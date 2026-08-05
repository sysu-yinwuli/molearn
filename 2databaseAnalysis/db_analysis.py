#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db_analysis.py — 数据库可视化分析脚本（功能 2 + 3）
================================================================
功能：
  1. 从 .npy 文件读取分子数据库
  2. 多种图表输出：分布直方图、箱线图、小提琴图、KDE 图、
     散点图矩阵（Pair Plot）、热力图相关性、累积分布（CDF）、
     QQ 图、t-SNE/PCA/UMAP 降维可视化
  3. 基于分子指纹的分析：
     官能团频率统计、结构多样性（Tanimoto 距离）、
     子结构分布热力图、Morgan 位比特分析

所有参数均在顶部 CONFIG 区域集中管理。
"""

# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据输入 ----------
INPUT_NPY     = 'your_database.npy'    # 输入 npy 文件路径
SMILES_FIELD  = 'smiles'               # 分子 SMILES 字段名（npy dict 中的 key）
Y_FIELD       = 'y'                    # 目标属性字段名
NAME_FIELD    = 'name'                 # 分子名称字段名
OUTPUT_DIR    = 'analysis_output'      # 所有图表输出目录

# ---------- 通用图表设置 ----------
PLOT_DPI      = 150
PLOT_STYLE    = 'whitegrid'      # seaborn 样式
PLOT_PALETTE  = 'tab10'
PLOT_FONT_SIZE = 11
PLOT_FIGSIZE  = (10, 6)          # 默认图尺寸

# ---------- Y 分布分析 ----------
PLOT_HIST          = True        # 属性值直方图
HIST_BINS          = 30          # 分箱数
HIST_KDE           = True        # 叠加 KDE 曲线
HIST_COLOR         = '#3498db'
PLOT_CDF           = True        # 累积分布函数
PLOT_QQ            = True        # QQ 图（正态性检验）
PLOT_BOXPLOT       = True        # 箱线图
PLOT_VIOLIN        = True        # 小提琴图

# ---------- 描述符相关性分析 ----------
PLOT_CORR_HEATMAP  = True        # 描述符相关性热力图（只用前 N 列）
CORR_MAX_FEATURES  = 30          # 热力图最多展示几个特征（按方差排序）
CORR_METHOD        = 'pearson'   # 'pearson' | 'spearman' | 'kendall'
CORR_CMAP          = 'coolwarm'

# ---------- 降维可视化 ----------
PLOT_PCA_2D        = True        # PCA 二维投影图
PLOT_TSNE_2D       = True        # t-SNE 二维投影图
PLOT_UMAP_2D       = False       # UMAP 二维投影图（需 pip install umap-learn）
DIM_COLOR_BY       = 'y'         # 点颜色：'y'（目标值）| None（单色）
TSNE_PERPLEXITY    = 30
TSNE_N_ITER        = 1000
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1

# ---------- 分子指纹分析（需要 SMILES 字段）----------
FINGERPRINT_ANALYSIS  = True     # 整体开关
FP_TYPE               = 'morgan' # 'morgan' | 'maccs' | 'rdkit'
MORGAN_RADIUS         = 2
MORGAN_NBITS          = 2048

# 官能团分析
PLOT_FG_FREQ          = True     # 官能团出现频率柱状图
PLOT_FG_HEATMAP       = True     # 官能团 × 分子热力图（分子数 ≤ 200 时）
FG_TOP_N              = 20       # 只展示出现频率最高的 N 个官能团

# 分子多样性（基于 Tanimoto 距离）
PLOT_DIVERSITY_HIST   = True     # 分子间 Tanimoto 相似度分布直方图
DIVERSITY_MAX_MOLS    = 500      # 多样性计算时最多使用的分子数（避免 O(n²) 过慢）

# Morgan 位分析
PLOT_MORGAN_BITS      = True     # 高频 Morgan bit 柱状图
MORGAN_TOP_BITS       = 30       # 展示前 N 个高频 bit

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os, sys, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# =============================================================================
# ── MOLEARN_ 环境变量覆盖（由 molearn_run.py 自动设置，单独运行时忽略）─────────
# MOLEARN_INPUT_NPY  : 覆盖 INPUT_NPY
# MOLEARN_OUTPUT_DIR : 覆盖 OUTPUT_DIR
# =============================================================================
_env_input  = os.environ.get('MOLEARN_INPUT_NPY', '').strip()
_env_outdir = os.environ.get('MOLEARN_OUTPUT_DIR', '').strip()
if _env_input:
    INPUT_NPY  = _env_input
if _env_outdir:
    OUTPUT_DIR = _env_outdir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── seaborn ───────────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    sns.set_style(PLOT_STYLE)
    sns.set_palette(PLOT_PALETTE)
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    print("[WARN] seaborn 未安装，部分图表降级为 matplotlib 版本（pip install seaborn）")

plt.rcParams.update({'font.size': PLOT_FONT_SIZE,
                     'axes.titlesize': PLOT_FONT_SIZE + 1,
                     'figure.dpi': PLOT_DPI})

# ── RDKit（指纹分析可选）──────────────────────────────────────────────────────
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdmolops
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    print("[WARN] RDKit 未安装，指纹分析功能将跳过（pip install rdkit）")

# ── scipy（QQ 图）────────────────────────────────────────────────────────────
try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# =============================================================================
# 辅助函数
# =============================================================================

def _save(fig, fname, tight=True):
    path = os.path.join(OUTPUT_DIR, fname)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {path}")


def _load_npy(path):
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        return list(raw['successful'])
    if isinstance(raw, np.ndarray):
        return list(raw)
    raise ValueError(f"无法解析 npy 格式: {path}")


def _get_smiles(mol_list):
    """从分子列表提取 SMILES 列表（过滤 None）"""
    smis = []
    for d in mol_list:
        s = d.get(SMILES_FIELD)
        smis.append(s if isinstance(s, str) and s.strip() else None)
    return smis


def _mols_from_smiles(smis):
    """SMILES → RDKit Mol 列表（None 表示解析失败）"""
    if not _HAS_RDKIT:
        return []
    mols = []
    for s in smis:
        if s:
            try:
                mols.append(Chem.MolFromSmiles(s))
            except Exception:
                mols.append(None)
        else:
            mols.append(None)
    return mols


def _get_fingerprint(mol, fp_type='morgan'):
    """单分子指纹（返回 RDKit ExplicitBitVect）"""
    if mol is None:
        return None
    try:
        if fp_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_NBITS)
        if fp_type == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        if fp_type == 'rdkit':
            return rdmolops.RDKFingerprint(mol)
    except Exception:
        return None
    return None


# ── 官能团 SMARTS 定义 ────────────────────────────────────────────────────────
_FUNCTIONAL_GROUPS = {
    'Hydroxyl (-OH)':          '[OX2H]',
    'Amine (-NH2)':            '[NX3H2]',
    'Secondary Amine (-NH-)':  '[NX3H1]',
    'Tertiary Amine (-N<)':    '[NX3H0;!$(N=*)]',
    'Carbonyl (C=O)':          '[CX3]=[OX1]',
    'Aldehyde (-CHO)':         '[CX3H1](=O)',
    'Ketone (R-CO-R)':         '[CX3](=O)[#6]',
    'Carboxyl (-COOH)':        '[CX3](=O)[OX2H1]',
    'Ester (-COO-)':           '[CX3](=O)[OX2][#6]',
    'Amide (-CONH-)':          '[CX3](=O)[NX3]',
    'Nitro (-NO2)':            '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
    'Nitrile (-C≡N)':          '[NX1]#[CX2]',
    'Sulfhydryl (-SH)':        '[SX2H]',
    'Thioether (-S-)':         '[SX2;!$([SX2H])]',
    'Sulfone (-SO2-)':         '[SX4](=O)(=O)',
    'Halide (F)':              '[F]',
    'Halide (Cl)':             '[Cl]',
    'Halide (Br)':             '[Br]',
    'Halide (I)':              '[I]',
    'Aromatic Ring':           'c1ccccc1',
    'Alkene (C=C)':            '[CX3]=[CX3]',
    'Alkyne (C≡C)':            '[CX2]#[CX2]',
    'Phosphate (-PO4-)':       '[PX4](=O)([OX2H])([OX2H])[OX2H]',
    'Epoxide':                 '[OX2r3]1CC1',
    'Pyridine ring':           'n1ccccc1',
    'Imidazole ring':          'c1cnc[nH]1',
    'Furan ring':              'o1cccc1',
    'Thiophene ring':          's1cccc1',
}

# =============================================================================
# 1. 加载数据
# =============================================================================
print("=" * 65)
print("  数据库分析脚本启动")
print("=" * 65)

_npy_path = INPUT_NPY
if not os.path.isabs(_npy_path):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    cand  = os.path.join(_HERE, _npy_path)
    _npy_path = cand if os.path.isfile(cand) else _npy_path

mol_list = _load_npy(_npy_path)
print(f"\n[INFO] 分子总数: {len(mol_list)}")

# 提取标量属性
y_vals  = np.array([d.get(Y_FIELD, np.nan) for d in mol_list], dtype=np.float64)
names   = [d.get(NAME_FIELD, f'mol_{i}') for i, d in enumerate(mol_list)]
y_valid = y_vals[~np.isnan(y_vals)]
print(f"[INFO] 有效标签数: {len(y_valid)} / {len(y_vals)}")
print(f"[INFO] Y 统计: min={y_valid.min():.4f}  max={y_valid.max():.4f}  "
      f"mean={y_valid.mean():.4f}  std={y_valid.std():.4f}")

# 提取 SMILES
smis = _get_smiles(mol_list)
n_smis = sum(s is not None for s in smis)
print(f"[INFO] 含 SMILES 字段的分子: {n_smis}")

# =============================================================================
# 2. Y 分布图
# =============================================================================
print(f"\n{'─'*65}")
print("  [模块 1] Y 值分布分析")
print(f"{'─'*65}")

if PLOT_HIST and len(y_valid) > 0:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    if _HAS_SNS and HIST_KDE:
        sns.histplot(y_valid, bins=HIST_BINS, kde=True, color=HIST_COLOR, ax=ax,
                     edgecolor='white', linewidth=0.5)
    else:
        ax.hist(y_valid, bins=HIST_BINS, color=HIST_COLOR, edgecolor='white', density=HIST_KDE)
    ax.set_xlabel(f'Property ({Y_FIELD})', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Count' if not HIST_KDE else 'Density', fontsize=PLOT_FONT_SIZE)
    ax.set_title(f'Distribution of {Y_FIELD}  (n={len(y_valid)})', fontsize=PLOT_FONT_SIZE + 1)
    _save(fig, 'plot_y_histogram.png')

if PLOT_CDF and len(y_valid) > 0:
    sorted_y = np.sort(y_valid)
    cdf      = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    fig, ax  = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(sorted_y, cdf, color='#2980b9', linewidth=2)
    ax.fill_between(sorted_y, cdf, alpha=0.15, color='#2980b9')
    ax.set_xlabel(f'Property ({Y_FIELD})', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Cumulative Probability', fontsize=PLOT_FONT_SIZE)
    ax.set_title(f'Cumulative Distribution Function — {Y_FIELD}', fontsize=PLOT_FONT_SIZE + 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    _save(fig, 'plot_y_cdf.png')

if PLOT_QQ and len(y_valid) > 1 and _HAS_SCIPY:
    fig, ax = plt.subplots(figsize=(6, 6))
    _sp_stats.probplot(y_valid, dist='norm', plot=ax)
    ax.set_title(f'QQ Plot — {Y_FIELD} vs Normal Distribution', fontsize=PLOT_FONT_SIZE + 1)
    ax.get_lines()[0].set(markersize=3, alpha=0.6, color='#2980b9')
    ax.get_lines()[1].set(color='#e74c3c', linewidth=2)
    _save(fig, 'plot_y_qqplot.png')

if PLOT_BOXPLOT and len(y_valid) > 0:
    fig, ax = plt.subplots(figsize=(5, 6))
    if _HAS_SNS:
        sns.boxplot(y=y_valid, ax=ax, color='#3498db', width=0.4,
                    medianprops={'color': 'red', 'linewidth': 2})
    else:
        ax.boxplot(y_valid, patch_artist=True,
                   boxprops=dict(facecolor='#3498db'),
                   medianprops=dict(color='red', linewidth=2))
    # 统计注释
    ax.text(0.98, 0.98,
            f'n={len(y_valid)}\nmean={y_valid.mean():.3f}\nstd={y_valid.std():.3f}\n'
            f'median={np.median(y_valid):.3f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=PLOT_FONT_SIZE - 1,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.set_ylabel(Y_FIELD, fontsize=PLOT_FONT_SIZE)
    ax.set_title(f'Boxplot — {Y_FIELD}', fontsize=PLOT_FONT_SIZE + 1)
    _save(fig, 'plot_y_boxplot.png')

if PLOT_VIOLIN and len(y_valid) > 0:
    fig, ax = plt.subplots(figsize=(5, 6))
    if _HAS_SNS:
        sns.violinplot(y=y_valid, ax=ax, color='#9b59b6', inner='box',
                       cut=0, bw_adjust=0.8)
    else:
        vp = ax.violinplot([y_valid], showmeans=True, showmedians=True)
        vp['bodies'][0].set_facecolor('#9b59b6')
    ax.set_ylabel(Y_FIELD, fontsize=PLOT_FONT_SIZE)
    ax.set_title(f'Violin Plot — {Y_FIELD}', fontsize=PLOT_FONT_SIZE + 1)
    _save(fig, 'plot_y_violin.png')

# =============================================================================
# 3. 描述符特征相关性热力图
# =============================================================================
print(f"\n{'─'*65}")
print("  [模块 2] 描述符相关性分析")
print(f"{'─'*65}")

# 尝试从 npy 中提取数值特征（rdkit / mordred / maccs / morgan）
_feat_keys = ['rdkit_descriptor', 'mordred_descriptor', 'maccs_descriptor',
              'morgan_descriptor', 'extra_d']
feat_matrix = []
feat_source = None
for fk in _feat_keys:
    sample = next((d.get(fk) for d in mol_list if d.get(fk)), None)
    if sample is not None:
        vecs = [np.array(d.get(fk, [0]*len(sample)), dtype=float) for d in mol_list]
        try:
            feat_matrix = np.stack(vecs)
            feat_source = fk
            break
        except Exception:
            pass

if PLOT_CORR_HEATMAP and feat_matrix is not None and len(feat_matrix) > 0:
    try:
        feat_df  = pd.DataFrame(feat_matrix)
        feat_df  = feat_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
        n_cols   = min(CORR_MAX_FEATURES, feat_df.shape[1])
        # 按方差降序取前 n_cols 个特征
        variances = feat_df.var().nlargest(n_cols).index
        feat_sub  = feat_df[variances]
        corr_mat  = feat_sub.corr(method=CORR_METHOD)
        corr_mat.columns = [f'F{i}' for i in range(len(corr_mat.columns))]
        corr_mat.index   = corr_mat.columns

        fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.4), max(6, n_cols * 0.4)))
        if _HAS_SNS:
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(corr_mat, mask=mask, cmap=CORR_CMAP, ax=ax,
                        vmin=-1, vmax=1, center=0,
                        annot=(n_cols <= 15), fmt='.2f', linewidths=0.3,
                        square=True, cbar_kws={'shrink': 0.7})
        else:
            im = ax.imshow(corr_mat.values, cmap=CORR_CMAP, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
        ax.set_title(f'Feature Correlation Heatmap ({feat_source}, top-{n_cols} by variance)',
                     fontsize=PLOT_FONT_SIZE)
        _save(fig, 'plot_feature_correlation.png')
        print(f"  特征来源: {feat_source}，展示 top-{n_cols} 特征")
    except Exception as e:
        print(f"  [WARN] 相关性热力图失败: {e}")
else:
    print("  跳过（无数值特征字段）")

# =============================================================================
# 4. 降维可视化（PCA / t-SNE / UMAP）
# =============================================================================
print(f"\n{'─'*65}")
print("  [模块 3] 降维可视化")
print(f"{'─'*65}")

def _plot_2d(X2d, y_color, title, fname, cmap='viridis'):
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    if y_color is not None:
        sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=y_color, cmap=cmap,
                        s=20, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=Y_FIELD)
    else:
        ax.scatter(X2d[:, 0], X2d[:, 1], s=20, alpha=0.7, edgecolors='none')
    ax.set_title(title, fontsize=PLOT_FONT_SIZE + 1)
    ax.set_xlabel('Dim 1', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Dim 2', fontsize=PLOT_FONT_SIZE)
    _save(fig, fname)

if feat_matrix is not None and len(feat_matrix) > 0:
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.impute import SimpleImputer
    _imputer = SimpleImputer(strategy='mean')
    _scaler  = _SS()
    X_clean  = _scaler.fit_transform(_imputer.fit_transform(
        np.clip(feat_matrix, -1e30, 1e30)))
    y_color  = y_vals if DIM_COLOR_BY == 'y' else None

    if PLOT_PCA_2D:
        try:
            from sklearn.decomposition import PCA
            X2d = PCA(n_components=2, random_state=42).fit_transform(X_clean)
            _plot_2d(X2d, y_color,
                     f'PCA 2D Projection ({feat_source})',
                     'plot_pca_2d.png')
        except Exception as e:
            print(f"  [WARN] PCA 失败: {e}")

    if PLOT_TSNE_2D:
        try:
            from sklearn.manifold import TSNE
            _X_tsne = X_clean if X_clean.shape[0] <= 2000 else X_clean[:2000]
            _y_tsne = y_color[:2000] if y_color is not None else None
            X2d = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                       n_iter=TSNE_N_ITER, random_state=42).fit_transform(_X_tsne)
            _plot_2d(X2d, _y_tsne,
                     f't-SNE 2D Projection ({feat_source})',
                     'plot_tsne_2d.png')
        except Exception as e:
            print(f"  [WARN] t-SNE 失败: {e}")

    if PLOT_UMAP_2D:
        try:
            from umap import UMAP
            X2d = UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
                       min_dist=UMAP_MIN_DIST, random_state=42).fit_transform(X_clean)
            _plot_2d(X2d, y_color,
                     f'UMAP 2D Projection ({feat_source})',
                     'plot_umap_2d.png')
        except ImportError:
            print("  [SKIP] UMAP 需要 pip install umap-learn")
        except Exception as e:
            print(f"  [WARN] UMAP 失败: {e}")
else:
    print("  无数值特征矩阵，跳过降维可视化")

# =============================================================================
# 5. 分子指纹分析（需 RDKit + SMILES）
# =============================================================================
print(f"\n{'─'*65}")
print("  [模块 4] 分子指纹 / 官能团分析")
print(f"{'─'*65}")

if not FINGERPRINT_ANALYSIS:
    print("  FINGERPRINT_ANALYSIS=False，跳过")
elif not _HAS_RDKIT:
    print("  RDKit 未安装，跳过")
elif n_smis == 0:
    print(f"  .npy 中未检测到 '{SMILES_FIELD}' 字段，跳过")
else:
    # ── 解析所有分子 ────────────────────────────────────────────────────────
    mols = _mols_from_smiles(smis)
    valid_mols    = [(i, m) for i, m in enumerate(mols) if m is not None]
    print(f"  有效分子（RDKit 解析成功）: {len(valid_mols)} / {len(mol_list)}")

    # ── 5-A 官能团分析 ───────────────────────────────────────────────────────
    if PLOT_FG_FREQ:
        print("  计算官能团频率 ...")
        fg_counts = {}
        for fg_name, smarts in _FUNCTIONAL_GROUPS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            cnt = sum(1 for _, m in valid_mols if m.HasSubstructMatch(patt))
            fg_counts[fg_name] = cnt

        fg_df = pd.DataFrame.from_dict(fg_counts, orient='index', columns=['count'])
        fg_df['frequency'] = fg_df['count'] / max(len(valid_mols), 1)
        fg_df = fg_df.sort_values('count', ascending=False).head(FG_TOP_N)
        fg_df.to_csv(os.path.join(OUTPUT_DIR, 'functional_groups.csv'), float_format='%.4f')

        # 频率柱状图
        fig, ax = plt.subplots(figsize=(12, max(5, FG_TOP_N * 0.35)))
        colors_fg = plt.cm.tab20(np.linspace(0, 1, len(fg_df)))
        bars = ax.barh(fg_df.index[::-1], fg_df['frequency'][::-1],
                       color=colors_fg[::-1], edgecolor='white')
        ax.set_xlabel('Frequency (fraction of molecules)', fontsize=PLOT_FONT_SIZE)
        ax.set_title(f'Top-{FG_TOP_N} Functional Groups  (n={len(valid_mols)} mols)',
                     fontsize=PLOT_FONT_SIZE + 1)
        for bar, val in zip(bars, fg_df['frequency'][::-1]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2%}', va='center', fontsize=PLOT_FONT_SIZE - 2)
        _save(fig, 'plot_functional_group_frequency.png')

    # ── 5-B 官能团 × 分子热力图 ─────────────────────────────────────────────
    if PLOT_FG_HEATMAP and len(valid_mols) <= 200:
        print("  绘制官能团 × 分子热力图 ...")
        top_fgs = list(fg_df.index) if 'fg_df' in dir() else list(_FUNCTIONAL_GROUPS.keys())[:FG_TOP_N]
        rows = []
        idx_used = [i for i, _ in valid_mols]
        for fg_name in top_fgs:
            patt = Chem.MolFromSmarts(_FUNCTIONAL_GROUPS[fg_name])
            if patt is None:
                rows.append([0] * len(valid_mols))
            else:
                rows.append([int(m.HasSubstructMatch(patt)) for _, m in valid_mols])
        mat_df = pd.DataFrame(rows, index=top_fgs,
                              columns=[names[i] for i in idx_used])
        fig, ax = plt.subplots(figsize=(max(10, len(valid_mols) * 0.1 + 2),
                                        max(5, len(top_fgs) * 0.35)))
        if _HAS_SNS:
            sns.heatmap(mat_df, ax=ax, cmap='YlOrRd', linewidths=0.1,
                        cbar_kws={'label': 'Presence'}, yticklabels=True,
                        xticklabels=(len(valid_mols) <= 50))
        else:
            ax.imshow(mat_df.values, cmap='YlOrRd', aspect='auto')
            ax.set_yticks(range(len(top_fgs)))
            ax.set_yticklabels(top_fgs, fontsize=8)
        ax.set_title('Functional Group × Molecule Heatmap', fontsize=PLOT_FONT_SIZE + 1)
        _save(fig, 'plot_fg_molecule_heatmap.png')

    # ── 5-C 分子多样性（Tanimoto 距离分布）──────────────────────────────────
    if PLOT_DIVERSITY_HIST:
        print("  计算 Tanimoto 多样性分布 ...")
        sample_mols = valid_mols[:DIVERSITY_MAX_MOLS]
        fps  = [_get_fingerprint(m, FP_TYPE) for _, m in sample_mols]
        fps  = [fp for fp in fps if fp is not None]
        sims = []
        for i in range(len(fps)):
            bulk = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            sims.extend(bulk)
        if sims:
            sims = np.array(sims, dtype=float)
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            if _HAS_SNS:
                sns.histplot(sims, bins=40, kde=True, ax=ax, color='#27ae60',
                             edgecolor='white')
            else:
                ax.hist(sims, bins=40, color='#27ae60', edgecolor='white')
            ax.axvline(sims.mean(), color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean={sims.mean():.3f}')
            ax.legend(fontsize=PLOT_FONT_SIZE - 1)
            ax.set_xlabel(f'Tanimoto Similarity ({FP_TYPE})', fontsize=PLOT_FONT_SIZE)
            ax.set_ylabel('Count', fontsize=PLOT_FONT_SIZE)
            ax.set_title(f'Pairwise Similarity Distribution  '
                         f'(n={len(fps)} mols, {len(sims)} pairs)',
                         fontsize=PLOT_FONT_SIZE + 1)
            _save(fig, 'plot_tanimoto_diversity.png')
            print(f"  多样性: mean_sim={sims.mean():.4f}  "
                  f"diversity={1 - sims.mean():.4f}  "
                  f"std={sims.std():.4f}")

    # ── 5-D Morgan Bit 分析 ─────────────────────────────────────────────────
    if PLOT_MORGAN_BITS and FP_TYPE == 'morgan':
        print("  分析 Morgan Bit 分布 ...")
        bit_counts = np.zeros(MORGAN_NBITS, dtype=int)
        for _, m in valid_mols:
            fp = _get_fingerprint(m, 'morgan')
            if fp is not None:
                arr = np.array(fp)
                bit_counts += arr
        top_bits_idx = np.argsort(bit_counts)[::-1][:MORGAN_TOP_BITS]
        top_bits_cnt = bit_counts[top_bits_idx]
        top_bits_frq = top_bits_cnt / max(len(valid_mols), 1)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(MORGAN_TOP_BITS), top_bits_frq,
               color='#8e44ad', edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(MORGAN_TOP_BITS))
        ax.set_xticklabels([f'bit{i}' for i in top_bits_idx], rotation=45, ha='right',
                           fontsize=PLOT_FONT_SIZE - 3)
        ax.set_xlabel('Morgan Bit Index', fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('Frequency', fontsize=PLOT_FONT_SIZE)
        ax.set_title(f'Top-{MORGAN_TOP_BITS} Most Frequent Morgan Bits '
                     f'(radius={MORGAN_RADIUS}, nbits={MORGAN_NBITS})',
                     fontsize=PLOT_FONT_SIZE + 1)
        _save(fig, 'plot_morgan_bits_frequency.png')

        # 保存 bit 统计 CSV
        bits_df = pd.DataFrame({'bit_index': top_bits_idx,
                                 'count': top_bits_cnt,
                                 'frequency': top_bits_frq})
        bits_df.to_csv(os.path.join(OUTPUT_DIR, 'morgan_bits.csv'), index=False)

# =============================================================================
# 6. 汇总报告
# =============================================================================
summary_path = os.path.join(OUTPUT_DIR, 'analysis_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  数据库分析汇总报告\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"输入文件:  {INPUT_NPY}\n")
    f.write(f"分子总数:  {len(mol_list)}\n")
    f.write(f"有效标签:  {len(y_valid)}\n")
    if len(y_valid) > 0:
        f.write(f"Y 统计:    min={y_valid.min():.4f}  max={y_valid.max():.4f}  "
                f"mean={y_valid.mean():.4f}  std={y_valid.std():.4f}\n")
    f.write(f"\n含 SMILES 分子: {n_smis}\n")
    if FINGERPRINT_ANALYSIS and _HAS_RDKIT and n_smis > 0:
        f.write(f"RDKit 解析成功: {len(valid_mols)}\n")
    f.write(f"\n[输出文件]\n")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        f.write(f"  {fname}\n")

print(f"\n  汇总报告: {summary_path}")
print(f"\n{'='*65}")
print("  数据库分析完成！")
print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}/")
print(f"{'='*65}")
