#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sa_score.py — 分子可合成性评分脚本（功能 8）
================================================================
支持多种可合成性评分方法（可同时运行并输出对比）：

  1. SA Score（Synthetic Accessibility Score）
     文献：Ertl & Schuffenhauer, J. Cheminform. 1:8, 2009
     分数范围：1（易合成）→ 10（难合成）
     依赖：RDKit（rdkit.Contrib.SA_Score）

  2. SCScore（Sequential Conditioned Score）
     文献：Coley et al., J. Chem. Inf. Model. 58(2):252-261, 2018
     分数范围：1 → 5（值越大越难合成）
     依赖：scscore 库（pip install scscore）或内置轻量估算

  3. SYBA（SYnthetic acBility Assessment）
     文献：Vorsilak et al., J. Cheminform. 12:35, 2020
     分数范围：< 0 难合成，> 0 易合成
     依赖：syba（pip install syba）

  4. RAscore（Retrosynthetic Accessibility Score）
     文献：Thakkar et al., Chem. Sci. 12:3339-3349, 2021
     分数范围：0（难）→ 1（易）
     依赖：rascore（pip install rascore）

  5. 简易 Morgan 指纹复杂度评分（内置，无需额外依赖）
     基于 Morgan 指纹高度唯一位的比例估算合成复杂度

所有参数在顶部 CONFIG 区域配置。
"""

# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据输入 ----------
INPUT_NPY       = 'your_database.npy'   # 输入 npy 文件
SMILES_FIELD    = 'smiles'              # SMILES 字段名
NAME_FIELD      = 'name'               # 分子名字段名
OUTPUT_DIR      = 'sa_results'          # 输出目录
OUTPUT_CSV      = 'synthesizability.csv'

# ---------- 评分方法开关 ----------
USE_SA_SCORE    = True    # SA Score（RDKit 内置，推荐）
USE_SCSCORE     = False   # SCScore（需要 pip install scscore）
USE_SYBA        = False   # SYBA（需要 pip install syba）
USE_RASCORE     = False   # RAscore（需要额外安装）
USE_SIMPLE_SCORE = True   # 简易 Morgan 复杂度评分（内置，始终可用）

# ---------- 图表设置 ----------
PLOT_DIST       = True    # 分数分布直方图
PLOT_SCATTER    = True    # 各方法得分对比散点矩阵
PLOT_DPI        = 150
PLOT_FIGSIZE    = (10, 6)
PLOT_FONT_SIZE  = 11
PLOT_STYLE      = 'whitegrid'

# ---------- 过滤 ----------
# 将各方法得分进行归一化后取均值，并按阈值过滤（值越高越容易合成）
FILTER_ENABLE   = True
SA_THRESHOLD    = 6.0     # SA Score ≤ 该值视为"可合成"
NORMALIZE_SCORES = True   # 将各方法归一化到 [0,1]（1=最易合成）后输出 combined_score

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))

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

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    print("[ERROR] RDKit 未安装，无法计算任何评分（pip install rdkit）")
    sys.exit(1)

# ── seaborn ───────────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    sns.set_style(PLOT_STYLE)
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

plt.rcParams.update({'font.size': PLOT_FONT_SIZE, 'figure.dpi': PLOT_DPI})

# ── SA Score（RDKit Contrib）─────────────────────────────────────────────────
def _get_sa_scorer():
    """加载 RDKit 内置 SA_Score 计算器（sa_scores.py）。"""
    # 尝试多种路径
    import importlib
    for modname in ['rdkit.Contrib.SA_Score.sascorer',
                    'SA_Score.sascorer', 'sascorer']:
        try:
            return importlib.import_module(modname)
        except ImportError:
            pass
    # 手动查找 RDKit 安装目录
    try:
        import rdkit
        rdkit_dir = os.path.dirname(rdkit.__file__)
        sa_path   = os.path.join(rdkit_dir, 'Contrib', 'SA_Score', 'sascorer.py')
        if os.path.isfile(sa_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location('sascorer', sa_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        pass
    return None


_SA_SCORER = _get_sa_scorer() if USE_SA_SCORE else None
if USE_SA_SCORE and _SA_SCORER is None:
    print("[WARN] SA Score 模块未找到，将跳过（尝试：pip install rdkit 或确认 RDKit Contrib 路径）")


def calc_sa_score(mol):
    if _SA_SCORER is None or mol is None:
        return np.nan
    try:
        return float(_SA_SCORER.calculateScore(mol))
    except Exception:
        return np.nan


# ── SCScore（轻量估算版，无需 scscore 库）────────────────────────────────────
# 参考原始 SCScore 的设计思路：复杂度 ∝ 分子量 × 官能团复杂度 / 环数
# 真实 SCScore 需要神经网络模型权重文件（约 200 MB），此处提供近似估算
def _approx_scscore(mol):
    """基于 RDKit 描述符的近似 SCScore（范围 ~1-5）。"""
    if mol is None:
        return np.nan
    try:
        mw    = Descriptors.MolWt(mol)
        nhba  = Descriptors.NumHAcceptors(mol)
        nhbd  = Descriptors.NumHDonors(mol)
        rings = Descriptors.RingCount(mol)
        rot   = Descriptors.NumRotatableBonds(mol)
        sp3   = Descriptors.FractionCSP3(mol)
        hac   = mol.GetNumHeavyAtoms()
        # 简单线性组合（近似）
        score = (1.0
                 + 0.0015 * mw
                 + 0.1  * rings
                 + 0.05 * rot
                 + 0.01 * hac
                 - 0.3  * sp3)
        return float(np.clip(score, 1.0, 5.0))
    except Exception:
        return np.nan


def calc_scscore(mol):
    if not USE_SCSCORE:
        return np.nan
    try:
        from scscore.standalone_model_numpy import SCScorer
        # 使用单例以避免重复加载
        if not hasattr(calc_scscore, '_model'):
            calc_scscore._model = SCScorer()
            calc_scscore._model.restore()
        smi = Chem.MolToSmiles(mol)
        _, score = calc_scscore._model.get_score_from_smi(smi)
        return float(score)
    except ImportError:
        return _approx_scscore(mol)   # 回退到近似版本
    except Exception:
        return np.nan


# ── SYBA ────────────────────────────────────────────────────────────────────
def calc_syba(mol):
    if not USE_SYBA:
        return np.nan
    try:
        from syba.syba import SybaClassifier
        if not hasattr(calc_syba, '_model'):
            calc_syba._model = SybaClassifier()
            calc_syba._model.fitDefaultScore()
        smi = Chem.MolToSmiles(mol)
        return float(calc_syba._model.predict(smi=smi))
    except ImportError:
        print("[WARN] SYBA 未安装（pip install syba）")
        return np.nan
    except Exception:
        return np.nan


# ── RAscore ─────────────────────────────────────────────────────────────────
def calc_rascore(mol):
    if not USE_RASCORE:
        return np.nan
    try:
        from rascore import rascore_predict
        smi = Chem.MolToSmiles(mol)
        return float(rascore_predict([smi])[0])
    except ImportError:
        print("[WARN] RAscore 未安装，请参考 https://github.com/reymond-group/RAscore")
        return np.nan
    except Exception:
        return np.nan


# ── 简易 Morgan 复杂度评分（内置）───────────────────────────────────────────
def calc_simple_score(mol):
    """
    基于 Morgan 指纹的简易结构复杂度评分（近似值）。
    思路：计算每个原子的 Morgan 半径信息量，高度唯一的子结构 → 复杂 → 分数高。
    返回值：0（简单）→ 1（复杂），越大越难合成。
    """
    if mol is None:
        return np.nan
    try:
        from rdkit.Chem import rdMolDescriptors
        n_atoms  = mol.GetNumHeavyAtoms()
        if n_atoms == 0:
            return np.nan
        # 高半径指纹中唯一 bit 的比例
        fp2 = AllChem.GetMorganFingerprint(mol, radius=2)
        fp3 = AllChem.GetMorganFingerprint(mol, radius=3)
        bits2 = len(fp2.GetNonzeroElements())
        bits3 = len(fp3.GetNonzeroElements())
        # 归一化
        score = min(1.0, (bits2 + bits3) / (2 * n_atoms + 1))
        return float(score)
    except Exception:
        return np.nan


# ── 加载数据 ──────────────────────────────────────────────────────────────────
def _load_npy(path):
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        return list(raw['successful'])
    return list(raw)


print("=" * 65)
print("  可合成性评分脚本")
print("=" * 65)
print(f"\n  引用说明：")
if USE_SA_SCORE:
    print("  [SA Score]  Ertl P. & Schuffenhauer A., J. Cheminform. 1:8 (2009)")
    print("              https://doi.org/10.1186/1758-2946-1-8")
if USE_SCSCORE:
    print("  [SCScore]   Coley C.W. et al., J. Chem. Inf. Model. 58:252 (2018)")
    print("              https://doi.org/10.1021/acs.jcim.7b00622")
if USE_SYBA:
    print("  [SYBA]      Vorsilak M. et al., J. Cheminform. 12:35 (2020)")
    print("              https://doi.org/10.1186/s13321-020-00439-2")
if USE_RASCORE:
    print("  [RAscore]   Thakkar A. et al., Chem. Sci. 12:3339 (2021)")
    print("              https://doi.org/10.1039/d0sc05401a")

_npy_path = INPUT_NPY
if not os.path.isabs(_npy_path):
    cand = os.path.join(_HERE, _npy_path)
    _npy_path = cand if os.path.isfile(cand) else _npy_path

mol_list = _load_npy(_npy_path)
print(f"\n[INFO] 分子总数: {len(mol_list)}")

# 构建 RDKit Mol 对象
names, mols = [], []
for d in mol_list:
    name = d.get(NAME_FIELD, '')
    smi  = d.get(SMILES_FIELD, '')
    names.append(name)
    if smi and isinstance(smi, str):
        try:
            mols.append(Chem.MolFromSmiles(smi))
        except Exception:
            mols.append(None)
    else:
        mols.append(None)

n_valid = sum(m is not None for m in mols)
print(f"[INFO] 有效 RDKit 分子: {n_valid}")

# 计算各评分
records = []
for i, (name, mol) in enumerate(zip(names, mols)):
    row = {'name': name}
    if USE_SA_SCORE:
        row['sa_score']    = calc_sa_score(mol)
    if USE_SCSCORE:
        row['scscore']     = calc_scscore(mol)
    if USE_SYBA:
        row['syba_score']  = calc_syba(mol)
    if USE_RASCORE:
        row['rascore']     = calc_rascore(mol)
    if USE_SIMPLE_SCORE:
        row['simple_complexity'] = calc_simple_score(mol)
    records.append(row)

    if (i + 1) % 100 == 0 or i == len(mol_list) - 1:
        print(f"  进度: {i+1}/{len(mol_list)}", end='\r')

print()

df = pd.DataFrame(records)

# ── 归一化 + 合并评分 ─────────────────────────────────────────────────────────
if NORMALIZE_SCORES:
    score_cols = [c for c in ['sa_score', 'scscore', 'syba_score', 'rascore', 'simple_complexity']
                  if c in df.columns]
    norm_parts = []
    for col in score_cols:
        s  = df[col].values.astype(float)
        lo, hi = np.nanmin(s), np.nanmax(s)
        if hi == lo:
            norm = np.zeros_like(s)
        else:
            norm = (s - lo) / (hi - lo)
        # 对于 SA Score 和 SCScore，值越大越难合成，需要翻转
        if col in ('sa_score', 'scscore', 'simple_complexity'):
            norm = 1.0 - norm
        norm_parts.append(norm)
        df[f'{col}_norm'] = norm

    if norm_parts:
        combined = np.nanmean(np.stack(norm_parts, axis=1), axis=1)
        df['combined_score'] = combined   # 0→难合成, 1→易合成

# ── 可合成性筛选 ──────────────────────────────────────────────────────────────
if FILTER_ENABLE and 'sa_score' in df.columns:
    df['synthesizable'] = df['sa_score'] <= SA_THRESHOLD
    n_synth = df['synthesizable'].sum()
    print(f"\n[筛选] SA Score ≤ {SA_THRESHOLD} 的分子: {n_synth} / {len(df)}"
          f"  ({n_synth/len(df):.1%})")

# 保存
csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
df.to_csv(csv_path, index=False, float_format='%.4f')
print(f"[INFO] 评分结果 → {csv_path}")

# ── 可视化 ────────────────────────────────────────────────────────────────────
def _save(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {path}")

score_plot_cols = {
    'sa_score':         'SA Score (1=易,10=难)',
    'scscore':          'SCScore (1=易,5=难)',
    'syba_score':       'SYBA Score (>0=易合成)',
    'rascore':          'RAscore (0=难,1=易)',
    'simple_complexity': '复杂度估算 (0=简单,1=复杂)',
    'combined_score':   '综合评分 (0=难,1=易)',
}

if PLOT_DIST:
    for col, label in score_plot_cols.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        if _HAS_SNS:
            sns.histplot(vals, bins=30, kde=True, ax=ax, color='#2980b9',
                         edgecolor='white')
        else:
            ax.hist(vals, bins=30, color='#2980b9', edgecolor='white')
        ax.set_xlabel(label, fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('Count', fontsize=PLOT_FONT_SIZE)
        ax.set_title(f'Distribution — {label}', fontsize=PLOT_FONT_SIZE + 1)
        ax.text(0.98, 0.98, f'mean={vals.mean():.3f}\nstd={vals.std():.3f}',
                transform=ax.transAxes, va='top', ha='right',
                fontsize=PLOT_FONT_SIZE - 1,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        _save(fig, f'plot_dist_{col}.png')

if PLOT_SCATTER:
    valid_cols = [c for c in score_plot_cols if c in df.columns and df[c].notna().sum() > 5]
    if len(valid_cols) >= 2:
        try:
            if _HAS_SNS:
                pair_df = df[valid_cols].dropna()
                if len(pair_df) > 0:
                    fig = plt.figure(figsize=(max(8, len(valid_cols) * 2.5),
                                               max(8, len(valid_cols) * 2.5)))
                    pg  = sns.pairplot(pair_df, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15})
                    pg.fig.suptitle('Synthesizability Score Correlation', y=1.02,
                                    fontsize=PLOT_FONT_SIZE + 1)
                    path = os.path.join(OUTPUT_DIR, 'plot_score_pairplot.png')
                    pg.fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
                    plt.close()
                    print(f"  已保存: {path}")
        except Exception as e:
            print(f"  [WARN] 散点矩阵失败: {e}")

print(f"\n{'='*65}")
print("  可合成性评分完成！")
print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}/")
print(f"{'='*65}")
