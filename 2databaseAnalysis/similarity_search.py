#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
similarity_search.py — 分子相似度打分脚本（功能 9）
================================================================
功能：
  以目标集 npy（含若干查询分子）为基准，
  与数据库 npy（含全部数据库分子）进行逐一 Tanimoto 相似度计算。

支持的分子指纹（选其中一种）：
  'morgan'  : Morgan 圆形指纹（ECFP）
  'maccs'   : MACCS Keys 167 位
  'rdkit'   : RDKit 拓扑指纹
  'atompair': AtomPair 指纹
  'torsion' : Topological Torsion 指纹

输出：
  - similarity_matrix.csv : 行=查询分子，列=数据库分子，值=Tanimoto 相似度
  - similarity_hits.csv   : 超过阈值的（查询, 数据库, 相似度）三元组
  - similarity_top_n.csv  : 每个查询分子的 Top-N 最相似分子
  - 可视化：相似度热力图、分布图、Top-N 柱状图

所有参数在顶部 CONFIG 区域配置。
"""

# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据输入 ----------
QUERY_NPY       = 'query_mols.npy'      # 查询分子 npy（含若干目标分子）
DATABASE_NPY    = 'database_mols.npy'   # 数据库 npy（全量数据库）
SMILES_FIELD    = 'smiles'              # SMILES 字段名
NAME_FIELD      = 'name'               # 分子名字段名
OUTPUT_DIR      = 'similarity_results'  # 输出目录

# ---------- 指纹设置 ----------
FP_TYPE         = 'morgan'   # 'morgan' | 'maccs' | 'rdkit' | 'atompair' | 'torsion'
MORGAN_RADIUS   = 2          # 仅 Morgan：半径
MORGAN_NBITS    = 2048       # 仅 Morgan/RDKit：指纹长度
MORGAN_USE_FEAT = False      # True=FCFP, False=ECFP

# ---------- 相似度设置 ----------
SIMILARITY_METRIC = 'tanimoto'   # 目前仅支持 'tanimoto'

# ---------- 阈值筛选 ----------
HIT_THRESHOLD   = 0.7        # 相似度高于此值输出到 hits 文件
                              # 设为 0 输出全部（慎用：若数据库很大会导致大文件）

# ---------- Top-N 输出 ----------
TOP_N           = 10         # 每个查询分子输出前 N 个最相似的数据库分子

# ---------- 矩阵输出 ----------
OUTPUT_MATRIX   = True       # 输出完整相似度矩阵（行=查询，列=数据库）
                              # 若数据库很大（>5000），建议改为 False

# ---------- 可视化设置 ----------
PLOT_HEATMAP    = True       # 相似度热力图（查询数 × 数据库数 ≤ 200×200）
PLOT_DIST       = True       # 查询与数据库的相似度分布图
PLOT_TOPN_BAR   = True       # 每个查询分子 Top-N 柱状图（查询数 ≤ 20 时）
PLOT_DPI        = 150
PLOT_FIGSIZE    = (12, 7)
PLOT_FONT_SIZE  = 11
PLOT_STYLE      = 'whitegrid'
HEATMAP_CMAP    = 'YlOrRd'

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── seaborn ───────────────────────────────────────────────────────────────────
try:
    import seaborn as sns
    sns.set_style(PLOT_STYLE)
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

plt.rcParams.update({'font.size': PLOT_FONT_SIZE, 'figure.dpi': PLOT_DPI})

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, MACCSkeys, rdmolops
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    print("[ERROR] RDKit 未安装（pip install rdkit）")
    sys.exit(1)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _load_npy(path):
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        return list(raw['successful'])
    return list(raw)


def _get_fp(mol):
    """计算单分子指纹。"""
    if mol is None:
        return None
    try:
        if FP_TYPE == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(
                mol, MORGAN_RADIUS, MORGAN_NBITS, useFeatures=MORGAN_USE_FEAT)
        if FP_TYPE == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        if FP_TYPE == 'rdkit':
            return rdmolops.RDKFingerprint(mol, fpSize=MORGAN_NBITS)
        if FP_TYPE == 'atompair':
            return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=MORGAN_NBITS)
        if FP_TYPE == 'torsion':
            return AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=MORGAN_NBITS)
    except Exception:
        return None
    return None


def _batch_fps(mol_list):
    """从分子列表批量提取指纹，返回 (names, fps) 列表（过滤失败项）。"""
    names, fps = [], []
    for d in mol_list:
        name = d.get(NAME_FIELD, '')
        smi  = d.get(SMILES_FIELD, '')
        mol  = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        fp   = _get_fp(mol)
        if fp is not None:
            names.append(name)
            fps.append(fp)
    return names, fps


def _save(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {path}")


# =============================================================================
# 主程序
# =============================================================================
print("=" * 65)
print("  分子相似度打分脚本")
print("=" * 65)
print(f"  指纹类型:   {FP_TYPE.upper()}")
print(f"  相似度度量: {SIMILARITY_METRIC}")
print(f"  命中阈值:   {HIT_THRESHOLD}")
print(f"  Top-N:      {TOP_N}\n")

# ── 加载数据 ──────────────────────────────────────────────────────────────────
_q_path  = QUERY_NPY    if os.path.isabs(QUERY_NPY)    else os.path.join(_HERE, QUERY_NPY)
_db_path = DATABASE_NPY if os.path.isabs(DATABASE_NPY) else os.path.join(_HERE, DATABASE_NPY)

print("[INFO] 加载查询分子 ...")
q_mols   = _load_npy(_q_path)
print(f"  查询集分子数: {len(q_mols)}")

print("[INFO] 加载数据库分子 ...")
db_mols  = _load_npy(_db_path)
print(f"  数据库分子数: {len(db_mols)}")

# ── 计算指纹 ──────────────────────────────────────────────────────────────────
print("\n[INFO] 计算查询分子指纹 ...")
q_names, q_fps = _batch_fps(q_mols)
print(f"  有效查询分子: {len(q_names)}")

print("[INFO] 计算数据库分子指纹 ...")
db_names, db_fps = _batch_fps(db_mols)
print(f"  有效数据库分子: {len(db_names)}")

if len(q_fps) == 0 or len(db_fps) == 0:
    print("[ERROR] 无有效指纹（检查 SMILES 字段和 npy 格式）")
    sys.exit(1)

# ── 计算相似度矩阵 ────────────────────────────────────────────────────────────
print(f"\n[INFO] 计算 Tanimoto 相似度矩阵 "
      f"({len(q_names)} × {len(db_names)}) ...")

sim_matrix = np.zeros((len(q_names), len(db_names)), dtype=np.float32)
for i, q_fp in enumerate(q_fps):
    sims = DataStructs.BulkTanimotoSimilarity(q_fp, db_fps)
    sim_matrix[i] = sims
    if (i + 1) % 10 == 0 or i == len(q_fps) - 1:
        print(f"  进度: {i+1}/{len(q_names)}", end='\r')
print()

# ── 输出相似度矩阵 CSV ────────────────────────────────────────────────────────
if OUTPUT_MATRIX:
    mat_df = pd.DataFrame(sim_matrix, index=q_names, columns=db_names)
    mat_path = os.path.join(OUTPUT_DIR, 'similarity_matrix.csv')
    mat_df.to_csv(mat_path, float_format='%.4f')
    print(f"[INFO] 相似度矩阵 → {mat_path}")

# ── Hits：超过阈值的分子对 ────────────────────────────────────────────────────
print(f"\n[INFO] 筛选相似度 ≥ {HIT_THRESHOLD} 的分子对 ...")
hit_rows = []
for i, q_name in enumerate(q_names):
    for j, db_name in enumerate(db_names):
        sim = float(sim_matrix[i, j])
        if sim >= HIT_THRESHOLD:
            hit_rows.append({'query': q_name, 'database': db_name,
                             'similarity': sim})

hit_df = pd.DataFrame(hit_rows)
if len(hit_df) > 0:
    hit_df = hit_df.sort_values(['query', 'similarity'], ascending=[True, False])
hits_path = os.path.join(OUTPUT_DIR, 'similarity_hits.csv')
hit_df.to_csv(hits_path, index=False, float_format='%.4f')
print(f"  命中分子对数: {len(hit_df)}")
print(f"  结果 → {hits_path}")

# 输出命中分子名称列表（按查询分组）
hits_by_query_path = os.path.join(OUTPUT_DIR, 'hits_by_query.txt')
with open(hits_by_query_path, 'w', encoding='utf-8') as f:
    f.write(f"# 相似度 ≥ {HIT_THRESHOLD} 的命中分子（FP: {FP_TYPE.upper()}）\n\n")
    if len(hit_df) > 0:
        for q_name in q_names:
            subset = hit_df[hit_df['query'] == q_name]
            if len(subset) > 0:
                f.write(f"查询分子: {q_name}  ({len(subset)} 个命中)\n")
                for _, row in subset.iterrows():
                    f.write(f"  {row['database']:<40s}  sim={row['similarity']:.4f}\n")
                f.write("\n")
    else:
        f.write("无命中分子（所有相似度均低于阈值）\n")
print(f"  命中分子列表 → {hits_by_query_path}")

# ── Top-N 最相似分子 ─────────────────────────────────────────────────────────
print(f"\n[INFO] 为每个查询分子提取 Top-{TOP_N} 最相似的数据库分子 ...")
topn_rows = []
for i, q_name in enumerate(q_names):
    top_idx = np.argsort(sim_matrix[i])[::-1][:TOP_N]
    for rank, j in enumerate(top_idx, 1):
        topn_rows.append({'query': q_name,
                          'rank': rank,
                          'database': db_names[j],
                          'similarity': float(sim_matrix[i, j])})

topn_df = pd.DataFrame(topn_rows)
topn_path = os.path.join(OUTPUT_DIR, 'similarity_top_n.csv')
topn_df.to_csv(topn_path, index=False, float_format='%.4f')
print(f"  Top-{TOP_N} 结果 → {topn_path}")

# ── 统计摘要 ─────────────────────────────────────────────────────────────────
print("\n[INFO] 相似度统计：")
all_sims = sim_matrix.flatten()
print(f"  全局 mean={all_sims.mean():.4f}  std={all_sims.std():.4f}  "
      f"max={all_sims.max():.4f}  min={all_sims.min():.4f}")

# 每个查询分子的 max 相似度
max_sims = sim_matrix.max(axis=1)
for i, (q_name, ms) in enumerate(zip(q_names, max_sims)):
    best_j  = int(np.argmax(sim_matrix[i]))
    print(f"  {q_name:<40s} 最大相似度={ms:.4f}  vs. {db_names[best_j]}")

# ── 可视化 ────────────────────────────────────────────────────────────────────

# 热力图
if PLOT_HEATMAP and len(q_names) <= 200 and len(db_names) <= 200:
    fig, ax = plt.subplots(figsize=(max(8, len(db_names) * 0.15 + 2),
                                     max(5, len(q_names) * 0.3 + 1)))
    if _HAS_SNS:
        sns.heatmap(sim_matrix, ax=ax, cmap=HEATMAP_CMAP,
                    xticklabels=(db_names if len(db_names) <= 30 else False),
                    yticklabels=(q_names  if len(q_names) <= 30 else True),
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Tanimoto Similarity'})
    else:
        im = ax.imshow(sim_matrix, cmap=HEATMAP_CMAP, aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Tanimoto Similarity')
    ax.set_title(f'Similarity Matrix ({FP_TYPE.upper()})  '
                 f'Threshold={HIT_THRESHOLD}',
                 fontsize=PLOT_FONT_SIZE + 1)
    ax.set_xlabel('Database Molecules', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Query Molecules', fontsize=PLOT_FONT_SIZE)
    _save(fig, 'plot_similarity_heatmap.png')

# 分布图（全局相似度）
if PLOT_DIST:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    if _HAS_SNS:
        sns.histplot(all_sims, bins=50, kde=True, ax=ax,
                     color='#27ae60', edgecolor='white')
    else:
        ax.hist(all_sims, bins=50, color='#27ae60', edgecolor='white', density=True)
    ax.axvline(HIT_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold={HIT_THRESHOLD}')
    ax.set_xlabel(f'Tanimoto Similarity ({FP_TYPE.upper()})', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Density', fontsize=PLOT_FONT_SIZE)
    ax.set_title('Pairwise Similarity Distribution (Query vs. Database)',
                 fontsize=PLOT_FONT_SIZE + 1)
    ax.legend(fontsize=PLOT_FONT_SIZE - 1)
    ax.text(0.98, 0.98,
            f'mean={all_sims.mean():.3f}\nstd={all_sims.std():.3f}\n'
            f'hits={len(hit_df)}({len(hit_df)/(len(q_names)*len(db_names)+1e-9):.1%})',
            transform=ax.transAxes, va='top', ha='right',
            fontsize=PLOT_FONT_SIZE - 1,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    _save(fig, 'plot_similarity_distribution.png')

# Top-N 柱状图（每个查询分子）
if PLOT_TOPN_BAR and len(q_names) <= 20:
    for q_name in q_names:
        subset = topn_df[topn_df['query'] == q_name].head(TOP_N)
        if len(subset) == 0:
            continue
        fig, ax = plt.subplots(figsize=(max(8, TOP_N * 0.7), 5))
        colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(subset)))[::-1]
        bars = ax.bar(range(len(subset)), subset['similarity'].values,
                      color=colors_bar, edgecolor='white')
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset['database'].values,
                           rotation=40, ha='right', fontsize=PLOT_FONT_SIZE - 2)
        ax.set_ylabel('Tanimoto Similarity', fontsize=PLOT_FONT_SIZE)
        ax.set_ylim(0, 1.05)
        ax.axhline(HIT_THRESHOLD, color='red', linestyle='--', linewidth=1.2,
                   label=f'Threshold={HIT_THRESHOLD}')
        ax.legend(fontsize=PLOT_FONT_SIZE - 1)
        ax.set_title(f'Top-{TOP_N} Similar Molecules  Query: {q_name}',
                     fontsize=PLOT_FONT_SIZE + 1)
        for bar, sim_val in zip(bars, subset['similarity'].values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{sim_val:.3f}', ha='center', va='bottom',
                    fontsize=PLOT_FONT_SIZE - 3)
        safe_name = q_name.replace('/', '_').replace('\\', '_')
        _save(fig, f'plot_topn_{safe_name}.png')

# ── 写汇总 txt ────────────────────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, 'similarity_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  分子相似度搜索汇总报告\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"查询集:     {QUERY_NPY}\n")
    f.write(f"数据库:     {DATABASE_NPY}\n")
    f.write(f"指纹类型:   {FP_TYPE.upper()}\n")
    f.write(f"相似度度量: Tanimoto\n")
    f.write(f"命中阈值:   {HIT_THRESHOLD}\n")
    f.write(f"\n[规模]\n")
    f.write(f"  有效查询分子:   {len(q_names)}\n")
    f.write(f"  有效数据库分子: {len(db_names)}\n")
    f.write(f"\n[相似度统计]\n")
    f.write(f"  全局 mean={all_sims.mean():.4f}  std={all_sims.std():.4f}  "
            f"max={all_sims.max():.4f}  min={all_sims.min():.4f}\n")
    f.write(f"  命中分子对数 (≥{HIT_THRESHOLD}): {len(hit_df)}\n")
    f.write(f"\n[每个查询分子最优命中]\n")
    for i, (q_name, ms) in enumerate(zip(q_names, max_sims)):
        best_j = int(np.argmax(sim_matrix[i]))
        f.write(f"  {q_name:<40s}  max_sim={ms:.4f}  best={db_names[best_j]}\n")
    f.write(f"\n[输出文件]\n")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        f.write(f"  {fname}\n")

print(f"\n  汇总报告 → {summary_path}")
print(f"\n{'='*65}")
print("  相似度搜索完成！")
print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}/")
print(f"{'='*65}")
