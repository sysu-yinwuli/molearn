#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_npy.py — 数据库抽样脚本（功能 5）
================================================================
从主数据库 .npy 文件抽样产生多个子数据库，支持：
  1. 随机抽样（Random Sampling）
  2. 系统/均匀抽样（Systematic Sampling）
  3. 分层抽样（Stratified Sampling，按 y 值分箱）
  4. 正交抽样（Latin Hypercube Sampling，基于描述符主成分）
  5. 基于分子指纹的多样性抽样（MaxMin / Sphere Exclusion）

一次运行可通过不同随机数种子生成多个子数据集。
所有参数在顶部 CONFIG 区域配置。
"""

# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据输入 ----------
INPUT_NPY      = 'poly-all.npy'       # 主数据库 npy 文件路径
OUTPUT_PREFIX  = 'sub'                # 输出文件名前缀，实际文件名: sub_method_n_seed.npy
OUTPUT_DIR     = 'sampled_npy'        # 输出目录

# ---------- 抽样目标 ----------
# 每个子数据集的大小：可以是绝对数量（int）或比例（0~1 float）
SAMPLE_SIZE    = 100                  # 例如 100 个分子，或 0.1 表示 10%

# ---------- 方法开关（可同时开启多种方法）----------
METHOD_RANDOM       = True    # 随机抽样
METHOD_SYSTEMATIC   = False   # 系统/均匀抽样（按排列顺序等间隔取）
METHOD_STRATIFIED   = True    # 分层抽样（按 y 值分箱层内随机取）
METHOD_LHS          = True    # Latin Hypercube（正交抽样）
METHOD_DIVERSITY    = True    # 基于指纹多样性抽样（MaxMin 算法）

# ---------- 多次抽样（不同随机种子产生多个子集）----------
SEEDS              = [42, 123, 456]   # 每个种子产生一套子集
# 注意：METHOD_SYSTEMATIC 和 METHOD_DIVERSITY 受 seed 影响较小，但仍保持一致性

# ---------- 分层抽样参数 ----------
STRATIFIED_N_BINS  = 5              # y 值分箱数
STRATIFIED_Y_FIELD = 'y'           # 分层依据的字段名

# ---------- LHS 正交抽样参数 ----------
LHS_N_COMPONENTS   = 20            # PCA 降维后的主成分数（用于构造 LHS 空间）
LHS_FEAT_FIELD     = None          # 描述符字段（None=自动检测第一个可用描述符）

# ---------- 多样性抽样参数 ----------
DIVERSITY_FP_TYPE  = 'morgan'      # 'morgan' | 'maccs' | 'rdkit'
DIVERSITY_RADIUS   = 2             # Morgan 指纹半径
DIVERSITY_NBITS    = 2048          # Morgan/RDKit 指纹长度
DIVERSITY_SMILES_FIELD = 'smiles'  # SMILES 字段名
# MaxMin 初始分子选择策略：'random'（随机取一个起始）| 'center'（取最中心的）
DIVERSITY_INIT     = 'random'

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os, sys, warnings
import numpy as np
import warnings
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


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _load_npy(path):
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        return list(raw['successful'])
    return list(raw)


def _save_npy(mol_list, method, n_target, seed, suffix=''):
    fname = f"{OUTPUT_PREFIX}_{method}_n{len(mol_list)}_seed{seed}{suffix}.npy"
    path  = os.path.join(OUTPUT_DIR, fname)
    save_dict = {
        'successful':    mol_list,
        'failed_count':  0,
        'error_stats':   {},
        'sample_info': {
            'method':      method,
            'source':      INPUT_NPY,
            'seed':        seed,
            'n_target':    n_target,
            'n_actual':    len(mol_list),
        }
    }
    np.save(path, save_dict, allow_pickle=True)
    print(f"    ✓ 已保存: {path}  ({len(mol_list)} 分子)")
    return path


def _resolve_size(total: int) -> int:
    """将 SAMPLE_SIZE 解析为绝对数量。"""
    if isinstance(SAMPLE_SIZE, float) and 0 < SAMPLE_SIZE <= 1:
        return max(1, int(round(SAMPLE_SIZE * total)))
    return min(int(SAMPLE_SIZE), total)


def _get_feat_matrix(mol_list):
    """自动检测并提取数值特征矩阵（用于 PCA / LHS）。"""
    feat_keys = ['rdkit_descriptor', 'mordred_descriptor',
                 'maccs_descriptor', 'morgan_descriptor', 'extra_d']
    fk_use    = LHS_FEAT_FIELD or next(
        (k for k in feat_keys if mol_list[0].get(k) is not None), None)
    if fk_use is None:
        return None, None

    n   = len(mol_list)
    dim = len(mol_list[0].get(fk_use, []))
    if dim == 0:
        return None, None

    mat = np.zeros((n, dim), dtype=np.float32)
    for i, d in enumerate(mol_list):
        v = d.get(fk_use, [])
        if v:
            mat[i] = np.clip(np.array(v, dtype=np.float32), -1e10, 1e10)

    # 去 NaN
    col_mean = np.nanmean(mat, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    mask = np.isnan(mat)
    mat[mask] = np.take(col_mean, np.where(mask)[1])
    return mat, fk_use


def _get_fingerprints(mol_list):
    """提取分子指纹列表（需 RDKit）。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys, rdmolops
    except ImportError:
        return None

    fps = []
    for d in mol_list:
        smi = d.get(DIVERSITY_SMILES_FIELD)
        if not smi:
            fps.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(None)
                continue
            if DIVERSITY_FP_TYPE == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, DIVERSITY_RADIUS, DIVERSITY_NBITS)
            elif DIVERSITY_FP_TYPE == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            else:
                fp = rdmolops.RDKFingerprint(mol, fpSize=DIVERSITY_NBITS)
            fps.append(fp)
        except Exception:
            fps.append(None)
    return fps


# ── 抽样方法 ──────────────────────────────────────────────────────────────────

def sample_random(mol_list, n, rng):
    """全随机抽样。"""
    idx = rng.choice(len(mol_list), size=n, replace=False)
    return [mol_list[i] for i in sorted(idx)]


def sample_systematic(mol_list, n):
    """系统/均匀抽样（按原始顺序等间隔）。"""
    total = len(mol_list)
    step  = total / n
    idx   = [int(i * step) for i in range(n)]
    return [mol_list[i] for i in idx]


def sample_stratified(mol_list, n, rng):
    """分层抽样（按 y 值分箱，各层按比例抽取）。"""
    y_vals = np.array([d.get(STRATIFIED_Y_FIELD, np.nan) for d in mol_list],
                      dtype=np.float64)
    valid  = np.where(~np.isnan(y_vals))[0]
    if len(valid) == 0:
        print("  [WARN] 无有效 y 值，回退到随机抽样")
        return sample_random(mol_list, n, rng)

    # 分箱
    y_valid = y_vals[valid]
    bins    = np.percentile(y_valid, np.linspace(0, 100, STRATIFIED_N_BINS + 1))
    bins    = np.unique(bins)
    labels  = np.digitize(y_valid, bins[1:-1])  # 0 ~ n_bins-1

    selected = []
    for b in np.unique(labels):
        bin_idx    = valid[labels == b]
        n_take     = max(1, round(n * len(bin_idx) / len(valid)))
        take       = rng.choice(bin_idx, size=min(n_take, len(bin_idx)), replace=False)
        selected.extend(take.tolist())

    # 按总数调整（可能因取整误差而多/少几个）
    selected = list(set(selected))
    if len(selected) < n:
        remaining = list(set(range(len(mol_list))) - set(selected))
        extra     = rng.choice(remaining, size=n - len(selected), replace=False)
        selected.extend(extra.tolist())
    selected = selected[:n]
    return [mol_list[i] for i in sorted(selected)]


def sample_lhs(mol_list, n, rng):
    """
    Latin Hypercube 抽样（正交抽样）。
    思路：
      1. PCA 降维到 LHS_N_COMPONENTS 维
      2. 每个主成分方向均匀分 n 格，LHS 保证每格各有一个样本
      3. 依据 LHS 中每格对应的分子指标选择真实分子
    """
    mat, fk = _get_feat_matrix(mol_list)
    if mat is None:
        print("  [WARN] 无可用描述符，LHS 回退到随机抽样")
        return sample_random(mol_list, n, rng)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X     = StandardScaler().fit_transform(mat)
    n_pca = min(LHS_N_COMPONENTS, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    total = len(mol_list)
    dim   = X_pca.shape[1]

    # LHS：在每维各分 n 格，随机排列后组合
    lhs_grid = np.zeros((n, dim), dtype=float)
    for d in range(dim):
        perm           = rng.permutation(n)
        lhs_grid[:, d] = (perm + rng.random(n)) / n   # ∈ [0,1)

    # 将 X_pca 归一化到 [0,1)，然后为每个 LHS 点找最近邻
    X_min = X_pca.min(axis=0)
    X_max = X_pca.max(axis=0) + 1e-9
    X_norm = (X_pca - X_min) / (X_max - X_min)

    # 用最近邻找最接近每个 LHS 点的真实分子（禁止重复选取）
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_norm)
        _, indices = nn.kneighbors(lhs_grid)
        selected = list(dict.fromkeys(indices.flatten().tolist()))  # 去重保序
    except Exception:
        # 退化为欧氏最近邻暴力搜索
        selected = []
        used     = set()
        for lp in lhs_grid:
            dists = np.sum((X_norm - lp) ** 2, axis=1)
            for rank_idx in np.argsort(dists):
                if rank_idx not in used:
                    selected.append(int(rank_idx))
                    used.add(rank_idx)
                    break

    # 补充不足的（可能因最近邻重复导致）
    if len(selected) < n:
        remaining = list(set(range(total)) - set(selected))
        extra     = rng.choice(remaining,
                               size=min(n - len(selected), len(remaining)),
                               replace=False)
        selected.extend(extra.tolist())

    selected = selected[:n]
    return [mol_list[i] for i in sorted(selected)]


def sample_diversity_maxmin(mol_list, n, rng):
    """
    基于分子指纹相似度的多样性抽样（MaxMin 算法）。
    每步选择与当前已选集合中所有分子的最小 Tanimoto 相似度最大的分子。
    """
    try:
        from rdkit import DataStructs
    except ImportError:
        print("  [WARN] RDKit 未安装，多样性抽样回退到随机抽样")
        return sample_random(mol_list, n, rng)

    fps = _get_fingerprints(mol_list)
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]

    if len(valid_idx) < n:
        print(f"  [WARN] 有效指纹数 ({len(valid_idx)}) < 目标数 ({n})，返回所有有效分子")
        return [mol_list[i] for i in valid_idx]

    # 初始化：选最中心的（mean 相似度最高）或随机
    if DIVERSITY_INIT == 'center' and len(valid_idx) <= 2000:
        # 计算每个分子与其他分子的平均相似度，选最中心的
        mean_sims = []
        for i in valid_idx:
            others = [fps[j] for j in valid_idx if j != i and fps[j] is not None]
            if others:
                mean_sims.append(np.mean(DataStructs.BulkTanimotoSimilarity(fps[i], others)))
            else:
                mean_sims.append(0.0)
        init_local = int(np.argmax(mean_sims))
        init_global = valid_idx[init_local]
    else:
        init_global = rng.choice(valid_idx)

    selected   = [init_global]
    remaining  = set(valid_idx) - {init_global}
    # min_sim[i] = 与已选集合的最小相似度（初始为对应单个分子的相似度）
    min_sim    = {i: DataStructs.TanimotoSimilarity(fps[i], fps[init_global])
                  for i in remaining}

    for _ in range(n - 1):
        if not remaining:
            break
        # 选 min_sim 最小（即与已选集合差异最大）的分子
        best      = max(remaining, key=lambda i: -min_sim[i])
        # 注意：maxmin 选的是"距离最大"，即相似度最小
        # 但上面的 best = max(... -min_sim) 等效于 argmin(min_sim)
        selected.append(best)
        remaining.remove(best)
        # 更新剩余分子的 min_sim
        new_sims = DataStructs.BulkTanimotoSimilarity(fps[best],
                                                       [fps[i] for i in remaining])
        for idx_r, i in enumerate(remaining):
            sim = new_sims[idx_r] if idx_r < len(new_sims) else 0.0
            if sim < min_sim[i]:
                min_sim[i] = sim

    return [mol_list[i] for i in sorted(selected[:n])]


# =============================================================================
# 主程序
# =============================================================================
print("=" * 65)
print("  数据库抽样脚本")
print("=" * 65)

# 加载主数据库
_input_path = INPUT_NPY
if not os.path.isabs(_input_path):
    cand = os.path.join(_HERE, _input_path)
    _input_path = cand if os.path.isfile(cand) else _input_path

mol_list = _load_npy(_input_path)
total    = len(mol_list)
n_target = _resolve_size(total)
print(f"\n[INFO] 主数据库分子数: {total}")
print(f"[INFO] 抽样目标数:     {n_target}")
print(f"[INFO] 随机种子列表:   {SEEDS}")
print(f"[INFO] 输出目录:       {os.path.abspath(OUTPUT_DIR)}\n")

if n_target >= total:
    print("[WARN] 目标数量 ≥ 总数，将直接使用全量数据库。")

results_log = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    print(f"{'─'*65}")
    print(f"  Seed = {seed}")
    print(f"{'─'*65}")

    if METHOD_RANDOM:
        print("  [随机抽样] ...")
        sub = sample_random(mol_list, n_target, rng)
        p   = _save_npy(sub, 'random', n_target, seed)
        results_log.append({'method': 'random', 'seed': seed, 'n': len(sub), 'path': p})

    if METHOD_SYSTEMATIC:
        print("  [系统抽样] ...")
        sub = sample_systematic(mol_list, n_target)
        p   = _save_npy(sub, 'systematic', n_target, seed)
        results_log.append({'method': 'systematic', 'seed': seed, 'n': len(sub), 'path': p})

    if METHOD_STRATIFIED:
        print("  [分层抽样] ...")
        sub = sample_stratified(mol_list, n_target, rng)
        p   = _save_npy(sub, 'stratified', n_target, seed)
        results_log.append({'method': 'stratified', 'seed': seed, 'n': len(sub), 'path': p})

    if METHOD_LHS:
        print("  [正交抽样 LHS] ...")
        sub = sample_lhs(mol_list, n_target, rng)
        p   = _save_npy(sub, 'lhs', n_target, seed)
        results_log.append({'method': 'lhs', 'seed': seed, 'n': len(sub), 'path': p})

    if METHOD_DIVERSITY:
        print("  [多样性抽样 MaxMin] ...")
        sub = sample_diversity_maxmin(mol_list, n_target, rng)
        p   = _save_npy(sub, 'diversity', n_target, seed)
        results_log.append({'method': 'diversity', 'seed': seed, 'n': len(sub), 'path': p})

# 写抽样日志
import csv
log_path = os.path.join(OUTPUT_DIR, 'sampling_log.csv')
with open(log_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['method', 'seed', 'n', 'path'])
    w.writeheader()
    w.writerows(results_log)

print(f"\n{'='*65}")
print(f"  抽样完成！共产生 {len(results_log)} 个子数据集")
print(f"  日志文件: {log_path}")
print(f"{'='*65}")
