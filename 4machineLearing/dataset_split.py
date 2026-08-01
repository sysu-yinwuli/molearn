#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_split.py — 数据集划分工具（功能 7）
================================================================
支持三种划分策略：
  1. 全随机划分（Random Split）
  2. 分层采样划分（Stratified Split，按 y 值分位数）
  3. 骨架划分（Scaffold Split，基于分子拓扑骨架）
     支持骨架类型：
       - 'bemis_murcko'  : Bemis-Murcko 骨架（经典）
       - 'murcko_generic': 通用 Bemis-Murcko（去除原子差异）
       - 'morgan'        : Morgan 指纹（将相似指纹视为同一"骨架"）
       - 'maccs'         : MACCS Keys 簇
       - 'rdkit'         : RDKit 拓扑指纹簇

输出：
  - train.npy / valid.npy / test.npy（保持原始 dict 格式）
  - split_info.csv（记录每个分子的归属）
  - split_summary.txt

所有参数在顶部 CONFIG 区域配置。
"""

# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据输入 ----------
INPUT_NPY       = 'your_database.npy'   # 输入 npy 文件
CONFIG_TXT      = 'config-full-1.txt'   # 特征配置文件（用于解析 smiles 等字段）
OUTPUT_DIR      = 'split_output'        # 输出目录

# ---------- 划分比例 ----------
TRAIN_RATIO     = 0.8     # 训练集比例
VALID_RATIO     = 0.1     # 验证集比例（0 = 不划分验证集）
TEST_RATIO      = 0.1     # 测试集比例

# ---------- 划分策略 ----------
# 'random' | 'stratified' | 'scaffold'
SPLIT_METHOD    = 'random'

# ---------- 随机划分参数 ----------
SPLIT_SEED      = 42

# ---------- 分层划分参数 ----------
STRATIFIED_Y_FIELD  = 'y'    # 分层依据的字段（通常是目标属性）
STRATIFIED_N_BINS   = 5      # 分箱数

# ---------- 骨架划分参数 ----------
SCAFFOLD_TYPE       = 'bemis_murcko'   # 见上方说明
SCAFFOLD_SMILES_FLD = 'smiles'         # SMILES 字段名
# 指纹骨架聚类（当 SCAFFOLD_TYPE 为 morgan/maccs/rdkit 时）
SCAFFOLD_CLUSTER_N  = None    # None=按实际骨架数，int=强制聚类到 N 个簇
SCAFFOLD_FP_MORGAN_R = 2
SCAFFOLD_FP_NBITS    = 2048
# 骨架划分分配策略：'train_largest'（最大骨架组优先进训练集）
#                  'random'（骨架组随机分配）
SCAFFOLD_ASSIGN     = 'train_largest'
SCAFFOLD_SEED        = 42

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os, sys, csv, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from feature_utils import load_npy, load_config, resolve_path

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _check_ratios():
    total = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"TRAIN+VALID+TEST 比例之和必须为 1.0，当前={total:.4f}")

def _save_split(mol_list, fname, split_label):
    path = os.path.join(OUTPUT_DIR, fname)
    d = {
        'successful': mol_list,
        'failed_count': 0,
        'error_stats': {},
        'split_info': {
            'method': SPLIT_METHOD,
            'split':  split_label,
            'n':      len(mol_list),
            'source': INPUT_NPY,
        }
    }
    np.save(path, d, allow_pickle=True)
    print(f"  ✓ {split_label:6s}: {len(mol_list):5d} 分子 → {path}")
    return path


def _write_split_info(mol_list, assignments):
    """assignments: list of ('train'/'valid'/'test') 与 mol_list 等长。"""
    path = os.path.join(OUTPUT_DIR, 'split_info.csv')
    rows = [{'name': d.get('name', f'mol_{i}'),
             'y':    d.get('y', ''),
             'split': assignments[i]}
            for i, d in enumerate(mol_list)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  ✓ split_info.csv → {path}")
    return df


# ── 划分策略 ──────────────────────────────────────────────────────────────────

def split_random(mol_list):
    """全随机划分。"""
    rng    = np.random.default_rng(SPLIT_SEED)
    idx    = rng.permutation(len(mol_list))
    n      = len(mol_list)
    n_te   = max(1, round(n * TEST_RATIO))
    n_va   = max(1, round(n * VALID_RATIO)) if VALID_RATIO > 0 else 0
    n_tr   = n - n_te - n_va

    idx_tr = idx[:n_tr]
    idx_va = idx[n_tr:n_tr + n_va]
    idx_te = idx[n_tr + n_va:]

    assign = [''] * n
    for i in idx_tr: assign[i] = 'train'
    for i in idx_va: assign[i] = 'valid'
    for i in idx_te: assign[i] = 'test'

    return (sorted(idx_tr.tolist()),
            sorted(idx_va.tolist()),
            sorted(idx_te.tolist()),
            assign)


def split_stratified(mol_list):
    """分层采样划分。"""
    y_vals  = np.array([d.get(STRATIFIED_Y_FIELD, np.nan) for d in mol_list],
                        dtype=np.float64)
    valid   = np.where(~np.isnan(y_vals))[0]
    invalid = np.where(np.isnan(y_vals))[0]
    n       = len(mol_list)

    if len(valid) == 0:
        print("  [WARN] 无有效 y 值，回退到随机划分")
        return split_random(mol_list)

    # 分箱
    bins   = np.percentile(y_vals[valid],
                            np.linspace(0, 100, STRATIFIED_N_BINS + 1))
    bins   = np.unique(bins)
    labels = np.digitize(y_vals[valid], bins[1:-1])

    rng    = np.random.default_rng(SPLIT_SEED)
    idx_tr, idx_va, idx_te = [], [], []

    for b in np.unique(labels):
        bin_idx  = valid[labels == b]
        rng.shuffle(bin_idx)
        nb       = len(bin_idx)
        n_te_b   = max(1, round(nb * TEST_RATIO))
        n_va_b   = max(1, round(nb * VALID_RATIO)) if VALID_RATIO > 0 else 0
        n_tr_b   = nb - n_te_b - n_va_b
        idx_tr.extend(bin_idx[:n_tr_b].tolist())
        idx_va.extend(bin_idx[n_tr_b:n_tr_b + n_va_b].tolist())
        idx_te.extend(bin_idx[n_tr_b + n_va_b:].tolist())

    # 无效 y 的分子全部放训练集
    idx_tr.extend(invalid.tolist())

    assign = [''] * n
    for i in idx_tr: assign[i] = 'train'
    for i in idx_va: assign[i] = 'valid'
    for i in idx_te: assign[i] = 'test'

    return (sorted(idx_tr), sorted(idx_va), sorted(idx_te), assign)


def _get_scaffold(mol, scaffold_type):
    """提取分子骨架标识符（字符串）。"""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if scaffold_type == 'bemis_murcko':
        try:
            smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            return smi if smi else '__no_scaffold__'
        except Exception:
            return '__error__'
    if scaffold_type == 'murcko_generic':
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            smi  = MurckoScaffold.MakeScaffoldGeneric(core)
            from rdkit import Chem
            return Chem.MolToSmiles(smi, canonical=True) if smi else '__no_scaffold__'
        except Exception:
            return '__error__'
    # 指纹聚类类骨架 → 返回 bit 向量字符串（不实际聚类，依靠聚类步骤）
    from rdkit.Chem import AllChem, MACCSkeys, rdmolops
    try:
        if scaffold_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, SCAFFOLD_FP_MORGAN_R, SCAFFOLD_FP_NBITS)
        elif scaffold_type == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            fp = rdmolops.RDKFingerprint(mol, fpSize=SCAFFOLD_FP_NBITS)
        return fp.ToBitString()
    except Exception:
        return '__error__'


def split_scaffold(mol_list):
    """骨架划分。"""
    try:
        from rdkit import Chem, DataStructs
    except ImportError:
        print("  [WARN] RDKit 未安装，骨架划分回退到随机划分")
        return split_random(mol_list)

    n = len(mol_list)

    # ── 提取骨架 ────────────────────────────────────────────────────────────
    print(f"  提取骨架（type={SCAFFOLD_TYPE}）...")
    scaffolds   = {}    # scaffold_id → [mol_indices]
    mol_scaffold = [''] * n   # 每个分子的骨架标识

    for i, d in enumerate(mol_list):
        smi = d.get(SCAFFOLD_SMILES_FLD)
        if not smi:
            key = '__no_smiles__'
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                key = '__invalid_smiles__'
            else:
                key = _get_scaffold(mol, SCAFFOLD_TYPE)
        mol_scaffold[i] = key
        scaffolds.setdefault(key, []).append(i)

    print(f"  骨架种类: {len(scaffolds)}")

    # ── 指纹骨架聚类（可选）────────────────────────────────────────────────
    if SCAFFOLD_TYPE in ('morgan', 'maccs', 'rdkit') and SCAFFOLD_CLUSTER_N is not None:
        print(f"  进行 K-Means 聚类 → {SCAFFOLD_CLUSTER_N} 个簇 ...")
        try:
            from sklearn.cluster import MiniBatchKMeans
            # 将 bit 字符串转为 numpy 数组
            unique_scaf = list(scaffolds.keys())
            X_scaf = np.array([[int(c) for c in s] for s in unique_scaf
                               if len(s) > 1], dtype=np.uint8)
            if X_scaf.shape[0] > SCAFFOLD_CLUSTER_N:
                km = MiniBatchKMeans(n_clusters=SCAFFOLD_CLUSTER_N,
                                      random_state=SCAFFOLD_SEED, n_init=10)
                km.fit(X_scaf)
                # 重新映射
                new_scaffolds = {}
                for j, s in enumerate(unique_scaf):
                    if len(s) > 1:
                        cluster_id = f"cluster_{km.labels_[j]}"
                        new_scaffolds.setdefault(cluster_id, [])
                        new_scaffolds[cluster_id].extend(scaffolds[s])
                        for idx in scaffolds[s]:
                            mol_scaffold[idx] = cluster_id
                scaffolds = new_scaffolds
                print(f"  聚类完成: {len(scaffolds)} 个骨架簇")
        except Exception as e:
            print(f"  [WARN] 聚类失败，使用原始骨架: {e}")

    # ── 分配骨架到 train/valid/test ──────────────────────────────────────────
    scaf_list = sorted(scaffolds.keys(),
                        key=lambda s: len(scaffolds[s]), reverse=True)

    # 预算容量
    n_te = max(1, round(n * TEST_RATIO))
    n_va = max(1, round(n * VALID_RATIO)) if VALID_RATIO > 0 else 0
    n_tr = n - n_te - n_va

    rng = np.random.default_rng(SCAFFOLD_SEED)
    if SCAFFOLD_ASSIGN == 'random':
        rng.shuffle(scaf_list := list(scaf_list))

    idx_tr, idx_va, idx_te = [], [], []

    for scaf in scaf_list:
        group = scaffolds[scaf]
        if len(idx_te) < n_te:
            idx_te.extend(group)
        elif VALID_RATIO > 0 and len(idx_va) < n_va:
            idx_va.extend(group)
        else:
            idx_tr.extend(group)

    assign = ['train'] * n   # 默认全训练
    for i in idx_te: assign[i] = 'test'
    for i in idx_va: assign[i] = 'valid'
    # 剩余全为训练
    idx_tr = [i for i in range(n) if assign[i] == 'train']

    return (sorted(idx_tr), sorted(idx_va), sorted(idx_te), assign)


# =============================================================================
# 主程序
# =============================================================================
print("=" * 65)
print("  数据集划分工具")
print("=" * 65)
_check_ratios()

_npy_path = resolve_path(INPUT_NPY, _HERE)
mol_list  = load_npy(_npy_path)
n         = len(mol_list)
print(f"\n[INFO] 输入文件:  {INPUT_NPY}")
print(f"[INFO] 分子总数:  {n}")
print(f"[INFO] 划分策略:  {SPLIT_METHOD}")
print(f"[INFO] 划分比例:  train={TRAIN_RATIO}  valid={VALID_RATIO}  test={TEST_RATIO}\n")

if SPLIT_METHOD == 'random':
    idx_tr, idx_va, idx_te, assign = split_random(mol_list)
elif SPLIT_METHOD == 'stratified':
    idx_tr, idx_va, idx_te, assign = split_stratified(mol_list)
elif SPLIT_METHOD == 'scaffold':
    idx_tr, idx_va, idx_te, assign = split_scaffold(mol_list)
else:
    raise ValueError(f"未知划分方法: '{SPLIT_METHOD}'，可选: random/stratified/scaffold")

# 输出统计
print(f"\n[划分结果]")
print(f"  训练集: {len(idx_tr):5d}  ({len(idx_tr)/n:.1%})")
if VALID_RATIO > 0:
    print(f"  验证集: {len(idx_va):5d}  ({len(idx_va)/n:.1%})")
print(f"  测试集: {len(idx_te):5d}  ({len(idx_te)/n:.1%})")

# 保存
_save_split([mol_list[i] for i in idx_tr], 'train.npy', 'train')
if VALID_RATIO > 0:
    _save_split([mol_list[i] for i in idx_va], 'valid.npy', 'valid')
_save_split([mol_list[i] for i in idx_te], 'test.npy',  'test')

# split_info.csv
split_df = _write_split_info(mol_list, assign)

# 汇总报告
summary_path = os.path.join(OUTPUT_DIR, 'split_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  数据集划分报告\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"输入文件: {INPUT_NPY}\n")
    f.write(f"划分方法: {SPLIT_METHOD}\n")
    if SPLIT_METHOD == 'random':
        f.write(f"随机种子: {SPLIT_SEED}\n")
    elif SPLIT_METHOD == 'stratified':
        f.write(f"Y 字段:   {STRATIFIED_Y_FIELD}\n")
        f.write(f"分箱数:   {STRATIFIED_N_BINS}\n")
        f.write(f"随机种子: {SPLIT_SEED}\n")
    elif SPLIT_METHOD == 'scaffold':
        f.write(f"骨架类型: {SCAFFOLD_TYPE}\n")
        f.write(f"分配策略: {SCAFFOLD_ASSIGN}\n")
    f.write(f"\n比例设置: train={TRAIN_RATIO}  valid={VALID_RATIO}  test={TEST_RATIO}\n")
    f.write(f"\n[实际划分]\n")
    f.write(f"  训练集: {len(idx_tr):5d}  ({len(idx_tr)/n:.1%})\n")
    if VALID_RATIO > 0:
        f.write(f"  验证集: {len(idx_va):5d}  ({len(idx_va)/n:.1%})\n")
    f.write(f"  测试集: {len(idx_te):5d}  ({len(idx_te)/n:.1%})\n")

    # Y 值统计对比
    y_all   = np.array([d.get('y', np.nan) for d in mol_list], dtype=np.float64)
    y_tr    = y_all[idx_tr]; y_te = y_all[idx_te]
    if not np.all(np.isnan(y_all)):
        f.write(f"\n[Y 值统计对比]\n")
        for split_name, y_s in [('全量', y_all), ('训练集', y_tr), ('测试集', y_te)]:
            yv = y_s[~np.isnan(y_s)]
            if len(yv) > 0:
                f.write(f"  {split_name}: mean={yv.mean():.4f}  "
                        f"std={yv.std():.4f}  "
                        f"min={yv.min():.4f}  max={yv.max():.4f}\n")

print(f"\n  汇总报告: {summary_path}")
print(f"\n{'='*65}")
print("  数据集划分完成！")
print(f"{'='*65}")
