#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pearson_filter.py — 描述符皮尔逊相关性共线性剔除
================================================================
功能：
  读取带描述符的 .npy 文件，计算所有描述符之间的皮尔逊相关系数，
  对相关系数绝对值超过阈值的描述符对，保留方差更大的一个，剔除另一个。
  输出：
    ① 剔除后的 .npy 文件（描述符字段中移除共线描述符列）
    ② 共线性剔除报告 Excel（pearson_removal_report.xlsx）
    ③ 相关性热图 PNG（可选，仅对 ≤500 维描述符生成）

独立使用（可不依赖 molearn_run.py）：
    python 3descriptor/pearson_filter.py

与 create_by_fp.py 集成：
    在 create_by_fp.py CONFIG 中设置 'pearson_filter': True
    → 在描述符计算完成后自动执行过滤

参数调整：修改 CONFIG 区域即可，代码体勿改。
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations

# ======================================================================
# ========== 配置区域 ==========
# ======================================================================
CONFIG = {
    # ---- 路径 ----
    # 输入：带描述符的 .npy 文件（create_by_fp.py 的输出）
    'input_npy':     'poly-all-fp.npy',

    # 输出：过滤后的 .npy 文件名（留空则自动命名为 <input>_pearson.npy）
    'output_npy':    '',

    # 剔除报告 Excel 路径（留空则自动命名为 pearson_removal_report.xlsx）
    'report_xlsx':   '',

    # 输出目录（留空则与 input_npy 同目录）
    'output_dir':    '',

    # ---- 过滤参数 ----
    # 皮尔逊相关系数绝对值阈值：|r| > threshold 的描述符对中剔除方差小的
    'threshold':     0.95,

    # 用哪些描述符字段做共线性分析（list of str）；留空则自动取全部描述符字段
    # 可选值：'rdkit_descriptor', 'maccs_descriptor', 'morgan_descriptor',
    #         'atompair_descriptor', 'torsion_descriptor', 'avalon_descriptor',
    #         'soap_descriptor', 'acsf_descriptor', 'mbtr_descriptor',
    #         'mordred_descriptor'
    'descriptor_fields': [],

    # ---- 热图选项 ----
    # 是否生成相关性热图（仅当描述符总维数 ≤ heatmap_max_dim 时生成）
    'gen_heatmap':       True,
    'heatmap_max_dim':   300,   # 超过此维数跳过热图（避免图片过大/超慢）
    'heatmap_filename':  '',    # 留空则自动命名

    # ---- 运行模式 ----
    # 'filter'  : 完整执行（过滤 + 保存 .npy + 报告 + 热图）
    # 'report'  : 仅分析和出报告，不修改 .npy
    # 'heatmap' : 仅生成热图，不过滤不保存
    'mode':              'filter',
}
# ======================================================================
# 以下代码不再出现任何硬编码参数
# ======================================================================

# ── 已知的描述符字段列表（按 create_by_fp.py 的字段名）
_ALL_DESCRIPTOR_FIELDS = [
    'rdkit_descriptor',
    'maccs_descriptor',
    'morgan_descriptor',
    'atompair_descriptor',
    'torsion_descriptor',
    'avalon_descriptor',
    'soap_descriptor',
    'acsf_descriptor',
    'mbtr_descriptor',
    'mordred_descriptor',
]

# ── 字段名 → 列名前缀映射
_FIELD_PREFIX = {
    'rdkit_descriptor':     'rdkit',
    'maccs_descriptor':     'maccs',
    'morgan_descriptor':    'morgan',
    'atompair_descriptor':  'AP',
    'torsion_descriptor':   'TT',
    'avalon_descriptor':    'Avalon',
    'soap_descriptor':      'SOAP',
    'acsf_descriptor':      'ACSF',
    'mbtr_descriptor':      'MBTR',
    'mordred_descriptor':   'mordred',
}


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────

def _load_npy(path: str):
    """加载 .npy，返回 (molecules_list, raw_dict)。"""
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        return list(raw['successful']), raw
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        return list(raw), {'successful': list(raw)}
    raise ValueError(f"无法识别 .npy 格式: {path}")


def _detect_descriptor_fields(molecules: list) -> list:
    """自动检测 molecules 中存在的描述符字段。"""
    found = []
    if not molecules:
        return found
    d0 = molecules[0]
    for field in _ALL_DESCRIPTOR_FIELDS:
        if field in d0 and d0[field]:
            found.append(field)
    return found


def _build_matrix(molecules: list, fields: list):
    """
    将 molecules 中指定 fields 的描述符拼合为矩阵和列名列表。
    返回：
      X    : np.ndarray  (n_samples, n_features)
      cols : list[str]   列名列表
    """
    rows, cols = [], []
    cols_built = False
    for mol in molecules:
        row = []
        for field in fields:
            vals = mol.get(field, [])
            row.extend(vals)
            if not cols_built:
                prefix = _FIELD_PREFIX.get(field, field)
                # 尝试用 .mordred_f / .rdkit_f 等已有的列名
                names_field = field.replace('_descriptor', '_f')
                feat_names = mol.get(names_field, None)
                if feat_names and len(feat_names) == len(vals):
                    cols.extend([f"{prefix}_{n}" for n in feat_names])
                else:
                    cols.extend([f"{prefix}_{i}" for i in range(len(vals))])
        if not cols_built:
            cols_built = True
        rows.append(row)

    X = np.array(rows, dtype=np.float64)
    # 清洗：clip + inf→nan → 列均值填充
    X = np.clip(X, -1e30, 1e30)
    X[~np.isfinite(X)] = np.nan
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X, cols


def pearson_collinear_filter(X: np.ndarray, col_names: list, threshold: float):
    """
    皮尔逊相关系数共线性剔除算法。

    策略：
      1. 计算所有特征对之间的皮尔逊 |r|
      2. 对 |r| > threshold 的每一对，标记**方差较小**的那列为"待剔除"
      3. 重复直到无新标记（等价于一次性贪心：先按方差降序排，从高方差开始保留）

    返回：
      keep_mask    : np.ndarray[bool]  True=保留，False=剔除
      report_rows  : list[dict]        每对共线关系的报告行
    """
    n_feat = X.shape[1]
    variances = np.var(X, axis=0)

    # 按方差降序排列索引（方差大的优先保留）
    order = np.argsort(-variances)

    keep = np.ones(n_feat, dtype=bool)
    report_rows = []

    # 计算相关矩阵（按块计算，节约内存）
    # 对大维数做分块计算
    BLOCK = 1000
    n_blk = (n_feat + BLOCK - 1) // BLOCK

    # 标准化 X（计算 Pearson）
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_std = (X - X.mean(axis=0)) / std

    removed_by = {}  # col_idx → col_idx (被谁触发剔除的)

    for i_pos, i in enumerate(order):
        if not keep[i]:
            continue
        # 与所有 j > i (按 order 排列中在 i 之后) 计算相关
        for j_pos in range(i_pos + 1, n_feat):
            j = order[j_pos]
            if not keep[j]:
                continue
            # Pearson r = (X_std[:, i] · X_std[:, j]) / n
            r = float(np.dot(X_std[:, i], X_std[:, j])) / X.shape[0]
            if abs(r) > threshold:
                # 保留 i (高方差), 剔除 j (低方差)
                keep[j] = False
                removed_by[j] = i
                report_rows.append({
                    '保留特征':       col_names[i],
                    '保留特征方差':   float(variances[i]),
                    '剔除特征':       col_names[j],
                    '剔除特征方差':   float(variances[j]),
                    '皮尔逊|r|':      round(abs(r), 6),
                    '阈值':           threshold,
                    '触发原因':       f"|r|={abs(r):.4f} > 阈值 {threshold}",
                })

    return keep, report_rows


def _apply_filter_to_molecules(molecules: list, fields: list,
                                keep_mask: np.ndarray, col_names: list) -> list:
    """
    将 keep_mask 应用到每个分子的描述符字段，
    按照 keep_mask 截取各字段对应的位段，返回新的分子列表（深复制）。
    """
    import copy

    # 建立每个字段在完整向量中的起止索引
    # 先用第一个分子确定各字段维度
    d0 = molecules[0]
    field_slices = {}
    ptr = 0
    for field in fields:
        length = len(d0.get(field, []))
        field_slices[field] = (ptr, ptr + length)
        ptr += length

    new_mols = []
    for mol in molecules:
        m = copy.deepcopy(mol)
        for field, (s, e) in field_slices.items():
            if field not in m:
                continue
            old_vals = m[field]
            # keep_mask 中对应段
            mask_seg = keep_mask[s:e]
            m[field] = [v for v, k in zip(old_vals, mask_seg) if k]
            # 同步更新列名字段（如 rdkit_f / mordred_f 等）
            names_field = field.replace('_descriptor', '_f')
            if names_field in m and m[names_field]:
                old_names = m[names_field]
                m[names_field] = [n for n, k in zip(old_names, mask_seg) if k]
        new_mols.append(m)
    return new_mols


def _gen_heatmap(X: np.ndarray, col_names: list, out_path: str, threshold: float):
    """生成皮尔逊相关系数热图（仅对 ≤500 维）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        corr = np.corrcoef(X.T)
        n = len(col_names)
        # 截短列名，避免热图过于拥挤
        short_names = [n[:12] if len(n) > 12 else n for n in col_names]

        fig_size = max(10, n // 10)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            pd.DataFrame(corr, index=short_names, columns=short_names),
            ax=ax, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1,
            xticklabels=(n <= 100), yticklabels=(n <= 100),
            cbar_kws={'label': 'Pearson r'},
        )
        ax.set_title(f'描述符皮尔逊相关系数矩阵（阈值 |r|>{threshold}，共{n}维）', pad=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  热图已保存 → {out_path}")
    except ImportError as e:
        print(f"  [WARN] 热图生成失败（缺少依赖 {e}），跳过")
    except Exception as e:
        print(f"  [WARN] 热图生成异常: {e}，跳过")


# ──────────────────────────────────────────────────────────────────────
# 主函数（可作为库函数调用，也可直接运行）
# ──────────────────────────────────────────────────────────────────────

def run_pearson_filter(cfg: dict = None) -> dict:
    """
    执行皮尔逊相关性过滤主流程。

    参数
    ----
    cfg : dict  配置字典（留空则使用模块级 CONFIG）

    返回
    ----
    dict with keys:
      'keep_mask'      : np.ndarray[bool]
      'col_names'      : list[str]
      'n_before'       : int
      'n_after'        : int
      'n_removed'      : int
      'removed_count'  : int
      'output_npy'     : str  (mode='filter' 时)
      'report_xlsx'    : str
    """
    if cfg is None:
        cfg = CONFIG

    input_npy   = cfg['input_npy']
    threshold   = float(cfg.get('threshold', 0.95))
    mode        = cfg.get('mode', 'filter')
    fields_cfg  = cfg.get('descriptor_fields', [])

    # ── 路径解析 ──────────────────────────────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(input_npy):
        _try = os.path.join(_script_dir, input_npy)
        input_npy = _try if os.path.isfile(_try) else input_npy

    out_dir = cfg.get('output_dir', '') or os.path.dirname(os.path.abspath(input_npy))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_npy))[0]

    out_npy = cfg.get('output_npy') or os.path.join(out_dir, f"{base}_pearson.npy")
    if not os.path.isabs(out_npy):
        out_npy = os.path.join(out_dir, os.path.basename(out_npy))

    report_xlsx = cfg.get('report_xlsx') or os.path.join(out_dir, 'pearson_removal_report.xlsx')
    if not os.path.isabs(report_xlsx):
        report_xlsx = os.path.join(out_dir, os.path.basename(report_xlsx))

    heatmap_file = cfg.get('heatmap_filename') or os.path.join(out_dir, f"{base}_pearson_heatmap.png")
    if not os.path.isabs(heatmap_file):
        heatmap_file = os.path.join(out_dir, os.path.basename(heatmap_file))

    # ── 加载数据 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  pearson_filter.py — 描述符共线性剔除")
    print(f"{'='*60}")
    print(f"  输入 npy  : {input_npy}")
    print(f"  阈值 |r|> : {threshold}")
    print(f"  模式      : {mode}")

    molecules, raw_dict = _load_npy(input_npy)
    print(f"  样本数    : {len(molecules)}")

    # ── 确定需要分析的描述符字段 ──────────────────────────────────────
    if fields_cfg:
        fields = [f for f in fields_cfg if f in _ALL_DESCRIPTOR_FIELDS]
    else:
        fields = _detect_descriptor_fields(molecules)

    if not fields:
        print("  [ERROR] 未检测到任何描述符字段，请确认 input_npy 包含描述符数据。")
        sys.exit(1)

    print(f"  分析字段  : {', '.join(fields)}")

    # ── 构建描述符矩阵 ────────────────────────────────────────────────
    print("\n  正在构建描述符矩阵...")
    X, col_names = _build_matrix(molecules, fields)
    n_before = X.shape[1]
    print(f"  总描述符维数: {n_before}")

    # ── 删除常数列（方差=0，无信息量，必须先剔除）────────────────────
    variances = np.var(X, axis=0)
    nonconst_mask = variances > 0
    n_const = int((~nonconst_mask).sum())
    if n_const > 0:
        print(f"  剔除常数列: {n_const} 个（方差=0）")
        X          = X[:, nonconst_mask]
        col_names  = [c for c, k in zip(col_names, nonconst_mask) if k]

    # ── 皮尔逊过滤 ────────────────────────────────────────────────────
    print(f"\n  执行皮尔逊相关性过滤（阈值={threshold}）...")
    keep_mask_local, report_rows = pearson_collinear_filter(X, col_names, threshold)
    n_after   = int(keep_mask_local.sum())
    n_removed = n_before - n_after

    print(f"  过滤前维数: {n_before}")
    print(f"  常数列剔除: {n_const}")
    print(f"  共线性剔除: {n_before - n_const - n_after}")
    print(f"  过滤后维数: {n_after}")
    print(f"  剔除总计  : {n_removed}")

    # ── 生成报告 ──────────────────────────────────────────────────────
    if report_rows:
        df_report = pd.DataFrame(report_rows)
    else:
        df_report = pd.DataFrame(columns=['保留特征', '保留特征方差', '剔除特征',
                                           '剔除特征方差', '皮尔逊|r|', '阈值', '触发原因'])

    # 汇总表（保留和剔除特征名单）
    kept_names    = [c for c, k in zip(col_names, keep_mask_local) if k]
    removed_names = [c for c, k in zip(col_names, keep_mask_local) if not k]
    n_pad = max(len(kept_names), len(removed_names))
    df_summary = pd.DataFrame({
        '保留特征列表': kept_names + [''] * (n_pad - len(kept_names)),
        '剔除特征列表': removed_names + [''] * (n_pad - len(removed_names)),
    })

    with pd.ExcelWriter(report_xlsx, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='过滤汇总', index=False)
        df_report.to_excel(writer, sheet_name='共线对明细', index=False)
        # 写统计信息
        stats_df = pd.DataFrame([
            {'指标': '输入文件',        '值': input_npy},
            {'指标': '输出文件',        '值': out_npy},
            {'指标': '阈值 |r|>',       '值': threshold},
            {'指标': '过滤前描述符维数', '值': n_before},
            {'指标': '常数列剔除',       '值': n_const},
            {'指标': '共线性剔除',       '值': n_before - n_const - n_after},
            {'指标': '过滤后描述符维数', '值': n_after},
            {'指标': '剔除比例',         '值': f"{n_removed/n_before*100:.1f}%"},
        ])
        stats_df.to_excel(writer, sheet_name='统计摘要', index=False)

    print(f"\n  剔除报告  → {report_xlsx}")

    # ── 热图 ──────────────────────────────────────────────────────────
    if cfg.get('gen_heatmap', True) and mode != 'report':
        if n_before <= cfg.get('heatmap_max_dim', 300):
            print(f"  生成相关性热图（{n_before}维）...")
            _gen_heatmap(X, col_names, heatmap_file, threshold)
        else:
            print(f"  [跳过热图] 维数 {n_before} > heatmap_max_dim {cfg.get('heatmap_max_dim', 300)}")

    if mode == 'report':
        print("\n  report 模式：仅生成报告，未修改 .npy 文件")
        return {
            'keep_mask': keep_mask_local,
            'col_names': col_names,
            'n_before': n_before,
            'n_after': n_after,
            'n_removed': n_removed,
            'removed_count': n_removed,
            'report_xlsx': report_xlsx,
        }

    # ── 将 keep_mask 映射回原始完整向量（含常数列）────────────────────
    # nonconst_mask 是完整向量中的非常数位
    # keep_mask_local 是在非常数子集中的保留位
    # 综合得到原始维度中的 keep 位
    full_keep = nonconst_mask.copy()
    # 对 nonconst 区域内再应用 keep_mask_local
    nonconst_idx = np.where(nonconst_mask)[0]
    for local_i, global_i in enumerate(nonconst_idx):
        full_keep[global_i] = keep_mask_local[local_i]

    # ── 修改分子数据 ──────────────────────────────────────────────────
    if mode == 'filter':
        print("\n  正在修改分子描述符字段...")
        # 重新构建完整keep_mask（包含常数列位置，均设为False）
        new_molecules = _apply_filter_to_molecules(molecules, fields, full_keep,
                                                    col_names + [''] * n_const)

        # 更新 raw_dict 保存
        new_dict = dict(raw_dict)
        new_dict['successful']      = new_molecules
        new_dict['pearson_filter']  = {
            'threshold':       threshold,
            'n_before':        n_before,
            'n_after':         n_after,
            'n_removed':       n_removed,
            'removed_fields':  fields,
            'kept_col_names':  kept_names,
        }

        np.save(out_npy, new_dict, allow_pickle=True)
        print(f"  过滤后 npy → {out_npy}")

    result = {
        'keep_mask':    full_keep,
        'col_names':    col_names,
        'n_before':     n_before,
        'n_after':      n_after,
        'n_removed':    n_removed,
        'removed_count': n_removed,
        'output_npy':   out_npy,
        'report_xlsx':  report_xlsx,
    }
    print(f"\n  ✓ 皮尔逊过滤完成！保留 {n_after}/{n_before} 个描述符\n")
    return result


# ──────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='皮尔逊相关性共线性描述符剔除',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pearson_filter.py
  python pearson_filter.py --input data/descriptors/dataset-fp.npy --threshold 0.90
  python pearson_filter.py --input x.npy --output x_clean.npy --mode report
  python pearson_filter.py --threshold 0.95 --no-heatmap
        """
    )
    parser.add_argument('--input',     '-i', type=str, default='',
                        help='输入 .npy 文件路径（覆盖 CONFIG）')
    parser.add_argument('--output',    '-o', type=str, default='',
                        help='输出 .npy 文件路径（覆盖 CONFIG）')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='皮尔逊阈值，如 0.95（覆盖 CONFIG）')
    parser.add_argument('--report',         type=str, default='',
                        help='剔除报告 xlsx 路径（覆盖 CONFIG）')
    parser.add_argument('--outdir',    '-d', type=str, default='',
                        help='输出目录（覆盖 CONFIG）')
    parser.add_argument('--mode',           type=str, default='',
                        choices=['filter', 'report', 'heatmap'],
                        help='运行模式（覆盖 CONFIG）')
    parser.add_argument('--no-heatmap',     action='store_true',
                        help='禁止生成热图（覆盖 CONFIG）')
    parser.add_argument('--fields',         nargs='+', default=[],
                        help='指定分析的描述符字段名（空格分隔）')
    args = parser.parse_args()

    # 命令行参数覆盖 CONFIG
    cfg = dict(CONFIG)
    if args.input:     cfg['input_npy']          = args.input
    if args.output:    cfg['output_npy']          = args.output
    if args.threshold is not None: cfg['threshold'] = args.threshold
    if args.report:    cfg['report_xlsx']         = args.report
    if args.outdir:    cfg['output_dir']          = args.outdir
    if args.mode:      cfg['mode']                = args.mode
    if args.no_heatmap: cfg['gen_heatmap']        = False
    if args.fields:    cfg['descriptor_fields']   = args.fields

    run_pearson_filter(cfg)
