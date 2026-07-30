#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use_tsne-label-fixed-v2.py
- 兼容 dict/列表两种 npy 结构
- 支持复用已有降维结果
- 带 tqdm 进度条
- 修复 SIZE_INVERT：真正反转大小映射，而非反转数组顺序
- 修复图例：与主体散点大小完全一致，且动态范围放大到 30-700
"""
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

# ================= 配置区域 =================
in_npy         = 'hjf-all-fp-score.npy'
out_npy        = 'hjf-all-fp-score-tsne.npy'
IMG_FNAME      = 'hjf-all-fp-score-tsne.png'
descriptor     = 'rdkit_descriptor'
remain_num     = 2                # 2 或 3
threshold      = 0.1
COLOR_FIELD    = 'score_d'
SIZE_FIELD     = 'score_d'
REUSE_TSNE_NPY = False            # True=只读已有文件，False=重新降维
SIZE_INVERT    = False            # True=标签值越大点越小；False=标签值越大点越大
# 新增：面积上下限
SIZE_RANGE     = (1, 30)       # 主体散点与图例共用
# ==========================================

# 1. 加载数据（兼容两种格式）
_raw = np.load(in_npy, allow_pickle=True)
if _raw.ndim == 0:
    _raw = _raw.item()
data = _raw['successful'] if isinstance(_raw, dict) and 'successful' in _raw else _raw
print(f"成功加载 {len(data)} 个分子数据")

# 2. 提取描述符矩阵
try:
    descriptors = np.array([mol[descriptor] for mol in data])
except KeyError as e:
    raise KeyError(f"数据中缺少必要的键: {e}")

# 3. 修复非法字符串 → nan
descriptors = np.array(descriptors, dtype=str)
descriptors[np.isin(descriptors, ['', 'None', '#NUM!'])] = 'nan'
descriptors = descriptors.astype(float)

# 4. 零比例过滤
zero_ratios = np.mean(descriptors == 0, axis=0)
columns_to_keep = zero_ratios < threshold
descriptors = descriptors[:, columns_to_keep]
print(f"过滤后保留 {descriptors.shape[1]} 个特征")

# 5. 缺失值/无穷值处理
features_copy = descriptors.copy()
features_copy[np.isinf(features_copy)] = np.nan
column_means = np.nanmean(features_copy, axis=0)
column_means = np.array([column_means]) if np.isscalar(column_means) else column_means
column_means[np.isnan(column_means)] = 0
nan_indices = np.argwhere(np.isnan(features_copy))
for idx in nan_indices:
    row, col = idx
    features_copy[row, col] = column_means[col]

# 6. 标准化
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_copy)
descriptors = features_normalized.astype(np.float32)

# 7. t-SNE 降维 / 复用旧结果
if REUSE_TSNE_NPY and os.path.isfile(out_npy):
    print(f"发现已存在 {out_npy}，直接加载降维结果...")
    _cached = np.load(out_npy, allow_pickle=True).item()
    reduced_descriptors = np.array([mol[descriptor] for mol in _cached['successful']])
else:
    print("运行 t-SNE 降维...")
    tsne = TSNE(n_components=remain_num, perplexity=30, early_exaggeration=12,
                learning_rate=200, n_iter=1000, init='pca', random_state=0, verbose=1)
    with tqdm(total=tsne.n_iter, desc="t-SNE iter") as bar:
        def _update(*_):
            bar.update(1)
        tsne._callback = _update
        reduced_descriptors = tsne.fit_transform(descriptors)

# 8. 颜色 / 大小
def safe_float(x):
    try:
        if isinstance(x, (list, np.ndarray)):
            return float(np.mean(x)) if len(x) else 0.0
        return float(x)
    except (ValueError, TypeError):
        return float(hash(str(x)) % 100)

color_vals = np.array([safe_float(mol[COLOR_FIELD]) for mol in data])
size_vals  = np.array([safe_float(mol[SIZE_FIELD]) for mol in data])
c_min, c_max = color_vals.min(), color_vals.max()
s_min, s_max = size_vals.min(), size_vals.max()

# 9. 映射到散点大小（与图例完全同步）
s_lo, s_hi = SIZE_RANGE
sizes_raw = s_lo + (size_vals - s_min) * (s_hi - s_lo) / max(s_max - s_min, 1e-8)
if SIZE_INVERT:
    sizes = s_hi - (sizes_raw - s_lo)          # 值越大 → 面积越小
else:
    sizes = sizes_raw

# 10. 绘图
plt.figure(figsize=(12, 8))
plt.grid(False)
if remain_num == 2:
    scatter = plt.scatter(reduced_descriptors[:, 0], reduced_descriptors[:, 1],
                          c=color_vals, s=sizes, cmap='viridis', alpha=0.7,
                          edgecolor='w', linewidth=0.3)
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
else:
    ax = plt.axes(projection='3d')
    ax.grid(False)
    scatter = ax.scatter3D(reduced_descriptors[:, 0], reduced_descriptors[:, 1], reduced_descriptors[:, 2],
                           c=color_vals, s=sizes, cmap='viridis', alpha=0.7)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.set_zlabel('t-SNE 3')

plt.colorbar(scatter, pad=0.05, aspect=40).set_label(f'{COLOR_FIELD}', rotation=270, labelpad=20)

# --------  大小图例（与主体完全一致）  --------
# 取 5 个示意点
n_leg = 3
leg_sizes = np.linspace(s_lo, s_hi, n_leg)
leg_vals  = np.linspace(s_min, s_max, n_leg)
if SIZE_INVERT:
    leg_sizes = leg_sizes[::-1]          # 图例顺序也反转，视觉一致
size_legend = [plt.scatter([], [], s=ls, c='gray', alpha=0.6,
                           label=f'{lv:.2f}') for ls, lv in zip(leg_sizes, leg_vals)]
plt.legend(handles=size_legend,
           title=f'{SIZE_FIELD} → Size',
           loc='lower left', bbox_to_anchor=(0.05, 0.05), framealpha=0.7)

plt.title(f't-SNE {remain_num}D  Color={COLOR_FIELD}  Size={SIZE_FIELD}')
plt.tight_layout()
plt.savefig(IMG_FNAME, dpi=300, bbox_inches='tight')
plt.show()

# 11. 保存
if not (REUSE_TSNE_NPY and os.path.isfile(out_npy)):
    reduced_data = []
    for i, mol in enumerate(data):
        new_entry = mol.copy()
        new_entry[descriptor] = reduced_descriptors[i]
        reduced_data.append(new_entry)
    save_dict = {'successful': reduced_data,
                 'config': {'in_npy': in_npy, 'out_npy': out_npy,
                            'descriptor': descriptor, 'remain_num': remain_num,
                            'COLOR_FIELD': COLOR_FIELD, 'SIZE_FIELD': SIZE_FIELD,
                            'SIZE_INVERT': SIZE_INVERT, 'SIZE_RANGE': SIZE_RANGE}}
    np.save(out_npy, save_dict, allow_pickle=True)
    print(f"结果已保存至: {out_npy}")
else:
    print("未重新降维，未覆盖原有文件。")