#!/usr/bin/env python3
# ablation_study.py  —— 描述符消融实验 + 多种可视化
# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据配置（与 ml-m-full.py 保持一致）----------
CONFIG_TXT = 'config-full-1.txt'

# ---------- 消融目标模型 ----------
# 消融实验只用一个代表性模型做评分，建议选你最关注的那个
ABLATION_MODEL   = 'RandomForest'

# ---------- 实验模式 ----------
# 'single'     : 每次只剔除一种描述符（所有开启的描述符各剔除一次），看单独影响
# 'sequential' : 从影响最小的描述符开始，依次累积剔除（贪心前向消融）
# 'both'       : 同时执行以上两种模式
ABLATION_MODE    = 'both'

# ---------- 评估设置 ----------
ABLATION_SEED       = 42     # 数据划分 seed
ABLATION_TEST_SIZE  = 0.2    # 测试集比例
ABLATION_CV         = 5      # 交叉验证折数（用于 sequential 模式的贡献度打分）

# ---------- 输出目录 ----------
ABLATION_OUT_DIR    = 'ablation_results'

# =============================================================================
# ====================== 作图配置（可按需微调） =================================
# =============================================================================

# ---------- 通用画布设置 ----------
PLOT_DPI       = 150          # 图片分辨率
PLOT_STYLE     = 'whitegrid'  # seaborn 风格: 'whitegrid'|'darkgrid'|'ticks'|'white'
PLOT_PALETTE   = 'tab10'      # 颜色方案: 'tab10'|'Set2'|'husl'|'viridis'|'coolwarm'
PLOT_FONT_SIZE = 11           # 基础字体大小

# ---------- 启用哪些图 ----------
PLOT_BAR           = True    # 柱状图：single 消融 MAE/R² 对比
PLOT_GROUPED_BAR   = True    # 分组柱状图：MAE + R² 同时展示（双 Y 轴）
PLOT_LINE          = True    # 点线图：sequential 消融趋势（MAE & R²）
PLOT_AREA          = True    # 面积/山脉图：sequential 趋势的填充版本
PLOT_HEATMAP       = True    # 热力图：single 模式各描述符的 MAE/R²/Delta 对比
PLOT_PIE           = True    # 饼状图：各描述符对性能下降的贡献占比（single 模式）
PLOT_RADAR         = True    # 雷达图：各描述符综合得分

# ---------- 柱状图细节 ----------
BAR_FIGSIZE        = (10, 5)  # (宽, 高) 英寸
BAR_METRIC         = 'r2'     # 'r2' | 'mae'
BAR_SHOW_BASELINE  = True     # 是否画全量描述符的基线
BAR_BASELINE_COLOR = '#e74c3c'
BAR_BAR_COLOR      = '#3498db'
BAR_EDGE_COLOR     = 'white'
BAR_ANNOTATE       = True     # 柱顶标注数值

# ---------- 点线图细节 ----------
LINE_FIGSIZE       = (11, 5)
LINE_R2_COLOR      = '#2980b9'
LINE_MAE_COLOR     = '#e67e22'
LINE_MARKER        = 'o'      # 点形状: 'o'|'s'|'^'|'D'
LINE_MARKER_SIZE   = 7
LINE_SHOW_REMOVED  = True     # X 轴标签显示被剔除的描述符名

# ---------- 面积图细节 ----------
AREA_FIGSIZE       = (11, 5)
AREA_ALPHA         = 0.25     # 填充透明度

# ---------- 热力图细节 ----------
HEATMAP_FIGSIZE    = (9, 4)
HEATMAP_CMAP_POS   = 'Blues'   # R² 热力图配色（正向指标）
HEATMAP_CMAP_NEG   = 'Reds'    # MAE 热力图配色（负向指标，值越大越红）
HEATMAP_ANNOT      = True      # 单元格内显示数值
HEATMAP_FMT        = '.3f'     # 数值格式

# ---------- 饼状图细节 ----------
PIE_FIGSIZE        = (7, 7)
PIE_AUTOPCT        = '%1.1f%%' # 百分比格式
PIE_STARTANGLE     = 90
PIE_EXPLODE_MAX    = True      # 最大贡献者稍微突出

# ---------- 雷达图细节 ----------
RADAR_FIGSIZE      = (7, 7)
RADAR_FILL_ALPHA   = 0.20
RADAR_LINE_WIDTH   = 2

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os
import sys
import copy
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── 导入公共工具 ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from feature_utils import (load_npy, extract_features, build_header,
                            clean_features, load_config, parse_flags, resolve_path,
                            FEATURE_FLAG_KEYS, _DESCRIPTOR_MAP)

# ── 导入 sklearn ──────────────────────────────────────────────────────────────
from sklearn.linear_model import (LinearRegression, Ridge, LassoCV, ElasticNetCV,
                                   HuberRegressor, BayesianRidge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor,
                               BaggingRegressor, HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

try:
    import seaborn as sns
    sns.set_style(PLOT_STYLE)
    sns.set_palette(PLOT_PALETTE)
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    print("[WARN] seaborn 未安装，将使用 matplotlib 默认样式（pip install seaborn）")

plt.rcParams['font.size'] = PLOT_FONT_SIZE
plt.rcParams['axes.titlesize'] = PLOT_FONT_SIZE + 1
plt.rcParams['figure.dpi'] = PLOT_DPI

# ── 导入 MODEL_PARAMS（从 ml-m-full.py 复用，或此处内嵌一份精简版）─────────────
# 为避免强依赖 ml-m-full.py，这里内嵌与其完全一致的精简参数
_ALPHA_SPACE_AB = np.logspace(-4, 1, 20)

_MODEL_PARAMS = {
    "LinearRegression":    {},
    "Ridge":               {"model__alpha": 10.0},
    "LassoCV":             {"model__cv": 5, "model__alphas": _ALPHA_SPACE_AB,
                            "model__max_iter": 10000, "model__random_state": 42},
    "ElasticNetCV":        {"model__cv": 5, "model__alphas": _ALPHA_SPACE_AB,
                            "model__l1_ratio": 0.5, "model__max_iter": 10000,
                            "model__random_state": 42},
    "HuberRegressor":      {"model__epsilon": 1.35, "model__alpha": 1e-4,
                            "model__max_iter": 300},
    "BayesianRidge":       {"model__max_iter": 300, "model__tol": 1e-3},
    "DecisionTree":        {"model__max_depth": 5,  "model__random_state": 42},
    "RandomForest":        {"model__n_estimators": 100, "model__max_depth": 5,
                            "model__random_state": 42, "model__n_jobs": -1},
    "ExtraTreesRegressor": {"model__n_estimators": 100, "model__max_depth": 5,
                            "model__random_state": 42, "model__n_jobs": -1},
    "GradientBoosting":    {"model__n_estimators": 100, "model__learning_rate": 0.1,
                            "model__max_depth": 3,  "model__random_state": 42},
    "HistGBR":             {"model__max_iter": 200, "model__learning_rate": 0.1,
                            "model__max_depth": 5,  "model__random_state": 42},
    "AdaBoost":            {"model__n_estimators": 100, "model__learning_rate": 0.1,
                            "model__random_state": 42},
    "BaggingRegressor":    {"model__n_estimators": 20,  "model__random_state": 42,
                            "model__n_jobs": -1},
    "SVR":                 {"model__kernel": "rbf", "model__C": 100,
                            "model__gamma": "scale", "model__epsilon": 0.01},
    "KNeighbors":          {"model__n_neighbors": 5, "model__n_jobs": -1},
    "MLP":                 {"model__hidden_layer_sizes": (100,), "model__activation": "tanh",
                            "model__solver": "adam",  "model__learning_rate_init": 1e-3,
                            "model__max_iter": 1000,  "model__early_stopping": True,
                            "model__validation_fraction": 0.1, "model__n_iter_no_change": 20,
                            "model__tol": 1e-4,       "model__random_state": 42},
}

_ALL_BASE_PIPELINES = {
    "LinearRegression":    Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    "Ridge":               Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
    "LassoCV":             Pipeline([('scaler', StandardScaler()), ('model', LassoCV())]),
    "ElasticNetCV":        Pipeline([('scaler', StandardScaler()), ('model', ElasticNetCV())]),
    "HuberRegressor":      Pipeline([('scaler', StandardScaler()), ('model', HuberRegressor())]),
    "BayesianRidge":       Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())]),
    "DecisionTree":        Pipeline([('model', DecisionTreeRegressor())]),
    "RandomForest":        Pipeline([('model', RandomForestRegressor())]),
    "ExtraTreesRegressor": Pipeline([('model', ExtraTreesRegressor())]),
    "GradientBoosting":    Pipeline([('model', GradientBoostingRegressor())]),
    "HistGBR":             Pipeline([('model', HistGradientBoostingRegressor())]),
    "AdaBoost":            Pipeline([('model', AdaBoostRegressor())]),
    "BaggingRegressor":    Pipeline([('model', BaggingRegressor())]),
    "SVR":                 Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
    "KNeighbors":          Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())]),
    "MLP":                 Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor())]),
}

def _get_pipeline(model_name: str) -> Pipeline:
    """返回带参数注入的深拷贝 Pipeline。"""
    pipe = copy.deepcopy(_ALL_BASE_PIPELINES[model_name])
    params = _MODEL_PARAMS.get(model_name, {})
    if params:
        pipe.set_params(**params)
    return pipe

# ─────────────────────────────────────────────────────────────────────────────
# 工具：根据 flags 字典构建列名前缀 → 列索引范围映射
# ─────────────────────────────────────────────────────────────────────────────
def _build_desc_col_ranges(header: list) -> dict:
    """
    将列名列表按前缀分组，返回
    {desc_label: [col_idx, ...]}
    desc_label 格式：'<prefix>_<file_idx>'  例如 'rdkit_0', 'extra_1'
    """
    ranges = {}
    for i, col in enumerate(header):
        # 列名格式：  prefix_fileidx_colnum  或  fileidx_extra_name
        parts = col.split('_')
        if len(parts) >= 2:
            # extra 列名格式 "fileidx_xxxx"，以数字开头
            if parts[0].isdigit():
                key = f"extra_{parts[0]}"
            else:
                # 其他：prefix_fileidx_...
                key = f"{parts[0]}_{parts[1]}"
        else:
            key = col
        ranges.setdefault(key, []).append(i)
    return ranges

# ─────────────────────────────────────────────────────────────────────────────
# 工具：训练 + 评估单次（在标准化标签空间做预测，然后逆变换）
# ─────────────────────────────────────────────────────────────────────────────
def _train_eval(X_tr, y_tr_s, X_te, y_te, y_scaler, model_name):
    """返回 (r2, mae, mse)，失败时返回 (nan, nan, nan)。"""
    try:
        pipe = _get_pipeline(model_name)
        pipe.fit(X_tr, y_tr_s)
        y_pred = y_scaler.inverse_transform(
            pipe.predict(X_te).reshape(-1, 1)
        ).flatten()
        r2  = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        mse = mean_squared_error(y_te, y_pred)
        return r2, mae, mse
    except Exception as e:
        print(f"    [WARN] 训练失败: {e}")
        return np.nan, np.nan, np.nan

def _cv_score(X_tr, y_tr_s, model_name):
    """交叉验证 R²，用于 sequential 模式快速排序描述符贡献。"""
    try:
        pipe = _get_pipeline(model_name)
        scores = cross_val_score(pipe, X_tr, y_tr_s,
                                  cv=ABLATION_CV, scoring='r2', n_jobs=-1)
        return float(scores.mean())
    except Exception:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 颜色辅助
# ─────────────────────────────────────────────────────────────────────────────
def _get_colors(n):
    """从配置的 palette 取 n 种颜色。"""
    if _HAS_SNS:
        return sns.color_palette(PLOT_PALETTE, n)
    cmap = plt.get_cmap('tab10')
    return [cmap(i / max(n - 1, 1)) for i in range(n)]

# =============================================================================
# ── 1. 读取配置 & 加载数据 ────────────────────────────────────────────────────
# =============================================================================
print("=" * 65)
print("  描述符消融实验")
print("=" * 65)

_config_path = resolve_path(CONFIG_TXT, _HERE)
config = load_config(_config_path)

print("\n[配置]")
for k, v in config.items():
    print(f"  {k}: {v}")

ml_npy  = [p.strip() for p in config['npy_path'].split(',')]
n_files = len(ml_npy)
datas   = [load_npy(resolve_path(p, _HERE)) for p in ml_npy]

sample_counts = [len(d) for d in datas]
if len(set(sample_counts)) != 1:
    raise ValueError(f"所有 .npy 样本数量必须一致，当前: {sample_counts}")

flags          = parse_flags(config, n_files)
features_raw, labels = extract_features(datas, flags)
features       = clean_features(features_raw)
header         = build_header(datas, flags)

if len(header) != features.shape[1]:
    raise RuntimeError(f"列名数量 ({len(header)}) 与特征维度 ({features.shape[1]}) 不一致")

X_all = pd.DataFrame(features, columns=header)
y_all = pd.Series(labels)
print(f"\n[INFO] 特征矩阵: {X_all.shape}  有效样本: {(~y_all.isna()).sum()}")

# 构建描述符分组（label → col_indices）
desc_groups = _build_desc_col_ranges(header)
active_descs = sorted(desc_groups.keys())
print(f"\n[INFO] 检测到 {len(active_descs)} 个活跃描述符组: {active_descs}")

# 数据划分
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_all, y_all, test_size=ABLATION_TEST_SIZE, random_state=ABLATION_SEED
)
y_scaler   = StandardScaler()
y_train_s  = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_np  = y_test.values

# 全量基线
print(f"\n[BASELINE] 使用全部描述符训练 {ABLATION_MODEL} ...")
base_r2, base_mae, base_mse = _train_eval(
    X_train_full, y_train_s, X_test_full, y_test_np, y_scaler, ABLATION_MODEL
)
print(f"  Baseline  R²={base_r2:.4f}  MAE={base_mae:.4f}  MSE={base_mse:.4f}")

# 输出目录
os.makedirs(ABLATION_OUT_DIR, exist_ok=True)

# =============================================================================
# ── 2. SINGLE 消融：每次剔除一种描述符 ───────────────────────────────────────
# =============================================================================
single_records = []

if ABLATION_MODE in ('single', 'both'):
    print(f"\n{'─'*65}")
    print("  [模式 1] SINGLE 消融：依次剔除单个描述符组")
    print(f"{'─'*65}")

    for desc in active_descs:
        drop_cols = desc_groups[desc]
        keep_cols = [c for c in range(X_all.shape[1]) if c not in drop_cols]

        X_tr = X_train_full.iloc[:, keep_cols]
        X_te = X_test_full.iloc[:, keep_cols]

        r2, mae, mse = _train_eval(X_tr, y_train_s, X_te, y_test_np, y_scaler, ABLATION_MODEL)
        delta_r2  = r2  - base_r2
        delta_mae = mae - base_mae

        single_records.append({
            'descriptor':     desc,
            'n_features_dropped': len(drop_cols),
            'r2':             r2,
            'mae':            mae,
            'mse':            mse,
            'delta_r2':       delta_r2,
            'delta_mae':      delta_mae,
        })
        print(f"  剔除 {desc:20s}  R²={r2:.4f} (Δ{delta_r2:+.4f})  "
              f"MAE={mae:.4f} (Δ{delta_mae:+.4f})")

    single_df = pd.DataFrame(single_records).sort_values('delta_r2')
    csv_path  = os.path.join(ABLATION_OUT_DIR, 'single_ablation.csv')
    single_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  结果已保存: {csv_path}")

# =============================================================================
# ── 3. SEQUENTIAL 消融：贪心累积剔除 ──────────────────────────────────────────
# =============================================================================
seq_records = []

if ABLATION_MODE in ('sequential', 'both'):
    print(f"\n{'─'*65}")
    print("  [模式 2] SEQUENTIAL 消融：贪心依次剔除影响最小的描述符")
    print(f"{'─'*65}")

    remaining  = list(active_descs)           # 尚未剔除的描述符
    removed_so_far: list[str] = []            # 已剔除的描述符（有序）
    current_keep = list(range(X_all.shape[1]))  # 当前保留的列索引

    # 记录初始状态（全量）
    seq_records.append({
        'step':             0,
        'removed_desc':     '（全量）',
        'removed_so_far':   '',
        'n_features_remain': len(current_keep),
        'r2':               base_r2,
        'mae':              base_mae,
        'mse':              base_mse,
    })

    while len(remaining) > 1:
        # ── 对每个剩余描述符打分：剔除后的交叉验证 R² ──
        scores = {}
        X_tr_cur = X_train_full.iloc[:, current_keep]

        for desc in remaining:
            drop_cols = desc_groups[desc]
            keep_try  = [c for c in current_keep if c not in drop_cols]
            X_try     = X_train_full.iloc[:, keep_try]
            scores[desc] = _cv_score(X_try, y_train_s, ABLATION_MODEL)

        # 选贡献最小（剔除后 CV 分下降最小）的描述符
        best_desc = max(scores, key=lambda d: scores[d] if not np.isnan(scores[d]) else -1e9)

        # 真正剔除
        drop_cols     = desc_groups[best_desc]
        current_keep  = [c for c in current_keep if c not in drop_cols]
        remaining.remove(best_desc)
        removed_so_far.append(best_desc)

        X_tr = X_train_full.iloc[:, current_keep]
        X_te = X_test_full.iloc[:, current_keep]

        r2, mae, mse = _train_eval(X_tr, y_train_s, X_te, y_test_np, y_scaler, ABLATION_MODEL)
        step = len(removed_so_far)

        seq_records.append({
            'step':              step,
            'removed_desc':      best_desc,
            'removed_so_far':    ' → '.join(removed_so_far),
            'n_features_remain': len(current_keep),
            'r2':                r2,
            'mae':               mae,
            'mse':               mse,
        })
        print(f"  Step {step:02d}  剔除 {best_desc:20s}  "
              f"R²={r2:.4f}  MAE={mae:.4f}  剩余特征={len(current_keep)}")

    seq_df   = pd.DataFrame(seq_records)
    csv_path = os.path.join(ABLATION_OUT_DIR, 'sequential_ablation.csv')
    seq_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  结果已保存: {csv_path}")

    # 写消融顺序摘要
    order_path = os.path.join(ABLATION_OUT_DIR, 'removal_order.txt')
    with open(order_path, 'w', encoding='utf-8') as f:
        f.write(f"# Sequential 消融顺序（模型: {ABLATION_MODEL}，Seed: {ABLATION_SEED}）\n")
        f.write(f"# 全量基线  R²={base_r2:.6f}  MAE={base_mae:.6f}\n\n")
        for rec in seq_records[1:]:
            f.write(f"Step {rec['step']:02d}: 剔除 {rec['removed_desc']:<20s} "
                    f"→ R²={rec['r2']:.6f}  MAE={rec['mae']:.6f}  "
                    f"剩余特征={rec['n_features_remain']}\n")
    print(f"  消融顺序已保存: {order_path}")

# =============================================================================
# ── 4. 作图 ───────────────────────────────────────────────────────────────────
# =============================================================================
print(f"\n{'─'*65}")
print("  生成图表 ...")
print(f"{'─'*65}")

# ── 辅助：保存图片 ────────────────────────────────────────────────────────────
def _save(fig, fname):
    path = os.path.join(ABLATION_OUT_DIR, fname)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 图 1：柱状图 —— single 消融，按 BAR_METRIC 排序
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_BAR and ABLATION_MODE in ('single', 'both') and single_records:
    df = single_df.copy()
    metric = BAR_METRIC.lower()
    ylabel = 'R²' if metric == 'r2' else 'MAE'
    df_sorted = df.sort_values(metric, ascending=(metric == 'mae'))

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    bars = ax.bar(df_sorted['descriptor'], df_sorted[metric],
                  color=BAR_BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=0.8)

    if BAR_SHOW_BASELINE:
        bval = base_r2 if metric == 'r2' else base_mae
        ax.axhline(bval, color=BAR_BASELINE_COLOR, linestyle='--', linewidth=1.5,
                   label=f'Baseline ({ylabel}={bval:.3f})')
        ax.legend(fontsize=PLOT_FONT_SIZE - 1)

    if BAR_ANNOTATE:
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if metric == 'r2' else 0),
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=PLOT_FONT_SIZE - 2, rotation=0)

    ax.set_xlabel('剔除的描述符组', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE)
    ax.set_title(f'Single 消融 — {ylabel}（模型: {ABLATION_MODEL}）',
                 fontsize=PLOT_FONT_SIZE + 1)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    _save(fig, 'plot_bar_single.png')

# ─────────────────────────────────────────────────────────────────────────────
# 图 2：分组柱状图 —— 双 Y 轴同时展示 R² + MAE
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_GROUPED_BAR and ABLATION_MODE in ('single', 'both') and single_records:
    df = single_df.sort_values('delta_r2')   # delta_r2 最负的（影响最大）排左
    labels_x = list(df['descriptor'])
    x = np.arange(len(labels_x))
    w = 0.38

    colors = _get_colors(2)
    fig, ax1 = plt.subplots(figsize=BAR_FIGSIZE)
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, df['r2'],  width=w, color=colors[0],
                    alpha=0.85, label='R²', edgecolor='white')
    bars2 = ax2.bar(x + w / 2, df['mae'], width=w, color=colors[1],
                    alpha=0.85, label='MAE', edgecolor='white')

    ax1.axhline(base_r2,  color=colors[0], linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.axhline(base_mae, color=colors[1], linestyle='--', linewidth=1.2, alpha=0.7)

    ax1.set_xlabel('剔除的描述符组', fontsize=PLOT_FONT_SIZE)
    ax1.set_ylabel('R²',  color=colors[0], fontsize=PLOT_FONT_SIZE)
    ax2.set_ylabel('MAE', color=colors[1], fontsize=PLOT_FONT_SIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_x, rotation=30, ha='right')
    ax1.set_title(f'Single 消融 — R² & MAE 分组柱状图（{ABLATION_MODEL}）',
                  fontsize=PLOT_FONT_SIZE + 1)
    handles = [mpatches.Patch(color=colors[0], label='R²'),
               mpatches.Patch(color=colors[1], label='MAE')]
    ax1.legend(handles=handles, loc='lower left', fontsize=PLOT_FONT_SIZE - 1)
    fig.tight_layout()
    _save(fig, 'plot_grouped_bar.png')

# ─────────────────────────────────────────────────────────────────────────────
# 图 3：点线图 —— sequential 消融趋势
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_LINE and ABLATION_MODE in ('sequential', 'both') and seq_records:
    df = seq_df.copy()
    steps = df['step'].tolist()
    r2s   = df['r2'].tolist()
    maes  = df['mae'].tolist()

    # X 轴标签：step 0 = 全量，后续 = 被剔除的描述符名
    if LINE_SHOW_REMOVED:
        xlabels = ['全量'] + [r['removed_desc'] for r in seq_records[1:]]
    else:
        xlabels = [str(s) for s in steps]

    fig, ax1 = plt.subplots(figsize=LINE_FIGSIZE)
    ax2 = ax1.twinx()

    ax1.plot(steps, r2s, color=LINE_R2_COLOR, marker=LINE_MARKER,
             markersize=LINE_MARKER_SIZE, linewidth=2, label='R²')
    ax2.plot(steps, maes, color=LINE_MAE_COLOR, marker=LINE_MARKER,
             markersize=LINE_MARKER_SIZE, linewidth=2, linestyle='--', label='MAE')

    ax1.set_xticks(steps)
    ax1.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=PLOT_FONT_SIZE - 2)
    ax1.set_xlabel('剔除步骤（→ 被剔除的描述符）', fontsize=PLOT_FONT_SIZE)
    ax1.set_ylabel('R²',  color=LINE_R2_COLOR,  fontsize=PLOT_FONT_SIZE)
    ax2.set_ylabel('MAE', color=LINE_MAE_COLOR, fontsize=PLOT_FONT_SIZE)
    ax1.set_title(f'Sequential 消融趋势 — 点线图（{ABLATION_MODEL}）',
                  fontsize=PLOT_FONT_SIZE + 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best', fontsize=PLOT_FONT_SIZE - 1)
    fig.tight_layout()
    _save(fig, 'plot_line_sequential.png')

# ─────────────────────────────────────────────────────────────────────────────
# 图 4：面积图 —— sequential 消融趋势（填充版）
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_AREA and ABLATION_MODE in ('sequential', 'both') and seq_records:
    df = seq_df.copy()
    steps = df['step'].tolist()
    r2s   = df['r2'].tolist()
    maes  = df['mae'].tolist()
    colors = _get_colors(2)

    if LINE_SHOW_REMOVED:
        xlabels = ['全量'] + [r['removed_desc'] for r in seq_records[1:]]
    else:
        xlabels = [str(s) for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=AREA_FIGSIZE, sharex=True)

    ax1.plot(steps, r2s,  color=colors[0], linewidth=2, marker='o', markersize=5)
    ax1.fill_between(steps, r2s,  min(r2s),  alpha=AREA_ALPHA, color=colors[0])
    ax1.axhline(base_r2, color='gray', linestyle=':', linewidth=1)
    ax1.set_ylabel('R²', fontsize=PLOT_FONT_SIZE)
    ax1.set_title(f'Sequential 消融趋势 — 面积图（{ABLATION_MODEL}）',
                  fontsize=PLOT_FONT_SIZE + 1)

    ax2.plot(steps, maes, color=colors[1], linewidth=2, marker='s', markersize=5)
    ax2.fill_between(steps, maes, min(maes), alpha=AREA_ALPHA, color=colors[1])
    ax2.axhline(base_mae, color='gray', linestyle=':', linewidth=1)
    ax2.set_ylabel('MAE', fontsize=PLOT_FONT_SIZE)
    ax2.set_xlabel('剔除步骤', fontsize=PLOT_FONT_SIZE)
    ax2.set_xticks(steps)
    ax2.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=PLOT_FONT_SIZE - 2)

    fig.tight_layout()
    _save(fig, 'plot_area_sequential.png')

# ─────────────────────────────────────────────────────────────────────────────
# 图 5：热力图 —— single 消融，展示各描述符的 R² / MAE / ΔR² / ΔMAE
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_HEATMAP and ABLATION_MODE in ('single', 'both') and single_records:
    df = single_df.set_index('descriptor')[['r2', 'mae', 'delta_r2', 'delta_mae']]
    df.columns = ['R²', 'MAE', 'ΔR²', 'ΔMAE']

    # 两张子图：R²/ΔR² 用蓝（越大越好）；MAE/ΔMAE 用红（越大越差）
    fig, axes = plt.subplots(1, 2, figsize=HEATMAP_FIGSIZE)

    if _HAS_SNS:
        sns.heatmap(df[['R²', 'ΔR²']], ax=axes[0], cmap=HEATMAP_CMAP_POS,
                    annot=HEATMAP_ANNOT, fmt=HEATMAP_FMT,
                    linewidths=0.5, linecolor='white')
        sns.heatmap(df[['MAE', 'ΔMAE']], ax=axes[1], cmap=HEATMAP_CMAP_NEG,
                    annot=HEATMAP_ANNOT, fmt=HEATMAP_FMT,
                    linewidths=0.5, linecolor='white')
    else:
        # 无 seaborn 时用 imshow 代替
        for ax_idx, (cols, cmap) in enumerate(
                [(['R²', 'ΔR²'], HEATMAP_CMAP_POS),
                 (['MAE', 'ΔMAE'], HEATMAP_CMAP_NEG)]):
            mat = df[cols].values.astype(float)
            axes[ax_idx].imshow(mat, aspect='auto', cmap=cmap)
            axes[ax_idx].set_xticks(range(len(cols)))
            axes[ax_idx].set_xticklabels(cols)
            axes[ax_idx].set_yticks(range(len(df)))
            axes[ax_idx].set_yticklabels(df.index, fontsize=PLOT_FONT_SIZE - 2)
            if HEATMAP_ANNOT:
                for r in range(mat.shape[0]):
                    for c in range(mat.shape[1]):
                        axes[ax_idx].text(c, r, f'{mat[r,c]:{HEATMAP_FMT}}',
                                          ha='center', va='center',
                                          fontsize=PLOT_FONT_SIZE - 3)

    axes[0].set_title('R² & ΔR²', fontsize=PLOT_FONT_SIZE)
    axes[1].set_title('MAE & ΔMAE', fontsize=PLOT_FONT_SIZE)
    fig.suptitle(f'Single 消融热力图（{ABLATION_MODEL}）', fontsize=PLOT_FONT_SIZE + 1)
    fig.tight_layout()
    _save(fig, 'plot_heatmap_single.png')

# ─────────────────────────────────────────────────────────────────────────────
# 图 6：饼状图 —— 各描述符对 R² 下降的贡献占比（single 模式）
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_PIE and ABLATION_MODE in ('single', 'both') and single_records:
    df = single_df.copy()
    # 贡献度 = baseline - 剔除后的 r2（正值=剔除后性能下降，即该描述符有贡献）
    df['contribution'] = base_r2 - df['r2']
    # 只保留正贡献（性能确实下降的描述符）
    df_pos = df[df['contribution'] > 0].copy()
    if df_pos.empty:
        print("  [INFO] 没有描述符剔除后导致性能下降，跳过饼图")
    else:
        total = df_pos['contribution'].sum()
        df_pos['pct'] = df_pos['contribution'] / total

        colors_pie = _get_colors(len(df_pos))
        explode = None
        if PIE_EXPLODE_MAX:
            max_idx = df_pos['contribution'].idxmax()
            explode = [0.08 if i == max_idx else 0
                       for i in df_pos.index]

        fig, ax = plt.subplots(figsize=PIE_FIGSIZE)
        wedges, texts, autotexts = ax.pie(
            df_pos['contribution'],
            labels=df_pos['descriptor'],
            autopct=PIE_AUTOPCT,
            startangle=PIE_STARTANGLE,
            colors=colors_pie,
            explode=explode,
            pctdistance=0.82,
        )
        for at in autotexts:
            at.set_fontsize(PLOT_FONT_SIZE - 2)
        ax.set_title(f'描述符对 R² 下降的贡献占比\n（{ABLATION_MODEL}，基线 R²={base_r2:.3f}）',
                     fontsize=PLOT_FONT_SIZE + 1)
        fig.tight_layout()
        _save(fig, 'plot_pie_contribution.png')

# ─────────────────────────────────────────────────────────────────────────────
# 图 7：雷达图 —— 每种描述符的综合得分（归一化）
# ─────────────────────────────────────────────────────────────────────────────
if PLOT_RADAR and ABLATION_MODE in ('single', 'both') and single_records:
    df = single_df.copy()
    # 综合得分：将 R²（越大越好）和 1/MAE（越大越好）归一化后取均值
    df['score_r2']  = (df['r2']  - df['r2'].min())  / (df['r2'].max()  - df['r2'].min() + 1e-9)
    df['score_mae'] = 1 - (df['mae'] - df['mae'].min()) / (df['mae'].max() - df['mae'].min() + 1e-9)
    df['score']     = (df['score_r2'] + df['score_mae']) / 2.0

    categories = list(df['descriptor'])
    N = len(categories)
    if N < 3:
        print("  [INFO] 描述符组数 < 3，跳过雷达图")
    else:
        values = list(df['score'])
        angles = [2 * np.pi * i / N for i in range(N)] + [0]
        values_plot = values + [values[0]]

        colors_rad = _get_colors(1)
        fig, ax = plt.subplots(figsize=RADAR_FIGSIZE, subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, 'o-', linewidth=RADAR_LINE_WIDTH,
                color=colors_rad[0])
        ax.fill(angles, values_plot, alpha=RADAR_FILL_ALPHA, color=colors_rad[0])
        ax.set_thetagrids(np.degrees(angles[:-1]), categories,
                          fontsize=PLOT_FONT_SIZE - 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'描述符综合得分雷达图\n（{ABLATION_MODEL}）',
                     fontsize=PLOT_FONT_SIZE + 1, pad=20)
        fig.tight_layout()
        _save(fig, 'plot_radar.png')

# =============================================================================
# ── 5. 汇总报告 ───────────────────────────────────────────────────────────────
# =============================================================================
summary_path = os.path.join(ABLATION_OUT_DIR, 'ablation_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  描述符消融实验 汇总报告\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"模型:    {ABLATION_MODEL}\n")
    f.write(f"模式:    {ABLATION_MODE}\n")
    f.write(f"Seed:    {ABLATION_SEED}\n")
    f.write(f"配置文件: {CONFIG_TXT}\n\n")
    f.write(f"[基线] 全量描述符\n")
    f.write(f"  R²={base_r2:.6f}  MAE={base_mae:.6f}  MSE={base_mse:.6f}\n\n")

    if ABLATION_MODE in ('single', 'both') and single_records:
        f.write("[Single 消融结果（按 ΔR² 从小到大，影响最大在前）]\n")
        f.write(f"  {'描述符':<20s} {'R²':>8s} {'ΔR²':>9s} {'MAE':>9s} {'ΔMAE':>9s}  {'删除列数':>6s}\n")
        f.write("  " + "-" * 70 + "\n")
        for _, row in single_df.iterrows():
            f.write(f"  {row['descriptor']:<20s} {row['r2']:>8.4f} "
                    f"{row['delta_r2']:>+9.4f} {row['mae']:>9.4f} "
                    f"{row['delta_mae']:>+9.4f}  {int(row['n_features_dropped']):>6d}\n")

    if ABLATION_MODE in ('sequential', 'both') and seq_records:
        f.write("\n[Sequential 消融顺序]\n")
        for rec in seq_records:
            f.write(f"  Step {rec['step']:02d}: 剔除 {rec['removed_desc']:<20s} "
                    f"→ R²={rec['r2']:.4f}  MAE={rec['mae']:.4f}  "
                    f"剩余特征={rec['n_features_remain']}\n")

    f.write("\n[输出图表]\n")
    for fname in sorted(os.listdir(ABLATION_OUT_DIR)):
        if fname.endswith('.png'):
            f.write(f"  {fname}\n")

print(f"\n  汇总报告已保存: {summary_path}")
print(f"\n{'='*65}")
print("  消融实验完成！")
print(f"  所有输出位于: {os.path.abspath(ABLATION_OUT_DIR)}/")
print(f"{'='*65}")
