#!/usr/bin/env python3
# usemodel.py  —— 用已训练模型对新数据做推理（支持回归 + 分类双模式）
# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

PREDICT_NPY   = 'hjf-add-fp.npy'                      # 待预测 npy
OUTPUT_CSV    = 'wavelength-add-pred.csv'              # 输出 CSV
MODEL_DIR     = '../4machineLearing/results/seed_42'   # 训练输出目录（含 models/ 等子目录）
MODEL_NAME    = 'GradientBoosting'                     # 回归模型名（不含 .joblib）
                                                       # 分类时示例：'RandomForestClassifier'
CONFIG_TXT    = '../4machineLearing/config-full-1.txt' # 与训练时相同的配置文件

# 任务类型：'auto' = 自动读取 task_type.pkl（推荐）
#           'regression' | 'classification' = 手动指定（覆盖自动检测）
TASK_TYPE     = 'auto'

# 降维设置（与训练时保持一致）
# 若训练时启用了降维，此处自动检测 MODEL_DIR/dim_reducer.pkl 并应用
# 设为 False 可强制跳过（通常无需修改）
USE_DIM_REDUCTION = True

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib

# ── 公共工具（训练目录）────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.normpath(os.path.join(_HERE, '..', '4machineLearing'))
sys.path.insert(0, _ML_DIR)
from feature_utils import (load_npy, extract_features, clean_features,
                            load_config, parse_flags, resolve_path, DimReducer)

# ── 路径解析 ───────────────────────────────────────────────────────────────────
_model_dir   = resolve_path(MODEL_DIR,   _HERE)
_config_path = resolve_path(CONFIG_TXT,  _HERE)
_npy_path    = resolve_path(PREDICT_NPY, _HERE)

model_path      = os.path.join(_model_dir, 'models',  f'{MODEL_NAME}.joblib')
cols_path       = os.path.join(_model_dir, 'training_columns.pkl')
task_type_path  = os.path.join(_model_dir, 'task_type.pkl')
y_scaler_path   = os.path.join(_model_dir, 'y_scaler.pkl')
classes_path    = os.path.join(_model_dir, 'classes.pkl')
reducer_path    = os.path.join(_model_dir, 'dim_reducer.pkl')

# ── 加载推理资源（必须文件）────────────────────────────────────────────────────
for path, label in [(model_path, 'model'), (cols_path, 'training_columns')]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[usemodel] {label} 文件不存在: {path}\n"
            "请检查 MODEL_DIR / MODEL_NAME 配置，或确认训练脚本已正常运行并保存了所有文件。"
        )

pipe       = joblib.load(model_path)
train_cols = joblib.load(cols_path)
print(f"[INFO] 模型: {MODEL_NAME}   期望特征维度: {len(train_cols)}")

# ── 任务类型检测 ────────────────────────────────────────────────────────────────
if TASK_TYPE == 'auto':
    if os.path.isfile(task_type_path):
        _task_type = joblib.load(task_type_path)
        print(f"[INFO] 任务类型（自动检测）: {_task_type}")
    elif os.path.isfile(y_scaler_path):
        _task_type = 'regression'
        print(f"[INFO] 任务类型（由 y_scaler.pkl 推断）: {_task_type}")
    else:
        _task_type = 'regression'
        print(f"[WARN] 无法检测任务类型，默认 regression")
else:
    _task_type = TASK_TYPE
    print(f"[INFO] 任务类型（手动指定）: {_task_type}")

_IS_CLF = (_task_type == 'classification')

# ── 加载回归/分类专属资源 ───────────────────────────────────────────────────────
y_scaler = None
classes  = None

if _IS_CLF:
    if os.path.isfile(classes_path):
        classes = joblib.load(classes_path)
        print(f"[INFO] 分类类别: {classes}")
    else:
        print(f"[WARN] classes.pkl 不存在，输出列只含预测标签，无类别概率")
else:
    if not os.path.isfile(y_scaler_path):
        raise FileNotFoundError(
            f"[usemodel] 回归任务需要 y_scaler.pkl，但文件不存在: {y_scaler_path}\n"
            "请确认训练脚本已正常运行并保存了 y_scaler.pkl。"
        )
    y_scaler = joblib.load(y_scaler_path)

# ── 加载降维器（若存在）───────────────────────────────────────────────────────
dim_reducer = None
if USE_DIM_REDUCTION and os.path.isfile(reducer_path):
    dim_reducer = DimReducer.load(reducer_path)
    print(f"[INFO] 已加载降维器: method={dim_reducer.method}")
elif USE_DIM_REDUCTION:
    print(f"[INFO] 未检测到降维器文件（{reducer_path}），跳过降维")

# ── 读取配置 & 特征提取 ────────────────────────────────────────────────────────
config = load_config(_config_path)

# 预测场景只有单个 npy，固定 n_files=1
flags_single = parse_flags(config, n_files=1)

data  = load_npy(_npy_path)
names = [d.get('name', f'sample_{i}') for i, d in enumerate(data)]

# 标签（如有）—— 回归读 float，分类读 int
if _IS_CLF:
    ys = np.array([d.get('y', np.nan) for d in data])
    try:
        ys = ys.astype(int)
    except (ValueError, TypeError):
        ys = ys.astype(float)
else:
    ys = np.array([d.get('y', np.nan) for d in data], dtype=np.float64)

features_raw, _ = extract_features([data], flags_single)
features        = clean_features(features_raw)

# ── 降维（若训练时启用）─────────────────────────────────────────────────────────
if dim_reducer is not None:
    print(f"[INFO] 应用降维变换: {features.shape[1]} → ", end='')
    features = dim_reducer.transform(features)
    print(f"{features.shape[1]} 维")

# ── 列对齐（用 training_columns 确保特征顺序与训练一致）─────────────────────
X_df = pd.DataFrame(features, columns=train_cols)
print(f"[INFO] 预测样本数: {len(X_df)}")

# ── 预测 ───────────────────────────────────────────────────────────────────────
if _IS_CLF:
    # ── 分类预测 ───────────────────────────────────────────────────────────────
    y_pred_label = pipe.predict(X_df)

    out_dict = {
        'sample_name': names,
        'y_true':      ys,
        'y_pred':      y_pred_label,
    }

    # 概率列（如果模型支持）
    if hasattr(pipe, 'predict_proba') and classes is not None:
        try:
            proba = pipe.predict_proba(X_df)
            for i, cls in enumerate(classes):
                out_dict[f'prob_class_{cls}'] = proba[:, i]
        except Exception as _e:
            print(f"[WARN] 概率预测失败: {_e}")

    df_out = pd.DataFrame(out_dict)

else:
    # ── 回归预测 ───────────────────────────────────────────────────────────────
    y_pred_scaled = pipe.predict(X_df)
    y_pred        = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    df_out = pd.DataFrame({
        'sample_name': names,
        'y_true':      ys,
        'y_pred':      y_pred,
    })

# ── 输出 CSV ───────────────────────────────────────────────────────────────────
out_path = resolve_path(OUTPUT_CSV, _HERE)
df_out.to_csv(out_path, index=False, float_format='%.6f')

print(f"[INFO] 推理完成，结果已写入 {out_path}")
if _IS_CLF:
    from sklearn.metrics import accuracy_score
    _valid = ~np.isnan(ys.astype(float))
    if _valid.sum() > 0:
        acc = accuracy_score(ys[_valid], y_pred_label[_valid])
        print(f"[INFO] 有标签样本准确率: {acc:.4f}  (n={_valid.sum()})")
