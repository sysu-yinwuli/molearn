#!/usr/bin/env python3
# usemodel.py  —— 用已训练模型对新数据做推理
# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

PREDICT_NPY   = 'hjf-add-fp.npy'                      # 待预测 npy
OUTPUT_CSV    = 'wavelength-add-pred.csv'              # 输出 CSV
MODEL_DIR     = '../4machineLearing/results/seed_42'   # 训练输出目录（含 models/ 等子目录）
MODEL_NAME    = 'GradientBoosting'                     # 模型名（不含 .joblib）
CONFIG_TXT    = '../4machineLearing/config-full-1.txt' # 与训练时相同的配置文件

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
_HERE = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.normpath(os.path.join(_HERE, '..', '4machineLearing'))
sys.path.insert(0, _ML_DIR)
from feature_utils import (load_npy, extract_features, clean_features,
                            load_config, parse_flags, resolve_path, DimReducer)

# ── 路径解析 ───────────────────────────────────────────────────────────────────
_model_dir   = resolve_path(MODEL_DIR,   _HERE)
_config_path = resolve_path(CONFIG_TXT,  _HERE)
_npy_path    = resolve_path(PREDICT_NPY, _HERE)

model_path    = os.path.join(_model_dir, 'models',  f'{MODEL_NAME}.joblib')
cols_path     = os.path.join(_model_dir, 'training_columns.pkl')
scaler_path   = os.path.join(_model_dir, 'y_scaler.pkl')
reducer_path  = os.path.join(_model_dir, 'dim_reducer.pkl')

# ── 加载推理资源 ───────────────────────────────────────────────────────────────
for path, label in [(model_path,  'model'),
                    (cols_path,   'training_columns'),
                    (scaler_path, 'y_scaler')]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[usemodel] {label} 文件不存在: {path}\n"
            "请检查 MODEL_DIR / MODEL_NAME 配置，或确认训练脚本已正常运行并保存了所有文件。"
        )

pipe        = joblib.load(model_path)
train_cols  = joblib.load(cols_path)
y_scaler    = joblib.load(scaler_path)
print(f"[INFO] 模型: {MODEL_NAME}   期望特征维度: {len(train_cols)}")

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
# 特征开关取 config 中每个 flag 的第一个值（与训练时该文件对应的开关一致）
flags_single = parse_flags(config, n_files=1)

data  = load_npy(_npy_path)
names = [d.get('name', f'sample_{i}') for i, d in enumerate(data)]
ys    = np.array([d.get('y', np.nan) for d in data], dtype=np.float64)

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

# ── 预测 & 逆标准化 ────────────────────────────────────────────────────────────
y_pred_scaled = pipe.predict(X_df)
y_pred        = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# ── 输出 CSV ───────────────────────────────────────────────────────────────────
out_path = resolve_path(OUTPUT_CSV, _HERE)
pd.DataFrame({
    'sample_name':  names,
    'y_true':       ys,
    'y_pred':       y_pred,
}).to_csv(out_path, index=False, float_format='%.6f')

print(f"[INFO] 推理完成，结果已写入 {out_path}")
