#!/usr/bin/env python3
# ml-m-predict-final.py  —— 必须加载 y_scaler.pkl 的外推脚本
# ========= 配置区（全部可改） =========
PREDICT_NPY     = 'hjf-add-fp.npy'      # 待外推 npy
OUTPUT_CSV      = 'wavelength-add-pred.csv'  # 输出 csv
MODEL_PATH      = 'wavelength-GradientBoosting.joblib'  # 模型
COLS_PATH       = 'wavelength-training_columns.pkl'  # 列顺序
Y_SCALER_PATH   = 'wavelength-y_scaler.pkl'  # 必须存在！
CONFIG_TXT      = 'config-full-rdkit.txt'     # 训练配置（只拿特征开关）
# =====================================

import numpy as np
import pandas as pd
import joblib, os, sys

# ---------------- 内嵌：解析特征开关 ----------------
def parse_list(key, default='0'):
    val = config_dict.get(key, default)
    return int(val.split(',')[0])   # 单文件场景

# ---------------- 内嵌：数据清洗 ----------------
def clean_features(feat: np.ndarray) -> np.ndarray:
    if feat.size == 0:
        raise RuntimeError('[ERROR] 特征矩阵为 0 列！')
    X = feat.copy()
    X = np.clip(X, -1e30, 1e30)
    X[~np.isfinite(X)] = np.nan
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X

# ---------------- 1. 读训练配置 ----------------
config_dict = {}
with open(CONFIG_TXT, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            k, v = line.split(':', 1)
            config_dict[k.strip()] = v.strip()

# ---------------- 2. 加载外部资源 ----------------
if not os.path.exists(Y_SCALER_PATH):
    raise FileNotFoundError(
        f'[{Y_SCALER_PATH}] 不存在！请先在训练脚本里补存 '
        'joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.pkl"))'
    )

pipe       = joblib.load(MODEL_PATH)
train_cols = joblib.load(COLS_PATH)
y_scaler   = joblib.load(Y_SCALER_PATH)
print(f'[INFO] 期望特征维度：{len(train_cols)}')

# ---------------- 3. 加载待预测 npy ----------------
raw = np.load(PREDICT_NPY, allow_pickle=True)
if raw.ndim == 0:
    raw = raw.item()
if isinstance(raw, dict) and 'successful' in raw:
    data = raw['successful']
elif isinstance(raw, np.ndarray) and raw.dtype == object:
    data = raw
else:
    raise ValueError(f'{PREDICT_NPY} 格式无法识别')

# ---------------- 4. 特征拼接 ----------------
if_rdkit   = parse_list('if_rdkit',   '0')
if_soap    = parse_list('if_soap',    '0')
if_acsf    = parse_list('if_acsf',    '0')
if_mordred = parse_list('if_mordred', '0')
if_maccs   = parse_list('if_maccs',   '0')
if_morgan  = parse_list('if_morgan',  '0')
if_QC      = parse_list('if_QC',      '0')
if_m       = parse_list('if_m',       '0')
if_extra   = parse_list('if_extra',   '0')

xs, ys, names = [], [], []
for idx, d in enumerate(data):
    ft = []
    if if_rdkit:   ft += d.get('rdkit_descriptor', [])
    if if_soap:    ft += d.get('soap_descriptor', [])
    if if_acsf:    ft += d.get('acsf_descriptor', [])
    if if_mordred: ft += d.get('mordred_descriptor', [])
    if if_maccs:   ft += d.get('maccs_descriptor', [])
    if if_morgan:  ft += d.get('morgan_descriptor', [])
    if if_QC:      ft += d.get('g_d', [])
    if if_extra:   ft += d.get('extra_d', [])
    if if_m:
        for l in d.get('3DMatrix', []):
            ft += l
    xs.append(ft)
    ys.append(d.get('y', np.nan))
    names.append(d.get('name', f'sample_{idx}'))

X = np.array(xs, dtype=np.float64)
y = np.array(ys, dtype=np.float64)

# ---------------- 5. 数据清洗 + 对齐列 ----------------
X_clean = clean_features(X)
X_df    = pd.DataFrame(X_clean, columns=train_cols)

# ---------------- 6. 预测 + 逆标准化 ----------------
y_pred = pipe.predict(X_df)
y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# ---------------- 7. 写 csv ----------------
pd.DataFrame({
    'sample_name': names,
    'y': y,
    'prediction_y': y_pred
}).to_csv(OUTPUT_CSV, index=False, float_format='%.6f')

print(f'[INFO] 外推完成，结果已写入 {OUTPUT_CSV}')