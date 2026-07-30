#!/usr/bin/env python3
# ml-m-full-1.py  —— 训练 + 自动 SHAP 全合一
# ===== SHAP 关键设置（只改这里） =====
SHAP_SAMPLES      = 1000      # 用于 SHAP 的背景/解释样本数
SHAP_DO_SHAP      = True      # 总开关：True=做 SHAP，False=纯训练
SHAP_SUMMARY_PLOT = True      # 是否输出 summary_plot
SHAP_BAR_PLOT     = True      # 是否输出 bar plot
SHAP_FEATURE_PERT = 'interventional'  # TreeExplainer 扰动方式
# ===== 下面代码勿动 =====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import shap
from sklearn.linear_model import (LinearRegression, Ridge, LassoCV, ElasticNetCV)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ---------- 读取配置 ----------
config_dict = {}
with open('config-full-1.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split(':', 1)
            config_dict[key.strip()] = value.strip()

print("当前配置:")
for k, v in config_dict.items():
    print(f"{k}: {v}")

# ---------- 解析路径与开关 ----------
ml_npy = [p.strip() for p in config_dict['npy_path'].split(',')]
n_files = len(ml_npy)

def parse_list(key, default='0'):
    lst = [int(x) for x in config_dict.get(key, default).split(',')]
    if len(lst) == 1:
        lst = lst * n_files
    assert len(lst) == n_files, f"{key} 长度与 npy_path 不一致"
    return lst

if_rdkit   = parse_list('if_rdkit')
if_soap    = parse_list('if_soap')
if_acsf    = parse_list('if_acsf')
if_mordred = parse_list('if_mordred')
if_maccs   = parse_list('if_maccs')
if_morgan  = parse_list('if_morgan')
if_QC      = parse_list('if_QC')
if_m       = parse_list('if_m')
if_extra   = parse_list('if_extra')

# ---------- 加载所有 .npy ----------
datas = []
for m in ml_npy:
    raw = np.load(m, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        datas.append(raw['successful'])
    elif isinstance(raw, np.ndarray) and raw.dtype == object:
        datas.append(raw)
    else:
        raise ValueError(f"{m} 格式无法识别，请检查是否为 create_by_mmr-1.py 输出格式")
assert len({len(d) for d in datas}) == 1, "所有 .npy 样本数量必须一致"

# ---------- 特征提取 ----------
xs, ys = [], []
for idx, data in enumerate(datas):
    x_tmp, y_tmp = [], []
    for d in tqdm(data, desc=f"处理第 {idx+1} 个 .npy"):
        ft = []
        if if_rdkit[idx]:   ft += d.get('rdkit_descriptor', [])
        if if_soap[idx]:    ft += d.get('soap_descriptor', [])
        if if_acsf[idx]:    ft += d.get('acsf_descriptor', [])
        if if_mordred[idx]: ft += d.get('mordred_descriptor', [])
        if if_maccs[idx]:   ft += d.get('maccs_descriptor', [])
        if if_morgan[idx]:  ft += d.get('morgan_descriptor', [])
        if if_QC[idx]:      ft += d.get('g_d', [])
        if if_extra[idx]:   ft += d.get('extra_d', [])
        if if_m[idx]:
            for l in d.get('3DMatrix', []):
                ft += l
        x_tmp.append(ft)
        y_tmp.append(d.get('y', np.nan))
    xs.append(x_tmp)
    if not ys:
        ys = y_tmp

x_final = []
for i in range(len(xs[0])):
    x_final.append([item for j in range(n_files) for item in xs[j][i]])
features = np.array(x_final, dtype=np.float64)
labels   = np.array(ys, dtype=np.float64)
print(f"[DEBUG] 合并后特征矩阵形状: {features.shape}")

# ---------- 数据清洗 ----------
def clean_features(feat):
    if feat.size == 0:
        raise RuntimeError("[ERROR] 特征矩阵为 0 列，没有任何特征被选中！")
    feat = feat.copy()
    feat = np.clip(feat, -1e30, 1e30)
    feat[~np.isfinite(feat)] = np.nan
    col_means = np.nanmean(feat, axis=0)
    col_means[np.isnan(col_means)] = 0
    nan_mask = np.isnan(feat)
    feat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return feat

features_cleaned = clean_features(features)
X = pd.DataFrame(features_cleaned)
y = pd.Series(labels)

# ---------- 生成列名（关键修复：extra 用原名） ----------
header = []
for idx, d in enumerate(datas):
    d0 = d[0]
    if if_rdkit[idx]:   header += [f"rdkit_{idx}_{i}" for i in range(len(d0.get('rdkit_descriptor', [])))]
    if if_soap[idx]:    header += [f"soap_{idx}_{i}" for i in range(len(d0.get('soap_descriptor', [])))]
    if if_acsf[idx]:    header += [f"acsf_{idx}_{i}" for i in range(len(d0.get('acsf_descriptor', [])))]
    if if_mordred[idx]: header += [f"mordred_{idx}_{i}" for i in range(len(d0.get('mordred_descriptor', [])))]
    if if_maccs[idx]:   header += [f"maccs_{idx}_{i}" for i in range(len(d0.get('maccs_descriptor', [])))]
    if if_morgan[idx]:  header += [f"morgan_{idx}_{i}" for i in range(len(d0.get('morgan_descriptor', [])))]
    if if_QC[idx]:      header += [f"QC_{idx}_{i}" for i in range(len(d0.get('g_d', [])))]
    if if_extra[idx]:
        # ⬇️ 用 Excel 表头而非数字序号
        extra_names = d0.get('name_of_extra', [])
        header += [f"{idx}_{name}" for name in extra_names]
    if if_m[idx]:
        flat_len = sum(len(l) for l in d0.get('3DMatrix', []))
        header += [f"m_{idx}_{i}" for i in range(flat_len)]

X.columns = header        # 关键：让 DataFrame 列名就是人类可读名称
print(f"[DEBUG] 列名生成完成，共 {len(header)} 维")

# ---------- 模型定义 ----------
models = {
    "LinearRegression":  Pipeline([('scaler', StandardScaler()),
                                   ('model', LinearRegression())]),
    "Ridge":             Pipeline([('scaler', StandardScaler()),
                                   ('model', Ridge(alpha=10.0))]),
    "LassoCV":           Pipeline([('scaler', StandardScaler()),
                                   ('model', LassoCV(cv=5, alphas=np.logspace(-4, 1, 20),
                                                     max_iter=10000, random_state=42))]),
    "ElasticNetCV":      Pipeline([('scaler', StandardScaler()),
                                   ('model', ElasticNetCV(cv=5, alphas=np.logspace(-4, 1, 20),
                                                           l1_ratio=0.5, max_iter=10000, random_state=42))]),
    "DecisionTree":      Pipeline([('model', DecisionTreeRegressor(max_depth=5, random_state=42))]),
    "RandomForest":      Pipeline([('model', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))]),
    "GradientBoosting":  Pipeline([('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                                        max_depth=3, random_state=42))]),
    "SVR":               Pipeline([('scaler', StandardScaler()),
                                   ('model', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01))]),
    "KNeighbors":        Pipeline([('scaler', StandardScaler()),
                                   ('model', KNeighborsRegressor(n_neighbors=5))]),
    "MLP":               Pipeline([('scaler', StandardScaler()),
                                   ('model', MLPRegressor(
                                           hidden_layer_sizes=(100,),
                                           activation='tanh',
                                           solver='adam',
                                           learning_rate_init=1e-3,
                                           max_iter=1000,
                                           early_stopping=True,
                                           validation_fraction=0.1,
                                           n_iter_no_change=20,
                                           tol=1e-4,
                                           random_state=42))])
}

# ---------- 单次 80/20 划分 ----------
seeds = [int(s) for s in config_dict.get('seed', '42').split(',')]
output_base = config_dict.get('res_folder', 'results')

for s in seeds:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=s
    )

    # 标签标准化
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled  = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    out_dir = os.path.join(output_base, f"seed_{s}")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "shap"),  exist_ok=True)

    # 保存列名，供后续 SHAP 或任何后处理对齐
    joblib.dump(X_train.columns, os.path.join(out_dir, 'training_columns.pkl'))

    results = []
    for name, pipe in models.items():
        try:
            pipe.fit(X_train, y_train_scaled)
            y_pred_scaled = pipe.predict(X_test)
            y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_original = y_test.values

            mae = mean_absolute_error(y_test_original, y_pred_original)
            mse = mean_squared_error(y_test_original, y_pred_original)
            r2  = r2_score(y_test_original, y_pred_original)
            results.append((name, mae, mse, r2))

            joblib.dump(pipe, os.path.join(out_dir, "models", f"{name}.joblib"))

            # ------- 回归效果图 -------
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test_original, y_pred_original, alpha=0.5)
            plt.plot([y_test_original.min(), y_test_original.max()],
                     [y_test_original.min(), y_test_original.max()], 'k--')
            plt.xlabel('Actual'); plt.ylabel('Predicted')
            plt.title(f'{name} (R²={r2:.3f})')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "images", f"{name}.png"))
            plt.close()

            # ------- SHAP 分析 -------
            if not SHAP_DO_SHAP:
                continue
            model = pipe.named_steps['model']
            if not hasattr(model, 'apply'):          # 仅树模型
                continue
            background = X_test.sample(n=min(SHAP_SAMPLES, len(X_test)), random_state=s)
            explainer = shap.TreeExplainer(model, data=background, feature_perturbation=SHAP_FEATURE_PERT)
            shap_values = explainer.shap_values(background)
            if isinstance(shap_values, list):        # 多输出
                shap_values = shap_values[0]

            if SHAP_SUMMARY_PLOT:
                plt.figure()
                shap.summary_plot(shap_values, background, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap", f"{name}_summary.png"))
                plt.close()

            if SHAP_BAR_PLOT:
                plt.figure()
                shap.plots.bar(shap.Explanation(values=shap_values,
                                               data=background.values,
                                               feature_names=list(background.columns.astype(str))),
                              show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap", f"{name}_bar.png"))
                plt.close()
            print(f'{name} 训练+SHAP 完成')

        except Exception as e:
            print(f"{name} 训练失败: {e}")

    # 写结果
    with open(os.path.join(out_dir, "results", "results.txt"), 'w', encoding='utf-8') as f:
        for res in results:
            f.write(f"{res[0]}: MAE={res[1]:.4f}, MSE={res[2]:.4f}, R²={res[3]:.4f}\n")

print("\n所有训练 + SHAP 分析完成!")