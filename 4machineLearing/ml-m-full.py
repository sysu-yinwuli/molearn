#!/usr/bin/env python3
# ml-m-full.py  —— 训练 + SHAP + 超参数优化 全合一
# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 数据配置 ----------
CONFIG_TXT   = 'config-full-1.txt'   # 特征开关配置文件（npy_path、if_rdkit 等）

# ---------- 启用/禁用模型 ----------
# 设为 False 可跳过某个模型，减少训练时间；删除某行等同于 False
MODEL_ENABLE = {
    "LinearRegression":  True,
    "Ridge":             True,
    "LassoCV":           True,
    "ElasticNetCV":      True,
    "HuberRegressor":    True,   # 对异常值鲁棒的线性模型
    "BayesianRidge":     True,   # 贝叶斯岭回归，自动估计正则强度
    "DecisionTree":      True,
    "RandomForest":      True,
    "ExtraTreesRegressor": True, # 极端随机树，比 RF 更快
    "GradientBoosting":  True,
    "HistGBR":           True,   # 直方图梯度提升，大数据集首选
    "AdaBoost":          True,   # 自适应提升
    "BaggingRegressor":  True,   # 袋装集成
    "SVR":               True,
    "KNeighbors":        True,
    "MLP":               True,
}

# ---------- 模型超参数 ----------
# key 格式：Pipeline 步骤名 "model__参数名"，直接传给 pipe.set_params()
_ALPHA_SPACE = None   # None → 自动使用 np.logspace(-4, 1, 20)

MODEL_PARAMS = {
    "LinearRegression":    {},
    "Ridge":               {"model__alpha": 10.0},
    "LassoCV":             {"model__cv": 5, "model__alphas": _ALPHA_SPACE,
                            "model__max_iter": 10000, "model__random_state": 42},
    "ElasticNetCV":        {"model__cv": 5, "model__alphas": _ALPHA_SPACE,
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

# ---------- 超参数优化（HPO）配置 ----------
HPO_ENABLE   = False        # 总开关
HPO_METHOD   = 'optuna'     # 'grid' | 'random' | 'optuna'
HPO_CV       = 5            # 交叉验证折数
HPO_N_ITER   = 30           # random/optuna 试验次数
HPO_SCORING  = 'r2'         # 优化指标
HPO_N_JOBS   = -1           # 并行数（-1=全核）
HPO_MODELS   = ['RandomForest', 'GradientBoosting', 'SVR']  # 空列表 [] = 优化全部

# grid / random 搜索空间
HPO_PARAM_GRIDS = {
    "Ridge":               {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "HuberRegressor":      {"model__epsilon": [1.1, 1.35, 1.5, 2.0],
                            "model__alpha": [1e-5, 1e-4, 1e-3]},
    "DecisionTree":        {"model__max_depth": [3, 5, 8, None],
                            "model__min_samples_split": [2, 5, 10]},
    "RandomForest":        {"model__n_estimators": [50, 100, 200],
                            "model__max_depth": [3, 5, 8, None],
                            "model__min_samples_split": [2, 5]},
    "ExtraTreesRegressor": {"model__n_estimators": [50, 100, 200],
                            "model__max_depth": [3, 5, 8, None]},
    "GradientBoosting":    {"model__n_estimators": [50, 100, 200],
                            "model__learning_rate": [0.05, 0.1, 0.2],
                            "model__max_depth": [2, 3, 5]},
    "HistGBR":             {"model__max_iter": [100, 200, 300],
                            "model__learning_rate": [0.05, 0.1, 0.2],
                            "model__max_depth": [3, 5, 8]},
    "AdaBoost":            {"model__n_estimators": [50, 100, 200],
                            "model__learning_rate": [0.01, 0.1, 1.0]},
    "SVR":                 {"model__C": [0.1, 1, 10, 100],
                            "model__epsilon": [0.001, 0.01, 0.1],
                            "model__gamma": ["scale", "auto"]},
    "KNeighbors":          {"model__n_neighbors": [3, 5, 7, 10, 15]},
    "MLP":                 {"model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                            "model__alpha": [1e-4, 1e-3, 1e-2],
                            "model__learning_rate_init": [1e-4, 1e-3, 1e-2]},
}

# Optuna 搜索空间
def _optuna_spaces(trial, model_name):
    if trial is None:
        return {}
    if model_name == "Ridge":
        return {"model__alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if model_name == "HuberRegressor":
        return {"model__epsilon": trial.suggest_float("epsilon", 1.05, 3.0),
                "model__alpha":   trial.suggest_float("alpha",   1e-5, 1e-1, log=True)}
    if model_name == "DecisionTree":
        return {"model__max_depth": trial.suggest_int("max_depth", 2, 15),
                "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20)}
    if model_name == "RandomForest":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "model__max_depth":    trial.suggest_int("max_depth", 2, 15),
                "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 10)}
    if model_name == "ExtraTreesRegressor":
        return {"model__n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "model__max_depth":    trial.suggest_int("max_depth", 2, 15)}
    if model_name == "GradientBoosting":
        return {"model__n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                "model__max_depth":     trial.suggest_int("max_depth", 2, 6)}
    if model_name == "HistGBR":
        return {"model__max_iter":      trial.suggest_int("max_iter", 50, 400),
                "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                "model__max_depth":     trial.suggest_int("max_depth", 2, 10)}
    if model_name == "AdaBoost":
        return {"model__n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True)}
    if model_name == "SVR":
        return {"model__C":       trial.suggest_float("C", 1e-2, 1e4, log=True),
                "model__epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
                "model__gamma":   trial.suggest_categorical("gamma", ["scale", "auto"])}
    if model_name == "KNeighbors":
        return {"model__n_neighbors": trial.suggest_int("n_neighbors", 2, 20)}
    if model_name == "MLP":
        n1 = trial.suggest_int("n1", 32, 256)
        n2 = trial.suggest_int("n2", 0, 128)
        return {"model__hidden_layer_sizes": (n1,) if n2 == 0 else (n1, n2),
                "model__alpha":              trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "model__learning_rate_init": trial.suggest_float("lr_init", 1e-4, 1e-2, log=True)}
    return {}

# ---------- SHAP 配置 ----------
SHAP_ENABLE       = True              # 总开关
SHAP_SAMPLES      = 1000             # 背景/解释样本数上限
SHAP_SUMMARY_PLOT = True             # summary_plot
SHAP_BAR_PLOT     = True             # bar plot
SHAP_FEATURE_PERT = 'interventional' # TreeExplainer 扰动方式
# 仅对以下树模型做 SHAP（TreeExplainer 专用）
SHAP_TREE_MODELS  = {"DecisionTree", "RandomForest", "ExtraTreesRegressor",
                     "GradientBoosting", "HistGBR"}

# ---------- 降维配置 ----------
# method: 'none' | 'pca' | 'kpca' | 'tsvd' | 'umap' | 'autoencoder'
# 降维在特征清洗之后、训练集划分之后执行（仅在训练集上 fit，测试集 transform）
DIM_REDUCTION_CFG = {
    'method':        'none',   # 'none' 表示不降维（默认关闭）
    'n_components':  50,       # 目标维数（pca/kpca/tsvd/umap 使用）
    'whiten':        False,    # 仅 PCA：白化
    'kpca_kernel':   'rbf',    # 仅 KernelPCA：核函数
    'variance_ratio': None,    # PCA/TSVD：用方差解释率自动确定维数（如 0.95）
    'umap_n_neighbors': 15,
    'umap_min_dist':    0.1,
    'umap_metric':      'euclidean',
    'umap_random_state': 42,
    'ae_dims':       [256, 128, 64],
    'ae_epochs':     50,
    'ae_batch_size': 64,
    'ae_lr':         1e-3,
    'ae_activation': 'relu',
}

# ---------- 数据集划分配置（与 dataset_split.py 联动）----------
# 此处控制 ml-m-full.py 内置的划分方式
# 'random'：全随机；'stratified'：分层（按 y 分位数）；详见 dataset_split.py 做更细粒度控制
SPLIT_METHOD  = 'random'      # 'random' | 'stratified'
SPLIT_N_BINS  = 5             # stratified 时的分层分箱数

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os
import sys
import copy
import datetime
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.linear_model import (LinearRegression, Ridge, LassoCV, ElasticNetCV,
                                   HuberRegressor, BayesianRidge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor,
                               BaggingRegressor,
                               HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline

# 公共工具（同目录）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from feature_utils import (load_npy, extract_features, build_header,
                            clean_features, load_config, parse_flags, resolve_path,
                            fit_dim_reduction, DimReducer)

# ── 路径解析 ──────────────────────────────────────────────────────────────────
_config_path = resolve_path(CONFIG_TXT, _HERE)
config = load_config(_config_path)

print("当前配置:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ── 加载数据 ──────────────────────────────────────────────────────────────────
ml_npy = [p.strip() for p in config['npy_path'].split(',')]
n_files = len(ml_npy)

datas = [load_npy(resolve_path(p, _HERE)) for p in ml_npy]

sample_counts = [len(d) for d in datas]
if len(set(sample_counts)) != 1:
    raise ValueError(f"所有 .npy 样本数量必须一致，当前: {sample_counts}")

# ── 特征提取 & 列名 ───────────────────────────────────────────────────────────
flags = parse_flags(config, n_files)
features_raw, labels = extract_features(datas, flags)
print(f"[INFO] 原始特征矩阵形状: {features_raw.shape}")

features = clean_features(features_raw)
header   = build_header(datas, flags)

if len(header) != features.shape[1]:
    raise RuntimeError(
        f"[ERROR] 列名数量 ({len(header)}) 与特征维度 ({features.shape[1]}) 不一致。"
        "请检查各 .npy 第 0 个样本的描述符字段长度。"
    )

X = pd.DataFrame(features, columns=header)
y = pd.Series(labels)
print(f"[INFO] 特征矩阵: {X.shape}，有效样本: {(~y.isna()).sum()}")

# ── 构建模型 Pipeline ──────────────────────────────────────────────────────────
_alpha_space = np.logspace(-4, 1, 20)

# 模型基础定义（需要 scaler 的模型加 scaler 步骤；树/集成模型不需要）
_ALL_BASE_MODELS = {
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

def _build_models():
    """按 MODEL_ENABLE 过滤，深拷贝并注入 MODEL_PARAMS。"""
    models = {}
    for name, base_pipe in _ALL_BASE_MODELS.items():
        if not MODEL_ENABLE.get(name, True):
            continue
        pipe = copy.deepcopy(base_pipe)
        params = MODEL_PARAMS.get(name, {})
        resolved = {
            k: (_alpha_space if v is None and k == 'model__alphas' else v)
            for k, v in params.items()
        }
        if resolved:
            pipe.set_params(**resolved)
        models[name] = pipe
    return models

# ── 模型信息卡（Model Card）生成 ──────────────────────────────────────────────
# 每个模型训练完成后，自动在同一目录写入一份完整的 model_card.txt
# 记录：数据集、特征配置、模型参数、训练指标、降维配置、引用文献
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CITATIONS = {
    "LinearRegression": (
        "Linear Regression",
        "Pedregosa et al. Scikit-learn: Machine Learning in Python. "
        "JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org"
    ),
    "Ridge": (
        "Ridge Regression (L2 regularization)",
        "Hoerl & Kennard (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. "
        "Technometrics 12(1):55-67. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "LassoCV": (
        "LASSO Regression with cross-validated alpha (L1 regularization)",
        "Tibshirani R (1996). Regression Shrinkage and Selection via the Lasso. "
        "J. Royal Stat. Soc. B 58(1):267-288. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "ElasticNetCV": (
        "Elastic Net with cross-validated alpha/l1_ratio (L1+L2 regularization)",
        "Zou & Hastie (2005). Regularization and Variable Selection via the Elastic Net. "
        "J. Royal Stat. Soc. B 67(2):301-320. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "HuberRegressor": (
        "Huber Regression (robust to outliers)",
        "Huber PJ (1964). Robust Estimation of a Location Parameter. "
        "Ann. Math. Statist. 35(1):73-101. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "BayesianRidge": (
        "Bayesian Ridge Regression (automatic relevance determination)",
        "Tipping ME (2001). Sparse Bayesian Learning and the Relevance Vector Machine. "
        "JMLR 1:211-244. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "DecisionTree": (
        "Decision Tree Regressor (CART)",
        "Breiman et al. (1984). Classification and Regression Trees. Wadsworth & Brooks/Cole. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "RandomForest": (
        "Random Forest Regressor (bagging of decision trees)",
        "Breiman L (2001). Random Forests. Machine Learning 45:5-32. "
        "DOI: 10.1023/A:1010933404324. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "ExtraTreesRegressor": (
        "Extra Trees Regressor (extremely randomized trees)",
        "Geurts et al. (2006). Extremely Randomized Trees. Machine Learning 63:3-42. "
        "DOI: 10.1007/s10994-006-6226-1. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "GradientBoosting": (
        "Gradient Boosting Regressor (GBRT)",
        "Friedman JH (2001). Greedy Function Approximation: A Gradient Boosting Machine. "
        "Ann. Statist. 29(5):1189-1232. DOI: 10.1214/aos/1013203451. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "HistGBR": (
        "Histogram-based Gradient Boosting Regressor (LightGBM-style, large dataset optimized)",
        "Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. "
        "NeurIPS 30. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "AdaBoost": (
        "AdaBoost Regressor (adaptive boosting)",
        "Freund & Schapire (1997). A Decision-Theoretic Generalization of On-Line Learning "
        "and an Application to Boosting. J. Comput. System Sci. 55(1):119-139. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "BaggingRegressor": (
        "Bagging Regressor (bootstrap aggregating)",
        "Breiman L (1996). Bagging Predictors. Machine Learning 24:123-140. "
        "DOI: 10.1007/BF00058655. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "SVR": (
        "Support Vector Regression (SVR with RBF kernel)",
        "Vapnik VN (1995). The Nature of Statistical Learning Theory. Springer. "
        "Smola & Schölkopf (2004). A Tutorial on Support Vector Regression. "
        "Statistics and Computing 14:199-222. DOI: 10.1023/B:STCO.0000035301.49549.88. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "KNeighbors": (
        "K-Nearest Neighbors Regressor",
        "Cover & Hart (1967). Nearest Neighbor Pattern Classification. "
        "IEEE Trans. Inf. Theory 13(1):21-27. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
    "MLP": (
        "Multi-Layer Perceptron Regressor (feedforward neural network)",
        "Rumelhart et al. (1986). Learning Representations by Back-Propagating Errors. "
        "Nature 323:533-536. DOI: 10.1038/323533a0. "
        "Pedregosa et al. JMLR 12, 2825-2830, 2011."
    ),
}

_DESCRIPTOR_CITATIONS = {
    'if_rdkit':    ("RDKit fingerprints & descriptors",
                    "Landrum G. RDKit: Open-source cheminformatics. https://www.rdkit.org"),
    'if_maccs':    ("MACCS structural keys (167-bit)",
                    "MDL Information Systems (2002). MACCS Structural Keys. "
                    "Symyx/MDL Drug Data Report."),
    'if_morgan':   ("Morgan circular fingerprints (ECFP/FCFP)",
                    "Rogers D & Hahn M (2010). Extended-Connectivity Fingerprints. "
                    "J. Chem. Inf. Model. 50(5):742-754. DOI: 10.1021/ci100050t."),
    'if_atompair': ("Atom-pair fingerprints",
                    "Carhart et al. (1985). Atom Pairs as Molecular Features in Structure-Activity Studies. "
                    "J. Chem. Inf. Comput. Sci. 25(2):64-73. DOI: 10.1021/ci00046a002."),
    'if_torsion':  ("Topological torsion fingerprints",
                    "Nilakantan et al. (1987). Topological Torsion: A New Molecular Descriptor for SAR Applications. "
                    "J. Chem. Inf. Comput. Sci. 27(2):82-85. DOI: 10.1021/ci00054a008."),
    'if_avalon':   ("Avalon fingerprints",
                    "Gedeck et al. (2006). QSAR - How Good Is It in Practice? "
                    "J. Chem. Inf. Model. 46(5):1924-1936. DOI: 10.1021/ci050413p."),
    'if_soap':     ("SOAP (Smooth Overlap of Atomic Positions) descriptor",
                    "Bartók et al. (2013). On Representing Chemical Environments. "
                    "Phys. Rev. B 87, 184115. DOI: 10.1103/PhysRevB.87.184115. "
                    "DScribe library: Himanen et al. (2020). Comput. Phys. Commun. 247, 106949."),
    'if_acsf':     ("ACSF (Atom-Centered Symmetry Functions) descriptor",
                    "Behler & Parrinello (2007). Generalized Neural-Network Representation of High-Dimensional "
                    "Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401. DOI: 10.1103/PhysRevLett.98.146401. "
                    "DScribe library: Himanen et al. (2020). Comput. Phys. Commun. 247, 106949."),
    'if_mbtr':     ("MBTR (Many-Body Tensor Representation) descriptor",
                    "Huo & Rupp (2018). Unified Representation of Molecules and Crystals for Machine Learning. "
                    "arXiv:1704.06439. "
                    "DScribe library: Himanen et al. (2020). Comput. Phys. Commun. 247, 106949."),
    'if_mordred':  ("Mordred molecular descriptors",
                    "Moriwaki et al. (2018). Mordred: a Molecular Descriptor Calculator. "
                    "J. Cheminform. 10, 4. DOI: 10.1186/s13321-018-0258-y."),
    'if_prop':     ("Basic molecular properties (RDKit)",
                    "MolWt, HeavyAtomCount, HBA, HBD, LogP, TPSA, RotBonds, ArRings, NumRings, FractionCSP3, Heteroatoms. "
                    "Lipinski et al. (1997). Experimental and Computational Approaches to Estimate Solubility and "
                    "Permeability in Drug Discovery. Adv. Drug Deliv. Rev. 23:3-25. DOI: 10.1016/S0169-409X(96)00423-1."),
    'if_QC':       ("Quantum chemistry descriptors (g_d field)", "In-house QC calculations."),
    'if_extra':    ("Extra numerical descriptors (extra_d field)", "User-supplied external descriptors."),
    'if_m':        ("3D matrix descriptors (3DMatrix field)", "User-supplied 3D matrix descriptors."),
}

def _write_model_card(pipe, name, metrics, seed, config, flags,
                      npy_paths, dim_cfg, hpo_best, train_shape, test_shape,
                      train_time_s, out_path):
    """
    生成单个模型的完整信息卡（model_card.txt）。
    涵盖数据集、特征、模型参数、超参数优化、降维、评估指标、引用文献。
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_step = pipe.named_steps['model']
    params     = model_step.get_params()
    long_name, main_cite = _MODEL_CITATIONS.get(name, (name, "N/A"))

    sep = "=" * 70

    lines = [
        sep,
        f"  MOLEARN MODEL CARD",
        sep,
        f"  Generated : {now}",
        f"  Model     : {name}",
        f"  Seed      : {seed}",
        sep,
        "",
        "── 1. 数据集信息 (Dataset) ─────────────────────────────────────────",
        f"  npy_path    : {', '.join(npy_paths)}",
        f"  res_folder  : {config.get('res_folder', 'N/A')}",
        f"  config_file : {config.get('_config_file', 'N/A')}",
        f"  train_shape : {train_shape[0]} samples × {train_shape[1]} features",
        f"  test_shape  : {test_shape[0]} samples × {test_shape[1]} features",
        f"  split_method: {config.get('_split_method', 'random')}",
        f"  test_ratio  : 0.20 (80/20 split)",
        "",
        "── 2. 特征配置 (Feature Flags) ────────────────────────────────────",
    ]

    active_descriptors = []
    for flag_key, (desc_name, desc_cite) in _DESCRIPTOR_CITATIONS.items():
        flag_vals = flags.get(flag_key, [0])
        if any(v for v in flag_vals):
            lines.append(f"  [ON]  {flag_key:12s}: {desc_name}")
            active_descriptors.append((desc_name, desc_cite))
        else:
            lines.append(f"  [off] {flag_key:12s}: {desc_name}")

    lines += [
        "",
        "── 3. 降维配置 (Dimensionality Reduction) ──────────────────────────",
    ]
    dr_method = dim_cfg.get('method', 'none')
    if dr_method != 'none':
        lines.append(f"  method      : {dr_method}")
        for k, v in dim_cfg.items():
            if k != 'method':
                lines.append(f"  {k:12s}: {v}")
    else:
        lines.append("  method      : none (降维已禁用)")

    lines += [
        "",
        "── 4. 模型信息 (Model) ─────────────────────────────────────────────",
        f"  model_name  : {name}",
        f"  description : {long_name}",
        f"  class       : {type(model_step).__name__}",
        f"  module      : {type(model_step).__module__}",
    ]

    if 'scaler' in pipe.named_steps:
        scaler = pipe.named_steps['scaler']
        lines.append(f"  scaler      : {type(scaler).__name__} (特征标准化)")

    lines += [
        "",
        "── 5. 超参数 (Hyperparameters) ─────────────────────────────────────",
    ]
    for k, v in sorted(params.items()):
        lines.append(f"  {k:30s}: {v}")

    if hpo_best:
        lines += [
            "",
            "── 6. 超参数优化最优结果 (HPO Best Params) ────────────────────────",
            f"  hpo_method  : {config.get('_hpo_method', 'N/A')}",
        ]
        for k, v in hpo_best.items():
            lines.append(f"  {k:30s}: {v}")
    else:
        lines.append("")
        lines.append("── 6. 超参数优化 (HPO) ─────────────────────────────────────────────")
        lines.append("  HPO 未启用（使用手动设置的超参数）")

    lines += [
        "",
        "── 7. 测试集评估指标 (Test Set Metrics) ────────────────────────────",
    ]
    for metric, val in metrics.items():
        lines.append(f"  {metric:8s}: {val:.6f}")
    lines.append(f"  train_time  : {train_time_s:.2f} s")

    lines += [
        "",
        "── 8. 引用文献 (References) ────────────────────────────────────────",
        "",
        f"  [1] scikit-learn (主要框架)",
        f"      Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.",
        f"      Journal of Machine Learning Research 12, pp. 2825-2830.",
        f"      URL: https://scikit-learn.org",
        "",
        f"  [2] {name} 算法",
        f"      {main_cite}",
        "",
    ]
    ref_idx = 3
    for desc_name, desc_cite in active_descriptors:
        lines.append(f"  [{ref_idx}] {desc_name}")
        lines.append(f"      {desc_cite}")
        lines.append("")
        ref_idx += 1

    lines += [
        sep,
        f"  END OF MODEL CARD",
        sep,
    ]

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _dump_model_params(pipe, name, metrics: dict, seed: int, out_path: str):
    """
    轻量版参数报告（保留向后兼容）。
    完整信息卡由 _write_model_card() 生成。
    """
    model_step = pipe.named_steps['model']
    params = model_step.get_params()
    scaler_info = ""
    if 'scaler' in pipe.named_steps:
        scaler = pipe.named_steps['scaler']
        scaler_info = f"\n[Scaler]\ntype: {type(scaler).__name__}\n"

    lines = [
        f"# Model Parameter Report",
        f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Seed: {seed}",
        f"",
        f"[Model]",
        f"name: {name}",
        f"class: {type(model_step).__name__}",
        f"module: {type(model_step).__module__}",
    ]
    if scaler_info:
        lines.append(scaler_info.strip())

    lines += ["", "[Hyperparameters]"]
    for k, v in sorted(params.items()):
        lines.append(f"{k}: {v}")

    lines += ["", "[Metrics on Test Set]"]
    for metric, val in metrics.items():
        lines.append(f"{metric}: {val:.6f}")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

# ── 超参数优化 ────────────────────────────────────────────────────────────────
def _run_hpo(pipe, name, X_tr, y_tr, seed):
    """返回 (最优已fit Pipeline, 最优参数 dict)。"""
    target = set(HPO_MODELS) if HPO_MODELS else set(_ALL_BASE_MODELS)
    if name not in target:
        return pipe, {}
    if HPO_METHOD in ('grid', 'random') and name not in HPO_PARAM_GRIDS:
        print(f"  [HPO] {name} 无搜索空间，跳过")
        return pipe, {}

    print(f"  [HPO] {name}  method={HPO_METHOD}  scoring={HPO_SCORING} ...")

    if HPO_METHOD == 'grid':
        searcher = GridSearchCV(pipe, HPO_PARAM_GRIDS[name],
                                cv=HPO_CV, scoring=HPO_SCORING,
                                n_jobs=HPO_N_JOBS, refit=True, verbose=0)
        searcher.fit(X_tr, y_tr)
        print(f"  [HPO] 最优 {HPO_SCORING}={searcher.best_score_:.4f}  params={searcher.best_params_}")
        return searcher.best_estimator_, searcher.best_params_

    if HPO_METHOD == 'random':
        searcher = RandomizedSearchCV(pipe, HPO_PARAM_GRIDS[name],
                                      n_iter=HPO_N_ITER, cv=HPO_CV,
                                      scoring=HPO_SCORING, n_jobs=HPO_N_JOBS,
                                      random_state=seed, refit=True, verbose=0)
        searcher.fit(X_tr, y_tr)
        print(f"  [HPO] 最优 {HPO_SCORING}={searcher.best_score_:.4f}  params={searcher.best_params_}")
        return searcher.best_estimator_, searcher.best_params_

    if HPO_METHOD == 'optuna':
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("  [HPO] optuna 未安装（pip install optuna），跳过")
            return pipe, {}

        _probe = _optuna_spaces.__code__.co_consts
        if name not in str(_probe):
            print(f"  [HPO] {name} 无 Optuna 空间定义，跳过")
            return pipe, {}

        def objective(trial):
            params = _optuna_spaces(trial, name)
            if not params:
                raise optuna.exceptions.TrialPruned()
            trial_pipe = copy.deepcopy(pipe)
            trial_pipe.set_params(**params)
            return cross_val_score(trial_pipe, X_tr, y_tr,
                                   cv=HPO_CV, scoring=HPO_SCORING, n_jobs=1).mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=HPO_N_ITER, show_progress_bar=False)
        best_params = _optuna_spaces(study.best_trial, name)
        pipe.set_params(**best_params)
        pipe.fit(X_tr, y_tr)
        print(f"  [HPO] 最优 {HPO_SCORING}={study.best_value:.4f}  params={best_params}")
        return pipe, best_params

    raise ValueError(f"HPO_METHOD='{HPO_METHOD}' 无效，应为 'grid'/'random'/'optuna'")

# ── SHAP 分析 ──────────────────────────────────────────────────────────────────
def _run_shap(pipe, name, X_bg, seed, shap_dir):
    """对树模型做 SHAP 分析并保存图片，失败时打印警告不中断主流程。"""
    if not SHAP_ENABLE or name not in SHAP_TREE_MODELS:
        return
    try:
        model_step = pipe.named_steps['model']
        n_bg = min(SHAP_SAMPLES, len(X_bg))
        bg   = X_bg.sample(n=n_bg, random_state=seed)
        explainer   = shap.TreeExplainer(model_step, data=bg,
                                          feature_perturbation=SHAP_FEATURE_PERT)
        shap_values = explainer.shap_values(bg)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        explanation = shap.Explanation(
            values=shap_values,
            data=bg.values,
            feature_names=list(bg.columns.astype(str))
        )
        if SHAP_SUMMARY_PLOT:
            plt.figure()
            shap.summary_plot(shap_values, bg, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"{name}_summary.png"), dpi=150)
            plt.close()

        if SHAP_BAR_PLOT:
            plt.figure()
            shap.plots.bar(explanation, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"{name}_bar.png"), dpi=150)
            plt.close()

        print(f"  {name} SHAP 完成")
    except Exception as e:
        print(f"  [WARN] {name} SHAP 失败（不影响训练）: {e}")

# ── 主训练循环 ────────────────────────────────────────────────────────────────
seeds       = [int(s.strip()) for s in config.get('seed', '42').split(',')]
output_base = config.get('res_folder', 'results')

# 将训练上下文注入 config，供 model card 使用
config['_config_file'] = _config_path
config['_split_method'] = SPLIT_METHOD
config['_hpo_method']   = HPO_METHOD if HPO_ENABLE else 'disabled'

for seed in seeds:
    print(f"\n{'='*60}\n  Seed = {seed}\n{'='*60}")

    # ── 数据集划分 ────────────────────────────────────────────────────────────
    if SPLIT_METHOD == 'stratified':
        # 分层采样：按 y 分位数分箱，用箱编号作为 stratify 标签
        from sklearn.preprocessing import KBinsDiscretizer
        _binner = KBinsDiscretizer(n_bins=SPLIT_N_BINS, encode='ordinal', strategy='quantile')
        _strat  = _binner.fit_transform(y.values.reshape(-1, 1)).flatten().astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=_strat
        )
        print(f"  [划分] 分层采样（y 分 {SPLIT_N_BINS} 箱）")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        print(f"  [划分] 全随机划分")

    # 标签标准化（仅对训练集 fit，防止数据泄漏）
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # ── 降维（可选）────────────────────────────────────────────────────────────
    dim_reducer = None
    if DIM_REDUCTION_CFG.get('method', 'none') != 'none':
        print(f"  [降维] 开始降维: method={DIM_REDUCTION_CFG['method']}")
        dim_reducer, X_train_np = fit_dim_reduction(X_train.values, DIM_REDUCTION_CFG)
        X_test_np  = dim_reducer.transform(X_test.values)
        # 重新包装为 DataFrame（列名为 dr_0, dr_1, ...）
        dr_cols   = [f"dr_{i}" for i in range(X_train_np.shape[1])]
        X_train   = pd.DataFrame(X_train_np, columns=dr_cols, index=X_train.index)
        X_test    = pd.DataFrame(X_test_np,  columns=dr_cols, index=X_test.index)
    else:
        print(f"  [降维] 跳过（method=none）")

    # 输出目录
    out_dir = os.path.join(output_base, f"seed_{seed}")
    for sub in ("models", "images", "results", "shap"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # 保存推理所需资源
    joblib.dump(X_train.columns.tolist(), os.path.join(out_dir, 'training_columns.pkl'))
    joblib.dump(y_scaler,                 os.path.join(out_dir, 'y_scaler.pkl'))

    # 保存降维器（如果启用）
    if dim_reducer is not None:
        dim_reducer.save(os.path.join(out_dir, 'dim_reducer.pkl'))

    results     = []
    hpo_results = {}
    models      = _build_models()

    for name, pipe in models.items():
        try:
            # —— HPO 或直接训练 ——
            _t0 = datetime.datetime.now()
            if HPO_ENABLE:
                pipe, best = _run_hpo(pipe, name, X_train, y_train_s, seed)
                if best:
                    hpo_results[name] = best
                if not best:
                    pipe.fit(X_train, y_train_s)
            else:
                pipe.fit(X_train, y_train_s)
            _train_time = (datetime.datetime.now() - _t0).total_seconds()

            # —— 评估（在原始标签空间）——
            y_pred = y_scaler.inverse_transform(
                pipe.predict(X_test).reshape(-1, 1)
            ).flatten()
            y_true = y_test.values

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2  = r2_score(y_true, y_pred)
            results.append((name, mae, mse, r2))
            print(f"  {name:22s}  MAE={mae:.4f}  MSE={mse:.4f}  R²={r2:.4f}")

            # —— 保存模型 ——
            model_path = os.path.join(out_dir, "models", f"{name}.joblib")
            joblib.dump(pipe, model_path)

            # —— 导出模型参数报告（轻量版）——
            _dump_model_params(
                pipe, name,
                metrics={"MAE": mae, "MSE": mse, "R2": r2},
                seed=seed,
                out_path=os.path.join(out_dir, "models", f"{name}_params.txt")
            )

            # —— 生成完整模型信息卡（model_card.txt）——
            _write_model_card(
                pipe=pipe,
                name=name,
                metrics={"MAE": mae, "MSE": mse, "R2": r2},
                seed=seed,
                config=config,
                flags=flags,
                npy_paths=ml_npy,
                dim_cfg=DIM_REDUCTION_CFG,
                hpo_best=hpo_results.get(name, {}),
                train_shape=X_train.shape,
                test_shape=X_test.shape,
                train_time_s=_train_time,
                out_path=os.path.join(out_dir, "models", f"{name}_model_card.txt")
            )

            # —— 回归散点图 ——
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lim, lim, 'k--', lw=1)
            ax.set_xlabel('Actual');  ax.set_ylabel('Predicted')
            ax.set_title(f'{name}  R²={r2:.3f}')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "images", f"{name}.png"), dpi=150)
            plt.close(fig)

            # —— SHAP ——
            _run_shap(pipe, name, X_test, seed, os.path.join(out_dir, "shap"))

        except Exception as e:
            print(f"  [ERROR] {name} 失败: {e}")
            traceback.print_exc()

    # —— 写结果文件 ——
    with open(os.path.join(out_dir, "results", "results.txt"), 'w', encoding='utf-8') as f:
        f.write(f"# Seed={seed}\n")
        for name, mae, mse, r2 in results:
            f.write(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}\n")

    if HPO_ENABLE and hpo_results:
        with open(os.path.join(out_dir, "results", "hpo_best_params.txt"), 'w', encoding='utf-8') as f:
            f.write(f"# Seed={seed}  HPO_METHOD={HPO_METHOD}\n")
            for mname, params in hpo_results.items():
                f.write(f"{mname}: {params}\n")

print("\n所有训练完成！")
