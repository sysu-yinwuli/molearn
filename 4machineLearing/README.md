# 4machineLearing — 机器学习训练模块

## 文件清单

| 文件 | 功能 |
|------|------|
| `ml-m-full.py` | 主训练脚本：16 种模型 + HPO + SHAP + 降维 |
| `feature_utils.py` | 共享工具模块（特征提取、降维、配置解析） |
| `ablation_study.py` | 描述符消融实验（Single / Sequential 两种模式 + 7 种图） |
| `batch_train.py` | 批量训练（多 npy / 超参扫描）+ 计算资源控制 |
| `dataset_split.py` | 数据集划分（随机 / 分层 / 骨架划分） |
| `read_res.py` | 读取所有 `seed_*` 目录训练结果并输出 Excel |

---

## ml-m-full.py

### 配置区说明

```
CONFIG_TXT          = 'config-full-1.txt'   # 特征开关配置文件

MODEL_ENABLE        = {...}   # 16 个模型独立开关
MODEL_PARAMS        = {...}   # 每个模型的超参数（Pipeline key 格式）

HPO_ENABLE          = False   # 超参数优化总开关
HPO_METHOD          = 'optuna'   # 'grid' | 'random' | 'optuna'
HPO_CV              = 5
HPO_N_ITER          = 30

SHAP_ENABLE         = True    # SHAP 分析（仅树模型）

DIM_REDUCTION_CFG   = {       # 降维配置
    'method': 'none',         # 'none'|'pca'|'kpca'|'tsvd'|'umap'|'autoencoder'
    'n_components': 50,
    'variance_ratio': None,   # 如 0.95 则自动取主成分数
    ...
}

SPLIT_METHOD        = 'random'    # 'random' | 'stratified'
SPLIT_N_BINS        = 5           # 分层时的 y 分位数分箱数
```

### 支持的 16 种模型

| 模型 | 类型 |
|------|------|
| LinearRegression | 线性回归 |
| Ridge | L2 正则 |
| LassoCV | L1 正则（CV 自动 alpha）|
| ElasticNetCV | 弹性网络 |
| HuberRegressor | 鲁棒线性回归 |
| BayesianRidge | 贝叶斯岭回归 |
| DecisionTree | 决策树 |
| RandomForest | 随机森林 |
| ExtraTreesRegressor | 极端随机树 |
| GradientBoosting | 梯度提升树 |
| HistGBR | 直方图梯度提升（大数据集首选）|
| AdaBoost | 自适应提升 |
| BaggingRegressor | 袋装集成 |
| SVR | 支持向量回归 |
| KNeighbors | K 近邻 |
| MLP | 多层感知机 |

### 输出目录结构

```
outputs/training/          ← 推荐路径（通过 molearn.yaml 配置）
└── seed_42/
    ├── models/
    │   ├── RandomForest.joblib            ← 训练好的模型
    │   └── RandomForest_model_card.txt    ← 模型信息卡（见下节）
    ├── images/
    │   └── RandomForest.png               ← 预测 vs 真实散点图
    ├── shap/
    │   ├── RandomForest_summary.png
    │   └── RandomForest_bar.png
    ├── training_columns.pkl               ← 训练特征列名
    ├── y_scaler.pkl                       ← 标签 StandardScaler
    └── dim_reducer.pkl                    ← 降维器（若启用降维）
```

> **变更说明**：`RandomForest_params.txt`（旧版）已升级为 `RandomForest_model_card.txt`（新版），内容大幅扩充，包含完整的数据集信息、引用文献等。

### 模型信息卡 (`{ModelName}_model_card.txt`)

每个模型训练完成后，在 `models/` 目录下自动生成一份完整的信息卡文件，内容格式如下：

```
======================================================================
  MOLEARN MODEL CARD
======================================================================
  Generated : 2025-01-01 12:00:00
  Model     : RandomForest
  Seed      : 42

── 1. 数据集信息 (Dataset) ─────────────────────────────────────────
  npy_path    : data/descriptors/dataset_fp.npy
  train_shape : 800 samples × 2048 features
  test_shape  : 200 samples × 2048 features
  split_method: random

── 2. 特征配置 (Feature Flags) ────────────────────────────────────
  [ON]  if_rdkit     : RDKit 2D Descriptors (200 维)
  [ON]  if_morgan    : Morgan Fingerprints (ECFP)
  [off] if_soap      : SOAP descriptors
  ...

── 3. 降维配置 (Dimensionality Reduction) ──────────────────────────
  method      : none (降维已禁用)

── 4. 模型信息 (Model) ─────────────────────────────────────────────
  model_name  : RandomForest
  description : Random Forest Regressor (Breiman, 2001)
  class       : RandomForestRegressor
  module      : sklearn.ensemble._forest
  scaler      : StandardScaler (特征标准化)

── 5. 超参数 (Hyperparameters) ─────────────────────────────────────
  max_depth                     : None
  n_estimators                  : 300
  ...

── 6. 超参数优化 (HPO) ─────────────────────────────────────────────
  HPO 未启用（使用手动设置的超参数）

── 7. 测试集评估指标 (Test Set Metrics) ────────────────────────────
  MAE     : 0.123456
  MSE     : 0.031415
  R2      : 0.956789
  train_time  : 12.34 s

── 8. 引用文献 (References) ────────────────────────────────────────
  [1] scikit-learn (主要框架)
      Pedregosa et al. (2011). JMLR 12, pp. 2825-2830.
  [2] RandomForest 算法
      Breiman, L. (2001). Random Forests. Machine Learning 45(1), 5-32.
  [3] RDKit 2D Descriptors
      RDKit: Open-source cheminformatics. http://www.rdkit.org
  ...
```

**8 个章节说明：**

| 章节 | 内容 |
|------|------|
| 1. Dataset | npy 路径、训练/测试样本量和特征维度、划分方法、配置文件路径 |
| 2. Feature Flags | 14 种描述符的 `[ON]`/`[off]` 状态 |
| 3. Dim Reduction | 降维方法及所有参数（method=none 时显示"已禁用"） |
| 4. Model Info | 模型名、算法说明、Python 类名及模块、Scaler 类型 |
| 5. Hyperparameters | Pipeline 中模型的全部超参数（sorted 字母序） |
| 6. HPO Best Params | 超参数优化最优结果（HPO 未启用时说明） |
| 7. Test Set Metrics | MAE / MSE / R²、本次训练耗时（秒） |
| 8. References | 自动生成：scikit-learn + 模型算法 + 所有**已启用**描述符的学术引用 |

**涵盖的 16 种模型引用：**

| 模型 | 引用来源 |
|------|----------|
| LinearRegression | scikit-learn OLS |
| Ridge | Hoerl & Kennard (1970) |
| LassoCV | Tibshirani (1996) |
| ElasticNetCV | Zou & Hastie (2005) |
| HuberRegressor | Huber (1964) |
| BayesianRidge | MacKay (1992) |
| DecisionTree | Breiman et al. (1984) CART |
| RandomForest | Breiman (2001) |
| ExtraTreesRegressor | Geurts et al. (2006) |
| GradientBoosting | Friedman (2001) |
| HistGBR | Chen & Guestrin (2016) XGBoost 思想 |
| AdaBoost | Freund & Schapire (1997) |
| BaggingRegressor | Breiman (1996) Bagging |
| SVR | Vapnik (1995) SVM |
| KNeighbors | Cover & Hart (1967) |
| MLP | Rumelhart et al. (1986) Backprop |

---

## feature_utils.py — 共享工具 API

```python
from feature_utils import (
    load_npy,            # 加载 .npy 文件
    extract_features,    # 批量提取特征矩阵
    build_header,        # 生成列名
    clean_features,      # 数据清洗（裁剪、NaN 填充）
    load_config,         # 解析 config-full-*.txt
    parse_flags,         # 解析特征开关
    resolve_path,        # 路径解析工具
    DimReducer,          # 降维器类（pca/kpca/tsvd/umap/autoencoder）
    fit_dim_reduction,   # 便捷：fit + transform 训练集
    apply_dim_reduction, # 便捷：transform 新样本
)
```

### DimReducer 使用示例

```python
from feature_utils import DimReducer
import joblib

# 训练阶段
cfg = {'method': 'pca', 'n_components': 50}
reducer = DimReducer(cfg=cfg)
X_train_reduced = reducer.fit_transform(X_train)
reducer.save('results/seed_42/dim_reducer.pkl')

# 推理阶段
reducer = DimReducer.load('results/seed_42/dim_reducer.pkl')
X_new_reduced = reducer.transform(X_new)
```

---

## dataset_split.py — 数据集划分

### 配置

```python
INPUT_NPY    = 'your_database.npy'
SPLIT_METHOD = 'random'     # 'random' | 'stratified' | 'scaffold'
TRAIN_RATIO  = 0.8
VALID_RATIO  = 0.1
TEST_RATIO   = 0.1

# scaffold 模式
SCAFFOLD_TYPE = 'bemis_murcko'  # 'bemis_murcko'|'murcko_generic'|'morgan'|'maccs'|'rdkit'
SCAFFOLD_ASSIGN = 'train_largest'
```

### 输出

```
split_output/
├── train.npy
├── valid.npy  （若 VALID_RATIO > 0）
├── test.npy
├── split_info.csv    # 每个分子的归属
└── split_summary.txt
```

---

## batch_train.py — 批量训练

### 模式

| 模式 | 功能 |
|------|------|
| A | 对多个 npy 文件分别训练 |
| B | 对同一数据集批量扫描超参数（grid / random）|

### 资源控制

```python
RESOURCE_N_JOBS          = -1       # sklearn 并行核数（-1=全核）
RESOURCE_MAX_CORES       = None     # CPU 核数上限
RESOURCE_PROCESS_PRIORITY = 'low'   # 'low'|'normal'|'high'
RESOURCE_PARALLEL_JOBS   = 1        # 批量任务并发进程数
RESOURCE_MAX_MEM_GB      = None     # 内存上限（Linux only）
```

> **Windows 用户**：`RESOURCE_PARALLEL_JOBS = 1`，`RESOURCE_MAX_MEM_GB = None`

---

## ablation_study.py — 描述符消融实验

```python
ABLATION_MODEL  = 'RandomForest'
ABLATION_MODE   = 'both'     # 'single' | 'sequential' | 'both'
ABLATION_CV     = 5          # sequential 模式的 CV 折数
ABLATION_OUT_DIR = 'ablation_results'
```

### 7 种图表

| 图文件 | 类型 |
|--------|------|
| `plot_bar_single.png` | 柱状图 |
| `plot_grouped_bar.png` | 分组柱状图（双 Y 轴）|
| `plot_line_sequential.png` | 点线图（sequential 趋势）|
| `plot_area_sequential.png` | 面积图 |
| `plot_heatmap_single.png` | 热力图 |
| `plot_pie_contribution.png` | 饼状图 |
| `plot_radar.png` | 雷达图 |

---

## config-full-*.txt 格式

单独运行 `ml-m-full.py` 时使用本地 `config-full-*.txt`；通过 `molearn_run.py` 运行时由总控脚本自动生成临时配置并通过 `MOLEARN_CONFIG` 环境变量传入。

```
npy_path:    data/descriptors/dataset_fp.npy   # 逗号分隔，支持多文件
res_folder:  outputs/training                  # 结果输出目录
seed:        42,123                            # 多 seed 批量训练

# 描述符开关（14 种）
if_rdkit:    1    # RDKit 2D 描述符
if_maccs:    1    # MACCS Keys
if_morgan:   1    # Morgan / ECFP 指纹
if_atompair: 0    # AtomPair 指纹（新增）
if_torsion:  0    # TopologicalTorsion 指纹（新增）
if_avalon:   0    # Avalon 指纹（新增）
if_soap:     0    # SOAP 描述符
if_acsf:     0    # ACSF 描述符
if_mbtr:     0    # MBTR 描述符（新增）
if_mordred:  0    # Mordred 分子描述符
if_prop:     0    # 基础分子性质（新增）
if_QC:       0    # 量化化学描述符
if_extra:    0    # 用户自定义描述符
if_m:        0    # 3D 矩阵描述符
```

> **提示**：通过 `molearn.yaml` 的 `step6_train.features` 节统一管理这些开关，无需手动编辑此文件。

---

## 分类模式（Classification Mode）

`ml-m-full.py` 完全支持分类任务。只需在文件顶部将 `TASK_TYPE` 改为 `'classification'`：

```python
TASK_TYPE = 'classification'   # 'regression' | 'classification'
```

或在 `molearn.yaml` 中设置：

```yaml
step6_train:
  task_type: "classification"
```

### 支持的 12 种分类模型

| 模型名 | 类型 |
|--------|------|
| `LogisticRegression` | 线性分类器 |
| `RidgeClassifier` | 岭分类器 |
| `SVC` | 支持向量分类（RBF 核，probability=True）|
| `KNeighborsClassifier` | K 近邻分类器 |
| `DecisionTreeClassifier` | 决策树分类器 |
| `RandomForestClassifier` | 随机森林分类器 |
| `ExtraTreesClassifier` | 极端随机树分类器 |
| `GradientBoostingClassifier` | 梯度提升分类器 |
| `HistGBClassifier` | 直方图梯度提升分类器 |
| `AdaBoostClassifier` | 自适应提升分类器 |
| `MLPClassifier` | 多层感知机分类器 |
| `GaussianNB` | 高斯朴素贝叶斯分类器 |

### 分类标签格式

标签必须为整数（0, 1, 2, ... 等）。分子 npy 中 `y` 字段示例：

```python
# 二分类
{'name': 'mol_001', 'y': 0, ...}
{'name': 'mol_002', 'y': 1, ...}

# 四分类
{'name': 'mol_003', 'y': 0, ...}   # 类别 0
{'name': 'mol_004', 'y': 3, ...}   # 类别 3
```

### 分类任务输出

训练完成后，每个模型在 `seed_N/` 目录生成：

```
seed_42/
├── models/
│   ├── RandomForestClassifier.joblib      ← 分类模型文件
│   ├── RandomForestClassifier_params.txt  ← 轻量参数报告
│   ├── RandomForestClassifier_model_card.txt ← 完整信息卡（含分类指标）
│   ├── task_type.pkl                      ← 'classification' 字符串
│   └── classes.pkl                        ← 类别列表 [0, 1, 2, ...]
├── images/
│   └── RandomForestClassifier_cm.png      ← 混淆矩阵图
├── results/
│   ├── results.txt                        ← Accuracy / F1 / AUC 汇总
│   └── RandomForestClassifier_clf_report.txt ← sklearn classification_report
└── shap/
    └── RandomForestClassifier_summary.png ← SHAP 分析图（树模型）
```

### 分类评估指标

| 指标 | 说明 |
|------|------|
| `Accuracy` | 整体准确率 |
| `F1_weighted` | 加权平均 F1 分数（支持多分类）|
| `AUC` | ROC-AUC（二分类用概率，多分类用 ovr 加权均值）|

> 注：`RidgeClassifier` 不输出概率，AUC 计算使用 `decision_function`。`GaussianNB` 在部分高维场景下 AUC 可能为 N/A。
