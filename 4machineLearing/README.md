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
results/
└── seed_42/
    ├── models/
    │   ├── RandomForest.joblib
    │   └── RandomForest_params.txt   # 模型参数 + 训练指标
    ├── images/
    │   └── RandomForest.png          # 预测 vs 真实散点图
    ├── shap/
    │   ├── RandomForest_summary.png
    │   └── RandomForest_bar.png
    ├── training_columns.pkl          # 训练特征列名
    ├── y_scaler.pkl                  # 标签 StandardScaler
    └── dim_reducer.pkl               # 降维器（若启用降维）
```

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

```
npy_path: poly-all-fp.npy          # 逗号分隔，支持多文件
res_folder: results
seed: 42,123                       # 多 seed 批量训练
if_rdkit:  1
if_maccs:  1
if_morgan: 0
if_soap:   0
if_acsf:   0
if_mordred: 0
if_QC:     0
if_extra:  0
if_m:      0
```
