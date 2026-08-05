# Molearn — 分子机器学习全流程框架

> **Molecular Machine Learning Pipeline**  
> 从原始分子结构（.gjf/.xyz）到训练模型并推理预测的完整九步流水线，支持回归与分类双任务，内置 Pearson 共线性过滤、多种抽样与划分方法、16 种回归 / 12 种分类模型、SHAP 可解释性分析，以及 Web GUI。

---

## 目录

1. [项目结构](#1-项目结构)
2. [数据目录布局](#2-数据目录布局)
3. [快速开始](#3-快速开始)
4. [配置文件 molearn.yaml](#4-配置文件-molearnyyaml)
5. [总控脚本 molearn_run.py](#5-总控脚本-molearn_runpy)
6. [Web GUI](#6-web-gui)
7. [九步流水线详解](#7-九步流水线详解)
   - [Step 1 数据预处理](#step-1--数据预处理-create_npypy)
   - [Step 2 数据库分析](#step-2--数据库分析-db_analysispy)
   - [Step 3 描述符计算 + Pearson 过滤](#step-3--描述符计算--pearson-过滤-create_by_fppy)
   - [Step 4 数据抽样](#step-4--数据抽样-sample_npypy)
   - [Step 5 数据集划分](#step-5--数据集划分-dataset_splitpy)
   - [Step 6 模型训练](#step-6--模型训练-ml-m-fullpy)
   - [Step 7 可合成性打分](#step-7--可合成性打分-sa_scorepy)
   - [Step 8 相似度搜索](#step-8--相似度搜索-similarity_searchpy)
   - [Step 9 模型推理](#step-9--模型推理-usemodelpy)
8. [环境变量接口（MOLEARN_*）](#8-环境变量接口molearn_)
9. [分类任务使用说明](#9-分类任务使用说明)
10. [依赖安装](#10-依赖安装)

---

## 1. 项目结构

```
molearn/
├── molearn.yaml              # 全项目统一配置文件（所有参数集中此处）
├── molearn_run.py            # 总控脚本（CLI，调度所有步骤）
├── requirements.txt          # Python 依赖列表
├── INSTALL.md                # 安装说明
│
├── molearn_gui/              # Web 图形界面
│   ├── molearn_gui.py        # Flask 服务
│   └── templates/
│       └── index.html        # 前端页面（全功能，含所有 9 步配置面板）
│
├── 1dataProcess/             # Step 1 & 4：数据预处理 + 抽样
│   ├── create_npy.py         # xlsx+xyz → .npy
│   ├── sample_npy.py         # 数据库抽样（5 种方法）
│   ├── gjf2xyz.py            # .gjf → .xyz 转换
│   └── create_npy-dir.py     # 批量目录创建
│
├── 2databaseAnalysis/        # Step 2 & 7 & 8：数据库分析/打分/搜索
│   ├── db_analysis.py        # 数据库可视化分析
│   ├── sa_score.py           # 可合成性打分（SA/SC/SYBA/RA）
│   └── similarity_search.py  # 相似度搜索（Tanimoto）
│
├── 3descriptor/              # Step 3：描述符计算
│   ├── create_by_fp.py       # 主描述符计算脚本（11 种描述符）
│   └── pearson_filter.py     # Pearson 共线性过滤（可单独运行）
│
├── 4machineLearing/          # Step 5 & 6：划分 + 训练
│   ├── ml-m-full.py          # 全功能训练脚本（回归+分类）
│   ├── dataset_split.py      # 数据集划分
│   ├── feature_utils.py      # 公共特征工具（PathManager, npy_save/load）
│   └── batch_train.py        # 批量训练（多配置）
│
└── 5modelApplication/        # Step 9：模型推理
    └── usemodel.py           # 推理脚本（自动检测回归/分类）
```

---

## 2. 数据目录布局

**所有项目数据统一存放在 `data/` 目录，所有输出统一存放在 `outputs/` 目录。**

```
项目根/
├── data/
│   ├── raw/
│   │   ├── gjf/          ← 原始 .gjf 量化计算输入文件
│   │   ├── xyz/          ← .xyz 三维坐标文件（gjf2xyz 输出 / 手动放置）
│   │   └── xlsx/         ← Excel 配置/标签文件
│   ├── processed/        ← Step1 输出：create_npy 生成的分子 .npy
│   ├── descriptors/      ← Step3 输出：带描述符的 .npy
│   ├── pearson/          ← Step3+ 输出：Pearson 过滤后的 .npy（启用时）
│   ├── samples/          ← Step4 输出：抽样子集 .npy
│   └── splits/           ← Step5 输出：划分后的 train/valid/test .npy
│
└── outputs/
    ├── training/         ← Step6 输出：models/ images/ shap/ results/
    │   └── <项目名>/
    │       └── seed_42/  ← 每个随机种子独立存储
    ├── analysis/         ← Step2 输出：图表 PNG + CSV
    ├── sa_scores/        ← Step7 输出：合成性打分 CSV + 图表
    ├── similarity/       ← Step8 输出：相似度矩阵 CSV
    └── predictions/      ← Step9 输出：预测结果 CSV
```

> **重要**：以前各子脚本使用的本地目录（`sampled_npy/`、`analysis_output/`、`split_output/`、`sa_results/`、`similarity_results/` 等）已通过 `MOLEARN_*` 环境变量机制统一重定向到上述 `data/` 结构。单独运行子脚本时仍使用各自的本地默认目录；通过 `molearn_run.py` 或 GUI 运行时自动重定向。

---

## 3. 快速开始

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 初始化目录结构

```bash
python molearn_run.py --init-dirs
```

这会按 `molearn.yaml` 中的 `paths` 节点创建所有必要目录。

### 3.3 准备数据

1. 将 `.gjf` 文件放入 `data/raw/gjf/`，或直接将 `.xyz` 文件放入 `data/raw/xyz/`
2. 准备分子属性 Excel（分子名 + y 值），放入 `data/raw/xlsx/`
3. 编辑 `data/raw/creat_npy.xlsx` 配置表，填写 `xyz_path`、`xlsx_file`、`npy_path` 等列

### 3.4 编辑配置

```bash
# 编辑 molearn.yaml，设置项目名、文件路径、启用的步骤等
nano molearn.yaml
```

### 3.5 运行流水线

```bash
# 运行所有 enabled=true 的步骤
python molearn_run.py

# 查看步骤状态
python molearn_run.py --list

# 仅运行 Step 1~3（数据预处理到描述符计算）
python molearn_run.py --step 1-3

# 仅运行 Step 6（模型训练），覆盖种子和任务类型
python molearn_run.py --step 6 --seed 42,123 --task-type regression

# 预演（仅打印命令，不实际执行）
python molearn_run.py --dry-run
```

### 3.6 启动 Web GUI

```bash
python molearn_gui/molearn_gui.py
# 访问 http://localhost:5000
```

---

## 4. 配置文件 molearn.yaml

`molearn.yaml` 是整个项目的**单一配置源**，包含所有 9 个步骤的参数。结构如下：

```yaml
project:          # 项目基本信息（名称/描述/作者/版本）
paths:            # 统一目录路径（所有步骤的输入输出目录）
step1_data:       # Step 1 数据预处理参数
step2_analysis:   # Step 2 数据库分析参数
step3_descriptor: # Step 3 描述符计算参数（含 Pearson 过滤配置）
step4_sampling:   # Step 4 数据抽样参数
step5_split:      # Step 5 数据集划分参数
step6_train:      # Step 6 模型训练参数（含 task_type/models/clf_models/HPO/SHAP/降维）
step7_sa_score:   # Step 7 可合成性打分参数
step8_similarity: # Step 8 相似度搜索参数
step9_predict:    # Step 9 模型推理参数
resources:        # 系统资源控制（优先级/CPU/内存）
```

### 关键路径配置

```yaml
paths:
  processed_npy:   "data/processed"    # Step1 → Step3 输入
  descriptor_npy:  "data/descriptors"  # Step3 输出 → Step4/5/6 输入
  pearson_npy:     "data/pearson"      # Step3 Pearson 过滤输出（优先级高于 descriptor_npy）
  samples_dir:     "data/samples"      # Step4 输出
  splits_dir:      "data/splits"       # Step5 输出
  training_output: "outputs/training"  # Step6 输出
```

> **自动路径推断规则**：  
> - Step3 `input_npy` 留空 → 自动用 `data/processed/<step1.output_name>`  
> - Step4/5/6 `input_npy` 留空 → 若 `step3.pearson_filter=true` 则用 `data/pearson/<*_pearson.npy>`，否则用 `data/descriptors/<step3.output_name>`  
> - Step2/7/8/9 `input_npy` 留空 → 同 Step4/5/6 的推断逻辑

---

## 5. 总控脚本 molearn_run.py

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | 配置文件路径（默认 molearn.yaml）| `--config my.yaml` |
| `--step` | 指定步骤（可逗号/范围）| `--step 3,6` 或 `--step 1-6` |
| `--list` | 列出所有步骤状态后退出 | |
| `--dry-run` | 预演模式（只打印命令）| |
| `--seed` | 覆盖 yaml 中的 seeds | `--seed 42,123,456` |
| `--task-type` | 覆盖任务类型 | `--task-type classification` |
| `--init-dirs` | 创建所有目录后退出 | |

### 工作原理

1. 读取 `molearn.yaml` 获取所有配置
2. 为每个步骤构建 `MOLEARN_*` 环境变量（路径覆盖）
3. 以子进程方式启动各步骤的 Python 脚本
4. 各脚本读取环境变量，将输出重定向到 `data/` 统一目录

---

## 6. Web GUI

### 启动

```bash
python molearn_gui/molearn_gui.py [--host 0.0.0.0] [--port 5000] [--debug]
```

- 默认监听 `http://127.0.0.1:5000`（仅本机）
- `--host 0.0.0.0` 可局域网访问

### 功能面板说明

| 侧边栏入口 | 功能 |
|-----------|------|
| 总览 & 快速启动 | 项目摘要、一键运行、实时日志 |
| 流水线步骤 | 所有 9 步的开关管理与单独运行 |
| 结果浏览 | 浏览 outputs/ 下所有输出文件 |
| 项目信息 | 修改项目名称/描述/版本 |
| 目录设置 | 修改所有 14 个路径键，一键创建目录 |
| Step 1~9 | 每步独立配置面板（含所有参数）|
| HPO & SHAP | 超参数优化 + SHAP 分析设置 |
| 降维 | PCA/KPCA/TSVD/UMAP/Autoencoder 设置 |
| 特征开关 | 14 种特征类型开关 |
| 系统资源 | 进程优先级/CPU 核数/内存限制 |

### GUI API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET  /get_config` | GET | 获取当前 molearn.yaml 内容（JSON）|
| `POST /save_config` | POST | 深度合并并保存配置 |
| `POST /run` | POST | 启动流水线（`{steps, seeds, dry_run, task_type}`）|
| `POST /stop` | POST | 停止运行中的流水线 |
| `GET  /log_stream` | GET | SSE 实时日志流 |
| `GET  /log_poll` | GET | 备用日志轮询（`?offset=N`）|
| `GET  /status` | GET | 运行状态（idle/running/done/error）|
| `GET  /outputs` | GET | 列出所有输出文件 |
| `GET  /list_npy` | GET | 列出 data/ 下所有 .npy 文件 |
| `GET  /read_file` | GET | 读取输出文件内容（`?path=...`）|
| `POST /init_dirs` | POST | 创建所有目录 |

---

## 7. 九步流水线详解

### Step 1 · 数据预处理（create_npy.py）

**功能**：从 `.xyz` 文件 + Excel 标签文件生成标准化的分子 `.npy` 数据库。

**输入**：
- `data/raw/creat_npy.xlsx`：配置表，每行指定一批分子（xyz 目录、标签 xlsx、输出 npy 名）
- `data/raw/xyz/`：`.xyz` 格式三维坐标文件

**输出**：`data/processed/dataset.npy`

```yaml
step1_data:
  enabled:     true
  xlsx_config: "data/raw/creat_npy.xlsx"  # 配置表（含 xyz_path/xlsx_file/npy_path 列）
  output_name: "dataset.npy"
```

**creat_npy.xlsx 格式**：

| xyz_path | xlsx_file | npy_path | key_idx | y_idx | flag_idx |
|---------|-----------|---------|---------|-------|---------|
| data/raw/xyz | data/raw/project-1.xlsx | dataset.npy | 0 | 3 | 5 |

---

### Step 2 · 数据库分析（db_analysis.py）

**功能**：可视化分析分子数据库，包括 Y 属性分布、PCA/t-SNE/UMAP 散点图、官能团统计、Tanimoto 多样性、Morgan 位频率。

**输入**：`data/descriptors/dataset-fp.npy`（默认）或手动指定  
**输出**：`outputs/analysis/`（各类 PNG 图表 + CSV 统计文件）

```yaml
step2_analysis:
  enabled:          false
  input_npy:        ""           # 留空=自动推断
  run_y_dist:       true         # Y 分布（直方图/CDF/QQ/箱线/小提琴）
  run_correlation:  true         # 描述符相关性热图
  run_dim_scatter:  true         # PCA/t-SNE/UMAP 散点图
  run_func_group:   true         # 官能团 SMARTS 分析
  run_diversity:    true         # Tanimoto 多样性分布
  run_morgan_bits:  true         # Morgan 位频率统计
  y_label:          "Property"
  diversity_sample: 500          # 多样性计算采样数（大库时降低此值）
```

**单独运行**：
```bash
python 2databaseAnalysis/db_analysis.py
```

---

### Step 3 · 描述符计算 + Pearson 过滤（create_by_fp.py）

**功能**：为每个分子计算多类型分子描述符，可选执行 Pearson 共线性过滤。

**输入**：`data/processed/dataset.npy`（step1 输出）  
**输出**：
- 描述符 npy：`data/descriptors/dataset-fp.npy`
- Pearson 过滤后 npy（启用时）：`data/pearson/dataset-fp_pearson.npy`
- Pearson 报告：`data/pearson/pearson_removal_report.xlsx`（3 个 sheet）
- Pearson 热图：`data/pearson/pearson_heatmap.png`

#### 支持的描述符类型

| 描述符 | 字段名 | 依赖 | 说明 |
|--------|--------|------|------|
| RDKit 物化描述符 | `rdkit_descriptor` | rdkit | ~200 维物理化学描述符 |
| RDKit 拓扑指纹 | `rdkit_fp` | rdkit | 2048 位路径型指纹 |
| MACCS Keys | `maccs_descriptor` | rdkit | 167 位结构键 |
| Morgan / ECFP | `morgan_descriptor` | rdkit | 圆形指纹（可选 FCFP/计数）|
| AtomPair | `atompair_descriptor` | rdkit | 原子对指纹 |
| Torsion | `torsion_descriptor` | rdkit | 拓扑扭转指纹 |
| Avalon | `avalon_descriptor` | rdkit | Avalon 指纹 |
| SOAP | `soap_descriptor` | dscribe | 平滑重叠原子密度（需3D）|
| ACSF | `acsf_descriptor` | dscribe | 原子中心对称函数（需3D）|
| MBTR | `mbtr_descriptor` | dscribe | 多体张量表示（需3D）|
| Mordred | `mordred_descriptor` | mordred | 1800+ 2D/3D 描述符 |
| 分子属性 | `prop_descriptor` | rdkit | MW/HBA/HBD/LogP/TPSA/RotBonds/AromaticRings |

#### Pearson 共线性过滤配置

```yaml
step3_descriptor:
  pearson_filter:          false   # true=启用
  pearson_threshold:       0.95    # |r| > 阈值的对，删除低方差特征
  pearson_output_npy:      ""      # 留空=自动命名 dataset-fp_pearson.npy
  pearson_report_xlsx:     ""      # 留空=自动命名
  pearson_gen_heatmap:     true    # 是否生成热图
  pearson_heatmap_max_dim: 300     # 超过此维度不生成（避免图太大）
```

**Pearson 报告 Excel 内容（3 sheet）**：
1. `过滤汇总`：按列记录每列是否保留及被移除原因
2. `共线对明细`：所有 |r| > 阈值的特征对
3. `统计摘要`：总特征数、保留数、剔除数

**单独运行 pearson_filter.py**：
```bash
python 3descriptor/pearson_filter.py \
  --input data/descriptors/dataset-fp.npy \
  --threshold 0.90 \
  --output data/pearson/filtered.npy \
  --mode report    # report | filter | both（默认 both）
```

---

### Step 4 · 数据抽样（sample_npy.py）

**功能**：从主数据库中按多种策略抽取子样本，用于小规模实验或计算资源受限时的替代数据集。

**输入**：与 step4/5/6 相同的自动推断逻辑（descriptor_npy 或 pearson_npy）  
**输出**：`data/samples/sub_<method>_n<N>_seed<S>.npy`

```yaml
step4_sampling:
  enabled:        false
  sample_size:    100          # 绝对数 或 0~1 比例
  seeds:          [42,123,456] # 每个种子生成一套子集
  run_random:     true
  run_systematic: true         # 等间隔系统抽样
  run_stratified: true         # 按 y 值分层抽样
  run_lhs:        true         # Latin Hypercube 正交抽样
  run_diversity:  true         # MaxMin 最大化多样性
  stratified_bins: 5
  lhs_feature:   "morgan_descriptor"
```

---

### Step 5 · 数据集划分（dataset_split.py）

**功能**：将数据库划分为训练集、验证集、测试集。

**输出**：`data/splits/train.npy`、`valid.npy`、`test.npy`、`split_info.csv`

```yaml
step5_split:
  method:       "random"       # random | stratified | scaffold
  train_ratio:  0.8
  valid_ratio:  0.1
  test_ratio:   0.1
  n_bins:       5              # 分层抽样分箱数
  scaffold_type: "bemis_murcko"  # scaffold 时的骨架算法
  seed:         42
```

---

### Step 6 · 模型训练（ml-m-full.py）

**功能**：使用配置的模型集合训练机器学习模型，支持回归和分类双模式，包含交叉验证、HPO、SHAP 分析、降维、模型卡生成。

#### 任务类型

```yaml
step6_train:
  task_type: "regression"     # regression | classification
```

#### 回归模式（16 种模型）

LinearRegression, Ridge, LassoCV, ElasticNetCV, HuberRegressor, BayesianRidge, DecisionTree, RandomForest, ExtraTreesRegressor, GradientBoosting, HistGBR, AdaBoost, BaggingRegressor, SVR, KNeighbors, MLP

**评估指标**：MAE, RMSE, R², Pearson r

#### 分类模式（12 种模型）

LogisticRegression, RidgeClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGBClassifier, AdaBoostClassifier, MLPClassifier, GaussianNB

**评估指标**：Accuracy, F1（weighted）, ROC-AUC（OvR）  
**额外输出**：混淆矩阵 PNG、分类报告 txt、`task_type.pkl`、`classes.pkl`

#### 输出目录结构（每个种子）

```
outputs/training/<项目名>/seed_42/
├── models/
│   ├── GradientBoosting.joblib      # 训练好的模型（Pipeline）
│   ├── RandomForest.joblib
│   ├── y_scaler.pkl                 # 回归：y 标准化器
│   ├── task_type.pkl                # 任务类型标识
│   ├── classes.pkl                  # 分类：类别标签列表
│   ├── dim_reducer.pkl              # 降维模型（启用时）
│   └── training_columns.pkl         # 训练时使用的特征列名
├── images/
│   ├── pred_vs_true_*.png           # 回归：预测 vs 真实散点图
│   └── confusion_matrix_*.png       # 分类：混淆矩阵
├── shap/
│   ├── shap_bar_*.png              # SHAP 特征重要性柱状图
│   └── shap_beeswarm_*.png         # SHAP 蜂群图
├── results.txt                     # 所有模型的评估指标汇总
└── model_card.txt                  # 模型卡（含配置/参数/性能）
```

#### SHAP 分析

```yaml
step6_train:
  shap_enable:  true
  shap_samples: 1000     # 背景样本数（越多越精确但越慢）
```

#### 降维

```yaml
step6_train:
  dim_reduction:
    method:     "pca"        # none|pca|kpca|tsvd|umap|autoencoder
    n_components: 50
    variance_ratio: 0.95     # PCA 自动选维数（null=不用）
```

---

### Step 7 · 可合成性打分（sa_score.py）

**功能**：对分子库中每个分子计算合成可行性评分。

**输出**：`outputs/sa_scores/synthesizability.csv`、各方法得分分布图

```yaml
step7_sa_score:
  calc_sa_score:   true    # SA Score（RDKit 内置）：1=易合成，10=难合成
  calc_scscore:    true    # SCScore（需 pip install scscore）
  calc_syba:       false   # SYBA（需 pip install syba）
  calc_rascore:    false   # RAscore（需额外安装）
  calc_simple:     true    # 简易 Morgan 复杂度评分（内置）
  weights:                 # 加权综合评分权重（各方法权重之和建议为 1）
    sa_score: 0.4
    scscore:  0.3
    simple_score: 0.3
```

---

### Step 8 · 相似度搜索（similarity_search.py）

**功能**：给定查询分子集，在数据库中搜索 Tanimoto 相似度高于阈值的分子，返回 Top-N 结果。

**注意**：`query_npy` 必须手动指定（不可自动推断）。

**输出**：
- `outputs/similarity/similarity_hits.csv`：所有命中的查询-数据库对
- `outputs/similarity/hits_by_query.txt`：按查询分子分组的命中列表
- `outputs/similarity/similarity_top_n.csv`：每个查询的 Top-N 结果

```yaml
step8_similarity:
  enabled:         false
  query_npy:       "data/processed/query.npy"  # ⚠️ 必须手动指定
  database_npy:    ""          # 留空=自动推断
  fp_type:         "morgan"
  sim_threshold:   0.7
  top_n:           10
  save_full_matrix: false
```

---

### Step 9 · 模型推理（usemodel.py）

**功能**：使用训练好的模型对新分子数据进行预测，自动检测任务类型（回归/分类）。

**注意**：`predict_npy` 必须手动指定，且该 npy 需包含与训练时相同的描述符字段。

**输出**：`outputs/predictions/predictions.csv`

```yaml
step9_predict:
  enabled:      false
  predict_npy:  "data/processed/new_mols.npy"  # ⚠️ 必须手动指定
  model_name:   "GradientBoosting"   # 回归模型名（或 RandomForestClassifier 等）
  task_type:    "auto"    # auto=自动读取 task_type.pkl | regression | classification
  model_dir:    ""        # 留空=自动推断 outputs/training/<项目名>/seed_42/
```

**分类输出列**：`y_pred`（预测类别）+ `prob_class_0`、`prob_class_1` ... 等概率列

---

## 8. 环境变量接口（MOLEARN_*）

`molearn_run.py` 通过以下环境变量将路径传递给子脚本。各子脚本在读取后**优先使用环境变量值**，本地 CONFIG 作为后备默认值。

| 环境变量 | 使用步骤 | 说明 |
|---------|---------|------|
| `MOLEARN_XLSX` | Step 1 | create_npy.py 配置表路径 |
| `MOLEARN_OUTPUT_DIR` | 所有步骤 | 输出目录（覆盖脚本本地默认）|
| `MOLEARN_OUTPUT_NAME` | Step 1, 3 | 输出文件名 |
| `MOLEARN_INPUT_NPY` | Step 2~9 | 输入 npy 文件路径 |
| `MOLEARN_QUERY_NPY` | Step 8 | 相似度搜索查询分子 npy |
| `MOLEARN_PREDICT_NPY` | Step 9 | 推理输入 npy |
| `MOLEARN_MODEL_DIR` | Step 9 | 模型目录 |
| `MOLEARN_MODEL_NAME` | Step 9 | 模型名称 |
| `MOLEARN_TASK_TYPE` | Step 9 | 任务类型（auto/regression/classification）|
| `MOLEARN_PEARSON_FILTER` | Step 3 | 是否启用 Pearson 过滤 |
| `MOLEARN_PEARSON_THRESHOLD` | Step 3 | 相关性阈值 |
| `MOLEARN_PEARSON_OUTPUT_NPY` | Step 3 | Pearson 过滤输出路径 |
| `MOLEARN_PEARSON_REPORT` | Step 3 | Pearson 报告 xlsx 路径 |
| `MOLEARN_PEARSON_HEATMAP` | Step 3 | 是否生成热图 |
| `MOLEARN_PEARSON_MAX_DIM` | Step 3 | 热图最大维度 |
| `MOLEARN_CONFIG` | Step 6 | 临时 config txt 路径（ml-m-full.py）|

---

## 9. 分类任务使用说明

### 数据格式

- 分子 `.npy` 中的 `y` 字段必须为**整数类别标签**（如 0, 1, 2, 3）
- 不支持字符串类别，请预先编码为整数

### 配置

```yaml
step6_train:
  task_type: "classification"
  split_method: "stratified"    # 分类任务推荐分层划分
  clf_models:
    RandomForestClassifier:       true
    GradientBoostingClassifier:   true
    SVC:                          true
    # ... 其余 9 种模型按需开关
```

### 启动分类训练

```bash
# 通过 yaml 配置
python molearn_run.py --step 6

# CLI 覆盖（临时改为分类，不修改 yaml）
python molearn_run.py --step 6 --task-type classification
```

### 推理时自动检测

```bash
# 训练完成后 task_type.pkl 已保存，推理时自动检测
python molearn_run.py --step 9
```

---

## 10. 依赖安装

详见 `requirements.txt` 和 `INSTALL.md`。

### 必须依赖

```bash
pip install numpy scipy pandas scikit-learn joblib optuna shap \
            matplotlib seaborn rdkit openpyxl xlrd pyyaml flask flask-cors
```

### 可选依赖

```bash
pip install mordred          # Mordred 描述符
pip install dscribe ase      # SOAP/ACSF/MBTR 描述符（需 3D 坐标）
pip install umap-learn       # UMAP 降维
pip install torch            # Autoencoder 降维
pip install scscore          # SCScore 合成可行性
pip install syba             # SYBA 合成可行性
```

### 使用 conda（推荐）

```bash
conda create -n molearn python=3.10
conda activate molearn
conda install -c rdkit rdkit
pip install -r requirements.txt
```

---

## 常见问题

**Q: Step3 输出后 Step6 还是用了旧的 npy？**  
A: 检查 `step3_descriptor.pearson_filter` 是否为 `true`——启用时 Step6 自动使用 `data/pearson/` 下的过滤后 npy，而非 `data/descriptors/` 下的原始描述符 npy。

**Q: 单独运行子脚本时路径不对？**  
A: 单独运行时 `MOLEARN_*` 环境变量未设置，脚本使用各自顶部 CONFIG 区域的本地默认路径（如 `sampled_npy/`）。若需统一使用 `data/` 目录，请通过 `molearn_run.py` 或手动设置环境变量运行。

**Q: GUI 保存配置后注释丢失？**  
A: GUI 使用 `pyyaml.dump()` 保存，注释会丢失，这是已知限制。建议在 GUI 中修改参数后，再手动核对关键注释是否保留。

**Q: 分类任务标签如何准备？**  
A: `y` 字段需为整数（0, 1, 2...）。可在 Excel 中预处理（如 IC50 分级），再通过 Step1 的 `y_col` 读入正确列。
