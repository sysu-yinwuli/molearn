# Molearn — 分子机器学习工作流

**Molearn** 是一套面向分子/材料科学的端到端机器学习工作流，覆盖数据预处理、描述符计算、模型训练、可合成性打分和推理预测，支持命令行一键总控和浏览器图形界面两种操作方式。

---

## 目录结构总览

```
molearn/
├── molearn.yaml              ← 全项目统一配置文件（总控入口）
├── molearn_run.py            ← CLI 总控脚本（Linux / Windows PowerShell / VSCode）
├── molearn_gui/              ← Web 图形界面（Flask，浏览器操作）
│   ├── molearn_gui.py        ← Flask 服务器
│   ├── templates/index.html  ← 单页 Web UI
│   └── README.md             ← GUI 独立说明
├── INSTALL.md                ← 安装指南（Python 环境 + 可选依赖）
│
├── data/                     ← 统一数据目录（运行后自动创建）
│   ├── raw/                  ← 原始输入文件
│   │   ├── gjf/              ← Gaussian 输入文件 (.gjf)
│   │   └── xyz/              ← 坐标文件 (.xyz，gjf2xyz 输出)
│   ├── processed/            ← create_npy 生成的基础 .npy 数据集
│   ├── descriptors/          ← create_by_fp 生成的带描述符 .npy
│   ├── samples/              ← sample_npy 抽样子集
│   └── splits/               ← dataset_split 划分后的 train/valid/test
│
├── outputs/                  ← 统一输出目录（运行后自动创建）
│   ├── training/             ← ml-m-full.py 训练结果（模型、图表、模型卡）
│   ├── analysis/             ← db_analysis.py 数据库分析结果
│   ├── sa_scores/            ← sa_score.py 可合成性打分结果
│   ├── similarity/           ← similarity_search.py 相似度矩阵
│   └── predictions/          ← usemodel.py 推理结果
│
├── 1dataProcess/             ← 数据预处理脚本
├── 2databaseAnalysis/        ← 数据库分析 + 打分 + 相似度
├── 3descriptor/              ← 描述符计算
├── 4machineLearing/          ← 机器学习训练
└── 5modelApplication/        ← 模型推理
```

> **提示**：首次运行前执行 `python molearn_run.py --init-dirs` 可自动创建全部 `data/` 和 `outputs/` 子目录。

---

## 快速开始

### 前提条件

```bash
# 最小依赖
pip install numpy pandas scikit-learn rdkit scipy

# 完整功能（详见 INSTALL.md）
pip install flask pyyaml seaborn matplotlib joblib
```

### 方式一：命令行总控（推荐）

```bash
# 1. 查看所有步骤状态
python molearn_run.py --list

# 2. 初始化目录结构
python molearn_run.py --init-dirs

# 3. 运行所有 enabled 步骤
python molearn_run.py

# 4. 只运行特定步骤
python molearn_run.py --step 6            # 仅训练
python molearn_run.py --step 3,6          # 描述符 + 训练
python molearn_run.py --step 3-6          # 步骤 3~6 全部运行

# 5. 预演（打印命令但不执行）
python molearn_run.py --step 6 --dry-run

# 6. 覆盖随机种子
python molearn_run.py --step 6 --seed 42,123,456
```

### 方式二：Web 图形界面（Windows/Linux/macOS 通用）

```bash
# 安装 Flask（仅需一次）
pip install flask pyyaml

# 启动 GUI（默认 http://localhost:5000）
python molearn_gui/molearn_gui.py

# 局域网共享（其他设备可访问）
python molearn_gui/molearn_gui.py --host 0.0.0.0 --port 5000
```

浏览器访问 `http://localhost:5000` 即可使用全功能图形界面。

> **注意**：Web GUI 是独立组件，无需 `molearn_run.py` 即可单独运行；它通过 API 在后台调用 `molearn_run.py`。

---

## molearn.yaml — 全项目配置文件

`molearn.yaml` 是整个项目唯一需要编辑的配置文件，所有步骤的参数均在此统一设置。各子脚本（`ml-m-full.py`、`create_by_fp.py` 等）仍保留各自的独立配置，可单独运行。

### 配置文件结构

```yaml
project:
  name:        "my_molearn_project"   # 项目名称
  description: "分子机器学习项目"      # 项目描述
  author:      ""                      # 作者（写入 model_card.txt）

paths:                                 # 统一数据目录映射（见上方目录结构）
  data_root:        "data"
  raw_gjf:          "data/raw/gjf"
  processed_npy:    "data/processed"
  descriptor_npy:   "data/descriptors"
  training_output:  "outputs/training"
  # ... 其余路径字段

step1_data:                            # 各步骤开关 + 参数
  enabled: true
  xlsx_config: "data/raw/creat_npy.xlsx"

step3_descriptor:
  enabled: true
  input_npy:  ""                       # 空 = 自动链接上一步输出
  output_npy: "data/descriptors/dataset_fp.npy"
  if_rdkit:   1
  if_maccs:   1
  if_morgan:  1
  # ...

step6_train:
  enabled: true
  models:
    LinearRegression: true
    RandomForest:     true
    # ... 16 种模型
  features:
    if_rdkit:  1
    if_morgan: 1
    # ... 14 种描述符开关
  hpo_enable: false
  dim_reduction:
    method:      "none"                # none|pca|kpca|tsvd|umap|autoencoder
    n_components: 50

resources:
  priority:    "normal"               # low|normal|high
  cpu_cores:   -1                     # -1=全核；N=限制核数（Linux）
  mem_limit_gb: 0                     # 0=不限；N=内存上限（Linux）
```

### 步骤编号对照

| 步骤 | 名称 | 对应脚本 |
|------|------|----------|
| 1 | 数据预处理 | `1dataProcess/create_npy.py` |
| 2 | 数据库分析 | `2databaseAnalysis/db_analysis.py` |
| 3 | 描述符计算 | `3descriptor/create_by_fp.py` |
| 4 | 数据抽样 | `1dataProcess/sample_npy.py` |
| 5 | 数据集划分 | `4machineLearing/dataset_split.py` |
| 6 | 模型训练 | `4machineLearing/ml-m-full.py` |
| 7 | 可合成性打分 | `2databaseAnalysis/sa_score.py` |
| 8 | 相似度搜索 | `2databaseAnalysis/similarity_search.py` |
| 9 | 模型推理 | `5modelApplication/usemodel.py` |

---

## molearn_run.py — CLI 总控脚本

### 完整 CLI 参数

```
usage: molearn_run.py [-h] [--config CONFIG] [--step STEP] [--seed SEED]
                       [--dry-run] [--list] [--init-dirs]

选项：
  --config CONFIG   YAML 配置文件路径（默认：./molearn.yaml）
  --step  STEP      指定步骤：单个(6)、逗号(3,6)、范围(3-6)
  --seed  SEED      覆盖训练随机种子，逗号分隔（如 42,123）
  --dry-run         预演模式：打印命令但不实际执行
  --list            列出所有步骤及其 enabled 状态后退出
  --init-dirs       按 paths 节点创建全部目录后退出
```

### 使用示例

```bash
# Windows PowerShell / VSCode Terminal
python molearn_run.py --list
python molearn_run.py --step 6
python molearn_run.py --step 3-6 --dry-run
python molearn_run.py --config D:\projects\my.yaml --step 6 --seed 42

# Linux / macOS
python3 molearn_run.py
python3 molearn_run.py --step 3,5,6 --seed 42,123
python3 molearn_run.py --init-dirs
```

### 步骤链接（自动路径推断）

`molearn_run.py` 支持步骤间的**自动路径链接**：当 `molearn.yaml` 中某步骤的 `input_npy` 为空时，系统自动使用上一步的 `output_npy` 路径作为输入，无需手动填写每一步的输入输出路径。

### 独立运行子脚本

每个子脚本均可**完全独立运行**，不依赖总控脚本：

```bash
# 直接运行子脚本（使用各自目录中的配置区/配置文件）
cd 3descriptor && python create_by_fp.py
cd 4machineLearing && python ml-m-full.py
cd 1dataProcess && python sample_npy.py
```

---

## Web 图形界面 (molearn_gui)

详见 [`molearn_gui/README.md`](molearn_gui/README.md)。

功能摘要：
- **9 个配置面板**：项目信息、路径、每步参数、模型选择、特征开关、HPO、降维、资源控制
- **实时日志流**：训练过程日志通过 SSE 实时推送到浏览器
- **一键操作**：启动/停止流水线、创建目录、浏览输出文件
- **Windows 友好**：纯浏览器界面，无需安装额外 GUI 框架

---

## 训练输出目录结构

```
outputs/training/
└── seed_42/
    ├── models/
    │   ├── RandomForest.joblib            ← 训练好的模型
    │   └── RandomForest_model_card.txt    ← 模型信息卡（新增）
    ├── images/
    │   └── RandomForest.png               ← 预测 vs 真实散点图
    ├── shap/
    │   ├── RandomForest_summary.png
    │   └── RandomForest_bar.png
    ├── training_columns.pkl               ← 特征列名
    ├── y_scaler.pkl                       ← 标签 StandardScaler
    └── dim_reducer.pkl                    ← 降维器（启用降维时）
```

### 模型信息卡 (`{ModelName}_model_card.txt`)

每个训练完成的模型在 `models/` 目录下自动生成一份完整的信息卡，包含 8 个部分：

| 章节 | 内容 |
|------|------|
| 1. Dataset | npy路径、样本量、特征维度、划分方法 |
| 2. Feature Flags | 所有 14 种描述符的 `[ON]`/`[off]` 状态 |
| 3. Dim Reduction | 降维方法及参数（或"已禁用"） |
| 4. Model Info | 模型类名、所属模块、是否使用 StandardScaler |
| 5. Hyperparameters | 所有超参数（sorted 字母序）|
| 6. HPO Best Params | 超参数优化最优结果（或"HPO 未启用"） |
| 7. Test Set Metrics | MAE / MSE / R²、训练耗时 |
| 8. References | scikit-learn + 模型算法 + 活跃描述符的学术引用 |

---

## 子模块 README

| 目录 | 内容 |
|------|------|
| [`1dataProcess/README.md`](1dataProcess/README.md) | 数据预处理、5 种抽样方法 |
| [`2databaseAnalysis/README.md`](2databaseAnalysis/README.md) | 数据库分析、可合成性打分、相似度搜索 |
| [`3descriptor/README.md`](3descriptor/README.md) | 12 种描述符类型、维度参考、选择指南 |
| [`4machineLearing/README.md`](4machineLearing/README.md) | 16 种模型、HPO、SHAP、批量训练、模型卡 |
| [`5modelApplication/README.md`](5modelApplication/README.md) | 模型推理、降维自动加载、错误排查 |
| [`INSTALL.md`](INSTALL.md) | Python 环境配置、可选依赖安装 |
| [`molearn_gui/README.md`](molearn_gui/README.md) | Web GUI 安装与使用 |
