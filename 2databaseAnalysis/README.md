# 2databaseAnalysis — 数据库分析与可视化

本目录提供对已生成 `.npy` 数据集的统计分析和可视化功能，帮助了解数据分布、筛选结果、分子结构特征、可合成性以及相似度关系。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `add_score.py` | 将外部评分/筛选列从 Excel 注入现有 `.npy` 文件 |
| `image-database.py` | 生成数据库整体可视化图（属性分布直方图、箱线图等） |
| `model_result_hot.py` | 绘制模型预测结果热图 |
| `use_tsne-score.py` | 对特征做 t-SNE 降维可视化，并按评分着色 |
| `db_analysis.py` | **综合数据库分析**：Y分布多图 + 相关性 + PCA/t-SNE/UMAP + 官能团 + 多样性 + Morgan位分析 |
| `sa_score.py` | **可合成性打分**：SA Score / SCScore / SYBA / RAscore / Simple，附引用说明 |
| `similarity_search.py` | **相似度搜索**：查询分子 vs 数据库全集 Tanimoto 矩阵 + 阈值命中 + Top-N |

---

## 脚本详解

### add_score.py

将 Excel 中某几列数值（如筛选得分、百分位数）附加到 `.npy` 文件，以 `score_d` 字段存储。

**修改脚本顶部参数：**

```python
xlsx_file = 'sample-space-all-scored-percentile.xlsx'  # 包含评分的 Excel 文件
de_list   = [5]           # 要读取的列索引列表（0-based，可指定多列）
in_npy    = 'hjf-all-fp.npy'   # 输入 npy（已有特征）
out_npy   = 'hjf-all-fp-score.npy'  # 输出 npy（附加了 score_d）
```

**运行：**

```bash
python add_score.py
```

---

### image-database.py

对 `.npy` 中的目标属性 `y` 及可选特征进行批量可视化。输出图片保存到当前目录。

---

### model_result_hot.py

读取模型预测 CSV，绘制真实值 vs 预测值热图或残差分布图。

---

### use_tsne-score.py

对 `.npy` 中的特征矩阵进行 t-SNE 二维降维，并用 `score_d` 字段的值着色散点图。

---

## db_analysis.py — 综合数据库分析

### 功能概览

对单个 `.npy` 数据集执行全方位分析，输出统计图、CSV 汇总和文本报告。

### 配置参数（脚本顶部）

```python
# ---- 输入/输出 ----
INPUT_NPY    = '../3descriptor/output-fp.npy'   # 已计算描述符的 npy
OUTPUT_DIR   = 'db_analysis_output'             # 所有输出保存的目录

# ---- 模块开关（True = 运行该模块）----
RUN_Y_DIST      = True   # Y 属性分布（直方图/CDF/QQ图/箱线图/小提琴图）
RUN_CORRELATION = True   # 描述符相关性热图
RUN_DIM_SCATTER = True   # PCA / t-SNE / UMAP 二维散点（按 y 着色）
RUN_FUNC_GROUP  = True   # 官能团分析（SMARTS 频率统计）
RUN_DIVERSITY   = True   # Tanimoto 多样性分布
RUN_MORGAN_BITS = True   # Morgan 位频率分析

# ---- Y 分布参数 ----
Y_BINS      = 30     # 直方图分箱数
Y_LABEL     = 'Property (eV)'

# ---- 相关性热图参数 ----
CORR_FEATURE     = 'morgan_descriptor'  # 用于相关性的描述符字段
CORR_MAX_FEATURES = 50                  # 最多展示的特征维度（太高会导致图太密）

# ---- 降维散点参数 ----
DIM_FEATURE  = 'morgan_descriptor'   # 用于降维的描述符字段
TSNE_PERPLEXITY  = 30
TSNE_N_ITER      = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

# ---- 官能团分析参数 ----
FG_TOP_N         = 25   # 展示频率最高的 N 个官能团

# ---- Tanimoto 多样性参数 ----
DIVERSITY_FP     = 'morgan'   # 'morgan' | 'maccs' | 'rdkit'
DIVERSITY_SAMPLE = 500        # 随机采样分子数（大数据集加速）

# ---- Morgan 位分析参数 ----
MORGAN_TOP_N     = 50    # 展示频率最高的 N 个 Morgan 位
```

### 输出文件列表

| 输出文件 | 内容 |
|---------|------|
| `plot_y_histogram.png` | Y 属性值直方图（含 KDE 曲线） |
| `plot_y_cdf.png` | Y 属性值累积分布函数（CDF）曲线 |
| `plot_y_qqplot.png` | Y 属性值 QQ 图（正态性检验） |
| `plot_y_boxplot.png` | Y 属性值箱线图（含异常值标记） |
| `plot_y_violin.png` | Y 属性值小提琴图（分布形状 + 四分位） |
| `plot_feature_correlation.png` | 描述符特征间 Pearson 相关系数热图 |
| `plot_pca_2d.png` | PCA 二维散点（按 y 值着色） |
| `plot_tsne_2d.png` | t-SNE 二维散点（按 y 值着色） |
| `plot_umap_2d.png` | UMAP 二维散点（按 y 值着色，需安装 umap-learn） |
| `plot_functional_group_frequency.png` | 官能团出现频率条形图 |
| `plot_fg_molecule_heatmap.png` | 分子 × 官能团存在性热图（Top-N 分子） |
| `plot_tanimoto_diversity.png` | 分子间 Tanimoto 相似度分布直方图 |
| `plot_morgan_bits_frequency.png` | Morgan 位出现频率柱状图 |
| `functional_groups.csv` | 25+ 官能团频率统计表（含 SMARTS 模式） |
| `morgan_bits.csv` | Morgan 位 ID 及出现频率统计表 |
| `analysis_summary.txt` | 数值摘要：Y 分布统计量、多样性指标等 |

### 内置官能团（25+ 种 SMARTS 模式）

脚本内置 `_FUNCTIONAL_GROUPS` 字典，涵盖：
- 羟基 (`-OH`)、氨基 (`-NH₂/-NH-`)、羰基 (`-C=O`)、羧基 (`-COOH`)
- 酯基、酰胺、磺酰基、硝基、氰基、叠氮基
- 卤素（F/Cl/Br/I）、醚键、硫醚
- 苯环、芳香杂环（吡啶/呋喃/噻吩等）
- 双键、三键、烷基链

### 运行

```bash
cd 2databaseAnalysis/
python db_analysis.py
```

### 依赖

```bash
# 必选
pip install numpy matplotlib scikit-learn rdkit

# 可选（有则使用，无则跳过对应模块）
pip install seaborn umap-learn scipy
```

---

## sa_score.py — 可合成性打分

### 功能概览

对 `.npy` 数据集中的每个分子计算多种可合成性得分，归一化后输出综合排名，帮助筛选合成可行性更高的化合物。

### 配置参数（脚本顶部）

```python
# ---- 输入/输出 ----
INPUT_NPY   = '../3descriptor/output-fp.npy'
OUTPUT_DIR  = 'sa_score_output'

# ---- 打分方法开关（True = 启用）----
CALC_SA_SCORE   = True    # SA Score（Ertl & Schuffenhauer 2009）
CALC_SCSCORE    = True    # SCScore（Coley et al. 2018）
CALC_SYBA       = False   # SYBA（Vorsilak et al. 2020，需安装 syba 库）
CALC_RASCORE    = False   # RAscore（Thakkar et al. 2021，需安装 rascore 库）
CALC_SIMPLE     = True    # Simple Morgan 复杂度（内置，无需额外依赖）

# ---- 权重（各方法的综合分权重，总和建议为 1）----
SCORE_WEIGHTS = {
    'sa_score':     0.4,
    'scscore':      0.3,
    'syba':         0.0,
    'rascore':      0.0,
    'simple_score': 0.3,
}

# ---- 绘图参数 ----
PLOT_DIST       = True   # 绘制得分分布直方图
PLOT_SCATTER    = True   # 绘制各方法得分两两散点图
TOP_N_EASY      = 20     # 输出最易合成 Top-N 分子名
```

### 打分方法说明

| 方法 | 分值范围 | 含义 | 引用 |
|------|---------|------|------|
| **SA Score** | 1–10 | 合成可及性，越低越易合成 | Ertl & Schuffenhauer, *J. Cheminform.* **1**, 8 (2009) |
| **SCScore** | 1–5 | 合成复杂度，越低越易合成 | Coley et al., *J. Chem. Inf. Model.* **58**, 252 (2018) |
| **SYBA** | −∞ ~ +∞ | 正值=易合成，负值=难合成 | Vorsilak et al., *J. Cheminform.* **12**, 35 (2020) |
| **RAscore** | 0–1 | 越高表示越易合成 | Thakkar et al., *Chem. Sci.* **12**, 3339 (2021) |
| **Simple Score** | 0–1 | Morgan 位稀有度代理，越高越复杂 | 内置近似 |

所有得分归一化到 **[0, 1]**（0 = 最易合成，1 = 最难合成），并按 `SCORE_WEIGHTS` 加权得到 `combined_score`。

### 输出文件

| 文件 | 内容 |
|------|------|
| `sa_scores.csv` | 每个分子的各项得分 + 归一化值 + 综合得分 |
| `top_easy.txt` | 最易合成 Top-N 分子名单 |
| `plot_score_distribution.png` | 各方法得分分布直方图 |
| `plot_score_scatter.png` | 各方法得分两两散点相关图 |
| `score_summary.txt` | 分布统计量 + 引用说明 |

### 安装可合成性库

```bash
# SA Score：RDKit 内置，无需额外安装
# SCScore（近似模式内置，精确模式需安装）：
pip install scscore

# SYBA：
pip install syba

# RAscore：
pip install rascore
```

> **注意**：若未安装对应库，该方法自动跳过（或使用内置近似），不会中断程序。

### 运行

```bash
cd 2databaseAnalysis/
python sa_score.py
```

---

## similarity_search.py — 相似度搜索

### 功能概览

将**查询集**（若干目标分子）与**数据库**（完整分子库）逐一做 Tanimoto 相似度比较，输出相似度矩阵、高相似分子命中列表（阈值筛选）、Top-N 结果及可视化图表。

### 配置参数（脚本顶部）

```python
# ---- 输入文件 ----
QUERY_NPY    = 'query.npy'       # 查询分子 npy（含 1~N 个分子）
DATABASE_NPY = '../3descriptor/all-fp.npy'  # 数据库 npy（全集）

# ---- 输出目录 ----
OUTPUT_DIR   = 'similarity_output'

# ---- 指纹类型（选一种作为相似度计算依据）----
FP_TYPE = 'morgan'   # 'morgan' | 'maccs' | 'rdkit' | 'atompair' | 'torsion'

# ---- Morgan 参数（仅 FP_TYPE='morgan' 时有效）----
MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048

# ---- 阈值筛选 ----
SIM_THRESHOLD = 0.7   # Tanimoto ≥ 此值视为"命中"

# ---- Top-N 输出 ----
TOP_N = 10   # 每个查询分子输出最相似的 N 个数据库分子

# ---- 输出选项 ----
SAVE_FULL_MATRIX = True   # 是否保存完整相似度矩阵（查询数×数据库数的 CSV，可能很大）
PLOT_HEATMAP     = True   # 是否绘制相似度热图
PLOT_DIST        = True   # 是否绘制相似度分布直方图
PLOT_TOPN_BAR    = True   # 是否为每个查询分子绘制 Top-N 条形图
```

### 支持的指纹类型

| `FP_TYPE` | 描述 | 适用场景 |
|-----------|------|---------|
| `morgan` | ECFP（Morgan 圆形指纹） | 最常用，子结构敏感 |
| `maccs` | MACCS 167-bit 结构键 | 官能团层面相似度 |
| `rdkit` | RDKit 路径指纹 | 拓扑路径相似度 |
| `atompair` | 原子对指纹 | 关注原子对距离 |
| `torsion` | 拓扑扭转指纹 | 关注键-扭转路径 |

### 输出文件

| 文件 | 内容 |
|------|------|
| `similarity_matrix.csv` | 完整相似度矩阵（行=查询，列=数据库，值=Tanimoto） |
| `similarity_hits.csv` | 所有"命中"记录（相似度 ≥ 阈值），含查询名/数据库分子名/相似度 |
| `hits_by_query.txt` | 按查询分子分组，列出每个查询的命中分子名单 |
| `similarity_top_n.csv` | 每个查询分子的 Top-N 最相似数据库分子，按相似度降序排列 |
| `plot_similarity_heatmap.png` | 相似度矩阵热图（查询 × 数据库子集） |
| `plot_similarity_distribution.png` | 整体相似度分布直方图（含阈值竖线标记） |
| `plot_topn_<query_name>.png` | 每个查询分子的 Top-N 相似度条形图 |

### 典型使用场景

**场景 1：虚拟筛选命中验证**

已知数个活性分子（查询集），想从大型数据库中找出结构相似的候选：

```python
QUERY_NPY    = 'active_hits.npy'       # 已知活性分子
DATABASE_NPY = 'virtual_library.npy'   # 虚拟库
SIM_THRESHOLD = 0.6                    # 相似度阈值
TOP_N = 20
```

**场景 2：去重/多样性筛选**

检查两个数据集之间是否有高度重复分子：

```python
QUERY_NPY    = 'new_compounds.npy'
DATABASE_NPY = 'existing_database.npy'
SIM_THRESHOLD = 0.9   # 高阈值 = 近似重复
```

**场景 3：骨架跳跃分析**

寻找与查询分子相似但骨架不同的分子：

```python
FP_TYPE = 'maccs'   # 官能团层面相似度
SIM_THRESHOLD = 0.5
```

### 运行

```bash
cd 2databaseAnalysis/
python similarity_search.py
```

### 依赖

```bash
pip install numpy pandas matplotlib rdkit
```

---

## 综合工作流

```
1dataProcess/dataset.npy
      ↓
3descriptor/create_by_fp.py   →   dataset-fp.npy
      ↓
┌────────────────────────────────────────────────────┐
│  2databaseAnalysis/                                │
│                                                    │
│  db_analysis.py         ← 全面数据分布分析          │
│  sa_score.py            ← 可合成性筛选              │
│  similarity_search.py   ← 结构相似度搜索            │
│  add_score.py           ← 注入外部评分              │
│  image-database.py      ← 快速可视化                │
└────────────────────────────────────────────────────┘
      ↓
4machineLearing/ml-m-full.py   （模型训练）
```

---

## 注意事项

- `db_analysis.py` 中 t-SNE 对大数据集（> 5000 分子）运行较慢，建议设置 `DIVERSITY_SAMPLE` 限制采样数，或先使用 PCA 降维再做 t-SNE。
- `similarity_search.py` 的完整相似度矩阵（`SAVE_FULL_MATRIX = True`）在查询数 × 数据库数较大时可能产生 GB 级 CSV，建议设为 `False` 仅保留命中和 Top-N 结果。
- `sa_score.py` 的 SCScore 在未安装 `scscore` 库时会自动使用内置近似（基于 RDKit 描述符），精度略低但无需额外依赖。
- UMAP 需安装 `umap-learn`（`pip install umap-learn`），未安装时 `db_analysis.py` 会跳过 UMAP 图并提示。
