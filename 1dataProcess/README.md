# 1dataProcess — 原始数据预处理与数据抽样

本目录负责将原始量化计算文件（`.gjf` / `.xyz`）和标签 Excel 合并，生成下游步骤所需的 `.npy` 数据集；同时提供从已有数据集中灵活抽样生成子数据集的功能。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `gjf2xyz.py` | 批量将 Gaussian `.gjf` 文件转换为 `.xyz` 格式 |
| `create_npy.py` | 从 Excel 配置表批量读取 xyz + 标签，生成多个 `.npy` 文件 |
| `create_npy-dir.py` | 扫描整个 xyz 目录直接生成单个 `.npy`（无 Excel 控制，适合无标签场景） |
| `creat_npy.xlsx` | `create_npy.py` 的批量任务配置表 |
| `sample_npy.py` | **数据抽样脚本**：从主数据集 npy 中以多种策略抽样生成若干子数据集 |

---

## 脚本详解

### gjf2xyz.py

将指定文件夹中所有 `.gjf` 文件批量转为 `.xyz`。

**修改参数（脚本底部）：**

```python
src_folder  = 'gjf'   # 输入：存放 .gjf 文件的文件夹路径
dest_folder = 'xyz'   # 输出：转换后 .xyz 文件存放路径
```

**运行：**

```bash
python gjf2xyz.py
```

**`.xyz` 文件格式：**

```
<原子总数>
<注释行（可为空）>
<元素符号>  <x>  <y>  <z>
<元素符号>  <x>  <y>  <z>
...
```

---

### create_npy.py

读取 `creat_npy.xlsx` 中的任务配置，对每一行生成一个 `.npy` 文件。

**`creat_npy.xlsx` 格式（每行一个任务）：**

| 列名 | 含义 | 示例 |
|------|------|------|
| `xyz_path` | xyz 文件所在目录 | `xyz` |
| `xlsx_file` | 包含分子名和标签的 Excel 文件路径 | `project-1.xlsx` |
| `key_idx` | 分子名所在列的索引（从 0 开始） | `0` |
| `y_idx` | 目标属性值所在列的索引 | `3` |
| `flag_idx` | 是否使用该样本的标志列索引（值为 0 时跳过） | `5` |
| `npy_path` | 输出 `.npy` 文件路径 | `dataset.npy` |

**运行：**

```bash
python create_npy.py
```

**输出 `.npy` 结构（每个元素为一个 dict）：**

```python
{
    'name':        'mol_001',        # 分子名（字符串）
    'elements':    [6, 6, 8, ...],   # 原子序数列表
    'coordinates': [[x,y,z], ...],   # 原子坐标列表（Å）
    'atom_count':  15,               # 原子数
    'y':           3.14              # 目标属性值
}
```

---

### create_npy-dir.py

扫描整个 xyz 目录，无需 Excel 标签文件，适用于待预测的新数据集（`y` 字段填充为占位值 `1`）。

**修改脚本顶部参数：**

```python
XYZ_DIR = 'xyz-add'       # 待扫描的 xyz 文件夹
OUT_NPY = 'hjf-add.npy'   # 输出 npy 文件名
```

**运行：**

```bash
python create_npy-dir.py
```

---

## sample_npy.py — 数据抽样

### 功能概览

从**主数据集**（已计算描述符的 `.npy`）中按指定策略抽取子集，生成若干**子数据集**（新的 `.npy` 文件）。支持：

- 多种抽样方式：随机、系统性、分层、拉丁超立方（LHS）、MaxMin 多样性
- 一次指定多个随机种子，批量生成多个子数据集（用于统计稳健性验证）
- 日志 CSV 记录每次抽样的元数据

### 配置参数（脚本顶部）

```python
# ---- 输入/输出 ----
INPUT_NPY   = '../3descriptor/output-fp.npy'   # 主数据集（需已计算描述符）
OUTPUT_DIR  = 'samples'                         # 子数据集保存目录

# ---- 抽样大小 ----
SAMPLE_SIZE = 100   # 每次抽取的分子数量

# ---- 随机种子列表（每个种子生成一个独立的子数据集）----
SEEDS = [42, 123, 456]   # 生成 3 个子数据集

# ---- 抽样方法开关（True = 运行该方法）----
RUN_RANDOM     = True    # 完全随机抽样
RUN_SYSTEMATIC = True    # 系统性等间隔抽样
RUN_STRATIFIED = True    # 分层抽样（按 y 值分箱）
RUN_LHS        = True    # 拉丁超立方抽样（PCA → LHS 网格 → KNN 匹配）
RUN_DIVERSITY  = True    # MaxMin 多样性抽样（基于 Tanimoto 相似度贪心选择）

# ---- 分层抽样参数 ----
STRATIFIED_N_BINS = 5    # y 值分箱数

# ---- LHS 参数 ----
LHS_FEATURE   = 'morgan_descriptor'   # 用于 PCA 的特征字段（需已计算）
LHS_PCA_DIMS  = 5                     # PCA 降至几维后做 LHS 网格

# ---- 多样性抽样参数 ----
DIV_FP_FIELD  = 'morgan_descriptor'  # 用于 Tanimoto 相似度的指纹字段
```

### 抽样方法说明

#### 1. 随机抽样（`RUN_RANDOM`）

最简单的无放回随机采样，每个分子被选中的概率相等。

- **优点**：无偏、快速
- **适用**：数据分布均匀时的基线方法

#### 2. 系统性抽样（`RUN_SYSTEMATIC`）

按索引等间隔选取（每 `N/n` 个取一个），随机起始偏移。

- **优点**：对有序数据保持均匀覆盖
- **适用**：数据集按某种物理量顺序存储时

#### 3. 分层抽样（`RUN_STRATIFIED`）

将目标属性 `y` 分成 `STRATIFIED_N_BINS` 个箱，在每个箱内按比例随机采样。

- **优点**：确保子集的 y 分布与原数据集一致，避免某些属性范围欠代表
- **适用**：有明确目标属性且分布不均匀时

#### 4. 拉丁超立方抽样（`RUN_LHS`）

基于分子描述符的 PCA 降维 + LHS 网格 + KNN 匹配实现正交设计抽样。

**流程：**
1. 对 `LHS_FEATURE` 字段做 PCA 降至 `LHS_PCA_DIMS` 维
2. 在低维空间生成 `SAMPLE_SIZE` 个 LHS 网格点（每维均匀覆盖）
3. 用 KNN 找到每个网格点最近的真实分子

- **优点**：兼顾覆盖均匀性（高维空间填充）和多样性，适合实验设计
- **适用**：化学空间探索，希望子集在特征空间中均匀分布

#### 5. MaxMin 多样性抽样（`RUN_DIVERSITY`）

基于 Tanimoto 指纹相似度的贪心最大最小距离选择：

1. 随机选取第一个分子作为初始集
2. 每次选取与已选集中**最相似分子的相似度最小**的分子（最大化最小相似度）

- **优点**：最大化分子多样性，去相关；适合生成多样化训练集
- **适用**：当主数据集中存在大量相似结构时，帮助提取代表性子集

### 输出文件结构

```
samples/
├── sample_random_seed42.npy         # 随机抽样，种子 42
├── sample_random_seed123.npy        # 随机抽样，种子 123
├── sample_random_seed456.npy        # 随机抽样，种子 456
├── sample_systematic_seed42.npy     # 系统性抽样（随机起始偏移随种子变化）
├── sample_systematic_seed123.npy
├── sample_systematic_seed456.npy
├── sample_stratified_seed42.npy     # 分层抽样
├── sample_stratified_seed123.npy
├── sample_stratified_seed456.npy
├── sample_lhs_seed42.npy            # LHS 抽样
├── sample_lhs_seed123.npy
├── sample_lhs_seed456.npy
├── sample_diversity_seed42.npy      # MaxMin 多样性抽样
├── sample_diversity_seed123.npy
├── sample_diversity_seed456.npy
└── sampling_log.csv                 # 抽样日志：方法、种子、实际抽样数、时间
```

### 子数据集 `.npy` 结构

每个子数据集 npy 与原数据集结构相同（保留所有字段），额外添加 `sample_info` 元数据：

```python
{
    ...原有字段...,
    'sample_info': {
        'method':      'random',         # 抽样方法名
        'seed':        42,               # 随机种子
        'sample_size': 100,             # 目标抽样数
        'actual_size': 100,             # 实际抽样数（可能因重复过滤而略少）
        'source_size': 5000,            # 源数据集总分子数
        'timestamp':   '2025-01-01T12:00:00',
    }
}
```

### 抽样日志 `sampling_log.csv`

```csv
method,seed,target_size,actual_size,source_size,timestamp,output_file
random,42,100,100,5000,2025-01-01T12:00:00,sample_random_seed42.npy
random,123,100,100,5000,2025-01-01T12:00:01,sample_random_seed123.npy
...
```

### 运行

```bash
cd 1dataProcess/
python sample_npy.py
```

### 依赖

```bash
# 基础（必选）
pip install numpy scikit-learn rdkit

# LHS 抽样（可选，无则跳过 LHS）
pip install scipy   # 用于 stats.qmc.LatinHypercube（scipy >= 1.7）
```

### 典型使用场景

**场景 1：生成多个随机子集用于集成模型验证**

```python
SEEDS = [1, 2, 3, 4, 5]   # 5 个种子
SAMPLE_SIZE = 200
RUN_RANDOM = True
# 其他方法关闭
```

**场景 2：构建化学空间均匀覆盖的实验设计集**

```python
SEEDS = [42]
SAMPLE_SIZE = 50
RUN_LHS = True
LHS_FEATURE = 'morgan_descriptor'
LHS_PCA_DIMS = 8
```

**场景 3：从高度相似的大型库中筛选多样化代表子集**

```python
SEEDS = [42, 100, 200]
SAMPLE_SIZE = 100
RUN_DIVERSITY = True
DIV_FP_FIELD = 'morgan_descriptor'
```

---

## 注意事项

- `.xyz` 文件第一列可以是**元素符号**（如 `C`）或**原子序数**（如 `6`），两个脚本均自动兼容。
- 分子名必须与 xyz 文件名（去扩展名）完全一致，区分大小写。
- `sample_npy.py` 的 LHS 和 MaxMin 方法需要 npy 中已包含描述符字段（`LHS_FEATURE` / `DIV_FP_FIELD`），请先运行 `3descriptor/create_by_fp.py`。
- 若某种方法的实际抽样数少于 `SAMPLE_SIZE`（如分子总数不足，或过滤后去重导致），脚本会打印 `[Warning]` 并返回所有可用分子。
- MaxMin 多样性抽样在大数据集（> 5000 分子）时速度较慢，可降低 `SAMPLE_SIZE` 或先对数据集随机预筛选。
