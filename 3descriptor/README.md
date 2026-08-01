# 3descriptor — 分子描述符计算

本目录负责对 `.npy` 分子数据集计算各类分子描述符，并将结果写回 `.npy`，供 `4machineLearing` 模块读取。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `create_by_fp.py` | **主描述符计算脚本**：支持 12 类描述符，涵盖 RDKit / MACCS / Morgan / AtomPair / Torsion / Avalon / SOAP / ACSF / MBTR / Mordred / 分子属性 |
| `add_extra_de.py` | 从单个 Excel 文件向 npy 注入额外数值描述符（`extra_d` 字段） |
| `add_extra_de-multi.py` | 从多个 Excel 文件合并注入额外数值描述符（多分表场景） |

---

## create_by_fp.py

核心描述符计算脚本，支持 **12 类描述符**，所有参数集中在顶部 `CONFIG` 字典中。

### CONFIG 参数说明

```python
CONFIG = {
    # ---- 路径 ----
    'xyz_folder':     '../1dataProcess/xyz/',  # xyz 文件所在目录
    'smi_file':       'output.smi',            # obabel 中间 SMILES 文件（临时）
    'input_npy':      'input.npy',             # 输入 npy（来自 1dataProcess）
    'output_npy':     'output-fp.npy',         # 输出 npy（附加了描述符）
    'failed_record':  'failed.xlsx',           # 失败分子记录 Excel

    # ---- 元素全集（影响 SOAP/ACSF/MBTR 维度）----
    'global_species': ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'],

    # ---- 描述符总开关（True/False）----
    'calc_rdkit':     True,    # RDKit 物化描述符 + RDKit 路径指纹
    'calc_maccs':     True,    # MACCS 167-bit 结构键
    'calc_morgan':    True,    # Morgan 圆形指纹（ECFP / FCFP，位向量 / 计数）
    'calc_atompair':  False,   # AtomPair 指纹（原子对拓扑距离编码）
    'calc_torsion':   False,   # Topological Torsion 指纹（四原子路径扭转）
    'calc_avalon':    False,   # Avalon 指纹（需 rdkit.Avalon 或 rdBase）
    'calc_soap':      False,   # SOAP 对称函数（需 dscribe + ase）
    'calc_acsf':      False,   # ACSF 对称函数（需 dscribe + ase）
    'calc_mbtr':      False,   # MBTR 多体张量（需 dscribe + ase）
    'calc_mordred':   False,   # Mordred 全局描述符（1800+维，需 mordred 库）
    'calc_prop':      True,    # 基本分子属性（11 个 RDKit 计算量，无需额外依赖）

    # ---- RDKit 指纹参数 ----
    'rdkit_fp_size':  2048,    # RDKit 路径指纹位数

    # ---- Morgan 参数 ----
    'morgan_radius':        2,      # 圆半径（2 = ECFP4，3 = ECFP6）
    'morgan_nbits':         2048,   # 指纹位数
    'morgan_use_features':  False,  # False = ECFP（原子标识符），True = FCFP（特征标识符）
    'morgan_use_counts':    False,  # False = 位向量，True = 计数向量（值 > 1 表示重复子结构）

    # ---- AtomPair 参数 ----
    'atompair_nbits': 2048,    # AtomPair 指纹位数

    # ---- Torsion 参数 ----
    'torsion_nbits':  2048,    # Topological Torsion 指纹位数

    # ---- Avalon 参数 ----
    'avalon_nbits':   2048,    # Avalon 指纹位数

    # ---- SOAP 超参（已优化默认值）----
    'soap': {
        'r_cut':    5.0,      # 截断半径（Å）
        'n_max':    8,        # 径向基函数数（原 6，提升精度）
        'l_max':    6,        # 球谐函数最大阶数（原 4，提升角度分辨率）
        'sigma':    0.5,      # 高斯展宽（原 0.1，减少过拟合）
        'rbf':      'gto',    # 径向基类型：'gto' 或 'polynomial'
        'periodic': False,    # 是否周期性边界条件
    },

    # ---- ACSF 超参（已优化默认值）----
    'acsf': {
        'r_cut':     6.0,
        'g2_params': [        # G2 对称函数参数 [eta, Rs]（已扩充）
            [0.5, 0], [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 1], [2, 2], [2, 3], [4, 2], [4, 4],
        ],
        'g4_params': [        # G4 对称函数参数 [eta, zeta, lambda]（已扩充）
            [1, 1,  1], [1, 1, -1], [1, 2,  1], [1, 2, -1],
            [2, 1,  1], [2, 1, -1], [2, 2,  1], [2, 2, -1],
        ],
        'periodic': False,
    },

    # ---- MBTR 超参 ----
    'mbtr': {
        'geometry':     {'function': 'inverse_distance'},  # 几何函数
        'grid':         {'min': 0, 'max': 1, 'n': 100, 'sigma': 0.01},
        'weighting':    {'function': 'exp', 'scale': 0.5, 'threshold': 1e-3},
        'periodic':     False,
        'normalization': 'l2',
    },

    # ---- Mordred 选项 ----
    'mordred_ignore_3D': True,   # True = 只算 2D 描述符（速度快，推荐）

    # ---- 分子属性（calc_prop = True 时有效）----
    # 自动计算以下 11 个 RDKit 属性：
    # MolWt, HeavyAtomCount, NumHAcceptors, NumHDonors,
    # MolLogP, TPSA, NumRotatableBonds, NumAromaticRings,
    # NumRings, FractionCSP3, NumHeteroatoms
}
```

### 描述符类型详解

#### 1. RDKit 描述符（`calc_rdkit`）

- **内容**：RDKit 物化描述符（约 200 维） + RDKit 路径指纹（2048 bit）
- **特点**：速度快，覆盖面广；路径指纹捕捉分子拓扑路径
- **npy 字段**：`rdkit_descriptor`（数值列表）、`rdkit_f`（描述符名称列表）

#### 2. MACCS 结构键（`calc_maccs`）

- **内容**：167-bit 二进制结构键，每位对应一个预定义子结构
- **特点**：维度固定，解释性强，适合官能团层面相似度计算
- **npy 字段**：`maccs_descriptor`

#### 3. Morgan 圆形指纹（`calc_morgan`）

- **ECFP**（`morgan_use_features=False`）：基于原子标识符（原子序数 + 手性等）
- **FCFP**（`morgan_use_features=True`）：基于药效团特征（供/受体、芳香性等），更抽象
- **位向量**（`morgan_use_counts=False`）：每位 0/1，指示子结构是否存在
- **计数向量**（`morgan_use_counts=True`）：每位计数，值>1 表示重复子结构出现次数
- **npy 字段**：`morgan_descriptor`

#### 4. AtomPair 指纹（`calc_atompair`）

- **内容**：编码所有原子对的原子类型 + 拓扑距离，捕捉分子形状和原子间距关系
- **特点**：对分子拓扑距离敏感；`atompair_nbits` 控制哈希位数
- **npy 字段**：`atompair_descriptor`

#### 5. Topological Torsion 指纹（`calc_torsion`）

- **内容**：编码四个连续重原子的路径信息（原子类型 + 支化度 + 键类型）
- **特点**：捕捉四原子拓扑扭转模式，对分子骨架形状敏感
- **npy 字段**：`torsion_descriptor`

#### 6. Avalon 指纹（`calc_avalon`）

- **内容**：基于 Avalon Chemoinformatics Toolkit 的结构指纹（2048 bit）
- **特点**：与 Daylight 指纹类似，基于分子图路径枚举；常用于专利检索领域
- **依赖**：需要 RDKit 中的 `rdkit.Avalon` 模块（conda 安装的 RDKit 默认包含）
- **npy 字段**：`avalon_descriptor`

#### 7. SOAP 描述符（`calc_soap`）

- **内容**：光滑重叠原子势（Smooth Overlap of Atomic Positions），基于原子局部环境的连续对称不变描述符
- **特点**：对分子几何结构高度敏感，常用于量子化学性质预测；需要 xyz 坐标
- **优化参数**：`n_max=8`（精度提升）、`l_max=6`（角度分辨率提升）、`sigma=0.5`（减少过平滑）
- **npy 字段**：`soap_descriptor`（原子 SOAP 向量的平均值）

#### 8. ACSF 描述符（`calc_acsf`）

- **内容**：原子中心对称函数（Atom-Centered Symmetry Functions），用 G2/G4 对称函数描述局部原子环境
- **特点**：可微分，适合深度学习势能面拟合；需要 xyz 坐标
- **优化参数**：G2 参数扩充至 10 组，G4 参数扩充至 8 组
- **npy 字段**：`acsf_descriptor`（原子 ACSF 向量的平均值）

#### 9. MBTR 描述符（`calc_mbtr`）

- **内容**：多体张量表示（Many-Body Tensor Representation），同时编码 1/2/3 体相互作用
- **特点**：包含完整的分子几何信息（键长、键角、扭转），全局描述符（不依赖原子顺序）
- **依赖**：需要 `dscribe`（`pip install dscribe`）和 `ase`（`pip install ase`）
- **npy 字段**：`mbtr_descriptor`

#### 10. Mordred 描述符（`calc_mordred`）

- **内容**：1800+ 个分子描述符，涵盖拓扑、几何、电子、物化属性
- **特点**：维度极高，信息全面；`mordred_ignore_3D=True` 只算 2D（约 1600 维，速度快）
- **npy 字段**：`mordred_descriptor`

#### 11. 分子属性（`calc_prop`）

- **内容**：11 个直接由 RDKit 计算的基本分子属性，无需额外依赖
- **属性列表**：

| 属性 | 说明 |
|------|------|
| `MolWt` | 分子量（g/mol） |
| `HeavyAtomCount` | 重原子数 |
| `NumHAcceptors` | 氢键受体数 |
| `NumHDonors` | 氢键供体数 |
| `MolLogP` | 辛醇/水分配系数（亲脂性） |
| `TPSA` | 拓扑极性表面积（Å²） |
| `NumRotatableBonds` | 可旋转键数 |
| `NumAromaticRings` | 芳香环数 |
| `NumRings` | 总环数 |
| `FractionCSP3` | sp3 碳原子分数 |
| `NumHeteroatoms` | 杂原子数 |

- **npy 字段**：`prop_descriptor`

### 描述符选择建议

| 场景 | 推荐描述符组合 |
|------|--------------|
| 快速基线 | `rdkit` + `maccs` + `prop` |
| QSAR/药物发现 | `morgan`（ECFP4）+ `maccs` + `prop` |
| 骨架多样性分析 | `morgan`（ECFP4）+ `atompair` |
| 官能团相似度 | `morgan`（FCFP4）+ `maccs` |
| 重复子结构检测 | `morgan_use_counts=True` |
| 量子化学性质预测 | `soap` + `acsf`（需 xyz 坐标） |
| 全特征融合 | `rdkit` + `maccs` + `morgan` + `atompair` + `prop` |

### 依赖工具

| 工具 | 描述符类型 | 安装方式 |
|------|-----------|---------|
| RDKit | rdkit / MACCS / Morgan / AtomPair / Torsion / Avalon / prop | `conda install -c conda-forge rdkit` |
| Open Babel (`obabel`) | xyz → SMILES 转换 | `conda install -c conda-forge openbabel` |
| dscribe | SOAP / ACSF / MBTR | `pip install dscribe` |
| ase | 原子结构对象（SOAP/ACSF/MBTR 依赖） | `pip install ase` |
| mordred | Mordred 描述符 | `pip install mordred` |

### 运行

```bash
cd 3descriptor/
python create_by_fp.py
```

### 输出 `.npy` 新增字段

计算完成后每个分子 dict 新增以下字段（视开关而定）：

```python
{
    ...原有字段...,
    'rdkit_descriptor':     [...],   # RDKit 物化描述符 + 路径指纹（~2200 维）
    'rdkit_f':              [...],   # 对应描述符名称列表
    'maccs_descriptor':     [...],   # 167-bit MACCS 指纹
    'morgan_descriptor':    [...],   # Morgan 指纹（位或计数向量）
    'atompair_descriptor':  [...],   # AtomPair 指纹（2048 bit）
    'torsion_descriptor':   [...],   # Topological Torsion 指纹（2048 bit）
    'avalon_descriptor':    [...],   # Avalon 指纹（2048 bit）
    'soap_descriptor':      [...],   # SOAP 向量（原子平均）
    'acsf_descriptor':      [...],   # ACSF 向量（原子平均）
    'mbtr_descriptor':      [...],   # MBTR 多体张量向量
    'mordred_descriptor':   [...],   # Mordred 全描述符值（1600~1800 维）
    'prop_descriptor':      [...],   # 11 个基本分子属性
}
```

### 失败处理

- 每个失败分子的原因记录到 `failed_record` 指定的 Excel 文件。
- 失败分子**不会**进入输出 npy，保证输出数据均为有效数据。
- 失败类型分为三类：
  - `xyz_conversion`：obabel xyz→SMILES 转换失败
  - `mol_creation`：SMILES → RDKit Mol 对象创建失败
  - `descriptor_calculation`：描述符计算失败（如 MBTR/SOAP 依赖缺失）

---

## add_extra_de.py

将单个 Excel 文件中的数值列注入到 `.npy`，存为 `extra_d` 字段。

### 参数说明

```python
xlsx_file = 'my_qc_data.xlsx'   # 数值 Excel 文件
de_list   = [1,2,3,4,5]         # 要读取的列索引（0-based）
in_npy    = 'input.npy'         # 输入 npy
out_npy   = 'output-qc.npy'     # 输出 npy
```

### Excel 格式

- 第一行：表头（列名会保存为 `name_of_extra`）
- 第 `0` 列：分子名（与 npy 中 `name` 字段对应）
- `de_list` 指定列：数值

### 注入后新增字段

```python
{
    'extra_d':       [val1, val2, ...],          # 数值列表
    'name_of_extra': ['col_A', 'col_B', ...]     # 列名列表
}
```

---

## add_extra_de-multi.py

与 `add_extra_de.py` 功能相同，但支持从**多个 Excel 分表**合并读取（分子分散在不同文件中）。

### 参数说明

```python
EXCEL_FILES = ['file1.xlsx', 'file2.xlsx', ...]  # 分表路径列表
IN_NPY      = 'input.npy'
OUT_NPY     = 'output-qc.npy'
DE_COLS     = list(range(1, 34))   # 要读取的列索引
```

---

## 典型工作流

```
1dataProcess/dataset.npy
         ↓
3descriptor/create_by_fp.py  →  dataset-fp.npy  （选择所需描述符类型）
         ↓
3descriptor/add_extra_de.py  →  dataset-fp-qc.npy  （可选：额外 QC 描述符）
         ↓
4machineLearing/ml-m-full.py  （特征开关在 config 文件中控制）
```

## 描述符维度参考

| 描述符类型 | 典型维度 | 说明 |
|-----------|---------|------|
| rdkit | ~2200 | 物化值 ~200 + 路径指纹 2048 |
| maccs | 167 | 固定 167 位 |
| morgan（位向量） | 2048 | 可通过 `morgan_nbits` 调整 |
| morgan（计数向量） | 2048 | 值 > 1 表示计数 |
| atompair | 2048 | 可通过 `atompair_nbits` 调整 |
| torsion | 2048 | 可通过 `torsion_nbits` 调整 |
| avalon | 2048 | 可通过 `avalon_nbits` 调整 |
| soap | 与元素数/n_max/l_max 有关 | 典型值 3000~10000 |
| acsf | 与元素数/G2+G4 参数数有关 | 典型值 500~2000 |
| mbtr | 与 grid['n'] 和元素数有关 | 典型值 2000~5000 |
| mordred | ~1600（2D）/ ~1800（全） | 高维，含 NaN 需清洗 |
| prop | 11 | 固定 11 个属性 |
