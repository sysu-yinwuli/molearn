# 5modelApplication — 模型推理（新数据预测）

本目录使用训练好的模型对新分子数据集进行预测，并将结果输出为 CSV 文件。支持自动加载降维器（`dim_reducer.pkl`），与训练流程完全对应。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `usemodel.py` | 模型推理脚本：加载模型 → 提取特征 → （可选降维）→ 预测 → 输出 CSV |
| `config-full.txt` | 参考配置文件（特征开关，需与训练时一致） |

---

## 前提条件

在运行推理前，必须先完成以下步骤：

1. **`1dataProcess`**：生成新数据的 `.npy` 文件（使用 `create_npy-dir.py` 或 `create_npy.py`）
2. **`3descriptor`**：对新数据的 `.npy` 计算描述符（使用 `create_by_fp.py`，**开关须与训练时相同**）
3. **`4machineLearing`**：完成模型训练，确认 `seed_xxx/` 目录下存在以下文件：
   - `models/<ModelName>.joblib`
   - `training_columns.pkl`
   - `y_scaler.pkl`
   - `dim_reducer.pkl`（**如训练时启用了降维，此文件必须存在**）

---

## usemodel.py 使用说明

### 配置区域（只改这里）

```python
PREDICT_NPY = 'hjf-add-fp.npy'                       # 待预测 npy（已计算描述符）
OUTPUT_CSV  = 'wavelength-add-pred.csv'               # 输出 CSV 文件路径
MODEL_DIR   = '../4machineLearing/results/seed_42'    # 训练输出目录（包含 models/ 子目录）
MODEL_NAME  = 'GradientBoosting'                      # 使用的模型名（不含 .joblib）
CONFIG_TXT  = '../4machineLearing/config-full-1.txt'  # 训练时使用的配置文件

# ---- 降维设置 ----
USE_DIM_REDUCTION = True   # True = 自动检测并加载 dim_reducer.pkl
                           # False = 跳过降维（即使 dim_reducer.pkl 存在）
```

### 降维自动检测逻辑

当 `USE_DIM_REDUCTION = True` 时，脚本会在 `MODEL_DIR` 中查找 `dim_reducer.pkl`：

```
results/
└── seed_42/              ← MODEL_DIR 指向这里
    ├── models/
    │   ├── GradientBoosting.joblib
    │   └── RandomForest.joblib
    ├── training_columns.pkl
    ├── y_scaler.pkl
    └── dim_reducer.pkl    ← 自动检测到时，推理前会自动应用降维变换
```

| 场景 | 行为 |
|------|------|
| `USE_DIM_REDUCTION=True` 且 `dim_reducer.pkl` 存在 | 自动加载并应用降维变换 |
| `USE_DIM_REDUCTION=True` 且 `dim_reducer.pkl` **不存在** | 打印提示，跳过降维，直接用原始特征预测 |
| `USE_DIM_REDUCTION=False` | 始终跳过降维，即使文件存在 |

> **重要**：如果训练时使用了降维（`DIM_REDUCTION_CFG['method'] != 'none'`），推理时**必须**启用 `USE_DIM_REDUCTION=True` 且确保 `dim_reducer.pkl` 存在，否则特征维度不匹配会报错。

**`MODEL_DIR` 与 `MODEL_NAME` 对应关系：**

```
results/
└── seed_42/              ← MODEL_DIR 指向这里
    ├── models/
    │   ├── GradientBoosting.joblib   ← MODEL_NAME = 'GradientBoosting'
    │   └── RandomForest.joblib
    ├── training_columns.pkl
    ├── y_scaler.pkl
    └── dim_reducer.pkl               ← 降维器（训练时保存，推理时自动加载）
```

### 运行

```bash
cd 5modelApplication/
python usemodel.py
```

### 关键注意事项

1. **配置文件必须与训练时完全一致**：`CONFIG_TXT` 中的 `if_xxx` 特征开关决定了从 npy 中提取哪些特征，必须与训练时完全一样，否则特征维度不匹配会报错。

2. **npy 必须已计算对应的描述符**：如果训练时用了 `if_rdkit: 1`，待预测 npy 也必须含有 `rdkit_descriptor` 字段。

3. **降维器与模型绑定**：`dim_reducer.pkl` 中保存的是已经在训练集上 `fit` 过的降维器，不可用其他降维器替换。

4. **`y_true` 列可以是 NaN**：若新数据没有真实标签（纯预测场景），npy 中 `y` 字段为 `1`（`create_npy-dir.py` 的默认值），输出 CSV 中 `y_true` 列会显示占位值，预测结果在 `y_pred` 列。

5. **t-SNE 降维无法用于推理**：t-SNE 没有 `transform` 方法（无法对新样本降维），若训练时使用了 `tsne` 降维，推理时会报错并提示改用 PCA/UMAP 等支持 `transform` 的方法。

### 输出 CSV 格式

```csv
sample_name,y_true,y_pred
mol_001,3.120000,3.087234
mol_002,NaN,4.512789
mol_003,2.890000,2.934156
```

| 列名 | 说明 |
|------|------|
| `sample_name` | 分子名（来自 npy 中的 `name` 字段） |
| `y_true` | 真实值（npy 中的 `y` 字段，无标签时为占位值） |
| `y_pred` | 模型预测值（已逆归一化至原始标签空间） |

---

## 错误排查

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `FileNotFoundError: model 文件不存在` | `MODEL_DIR` 或 `MODEL_NAME` 配置错误 | 检查路径，确认 `.joblib` 文件存在 |
| `FileNotFoundError: y_scaler 文件不存在` | 训练脚本版本过旧（旧版不保存 y_scaler） | 重新运行训练脚本 |
| `ValueError: 特征维度不匹配` | 预测 npy 的描述符开关与训练时不一致，或训练用了降维但推理未加载 `dim_reducer.pkl` | 统一 `CONFIG_TXT` 特征开关；设置 `USE_DIM_REDUCTION=True` |
| `KeyError: rdkit_descriptor` | 预测 npy 未计算该描述符 | 先运行 `3descriptor/create_by_fp.py` |
| `RuntimeError: t-SNE 无 transform 方法` | 训练时使用了 t-SNE 降维，推理阶段无法变换新样本 | 改用 `pca`/`umap`/`kpca`/`tsvd`/`autoencoder` 等支持 transform 的方法 |
| `ModuleNotFoundError: No module named 'umap'` | 训练使用了 UMAP，但推理环境未安装 | `pip install umap-learn` |

---

## 推理流程说明

```
待预测 npy（已有描述符）
         ↓
读取 CONFIG_TXT（特征开关）
         ↓
提取原始特征矩阵 X
         ↓
（若 USE_DIM_REDUCTION=True 且 dim_reducer.pkl 存在）
    加载 DimReducer → reducer.transform(X)
         ↓
加载 training_columns.pkl（特征列对齐）
         ↓
加载 y_scaler.pkl（逆变换预测值）
         ↓
加载模型（MODEL_NAME.joblib）→ model.predict(X)
         ↓
逆归一化 → y_pred
         ↓
输出 OUTPUT_CSV
```

---

## config-full.txt 格式参考

此文件需与训练时的配置文件**完全一致**（特征开关部分）。

```ini
npy_path: ../3descriptor/Osc-fp.npy
res_folder: res/Osc-maccs
if_rdkit:    1
if_maccs:    0
if_morgan:   0
if_atompair: 0
if_torsion:  0
if_avalon:   0
if_soap:     0
if_acsf:     0
if_mbtr:     0
if_mordred:  0
if_prop:     1
if_QC:       0
if_extra:    0
if_m:        0
seed: 42
```

> 新版训练脚本（Round 5+）新增 `if_atompair`、`if_torsion`、`if_avalon`、`if_mbtr`、`if_prop` 选项，推理时配置文件需同步更新。
