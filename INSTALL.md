# 环境安装指南

本文档说明如何从零开始配置运行本项目所需的完整 Python 环境。

---

## 系统要求

| 项目 | 最低要求 |
|------|---------|
| 操作系统 | Linux（推荐 Ubuntu 20.04+）/ macOS 12+ / Windows 10（WSL2 推荐） |
| Python | 3.9 ~ 3.11（推荐 3.10） |
| 内存 | 8 GB+（大数据集建议 32 GB+） |
| 磁盘 | 10 GB+（含 conda 环境） |

---

## 第一步：安装 Miniconda

强烈推荐使用 conda 管理环境，因为 RDKit 和 Open Babel 通过 conda 安装最可靠。

```bash
# Linux / macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示完成安装，重启终端后生效

# 验证安装
conda --version
```

> Windows 用户请下载 [Miniconda Windows 安装包](https://docs.conda.io/en/latest/miniconda.html)，或在 WSL2 中按 Linux 方式安装。

---

## 第二步：创建并激活虚拟环境

```bash
conda create -n molearn python=3.10 -y
conda activate molearn
```

---

## 第三步：安装核心依赖

### 3.1 通过 conda 安装（必须先于 pip 安装）

```bash
# RDKit（化学信息学库，必选）
conda install -c conda-forge rdkit -y

# Open Babel（xyz → SMILES 转换，3descriptor 需要）
conda install -c conda-forge openbabel -y
```

### 3.2 通过 pip 安装 Python 包

```bash
pip install numpy pandas scikit-learn matplotlib joblib tqdm openpyxl shap scipy
```

**各包用途说明：**

| 包名 | 版本建议 | 用途 |
|------|---------|------|
| `numpy` | ≥ 1.24 | 数组运算 |
| `pandas` | ≥ 2.0 | 数据处理、Excel 读写 |
| `scikit-learn` | ≥ 1.3 | 机器学习模型（含 HistGBR 等新模型）、降维（PCA/KernelPCA/TruncatedSVD）、分层采样 |
| `matplotlib` | ≥ 3.7 | 绘图 |
| `joblib` | ≥ 1.3 | 模型保存/加载、降维器序列化 |
| `tqdm` | ≥ 4.65 | 进度条 |
| `openpyxl` | ≥ 3.1 | Excel 读写（`.xlsx`） |
| `shap` | ≥ 0.44 | SHAP 特征重要性分析 |
| `scipy` | ≥ 1.7 | LHS 抽样（`stats.qmc.LatinHypercube`）、统计检验（QQ 图） |

### 3.3 安装可选依赖

根据使用的功能选择安装：

#### 3.3.1 描述符相关

```bash
# SOAP / ACSF / MBTR 描述符（3descriptor/create_by_fp.py 使用）
pip install dscribe ase

# Mordred 描述符（大量化学描述符，3descriptor/create_by_fp.py 使用）
pip install mordred
```

#### 3.3.2 降维相关

```bash
# UMAP 降维（4machineLearing/feature_utils.py，DimReducer method='umap'）
pip install umap-learn

# PyTorch（Autoencoder 降维，DimReducer method='autoencoder'）
# CPU 版（无 GPU 时）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU 版（CUDA 12.1）：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 3.3.3 超参数优化

```bash
# Optuna（4machineLearing/ml-m-full.py HPO_METHOD='optuna' 时需要）
pip install optuna
```

#### 3.3.4 可合成性打分

```bash
# SYBA（2databaseAnalysis/sa_score.py CALC_SYBA=True 时需要）
pip install syba

# SCScore（精确版，未安装时自动切换内置近似版）
pip install scscore

# RAscore（2databaseAnalysis/sa_score.py CALC_RASCORE=True 时需要）
# 注意：RAscore 需要额外模型权重文件，请参阅官方仓库
pip install rascore
# 或从 GitHub 安装：
# pip install git+https://github.com/reymond-group/RAscore.git
```

#### 3.3.5 数据库分析可视化增强

```bash
# Seaborn（db_analysis.py 可选，有则使用更美观的可视化样式）
pip install seaborn
```

---

## 功能–依赖矩阵

| 功能模块 | 必选包 | 可选包（功能增强） |
|---------|-------|-----------------|
| 数据预处理（1dataProcess）| numpy, tqdm, openpyxl | scipy（LHS 抽样）|
| 数据抽样（sample_npy.py）| numpy, scikit-learn, rdkit | scipy（LHS 必须）|
| 数据库分析（db_analysis.py）| numpy, pandas, matplotlib, scikit-learn, rdkit | seaborn, umap-learn, scipy |
| 可合成性打分（sa_score.py）| numpy, pandas, matplotlib, rdkit | scscore, syba, rascore |
| 相似度搜索（similarity_search.py）| numpy, pandas, matplotlib, rdkit | — |
| 描述符计算（create_by_fp.py）| rdkit, openbabel | dscribe, ase, mordred |
| 模型训练（ml-m-full.py）| numpy, pandas, scikit-learn, matplotlib, joblib, shap, tqdm | optuna, umap-learn, torch |
| 降维（feature_utils.py DimReducer）| scikit-learn, joblib | umap-learn, torch |
| 批量训练（batch_train.py）| scikit-learn, numpy | — |
| 数据集划分（dataset_split.py）| scikit-learn, numpy, rdkit | — |
| 模型推理（usemodel.py）| numpy, pandas, scikit-learn, joblib | umap-learn（若训练用了 UMAP）|

---

## 第四步：验证安装

运行以下命令，所有导入均无报错则安装成功：

```bash
python -c "
import numpy, pandas, sklearn, matplotlib, joblib, shap, tqdm, openpyxl, scipy
from rdkit import Chem
print('核心依赖:', numpy.__version__, sklearn.__version__)
print('RDKit: OK')
print('scipy:', scipy.__version__)
print('所有核心依赖安装成功！')
"
```

验证可选依赖：

```bash
# 描述符
python -c "import dscribe; print('dscribe OK')"
python -c "import mordred; print('mordred OK')"

# 降维
python -c "import umap; print('umap-learn OK')"
python -c "import torch; print('torch OK, version:', torch.__version__)"

# 超参数优化
python -c "import optuna; print('optuna OK')"

# 可合成性打分
python -c "import syba; print('syba OK')"
python -c "import scscore; print('scscore OK')"
python -c "import rascore; print('rascore OK')"

# 可视化增强
python -c "import seaborn; print('seaborn OK')"
```

验证 Open Babel（命令行工具）：

```bash
obabel --version
```

---

## 第五步：克隆项目

```bash
git clone https://github.com/sysu-yinwuli/molearn.git
cd molearn
```

---

## 完整一键安装脚本

将以下内容保存为 `setup_env.sh`，运行 `bash setup_env.sh`：

```bash
#!/bin/bash
set -e

ENV_NAME="molearn"

echo "=== 创建 conda 环境 ==="
conda create -n $ENV_NAME python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== 安装 conda 依赖 ==="
conda install -c conda-forge rdkit openbabel -y

echo "=== 安装核心 pip 依赖 ==="
pip install numpy pandas "scikit-learn>=1.3" matplotlib joblib tqdm openpyxl shap scipy

echo "=== 安装可选依赖（描述符）==="
pip install dscribe ase mordred

echo "=== 安装可选依赖（降维）==="
pip install umap-learn
# torch CPU 版（如有 GPU 请按需修改）
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== 安装可选依赖（超参/打分/可视化）==="
pip install optuna seaborn
# pip install scscore syba rascore  # 可合成性打分（按需）

echo "=== 验证核心依赖 ==="
python -c "
import numpy, pandas, sklearn, matplotlib, joblib, shap, tqdm, openpyxl, scipy
from rdkit import Chem
print('核心依赖全部 OK')
print('sklearn 版本:', sklearn.__version__)
print('scipy 版本:', scipy.__version__)
"

echo "=== 安装完成！激活环境: conda activate $ENV_NAME ==="
```

---

## 常见问题

### Q1：RDKit 安装后 `import rdkit` 报错

**原因**：pip 安装的 rdkit 版本与系统不兼容。

**解决**：必须使用 conda 安装：

```bash
conda install -c conda-forge rdkit -y
```

### Q2：`obabel` 命令找不到

**原因**：Open Babel 的命令行工具路径未加入 PATH。

**解决**：

```bash
conda install -c conda-forge openbabel -y
# 重新激活环境
conda deactivate && conda activate molearn
obabel --version
```

### Q3：`dscribe` 安装失败（编译报错）

**原因**：缺少 C++ 编译器或 libint2 库。

**解决（Linux）：**

```bash
sudo apt-get install build-essential libffi-dev
pip install dscribe
```

**解决（macOS）：**

```bash
xcode-select --install
pip install dscribe
```

### Q4：SHAP 安装成功但 TreeExplainer 报错

**原因**：shap 版本过旧（< 0.40）。

**解决**：

```bash
pip install --upgrade shap
```

### Q5：`scikit-learn` 版本过低，`HistGradientBoostingRegressor` 功能缺失

**解决**：

```bash
pip install "scikit-learn>=1.3"
```

### Q6：Windows 下 Open Babel 无法通过 conda 安装

**推荐方案**：使用 WSL2（Windows Subsystem for Linux），在其中按 Linux 方式安装。

**备选方案**：从 [Open Babel 官网](http://openbabel.org/wiki/Category:Installation) 下载 Windows 安装包，安装后将安装目录加入系统 PATH。

### Q7：`umap-learn` 安装后 `import umap` 报错

**原因**：`umap-learn` 包的 import 名是 `umap`，不是 `umap_learn`。

**解决**：

```bash
pip install umap-learn
python -c "import umap; print('OK')"
```

### Q8：`sa_score.py` 提示 SA Score 找不到

**原因**：`SA_Score.py` 位于 RDKit Contrib 目录，路径因安装方式不同而异。

**解决**：脚本会自动搜索多个路径（conda 环境、系统路径、rdkit 安装目录），通常无需手动干预。若仍失败：

```bash
# 找到 SA_Score.py 位置
find $(conda info --base) -name "SA_Score.py" 2>/dev/null
# 将路径写入环境变量
export RDBASE=/path/to/rdkit
```

### Q9：`scscore` 安装后模型权重文件缺失

**原因**：SCScore 需要预训练的模型权重文件（`.h5` 格式）。

**解决**：`sa_score.py` 在 `scscore` 库异常时会自动切换到内置近似版本，精度略低但无需额外文件。若需精确版本，请参阅 [SCScore 官方仓库](https://github.com/connorcoley/scscore)。

### Q10：`rascore` 安装失败

**原因**：RAscore 依赖 TensorFlow，版本兼容性要求较严格。

**解决**：

```bash
# 建议使用专用 conda 环境测试
conda create -n rascore_env python=3.8 -y
conda activate rascore_env
pip install tensorflow==2.6.0
pip install rascore
```

或直接将 `CALC_RASCORE = False` 跳过该方法。

---

## 环境导出与迁移

```bash
# 导出当前环境（保存给他人复现）
conda activate molearn
conda env export > environment.yml

# 从 environment.yml 重建环境
conda env create -f environment.yml
conda activate molearn
```

---

## 版本参考

以下是经过测试的依赖版本组合：

```
python       = 3.10.14
rdkit        = 2024.03.3
numpy        = 1.26.4
pandas       = 2.2.2
scikit-learn = 1.5.1
matplotlib   = 3.9.1
joblib       = 1.4.2
shap         = 0.45.1
tqdm         = 4.66.4
openpyxl     = 3.1.5
scipy        = 1.13.1
dscribe      = 2.1.1
ase          = 3.23.0
mordred      = 1.2.0
optuna       = 3.6.1
umap-learn   = 0.5.6
torch        = 2.3.0 (CPU)
seaborn      = 0.13.2
```
