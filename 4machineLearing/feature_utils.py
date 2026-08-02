"""
feature_utils.py  —— 训练脚本与预测脚本共用的特征工程工具

导入方式（两个脚本同目录）：
    from feature_utils import load_npy, extract_features, build_header, clean_features,
                             apply_dim_reduction, fit_dim_reduction, DimReducer,
                             PathManager, npy_save, npy_load

新增功能：
  - PathManager  : 全项目统一路径管理类
  - npy_save     : 统一 .npy 保存接口（自动创建目录）
  - npy_load     : 统一 .npy 加载接口（兼容旧/新格式）
  - TASK_MODE    : 全局任务类型标记（'regression' | 'classification'）
"""

import os
import joblib
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# 路径管理器（PathManager）
# 统一管理项目中所有数据/输出文件的路径，
# 支持从 molearn.yaml 的 paths: 节读取，也可直接构造
# ══════════════════════════════════════════════════════════════════════

class PathManager:
    """
    统一路径管理：提供各阶段输入/输出文件的完整路径。

    用法：
        pm = PathManager(base_dir='/path/to/project')
        # 或从 molearn.yaml 中的 paths: 节初始化
        pm = PathManager.from_yaml_paths(yaml_paths_dict, base_dir)

        # 路径访问（自动创建目录）
        pm.descriptor_npy('dataset-fp.npy')
        pm.training_output('seed_42/models/RandomForest.joblib')
        pm.prediction_output('result.csv')
    """

    # 路径键 → 默认子目录名
    _DEFAULTS = {
        'raw_gjf':          'data/raw/gjf',
        'raw_xyz':          'data/raw/xyz',
        'processed_npy':    'data/processed',
        'descriptor_npy':   'data/descriptors',
        'samples_dir':      'data/samples',
        'splits_dir':       'data/splits',
        'training_output':  'outputs/training',
        'analysis_output':  'outputs/analysis',
        'sa_score_output':  'outputs/sa_scores',
        'similarity_output':'outputs/similarity',
        'prediction_output':'outputs/predictions',
    }

    def __init__(self, base_dir: str = '.', paths_cfg: dict = None):
        """
        参数
        ----
        base_dir  : 项目根目录（所有相对路径基于此解析）
        paths_cfg : 路径配置字典（对应 molearn.yaml paths: 节），
                    缺失的键使用 _DEFAULTS 补全
        """
        self.base_dir = os.path.abspath(base_dir)
        self._cfg     = dict(self._DEFAULTS)
        if paths_cfg:
            self._cfg.update(paths_cfg)

    @classmethod
    def from_yaml_paths(cls, yaml_paths: dict, base_dir: str = '.') -> 'PathManager':
        """从 molearn.yaml paths: 节构造 PathManager。"""
        return cls(base_dir=base_dir, paths_cfg=yaml_paths)

    def _resolve(self, key: str, filename: str = '') -> str:
        """返回 key 对应目录下的完整路径，自动创建目录。"""
        rel_dir = self._cfg.get(key, self._DEFAULTS.get(key, key))
        if os.path.isabs(rel_dir):
            full_dir = rel_dir
        else:
            full_dir = os.path.join(self.base_dir, rel_dir)
        os.makedirs(full_dir, exist_ok=True)
        return os.path.join(full_dir, filename) if filename else full_dir

    # ── 常用路径快捷方法 ──────────────────────────────────────────────
    def raw_gjf(self,       f: str = '') -> str: return self._resolve('raw_gjf',          f)
    def raw_xyz(self,       f: str = '') -> str: return self._resolve('raw_xyz',           f)
    def processed_npy(self, f: str = '') -> str: return self._resolve('processed_npy',    f)
    def descriptor_npy(self,f: str = '') -> str: return self._resolve('descriptor_npy',   f)
    def samples_dir(self,   f: str = '') -> str: return self._resolve('samples_dir',      f)
    def splits_dir(self,    f: str = '') -> str: return self._resolve('splits_dir',       f)
    def training_output(self,f: str= '') -> str: return self._resolve('training_output',  f)
    def analysis_output(self,f: str= '') -> str: return self._resolve('analysis_output',  f)
    def sa_score_output(self,f: str= '') -> str: return self._resolve('sa_score_output',  f)
    def similarity_output(self,f:str='')-> str: return self._resolve('similarity_output', f)
    def prediction_output(self,f:str='')-> str: return self._resolve('prediction_output', f)

    def get(self, key: str, filename: str = '') -> str:
        """通用方法：按键名获取路径。"""
        return self._resolve(key, filename)

    def __repr__(self):
        lines = [f"PathManager(base_dir={self.base_dir})"]
        for k, v in self._cfg.items():
            lines.append(f"  {k}: {v}")
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════
# 统一 .npy 存取接口
# ══════════════════════════════════════════════════════════════════════

def npy_save(path: str, data, extra_meta: dict = None):
    """
    保存 .npy 文件，自动创建父目录。

    data 可以是：
      - dict（已有 'successful' 等键）  → 直接保存
      - list（分子列表）                → 包装为 {'successful': data}
      - np.ndarray                       → 直接保存

    extra_meta : 追加到 dict 中的元数据（如 pearson_filter 统计信息）
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(data, list):
        save_obj = {'successful': data}
    elif isinstance(data, dict):
        save_obj = data
    else:
        np.save(path, data, allow_pickle=True)
        return
    if extra_meta:
        save_obj.update(extra_meta)
    np.save(path, save_obj, allow_pickle=True)


def npy_load(path: str) -> list:
    """
    加载 .npy 文件，返回分子列表。
    兼容：dict with 'successful'  /  object-dtype ndarray
    """
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, dict) and 'successful' in raw:
        return list(raw['successful'])
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        return list(raw)
    raise ValueError(
        f"[npy_load] {path} 格式无法识别。"
        "期望：dict with 'successful' key，或 object-dtype ndarray。"
    )


# ──────────────────────────────────────────────
# 描述符键名映射表（npy 字段名  →  列名前缀）
# 若将来新增描述符类型，只需在此表追加一行即可。
# ──────────────────────────────────────────────
_DESCRIPTOR_MAP = [
    # (if_flag_key, npy_field,          col_prefix)
    ('if_rdkit',     'rdkit_descriptor',   'rdkit'),
    ('if_soap',      'soap_descriptor',    'soap'),
    ('if_acsf',      'acsf_descriptor',    'acsf'),
    ('if_mordred',   'mordred_descriptor', 'mordred'),
    ('if_maccs',     'maccs_descriptor',   'maccs'),
    ('if_morgan',    'morgan_descriptor',  'morgan'),
    ('if_atompair',  'atompair_descriptor','AP'),
    ('if_torsion',   'torsion_descriptor', 'TT'),
    ('if_avalon',    'avalon_descriptor',  'Avalon'),
    ('if_mbtr',      'mbtr_descriptor',    'MBTR'),
    ('if_prop',      'prop_descriptor',    'prop'),
    ('if_QC',        'g_d',                'QC'),
    ('if_extra',     'extra_d',            'extra'),   # 列名特殊处理，见 build_header
    ('if_m',         '3DMatrix',           'm'),        # 需要展平，见 extract_features
]

# 标准特征开关键列表（顺序与 _DESCRIPTOR_MAP 一致）
FEATURE_FLAG_KEYS = [row[0] for row in _DESCRIPTOR_MAP]


# ──────────────────────────────────────────────
# 1. 加载单个 .npy 文件，返回分子列表
# ──────────────────────────────────────────────
def load_npy(path: str) -> list:
    """
    加载 create_by_fp.py 生成的 .npy 文件（npy_load 的别名，保持向后兼容）。
    支持两种格式：
      - dict with key 'successful'  （create_by_fp.py 输出）
      - object-dtype ndarray         （旧格式）
    """
    return npy_load(path)


# ──────────────────────────────────────────────
# 2. 从单个分子 dict 提取特征向量
# ──────────────────────────────────────────────
def _extract_one(d: dict, flags: dict) -> list:
    """
    flags: {flag_key: bool/int, ...}  例如 {'if_rdkit': 1, 'if_soap': 0, ...}
    返回一个 list[float]，顺序与 _DESCRIPTOR_MAP 一致。

    特殊处理：
      - '3DMatrix'      : 每行展平后拼接
      - 'prop_descriptor': create_by_fp.py 将基础属性存为独立字段（dict key），
                          此处优先读取 'prop_descriptor' 向量；若不存在，回退到
                          各属性字段（MolWt/MolLogP 等）收集
    """
    ft = []
    for flag_key, npy_field, _ in _DESCRIPTOR_MAP:
        if not flags.get(flag_key, 0):
            continue
        if npy_field == '3DMatrix':
            for row in d.get('3DMatrix', []):
                ft.extend(row)
        elif npy_field == 'prop_descriptor':
            # 优先读取预存向量
            if 'prop_descriptor' in d and d['prop_descriptor']:
                ft.extend(d['prop_descriptor'])
            else:
                # 回退：从个别属性字段收集（兼容旧版 create_by_fp.py）
                _PROP_KEYS = ['MolWt', 'HeavyAtomCount', 'NumHAcceptors',
                              'NumHDonors', 'MolLogP', 'TPSA',
                              'NumRotatableBonds', 'NumAromaticRings',
                              'NumRings', 'FractionCSP3', 'NumHeteroatoms']
                ft.extend([float(d.get(k, float('nan'))) for k in _PROP_KEYS])
        else:
            ft.extend(d.get(npy_field, []))
    return ft


# ──────────────────────────────────────────────
# 3. 批量提取特征（多 npy 文件横向拼接）
# ──────────────────────────────────────────────
def extract_features(datas: list, flags_per_file: dict) -> tuple:
    """
    参数
    ----
    datas          : list of list-of-dicts，每个元素对应一个 npy 文件的分子列表
    flags_per_file : {flag_key: [int, ...]}，每个 flag 对每个文件的开关值列表

    返回
    ----
    features : np.ndarray  shape (n_samples, n_features)
    labels   : np.ndarray  shape (n_samples,)
    """
    n_files = len(datas)
    n_samples = len(datas[0])

    # 按文件提取，xs[file_idx][sample_idx] = feature list
    xs = []
    labels = None
    for idx, data in enumerate(datas):
        flags = {k: flags_per_file[k][idx] for k in FEATURE_FLAG_KEYS}
        x_tmp, y_tmp = [], []
        for d in tqdm(data, desc=f"提取特征 [{idx+1}/{n_files}]", leave=False):
            x_tmp.append(_extract_one(d, flags))
            y_tmp.append(d.get('y', np.nan))
        xs.append(x_tmp)
        if labels is None:
            labels = y_tmp

    # 横向拼接：每个样本把所有文件的特征合并
    combined = [
        [item for j in range(n_files) for item in xs[j][i]]
        for i in range(n_samples)
    ]
    features = np.array(combined, dtype=np.float64)
    labels   = np.array(labels,   dtype=np.float64)
    return features, labels


# ──────────────────────────────────────────────
# 4. 生成列名（与 extract_features 保持相同顺序）
# ──────────────────────────────────────────────
def build_header(datas: list, flags_per_file: dict) -> list:
    """
    利用每个文件第 0 个样本的字段长度来生成列名。
    extra 描述符使用 name_of_extra 中存储的真实名称。
    mordred/rdkit 等已存有列名的字段使用已有名称（更易读）。
    """
    _PROP_KEYS = ['MolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
                  'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
                  'NumRings', 'FractionCSP3', 'NumHeteroatoms']

    header = []
    n_files = len(datas)
    for idx, data in enumerate(datas):
        d0    = data[0]
        flags = {k: flags_per_file[k][idx] for k in FEATURE_FLAG_KEYS}
        for flag_key, npy_field, prefix in _DESCRIPTOR_MAP:
            if not flags.get(flag_key, 0):
                continue

            if npy_field == 'extra_d':
                names = d0.get('name_of_extra', [])
                header += [f"{idx}_{n}" for n in names]

            elif npy_field == '3DMatrix':
                flat_len = sum(len(row) for row in d0.get('3DMatrix', []))
                header += [f"m_{idx}_{i}" for i in range(flat_len)]

            elif npy_field == 'prop_descriptor':
                # 优先读向量长度；回退到预定义的属性名
                if 'prop_descriptor' in d0 and d0['prop_descriptor']:
                    length = len(d0['prop_descriptor'])
                    header += [f"{prefix}_{idx}_{i}" for i in range(length)]
                else:
                    header += [f"prop_{idx}_{k}" for k in _PROP_KEYS]

            else:
                # 使用已存的列名字段（如 rdkit_f / mordred_f / morgan_f 等）
                names_field = npy_field.replace('_descriptor', '_f')
                if names_field in d0 and d0[names_field]:
                    header += [f"{prefix}_{idx}_{n}" for n in d0[names_field]]
                else:
                    length = len(d0.get(npy_field, []))
                    header += [f"{prefix}_{idx}_{i}" for i in range(length)]
    return header


# ──────────────────────────────────────────────
# 5. 数据清洗：裁剪极值 → NaN → 列均值填充
# ──────────────────────────────────────────────
def clean_features(feat: np.ndarray) -> np.ndarray:
    """
    原地安全版：不修改输入数组。
    步骤：① 裁剪 ±1e30  ② inf→NaN  ③ 列均值填充 NaN
    """
    if feat.size == 0:
        raise RuntimeError("[clean_features] 特征矩阵为空（0 列），请检查特征开关配置。")
    feat = feat.copy()
    feat = np.clip(feat, -1e30, 1e30)
    feat[~np.isfinite(feat)] = np.nan
    col_means = np.nanmean(feat, axis=0)
    col_means[np.isnan(col_means)] = 0.0          # 全 NaN 的列用 0 填
    nan_mask = np.isnan(feat)
    feat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return feat


# ──────────────────────────────────────────────
# 6. 解析 config-full-*.txt 配置文件
# ──────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """
    解析 key: value 格式的配置文件，忽略空行和 # 注释行。
    """
    cfg = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or ':' not in line:
                continue
            key, val = line.split(':', 1)
            cfg[key.strip()] = val.strip()
    return cfg


def parse_flags(config: dict, n_files: int) -> dict:
    """
    从 config dict 中解析所有 if_xxx 开关，统一广播到长度 n_files 的列表。
    支持 str（"1,0"格式）、int/bool、list 类型的 config 值。
    返回 {flag_key: [int, ...]}
    """
    result = {}
    for key in FEATURE_FLAG_KEYS:
        raw = config.get(key, 0)
        # 支持多种输入类型
        if isinstance(raw, (list, tuple)):
            lst = [int(v) for v in raw]
        elif isinstance(raw, str):
            lst = [int(x.strip()) for x in raw.split(',')]
        else:
            lst = [int(raw)]
        if len(lst) == 1:
            lst = lst * n_files
        if len(lst) != n_files:
            raise ValueError(
                f"[parse_flags] '{key}' 长度 ({len(lst)}) 与文件数 ({n_files}) 不一致"
            )
        result[key] = lst
    return result


# ──────────────────────────────────────────────
# 7. 解析脚本相对 / 绝对路径的通用工具
# ──────────────────────────────────────────────
def resolve_path(path: str, base_dir: str) -> str:
    """
    若 path 是绝对路径直接返回；否则先尝试 base_dir/path，找不到再返回原 path（cwd 相对）。
    """
    if os.path.isabs(path):
        return path
    candidate = os.path.join(base_dir, path)
    return candidate if os.path.isfile(candidate) else path


# ══════════════════════════════════════════════════════════════
# 8. 降维工具（DimReducer）
#    支持：PCA / KernelPCA / TruncatedSVD / UMAP / t-SNE / Autoencoder
#    训练时：fit_dim_reduction(X_train, cfg) → (reducer, X_reduced)
#    推理时：apply_dim_reduction(X, reducer_path) → X_reduced
# ══════════════════════════════════════════════════════════════

# 默认降维配置 —— 可在训练脚本配置区覆盖
DIM_REDUCTION_DEFAULT = {
    'method':        'pca',   # 'none'|'pca'|'kpca'|'tsvd'|'umap'|'tsne'|'autoencoder'
    'n_components':  50,      # 目标维数（autoencoder 忽略此项，用 ae_dims）
    'whiten':        False,   # 仅 PCA：是否白化
    'kpca_kernel':   'rbf',   # 仅 KernelPCA：核函数
    'variance_ratio': None,   # PCA/TSVD：用方差解释率自动决定维数（0~1，None 表示用 n_components）
    # UMAP 额外参数
    'umap_n_neighbors': 15,
    'umap_min_dist':    0.1,
    'umap_metric':      'euclidean',
    'umap_random_state': 42,
    # t-SNE 额外参数（注意：t-SNE 无法用于推理新样本，不建议用于训练流）
    'tsne_perplexity':  30,
    'tsne_n_iter':      1000,
    'tsne_random_state': 42,
    # Autoencoder 额外参数
    'ae_dims':       [256, 128, 64],  # 编码器各层节点数；解码器对称
    'ae_epochs':     50,
    'ae_batch_size': 64,
    'ae_lr':         1e-3,
    'ae_activation': 'relu',
}


class _SklearnReducer:
    """包装 sklearn/umap 的 fit/transform 接口，统一 DimReducer 调用格式。"""
    def __init__(self, estimator, method: str):
        self.estimator = estimator
        self.method    = method

    def fit_transform(self, X):
        return self.estimator.fit_transform(X)

    def transform(self, X):
        if self.method == 'tsne':
            raise RuntimeError("t-SNE 不支持 transform 新样本，请换用 pca/umap 等方法。")
        return self.estimator.transform(X)


class _AutoencoderReducer:
    """基于 PyTorch 的轻量 Autoencoder 降维器（可选依赖）。"""

    def __init__(self, input_dim: int, cfg: dict):
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn    = nn
        except ImportError:
            raise ImportError("Autoencoder 降维需要 PyTorch（pip install torch）")

        self.cfg       = cfg
        self.input_dim = input_dim
        dims           = cfg.get('ae_dims', [256, 128, 64])
        act_map        = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        act            = act_map.get(cfg.get('ae_activation', 'relu'), nn.ReLU)

        # 编码器
        enc_layers = []
        prev = input_dim
        for d in dims:
            enc_layers += [nn.Linear(prev, d), act()]
            prev = d
        self.encoder = nn.Sequential(*enc_layers)

        # 解码器（对称）
        dec_layers = []
        for d in reversed(dims[:-1]):
            dec_layers += [nn.Linear(prev, d), act()]
            prev = d
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._model_trained = False

    def _get_model(self):
        import torch.nn as nn
        class AE(nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.encoder = enc
                self.decoder = dec
            def forward(self, x):
                return self.decoder(self.encoder(x))
        return AE(self.encoder, self.decoder)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.cfg.get('ae_batch_size', 64),
                            shuffle=True)
        model  = self._get_model()
        opt    = torch.optim.Adam(model.parameters(), lr=self.cfg.get('ae_lr', 1e-3))
        loss_fn = nn.MSELoss()

        epochs = self.cfg.get('ae_epochs', 50)
        for ep in range(epochs):
            total = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item()
            if (ep + 1) % 10 == 0:
                print(f"    [AE] epoch {ep+1}/{epochs}  loss={total/len(loader):.4f}")

        self.encoder = model.encoder
        self.decoder = model.decoder
        self._model_trained = True

        with torch.no_grad():
            return self.encoder(X_t).numpy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch
        if not self._model_trained:
            raise RuntimeError("Autoencoder 尚未训练，请先调用 fit_transform。")
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.encoder(X_t).numpy()


class DimReducer:
    """
    统一降维接口。

    用法（训练阶段）：
        reducer = DimReducer(cfg=DIM_REDUCTION_CFG)
        X_reduced = reducer.fit_transform(X_train)
        reducer.save('results/seed_42/dim_reducer.pkl')

    用法（推理阶段）：
        reducer = DimReducer.load('results/seed_42/dim_reducer.pkl')
        X_reduced = reducer.transform(X_new)
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg     = {**DIM_REDUCTION_DEFAULT, **(cfg or {})}
        self.method  = self.cfg['method'].lower()
        self._inner  = None  # 实际的降维器实例

    # ── 构建 ──────────────────────────────────────────────────────────────────
    def _build(self, input_dim: int):
        m   = self.method
        cfg = self.cfg
        n   = cfg.get('n_components', 50)
        vr  = cfg.get('variance_ratio')

        if m == 'none':
            return None

        if m == 'pca':
            from sklearn.decomposition import PCA
            nc = (vr if vr else n)
            return _SklearnReducer(PCA(n_components=nc, whiten=cfg.get('whiten', False),
                                       random_state=42), 'pca')

        if m == 'kpca':
            from sklearn.decomposition import KernelPCA
            return _SklearnReducer(KernelPCA(n_components=n,
                                              kernel=cfg.get('kpca_kernel', 'rbf'),
                                              random_state=42, fit_inverse_transform=False), 'kpca')

        if m == 'tsvd':
            from sklearn.decomposition import TruncatedSVD
            nc = (vr if vr else n)
            return _SklearnReducer(TruncatedSVD(n_components=min(n, input_dim - 1),
                                                 random_state=42), 'tsvd')

        if m == 'umap':
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError("UMAP 降维需要 umap-learn（pip install umap-learn）")
            return _SklearnReducer(UMAP(n_components=n,
                                        n_neighbors=cfg.get('umap_n_neighbors', 15),
                                        min_dist=cfg.get('umap_min_dist', 0.1),
                                        metric=cfg.get('umap_metric', 'euclidean'),
                                        random_state=cfg.get('umap_random_state', 42)), 'umap')

        if m == 'tsne':
            from sklearn.manifold import TSNE
            print("[WARN] t-SNE 不支持 transform 新样本，仅建议用于可视化，不建议用于训练流！")
            return _SklearnReducer(TSNE(n_components=min(n, 3),
                                        perplexity=cfg.get('tsne_perplexity', 30),
                                        n_iter=cfg.get('tsne_n_iter', 1000),
                                        random_state=cfg.get('tsne_random_state', 42)), 'tsne')

        if m == 'autoencoder':
            return _AutoencoderReducer(input_dim, cfg)

        raise ValueError(f"[DimReducer] 不支持的降维方法: '{m}'，"
                         "可选: none/pca/kpca/tsvd/umap/tsne/autoencoder")

    # ── 训练 & 变换 ────────────────────────────────────────────────────────────
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在训练集上拟合并降维，返回降维后特征矩阵。"""
        if self.method == 'none':
            print("[DimReducer] 降维方法=none，跳过降维")
            return X
        self._inner = self._build(X.shape[1])
        print(f"[DimReducer] method={self.method}  输入维度={X.shape[1]}")
        X_r = self._inner.fit_transform(X)
        # 若使用方差比例，打印实际保留维数
        if self.method == 'pca' and hasattr(self._inner.estimator, 'n_components_'):
            print(f"[DimReducer] PCA 方差解释率={self.cfg['variance_ratio']}  "
                  f"保留主成分={self._inner.estimator.n_components_}")
        print(f"[DimReducer] 输出维度={X_r.shape[1]}")
        return X_r

    def transform(self, X: np.ndarray) -> np.ndarray:
        """用已拟合的降维器变换新样本。"""
        if self.method == 'none' or self._inner is None:
            return X
        return self._inner.transform(X)

    # ── 持久化 ────────────────────────────────────────────────────────────────
    def save(self, path: str):
        """保存降维器到 .pkl 文件（使用 joblib）。"""
        joblib.dump(self, path)
        print(f"[DimReducer] 已保存到: {path}")

    @staticmethod
    def load(path: str) -> 'DimReducer':
        """从 .pkl 文件加载降维器。"""
        obj = joblib.load(path)
        print(f"[DimReducer] 已加载自: {path}")
        return obj


def fit_dim_reduction(X_train: np.ndarray, cfg: dict) -> tuple:
    """
    便捷函数：拟合并降维训练集。
    返回 (DimReducer 实例, X_train_reduced)
    """
    reducer = DimReducer(cfg=cfg)
    X_r     = reducer.fit_transform(X_train)
    return reducer, X_r


def apply_dim_reduction(X: np.ndarray, reducer_or_path) -> np.ndarray:
    """
    便捷函数：用已有 DimReducer 或其路径对 X 做变换。
    reducer_or_path: DimReducer 实例 或 str(路径)
    """
    if isinstance(reducer_or_path, str):
        reducer = DimReducer.load(reducer_or_path)
    else:
        reducer = reducer_or_path
    return reducer.transform(X)
