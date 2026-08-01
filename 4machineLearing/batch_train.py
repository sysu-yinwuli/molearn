#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_train.py — 批量模型训练脚本（功能 6）
================================================================
支持三种批量训练模式（可同时开启）：
  模式 A：对多个 npy 文件分别训练（每个 npy → 独立结果目录）
  模式 B：对同一数据集批量扫描超参数空间（网格 / 随机 / Optuna）
  模式 C：对多个 npy 文件 × 多种模型 × 多组超参数全排列

计算资源控制（Windows / Linux 均支持）：
  - CPU 核数限制（sklearn n_jobs）
  - 进程优先级（nice/ionice）
  - 可选内存上限
  - 可选并发进程数（multiprocessing Pool）

所有参数在顶部 CONFIG 区域集中配置。
"""

# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

# ---------- 模式 A：多 npy 批量训练 ----------
MODE_A_ENABLE   = True
MODE_A_NPY_LIST = [
    # 每个元素是一个 npy 文件路径（相对或绝对）
    'data1.npy',
    'data2.npy',
]
# 可选：每个 npy 对应的配置文件（None=使用默认配置）
MODE_A_CONFIGS  = None   # None 表示全部用 MODE_A_DEFAULT_CONFIG
MODE_A_DEFAULT_CONFIG = 'config-full-1.txt'

# ---------- 模式 B：超参数批量扫描 ----------
MODE_B_ENABLE   = False
MODE_B_NPY      = 'data.npy'          # 使用的数据集
MODE_B_CONFIG   = 'config-full-1.txt'
MODE_B_MODEL    = 'RandomForest'       # 要扫描的模型名称
# 超参数网格（key = 完整 pipeline 参数名，如 model__n_estimators）
MODE_B_PARAM_GRID = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth':    [3, 5, 8, None],
    'model__max_features': ['sqrt', 'log2', 0.5],
}
MODE_B_SCAN_METHOD = 'grid'    # 'grid'（全网格）| 'random'（随机采样）
MODE_B_N_SAMPLES   = 20        # 仅 random 模式：采样组数
MODE_B_CV          = 5         # 交叉验证折数
MODE_B_SCORING     = 'r2'      # 评分指标

# ---------- 通用训练参数 ----------
COMMON_MODELS_ENABLE = {
    'RandomForest':     True,
    'GradientBoosting': True,
    'SVR':              False,
    'Ridge':            True,
    'MLP':              False,
    # 其余模型默认关闭（批量时减少时间）
}
COMMON_TEST_SIZE = 0.2
COMMON_SEEDS     = [42]        # 数据划分种子列表

# ---------- 计算资源控制 ----------
RESOURCE_N_JOBS       = -1     # sklearn 并行核数：-1=全核，1=单核，N=N核
RESOURCE_MAX_CORES    = None   # 整体 CPU 核数上限（None=不限）
                               # 若设置，将 pin 进程到前 N 个核（Linux: taskset，Windows: affinity）
RESOURCE_PROCESS_PRIORITY = 'normal'  # 'low' | 'normal' | 'high'
                                       # low: nice 19 / IDLE_PRIORITY_CLASS
                                       # high: nice -10 / HIGH_PRIORITY_CLASS
RESOURCE_PARALLEL_JOBS = 1     # 批量任务的进程并发数（>1 需要多进程支持）
                               # 建议 Windows 设为 1，Linux 可根据核数设置
# 内存限制（仅 Linux）
RESOURCE_MAX_MEM_GB   = None   # 设置后若超过将被系统 kill（None=不限）

# ---------- 输出 ----------
BATCH_OUTPUT_BASE  = 'batch_results'   # 批量结果根目录
BATCH_SUMMARY_CSV  = 'batch_summary.csv'

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os, sys, copy, csv, time, datetime, platform, itertools, warnings
import numpy as np
import pandas as pd
import joblib
import traceback
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sklearn.linear_model import (LinearRegression, Ridge, LassoCV, ElasticNetCV,
                                   HuberRegressor, BayesianRidge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor,
                               BaggingRegressor, HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, ParameterGrid, ParameterSampler)
from sklearn.pipeline import Pipeline

from feature_utils import (load_npy, extract_features, build_header,
                            clean_features, load_config, parse_flags, resolve_path)

# ── 资源控制 ──────────────────────────────────────────────────────────────────

def _set_process_priority():
    pri = RESOURCE_PROCESS_PRIORITY.lower()
    plat = platform.system()
    try:
        if plat == 'Windows':
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            CLASS_MAP = {
                'low':    0x00000040,  # IDLE_PRIORITY_CLASS
                'normal': 0x00000020,  # NORMAL_PRIORITY_CLASS
                'high':   0x00000080,  # HIGH_PRIORITY_CLASS
            }
            ctypes.windll.kernel32.SetPriorityClass(handle, CLASS_MAP.get(pri, 0x00000020))
        else:  # Linux / macOS
            nice_map = {'low': 19, 'normal': 0, 'high': -10}
            os.nice(nice_map.get(pri, 0))
        print(f"[资源] 进程优先级设置为: {pri}")
    except Exception as e:
        print(f"[资源] 优先级设置失败（可忽略）: {e}")


def _set_cpu_affinity():
    if RESOURCE_MAX_CORES is None:
        return
    plat = platform.system()
    try:
        if plat == 'Windows':
            import ctypes
            mask = (1 << RESOURCE_MAX_CORES) - 1
            ctypes.windll.kernel32.SetProcessAffinityMask(
                ctypes.windll.kernel32.GetCurrentProcess(), mask)
        else:
            cores = list(range(RESOURCE_MAX_CORES))
            pid   = os.getpid()
            os.sched_setaffinity(pid, cores)
        print(f"[资源] CPU 亲和性限制到前 {RESOURCE_MAX_CORES} 个核")
    except Exception as e:
        print(f"[资源] CPU 亲和性设置失败: {e}")


def _set_mem_limit():
    if RESOURCE_MAX_MEM_GB is None or platform.system() == 'Windows':
        return
    try:
        import resource
        limit = int(RESOURCE_MAX_MEM_GB * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print(f"[资源] 内存限制: {RESOURCE_MAX_MEM_GB} GB")
    except Exception as e:
        print(f"[资源] 内存限制设置失败: {e}")


# ── 决定并行 n_jobs ────────────────────────────────────────────────────────────
def _get_n_jobs():
    n = RESOURCE_N_JOBS
    if RESOURCE_MAX_CORES is not None:
        import multiprocessing
        avail = multiprocessing.cpu_count()
        cap   = min(RESOURCE_MAX_CORES, avail)
        if n < 0 or n > cap:
            n = cap
    return n


# ── 模型定义 ──────────────────────────────────────────────────────────────────
_N_JOBS = _get_n_jobs()

_BASE_MODELS = {
    'LinearRegression':    Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge':               Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
    'LassoCV':             Pipeline([('scaler', StandardScaler()), ('model', LassoCV())]),
    'ElasticNetCV':        Pipeline([('scaler', StandardScaler()), ('model', ElasticNetCV())]),
    'HuberRegressor':      Pipeline([('scaler', StandardScaler()), ('model', HuberRegressor())]),
    'BayesianRidge':       Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())]),
    'DecisionTree':        Pipeline([('model', DecisionTreeRegressor())]),
    'RandomForest':        Pipeline([('model', RandomForestRegressor(n_jobs=_N_JOBS))]),
    'ExtraTreesRegressor': Pipeline([('model', ExtraTreesRegressor(n_jobs=_N_JOBS))]),
    'GradientBoosting':    Pipeline([('model', GradientBoostingRegressor())]),
    'HistGBR':             Pipeline([('model', HistGradientBoostingRegressor())]),
    'AdaBoost':            Pipeline([('model', AdaBoostRegressor())]),
    'BaggingRegressor':    Pipeline([('model', BaggingRegressor(n_jobs=_N_JOBS))]),
    'SVR':                 Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
    'KNeighbors':          Pipeline([('scaler', StandardScaler()),
                                     ('model', KNeighborsRegressor(n_jobs=_N_JOBS))]),
    'MLP':                 Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor())]),
}

_DEFAULT_PARAMS = {
    'RandomForest':        {'model__n_estimators': 100, 'model__max_depth': 5,
                            'model__random_state': 42},
    'GradientBoosting':    {'model__n_estimators': 100, 'model__learning_rate': 0.1,
                            'model__max_depth': 3, 'model__random_state': 42},
    'ExtraTreesRegressor': {'model__n_estimators': 100, 'model__max_depth': 5,
                            'model__random_state': 42},
    'Ridge':               {'model__alpha': 10.0},
    'SVR':                 {'model__kernel': 'rbf', 'model__C': 100,
                            'model__gamma': 'scale'},
    'MLP':                 {'model__hidden_layer_sizes': (100,), 'model__activation': 'tanh',
                            'model__solver': 'adam', 'model__max_iter': 1000,
                            'model__early_stopping': True, 'model__random_state': 42},
    'DecisionTree':        {'model__max_depth': 5, 'model__random_state': 42},
    'HistGBR':             {'model__max_iter': 200, 'model__learning_rate': 0.1,
                            'model__max_depth': 5, 'model__random_state': 42},
    'AdaBoost':            {'model__n_estimators': 100, 'model__learning_rate': 0.1,
                            'model__random_state': 42},
    'BaggingRegressor':    {'model__n_estimators': 20, 'model__random_state': 42},
    'LassoCV':             {'model__cv': 5, 'model__max_iter': 10000,
                            'model__random_state': 42},
    'ElasticNetCV':        {'model__cv': 5, 'model__l1_ratio': 0.5,
                            'model__max_iter': 10000, 'model__random_state': 42},
    'HuberRegressor':      {'model__epsilon': 1.35, 'model__alpha': 1e-4,
                            'model__max_iter': 300},
    'BayesianRidge':       {'model__max_iter': 300},
    'KNeighbors':          {'model__n_neighbors': 5},
}


def _make_pipe(name: str, extra_params: dict = None) -> Pipeline:
    pipe   = copy.deepcopy(_BASE_MODELS[name])
    params = {**_DEFAULT_PARAMS.get(name, {}), **(extra_params or {})}
    if params:
        pipe.set_params(**params)
    return pipe


# ── 核心训练函数 ──────────────────────────────────────────────────────────────

def _load_data(npy_path, config_path):
    """加载 npy + config，返回 (X, y, header)。"""
    config  = load_config(resolve_path(config_path, _HERE))
    npy_lst = [p.strip() for p in config['npy_path'].split(',')]
    datas   = [load_npy(resolve_path(p, _HERE)) for p in npy_lst]
    n_files = len(datas)
    flags   = parse_flags(config, n_files)
    feat, labels = extract_features(datas, flags)
    feat    = clean_features(feat)
    header  = build_header(datas, flags)
    X = pd.DataFrame(feat, columns=header)
    y = pd.Series(labels)
    return X, y


def _train_one(name, pipe, X_tr, y_tr_s, X_te, y_te, y_scaler, seed, extra_info=''):
    """训练一个模型，返回 metrics dict。"""
    t0 = time.time()
    try:
        pipe.fit(X_tr, y_tr_s)
        y_pred = y_scaler.inverse_transform(
            pipe.predict(X_te).reshape(-1, 1)).flatten()
        r2  = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        mse = mean_squared_error(y_te, y_pred)
        elapsed = time.time() - t0
        return {'model': name, 'r2': r2, 'mae': mae, 'mse': mse,
                'time_s': elapsed, 'extra': extra_info, 'status': 'ok'}
    except Exception as e:
        elapsed = time.time() - t0
        return {'model': name, 'r2': np.nan, 'mae': np.nan, 'mse': np.nan,
                'time_s': elapsed, 'extra': extra_info, 'status': f'ERROR: {e}'}


def _run_single_npy(npy_path, config_path, out_subdir, seeds):
    """对一个 npy 文件运行标准多模型训练。"""
    os.makedirs(out_subdir, exist_ok=True)
    print(f"\n  npy: {npy_path}  config: {config_path}")

    try:
        X, y = _load_data(npy_path, config_path)
    except Exception as e:
        print(f"  [ERROR] 加载数据失败: {e}")
        return []

    print(f"  特征矩阵: {X.shape}   有效标签: {(~y.isna()).sum()}")
    records = []

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=COMMON_TEST_SIZE, random_state=seed)
        y_scaler  = StandardScaler()
        y_tr_s    = y_scaler.fit_transform(y_tr.values.reshape(-1, 1)).flatten()

        for name, enabled in COMMON_MODELS_ENABLE.items():
            if not enabled or name not in _BASE_MODELS:
                continue
            print(f"    [{name}] seed={seed} ...", end=' ', flush=True)
            pipe = _make_pipe(name)
            rec  = _train_one(name, pipe, X_tr, y_tr_s, X_te, y_te.values,
                               y_scaler, seed,
                               extra_info=f'npy={os.path.basename(npy_path)}')
            rec.update({'npy': os.path.basename(npy_path), 'seed': seed})
            records.append(rec)
            print(f"R²={rec['r2']:.4f}  MAE={rec['mae']:.4f}  [{rec['status']}]")

            # 保存模型
            if rec['status'] == 'ok':
                model_dir = os.path.join(out_subdir, f'seed_{seed}', 'models')
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(pipe, os.path.join(model_dir, f'{name}.joblib'))

    return records


def _run_param_scan(npy_path, config_path, model_name, param_grid_or_list,
                    out_subdir, cv, scoring):
    """超参数批量扫描。"""
    os.makedirs(out_subdir, exist_ok=True)
    print(f"\n  [超参扫描] model={model_name}  npy={npy_path}")

    try:
        X, y = _load_data(npy_path, config_path)
    except Exception as e:
        print(f"  [ERROR] 加载数据失败: {e}")
        return []

    seed = COMMON_SEEDS[0]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=COMMON_TEST_SIZE, random_state=seed)
    y_scaler  = StandardScaler()
    y_tr_s    = y_scaler.fit_transform(y_tr.values.reshape(-1, 1)).flatten()

    records = []
    for i, params in enumerate(param_grid_or_list):
        label = f"combo_{i:04d}"
        print(f"    [{i+1}/{len(param_grid_or_list)}] {params} ...", end=' ', flush=True)
        pipe = _make_pipe(model_name, params)
        # 交叉验证得分
        try:
            cv_scores = cross_val_score(pipe, X_tr, y_tr_s, cv=cv,
                                         scoring=scoring, n_jobs=_N_JOBS)
            cv_mean   = float(cv_scores.mean())
            cv_std    = float(cv_scores.std())
        except Exception as e:
            cv_mean, cv_std = np.nan, np.nan

        # 测试集得分
        rec = _train_one(model_name, _make_pipe(model_name, params),
                          X_tr, y_tr_s, X_te, y_te.values, y_scaler, seed,
                          extra_info=str(params))
        rec.update({'cv_r2_mean': cv_mean, 'cv_r2_std': cv_std,
                    'combo': label, 'params': str(params)})
        records.append(rec)
        print(f"CV_R²={cv_mean:.4f}±{cv_std:.4f}  Test_R²={rec['r2']:.4f}")

        # 保存最优模型（按 cv_mean 判断）
    if records:
        best = max([r for r in records if not np.isnan(r.get('cv_r2_mean', np.nan))],
                   key=lambda r: r['cv_r2_mean'], default=None)
        if best:
            best_pipe = _make_pipe(model_name,
                                   eval(best['params']) if isinstance(best['params'], str) else {})
            try:
                best_pipe.fit(X_tr, y_tr_s)
                joblib.dump(best_pipe, os.path.join(out_subdir, f'{model_name}_best.joblib'))
                print(f"\n  最优参数: {best['params']}")
                print(f"  最优 CV_R²={best['cv_r2_mean']:.4f}  Test_R²={best['r2']:.4f}")
            except Exception:
                pass
    return records


# =============================================================================
# 主程序
# =============================================================================
print("=" * 65)
print("  批量训练脚本")
print("=" * 65)
print(f"  日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  平台: {platform.system()} {platform.release()}")
print(f"  n_jobs: {_N_JOBS}")

# 应用资源限制
_set_process_priority()
_set_cpu_affinity()
_set_mem_limit()

os.makedirs(BATCH_OUTPUT_BASE, exist_ok=True)
all_records = []

# ── 模式 A：多 npy 文件批量训练 ──────────────────────────────────────────────
if MODE_A_ENABLE:
    print(f"\n{'─'*65}")
    print("  [模式 A] 多 npy 批量训练")
    print(f"{'─'*65}")
    for i, npy_p in enumerate(MODE_A_NPY_LIST):
        cfg_p  = (MODE_A_CONFIGS[i] if MODE_A_CONFIGS and i < len(MODE_A_CONFIGS)
                  else MODE_A_DEFAULT_CONFIG)
        tag    = os.path.splitext(os.path.basename(npy_p))[0]
        subdir = os.path.join(BATCH_OUTPUT_BASE, f'A_{tag}')
        recs   = _run_single_npy(npy_p, cfg_p, subdir, COMMON_SEEDS)
        for r in recs:
            r['batch_mode'] = 'A'
        all_records.extend(recs)

# ── 模式 B：超参数批量扫描 ───────────────────────────────────────────────────
if MODE_B_ENABLE:
    print(f"\n{'─'*65}")
    print("  [模式 B] 超参数批量扫描")
    print(f"{'─'*65}")

    if MODE_B_SCAN_METHOD == 'grid':
        param_list = list(ParameterGrid(MODE_B_PARAM_GRID))
        print(f"  网格法：共 {len(param_list)} 组参数组合")
    else:
        rng_b = np.random.RandomState(COMMON_SEEDS[0])
        param_list = list(ParameterSampler(MODE_B_PARAM_GRID, n_iter=MODE_B_N_SAMPLES,
                                            random_state=rng_b))
        print(f"  随机法：采样 {len(param_list)} 组参数组合")

    subdir_b = os.path.join(BATCH_OUTPUT_BASE, f'B_{MODE_B_MODEL}')
    recs_b   = _run_param_scan(MODE_B_NPY, MODE_B_CONFIG, MODE_B_MODEL,
                                param_list, subdir_b, MODE_B_CV, MODE_B_SCORING)
    for r in recs_b:
        r['batch_mode'] = 'B'
    all_records.extend(recs_b)

# ── 输出汇总 CSV ──────────────────────────────────────────────────────────────
if all_records:
    summary_df = pd.DataFrame(all_records)
    csv_path   = os.path.join(BATCH_OUTPUT_BASE, BATCH_SUMMARY_CSV)
    summary_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n[汇总] 结果已保存: {csv_path}")
    print(f"  总任务数: {len(all_records)}")
    ok = summary_df[summary_df['status'] == 'ok']
    if len(ok) > 0:
        print(f"  成功任务: {len(ok)}")
        print(f"  最优 R²:  {ok['r2'].max():.4f}  ({ok.loc[ok['r2'].idxmax(), 'model']})")

print(f"\n{'='*65}")
print("  批量训练完成！")
print(f"  输出目录: {os.path.abspath(BATCH_OUTPUT_BASE)}/")
print(f"{'='*65}")
