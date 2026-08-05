#!/usr/bin/env python3
# =============================================================================
#  molearn_run.py  —  Molearn 项目总控脚本
#  支持：Linux / macOS / Windows (PowerShell / VSCode Terminal)
#
#  用法示例：
#    python molearn_run.py                        # 运行 molearn.yaml 中所有 enabled 步骤
#    python molearn_run.py --step 3               # 仅运行第 3 步（描述符计算）
#    python molearn_run.py --step 3,6             # 仅运行第 3、6 步
#    python molearn_run.py --step 3-6             # 运行第 3~6 步（含两端）
#    python molearn_run.py --config my.yaml       # 使用自定义配置文件
#    python molearn_run.py --list                 # 列出所有步骤状态
#    python molearn_run.py --dry-run              # 预演（打印命令但不执行）
#    python molearn_run.py --step 6 --seed 42,123 # 覆盖 yaml 中的 seeds 配置
# =============================================================================

import argparse
import os
import sys
import subprocess
import platform
import tempfile
import datetime
import textwrap

# ── 可选 YAML 解析 ─────────────────────────────────────────────────────────────
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ── 项目根目录（本脚本所在位置）──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 步骤定义 ───────────────────────────────────────────────────────────────────
STEPS = {
    1: dict(name="数据预处理",     key="step1_data",       script="1dataProcess/create_npy.py"),
    2: dict(name="数据库分析",     key="step2_analysis",   script="2databaseAnalysis/db_analysis.py"),
    3: dict(name="描述符计算",     key="step3_descriptor", script="3descriptor/create_by_fp.py"),
    4: dict(name="数据抽样",       key="step4_sampling",   script="1dataProcess/sample_npy.py"),
    5: dict(name="数据集划分",     key="step5_split",      script="4machineLearing/dataset_split.py"),
    6: dict(name="模型训练",       key="step6_train",      script="4machineLearing/ml-m-full.py"),
    7: dict(name="可合成性打分",   key="step7_sa_score",   script="2databaseAnalysis/sa_score.py"),
    8: dict(name="相似度搜索",     key="step8_similarity", script="2databaseAnalysis/similarity_search.py"),
    9: dict(name="模型推理",       key="step9_predict",    script="5modelApplication/usemodel.py"),
}

# ─────────────────────────────────────────────────────────────────────────────
# YAML / dict config 解析
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    """加载 YAML 配置文件，返回 dict。未安装 PyYAML 时用内置简易解析器。"""
    if _HAS_YAML:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    # ── 简易解析器（仅支持 key: value 和缩进节）──────────────────────────────
    return _simple_yaml_parse(path)


def _simple_yaml_parse(path: str) -> dict:
    """无 PyYAML 时的简易 YAML 解析（只处理本项目 molearn.yaml 的格式）。"""
    import re
    result = {}
    stack  = [(result, -1)]

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
            indent = len(line) - len(stripped)

            # 弹出缩进比当前大的节
            while len(stack) > 1 and stack[-1][1] >= indent:
                stack.pop()

            parent = stack[-1][0]

            if ':' not in stripped:
                continue
            key, _, rest = stripped.partition(':')
            key  = key.strip()
            rest = rest.strip()

            # 去掉行内注释
            rest = re.sub(r'\s+#.*$', '', rest).strip()

            if rest == '' or rest == '{}':
                child = {}
                parent[key] = child
                stack.append((child, indent))
            elif rest.startswith('[') and rest.endswith(']'):
                inner = rest[1:-1]
                items = [x.strip().strip('"').strip("'") for x in inner.split(',') if x.strip()]
                try:
                    parent[key] = [int(x) for x in items]
                except ValueError:
                    parent[key] = items
            elif rest.lower() in ('true', 'yes'):
                parent[key] = True
            elif rest.lower() in ('false', 'no'):
                parent[key] = False
            elif rest.lower() in ('null', 'none', '~'):
                parent[key] = None
            else:
                try:
                    parent[key] = int(rest)
                except ValueError:
                    try:
                        parent[key] = float(rest)
                    except ValueError:
                        parent[key] = rest.strip('"').strip("'")
    return result


def _get(cfg: dict, *keys, default=None):
    """安全深度取值，任意节点为 None 时返回 default。"""
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val is None:
            return default
    return val


# ─────────────────────────────────────────────────────────────────────────────
# 目录初始化
# ─────────────────────────────────────────────────────────────────────────────

def _init_dirs(cfg: dict, dry_run: bool = False):
    """按 paths 节点创建所有输出目录。"""
    paths = cfg.get('paths', {})
    dirs = list(paths.values())
    for d in dirs:
        full = os.path.join(_ROOT, d)
        if not dry_run:
            os.makedirs(full, exist_ok=True)
        else:
            print(f"  [dry] mkdir -p {full}")


# ─────────────────────────────────────────────────────────────────────────────
# 临时 config-full.txt 生成（供各子脚本使用）
# ─────────────────────────────────────────────────────────────────────────────

def _make_config_txt(cfg: dict, step_key: str, overrides: dict = None) -> str:
    """
    从 molearn.yaml 中提取训练配置，生成临时 config-full-{step}.txt。
    返回该临时文件的路径。
    """
    s6 = cfg.get('step6_train', {})
    paths = cfg.get('paths', {})
    overrides = overrides or {}

    npy_rel = _get(cfg, 'step6_train', 'input_npy') or ''
    if not npy_rel:
        # 与 _resolve_input_npy 保持一致：优先使用 pearson 过滤后的 npy
        _pearson_on = bool(_get(cfg, 'step3_descriptor', 'pearson_filter', default=False))
        if _pearson_on:
            desc_dir  = _get(cfg, 'paths', 'pearson_npy', default='data/pearson')
            base_name = _get(cfg, 'step3_descriptor', 'output_name', default='dataset-fp.npy')
            custom    = _get(cfg, 'step3_descriptor', 'pearson_output_npy', default='')
            if custom:
                desc_name = os.path.basename(custom)
            else:
                desc_name = base_name.replace('.npy', '_pearson.npy') if base_name.endswith('.npy') \
                            else base_name + '_pearson.npy'
        else:
            desc_dir  = _get(cfg, 'paths', 'descriptor_npy', default='data/descriptors')
            desc_name = _get(cfg, 'step3_descriptor', 'output_name', default='dataset-fp.npy')
        npy_rel   = os.path.join(desc_dir, desc_name)

    res_folder = os.path.join(
        _get(cfg, 'paths', 'training_output', default='outputs/training'),
        _get(cfg, 'project', 'name', default='molearn_project')
    )

    seeds_raw = overrides.get('seeds', s6.get('seeds', [42]))
    seeds_str = ','.join(str(s) for s in (seeds_raw if isinstance(seeds_raw, list) else [seeds_raw]))

    feats     = s6.get('features', {})
    task_type = overrides.get('task_type', s6.get('task_type', 'regression'))

    lines = [
        f"# Auto-generated by molearn_run.py  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"npy_path: {os.path.join(_ROOT, npy_rel)}",
        f"res_folder: {os.path.join(_ROOT, res_folder)}",
        f"seed: {seeds_str}",
        f"task_type: {task_type}",
    ]
    feat_keys = [
        'if_rdkit','if_maccs','if_morgan','if_atompair','if_torsion',
        'if_avalon','if_soap','if_acsf','if_mbtr','if_mordred',
        'if_prop','if_QC','if_extra','if_m',
    ]
    for k in feat_keys:
        lines.append(f"{k}: {feats.get(k, 0)}")

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', prefix='molearn_cfg_',
        dir=os.path.join(_ROOT, '4machineLearing'),
        delete=False, encoding='utf-8'
    )
    tmp.write('\n'.join(lines) + '\n')
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# 资源控制（优先级 / CPU 亲和性 / 内存）
# ─────────────────────────────────────────────────────────────────────────────

def _apply_resource_limits(cfg: dict):
    """在当前进程上应用 resources 节点中的设置（Linux/Windows）。"""
    res = cfg.get('resources', {})
    if not res:
        return

    sys_name = platform.system()

    # ── 进程优先级 ──────────────────────────────────────────────────────────
    priority = str(res.get('priority', 'normal')).lower()
    if sys_name == 'Linux' or sys_name == 'Darwin':
        nice_map = {'low': 19, 'normal': 0, 'high': -10}
        try:
            os.nice(nice_map.get(priority, 0))
        except PermissionError:
            pass
    elif sys_name == 'Windows':
        try:
            import ctypes
            handle  = ctypes.windll.kernel32.GetCurrentProcess()
            cls_map = {'low': 0x40, 'normal': 0x20, 'high': 0x80}
            ctypes.windll.kernel32.SetPriorityClass(handle, cls_map.get(priority, 0x20))
        except Exception:
            pass

    # ── CPU 亲和性 ──────────────────────────────────────────────────────────
    cpu_cores = int(res.get('cpu_cores', -1))
    if cpu_cores > 0:
        if sys_name == 'Linux':
            try:
                cores = set(range(cpu_cores))
                os.sched_setaffinity(0, cores)
            except Exception:
                pass
        elif sys_name == 'Windows':
            try:
                import ctypes
                mask = (1 << cpu_cores) - 1
                ctypes.windll.kernel32.SetProcessAffinityMask(
                    ctypes.windll.kernel32.GetCurrentProcess(), mask)
            except Exception:
                pass

    # ── 内存限制（仅 Linux）──────────────────────────────────────────────────
    mem_gb = float(res.get('mem_limit_gb', 0))
    if mem_gb > 0 and sys_name == 'Linux':
        try:
            import resource
            limit = int(mem_gb * 1024 ** 3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 步骤执行
# ─────────────────────────────────────────────────────────────────────────────

def _run_script(script_rel: str, extra_env: dict = None,
                dry_run: bool = False) -> int:
    """运行单个子脚本，返回退出码。"""
    script_path = os.path.join(_ROOT, script_rel)
    if not os.path.isfile(script_path):
        print(f"  [WARN] 脚本不存在，跳过: {script_path}")
        return 0

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    cmd = [sys.executable, script_path]
    print(f"  → {' '.join(cmd)}")
    if dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=_ROOT, env=env)
    return proc.returncode


def _patch_script_config(script_rel: str, cfg: dict, step_key: str,
                          overrides: dict, dry_run: bool) -> int:
    """
    对于需要 config 文件的脚本（步骤6），先生成临时 config txt，
    通过环境变量 MOLEARN_CONFIG 传递给子脚本。
    子脚本如使用 molearn_run 兼容模式，可读取此环境变量。
    """
    tmp_cfg = _make_config_txt(cfg, step_key, overrides)
    print(f"  [config] 临时配置文件: {tmp_cfg}")
    ret = _run_script(script_rel,
                      extra_env={'MOLEARN_CONFIG': tmp_cfg},
                      dry_run=dry_run)
    if not dry_run:
        try:
            os.unlink(tmp_cfg)
        except OSError:
            pass
    return ret


# ─────────────────────────────────────────────────────────────────────────────
# 自动路径推断（上一步输出 → 下一步输入）
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_input_npy(cfg: dict, step_key: str) -> str:
    """
    如果某步骤的 input_npy 为空，自动推断上一步的输出路径。
    优先级规则：
      - step3 : step1 处理输出 (data/processed/)
      - step4/5/6 : 若 step3 开启了 pearson_filter，则用 pearson 过滤后 npy
                    (data/pearson/)；否则用描述符 npy (data/descriptors/)
      - step2/7/8/9 : 与 step4/5/6 相同的优先级逻辑
    """
    s = cfg.get(step_key, {})
    # 步骤内显式指定的 input_npy 最优先
    npy = (s.get('input_npy', '')
           or s.get('query_npy', '')
           or s.get('predict_npy', ''))
    if npy:
        return os.path.join(_ROOT, npy)

    paths = cfg.get('paths', {})

    # ── step3: 使用 step1 的输出 ─────────────────────────────────────────
    if step_key == 'step3_descriptor':
        d = paths.get('processed_npy', 'data/processed')
        n = _get(cfg, 'step1_data', 'output_name', default='dataset.npy')
        return os.path.join(_ROOT, d, n)

    # ── step4/5/6/2/7/8/9: 优先 pearson 过滤后 npy ──────────────────────
    _pearson_on = bool(_get(cfg, 'step3_descriptor', 'pearson_filter', default=False))
    if _pearson_on:
        d    = paths.get('pearson_npy', 'data/pearson')
        base = _get(cfg, 'step3_descriptor', 'output_name', default='dataset-fp.npy')
        # pearson_filter.py 的命名规则：<basename>_pearson.npy
        n    = base.replace('.npy', '_pearson.npy') if base.endswith('.npy') else base + '_pearson.npy'
        # 若用户手动指定了 pearson_output_npy，则用它
        custom = _get(cfg, 'step3_descriptor', 'pearson_output_npy', default='')
        if custom:
            n = os.path.basename(custom)
        return os.path.join(_ROOT, d, n)

    # ── 默认：descriptor npy ─────────────────────────────────────────────
    d = paths.get('descriptor_npy', 'data/descriptors')
    n = _get(cfg, 'step3_descriptor', 'output_name', default='dataset-fp.npy')
    return os.path.join(_ROOT, d, n)


# ─────────────────────────────────────────────────────────────────────────────
# 各步骤执行器
# ─────────────────────────────────────────────────────────────────────────────

def run_step(step_num: int, cfg: dict, overrides: dict, dry_run: bool) -> int:
    info = STEPS.get(step_num)
    if not info:
        print(f"[ERROR] 未知步骤: {step_num}")
        return 1

    step_key = info['key']
    s = cfg.get(step_key, {})
    print(f"\n{'─'*60}")
    print(f"  Step {step_num}: {info['name']}")
    print(f"{'─'*60}")

    if not s.get('enabled', False):
        print("  [跳过] enabled=false")
        return 0

    script = info['script']

    # ── 步骤特化逻辑（注入路径到环境变量）──────────────────────────────────
    env = {}
    if step_num == 1:
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('processed_npy', 'data/processed'))
        env['MOLEARN_OUTPUT_NAME'] = s.get('output_name', 'dataset.npy')
        env['MOLEARN_XLSX']        = os.path.join(_ROOT, s.get('xlsx_config', ''))

    elif step_num == 2:
        env['MOLEARN_INPUT_NPY']  = _resolve_input_npy(cfg, step_key)
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('analysis_output', 'outputs/analysis'))

    elif step_num == 3:
        env['MOLEARN_INPUT_NPY']  = _resolve_input_npy(cfg, step_key)
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('descriptor_npy', 'data/descriptors'))
        env['MOLEARN_OUTPUT_NAME'] = s.get('output_name', 'dataset-fp.npy')
        # ── 皮尔逊相关性筛选参数透传 ──────────────────────────────────────────
        pearson_on = s.get('pearson_filter', False)
        env['MOLEARN_PEARSON_FILTER']     = str(pearson_on).lower()
        env['MOLEARN_PEARSON_THRESHOLD']  = str(s.get('pearson_threshold', 0.95))
        # 自动为 pearson 输出 npy 设置完整路径（保存到 paths.pearson_npy 目录）
        p_out_npy = s.get('pearson_output_npy', '') or ''
        if pearson_on and not p_out_npy:
            _pearson_dir = os.path.join(
                _ROOT, cfg.get('paths', {}).get('pearson_npy', 'data/pearson'))
            _base_name   = s.get('output_name', 'dataset-fp.npy')
            _pearson_name = _base_name.replace('.npy', '_pearson.npy') \
                            if _base_name.endswith('.npy') else _base_name + '_pearson.npy'
            p_out_npy    = os.path.join(_pearson_dir, _pearson_name)
        env['MOLEARN_PEARSON_OUTPUT_NPY'] = p_out_npy
        # pearson report xlsx 也放到 pearson 目录
        p_report = s.get('pearson_report_xlsx', '') or ''
        if pearson_on and not p_report:
            _pearson_dir = os.path.join(
                _ROOT, cfg.get('paths', {}).get('pearson_npy', 'data/pearson'))
            p_report = os.path.join(_pearson_dir, 'pearson_removal_report.xlsx')
        env['MOLEARN_PEARSON_REPORT']     = p_report
        env['MOLEARN_PEARSON_HEATMAP']    = str(s.get('pearson_gen_heatmap', True)).lower()
        env['MOLEARN_PEARSON_MAX_DIM']    = str(s.get('pearson_heatmap_max_dim', 300))

    elif step_num == 4:
        env['MOLEARN_INPUT_NPY']  = _resolve_input_npy(cfg, step_key)
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('samples_dir', 'data/samples'))

    elif step_num == 5:
        env['MOLEARN_INPUT_NPY']  = _resolve_input_npy(cfg, step_key)
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('splits_dir', 'data/splits'))

    elif step_num == 6:
        # 步骤6使用 config txt，通过 MOLEARN_CONFIG 传递
        return _patch_script_config(script, cfg, step_key, overrides, dry_run)

    elif step_num == 7:
        env['MOLEARN_INPUT_NPY']  = _resolve_input_npy(cfg, step_key)
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('sa_score_output', 'outputs/sa_scores'))

    elif step_num == 8:
        q_npy = s.get('query_npy', '')
        env['MOLEARN_QUERY_NPY']  = os.path.join(_ROOT, q_npy) if q_npy else ''
        env['MOLEARN_INPUT_NPY']  = _resolve_input_npy(cfg, step_key)
        env['MOLEARN_OUTPUT_DIR'] = os.path.join(
            _ROOT, cfg.get('paths', {}).get('similarity_output', 'outputs/similarity'))

    elif step_num == 9:
        pred_npy = s.get('predict_npy', '')
        env['MOLEARN_PREDICT_NPY'] = os.path.join(_ROOT, pred_npy) if pred_npy else ''
        env['MOLEARN_OUTPUT_DIR']  = os.path.join(
            _ROOT, cfg.get('paths', {}).get('prediction_output', 'outputs/predictions'))
        # 推断训练输出目录
        model_dir = s.get('model_dir', '')
        if not model_dir:
            model_dir = os.path.join(
                _ROOT,
                cfg.get('paths', {}).get('training_output', 'outputs/training'),
                cfg.get('project', {}).get('name', 'molearn_project')
            )
        seeds = _get(cfg, 'step6_train', 'seeds', default=[42])
        if isinstance(seeds, list):
            seed0 = seeds[0]
        else:
            seed0 = seeds
        env['MOLEARN_MODEL_DIR']  = os.path.join(model_dir, f"seed_{seed0}")
        env['MOLEARN_MODEL_NAME'] = s.get('model_name', 'GradientBoosting')
        # 传递 task_type（auto=由 usemodel 自动检测）
        env['MOLEARN_TASK_TYPE']  = s.get('task_type', 'auto')

    return _run_script(script, extra_env=env, dry_run=dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def _parse_step_arg(s: str) -> list:
    """解析 '--step 3,6' 或 '--step 3-6' 为整数列表。"""
    nums = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            nums += list(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(part))
    return sorted(set(nums))


def _list_steps(cfg: dict):
    """打印所有步骤状态。"""
    print("\n  Molearn 流水线步骤")
    print("  " + "─" * 50)
    for num, info in sorted(STEPS.items()):
        s     = cfg.get(info['key'], {})
        state = "✓ ON " if s.get('enabled', False) else "✗ off"
        print(f"  Step {num}  [{state}]  {info['name']:<20}  ({info['script']})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='molearn_run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Molearn 总控脚本 — 按 molearn.yaml 配置运行完整机器学习流水线

            步骤编号:
              1  数据预处理 (create_npy.py)
              2  数据库分析 (db_analysis.py)
              3  描述符计算 (create_by_fp.py)
              4  数据抽样   (sample_npy.py)
              5  数据集划分 (dataset_split.py)
              6  模型训练   (ml-m-full.py)
              7  可合成性打分(sa_score.py)
              8  相似度搜索 (similarity_search.py)
              9  模型推理   (usemodel.py)
        """)
    )
    parser.add_argument(
        '--config', default='molearn.yaml',
        help='配置文件路径（默认: molearn.yaml）'
    )
    parser.add_argument(
        '--step', default=None,
        help='指定运行步骤，如 3 或 3,6 或 3-6（默认: 运行所有 enabled 步骤）'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='列出所有步骤及其状态后退出'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='预演模式：只打印命令，不实际执行'
    )
    parser.add_argument(
        '--seed', default=None,
        help='覆盖 yaml 中的 seeds（如 42 或 42,123,456）'
    )
    parser.add_argument(
        '--task-type', default=None,
        help='覆盖 yaml 中的 task_type（regression 或 classification）'
    )
    parser.add_argument(
        '--init-dirs', action='store_true',
        help='按 paths 节点创建所有目录后退出'
    )
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    config_path = os.path.join(_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        print("  请确认 molearn.yaml 存在于项目根目录，或通过 --config 指定路径。")
        sys.exit(1)

    print(f"[INFO] 加载配置: {config_path}")
    cfg = _load_yaml(config_path)

    # ── 初始化目录 ────────────────────────────────────────────────────────────
    if args.init_dirs:
        print("[INFO] 创建项目目录结构...")
        _init_dirs(cfg, dry_run=args.dry_run)
        print("[INFO] 目录创建完成。")
        return

    # ── 列出步骤 ──────────────────────────────────────────────────────────────
    if args.list:
        _list_steps(cfg)
        return

    # ── 确定要运行的步骤 ──────────────────────────────────────────────────────
    if args.step:
        steps_to_run = _parse_step_arg(args.step)
    else:
        steps_to_run = [n for n, info in sorted(STEPS.items())
                        if cfg.get(info['key'], {}).get('enabled', False)]

    if not steps_to_run:
        print("[INFO] 没有需要运行的步骤（所有步骤 enabled=false）。")
        print("  提示：编辑 molearn.yaml 中的 enabled: true，或使用 --step N 指定步骤。")
        return

    # ── 应用资源控制 ──────────────────────────────────────────────────────────
    if not args.dry_run:
        _apply_resource_limits(cfg)

    # ── 覆盖参数 ──────────────────────────────────────────────────────────────
    overrides = {}
    if args.seed:
        seeds_list = [int(s.strip()) for s in args.seed.split(',')]
        overrides['seeds'] = seeds_list
        # 同步到 cfg（让 _make_config_txt 读到）
        if 'step6_train' in cfg:
            cfg['step6_train']['seeds'] = seeds_list

    if getattr(args, 'task_type', None):
        overrides['task_type'] = args.task_type
        if 'step6_train' in cfg:
            cfg['step6_train']['task_type'] = args.task_type

    # ── 初始化目录（正常运行时也自动创建）────────────────────────────────────
    _init_dirs(cfg, dry_run=args.dry_run)

    # ── 执行步骤 ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] 待执行步骤: {steps_to_run}")
    if args.dry_run:
        print("[INFO] *** 预演模式：以下命令不会实际执行 ***\n")

    failed = []
    start_time = datetime.datetime.now()

    for num in steps_to_run:
        ret = run_step(num, cfg, overrides, dry_run=args.dry_run)
        if ret != 0:
            print(f"\n  [ERROR] Step {num} 退出码 {ret}，流水线中止。")
            failed.append(num)
            break

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    if failed:
        print(f"  流水线在 Step {failed[0]} 中止。用时 {elapsed:.1f}s")
    else:
        print(f"  所有步骤完成！用时 {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
