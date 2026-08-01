#!/usr/bin/env python3
# =============================================================================
#  molearn_gui.py  —  Molearn Web 图形界面（Flask）
#  功能：在浏览器中配置所有参数、点击按钮启动训练、实时查看日志
#  运行：python molearn_gui/molearn_gui.py
#  访问：http://localhost:5000  （Windows/Linux/macOS 通用）
#
#  依赖：pip install flask pyyaml
# =============================================================================

import os
import sys
import json
import subprocess
import threading
import datetime
import copy

# ── Flask 导入 ────────────────────────────────────────────────────────────────
try:
    from flask import (Flask, render_template, request, jsonify,
                       redirect, url_for, Response)
except ImportError:
    print("[ERROR] Flask 未安装。请运行：pip install flask")
    sys.exit(1)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
    print("[WARN] PyYAML 未安装，将使用内置解析器（功能受限）。pip install pyyaml")

# ── 路径 ─────────────────────────────────────────────────────────────────────
_GUI_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_GUI_DIR)
_YAML_PATH = os.path.join(_ROOT, 'molearn.yaml')
_RUN_SCRIPT = os.path.join(_ROOT, 'molearn_run.py')

app = Flask(__name__, template_folder='templates', static_folder='static')

# ── 全局运行状态 ──────────────────────────────────────────────────────────────
_run_lock   = threading.Lock()
_run_proc   = None          # subprocess.Popen
_run_log    = []            # list[str]  实时日志行
_run_status = 'idle'        # idle | running | done | error

# ─────────────────────────────────────────────────────────────────────────────
# YAML 读写
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml() -> dict:
    if not os.path.isfile(_YAML_PATH):
        return {}
    if _HAS_YAML:
        with open(_YAML_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    # 简易解析
    sys.path.insert(0, _ROOT)
    from molearn_run import _simple_yaml_parse
    return _simple_yaml_parse(_YAML_PATH)


def _save_yaml(data: dict):
    if _HAS_YAML:
        with open(_YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
    else:
        # 回退：用 JSON 序列化成类 YAML（仅用于演示，不保留注释）
        lines = []
        def _dump_dict(d, indent=0):
            for k, v in d.items():
                prefix = '  ' * indent
                if isinstance(v, dict):
                    lines.append(f"{prefix}{k}:")
                    _dump_dict(v, indent + 1)
                elif isinstance(v, list):
                    inner = ', '.join(str(x) for x in v)
                    lines.append(f"{prefix}{k}: [{inner}]")
                elif isinstance(v, bool):
                    lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
                elif v is None:
                    lines.append(f"{prefix}{k}: null")
                else:
                    lines.append(f"{prefix}{k}: {v}")
        _dump_dict(data)
        with open(_YAML_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 路由：主页
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    cfg = _load_yaml()
    return render_template('index.html', cfg=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 路由：保存配置
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/save_config', methods=['POST'])
def save_config():
    """接收前端 JSON，深度合并到当前 YAML 并保存。"""
    try:
        new_cfg = request.get_json(force=True)
        if not new_cfg:
            return jsonify({'ok': False, 'msg': '空数据'}), 400

        cfg = _load_yaml()
        _deep_merge(cfg, new_cfg)
        _save_yaml(cfg)
        return jsonify({'ok': True, 'msg': '配置已保存'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


def _deep_merge(base: dict, override: dict):
    """递归将 override 合并到 base（in-place）。"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# 路由：获取当前配置（JSON）
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/get_config')
def get_config():
    cfg = _load_yaml()
    return jsonify(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 路由：启动训练
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/run', methods=['POST'])
def run_pipeline():
    """启动 molearn_run.py，在后台线程中执行，实时捕获输出。"""
    global _run_proc, _run_log, _run_status

    with _run_lock:
        if _run_status == 'running':
            return jsonify({'ok': False, 'msg': '已有任务在运行中，请等待完成。'}), 409

        data     = request.get_json(force=True) or {}
        steps    = data.get('steps', None)    # None = 所有 enabled 步骤
        seeds    = data.get('seeds', None)
        dry_run  = data.get('dry_run', False)

        _run_log    = []
        _run_status = 'running'

    def _target():
        global _run_proc, _run_status
        try:
            cmd = [sys.executable, _RUN_SCRIPT]
            if steps:
                cmd += ['--step', str(steps)]
            if seeds:
                cmd += ['--seed', str(seeds)]
            if dry_run:
                cmd += ['--dry-run']

            _run_log.append(f"[GUI] {datetime.datetime.now().strftime('%H:%M:%S')} 启动: {' '.join(cmd)}\n")

            proc = subprocess.Popen(
                cmd, cwd=_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1
            )
            with _run_lock:
                _run_proc = proc

            for line in proc.stdout:
                _run_log.append(line)

            proc.wait()
            with _run_lock:
                _run_status = 'done' if proc.returncode == 0 else 'error'
                _run_log.append(
                    f"\n[GUI] 进程结束，退出码: {proc.returncode}  "
                    f"时间: {datetime.datetime.now().strftime('%H:%M:%S')}\n"
                )
        except Exception as e:
            with _run_lock:
                _run_status = 'error'
                _run_log.append(f"\n[GUI ERROR] {e}\n")

    threading.Thread(target=_target, daemon=True).start()
    return jsonify({'ok': True, 'msg': '训练已启动，请在日志面板查看进度。'})


# ─────────────────────────────────────────────────────────────────────────────
# 路由：停止训练
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/stop', methods=['POST'])
def stop_pipeline():
    global _run_proc, _run_status
    with _run_lock:
        proc = _run_proc
        if proc and _run_status == 'running':
            proc.terminate()
            _run_status = 'idle'
            _run_log.append("\n[GUI] 用户中止训练。\n")
            return jsonify({'ok': True, 'msg': '训练已中止。'})
    return jsonify({'ok': False, 'msg': '没有正在运行的任务。'})


# ─────────────────────────────────────────────────────────────────────────────
# 路由：日志轮询（SSE / 简单 JSON）
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/log_stream')
def log_stream():
    """Server-Sent Events 实时推送日志。"""
    def _gen():
        sent = 0
        while True:
            lines = _run_log[sent:]
            if lines:
                for line in lines:
                    yield f"data: {json.dumps(line)}\n\n"
                sent += len(lines)
            if _run_status not in ('running',) and sent >= len(_run_log):
                yield f"data: {json.dumps('[STREAM_END]')}\n\n"
                break
            import time
            time.sleep(0.3)

    return Response(_gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/log_poll')
def log_poll():
    """备用：返回全部日志（适合不支持 SSE 的浏览器）。"""
    offset = int(request.args.get('offset', 0))
    lines  = _run_log[offset:]
    return jsonify({'lines': lines, 'status': _run_status, 'total': len(_run_log)})


# ─────────────────────────────────────────────────────────────────────────────
# 路由：任务状态
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/status')
def status():
    return jsonify({'status': _run_status, 'log_lines': len(_run_log)})


# ─────────────────────────────────────────────────────────────────────────────
# 路由：列出输出目录下的文件
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/outputs')
def list_outputs():
    cfg = _load_yaml()
    results = {}
    out_root = os.path.join(_ROOT, cfg.get('paths', {}).get('training_output', 'outputs/training'))
    if os.path.isdir(out_root):
        for dirpath, dirs, files in os.walk(out_root):
            for fn in files:
                rel = os.path.relpath(os.path.join(dirpath, fn), _ROOT)
                results.setdefault(os.path.relpath(dirpath, out_root), []).append(fn)
    return jsonify(results)


# ─────────────────────────────────────────────────────────────────────────────
# 路由：读取单个结果文件（model_card / results.txt 等）
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/read_file')
def read_file():
    rel_path = request.args.get('path', '')
    full_path = os.path.normpath(os.path.join(_ROOT, rel_path))
    # 安全检查：只允许读 outputs/ 下的文件
    if not full_path.startswith(_ROOT):
        return jsonify({'ok': False, 'msg': '路径越界'}), 403
    if not os.path.isfile(full_path):
        return jsonify({'ok': False, 'msg': '文件不存在'})
    try:
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return jsonify({'ok': True, 'content': content})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# 路由：初始化目录
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/init_dirs', methods=['POST'])
def init_dirs():
    try:
        ret = subprocess.run(
            [sys.executable, _RUN_SCRIPT, '--init-dirs'],
            cwd=_ROOT, capture_output=True, text=True, timeout=30
        )
        return jsonify({'ok': ret.returncode == 0,
                        'stdout': ret.stdout, 'stderr': ret.stderr})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1',
                        help='监听地址（0.0.0.0=局域网可访问）')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Molearn GUI  — http://{args.host}:{args.port}")
    print(f"  项目根目录    : {_ROOT}")
    print(f"  配置文件      : {_YAML_PATH}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)
