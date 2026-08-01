#!/usr/bin/env python3
# read_res.py  —— 汇总所有实验 / 所有 seed 的结果到一个 Excel
# =============================================================================
# ============================= 配置区域（只改这里）============================
# =============================================================================

RES_FOLDER   = "results"       # 顶层结果目录（包含各实验子目录）
OUTPUT_EXCEL = "summary.xlsx"  # 输出 Excel 文件名
RESULT_FILE  = "results.txt"   # 每个 seed 目录下的结果文件名

# =============================================================================
# ============================= 以下代码勿动 ==================================
# =============================================================================

import os
import re
import pandas as pd


def parse_result_txt(path: str) -> list[tuple]:
    """
    解析单个 results.txt，返回 [(model, mae, mse, r2), ...]。
    自动从文件名所在路径读取 seed 信息（# Seed=xxx 注释行）。
    """
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(
                r'^(.*?):\s*MAE=([0-9eE.+-]+),\s*MSE=([0-9eE.+-]+),\s*R²=([0-9eE.+-]+)$',
                line
            )
            if m:
                model, mae, mse, r2 = m.groups()
                rows.append((model.strip(), float(mae), float(mse), float(r2)))
            else:
                print(f"  [WARN] 未匹配行: {line}")
    return rows


def collect_results(res_folder: str, result_file: str) -> pd.DataFrame:
    """
    遍历结构：res_folder/[experiment]/[seed_xxx]/results/result_file
    自动发现所有实验和所有 seed，无需手动指定。
    """
    all_rows = []
    if not os.path.isdir(res_folder):
        raise FileNotFoundError(f"结果目录不存在: {res_folder}")

    for exp in sorted(os.listdir(res_folder)):
        exp_dir = os.path.join(res_folder, exp)
        if not os.path.isdir(exp_dir):
            continue
        for seed_dir in sorted(os.listdir(exp_dir)):
            # 支持 seed_42、seed_0 等命名
            if not seed_dir.startswith('seed_'):
                continue
            result_path = os.path.join(exp_dir, seed_dir, 'results', result_file)
            if not os.path.isfile(result_path):
                print(f"  [SKIP] {result_path}")
                continue
            seed_val = seed_dir.split('_', 1)[1]
            for model, mae, mse, r2 in parse_result_txt(result_path):
                all_rows.append([exp, seed_val, model, mae, mse, r2])

    if not all_rows:
        raise RuntimeError("未解析到任何数据，请检查 RES_FOLDER 路径与目录结构。")

    return pd.DataFrame(all_rows, columns=['Experiment', 'Seed', 'Model', 'MAE', 'MSE', 'R2'])


if __name__ == '__main__':
    print(f"扫描目录: {RES_FOLDER}")
    df = collect_results(RES_FOLDER, RESULT_FILE)
    print(f"共收集 {len(df)} 条记录，涉及 {df['Experiment'].nunique()} 个实验，"
          f"{df['Seed'].nunique()} 个 seed，{df['Model'].nunique()} 个模型")
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"已保存: {OUTPUT_EXCEL}")
