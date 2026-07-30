import os
import pandas as pd
import re

# ================= 配置区域 =================
RES_FOLDER = "res"          # 根目录下的结果文件夹名
SEED_FOLDER = "seed_5"      # 种子文件夹名
RESULT_FILE = "results.txt"  # 结果文件名
OUTPUT_EXCEL = "summary.xlsx"  # 输出 Excel 文件名
# ===========================================

def parse_result_txt(content):
    data = []
    for line in content.strip().splitlines():
        match = re.match(r"^(.*?):\s*MAE=([0-9.-]+),\s*MSE=([0-9.-]+),\s*R²=([0-9.-]+)$", line.strip())
        if match:
            model, mae, mse, r2 = match.groups()
            data.append((model.strip(), float(mae), float(mse), float(r2)))
        elif line.strip():
            print(f"⚠️ 未匹配行: {line.strip()}")
    return data

def generate_excel():
    all_data = []
    for exp in os.listdir(RES_FOLDER):
        path = os.path.join(RES_FOLDER, exp, SEED_FOLDER, "results", RESULT_FILE)
        if not os.path.isfile(path):
            print(f"⚠️ 跳过: {path}")
            continue
        with open(path, encoding='utf-8') as f:
            for model, mae, mse, r2 in parse_result_txt(f.read()):
                all_data.append([exp, model, mae, mse, r2])

    if not all_data:
        print("❌ 没有解析到任何数据，Excel 未生成")
        return

    df = pd.DataFrame(all_data, columns=["Experiment", "Model", "MAE", "MSE", "R2"])
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"✅ Excel 已保存为: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    generate_excel()