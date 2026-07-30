import openpyxl
import numpy as np

xlsx_file = 'sample-space-all-scored-percentile.xlsx'
de_list = [5]
in_npy = "hjf-all-fp.npy"
out_npy = 'hjf-all-fp-score.npy'

# 读取Excel文件数据
workbook = openpyxl.load_workbook(xlsx_file)
worksheet = workbook.active
m_name_y = {}
header = []

# 获取header，假设头部在第一行
header_row = worksheet[1]
header = [header_row[d].value for d in de_list]

for row in worksheet.iter_rows(min_row=2, values_only=True):
    key = row[0]
    if key is None:
        continue
    key = str(key).split('.')[0]  # 去掉扩展名
    value = [row[d] for d in de_list]
    m_name_y[key] = value
workbook.close()

# 加载 .npy 数据
data = np.load(in_npy, allow_pickle=True)

# 修复：处理 0-d 数组
if data.ndim == 0:
    data = data.item()

# 确保是列表
if isinstance(data, dict) and 'successful' in data:
    molecules = data['successful']
else:
    molecules = data

# 添加额外特征
for d in molecules:
    key = d['name'].split('.')[0]
    if key in m_name_y:
        d["score_d"] = m_name_y[key]
        d["name_of_score"] = header
    else:
        print(f"[Warning] {key} not found in Excel, skipping score_d")

# 保存
np.save(out_npy, data if isinstance(data, dict) else molecules, allow_pickle=True)

