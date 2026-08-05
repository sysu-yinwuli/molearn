import os
import numpy as np
import openpyxl
#获得一个文件夹下所有文件的名字返回一个列表
def get_file_names_in_folder(folder_path):
    file_names = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_names.append(filename.split('.')[0])
    return file_names


def parse_xyz_file(file_path):
    element_symbols = []
    coordinates = []
    num_atoms = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines)==0:
            print(file_path)
            exit(0)
        num_atoms = int(lines[0].strip())

        for line in lines[2:num_atoms + 2]:
            parts = line.strip().split()
            element_symbols.append(parts[0])
            coordinates.append([float(coord) for coord in parts[1:]])

    # Mapping of element symbols to element numbers (1 to 90)
    element_number_map = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
        'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
        'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
        'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
        'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
        'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
        'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
        'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    }
    if element_symbols[0] in element_number_map.keys():
        element_numbers = [element_number_map[element] for element in element_symbols]
        return element_numbers, coordinates, num_atoms

    return element_symbols, coordinates,num_atoms




def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            file_list.append(relative_path)
    return file_list



import pandas as pd

# =============================================================================
# ── MOLEARN_ 环境变量覆盖（由 molearn_run.py 自动设置，单独运行时忽略）─────────
# MOLEARN_XLSX       : 覆盖配置 xlsx 路径
# MOLEARN_OUTPUT_DIR : 覆盖所有 npy 输出目录（xlsx 中的 npy_path 改为此目录下同名文件）
# =============================================================================
_env_xlsx       = os.environ.get('MOLEARN_XLSX', '').strip()
_env_output_dir = os.environ.get('MOLEARN_OUTPUT_DIR', '').strip()
if _env_output_dir:
    os.makedirs(_env_output_dir, exist_ok=True)

# 读取 Excel 文件
file_path = _env_xlsx if _env_xlsx else 'creat_npy.xlsx'
df = pd.read_excel(file_path)

# 将每一行转换为字典，并将所有字典放入列表
list_of_dicts = df.to_dict(orient='records')

# 打印结果（可选）
for record in list_of_dicts:



    xyz_folder = record['xyz_path']
    xlsx_file = record['xlsx_file']
    key_idx =  int(record['key_idx'])
    y_idx = int(record['y_idx'])
    flag_idx = int(record['flag_idx'])
    out_npy = record['npy_path']
    # 若 MOLEARN_OUTPUT_DIR 已设置，将输出重定向到统一目录
    if _env_output_dir:
        out_npy = os.path.join(_env_output_dir, os.path.basename(out_npy))


    directory_path = xyz_folder  # 替换成你要处理的文件夹路径
    file_list = list_files_in_directory(directory_path)
    for i in range(0,len(file_list)):
        file_list[i]= xyz_folder+'/'+file_list[i]

    files = {}
    for f in file_list:
        files[f.split('/')[-1].split('.')[0]] = f
    print(len(files))



    # 读取Excel文件数据
    workbook = openpyxl.load_workbook(xlsx_file)
    worksheet = workbook.active
    m_name_y = []
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        if row[flag_idx] == 0:
            continue

        d_t = {}
        key = row[key_idx].split('.')[0]
        value = row[y_idx]

        d_t['key'] = key
        d_t['value'] = value
        m_name_y.append(d_t)

    workbook.close()



    #处理原始数据
    mol_list = []
    from tqdm import tqdm
    for m in tqdm(m_name_y):
        key = m['key']
        value = m['value']
        mol = {}
        mol['name'] = key
        elements, coordinates, atom_count = parse_xyz_file(files[key])
        mol['elements'] = [int(e) for e in elements]
        mol['coordinates'] = coordinates
        mol['atom_count'] = atom_count
        mol['y'] = value
        mol_list.append(mol)

    print(len(mol_list))


    import random

    # my_list = [1, 2, 3, 4, 5]
    random.shuffle(mol_list)




    np.save(out_npy, mol_list, allow_pickle=True)

    # loaded_dict = np.load('data_2w.npy', allow_pickle=True)

