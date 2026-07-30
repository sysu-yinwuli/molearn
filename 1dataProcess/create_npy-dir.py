import os
import numpy as np
from tqdm import tqdm
import random

# ---------- 用户只需改这两个 ----------
XYZ_DIR = 'xyz-add'                # 与脚本同级的 xyz 文件夹
OUT_NPY = 'hjf-add.npy'       # 输出到脚本同级目录
# ------------------------------------

def list_xyz_in_dir(directory):
    """返回目录下所有 .xyz 的相对路径（相对于 directory）"""
    lst = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.xyz'):
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, directory)
                lst.append(rel)
    return lst

def parse_xyz(file_path):
    """兼容：第一列是元素符号 或 原子序数"""
    with open(file_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        raise ValueError('empty file')
    n = int(lines[0])
    elems, coords = [], []
    for l in lines[2:2+n]:
        parts = l.split()
        first = parts[0]
        # ******* 兼容分支 *******
        if first.isdigit():               # 第一列是原子序数
            elems.append(int(first))
        else:                             # 第一列是元素符号
            symbol2z = {
                'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
                'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,
                'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
                'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
                'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
                'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,
                'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,
                'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,
                'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90
            }
            elems.append(symbol2z[first])
        coords.append([float(x) for x in parts[1:4]])
    return elems, coords, n

# ---------- 主流程 ----------
xyz_files = list_xyz_in_dir(XYZ_DIR)
xyz_files = [os.path.join(XYZ_DIR, f) for f in xyz_files]

mol_list = []
for fp in tqdm(xyz_files, desc='Parsing'):
    try:
        elems, coords, n = parse_xyz(fp)
    except Exception as e:
        print(f'Skip {fp}: {e}')
        continue
    mol_list.append({
        'name': os.path.basename(fp)[:-4],
        'elements': elems,
        'coordinates': coords,
        'atom_count': n,
        'y': 1
    })

random.shuffle(mol_list)
np.save(OUT_NPY, mol_list, allow_pickle=True)
print(f'Done! {len(mol_list)} molecules → ./{OUT_NPY}')