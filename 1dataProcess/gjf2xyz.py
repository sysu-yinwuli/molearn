import os
from tqdm import tqdm

def gjf_to_xyz(gjf_path, xyz_path):
    # 读取GJF文件内容
    with open(gjf_path, 'r') as file:
        lines = file.readlines()

    # 查找开始读取原子坐标的行数，通常在第一个空行之后
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            start +=1
        if start==2:
            start=i+1
            break

    # 开始转换为XYZ格式
    atom_count = 0
    atoms = []
    for line in lines[start:]:
        if line.strip() == '':
            break
        parts = line.split()
        if len(parts) >= 4:
            element = parts[0]
            x = parts[1]
            y = parts[2]
            z = parts[3]
            atoms.append(f"{element} {x} {y} {z}")
            atom_count += 1

    # 将XYZ内容写入新文件
    with open(xyz_path, 'w') as file:
        file.write(f"{atom_count}\n\n")  # XYZ头部，原子总数和一个空行
        file.write("\n".join(atoms))
        file.write("\n\n\n")


def convert_folder(src_folder, dest_folder):
    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有文件
    for filename in tqdm(os.listdir(src_folder)):
        if filename.endswith('.gjf'):
            src_file = os.path.join(src_folder, filename)
            xyz_file = os.path.join(dest_folder, filename.replace('.gjf', '.xyz'))
            gjf_to_xyz(src_file, xyz_file)
            # print(f"Converted {src_file} to {xyz_file}")


# 使用示例
src_folder = 'gjf'
dest_folder = 'xyz'
convert_folder(src_folder, dest_folder)


