#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_by-fp-1.py  (参数化 + 修复 Mordred 报错)
1. 从 XYZ 文件生成 SMILES
2. 计算并整合 RDKit / MACCS / Morgan / SOAP / ACSF / Mordred
3. 所有描述符参数均在 CONFIG 中集中管理
"""

import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdmolops, AllChem
from tqdm import tqdm
import subprocess
import traceback
from rdkit import RDLogger
from collections import defaultdict
from dscribe.descriptors import SOAP, ACSF
from ase import Atoms
from ase.data import chemical_symbols
import mordred
from mordred import Calculator, descriptors

RDLogger.DisableLog('rdApp.*')

# ======================================================================
# ========== 用户配置区域（所有可调参数集中于此） ==========
# ======================================================================
CONFIG = {
    # ---- 路径 ----
    'xyz_folder': '../1dataProcess/xyz-all/',
    'smi_file':   'output-mmr-1.smi',
    'input_npy':  'poly-all.npy',
    'output_npy': 'poly-all-maccs.npy',
    'failed_record': 'failed_molecules-1.xlsx',

    # ---- 元素全集（1–6 周期 + 镧系，共 88 种）----
    'global_species': [
        'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl'
    ],

    # ---- 描述符总开关 ----
    'calc_rdkit':   False,
    'calc_maccs':   True,
    'calc_morgan':  False,
    'calc_soap':    False,
    'calc_acsf':    False,
    'calc_mordred': False,

    # ---- RDKit 指纹 ----
    'rdkit_fp_size': 2048,

    # ---- Morgan 指纹 ----
    'morgan_radius': 2,
    'morgan_nbits': 2048,

    # ---- SOAP 超参 ----
    'soap': {
        'r_cut': 5.0,
        'n_max': 6,
        'l_max': 4,
        'sigma': 0.1,
        'rbf': "gto",
        'periodic': False,
    },

    # ---- ACSF 超参 ----
    'acsf': {
        'r_cut': 6.0,
        'g2_params': [[1, 0], [1, 1], [1, 2], [1, 3],
                      [2, 1], [2, 2], [2, 3],
                      [3, 1], [3, 2], [3, 3]],
        'g4_params': [[1, 1, 1], [1, 1, -1], [1, 2, 1], [1, 2, -1],
                      [2, 1, 1], [2, 1, -1]],
        'periodic': False,
    },

    # ---- Mordred 选项 ----
    'mordred_ignore_3D': True,
}
# ======================================================================
# 以下代码不再出现任何硬编码参数
# ======================================================================

# ---------- 工具函数 ----------
def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (int, float, np.generic)):
        return [float(value)]
    if isinstance(value, (list, np.ndarray)):
        return value.astype(float).tolist() if isinstance(value, np.ndarray) else list(map(float, value))
    return []

def xyz_to_smiles(xyz_path, smi_file):
    result = subprocess.run(['obabel', xyz_path, '-O', smi_file],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Open Babel 错误: {result.stderr}")
    with open(smi_file, "r") as f:
        smiles = f.read().strip()
    return smiles.split('\t')[0]

def process_boron_smiles(smiles):
    modified = smiles.replace('[B](', '[B+](').replace(')[B]', ')[B-]')
    if '[B](' in modified and modified.count(')') > 3:
        modified = modified.replace('[B]', '[BH]')
    return modified

def create_molecule_from_smiles(smiles, name):
    max_attempts, attempts, error_log = 3, 0, []
    mol = None
    while attempts < max_attempts:
        try:
            if attempts == 0:
                mol = Chem.MolFromSmiles(smiles)
            elif attempts == 1:
                mod = process_boron_smiles(smiles)
                mol = Chem.MolFromSmiles(mod, sanitize=False)
                if mol:
                    mol.UpdatePropertyCache(strict=False)
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == 5:
                            atom.SetNumExplicitHs(0)
                            atom.SetNoImplicit(True)
                    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            else:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol:
                    mol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            if mol:
                return mol, error_log
        except Exception as e:
            error_log.append(f"Attempt {attempts+1} failed: {e}")
            if attempts == max_attempts - 1:
                error_log.append(traceback.format_exc())
        attempts += 1
    return None, error_log

# ------------------------------------------------------------
# 放在 CONFIG 区域之后，主程序之前
# ------------------------------------------------------------
from mordred import Descriptor as MordredDescriptor   # 用于类型判断

# 全局只初始化一次
_MORDRED_CALC = None
def _build_mordred_calc(ignore_3D=True):
    """只保留真正的 mordred.Descriptor 实例"""
    desc_list = [
        d for d in descriptors.all()
        if isinstance(d, MordredDescriptor) and callable(d)
    ]
    return Calculator(desc_list, ignore_3D=ignore_3D)

def calculate_mordred(mol, ignore_3D=True):
    if not CONFIG['calc_mordred']:
        return [], []
    global _MORDRED_CALC
    if _MORDRED_CALC is None:
        _MORDRED_CALC = _build_mordred_calc(ignore_3D)
    try:
        res = _MORDRED_CALC(mol)
        names = [str(k) for k in res.keys()]
        values = [float(v) if np.isfinite(v) else np.nan for v in res.values()]
        return values, names
    except Exception as e:
        return [], []
    
def calculate_ase_descriptors(xyz_path, name):
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    symbols, positions = [], []
    for line in lines[2:2+natoms]:
        parts = line.split()
        try:
            z = int(parts[0])
            symbols.append(chemical_symbols[z])
        except ValueError:
            symbols.append(parts[0])
        positions.append(list(map(float, parts[1:4])))
    atoms = Atoms(symbols=symbols, positions=positions)

    results = {}
    if CONFIG['calc_soap']:
        soap = SOAP(species=CONFIG['global_species'], **CONFIG['soap'])
        desc = soap.create(atoms).mean(axis=0)
        results['soap'] = (ensure_list(desc), [f'SOAP_{i}' for i in range(desc.shape[0])])

    if CONFIG['calc_acsf']:
        acsf = ACSF(species=CONFIG['global_species'], **CONFIG['acsf'])
        desc = acsf.create(atoms).mean(axis=0)
        results['acsf'] = (ensure_list(desc), [f'ACSF_{i}' for i in range(desc.shape[0])])
    return results

# ======================================================================
# ============================== 主程序 =================================
# ======================================================================
if __name__ == "__main__":
    desc_list = [(d[0], d[1]) for d in Descriptors._descList] if CONFIG['calc_rdkit'] else []
    desc_names = [d[0] for d in desc_list]

    data = np.load(CONFIG['input_npy'], allow_pickle=True)
    if data.ndim == 0:
        data = data.item()
    data = data['successful'] if isinstance(data, dict) and 'successful' in data else data

    d_r, failed_molecules, error_stats = [], [], defaultdict(int)

    print("\n========== 配置信息 ==========")
    print(f"XYZ 目录: {CONFIG['xyz_folder']}")
    print(f"输入文件: {CONFIG['input_npy']}")
    print(f"输出文件: {CONFIG['output_npy']}")
    print(f"失败记录: {CONFIG['failed_record']}")
    print("==============================\n")

    for d in tqdm(data, desc="处理分子"):
        info = {'name': d['name'], 'error_type': None, 'details': None,
                'smiles': None, 'traceback': None}
        try:
            d['smiles'] = xyz_to_smiles(os.path.join(CONFIG['xyz_folder'], d['name'] + '.xyz'),
                                        CONFIG['smi_file'])
        except Exception as e:
            info.update({'error_type': 'xyz_conversion', 'details': str(e),
                         'traceback': traceback.format_exc()})
            failed_molecules.append(info)
            error_stats['xyz_conversion'] += 1
            continue

        mol, errs = create_molecule_from_smiles(d['smiles'], d['name'])
        if mol is None:
            info.update({'error_type': 'mol_creation', 'details': errs,
                         'smiles': d['smiles'], 'traceback': traceback.format_exc()})
            failed_molecules.append(info)
            error_stats['mol_creation'] += 1
            continue

        errs = []

        # ---------- RDKit ----------
        if CONFIG['calc_rdkit']:
            row = []
            for n, f in desc_list:
                try:
                    row.append(float(f(mol)) if f(mol) is not None else np.nan)
                except Exception as e:
                    errs.append(f"RDKit {n} 失败: {e}")

            try:
                rdk_fp = rdmolops.RDKFingerprint(mol, fpSize=CONFIG['rdkit_fp_size'])
                rdk_bits = list(rdk_fp)
            except Exception as e:
                errs.append(f"RDK 指纹失败: {e}")
                rdk_bits = []

            d['rdkit_descriptor'] = row + rdk_bits
            d['rdkit_f'] = desc_names + [f'RDK_{i}' for i in range(len(rdk_bits))]

        # ---------- MACCS ----------
        if CONFIG['calc_maccs']:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                d['maccs_descriptor'] = list(fp)
                d['maccs_f'] = [f'MACCS_{i}' for i in range(len(fp))]
            except Exception as e:
                errs.append(f"MACCS 失败: {e}")

        # ---------- Morgan ----------
        if CONFIG['calc_morgan']:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    CONFIG['morgan_radius'],
                    CONFIG['morgan_nbits'])
                d['morgan_descriptor'] = list(fp)
                d['morgan_f'] = [f'Morgan_{i}' for i in range(len(fp))]
            except Exception as e:
                errs.append(f"Morgan 失败: {e}")

        # ---------- Mordred ----------
        if CONFIG['calc_mordred']:
            try:
                mordred_vals, mordred_names = calculate_mordred(mol, CONFIG['mordred_ignore_3D'])
                d['mordred_descriptor'] = mordred_vals
                d['mordred_f'] = mordred_names
            except Exception as e:
                errs.append(f"Mordred 失败: {e}")

        # ---------- SOAP / ACSF ----------
        if CONFIG['calc_soap'] or CONFIG['calc_acsf']:
            try:
                res = calculate_ase_descriptors(os.path.join(CONFIG['xyz_folder'], d['name'] + '.xyz'), d['name'])
                if CONFIG['calc_soap'] and 'soap' in res:
                    d['soap_descriptor'], d['soap_f'] = res['soap']
                if CONFIG['calc_acsf'] and 'acsf' in res:
                    d['acsf_descriptor'], d['acsf_f'] = res['acsf']
            except Exception as e:
                errs.append(f"ASE 描述符失败: {e}")

        if errs:
            info.update({'error_type': 'descriptor_calculation',
                         'details': "; ".join(errs),
                         'smiles': d['smiles'],
                         'traceback': traceback.format_exc()})
            failed_molecules.append(info)
            error_stats['descriptor_calculation'] += len(errs)
            continue

        d_r.append(d)

    # ---------- 保存失败记录 ----------
    pd.DataFrame(failed_molecules,
                 columns=['name', 'error_type', 'details', 'smiles', 'traceback']).to_excel(
        CONFIG['failed_record'], index=False, engine='openpyxl')

    print(f"\n成功: {len(d_r)}/{len(data)}")
    print("失败统计:", dict(error_stats))
    print(f"失败记录已保存到: {CONFIG['failed_record']}")

    # ---------- 保存结果 ----------
    save_dict = {
        'successful': d_r,
        'failed_count': len(failed_molecules),
        'error_stats': dict(error_stats),
        'config': CONFIG
    }
    np.save(CONFIG['output_npy'], save_dict, allow_pickle=True)
    print(f"结果已保存到: {CONFIG['output_npy']}\n所有计算完成!")

