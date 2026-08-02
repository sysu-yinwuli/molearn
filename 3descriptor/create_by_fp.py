#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_by_fp.py — 分子描述符计算脚本（全功能版）
================================================================
支持描述符类型：
  ① RDKit 物理化学描述符 + RDKit 拓扑指纹
  ② MACCS Keys（167 位）
  ③ Morgan 圆形指纹（ECFP，可选 count 或 bit 模式）
  ④ AtomPair 指纹
  ⑤ Topological Torsion 指纹
  ⑥ Avalon 指纹（需 rdkit.Chem.Avalontools）
  ⑦ SOAP（DScribe）
  ⑧ ACSF（DScribe）
  ⑨ MBTR（DScribe）——多体张量表示
  ⑩ Mordred（2D / 3D 描述符）
  ⑪ SMILES 属性（分子量、HBA、HBD、LogP、TPSA、RotBonds、AromaticRings）

所有参数在 CONFIG 中集中配置，代码体勿改。
"""

import pandas as pd
import numpy as np
import os
import subprocess
import traceback
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import (Descriptors, MACCSkeys, rdmolops, AllChem,
                         rdMolDescriptors, DataStructs)
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# ======================================================================
# ========== 用户配置区域（所有可调参数集中于此）==========
# ======================================================================
CONFIG = {
    # ---- 路径 ----
    'xyz_folder':     '../1dataProcess/xyz-all/',
    'smi_file':       'output-mmr-1.smi',
    'input_npy':      'poly-all.npy',
    'output_npy':     'poly-all-fp.npy',
    'failed_record':  'failed_molecules.xlsx',

    # ---- 皮尔逊相关性过滤（可选）----
    # 设为 True 后，在描述符计算完成后自动进行共线性剔除
    # 详细参数见 pearson_filter.py 的 CONFIG，此处只设核心开关和阈值
    'pearson_filter':           False,   # True=启用，False=跳过
    'pearson_threshold':        0.95,    # |r| 超过此值则剔除方差小的那列
    'pearson_output_npy':       '',      # 留空：自动命名为 <output_npy基名>_pearson.npy
    'pearson_report_xlsx':      '',      # 留空：自动命名为 pearson_removal_report.xlsx
    'pearson_gen_heatmap':      True,    # 是否生成热图（维数 ≤ 300 时有效）

    # ---- 元素全集（1–4 周期常用）----
    # 若分子含更重的元素，需在此追加
    'global_species': [
        'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
        'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    ],

    # ====== 描述符开关（True=计算，False=跳过）======
    'calc_rdkit':         True,    # RDKit 物化描述符 + 拓扑指纹
    'calc_maccs':         True,    # MACCS Keys 167 位
    'calc_morgan':        True,    # Morgan 圆形指纹（ECFP）
    'calc_atompair':      False,   # AtomPair 指纹
    'calc_torsion':       False,   # Topological Torsion 指纹
    'calc_avalon':        False,   # Avalon 指纹（需 rdkit.Contrib.Avalon）
    'calc_soap':          False,   # SOAP（DScribe）
    'calc_acsf':          False,   # ACSF（DScribe）
    'calc_mbtr':          False,   # MBTR（DScribe）
    'calc_mordred':       False,   # Mordred 描述符
    'calc_prop':          True,    # 基础分子属性（MW/HBA/HBD/LogP/TPSA 等）

    # ====== RDKit 描述符参数 ======
    'rdkit_include_desc': True,    # 是否包含 RDKit 物化描述符（约 200 个）
    'rdkit_fp_size':      2048,    # RDKit 拓扑指纹长度（bits）
    'rdkit_fp_minpath':   1,       # 最短路径
    'rdkit_fp_maxpath':   7,       # 最长路径
    'rdkit_fp_nBitsPerHash': 2,    # 每次哈希的位数

    # ====== Morgan 指纹参数 ======
    'morgan_radius':      2,       # 半径（2→ECFP4，3→ECFP6）
    'morgan_nbits':       2048,    # 指纹长度
    'morgan_use_features': False,  # True→FCFP，False→ECFP（基于原子不变量）
    'morgan_use_counts':  False,   # True→count 向量，False→bit 向量

    # ====== AtomPair 指纹参数 ======
    'atompair_nbits':     2048,
    'atompair_use_counts': False,

    # ====== Topological Torsion 指纹参数 ======
    'torsion_nbits':      2048,
    'torsion_use_counts': False,

    # ====== Avalon 指纹参数 ======
    'avalon_nbits':       512,

    # ====== SOAP 超参 ======
    'soap': {
        'r_cut':   5.0,      # 截断半径（Å）
        'n_max':   8,        # 径向基函数最大主量子数
        'l_max':   6,        # 球谐函数最大角量子数
        'sigma':   0.5,      # 原子高斯展宽（Å）
        'rbf':     'gto',    # 'gto' | 'polynomial'
        'periodic': False,
        'average': 'inner',  # 'inner' | 'outer' | 'off（per-atom）
    },

    # ====== ACSF 超参 ======
    'acsf': {
        'r_cut':    6.0,
        # G2: [eta, mu]；G4: [eta, zeta, lambda]
        'g2_params': [[0.5, 0], [1.0, 0], [2.0, 0], [4.0, 0],
                      [0.5, 2], [1.0, 2], [2.0, 4]],
        'g4_params': [[0.5, 1, 1], [0.5, 1, -1], [0.5, 2, 1], [0.5, 2, -1],
                      [1.0, 1, 1], [1.0, 1, -1]],
        'periodic':  False,
    },

    # ====== MBTR 超参 ======
    'mbtr': {
        'geometry':    {'function': 'inverse_distance'},
        'grid':        {'min': 0, 'max': 1, 'n': 100, 'sigma': 0.1},
        'weighting':   {'function': 'exp', 'scale': 0.5, 'threshold': 1e-3},
        'periodic':    False,
        'normalization': 'l2',
    },

    # ====== Mordred 选项 ======
    'mordred_ignore_3D': True,     # False 时需要 3D 坐标

    # ====== 基础分子属性 ======
    # 这些属性会以独立字段追加到 npy，不占用 descriptor 向量
    'prop_fields': [
        'MolWt', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
        'MolLogP', 'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
        'NumRings', 'FractionCSP3', 'NumHeteroatoms',
    ],
}
# ======================================================================
# 以下代码不再出现任何硬编码参数
# ======================================================================

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (int, float, np.generic)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, list):
        return list(map(float, value))
    return []


def xyz_to_smiles(xyz_path, smi_file):
    result = subprocess.run(
        ['obabel', xyz_path, '-O', smi_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Open Babel 错误: {result.stderr.strip()}")
    with open(smi_file, 'r') as f:
        smiles = f.read().strip()
    return smiles.split('\t')[0]


def process_boron_smiles(smiles):
    modified = smiles.replace('[B](', '[B+](').replace(')[B]', ')[B-]')
    if '[B](' in modified and modified.count(')') > 3:
        modified = modified.replace('[B]', '[BH]')
    return modified


def create_molecule_from_smiles(smiles, name):
    """尝试多策略从 SMILES 创建 RDKit Mol 对象。"""
    max_attempts, error_log = 3, []
    mol = None
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                mol = Chem.MolFromSmiles(smiles)
            elif attempt == 1:
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
            error_log.append(f"Attempt {attempt+1}: {e}")
    return None, error_log


# ── Mordred 计算器（全局单例）────────────────────────────────────────────────
_MORDRED_CALC = None

def _build_mordred_calc(ignore_3D=True):
    from mordred import Calculator, descriptors
    from mordred import Descriptor as _MD
    desc_list = [d for d in descriptors.all() if isinstance(d, _MD) and callable(d)]
    return Calculator(desc_list, ignore_3D=ignore_3D)


def calculate_mordred(mol):
    global _MORDRED_CALC
    if _MORDRED_CALC is None:
        _MORDRED_CALC = _build_mordred_calc(CONFIG['mordred_ignore_3D'])
    try:
        res = _MORDRED_CALC(mol)
        names  = [str(k) for k in res.keys()]
        values = [float(v) if (not isinstance(v, Exception) and np.isfinite(float(v) if not isinstance(v, Exception) else float('nan'))) else np.nan
                  for v in res.values()]
        return values, names
    except Exception:
        return [], []


# ── ASE / DScribe 描述符 ─────────────────────────────────────────────────────
def calculate_ase_descriptors(xyz_path):
    from ase import Atoms
    from ase.data import chemical_symbols

    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    natoms   = int(lines[0].strip())
    symbols, positions = [], []
    for line in lines[2: 2 + natoms]:
        parts = line.split()
        try:
            z = int(parts[0])
            symbols.append(chemical_symbols[z])
        except ValueError:
            symbols.append(parts[0])
        positions.append(list(map(float, parts[1:4])))
    atoms   = Atoms(symbols=symbols, positions=positions)
    species = CONFIG['global_species']
    results = {}

    if CONFIG['calc_soap']:
        from dscribe.descriptors import SOAP
        soap_cfg = {k: v for k, v in CONFIG['soap'].items() if k != 'average'}
        avg_mode = CONFIG['soap'].get('average', 'inner')
        soap = SOAP(species=species, **soap_cfg)
        desc = soap.create(atoms)
        # 对 per-atom 输出取平均（保证输出维数固定）
        if desc.ndim == 2:
            desc = desc.mean(axis=0)
        results['soap'] = (ensure_list(desc), [f'SOAP_{i}' for i in range(len(desc))])

    if CONFIG['calc_acsf']:
        from dscribe.descriptors import ACSF
        acsf = ACSF(species=species, **CONFIG['acsf'])
        desc = acsf.create(atoms).mean(axis=0)  # per-atom mean
        results['acsf'] = (ensure_list(desc), [f'ACSF_{i}' for i in range(len(desc))])

    if CONFIG['calc_mbtr']:
        from dscribe.descriptors import MBTR
        mbtr = MBTR(species=species, **CONFIG['mbtr'])
        desc = mbtr.create(atoms)
        if desc.ndim > 1:
            desc = desc.flatten()
        results['mbtr'] = (ensure_list(desc), [f'MBTR_{i}' for i in range(len(desc))])

    return results, atoms


# ── Avalon 指纹（rdkit.Contrib 可选）────────────────────────────────────────
def _get_avalon_fp(mol):
    try:
        from rdkit.Avalon import pyAvalonTools
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=CONFIG['avalon_nbits'])
        return list(fp), [f'Avalon_{i}' for i in range(CONFIG['avalon_nbits'])]
    except ImportError:
        try:
            from rdkit.Chem.Avalontools import GetAvalonFP
            fp = GetAvalonFP(mol, nBits=CONFIG['avalon_nbits'])
            return list(fp), [f'Avalon_{i}' for i in range(CONFIG['avalon_nbits'])]
        except Exception:
            raise ImportError("Avalon 指纹需要 rdkit Avalon Contrib 模块")


# ── 基础分子属性 ─────────────────────────────────────────────────────────────
_PROP_FUNCS = {
    'MolWt':            Descriptors.MolWt,
    'HeavyAtomCount':   Descriptors.HeavyAtomMolWt,
    'NumHAcceptors':    rdMolDescriptors.CalcNumHBA,
    'NumHDonors':       rdMolDescriptors.CalcNumHBD,
    'MolLogP':          Descriptors.MolLogP,
    'TPSA':             rdMolDescriptors.CalcTPSA,
    'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds,
    'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings,
    'NumRings':         rdMolDescriptors.CalcNumRings,
    'FractionCSP3':     rdMolDescriptors.CalcFractionCSP3,
    'NumHeteroatoms':   rdMolDescriptors.CalcNumHeteroatoms,
}

def calculate_props(mol):
    props = {}
    for name in CONFIG.get('prop_fields', []):
        func = _PROP_FUNCS.get(name)
        if func:
            try:
                props[name] = float(func(mol))
            except Exception:
                props[name] = np.nan
    return props


# ======================================================================
# ============================== 主程序 =================================
# ======================================================================
if __name__ == '__main__':
    # ── 打印配置摘要 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  create_by_fp.py — 分子描述符计算")
    print("=" * 60)
    print(f"  输入 npy  : {CONFIG['input_npy']}")
    print(f"  输出 npy  : {CONFIG['output_npy']}")
    print(f"  XYZ 目录  : {CONFIG['xyz_folder']}")
    enabled = [k.replace('calc_', '') for k, v in CONFIG.items()
               if k.startswith('calc_') and v]
    print(f"  启用描述符: {', '.join(enabled)}")
    print("=" * 60 + "\n")

    # ── RDKit 描述符名列表 ────────────────────────────────────────────────
    if CONFIG['calc_rdkit'] and CONFIG['rdkit_include_desc']:
        rdkit_desc_list  = [(n, f) for n, f in Descriptors._descList]
        rdkit_desc_names = [n for n, _ in rdkit_desc_list]
    else:
        rdkit_desc_list  = []
        rdkit_desc_names = []

    # ── 加载输入 npy ──────────────────────────────────────────────────────
    raw = np.load(CONFIG['input_npy'], allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    data = list(raw['successful'] if isinstance(raw, dict) and 'successful' in raw else raw)

    d_r, failed_molecules, error_stats = [], [], defaultdict(int)

    # ── 主循环 ────────────────────────────────────────────────────────────
    for d in tqdm(data, desc="计算描述符"):
        info = {'name': d['name'], 'error_type': None, 'details': None,
                'smiles': None, 'traceback': None}
        errs = []

        # 1) XYZ → SMILES
        xyz_path = os.path.join(CONFIG['xyz_folder'], d['name'] + '.xyz')
        try:
            d['smiles'] = xyz_to_smiles(xyz_path, CONFIG['smi_file'])
        except Exception as e:
            info.update({'error_type': 'xyz_conversion', 'details': str(e),
                         'traceback': traceback.format_exc()})
            failed_molecules.append(info)
            error_stats['xyz_conversion'] += 1
            continue

        # 2) SMILES → RDKit Mol
        mol, mol_errs = create_molecule_from_smiles(d['smiles'], d['name'])
        if mol is None:
            info.update({'error_type': 'mol_creation', 'details': mol_errs,
                         'smiles': d['smiles']})
            failed_molecules.append(info)
            error_stats['mol_creation'] += 1
            continue

        # 3) RDKit 描述符 + 拓扑指纹
        if CONFIG['calc_rdkit']:
            try:
                row = []
                for n, func in rdkit_desc_list:
                    try:
                        v = func(mol)
                        row.append(float(v) if v is not None else np.nan)
                    except Exception:
                        row.append(np.nan)
                # 拓扑指纹
                fp = rdmolops.RDKFingerprint(
                    mol,
                    fpSize=CONFIG['rdkit_fp_size'],
                    minPath=CONFIG['rdkit_fp_minpath'],
                    maxPath=CONFIG['rdkit_fp_maxpath'],
                    nBitsPerHash=CONFIG['rdkit_fp_nBitsPerHash'])
                rdk_bits = list(fp)
                d['rdkit_descriptor'] = row + rdk_bits
                d['rdkit_f'] = (rdkit_desc_names +
                                [f'RDK_{i}' for i in range(len(rdk_bits))])
            except Exception as e:
                errs.append(f"RDKit: {e}")

        # 4) MACCS Keys
        if CONFIG['calc_maccs']:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                d['maccs_descriptor'] = list(fp)
                d['maccs_f'] = [f'MACCS_{i}' for i in range(len(fp))]
            except Exception as e:
                errs.append(f"MACCS: {e}")

        # 5) Morgan 指纹（ECFP / FCFP）
        if CONFIG['calc_morgan']:
            try:
                if CONFIG['morgan_use_counts']:
                    fp_obj = AllChem.GetMorganFingerprint(
                        mol, CONFIG['morgan_radius'],
                        useFeatures=CONFIG['morgan_use_features'])
                    # 稀疏 → 密集位向量（用 bit 位的 count clip 到 1）
                    arr = np.zeros(CONFIG['morgan_nbits'], dtype=np.int32)
                    for bit, cnt in fp_obj.GetNonzeroElements().items():
                        arr[bit % CONFIG['morgan_nbits']] += cnt
                    d['morgan_descriptor'] = arr.tolist()
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, CONFIG['morgan_radius'],
                        nBits=CONFIG['morgan_nbits'],
                        useFeatures=CONFIG['morgan_use_features'])
                    d['morgan_descriptor'] = list(fp)
                d['morgan_f'] = [f'Morgan_{i}' for i in range(CONFIG['morgan_nbits'])]
            except Exception as e:
                errs.append(f"Morgan: {e}")

        # 6) AtomPair 指纹
        if CONFIG['calc_atompair']:
            try:
                if CONFIG['atompair_use_counts']:
                    fp_obj = AllChem.GetAtomPairFingerprint(mol)
                    arr = np.zeros(CONFIG['atompair_nbits'], dtype=np.int32)
                    for bit, cnt in fp_obj.GetNonzeroElements().items():
                        arr[bit % CONFIG['atompair_nbits']] += cnt
                    d['atompair_descriptor'] = arr.tolist()
                else:
                    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                        mol, nBits=CONFIG['atompair_nbits'])
                    d['atompair_descriptor'] = list(fp)
                d['atompair_f'] = [f'AP_{i}' for i in range(CONFIG['atompair_nbits'])]
            except Exception as e:
                errs.append(f"AtomPair: {e}")

        # 7) Topological Torsion 指纹
        if CONFIG['calc_torsion']:
            try:
                if CONFIG['torsion_use_counts']:
                    fp_obj = AllChem.GetTopologicalTorsionFingerprint(mol)
                    arr = np.zeros(CONFIG['torsion_nbits'], dtype=np.int32)
                    for bit, cnt in fp_obj.GetNonzeroElements().items():
                        arr[bit % CONFIG['torsion_nbits']] += cnt
                    d['torsion_descriptor'] = arr.tolist()
                else:
                    fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(
                        mol, nBits=CONFIG['torsion_nbits'])
                    d['torsion_descriptor'] = list(fp)
                d['torsion_f'] = [f'TT_{i}' for i in range(CONFIG['torsion_nbits'])]
            except Exception as e:
                errs.append(f"Torsion: {e}")

        # 8) Avalon 指纹
        if CONFIG['calc_avalon']:
            try:
                vals, names_av = _get_avalon_fp(mol)
                d['avalon_descriptor'] = vals
                d['avalon_f'] = names_av
            except ImportError as e:
                errs.append(str(e))
            except Exception as e:
                errs.append(f"Avalon: {e}")

        # 9) Mordred
        if CONFIG['calc_mordred']:
            try:
                mordred_vals, mordred_names = calculate_mordred(mol)
                if mordred_vals:
                    d['mordred_descriptor'] = mordred_vals
                    d['mordred_f'] = mordred_names
                else:
                    errs.append("Mordred: 返回空结果")
            except Exception as e:
                errs.append(f"Mordred: {e}")

        # 10) 基础分子属性
        if CONFIG['calc_prop']:
            try:
                props = calculate_props(mol)
                d.update(props)
            except Exception as e:
                errs.append(f"Props: {e}")

        # 11) SOAP / ACSF / MBTR（需要 xyz 文件）
        if CONFIG['calc_soap'] or CONFIG['calc_acsf'] or CONFIG['calc_mbtr']:
            try:
                res, _ = calculate_ase_descriptors(xyz_path)
                if CONFIG['calc_soap'] and 'soap' in res:
                    d['soap_descriptor'], d['soap_f'] = res['soap']
                if CONFIG['calc_acsf'] and 'acsf' in res:
                    d['acsf_descriptor'], d['acsf_f'] = res['acsf']
                if CONFIG['calc_mbtr'] and 'mbtr' in res:
                    d['mbtr_descriptor'], d['mbtr_f'] = res['mbtr']
            except Exception as e:
                errs.append(f"DScribe: {e}")

        # ── 错误记录 ──────────────────────────────────────────────────────
        if errs:
            info.update({'error_type': 'descriptor_calculation',
                         'details': '; '.join(errs),
                         'smiles': d['smiles'],
                         'traceback': traceback.format_exc()})
            failed_molecules.append(info)
            error_stats['descriptor_calculation'] += len(errs)
            # 即使有部分描述符失败，仍保留该分子（只跳过完全失败的情况）
            # 若希望严格模式（任意失败则排除），取消下一行注释：
            # continue

        d_r.append(d)

    # ── 输出失败记录 ──────────────────────────────────────────────────────
    pd.DataFrame(failed_molecules,
                 columns=['name', 'error_type', 'details', 'smiles', 'traceback']
                 ).to_excel(CONFIG['failed_record'], index=False, engine='openpyxl')

    print(f"\n成功处理: {len(d_r)} / {len(data)}")
    print("失败统计:", dict(error_stats))
    print(f"失败记录 → {CONFIG['failed_record']}")

    # ── 保存结果 ──────────────────────────────────────────────────────────
    save_dict = {
        'successful':   d_r,
        'failed_count': len(failed_molecules),
        'error_stats':  dict(error_stats),
        'config':       CONFIG,
    }
    np.save(CONFIG['output_npy'], save_dict, allow_pickle=True)
    print(f"结果已保存 → {CONFIG['output_npy']}")

    # ── 皮尔逊相关性过滤（可选）────────────────────────────────────────
    if CONFIG.get('pearson_filter', False):
        print(f"\n[INFO] 启动皮尔逊相关性过滤（阈值 |r|>{CONFIG['pearson_threshold']}）...")
        try:
            # 导入同目录的 pearson_filter.py
            _desc_dir = os.path.dirname(os.path.abspath(__file__))
            import sys as _sys
            if _desc_dir not in _sys.path:
                _sys.path.insert(0, _desc_dir)
            from pearson_filter import run_pearson_filter

            _pf_cfg = {
                'input_npy':        CONFIG['output_npy'],
                'output_npy':       CONFIG.get('pearson_output_npy', ''),
                'report_xlsx':      CONFIG.get('pearson_report_xlsx', ''),
                'output_dir':       os.path.dirname(os.path.abspath(CONFIG['output_npy'])),
                'threshold':        CONFIG.get('pearson_threshold', 0.95),
                'descriptor_fields': [],   # 自动检测所有字段
                'gen_heatmap':      CONFIG.get('pearson_gen_heatmap', True),
                'heatmap_max_dim':  300,
                'mode':             'filter',
            }
            pf_result = run_pearson_filter(_pf_cfg)
            print(f"[INFO] 皮尔逊过滤完成：{pf_result['n_before']}维 → {pf_result['n_after']}维")
            print(f"[INFO] 过滤后文件 → {pf_result['output_npy']}")
        except Exception as _pf_e:
            print(f"[WARN] 皮尔逊过滤执行失败: {_pf_e}")
            import traceback as _tb
            _tb.print_exc()

    print("全部计算完成！")
