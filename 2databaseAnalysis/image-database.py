import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def extract_target(name, target_list):
    """从name中提取匹配的目标列值（支持部分匹配）"""
    for target in target_list:
        if str(target) in name:  # 转换为字符串确保匹配
            return target
    return None

def set_color_scheme(scheme, n_colors):
    """设置颜色方案"""
    if scheme == 'pastel':
        cmap = plt.get_cmap('Pastel1')
    elif scheme == 'bright':
        cmap = plt.get_cmap('Set3')
    elif scheme == 'grayscale':
        cmap = LinearSegmentedColormap.from_list(
            'grayscale', ['#f0f0f0', '#636363'], N=n_colors)
    elif scheme == 'coolwarm':
        cmap = plt.get_cmap('coolwarm')
    else:
        cmap = plt.get_cmap('tab20')
    
    return [cmap(i % cmap.N) for i in range(n_colors)]

def create_boxplot(data, labels, target_column, 
                 colors, figsize, rotation, 
                 add_stats, show_outliers):
    """创建箱线图"""
    plt.figure(figsize=figsize)
    boxprops = dict(linestyle='-', linewidth=1.5, color='black')
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    whiskerprops = dict(linestyle='--', linewidth=1.5, color='black')
    capprops = dict(linestyle='-', linewidth=1.5, color='black')
    
    # 绘制箱线图
    bp = plt.boxplot(data, labels=labels, patch_artist=True,
                   showfliers=show_outliers,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   whiskerprops=whiskerprops,
                   capprops=capprops)
    
    # 设置箱体颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 添加统计信息
    if add_stats:
        for i, d in enumerate(data):
            stats_text = (f'n={len(d)}\n'
                        f'μ={np.mean(d):.2f}\n'
                        f'σ={np.std(d):.2f}')
            plt.text(i+1, np.max(np.concatenate(data))*1.05, 
                   stats_text, ha='center', va='bottom', 
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xticks(rotation=rotation)
    plt.xlabel(target_column, fontsize=12, labelpad=10)
    plt.ylabel('deltaG1', fontsize=12, labelpad=10)
    plt.title(f'Boxplot of deltaG1 by {target_column}\n', 
             fontsize=14, pad=20)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'enhanced_boxplot_by_{target_column}.png', 
              dpi=300, bbox_inches='tight')
    plt.show()  # 显示图片

def create_violinplot(data, labels, target_column, 
                    colors, figsize, rotation, 
                    add_stats, bandwidth):
    """创建小提琴图"""
    plt.figure(figsize=figsize)
    
    # 绘制小提琴图
    vp = plt.violinplot(data, showmeans=True, showmedians=True,
                      widths=0.8, bw_method=bandwidth)
    
    # 设置小提琴图颜色
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
    
    # 设置统计线样式
    vp['cmeans'].set_color('green')
    vp['cmeans'].set_linewidth(2)
    vp['cmedians'].set_color('red')
    vp['cmedians'].set_linewidth(2)
    vp['cbars'].set_color('black')
    vp['cbars'].set_linewidth(1)
    
    # 添加标签和标题
    plt.xticks(range(1, len(labels)+1), labels)  # 修复了这里的语法错误
    plt.xticks(rotation=rotation)
    plt.xlabel(target_column, fontsize=12, labelpad=10)
    plt.ylabel('deltaG1', fontsize=12, labelpad=10)
    plt.title(f'Violin plot of deltaG1 by {target_column}\n', 
             fontsize=14, pad=20)
    
    # 添加统计信息
    if add_stats:
        for i, d in enumerate(data):
            stats_text = (f'n={len(d)}\n'
                        f'μ={np.mean(d):.2f}\n'
                        f'σ={np.std(d):.2f}')
            plt.text(i+1, np.max(np.concatenate(data))*1.05,
                   stats_text, ha='center', va='bottom',
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'enhanced_violinplot_by_{target_column}.png',
              dpi=300, bbox_inches='tight')
    plt.show()  # 显示图片

def plot_graphs(database_df, example_df, target_column='coor', plot_type='both',
               color_scheme='pastel', figsize=(12, 7), rotation=45,
               add_stats=True, show_outliers=True, bandwidth=0.5):
    """
    绘制箱线图和小提琴图的增强函数
    
    参数:
    ----------
    database_df : DataFrame
        database.xlsx的数据
    example_df : DataFrame
        example.xlsx的数据
    target_column : str
        要分组的目标列 ('coor', 'ligand' 或 'group')
    plot_type : str
        绘图类型 ('box', 'violin', 'both')
    color_scheme : str
        颜色方案 ('pastel', 'bright', 'grayscale', 'coolwarm')
    figsize : tuple
        图像大小 (宽, 高)
    rotation : int
        x轴标签旋转角度
    add_stats : bool
        是否添加统计注释
    show_outliers : bool
        是否显示离群点
    bandwidth : float
        小提琴图的平滑带宽 (仅对violin plot有效)
    """
    # 1. 获取目标列的唯一值列表 (去除空值)
    target_values = database_df[target_column].dropna().unique()
    
    # 2. 在example中匹配目标列的值
    example_df[target_column] = example_df['name'].apply(
        lambda x: extract_target(x, target_values))
    
    # 3. 准备绘图数据
    plot_data = []
    labels = []
    for value in target_values:
        data = example_df[example_df[target_column] == value]['D_index'].dropna()
        if len(data) > 0:
            plot_data.append(data)
            labels.append(value)
    
    if not plot_data:
        print("没有找到匹配的数据!")
        return
    
    # 4. 设置颜色方案
    colors = set_color_scheme(color_scheme, len(labels))
    
    # 5. 绘图功能
    if plot_type in ['box', 'both']:
        create_boxplot(plot_data, labels, target_column,
                      colors, figsize, rotation,
                      add_stats, show_outliers)
    
    if plot_type in ['violin', 'both']:
        create_violinplot(plot_data, labels, target_column,
                        colors, figsize, rotation,
                        add_stats, bandwidth)

if __name__ == "__main__":
    # 读取数据文件
    try:
        database_df = pd.read_excel('database.xlsx', sheet_name='Sheet1')
        example_df = pd.read_excel('F1.xlsx', sheet_name='Sheet1')
    except FileNotFoundError as e:
        print(f"文件读取错误: {e}")
        exit(1)
    
    # 示例1: 绘制coor的两种图形，使用默认参数
    plot_graphs(database_df, example_df, target_column='Donor')
    
    # # 示例2: 绘制ligand的小提琴图，自定义参数
    # plot_graphs(
    #     database_df=database_df,
    #     example_df=example_df,
    #     target_column='ligand',
    #     plot_type='violin',
    #     color_scheme='coolwarm',
    #     figsize=(14, 8),
    #     rotation=90,
    #     bandwidth=0.3
    # )
    
    # # 示例3: 绘制group的箱线图，不显示离群点
    # plot_graphs(
    #     database_df=database_df,
    #     example_df=example_df,
    #     target_column='group',
    #     plot_type='box',
    #     show_outliers=False,
    #     rotation=60
    # )