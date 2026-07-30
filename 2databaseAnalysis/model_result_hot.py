import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_and_preprocess_data(file_path, custom_order=False, x_order=None, y_order=None):
    """加载并预处理数据
    Args:
        file_path: Excel文件路径
        custom_order: 是否使用自定义顺序 (True/False)
        x_order: 横坐标顺序列表 (仅在custom_order=True时使用)
        y_order: 纵坐标顺序列表 (仅在custom_order=True时使用)
    """
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df = df[["Descript", "Model", "R2"]].copy()
    
    # 处理SN特殊行
    df_sn = df[df["Model"] == "SN"]
    df_main = df[df["Model"] != "SN"]
    
    # 创建透视表并填充NaN
    pivot_main = df_main.pivot(index="Descript", columns="Model", values="R2").fillna(0)
    pivot_sn = df_sn.pivot(index="Descript", columns="Model", values="R2").fillna(0)
    
    # 只有当custom_order为True时才应用自定义顺序
    if custom_order:
        if x_order is not None:
            # 保留存在于数据中的列
            x_order = [x for x in x_order if x in pivot_main.columns]
            pivot_main = pivot_main[x_order]
        
        if y_order is not None:
            # 保留存在于数据中的行
            y_order = [y for y in y_order if y in pivot_main.index]
            pivot_main = pivot_main.loc[y_order]
    
    return pivot_main, pivot_sn

def create_custom_colormap():
    """创建蓝-白-红发散色阶"""
    colors = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
    return LinearSegmentedColormap.from_list("custom_diverging", colors)

def plot_heatmap(pivot_main, pivot_sn, output_file="model_r2_heatmap.png", x_label="Model", y_label="Descript"):
    """绘制热图"""
    # 创建图形时预留更多空间
    fig, ax = plt.subplots(figsize=(14, 8))  # 调整尺寸适应横向布局
    
    # 合并主数据和SN数据
    combined_data = pd.concat([pivot_main, pivot_sn])
    cmap = create_custom_colormap()
    
    # 绘制主热图
    im = ax.imshow(
        combined_data.values,
        cmap=cmap,
        vmin=-0.3,
        vmax=1.0,
        aspect="auto",
        origin="upper"
    )
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, label="R² Score")
    
    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(combined_data.columns)))
    ax.set_yticks(np.arange(len(combined_data.index)))
    ax.set_xticklabels(combined_data.columns, rotation=45, ha="right")
    ax.set_yticklabels(combined_data.index)
    
    # 添加单元格数值标注
    for i in range(len(combined_data.index)):
        for j in range(len(combined_data.columns)):
            val = combined_data.iloc[i, j]
            if val != 0:  # 只标注非零值
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="black" if abs(val - 0.5) < 0.3 else "white",
                    fontsize=8
                )
    
    # 标记SN列（如果有）
    if not pivot_sn.empty:
        sn_col = combined_data.columns.get_loc("SN")
        ax.add_patch(plt.Rectangle(
            (sn_col - 0.5, -0.5), 
            1, 
            len(combined_data.index), 
            fill=False, 
            edgecolor="purple", 
            lw=2,
            linestyle="--"
        ))
    
    # 调整布局
    plt.subplots_adjust(
        bottom=0.25,  # 增加底部边距
        top=0.95,     # 增加顶部边距
        left=0.15,    # 增加左侧边距
        right=0.9      # 增加右侧边距
    )
    
    # 美化图表
    ax.set_title("Machine Learning Model Performance by Descript", pad=20)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    
    # 保存和显示
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # 主程序
    input_file = "summary.xlsx"
    output_file = "model_r2_heatmap.png"
    
    # 控制是否使用自定义顺序
    USE_CUSTOM_ORDER = False  # 设置为False则不使用自定义顺序
    
    # 示例自定义顺序 (当USE_CUSTOM_ORDER为True时使用)
    custom_x_order = ["Ridge", "Lasso", "ElasticNet","DecisionTree","RandomForest","GradientBoosting","SVR","KNeighbors","MLP"]           # 横坐标顺序
    custom_y_order = ["delta_G1-maccs", "delta_G2-maccs", "delta_G3-maccs", "delta_G4-maccs","delta_G1-morgan", "delta_G2-morgan", "delta_G3-morgan", "delta_G4-morgan","delta_G1-rdkit", "delta_G2-rdkit", "delta_G3-rdkit", "delta_G4-rdkit","delta_G1-mrm", "delta_G2-mrm", "delta_G3-mrm", "delta_G4-mrm","delta_G1-qc", "delta_G2-qc", "delta_G3-qc", "delta_G4-qc"]  # 纵坐标顺序
    
    try:
        pivot_main, pivot_sn = load_and_preprocess_data(
            input_file, 
            custom_order=USE_CUSTOM_ORDER,
            x_order=custom_x_order if USE_CUSTOM_ORDER else None,
            y_order=custom_y_order if USE_CUSTOM_ORDER else None
        )
        plot_heatmap(pivot_main, pivot_sn, output_file, 
                    x_label="Machine Learning Models",  # 可自定义横坐标标签
                    y_label="Molecular Descripts")    # 可自定义纵坐标标签
        print(f"热图已保存至 {output_file}")
    except Exception as e:
        print(f"错误发生: {str(e)}")