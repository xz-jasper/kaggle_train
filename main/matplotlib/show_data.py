import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show(data: pd.DataFrame, precision=2, cmap="Greens", axix=0):
    df = data.copy()
    # 样式化数据，保留精度
    df = df.style.format(precision=precision)

    # 获取数值数据，用于背景渐变计算
    data_values = df.data.values

    # 创建渐变背景
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(data_values.min(), data_values.max())  # 标准化数据以适配颜色
    colors = cmap(norm(data_values))  # 计算每个单元格的渐变颜色

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")  # 不显示坐标轴

    # 绘制表格并应用渐变背景
    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f5f5f5"] * len(data.columns),
    )

    # 设置每个单元格的背景颜色
    for i, key in enumerate(table.get_celld().keys()):
        cell = table[key]
        row, col = key
        # 排除表头，设置内容区域的背景颜色
        if row > 0:  # 避免覆盖列标签
            cell.set_facecolor(colors[row - 1, col])  # 应用计算的颜色
            cell.set_fontsize(10)

    plt.show()


# 示例数据
data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]})

# 调用 show 方法显示
show(data, precision=2, cmap="Greens")


if __name__ == '__main__':
    print("end")