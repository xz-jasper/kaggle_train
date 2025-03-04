import matplotlib
matplotlib.use('Agg')  # 设置使用 Agg 后端
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("plot.png")  # 直接保存图片，不显示


if __name__ == '__main__':
    print("end")