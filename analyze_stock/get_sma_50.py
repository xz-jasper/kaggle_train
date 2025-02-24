import pandas as pd
import matplotlib.pyplot as plt

# 示例数据
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
prices = [100 + i for i in range(100)]
stock_df = pd.DataFrame({"日期": dates, "收盘": prices})

# 计算 50 日 SMA
stock_df['sma_50'] = stock_df['收盘'].rolling(window=50).mean()

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(stock_df['日期'], stock_df['收盘'], label="收盘价")
plt.plot(stock_df['日期'], stock_df['sma_50'], label="50 日 SMA")
plt.legend()
plt.title("50 日简单移动平均线")
plt.xlabel("日期")
plt.ylabel("价格")
plt.show()


if __name__ == '__main__':
    print("end")