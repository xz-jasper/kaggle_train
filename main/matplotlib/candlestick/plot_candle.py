import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc

# 从 akshare 拉取招商银行的历史股价数据
df = ak.stock_zh_a_hist(symbol="600036", period="daily", adjust="qfq")

# 选择需要的列，日期、开盘、最高、最低、收盘价
df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]

# 修改列名为 Matplotlib 所需的格式
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# 将 'Date' 列转换为 matplotlib 可识别的日期格式
matplotlib_date = mdates.date2num(pd.to_datetime(df['Date']))

# 创建 OHLC 数据（按日期、开盘、最高、最低、收盘的顺序）
ohlc = np.vstack((matplotlib_date, df['Open'], df['High'], df['Low'], df['Close'])).T

# 创建一个新的绘图
plt.figure(figsize=(15, 6))
ax = plt.subplot()

# 绘制 K 线图
candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')

# 设置日期格式
ax.xaxis_date()

# 添加标题和标签
plt.title('招商银行 K线图')
plt.xlabel('Date')
plt.ylabel('Price')

# 旋转 x 轴标签，避免重叠
plt.xticks(rotation=45)

# 显示网格
plt.grid(True)

# 显示图表
plt.show()


if __name__ == '__main__':
    print("end")