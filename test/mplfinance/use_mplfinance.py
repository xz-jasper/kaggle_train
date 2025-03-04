import akshare as ak
import mplfinance as mpf
import pandas as pd

# 获取招商银行的股票数据，假设获取的是日线数据
df = ak.stock_zh_a_hist(symbol="600036", period="daily", adjust="qfq")

# 修改列名为mplfinance所需的英文列名
df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]  # 选取必要的列
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']  # 重命名列

# 转换日期列为datetime类型，并设置为索引
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 使用mplfinance绘制K线图
mpf.plot(df, type='candle', style='charles', volume=True, title="招商银行 K线图", ylabel='价格', ylabel_lower='成交量')


if __name__ == '__main__':
    print("end")