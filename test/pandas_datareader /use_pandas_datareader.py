import pandas_datareader as pdr
import datetime

# 定义时间范围
start = datetime.datetime(2023, 1, 1)
end = datetime.datetime(2024, 1, 1)

# 获取股票数据
df = pdr.DataReader("AAPL", "yahoo", start, end)

# 显示数据
print(df.head())


#不能运行，如果需要正常拉取股票信息，国外使用yfinace,国内还是需要使用akshare

if __name__ == '__main__':
    print("use pandas_datareader")