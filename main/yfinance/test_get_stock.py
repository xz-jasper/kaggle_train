import yfinance as yf

# 获取微软的股票数据
stock_data = yf.download(["MSFT"], start="2023-01-01", end="2025-01-01")

# 打印前几行数据
print(stock_data.head())


if __name__ == '__main__':
    print("end")