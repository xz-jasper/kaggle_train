import akshare as ak

# 获取中国股票历史数据（以平安银行为例）
stock_data = ak.stock_zh_a_hist(symbol="000002", period="daily", start_date="20240101", end_date="20250201")
print(stock_data)

# symbol：股票代码（例如，000001 是平安银行的股票代码）。
# period：数据的时间粒度，常见的有 "daily"（日线）、"weekly"（周线）、"monthly"（月线）等。
# start_date 和 end_date：数据的时间范围，格式为 YYYYMMDD。

# 保存数据到 CSV 文件
stock_data.to_csv("stock_data.csv", index=False)
if __name__ == '__main__':
    print("end")