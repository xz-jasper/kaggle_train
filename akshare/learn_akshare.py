import akshare as ak

# 获取中国股票历史数据（以平安银行为例）
stock_data = ak.stock_zh_a_hist(symbol="000002", period="daily", start_date="20240101", end_date="20250201")
print(stock_data)


if __name__ == '__main__':
    print("end")