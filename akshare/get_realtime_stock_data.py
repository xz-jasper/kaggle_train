import akshare as ak
# 获取实时股票数据
stock_realtime_data = ak.stock_zh_a_spot()

# 显示前几行数据
print(stock_realtime_data.head())

if __name__ == '__main__':
    print("end")