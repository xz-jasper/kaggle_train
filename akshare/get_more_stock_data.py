import akshare as ak

stock_codes = ['000001', '600519', '000002']  # 可以填写多个股票代码
stock_data_list = []

for code in stock_codes:
    stock_data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20220101", end_date="20250201")
    stock_data['code'] = code  # 添加股票代码列
    stock_data_list.append(stock_data)

# 合并所有股票数据
import pandas as pd
all_stock_data = pd.concat(stock_data_list, ignore_index=True)

# 查看合并后的数据
print(all_stock_data.head())


if __name__ == '__main__':
    print("end")