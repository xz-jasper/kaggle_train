import akshare as ak

financial_data = ak.stock_yzxdr_em(symbol="600519")
print(financial_data.head())

if __name__ == '__main__':
    print("end")