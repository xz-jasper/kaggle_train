import akshare as ak

# 获取涨停股票池
zt_pool = ak.stock_zt_pool_em(date='20250212')

# 过滤出涨停股票
# zt_stocks = zt_pool[zt_pool['涨跌幅'] == '涨停']

# 查看涨停股票
print(zt_pool)


if __name__ == '__main__':
    print("end")