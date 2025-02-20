import tushare as ts

# 初始化并设置 Token（需要在 TuShare 注册获取 Token）
ts.set_token('0e2617b5080235a1bb1bf953f4ad6132ff58c74a7f864ac6bbb572a6')
pro = ts.pro_api()

# 获取某只股票的历史数据（以AAPL为例）
df = pro.daily(ts_code='000002.SZ', start_date='20240101', end_date='20250201')
print(df)



if __name__ == '__main__':
    print("end")
