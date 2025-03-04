from datetime import datetime, timedelta

# 获取当前时间
now = datetime.now()

# 创建一个 timedelta 表示 3 天
delta = timedelta(days=3)

# 计算 3 天后的日期
future_date = now + delta
print("现在时间：", now)
print("3天后的时间：", future_date)


if __name__ == '__main__':
    print("use timedelta")