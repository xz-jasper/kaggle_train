import akshare as ak
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# 获取股票数据（以贵州茅台 600519 为例，A 股采用新浪数据源）
stock_code = "600519"  # 贵州茅台
df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")  # 获取前复权数据

# 处理数据格式（Prophet 需要 `ds` 和 `y` 两列）
df = df[['日期', '收盘']].rename(columns={'日期': 'ds', '收盘': 'y'})
df['ds'] = pd.to_datetime(df['ds'])  # 确保日期格式正确

# 生成一个额外的预测因子（例如：随机生成一些“温度”数据）
np.random.seed(42)  # 保证结果可复现
temperature_data = np.random.normal(25, 5, len(df))  # 假设温度数据服从均值 25，标准差 5 的正态分布

# 将温度数据添加到 df 中
df['temperature'] = temperature_data

# 初始化 Prophet 模型，并添加温度作为额外的回归因子
model = Prophet()
model.add_regressor('temperature')  # 将温度列添加为回归因子

# 训练模型
model.fit(df)

# 生成未来 180 天的数据，确保传递的是 `periods` 作为关键字参数
future = model.make_future_dataframe(periods=180)  # 生成未来的日期数据
future['temperature'] = np.random.normal(25, 5, len(future))  # 为未来 180 天生成预测温度数据

# 进行预测
forecast = model.predict(future)

# 打印预测结果中的列
print(forecast.columns)

# 绘制预测结果
plt.figure(figsize=(12, 6))
model.plot(forecast)
plt.title(f"Stock Price Prediction for {stock_code}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.grid()
plt.show()






if __name__ == '__main__':
    print("end")