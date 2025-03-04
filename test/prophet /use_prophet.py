import akshare as ak
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 获取股票数据（以贵州茅台 600519 为例，A 股采用新浪数据源）
stock_code = "600519"  # 贵州茅台
df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")  # 获取前复权数据

# 处理数据格式（Prophet 需要 `ds` 和 `y` 两列）
df = df[['日期', '收盘']].rename(columns={'日期': 'ds', '收盘': 'y'})
df['ds'] = pd.to_datetime(df['ds'])  # 确保日期格式正确

# 初始化 Prophet 模型
model = Prophet()
model.fit(df)

# 生成未来 180 天的数据
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

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
    print("use prophet predict")