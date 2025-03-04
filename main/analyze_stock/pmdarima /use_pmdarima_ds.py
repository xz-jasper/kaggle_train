import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 忽略警告
warnings.filterwarnings('ignore')

# 设置全局字体（macOS 示例，Windows 可用 'Microsoft YaHei'）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 获取招商银行股票数据
stock_df = ak.stock_zh_a_hist(symbol="600036", period="daily", adjust="qfq")

# 重命名列名，确保与 ARIMA 训练一致
stock_df.rename(columns={"日期": "Date", "收盘": "Close"}, inplace=True)

# 转换 Date 列为 datetime 格式，并设为索引，按时间升序排列
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
stock_df.set_index("Date", inplace=True)
stock_df = stock_df.sort_index()

# 只保留 Close 列，并移除空值
df_close = stock_df["Close"].dropna()

# 设置索引频率为工作日（Business day），并填充因非交易日导致的缺失值
df_close = df_close.asfreq('B')
df_close = df_close.fillna(method='ffill')
df_close = df_close.replace([np.inf, -np.inf], np.nan).dropna()

# 对原始数据进行ADF检验
adf_result = adfuller(df_close)
print(f'ADF Statistic (原始数据): {adf_result[0]}')
print(f'p-value (原始数据): {adf_result[1]}')

# 如果 p-value > 0.05，则数据非平稳，需要进行差分处理
if adf_result[1] > 0.05:
    print("原始数据非平稳，进行一次差分处理...")
    df_diff = df_close.diff().dropna()
    diff_applied = True
else:
    df_diff = df_close.copy()
    diff_applied = False

# 对差分后的数据再进行ADF检验（可选）
adf_result_diff = adfuller(df_diff)
print(f'ADF Statistic (差分后): {adf_result_diff[0]}')
print(f'p-value (差分后): {adf_result_diff[1]}')

# 使用 auto_arima 在差分后的数据上建模，此时数据已平稳，设置 d=0
model_auto_arima = auto_arima(
    df_diff,
    start_p=0, start_q=0,
    max_p=5, max_q=5,  # 增加 p 和 q 的最大值
    d=0,              # 差分次数已在外部处理
    seasonal=False,   # 非季节性数据
    trace=True,       # 打印模型搜索过程
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# 打印自动选择的 ARIMA 模型摘要
print(model_auto_arima.summary())

# 预测未来 30 个交易日的数据（预测结果为差分数据的预测）
n_forecast = 30
forecast_diff, conf_int = model_auto_arima.predict(
    n_periods=n_forecast, return_conf_int=True
)
print("差分数据预测结果：")
print(forecast_diff)

# 生成预测日期序列，使用工作日频率
forecast_dates = pd.date_range(start=df_close.index[-1] + pd.Timedelta(days=1),
                               periods=n_forecast, freq='B')
forecast_series = pd.Series(forecast_diff, index=forecast_dates)

# 反差分：将预测的差分结果累加还原到原始尺度
# 使用原始数据最后一个值作为基准
if diff_applied:
    forecast_series = forecast_series.cumsum() + df_close.iloc[-1]

# 打印最终预测结果
print("预测值（原始尺度）：")
print(forecast_series)

# 计算预测误差（使用历史数据的一部分作为测试集）
train_size = int(len(df_close) * 0.8)
train, test = df_close[:train_size], df_close[train_size:]

# 在训练集上重新训练模型
model_train = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=0,
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# 预测测试集
forecast_test = model_train.predict(n_periods=len(test))

# 计算误差
mae = mean_absolute_error(test, forecast_test)
rmse = np.sqrt(mean_squared_error(test, forecast_test))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 绘制历史数据和预测数据
plt.figure(figsize=(12, 6))
plt.plot(df_close, label='历史数据', color='green')
plt.plot(forecast_series, label='预测数据', color='red')
plt.fill_between(forecast_dates,
                 conf_int[:, 0] + df_close.iloc[-1],
                 conf_int[:, 1] + df_close.iloc[-1],
                 color='pink', alpha=0.3, label='置信区间')
plt.title('招商银行股票预测')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.legend()
plt.show()

if __name__ == '__main__':
    print("ARIMA 预测结束")