import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 获取股票数据（例如平安银行，股票代码：000001）
stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20100101", end_date="20250101")

# 选择需要的列：日期和收盘价
stock_data = stock_data[['日期', '收盘']]
stock_data['日期'] = pd.to_datetime(stock_data['日期'])  # 转换为日期格式
stock_data.set_index('日期', inplace=True)  # 将日期列设置为索引

# 计算投资回报率 (ROI)
def calculate_roi(df, period=30):
    df['ROI'] = df['收盘'].pct_change(periods=period) * 100  # 计算30天的投资回报率（百分比）
    return df

# 计算过去 30 天的 ROI
stock_data = calculate_roi(stock_data, period=30)

# 删除空值（可能存在NaN值，因为是基于前期数据计算的）
stock_data.dropna(inplace=True)

# 创建目标变量：如果明天的股价上涨，则为1，否则为0
stock_data['Target'] = np.where(stock_data['收盘'].shift(-1) > stock_data['收盘'], 1, 0)

# 特征列：ROI 和过去几天的股价变化
features = stock_data[['收盘', 'ROI']]  # 可以添加更多技术指标如SMA, RSI等

# 特征归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# 目标列
target = stock_data['Target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, shuffle=False)

# 使用随机森林分类器训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在测试集上预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
plt.title('Stock Price Prediction with ROI as Feature')
plt.legend()
plt.show()


if __name__ == '__main__':
    print("end")