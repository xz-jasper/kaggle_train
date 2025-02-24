import akshare as ak
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 获取股票数据（以平安银行为例，股票代码：000001）
stock_data = ak.stock_zh_a_hist(
    symbol="000001", period="daily", start_date="20100101", end_date="20250101"
)

# 打印数据的前几行，查看实际的列名
print(stock_data.head())  # 查看数据的前几行
print(stock_data.columns)  # 查看数据的列名

# 假设返回的列名为 ['日期', '股票代码', '开盘', '收盘', ...]，选择 '日期' 和 '收盘' 列
stock_data = stock_data[["日期", "收盘"]]  # 使用实际的列名
stock_data["日期"] = pd.to_datetime(stock_data["日期"])  # 将日期列转换为日期格式
stock_data.set_index("日期", inplace=True)  # 设置日期为索引

# 使用 MinMaxScaler 归一化收盘价
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[["收盘"]])


# 创建数据集：构建 X（特征）和 y（标签）
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step), 0])  # 使用过去 60 天的数据来预测
        y.append(data[i + time_step, 0])  # 使用第 60 天后的收盘价作为标签
    return np.array(X), np.array(y)


time_step = 60  # 过去 60 天的数据
X, y = create_dataset(scaled_data, time_step)

# 将 X 转换为三维数据 [样本数, 时间步长, 特征数]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 转换为 PyTorch tensor
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


# 创建模型、定义损失函数和优化器
model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 获取模型的预测值
    y_pred = model(X_train)

    # 计算损失
    loss = criterion(y_pred, y_train.view(-1, 1))

    # 反向传播
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# 使用测试集进行预测
model.eval()  # 将模型设置为评估模式，这样可以避免训练过程中某些行为（如dropout）的影响
with torch.no_grad():  # 在预测时不计算梯度
    y_pred_test = model(X_test)  # 模型对测试集数据进行预测

# 反归一化预测结果
y_pred_test = y_pred_test.numpy()  # 将PyTorch tensor转换为Numpy数组
y_test = y_test.numpy()  # 同样将真实标签转换为Numpy数组

# 使用 scaler 对预测值和真实值进行反归一化，使得它们回到原始股价的范围
y_pred_test = scaler.inverse_transform(y_pred_test)  # 反归一化预测值
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # 反归一化真实值

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(y_test, color="blue", label="Actual Stock Price")  # 绘制真实股价
plt.plot(y_pred_test, color="red", label="Predicted Stock Price")  # 绘制预测股价
plt.title("Stock Price Prediction (LSTM) with PyTorch")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# 预测下一天的股价
# 假设你已经有了最新的 60 天的股价数据
last_60_days = stock_data[["收盘"]].tail(60)  # 获取最近 60 天的数据
scaled_last_60_days = scaler.transform(last_60_days)  # 对这些数据进行归一化

# 创建输入数据
X_new = []
X_new.append(scaled_last_60_days)

# 转换为符合 LSTM 输入的三维数组
X_new = np.array(X_new)
X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

# 使用训练好的模型进行预测
model.eval()  # 切换到评估模式
with torch.no_grad():
    next_day_pred = model(torch.Tensor(X_new))  # 预测下一天

# 反归一化预测结果
next_day_pred = next_day_pred.numpy()
next_day_pred = scaler.inverse_transform(next_day_pred)

print(f"Predicted stock price for the next day: {next_day_pred[0][0]}")



if __name__ == '__main__':
    print("end")