import akshare as ak
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 获取美的集团的股价数据（历史数据）
stock_data = ak.stock_zh_a_hist(
    symbol="000333", period="daily", start_date="20100101", end_date="20250101"
)
# 打印股价数据查看列名
print(stock_data.head())

# 获取美的集团的基本面数据（财务数据）
fundamentals_data = ak.stock_a_indicator_lg(symbol="000333")

# 查看数据
print(fundamentals_data.head())


stock_data = stock_data[["日期", "收盘"]]
stock_data["日期"] = pd.to_datetime(stock_data["日期"])
stock_data.set_index("日期", inplace=True)

# 提取基本面数据的所需列
fundamentals_data = fundamentals_data[
    ["trade_date", "pe", "pe_ttm", "pb", "dv_ttm", "ps", "ps_ttm", "total_mv"]
]
fundamentals_data["trade_date"] = pd.to_datetime(fundamentals_data["trade_date"])
fundamentals_data.set_index("trade_date", inplace=True)

# 合并股价数据和基本面数据，按日期对齐
merged_data = pd.merge(
    stock_data, fundamentals_data, left_index=True, right_index=True, how="inner"
)

# 计算投资回报率（ROI）
merged_data["ROI"] = (
    merged_data["收盘"].pct_change(periods=30) * 100
)  # 30天的投资回报率

# 删除缺失值
merged_data.dropna(inplace=True)

# 选择特征和目标
features = merged_data[
    ["收盘", "pe", "pe_ttm", "pb", "dv_ttm", "ps", "ps_ttm", "total_mv", "ROI"]
]
target = (merged_data["收盘"].shift(-1) > merged_data["收盘"]).astype(
    int
)  # 使用股价上涨（1）和下跌（0）作为目标变量

# 打印数据查看
print(merged_data.head())


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 对特征进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, target, test_size=0.2, shuffle=False
)

# 将数据转换为 PyTorch 的 Tensor
import torch

# 调整输入数据形状
X_train_tensor = torch.Tensor(X_train)  # 转换为tensor
X_test_tensor = torch.Tensor(X_test)  # 转换为tensor
y_train_tensor = torch.Tensor(y_train.values)  # 目标变量
y_test_tensor = torch.Tensor(y_test.values)  # 目标变量

# 确保数据是三维的 [batch_size, sequence_length, input_size]
X_train_tensor = X_train_tensor.unsqueeze(
    1
)  # 变为 [batch_size, sequence_length=1, features]
X_test_tensor = X_test_tensor.unsqueeze(
    1
)  # 变为 [batch_size, sequence_length=1, features]


# LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


# 初始化 LSTM 模型
input_size = X_train_tensor.shape[2]  # 特征数量
hidden_layer_size = 50  # 隐藏层单元数量
output_size = 1  # 二分类问题：预测涨跌（0 或 1）

model = LSTMModel(
    input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size
)

# 设置损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 使用二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 获取预测结果
    y_pred_train = model(X_train_tensor)

    # 计算损失
    loss = criterion(y_pred_train.squeeze(), y_train_tensor)

    # 反向传播
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

# 在测试集上进行预测
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)

# 转换为 0 或 1
y_pred_test = torch.round(torch.sigmoid(y_pred_test))

# 评估模型准确率
accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_test.numpy())
print(f"Accuracy: {accuracy * 100:.2f}%")


# 可视化预测与实际股价
plt.figure(figsize=(10, 6))
plt.plot(y_test_tensor.numpy(), label="Actual")
plt.plot(y_pred_test, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Stock Price Prediction with LSTM (ROI + Fundamentals)")
plt.show()


if __name__ == "__main__":
    print("end")
