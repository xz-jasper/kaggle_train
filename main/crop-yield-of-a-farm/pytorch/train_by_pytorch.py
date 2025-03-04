import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

import os

print(os.getcwd())  # 打印当前工作目录
os.chdir('pytorch')
print(os.getcwd())  # 打印当前工作目录
# 假设数据已加载为 X 和 y
data = pd.read_csv("../crop_yield_data.csv")

# 数据拆分
# Feature-target split
X = data.drop(columns=['crop_yield'])
y = data['crop_yield']

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 假设 X_train_scaled 和 y_train 已经经过标准化和处理
# PyTorch 需要 Tensor 格式的数据
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# 定义一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输入维度和输出维度

    def forward(self, x):
        return self.linear(x)

# 初始化模型
input_dim = X_train_tensor.shape[1]
model = LinearRegressionModel(input_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 模型训练
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "linear_regression_model.pth")
print("模型已保存为 'linear_regression_model.pth'")

# 保存标准化器
joblib.dump(scaler, 'scaler.pkl')
print("标准化器已保存为 'scaler.pkl'")

if __name__ == '__main__':

    print("train end")
