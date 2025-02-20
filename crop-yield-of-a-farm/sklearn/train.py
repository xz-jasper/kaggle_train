import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

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

# 创建并训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 保存模型
joblib.dump(model, "linear_regression_model.pkl")
print("模型已保存为 'linear_regression_model.pkl'")
joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    print("train end")
