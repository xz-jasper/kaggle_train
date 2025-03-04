from sklearn.metrics import mean_absolute_error, mean_squared_error

# 示例数据：真实值和预测值
y_true = [3, -0.5, 2, 7]  # 真实值
y_pred = [2.5, 0.0, 2, 8]  # 预测值

# 计算 MAE（Mean Absolute Error）
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# 计算 MSE（Mean Squared Error）
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse}")


if __name__ == '__main__':
    print("mae mse")