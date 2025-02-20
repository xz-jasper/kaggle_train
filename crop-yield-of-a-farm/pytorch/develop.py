from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import numpy as np

# 定义线性回归模型结构
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# 加载模型和标准化器
model_path = "linear_regression_model.pth"
scaler_path = "scaler.pkl"

scaler = joblib.load(scaler_path)
print("标准化器已加载")

# 初始化模型
input_dim = 5  # 假设特征数量为 4，需根据训练时的实际值设置
model = LinearRegressionModel(input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置模型为评估模式
print("模型已加载")

# 初始化 Flask 应用
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 从请求中获取数据
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        # 数据标准化
        features_scaled = scaler.transform(features)

        # 转为 PyTorch 张量
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # 进行预测
        with torch.no_grad():
            prediction = model(features_tensor).item()

        # 返回预测结果
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9066)
