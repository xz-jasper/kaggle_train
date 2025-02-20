from flask import Flask, request, jsonify
import joblib
import numpy as np



scaler = joblib.load('scaler.pkl')
# 加载保存的模型
model = joblib.load("linear_regression_model.pkl")

# 创建 Flask 应用
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取数据
    data = request.json
    features = np.array(data['features']).reshape(1, -1)

    # 对数据进行归一化
    features_normalized = scaler.transform(features)
    # 进行预测
    prediction = model.predict(features_normalized)

    # 返回预测结果
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9066)
