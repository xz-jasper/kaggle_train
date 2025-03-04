import joblib
from sklearn.ensemble import RandomForestClassifier


# 训练一个简单的模型
model = RandomForestClassifier(n_estimators=10)
model.fit([[0, 0], [1, 1]], [0, 1])

# 保存
joblib.dump(model, "model.joblib")

# 加载
loaded_model = joblib.load("model.joblib")


if __name__ == '__main__':
    print("joblib save ")