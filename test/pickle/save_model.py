from sklearn.ensemble import RandomForestClassifier
import pickle

# 训练一个简单的模型
model = RandomForestClassifier(n_estimators=10)
model.fit([[0, 0], [1, 1]], [0, 1])

# 保存模型
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 加载模型
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print(loaded_model.predict([[0, 0]]))  # 输出 [0]


if __name__ == '__main__':
    print("end")