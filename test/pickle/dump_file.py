import pickle

# 要保存的数据（任意 Python 对象）
data = {"name": "Alice", "age": 25, "city": "New York"}

# 以二进制写入模式（wb）打开文件
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)  # 序列化并写入文件

print("数据已保存！")


if __name__ == '__main__':
    print("end")