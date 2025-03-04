import pickle
#数据序列化工具


# 以二进制读取模式（rb）打开文件
with open("data.pkl", "rb") as file:
    loaded_data = pickle.load(file)  # 反序列化并加载数据

print("加载的数据：", loaded_data)


if __name__ == '__main__':
    print("load file")

