from sklearn.preprocessing import LabelEncoder

# 创建 LabelEncoder 对象
encoder = LabelEncoder()

# 示例数据（字符串类别）
fruits = ["apple", "banana", "cherry", "banana", "apple", "cherry"]

# 进行编码
encoded_labels = encoder.fit_transform(fruits)

print(encoded_labels)


if __name__ == '__main__':
    print("use label encoder")