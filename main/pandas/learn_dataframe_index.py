import pandas as pd

# 创建一个示例 DataFrame，假设日期是索引
data = {'Value': [10, 15, 12, 18, 25]}
dates = pd.date_range('2023-01-01', periods=5, freq='D')  # 生成 5 天的日期

# 创建 DataFrame，将日期作为索引
df = pd.DataFrame(data, index=dates)

# 显示原始数据
print("Original DataFrame:")
print(df)

# Step 1: Copy the index to a new column named 'Date'
df['Date'] = df.index

# Step 2: Reset the index, dropping the current index and replacing it with a default integer-based index
df.reset_index(drop=True, inplace=True)

# 显示处理后的 DataFrame
print("\nAfter copying index to 'Date' and resetting index:")
print(df)


if __name__ == '__main__':
    print("end")