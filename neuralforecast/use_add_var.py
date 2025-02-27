import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx

# 1. 创建历史数据（训练集）
df = pd.DataFrame({
    'unique_id': ['store_1'] * 10,  # 只有一个时间序列
    'ds': pd.date_range(start='2023-01-01', periods=10, freq='D'),  # 过去 10 天的日期
    'y': [100, 120, 130, 125, 140, 160, 180, 190, 210, 230],  # 目标变量
    'temperature': [15, 16, 14, 13, 17, 20, 22, 21, 19, 18],  # 过去的温度
    'holiday': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # 过去的假日情况
})

# 2. 创建未来数据（预测时使用的外部变量）
futr_df = pd.DataFrame({
    'unique_id': ['store_1'] * 3,  # 预测未来 3 天
    'ds': pd.date_range(start='2023-01-11', periods=3, freq='D'),  # 未来 3 天
    'temperature': [17, 16, 18],  # 未来 3 天的温度
    'holiday': [0, 1, 0]  # 未来 3 天的假日情况
})

# 3. 定义 NBEATSx 模型（支持外部变量）
model = NBEATSx(
    input_size=5,  # 过去 5 期作为输入
    h=3,  # 预测未来 3 期
    futr_exog_list=['temperature', 'holiday']  # 外部变量列表
)

# 4. 初始化预测器
nf = NeuralForecast(
    models=[model],
    freq='D'
)

# 5. 训练模型
nf.fit(df)

# 6. 预测未来 3 天（传入 `futr_df`）
forecast = nf.predict(futr_df=futr_df)

print("\n预测结果：")
print(forecast)


if __name__ == '__main__':
    print("end")