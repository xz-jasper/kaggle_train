from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.utils import AirPassengersDF

# 加载数据
df = AirPassengersDF

# 定义模型
model = NBEATS(input_size=12, h=12)

# 训练并预测
nf = NeuralForecast(models=[model], freq='M')
nf.fit(df)
forecast = nf.predict()
print(forecast)


if __name__ == '__main__':
    print("end")