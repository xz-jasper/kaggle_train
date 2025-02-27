import akshare as ak
import pandas as pd
from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM

# Step 1: 获取招商银行的历史股票数据
symbol = "600036"  # 招商银行的股票代码
stock_df = ak.stock_zh_a_hist(
    symbol=symbol, start_date="20210101", end_date="20221231", adjust="qfq"
)

# Step 2: 数据预处理
# 检查返回的数据，查看列名
print(stock_df.columns)

# 使用 'trade_date' 作为日期列名
stock_df["ds"] = pd.to_datetime(stock_df["日期"])  # 将日期列转换为 datetime 格式
stock_df["y"] = stock_df["收盘"]  # 将收盘价列命名为 'y'

# 为数据添加 unique_id 列，对于单一时间序列，可以给它一个常量值
stock_df["unique_id"] = 1  # 所有数据的 unique_id 设置为 1
# 选择需要的列：日期 ('ds') 和收盘价 ('y')
stock_df = stock_df[["unique_id", "ds", "y"]]

# # Step 3: 划分训练集和测试集
# train_size = int(len(stock_df) * 0.8)
# Y_train = stock_df.iloc[:train_size]
# Y_test = stock_df.iloc[train_size:]

# Step 4: 定义模型并训练
horizon = 30  # 设置预测的步数，比如预测未来 30 天的股价
Y_train = stock_df.iloc[:-horizon, :]
Y_test = stock_df.iloc[-horizon:, :]
# 创建模型列表：NBEATS、NHITS、LSTM
scale = 8
models = [
    NBEATS(
        input_size=scale * horizon,
        h=horizon,
        max_steps=100,
        batch_size=64,
        random_seed=0,
    ),
    NHITS(
        input_size=scale * horizon,
        h=horizon,
        max_steps=100,
        batch_size=64,
        random_seed=0,
    ),
    # LSTM(input_size=scale*horizon, h=horizon, max_steps=200, batch_size=64, random_seed=0)
]

# 使用 NeuralForecast 进行训练
nf = NeuralForecast(models=models, freq="D", local_scaler_type="robust")
nf.fit(df=Y_train)

# Step 5: 进行预测
Y_hat = nf.predict()
Y_hat = Y_hat.reset_index().drop(columns="unique_id")

# 更新预测结果的 'ds' 列，确保与测试集的日期一致
Y_hat["ds"] = list(Y_test["ds"])

# 设置预测结果的索引与实际数据对齐
Y_hat.index = Y_test["y"].index

# Step 6: 显示实际值与预测值的对比
from IPython.display import display


def show(data: pd.DataFrame, precision=2, cmap="Greens", axix=0):
    df = data.copy()
    df = df.style.format(precision=precision)
    df = df.background_gradient(cmap=cmap, axis=axix)
    # 使用原始数据来绘制表格
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")  # 不显示坐标轴
    ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f5f5f5"] * len(data.columns),
    )
    plt.show()


# 展示实际值与预测值的对比
show(pd.concat([Y_test["y"], Y_hat], axis=1)[["ds", "NBEATS", "NHITS", "y"]])


if __name__ == "__main__":
    print("end")
