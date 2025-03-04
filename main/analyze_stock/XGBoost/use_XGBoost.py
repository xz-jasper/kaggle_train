import akshare as ak
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import backtrader as bt
import xgboost as xgb
from sklearn.metrics import accuracy_score


# 1. 获取股票数据
def fetch_stock_data(symbol, start_date, end_date):
    """
    从 akshare 获取股票数据
    """
    stock_df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="hfq",  # 后复权
    )
    return stock_df


# 2. 数据预处理
def preprocess_data(stock_df):
    """
    数据预处理：计算特征和标签
    """
    # 计算每日收益率
    stock_df["return"] = stock_df["收盘"].pct_change()

    # 计算移动平均线
    stock_df["sma_10"] = stock_df["收盘"].rolling(window=10).mean()
    stock_df["sma_50"] = stock_df["收盘"].rolling(window=50).mean()

    # 计算波动率
    stock_df["volatility"] = stock_df["return"].rolling(window=10).std()

    # 计算标签：1 表示上涨，0 表示下跌
    stock_df["label"] = np.where(stock_df["return"].shift(-1) > 0, 1, 0)

    # 删除缺失值
    stock_df.dropna(inplace=True)

    return stock_df


# 3. 训练机器学习模型
def train_model(stock_df):
    """
    使用随机森林训练模型
    """
    # 特征列
    features = ["sma_10", "sma_50", "volatility"]
    X = stock_df[features]

    # 标签列
    y = stock_df["label"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # # 训练随机森林模型
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    }

    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)


    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    return model


# 4. 使用机器学习模型生成交易信号
def generate_signals(stock_df, model):
    """
    使用训练好的模型生成交易信号
    """
    features = ["sma_10", "sma_50", "volatility"]
    X = stock_df[features]

    # 预测标签
    stock_df["predicted_label"] = model.predict(X)

    # 生成交易信号：1 表示买入，-1 表示卖出
    stock_df["signal"] = 0
    stock_df.loc[stock_df["predicted_label"] == 1, "signal"] = 1
    stock_df.loc[stock_df["predicted_label"] == 0, "signal"] = -1

    return stock_df


# 5. 扩展 PandasData 以支持 signal 列
class SignalData(bt.feeds.PandasData):
    """
    自定义 PandasData，支持 signal 列
    """

    lines = ("signal",)  # 添加 signal 列
    params = (("signal", -1),)  # signal 列的默认值


# 6. 回测策略
class MLStrategy(bt.Strategy):
    """
    基于机器学习信号的策略
    """

    def __init__(self):
        self.signal = self.datas[0].signal  # 获取 signal 列

    def next(self):
        if self.signal[0] == 1:  # 买入信号
            if not self.position:
                self.buy()
        elif self.signal[0] == -1:  # 卖出信号
            if self.position:
                self.sell()


# 7. 主函数
def main():
    # 获取股票数据
    symbol = "000001"  # 平安银行
    start_date = "20200101"
    end_date = "20231001"
    stock_df = fetch_stock_data(symbol, start_date, end_date)

    # 数据预处理
    stock_df = preprocess_data(stock_df)

    # 训练机器学习模型
    model = train_model(stock_df)

    # 生成交易信号
    stock_df = generate_signals(stock_df, model)

    # 将数据转换为 Backtrader 格式
    stock_df["date"] = pd.to_datetime(stock_df["日期"])
    stock_df.set_index("date", inplace=True)
    stock_df["openinterest"] = 0

    # 使用自定义的 SignalData
    data = SignalData(
        dataname=stock_df,
        datetime=None,  # 使用索引作为日期
        open=2,  # 开盘价
        high=3,  # 最高价
        low=4,  # 最低价
        close=5,  # 收盘价
        volume=6,  # 成交量
        openinterest=7,  # 无意义，占位
        signal=8,  # 交易信号
    )

    # 初始化回测引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(MLStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # 运行回测
    print("初始资金: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("最终资金: %.2f" % cerebro.broker.getvalue())

    # 绘制回测结果
    cerebro.plot()


if __name__ == "__main__":
    main()
