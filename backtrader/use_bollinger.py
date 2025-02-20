import backtrader as bt
import pandas as pd
import akshare as ak
from datetime import datetime


# 定义一个简单的双均线策略
# 定义布林带策略
class MyStrategy(bt.Strategy):
    params = (
        ("sma_period", 50),  # 中间SMA周期
        ("bollinger_period", 20),  # 布林带的周期
        ("bollinger_dev", 2),  # 标准差倍数
    )

    def __init__(self):
        # 计算布林带的中间线（SMA）
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.sma_period
        )

        # 计算布林带上下轨
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bollinger_period,
            devfactor=self.params.bollinger_dev,
        )

    def next(self):
        if not self.position:  # 如果没有持仓
            # 如果当前价格突破下轨，买入
            if self.data.close[0] < self.bollinger.lines.bot[0]:
                self.buy()  # 买入
        else:
            # 如果当前价格突破上轨，卖出
            if self.data.close[0] > self.bollinger.lines.top[0]:
                self.sell()  # 卖出


# 获取股票数据（这里使用平安银行示例）
symbol = "000001"
stock_df = ak.stock_zh_a_hist(
    symbol=symbol,
    period="daily",
    start_date="20240101",
    end_date="20250201",
    adjust="hfq",
)

# 数据预处理
stock_df.rename(
    columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    },
    inplace=True,
)

# 转换日期格式并设置为索引
stock_df["date"] = pd.to_datetime(stock_df["date"])
stock_df.set_index("date", inplace=True)

# 添加openinterest列（Backtrader要求）
stock_df["openinterest"] = 0

# 创建Data Feed
data = bt.feeds.PandasData(
    dataname=stock_df,
    datetime=None,  # 使用索引作为日期
    open=1,  # open列的位置（从0开始计数）
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=6,
)

# 初始化回测引擎
cerebro = bt.Cerebro()

# 添加数据
cerebro.adddata(data)

# 添加策略
cerebro.addstrategy(MyStrategy)

# 设置初始资金
cerebro.broker.setcash(100000.0)

# 设置佣金
cerebro.broker.setcommission(commission=0.0001)

# 添加分析指标
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

print("初始资金: %.2f" % cerebro.broker.getvalue())

# 运行回测
results = cerebro.run()

# 输出结果
print("最终资金: %.2f" % cerebro.broker.getvalue())
print("年化回报率: %.2f%%" % (results[0].analyzers.returns.get_analysis()["rnorm100"]))
print("夏普比率: %.2f" % results[0].analyzers.sharpe.get_analysis()["sharperatio"])

# 绘制图表
cerebro.plot(style="candlestick", volume=False)

if __name__ == "__main__":
    print("回测完成")
