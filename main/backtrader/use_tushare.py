import backtrader as bt
import pandas as pd
import tushare as ts
from datetime import datetime

# 设置 Tushare 的 API Token
ts.set_token('0e2617b5080235a1bb1bf953f4ad6132ff58c74a7f864ac6bbb572a6')  # 替换为你的 Tushare API Token
pro = ts.pro_api()

# 定义一个简单的双均线策略
class MyStrategy(bt.Strategy):
    params = (
        ('short_period', 50),  # 短期SMA
        ('long_period', 200),  # 长期SMA
    )

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period)

    def next(self):
        if not self.position:  # 没有持仓
            if self.sma_short[0] > self.sma_long[0]:
                self.buy()  # 买入
        else:
            if self.sma_short[0] < self.sma_long[0]:
                self.sell()  # 卖出

# 获取股票数据（这里使用平安银行示例）
symbol = "000001.SZ"  # Tushare 的股票代码格式为 "代码.交易所"，例如平安银行为 "000001.SZ"
start_date = "20200101"
end_date = "20231001"

# 从 Tushare 获取日线数据
stock_df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)

# 数据预处理
stock_df.rename(columns={
    'trade_date': 'date',  # 将 'trade_date' 改为 'date'
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'vol': 'volume'  # 将 'vol' 改为 'volume'
}, inplace=True)

# 转换日期格式并设置为索引
stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y%m%d')
stock_df.set_index('date', inplace=True)

# 按日期升序排列（Tushare 数据默认是降序）
stock_df.sort_index(ascending=True, inplace=True)

# 添加 openinterest 列（Backtrader 要求）
stock_df['openinterest'] = 0

# 创建 Data Feed
data = bt.feeds.PandasData(
    dataname=stock_df,
    datetime=None,  # 使用索引作为日期
    open=1,         # open 列的位置（从 0 开始计数）
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=6
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
cerebro.broker.setcommission(commission=0.001)

# 添加分析指标
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

print('初始资金: %.2f' % cerebro.broker.getvalue())

# 运行回测
results = cerebro.run()

# 输出结果
print('最终资金: %.2f' % cerebro.broker.getvalue())
print('年化回报率: %.2f%%' % (results[0].analyzers.returns.get_analysis()['rnorm100']))
print('夏普比率: %.2f' % results[0].analyzers.sharpe.get_analysis()['sharperatio'])

# 绘制图表
cerebro.plot(style='candlestick', volume=False)

if __name__ == '__main__':
    print("回测完成")