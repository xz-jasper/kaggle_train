import akshare as ak
import plotly.graph_objects as go
import pandas as pd

# 获取招商银行的股票数据，假设获取的是日线数据
df = ak.stock_zh_a_hist(symbol="600036", period="daily", adjust="qfq")

# 修改列名为plotly所需的格式
df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

# 转换日期列为datetime类型
df['date'] = pd.to_datetime(df['date'])

# 计算布林带（Bollinger Bands）
# 计算20日简单移动平均（SMA）
df['SMA20'] = df['close'].rolling(window=20).mean()
# 计算标准差
df['std20'] = df['close'].rolling(window=20).std()
# 计算布林带
df['upper_band'] = df['SMA20'] + 2 * df['std20']
df['lower_band'] = df['SMA20'] - 2 * df['std20']

# 创建Plotly的Candlestick图
fig = go.Figure(data=[go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='green',
    decreasing_line_color='red',
    name='K线图'
)])

# 添加布林带的上轨、中轨和下轨
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['upper_band'],
    line=dict(color='red', width=1),
    name='上轨 (Upper Band)'
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['SMA20'],
    line=dict(color='blue', width=1),
    name='中轨 (Middle Band)'
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['lower_band'],
    line=dict(color='red', width=1),
    name='下轨 (Lower Band)'
))

# 添加成交量子图
fig.add_trace(go.Bar(
    x=df['date'],
    y=df['volume'],
    name="Volume",
    marker_color='rgba(246, 78, 139, 0.6)',
    yaxis='y2'
))

# 配置布局
fig.update_layout(
    title="招商银行 K线图与布林带",
    xaxis_title="日期",
    yaxis_title="价格",
    yaxis2=dict(
        title="成交量",
        overlaying='y',
        side='right'
    ),
    xaxis_rangeslider_visible=False  # 可选，是否显示滑动条
)

# 显示图表
fig.show()


if __name__ == '__main__':
    print("end")