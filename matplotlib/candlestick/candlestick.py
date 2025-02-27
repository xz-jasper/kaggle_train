import akshare as ak
import pandas as pd

# 获取招商银行股票数据
stock_data = ak.stock_zh_a_hist(symbol="600036", period="daily", adjust="qfq")

# 重命名列，使其符合 Plotly 的标准列名
stock_data = stock_data[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'volume']

# 转换日期列为 datetime 类型
stock_data['date'] = pd.to_datetime(stock_data['date'])

# 设置日期为索引
stock_data.set_index('date', inplace=True)

# 查看数据
print(stock_data.head())



from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_candlestick(stock_df, name="", rolling_avg=None, fig_size=(1100, 700)):
    """
    绘制蜡烛图
    Args:
        stock_df (pd.DataFrame): 股票数据
        name (str): 股票名称
        rolling_avg (list of int, optional): 滚动平均窗口的大小
        fig_size (tuple): 图表大小
    """
    # 复制数据避免修改原始数据
    stock_data = stock_df.copy()

    # 创建蜡烛图
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=stock_data.index,
                close=stock_data["close"],
                open=stock_data["open"],
                high=stock_data["high"],
                low=stock_data["low"],
                name="Candlesticks",
                increasing_line_color="green",
                decreasing_line_color="red",
                line=dict(width=1),
            )
        ]
    )

    # 如果指定了滚动平均，则绘制
    if rolling_avg:
        colors = [
            "rgba(0, 255, 255, 0.5)",  # cyan
            "rgba(255, 255, 0, 0.5)",  # yellow
            "rgba(255, 165, 0, 0.5)",  # orange
            "rgba(255, 105, 180, 0.5)",  # pink
            "rgba(165, 42, 42, 0.5)",  # brown
            "rgba(128, 128, 128, 0.5)",  # gray
            "rgba(128, 128, 0, 0.5)",  # olive
            "rgba(0, 0, 255, 0.5)",
        ]  # blue

        for i, avg in enumerate(rolling_avg):
            color = colors[i % len(colors)]
            ma_column = f"{avg}-day MA"
            stock_data[ma_column] = stock_data["close"].rolling(window=avg).mean()

            # 添加移动平均线
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data[ma_column],
                    mode="lines",
                    name=f"{avg}-day Moving Average",
                    line=dict(color=color),
                )
            )

    # 更新布局
    fig.update_layout(
        title=f"{name} Stock Price - Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        width=fig_size[0],
        height=fig_size[1],
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=14, label="2w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=2, label="2y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                ),
                bgcolor="pink",
                font=dict(color="black"),
                activecolor="lightgreen",
            )
        ),
    )
    fig.show(renderer="browser")  # 使用 'iframe' 渲染器


# 获取招商银行的数据并绘制蜡烛图
plot_candlestick(stock_data, name="招商银行", rolling_avg=[5, 10, 20])


if __name__ == '__main__':
    print("end")