import plotly.graph_objects as go

# 创建一个简单的折线图
fig = go.Figure()

# 添加折线数据
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines'))

# 设置图表标题
fig.update_layout(title="简单的折线图")

# 展示图表
fig.show()

if __name__ == '__main__':
    print("end")
