import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# 加载示例数据
df = px.data.iris()

# 初始化 Dash 应用
app = dash.Dash()

# 创建布局
app.layout = html.Div(
    [
        html.H1("Iris 数据集"),
        # 创建一个散点图组件
        dcc.Graph(
            id="scatter-plot",
            figure=px.scatter(df, x="sepal_width", y="sepal_length", color="species"),
        ),
        # 创建一个下拉菜单，用户可以选择不同的 X 轴数据
        dcc.Dropdown(
            id="x-axis-dropdown",
            options=[
                {"label": "花萼宽度", "value": "sepal_width"},
                {"label": "花瓣宽度", "value": "petal_width"},
            ],
            value="sepal_width",  # 默认选择
        ),
    ]
)


# 设置回调
@app.callback(
    dash.dependencies.Output("scatter-plot", "figure"),
    [dash.dependencies.Input("x-axis-dropdown", "value")],
)
def update_graph(selected_value):
    # 根据下拉菜单选择的值，更新图表的 X 轴数据
    return px.scatter(df, x=selected_value, y="sepal_length", color="species")


# 运行应用
if __name__ == "__main__":
    app.run_server(debug=True)
