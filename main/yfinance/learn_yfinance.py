# 下载单只股票的历史数据：
import yfinance as yf


def download_one_stock():
    stock = yf.Ticker("AAPL")  # 获取苹果公司（AAPL）的数据
    data = stock.history(period="1y")  # 获取过去一年的数据
    print(data)


def download_more_stock():
    data = yf.download(["AAPL", "GOOG", "MSFT"], start="2020-01-01", end="2021-01-01")
    print(data)


def get_financials():
    stock = yf.Ticker("AAPL")
    financials = stock.financials  # 获取财务报表
    print(financials)


if __name__ == "__main__":
    get_financials()
    print("end")
