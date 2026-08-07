import yfinance as yf

def load_stock(ticker,start,end):
    data = yf.download(
        ticker,
        start=start,
        end=end
    )

    return data

def calculate_daily_returns(prices):
    return prices.pct_change().dropna()