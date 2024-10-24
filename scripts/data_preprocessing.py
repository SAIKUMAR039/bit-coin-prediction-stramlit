import yfinance as yf
import pandas as pd

def fetch_data(ticker):
    data = yf.download(ticker, start="2021-01-01", end="2023-01-01")
    data.to_csv("../data/raw/bitcoin_data.csv")
    return data

def preprocess_data(data):
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    return X, y

if __name__ == "__main__":
    data = fetch_data("BTC-USD")
    X, y = preprocess_data(data)
