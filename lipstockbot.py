import yfinance as yf

import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

companies = {
    "Procter & Gamble": "PG",
    "Estee Lauder": "EL",
    "L'Oreal": "LRLCY",
    "Unilever": "UL",
    "Coty": "COTY",
    "Shiseido": "SSDOY",
    "Beiersdorf": "BDRFY",
    "Inter Parfums": "IPAR",
}


def download_data(tickers, start="2022-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(tickers=list(tickers.values()), start=start, end=end)
    return data


def analyze_lipstick_index(data):
    # Calculate daily returns
    daily_returns = data["Adj Close"].pct_change().dropna()

    # Average return
    avg_returns = daily_returns.mean()

    # Create a signal based on performance
    signals = avg_returns.apply(lambda x: "BUY" if x > 0 else "SELL")
    return signals


historical_data = download_data(companies)
signals = analyze_lipstick_index(historical_data)
print(signals)

# Alpaca API keys
API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_secret_key"
BASE_URL = "https://paper-api.alpaca.markets"  # Use the paper trading API for testing

# Initialize Alpaca API client
alpaca_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)


def execute_trades(signals):
    for company, signal in signals.items():
        ticker = companies[company]
        if signal == "BUY":
            # Code to place a buy order
            print(f"Buying {ticker}")
        elif signal == "SELL":
            # Code to place a sell order
            print(f"Selling {ticker}")


execute_trades(signals)
