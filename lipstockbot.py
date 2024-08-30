import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class TradingPlatform(ABC):
    @abstractmethod
    def buy(self, asset, amount):
        pass

    @abstractmethod
    def sell(self, asset, amount):
        pass


class StockTradingPlatform(TradingPlatform):
    def __init__(self, api_key, api_secret, base_url):
        # Initialize stock trading client here
        pass

    def buy(self, asset, amount):
        print(f"Buying {amount} of stock {asset}")

    def sell(self, asset, amount):
        print(f"Selling {amount} of stock {asset}")


class CryptoTradingPlatform(TradingPlatform):
    def __init__(self, api_key, api_secret):
        # Initialize crypto trading client here
        pass

    def buy(self, asset, amount):
        print(f"Buying {amount} of crypto {asset}")

    def sell(self, asset, amount):
        print(f"Selling {amount} of crypto {asset}")


class LipstickBot:
    def __init__(self, stock_platform, crypto_platform):
        self.lipstick_companies = {
            "Procter & Gamble": "PG",
            "Estee Lauder": "EL",
            "L'Oreal": "LRLCY",
            "Unilever": "UL",
            "Coty": "COTY",
            "Shiseido": "SSDOY",
            "Beiersdorf": "BDRFY",
            "Inter Parfums": "IPAR",
        }
        self.trading_stocks = [
            "AAPL",
            "GOOG",
            "MSFT",
            "AMZN",
            "FB",
            "TSLA",
            "NVDA",
            "JPM",
            "V",
            "JNJ",
        ]
        self.trading_crypto = [
            "BTC-USD",
            "ETH-USD",
            "ADA-USD",
            "XRP-USD",
            "DOT-USD",
            "SOL-USD",
        ]
        self.stock_platform = stock_platform
        self.crypto_platform = crypto_platform

    def download_data(self, tickers, start="2022-01-01", end=None):
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        data = yf.download(tickers=tickers, start=start, end=end)
        return data

    def analyze_lipstick_index(self, data):
        daily_returns = data["Adj Close"].pct_change().dropna()

        # Calculate 5-day and 20-day moving averages for the lipstick index
        lipstick_index = daily_returns[list(self.lipstick_companies.values())].mean(
            axis=1
        )
        ma5 = lipstick_index.rolling(window=5).mean()
        ma20 = lipstick_index.rolling(window=20).mean()

        # Get the most recent data point
        latest_date = lipstick_index.index[-1]
        latest_li = lipstick_index.iloc[-1]
        latest_ma5 = ma5.iloc[-1]
        latest_ma20 = ma20.iloc[-1]

        # Define market sentiment based on lipstick index
        if latest_li > latest_ma20 and latest_ma5 > latest_ma20:
            market_sentiment = "Bullish"
        elif latest_li < latest_ma20 and latest_ma5 < latest_ma20:
            market_sentiment = "Bearish"
        else:
            market_sentiment = "Neutral"

        return {
            "market_sentiment": market_sentiment,
            "lipstick_index": latest_li,
            "ma5": latest_ma5,
            "ma20": latest_ma20,
        }

    def generate_trading_signals(self, lipstick_index_result):
        market_sentiment = lipstick_index_result["market_sentiment"]

        if market_sentiment == "Bullish":
            signal = "BUY"
        elif market_sentiment == "Bearish":
            signal = "SELL"
        else:
            signal = "HOLD"

        stock_signals = {stock: signal for stock in self.trading_stocks}
        crypto_signals = {crypto: signal for crypto in self.trading_crypto}

        return stock_signals, crypto_signals

    def execute_trades(self, stock_signals, crypto_signals):
        for asset, signal in stock_signals.items():
            if signal == "BUY":
                self.stock_platform.buy(asset, amount=100)  # Example amount
            elif signal == "SELL":
                self.stock_platform.sell(asset, amount=100)  # Example amount

        for asset, signal in crypto_signals.items():
            if signal == "BUY":
                self.crypto_platform.buy(asset, amount=1)  # Example amount
            elif signal == "SELL":
                self.crypto_platform.sell(asset, amount=1)  # Example amount

    def generate_recommendations(self):
        # Download data for lipstick companies
        lipstick_data = self.download_data(list(self.lipstick_companies.values()))

        # Analyze lipstick index
        lipstick_index_result = self.analyze_lipstick_index(lipstick_data)

        # Generate trading signals based on lipstick index
        stock_signals, crypto_signals = self.generate_trading_signals(
            lipstick_index_result
        )

        # Execute trades
        self.execute_trades(stock_signals, crypto_signals)

        return {
            "lipstick_index_analysis": lipstick_index_result,
            "stock_recommendations": stock_signals,
            "crypto_recommendations": crypto_signals,
        }


if __name__ == "__main__":
    # Usage
    stock_platform = StockTradingPlatform(
        "your_stock_api_key", "your_stock_secret_key", "https://stock-api-url.com"
    )
    crypto_platform = CryptoTradingPlatform(
        "your_crypto_api_key", "your_crypto_secret_key"
    )

    bot = LipstickBot(stock_platform, crypto_platform)
    recommendations = bot.generate_recommendations()
    print(recommendations)
