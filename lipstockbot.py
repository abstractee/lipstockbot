import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List
from alpaca.trading.client import TradingClient
from binance import (
    Client as BinanceClient,
    ThreadedWebsocketManager,
    ThreadedDepthCacheManager,
)
import os


class DataDownloader:

    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _get_local_file_path(self, ticker):
        return os.path.join(self.save_dir, f"{ticker}.csv")

    def _load_local_data(self, ticker):
        file_path = self._get_local_file_path(ticker)
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col="Date", parse_dates=True)
        return None

    def _save_local_data(self, ticker, data):
        file_path = self._get_local_file_path(ticker)
        data.to_csv(file_path)

    def _merge_data(self, local_data, new_data):
        if local_data is not None:
            combined_data = (
                pd.concat([local_data, new_data]).drop_duplicates().sort_index()
            )
        else:
            combined_data = new_data
        return combined_data

    def download_data(self, tickers, start="2022-01-01", end=None):
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        all_data = {}
        for ticker in tickers.split():
            local_data = self._load_local_data(ticker)

            if local_data is not None:
                local_start = local_data.index.min().strftime("%Y-%m-%d")
                local_end = local_data.index.max().strftime("%Y-%m-%d")

                if local_start <= start and local_end >= end:
                    # Data exists locally, use it
                    data = local_data[
                        (local_data.index >= start) & (local_data.index <= end)
                    ]
                else:
                    # Data partially exists, fetch missing parts
                    missing_start = start if local_start > start else local_end
                    new_data = yf.download(ticker, start=missing_start, end=end)
                    data = self._merge_data(local_data, new_data)
                    self._save_local_data(ticker, data)
            else:
                # No local data, fetch everything
                data = yf.download(ticker, start=start, end=end)
                self._save_local_data(ticker, data)

            all_data[ticker] = data

        return all_data


class TradingPlatform(ABC):
    def __init__(self, assets: Dict[str, float]):
        self.assets = assets

    @abstractmethod
    def buy(self, asset: str, amount: float):
        pass

    @abstractmethod
    def sell(self, asset: str, amount: float):
        pass

    @abstractmethod
    def get_current_price(self, asset: str) -> float:
        pass


class AlpacaPlatform(TradingPlatform):
    def __init__(
        self, api_key: str, api_secret: str, base_url: str, assets: Dict[str, float]
    ):
        super().__init__(assets)
        self.api = TradingClient(api_key, api_secret, paper=True)

    def buy(self, asset: str, amount: float):
        if asset in self.assets:
            try:
                self.api.submit_order(
                    symbol=asset,
                    qty=amount,
                    side="buy",
                    type="market",
                    time_in_force="gtc",
                )
                self.assets[asset] += amount
                print(f"Bought {amount} of stock {asset}")
            except Exception as e:
                print(f"Error buying {asset}: {str(e)}")
        else:
            print(f"Asset {asset} not in portfolio. Cannot buy.")

    def sell(self, asset: str, amount: float):
        if asset in self.assets and self.assets[asset] >= amount:
            try:
                self.api.submit_order(
                    symbol=asset,
                    qty=amount,
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                self.assets[asset] -= amount
                print(f"Sold {amount} of stock {asset}")
            except Exception as e:
                print(f"Error selling {asset}: {str(e)}")
        else:
            print(f"Insufficient {asset} in portfolio. Cannot sell.")

    def get_current_price(self, asset: str) -> float:
        try:
            return float(self.api.get_last_trade(asset).price)
        except Exception as e:
            print(f"Error getting price for {asset}: {str(e)}")
            return 0.0


class BinancePlatform(TradingPlatform):
    def __init__(self, api_key: str, api_secret: str, assets: Dict[str, float]):
        super().__init__(assets)
        self.client = BinanceClient(api_key, api_secret)

    def buy(self, asset: str, amount: float):
        if asset in self.assets:
            try:
                self.client.create_order(
                    symbol=asset, side="BUY", type="MARKET", quantity=amount
                )
                self.assets[asset] += amount
                print(f"Bought {amount} of crypto {asset}")
            except Exception as e:
                print(f"Error buying {asset}: {str(e)}")
        else:
            print(f"Asset {asset} not in portfolio. Cannot buy.")

    def sell(self, asset: str, amount: float):
        if asset in self.assets and self.assets[asset] >= amount:
            try:
                self.client.create_order(
                    symbol=asset, side="SELL", type="MARKET", quantity=amount
                )
                self.assets[asset] -= amount
                print(f"Sold {amount} of crypto {asset}")
            except Exception as e:
                print(f"Error selling {asset}: {str(e)}")
        else:
            print(f"Insufficient {asset} in portfolio. Cannot sell.")

    def get_current_price(self, asset: str) -> float:
        try:
            return float(self.client.get_symbol_ticker(symbol=asset)["price"])
        except Exception as e:
            print(f"Error getting price for {asset}: {str(e)}")
            return 0.0


class TestPlatform(TradingPlatform):
    def __init__(self, assets: Dict[str, float]):
        super().__init__(assets)
        self.price_data = {}

    def load_price_data(self, start_date, end_date):
        tickers = list(self.assets.keys())
        self.price_data = yf.download(tickers, start=start_date, end=end_date)[
            "Adj Close"
        ]

    def get_current_price(self, asset: str, date=None) -> float:
        if date is None:
            date = self.price_data.index[-1]
        return self.price_data.loc[date, asset]

    def buy(self, asset: str, amount: float, date=None):
        if date is None:
            date = self.price_data.index[-1]
        price = self.get_current_price(asset, date)
        cost = price * amount
        if self.cash >= cost:
            self.cash -= cost
            self.assets[asset] += amount
            print(f"Bought {amount} of {asset} at {price} on {date}")
        else:
            print(f"Insufficient cash to buy {amount} of {asset}")

    def sell(self, asset: str, amount: float, date=None):
        if date is None:
            date = self.price_data.index[-1]
        if self.assets[asset] >= amount:
            price = self.get_current_price(asset, date)
            self.cash += price * amount
            self.assets[asset] -= amount
            print(f"Sold {amount} of {asset} at {price} on {date}")
        else:
            print(f"Insufficient {asset} to sell")


class LipstickBot:
    def __init__(self, platforms: List[TradingPlatform]):
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
        self.platforms = platforms
        self.downloader = DataDownloader()

    def download_data(self, tickers, start="2022-01-01", end=None):
        data = self.download_data(tickers, start, end)
        return data

    def analyze_lipstick_index(self, data):
        daily_returns = data["Adj Close"].pct_change().dropna()

        lipstick_index = daily_returns[list(self.lipstick_companies.values())].mean(
            axis=1
        )
        ma5 = lipstick_index.rolling(window=5).mean()
        ma20 = lipstick_index.rolling(window=20).mean()

        latest_li = lipstick_index.iloc[-1]
        latest_ma5 = ma5.iloc[-1]
        latest_ma20 = ma20.iloc[-1]

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

        signals = {}
        for platform in self.platforms:
            signals[type(platform).__name__] = {
                asset: signal for asset in platform.assets.keys()
            }

        return signals

    def execute_trades(self, signals):
        for platform in self.platforms:
            platform_name = type(platform).__name__
            for asset, signal in signals[platform_name].items():
                current_price = platform.get_current_price(asset)
                if current_price == 0:
                    continue  # Skip if we couldn't get the current price

                if signal == "BUY":
                    amount = 100 / current_price  # Buy $100 worth
                    platform.buy(asset, amount)
                elif signal == "SELL":
                    amount = min(
                        50 / current_price, platform.assets[asset]
                    )  # Sell $50 worth or all holdings
                    platform.sell(asset, amount)

    def generate_recommendations(self):
        lipstick_data = self.download_data(list(self.lipstick_companies.values()))
        lipstick_index_result = self.analyze_lipstick_index(lipstick_data)
        signals = self.generate_trading_signals(lipstick_index_result)
        self.execute_trades(signals)

        current_holdings = {
            type(platform).__name__: platform.assets for platform in self.platforms
        }

        return {
            "lipstick_index_analysis": lipstick_index_result,
            "trading_signals": signals,
            "current_holdings": current_holdings,
        }

    def backtest(self, start_date, end_date):
        # Load price data for the entire period
        for platform in self.platforms:
            if isinstance(platform, TestPlatform):
                platform.load_price_data(start_date, end_date)

        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        portfolio_values = []

        for date in date_range:
            self.generate_recommendations(date)

            # Calculate portfolio value
            portfolio_value = sum(platform.cash for platform in self.platforms)
            for platform in self.platforms:
                for asset, amount in platform.assets.items():
                    portfolio_value += amount * platform.get_current_price(asset, date)

            portfolio_values.append(portfolio_value)

        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change()
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[
            0
        ]
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

        return {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Final Portfolio Value": f"${portfolio_values[-1]:,.2f}",
        }


if __name__ == "__main__":
    # Usage
    # alpaca_assets = {"AAPL": 100, "GOOG": 50, "MSFT": 75, "AMZN": 30, "FB": 60}

    # binance_assets = {"BTCUSDT": 1.5, "ETHUSDT": 10, "ADAUSDT": 5000, "XRPUSDT": 10000}

    # alpaca_platform = AlpacaPlatform(
    #     "your_alpaca_api_key",
    #     "your_alpaca_secret_key",
    #     "https://paper-api.alpaca.markets",
    #     alpaca_assets,
    # )

    # binance_platform = BinancePlatform(
    #     "your_binance_api_key", "your_binance_secret_key", binance_assets
    # )

    test_assets = {
        "AAPL": 100,
        "GOOG": 50,
        "MSFT": 75,
        "AMZN": 30,
        "META": 60,
        "BTC-USD": 1.5,
        "ETH-USD": 10,
    }

    test_platform = TestPlatform(test_assets)

    bot = LipstickBot([test_assets])

    # Generate recommendations for the latest date
    recommendations = bot.generate_recommendations()
    print("Latest Recommendations:")
    print(recommendations)

    # Backtest
    backtest_results = bot.backtest("2022-01-01", "2023-12-31")
    print("\nBacktest Results:")
    print(backtest_results)
