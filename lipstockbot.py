import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List
from alpaca.trading.client import TradingClient
from binance import (
    Client as BinanceClient,
    ThreadedWebsocketManager,
    ThreadedDepthCacheManager,
)
import os
import pickle


class DataDownloader:

    def __init__(self, save_dir="data", pickle_file="all_tickers_data.pkl"):
        self.save_dir = save_dir
        self.pickle_file = pickle_file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _get_pickle_file_path(self):
        return os.path.join(self.save_dir, self.pickle_file)

    def _load_local_data(self):
        file_path = self._get_pickle_file_path()
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_local_data(self, data):
        file_path = self._get_pickle_file_path()
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def download_data(self, tickers, start=None, end=None):
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        if start is None:
            start = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        # Convert to string format "%Y-%m-%d"
        start = pd.to_datetime(start).strftime("%Y-%m-%d")
        end = pd.to_datetime(end).strftime("%Y-%m-%d")

        # Convert to pandas datetime with UTC timezone
        start = pd.to_datetime(start).tz_localize("UTC")
        end = pd.to_datetime(end).tz_localize("UTC")

        print(f"downloading for {tickers} from {start} to {end}")
        local_data = self._load_local_data()

        if local_data is not None:
            local_start = local_data.index.min().strftime("%Y-%m-%d")
            local_end = local_data.index.max().strftime("%Y-%m-%d")
            local_start = pd.to_datetime(local_start).tz_localize("UTC")
            local_end = pd.to_datetime(local_end).tz_localize("UTC")
            existing_tickers = set(
                local_data.columns.levels[1]
            )  # Assuming MultiIndex columns
            new_tickers = set(tickers) - existing_tickers

            if local_start <= start and local_end >= end and not new_tickers:
                # Data exists locally and covers the required date range and tickers
                data = local_data[
                    (local_data.index >= start) & (local_data.index <= end)
                ]
                print("date not within")
            else:
                # Data is not up to date or new tickers are requested
                print("data not uptodates")
                if new_tickers:
                    print("new tickers")
                    # Download data for new tickers using local date boundaries
                    new_data = yf.download(
                        tickers=list(new_tickers),
                        start=local_start,
                        end=local_end,
                        ignore_tz=False,
                    )
                    # Merge new data with existing data
                    merged_data = pd.concat([local_data, new_data], axis=1)
                else:
                    merged_data = local_data

                # Download any missing data for the requested date range
                if start < local_start or end > local_end:
                    additional_data = yf.download(
                        tickers=tickers, start=start, end=end, ignore_tz=False
                    )
                    merged_data = (
                        pd.concat([merged_data, additional_data])
                        .sort_index()
                        .drop_duplicates()
                    )

                self._save_local_data(merged_data)
                data = merged_data[
                    (merged_data.index >= start) & (merged_data.index <= end)
                ]
        else:
            print("fetching")
            # No local data, fetch everything
            data = yf.download(tickers=tickers, start=start, end=end, ignore_tz=False)
            self._save_local_data(data)

        return data


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
    def __init__(self, assets: Dict[str, float], cash: float = 10000.0):
        super().__init__(assets)
        self.price_data = None
        self.cash = cash
        self.downloader = DataDownloader()

    def load_price_data(self, start_date, end_date):
        tickers = list(self.assets.keys())
        self.price_data = self.downloader.download_data(
            tickers, start=start_date, end=end_date
        )["Adj Close"]
        # yf.download(tickers, start=start_date, end=end_date, ignore_tz=False)

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

    def download_data(self, tickers, start=None, end=None):
        data = self.downloader.download_data(tickers, start, end)
        # if end is None:
        #     end = datetime.today().strftime("%Y-%m-%d")
        # data = yf.download(tickers=tickers, start=start, end=end)
        return data

    def analyze_lipstick_index(self, data, date=None):
        daily_returns = data["Adj Close"].pct_change().dropna()

        lipstick_index = daily_returns[list(self.lipstick_companies.values())].mean(
            axis=1
        )
        lipstick_index.index = lipstick_index.index.normalize()

        ma5 = lipstick_index.rolling(window=5).mean()
        ma20 = lipstick_index.rolling(window=20).mean()

        if date is None:
            date = lipstick_index.index[-1]

        dateend = lipstick_index.index[-1]
        datestart = lipstick_index.index[0]
        latest_li = None
        latest_ma5 = None
        latest_ma20 = None
        try:
            latest_li = lipstick_index.loc[date]
            latest_ma5 = ma5.loc[date]
            latest_ma20 = ma20.loc[date]
        except KeyError:
            return False

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

    def execute_trades(self, signals, date=None):
        for platform in self.platforms:
            platform_name = type(platform).__name__
            for asset, signal in signals[platform_name].items():
                current_price = platform.get_current_price(asset, date)
                if current_price == 0:
                    continue  # Skip if we couldn't get the current price

                if signal == "BUY":
                    amount = 100 / current_price  # Buy $100 worth
                    platform.buy(asset, amount, date)
                elif signal == "SELL":
                    amount = min(
                        50 / current_price, platform.assets[asset]
                    )  # Sell $50 worth or all holdings
                    platform.sell(asset, amount, date)

    def generate_recommendations(self, date=None):
        lipstick_data = self.download_data(list(self.lipstick_companies.values()))

        lipstick_index_result = self.analyze_lipstick_index(lipstick_data, date)
        if not lipstick_index_result:
            return False

        signals = self.generate_trading_signals(lipstick_index_result)
        self.execute_trades(signals, date)

        current_holdings = {
            type(platform).__name__: platform.assets for platform in self.platforms
        }

        return {
            "lipstick_index_analysis": lipstick_index_result,
            "trading_signals": signals,
            "current_holdings": current_holdings,
        }

    def backtest(self, start_date, end_date):
        # Use a business day offset to adjust the start and end dates if necessary
        if not np.is_busday(start_date.date()):
            start_date = pd.offsets.BDay().rollback(start_date)
        if not np.is_busday(end_date.date()):
            end_date = pd.offsets.BDay().rollback(end_date)
        # Load price data for the entire period
        for platform in self.platforms:
            if isinstance(platform, TestPlatform):
                platform.load_price_data(start_date, end_date)

        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        portfolio_values = []

        for date in date_range:
            date_recommendations = self.generate_recommendations(date)
            if not date_recommendations:
                continue
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

    bot = LipstickBot([test_platform])

    # Generate recommendations for the latest date
    # recommendations = bot.generate_recommendations()
    # print("Latest Recommendations:")
    # print(recommendations)

    # Backtest

    start_date = pd.to_datetime("2022-01-01").tz_localize("UTC")
    end_date = pd.to_datetime("2023-12-31").tz_localize("UTC")

    backtest_results = bot.backtest(start_date, end_date)
    print("\nBacktest Results:")
    print(backtest_results)
