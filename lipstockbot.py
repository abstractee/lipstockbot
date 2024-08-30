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

    def download_data(self, tickers, start="2022-01-01", end=None):
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        data = yf.download(tickers=tickers, start=start, end=end)
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


if __name__ == "__main__":
    # Usage
    alpaca_assets = {"AAPL": 100, "GOOG": 50, "MSFT": 75, "AMZN": 30, "FB": 60}

    binance_assets = {"BTCUSDT": 1.5, "ETHUSDT": 10, "ADAUSDT": 5000, "XRPUSDT": 10000}

    alpaca_platform = AlpacaPlatform(
        "your_alpaca_api_key",
        "your_alpaca_secret_key",
        "https://paper-api.alpaca.markets",
        alpaca_assets,
    )

    binance_platform = BinancePlatform(
        "your_binance_api_key", "your_binance_secret_key", binance_assets
    )

    bot = LipstickBot([alpaca_platform, binance_platform])
    recommendations = bot.generate_recommendations()
    print(recommendations)
