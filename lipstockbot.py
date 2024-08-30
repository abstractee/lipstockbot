import yfinance as yf
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
