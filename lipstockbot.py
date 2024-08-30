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


historical_data = download_data(companies)

print(historical_data)
