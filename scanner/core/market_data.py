import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, period="2y"):

    return yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )


def series_for(data, ticker, field="Close"):

    if isinstance(data.columns, pd.MultiIndex):

        if ticker in data.columns.get_level_values(0):
            return data[ticker][field].dropna()

        if ticker in data.columns.get_level_values(1):
            return data[field][ticker].dropna()

    if field in data.columns:
        return data[field].dropna()

    return pd.Series(dtype=float)


def indicators(df):

    x = df.copy().dropna()

    if len(x) < 220:
        return None

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()
    x["sma200"] = close.rolling(200).mean()

    x["vol20"] = volume.rolling(20).mean()

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = true_range.rolling(14).mean()

    x["ret1m"] = close.pct_change(21)
    x["ret3m"] = close.pct_change(63)
    x["ret6m"] = close.pct_change(126)
    x["ret12m"] = close.pct_change(252)

    x["vol6m"] = (
        close.pct_change()
        .rolling(126)
        .std()
        * np.sqrt(252)
    )

    x["high20"] = high.rolling(20).max().shift(1)
    x["high52"] = high.rolling(252).max()

    x["dist52"] = (
        (x["high52"] - close)
        / x["high52"]
    )

    x["dist20res"] = (
        (x["high20"] - close)
        / x["high20"]
    )

    x["atr_pct"] = x["atr14"] / close

    x["atr_pct_20ago"] = x["atr_pct"].shift(20)

    x["vol20_prev"] = x["vol20"].shift(20)

    return x


def latest_metrics(x):

    row = x.iloc[-1]

    return {
        "price": float(row["Close"]),
        "sma20": float(row["sma20"]),
        "sma50": float(row["sma50"]),
        "sma200": float(row["sma200"]),
        "volume": float(row["Volume"]),
        "vol20": float(row["vol20"]),
        "atr_pct": float(row["atr_pct"]),
        "atr_pct_20ago": float(row["atr_pct_20ago"]),
        "ret1m": float(row["ret1m"]),
        "ret3m": float(row["ret3m"]),
        "ret6m": float(row["ret6m"]),
        "ret12m": float(row["ret12m"]),
        "vol6m": float(row["vol6m"]),
        "dist52": float(row["dist52"]),
        "dist20res": float(row["dist20res"]),
        "high20": float(row["high20"]),
    }
