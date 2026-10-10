import time

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, period="2y", batch_size=50):
    """Download price history in small batches with one retry per batch."""
    tickers = list(dict.fromkeys(
        ticker for ticker in tickers
        if isinstance(ticker, str) and ticker.strip()
    ))
    if not tickers:
        return pd.DataFrame()

    batches = []
    failed = []

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        result = None

        for attempt in range(2):
            try:
                candidate = yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                    timeout=30,
                )
                if candidate is not None and not candidate.empty:
                    result = candidate
                    break
            except Exception as exc:
                print(
                    f"Download batch {start + 1}-{start + len(batch)}, "
                    f"attempt {attempt + 1}: {exc}"
                )
            if attempt == 0:
                time.sleep(2)

        if result is None or result.empty:
            failed.extend(batch)
            print(f"WARNING: price download failed for batch starting at {start + 1}.")
            continue

        if not isinstance(result.columns, pd.MultiIndex):
            if len(batch) == 1:
                result = result.copy()
                result.columns = pd.MultiIndex.from_product([[batch[0]], result.columns])
            else:
                failed.extend(batch)
                print("WARNING: unexpected flat columns; skipped batch.")
                continue
        else:
            level0 = set(map(str, result.columns.get_level_values(0)))
            level1 = set(map(str, result.columns.get_level_values(1)))
            batch_set = set(batch)
            if level1.intersection(batch_set) and not level0.intersection(batch_set):
                result = result.swaplevel(0, 1, axis=1).sort_index(axis=1)

        batches.append(result)
        print(f"Downloaded batch {start + 1}-{start + len(batch)} of {len(tickers)} tickers.")
        time.sleep(0.25)

    if not batches:
        print("ERROR: no price batches downloaded.")
        return pd.DataFrame()

    combined = pd.concat(batches, axis=1)
    if isinstance(combined.columns, pd.MultiIndex):
        combined = combined.loc[:, ~combined.columns.duplicated()]
        count = len(set(map(str, combined.columns.get_level_values(0))))
    else:
        count = 1

    print(f"Price data available for approximately {count} tickers.")
    if failed:
        print(f"WARNING: {len(failed)} ticker(s) failed; examples: {', '.join(failed[:15])}")
    return combined


def series_for(data, ticker, field="Close"):
    if data is None or data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)

        if ticker in level0:
            block = data[ticker]
            if field in block.columns:
                return pd.to_numeric(block[field], errors="coerce").dropna()

        if ticker in level1:
            try:
                return pd.to_numeric(data[field][ticker], errors="coerce").dropna()
            except (KeyError, TypeError):
                return pd.Series(dtype=float)

    if field in data.columns:
        return pd.to_numeric(data[field], errors="coerce").dropna()
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
    x["vol6m"] = close.pct_change().rolling(126).std() * np.sqrt(252)

    x["high20"] = high.rolling(20).max().shift(1)
    x["high52"] = high.rolling(252).max()
    x["dist52"] = (x["high52"] - close) / x["high52"]
    x["dist20res"] = (x["high20"] - close) / x["high20"]
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
        "vol20_prev": (
            float(row["vol20_prev"])
            if pd.notna(row["vol20_prev"])
            else float(row["vol20"])
        ),
        "atr_pct": float(row["atr_pct"]),
        "atr_pct_20ago": (
            float(row["atr_pct_20ago"])
            if pd.notna(row["atr_pct_20ago"])
            else float(row["atr_pct"])
        ),
        "ret1m": float(row["ret1m"]),
        "ret3m": float(row["ret3m"]),
        "ret6m": float(row["ret6m"]),
        "ret12m": float(row["ret12m"]),
        "vol6m": float(row["vol6m"]),
        "dist52": float(row["dist52"]),
        "dist20res": float(row["dist20res"]),
        "high20": float(row["high20"]),
    }
