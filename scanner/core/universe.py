import pandas as pd
import requests
from io import StringIO

INDEX_URLS = {
    "NIFTY50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTYNEXT50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "NIFTYMIDCAP150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "NIFTYSMALLCAP250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC",
    "HINDUNILVR", "LT", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "AXISBANK", "MARUTI", "SUNPHARMA", "M&M", "BAJFINANCE",
    "HCLTECH", "TITAN", "ULTRACEMCO", "ASIANPAINT", "ADANIENT",
    "ADANIPORTS", "NTPC", "POWERGRID", "BEL", "TRENT",
]


def _get_csv(url):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*",
            "Referer": "https://www.niftyindices.com/",
        },
        timeout=25,
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def get_universe():
    frames = []
    failures = []

    for name, url in INDEX_URLS.items():
        try:
            df = _get_csv(url)
            df.columns = [str(c).strip() for c in df.columns]
            if "Symbol" not in df.columns:
                raise ValueError("Symbol column missing")

            sector_col = next(
                (c for c in ("Industry", "Sector") if c in df.columns),
                None,
            )
            sectors = (
                df[sector_col].fillna("Unknown").astype(str).str.strip()
                if sector_col
                else pd.Series("Unknown", index=df.index)
            )

            part = pd.DataFrame({
                "symbol": df["Symbol"].astype(str).str.upper().str.strip(),
                "index": name,
                "sector": sectors,
            })
            part = part[
                part["symbol"].str.fullmatch(r"[A-Z0-9&.-]+", na=False)
            ]
            frames.append(part)
            print(f"Universe source {name}: {len(part)} stocks loaded")
        except Exception as exc:
            failures.append(name)
            print(f"Universe source {name} failed: {exc}")

    if not frames:
        print("WARNING: all index downloads failed; using emergency fallback.")
        universe = pd.DataFrame({
            "symbol": FALLBACK,
            "index": "FALLBACK",
            "sector": "Unknown",
        })
    else:
        universe = pd.concat(frames, ignore_index=True)
        priority = {
            "NIFTY50": 0,
            "NIFTYNEXT50": 1,
            "NIFTYMIDCAP150": 2,
            "NIFTYSMALLCAP250": 3,
        }
        universe["_priority"] = universe["index"].map(priority).fillna(99)
        universe = (
            universe.sort_values("_priority")
            .drop_duplicates("symbol")
            .drop(columns="_priority")
            .reset_index(drop=True)
        )

    universe["ticker"] = universe["symbol"] + ".NS"
    print(
        f"Universe total: {len(universe)} unique stocks from "
        f"{universe['index'].nunique()} available index lists."
    )
    if failures:
        print("WARNING: unavailable index lists: " + ", ".join(failures))
    return universe
