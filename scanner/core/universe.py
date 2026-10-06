import pandas as pd
import requests
from io import StringIO


NIFTY50_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty50list.csv"
)

NEXT50_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_niftynext50list.csv"
)


def _get_csv(url):

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.niftyindices.com/",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    return pd.read_csv(
        StringIO(response.text)
    )


def _normalise_columns(df):

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


def get_universe():

    try:

        nifty50 = _normalise_columns(
            _get_csv(NIFTY50_URL)
        )

        next50 = _normalise_columns(
            _get_csv(NEXT50_URL)
        )

        frames = []

        for df, index_name in [
            (nifty50, "NIFTY50"),
            (next50, "NIFTYNEXT50"),
        ]:

            if "Symbol" not in df.columns:

                raise ValueError(
                    f"Symbol column missing from {index_name}"
                )

            if "Industry" in df.columns:

                industry = (
                    df["Industry"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                )

            else:

                industry = "Unknown"

            temp = pd.DataFrame(
                {
                    "symbol": (
                        df["Symbol"]
                        .astype(str)
                        .str.upper()
                        .str.strip()
                    ),

                    "index": index_name,

                    "sector": industry,
                }
            )

            frames.append(temp)

        universe = pd.concat(
            frames,
            ignore_index=True,
        )

        universe = (
            universe
            .drop_duplicates(
                subset=["symbol"]
            )
            .reset_index(drop=True)
        )

        universe["ticker"] = (
            universe["symbol"] + ".NS"
        )

        return universe

    except Exception as error:

        print(
            "NSE universe download failed:",
            error,
        )

        # Small emergency fallback.
        # The normal scanner should use the
        # current Nifty 50 + Next 50 files.

        symbols = [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "ITC",
            "HINDUNILVR",
            "LT",
            "SBIN",
            "BHARTIARTL",
            "KOTAKBANK",
            "AXISBANK",
            "MARUTI",
            "SUNPHARMA",
            "M&M",
            "BAJFINANCE",
            "HCLTECH",
            "TITAN",
            "ULTRACEMCO",
            "ASIANPAINT",
            "ADANIENT",
            "ADANIPORTS",
            "NTPC",
            "POWERGRID",
            "BEL",
            "TRENT",
        ]

        return pd.DataFrame(
            {
                "symbol": symbols,

                "index": [
                    "FALLBACK"
                    for _ in symbols
                ],

                "sector": [
                    "Unknown"
                    for _ in symbols
                ],

                "ticker": [
                    symbol + ".NS"
                    for symbol in symbols
                ],
            }
        )
