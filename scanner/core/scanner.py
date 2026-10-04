from pathlib import Path

import numpy as np
import pandas as pd

from .universe import get_universe
from .market_data import (
    download_prices,
    series_for,
    indicators,
    latest_metrics,
)
from .fundamentals import get_fundamentals
from .scoring import (
    score_fundamentals,
    technical_scores,
)


OUTPUT_DIR = Path("output")


def run():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = get_universe()

    tickers = universe[
        "ticker"
    ].tolist()

    data = download_prices(
        tickers
    )

    rows = []

    nifty_return_6m = 0.0

    try:

        import yfinance as yf

        nifty = yf.download(
            "^NSEI",
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if isinstance(
            nifty.columns,
            pd.MultiIndex,
        ):

            nifty = nifty.xs(
                "Close",
                axis=1,
                level=0,
            )

            nifty = nifty.iloc[:, 0]

        else:

            nifty = (
                nifty["Close"]
                if "Close" in nifty
                else nifty.iloc[:, 0]
            )

        nifty = nifty.dropna()

        if len(nifty) > 126:

            nifty_return_6m = float(
                nifty.pct_change(126).iloc[-1]
            )

    except Exception as error:

        print(
            "Nifty benchmark unavailable:",
            error,
        )

    for _, stock in universe.iterrows():

        ticker = stock["ticker"]

        try:

            close = series_for(
                data,
                ticker,
                "Close",
            )

            high = series_for(
                data,
                ticker,
                "High",
            )

            low = series_for(
                data,
                ticker,
                "Low",
            )

            volume = series_for(
                data,
                ticker,
                "Volume",
            )

            price_df = pd.concat(
                {
                    "Close": close,
                    "High": high,
                    "Low": low,
                    "Volume": volume,
                },
                axis=1,
            ).dropna()

            ind = indicators(
                price_df
            )

            if ind is None:
                continue

            metrics = latest_metrics(
                ind
            )

            metrics.update(
                {
                    "symbol": stock["symbol"],
                    "ticker": ticker,
                    "index": stock["index"],
                }
            )

            fundamentals = get_fundamentals(
                ticker
            )

            metrics.update(
                fundamentals
            )

            metrics[
                "sma200_20ago"
            ] = float(
                ind["sma200"].iloc[-21]
            )

            metrics[
                "sma50_20ago"
            ] = float(
                ind["sma50"].iloc[-21]
            )

            metrics[
                "sma20_20ago"
            ] = float(
                ind["sma20"].iloc[-21]
            )

            rows.append(metrics)

        except Exception as error:

            print(
                f"Skipping {ticker}: {error}"
            )

    if not rows:

        raise RuntimeError(
            "No stocks could be processed."
        )

    df = pd.DataFrame(rows)

    # FUNDAMENTAL SCORING

    df = score_fundamentals(
        df
    )

    # TECHNICAL SCORING

    df = technical_scores(
        df,
        nifty_return_6m,
    )

    # Temporary sector proxy.
    #
    # The next integration step should replace this
    # with true NSE sector membership and sector
    # breadth data.

    df["sector_score"] = (
        df["ret3m"]
        .rank(pct=True)
        * 100
    ).round(1)

    df["sector_class"] = pd.cut(
        df["sector_score"],
        bins=[
            -1,
            50,
            70,
            80,
            101,
        ],
        labels=[
            "Weak",
            "Neutral",
            "Improving",
            "Strong",
        ],
    ).astype(str)

    def sector_bonus(score):

        if score >= 80:
            return 5

        if score >= 70:
            return 3

        if score >= 50:
            return 0

        return -5

    df["sector_bonus"] = (
        df["sector_score"]
        .apply(sector_bonus)
    )

    df["final_score"] = (
        df["overall_score"]
        + df["sector_bonus"]
    ).clip(0, 100)

    # SETUP CLASSIFICATION

    df["setup"] = np.select(
        [
            (
                (df["final_score"] >= 80)
                &
                (df["base_score"] >= 9)
                &
                (df["momentum_score"] >= 24)
                &
                (df["sector_score"] >= 70)
                &
                (df["breakout_score"] > 0)
            ),

            (
                (df["final_score"] >= 80)
                &
                (df["base_score"] >= 9)
                &
                (df["momentum_score"] >= 24)
                &
                (df["sector_score"] >= 70)
            ),

            df["final_score"] >= 70,
        ],

        [
            "BREAKOUT",
            "PRE-BREAKOUT",
            "WATCH",
        ],

        default="REJECT",
    )

    # MASTER OUTPUT COLUMNS

    columns = [

        "symbol",
        "index",

        "price",

        "final_score",
        "overall_score",
        "early_score",

        "value_score",
        "quality_score",
        "momentum_score",
        "base_score",

        "sector_score",
        "sector_class",

        "setup",

        "ret1m",
        "ret3m",
        "ret6m",
        "ret12m",
        "relative_strength",

        "sma20",
        "sma50",
        "sma200",

        "dist52",
        "dist20res",

        "atr_pct",
        "vol20",
    ]

    for column in columns:

        if column not in df.columns:
            df[column] = np.nan

    # MASTER TOP 30

    df = df.sort_values(
        [
            "final_score",
            "early_score",
        ],
        ascending=False,
    )

    df[
        columns
    ].head(30).to_csv(
        OUTPUT_DIR
        / "master_top30.csv",
        index=False,
    )

    # EARLY MOMENTUM

    df.sort_values(
        "early_score",
        ascending=False,
    )[
        columns
    ].head(10).to_csv(
        OUTPUT_DIR
        / "top10_early_momentum.csv",
        index=False,
    )

    # VALUE + MOMENTUM

    df.sort_values(
        "final_score",
        ascending=False,
    )[
        columns
    ].head(10).to_csv(
        OUTPUT_DIR
        / "top10_value_momentum.csv",
        index=False,
    )

    # BREAKOUT READY

    breakout = df[
        df["setup"].isin(
            [
                "BREAKOUT",
                "PRE-BREAKOUT",
            ]
        )
    ].sort_values(
        "final_score",
        ascending=False,
    )

    breakout[
        columns
    ].head(10).to_csv(
        OUTPUT_DIR
        / "top10_breakout_ready.csv",
        index=False,
    )

    create_telegram_message(
        df
    )

    print(
        "Trading OS Value-Momentum "
        "scan completed."
    )


def create_telegram_message(df):

    top10 = df.head(10)

    lines = [

        "🚀 TRADING OS "
        "VALUE-MOMENTUM v1.0",

        "",

        f"Stocks processed: {len(df)}",

        "",
    ]

    for rank, (
        _,
        row,
    ) in enumerate(
        top10.iterrows(),
        1,
    ):

        lines.extend(
            [

                (
                    f"{rank}. "
                    f"{row['symbol']} | "
                    f"Score {row['final_score']:.0f} | "
                    f"Early {row['early_score']:.0f}"
                ),

                (
                    f"   Value "
                    f"{row['value_score']:.0f}/30 | "
                    f"Quality "
                    f"{row['quality_score']:.0f}/20 | "
                    f"Momentum "
                    f"{row['momentum_score']:.0f}/35 | "
                    f"Base "
                    f"{row['base_score']:.0f}/15"
                ),

                (
                    f"   Setup: "
                    f"{row['setup']} | "
                    f"Sector "
                    f"{row['sector_score']:.0f}"
                ),

                (
                    f"   Price ₹"
                    f"{row['price']:.2f} | "
                    f"6M "
                    f"{row['ret6m'] * 100:.1f}% | "
                    f"52W gap "
                    f"{row['dist52'] * 100:.1f}%"
                ),

                "",
            ]
        )

    message = "\n".join(
        lines
    )

    (
        OUTPUT_DIR
        / "telegram_message.txt"
    ).write_text(
        message,
        encoding="utf-8",
    )
