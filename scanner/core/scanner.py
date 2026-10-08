from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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


# =========================================================
# HELPERS
# =========================================================

def _safe_float(
    value,
    default=0.0
):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# =========================================================
# NIFTY 6M BENCHMARK
# =========================================================

def get_nifty_6m_return():

    try:

        nifty = yf.download(
            "^NSEI",
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if isinstance(
            nifty.columns,
            pd.MultiIndex
        ):

            if "Close" in nifty.columns.get_level_values(0):

                close = (
                    nifty
                    .xs(
                        "Close",
                        axis=1,
                        level=0
                    )
                    .iloc[:, 0]
                )

            else:

                close = nifty.iloc[:, 0]

        else:

            if "Close" in nifty.columns:

                close = nifty["Close"]

            else:

                close = nifty.iloc[:, 0]


        close = (
            pd.to_numeric(
                close,
                errors="coerce"
            )
            .dropna()
        )


        if len(close) <= 126:

            return 0.0


        return float(
            close.pct_change(126).iloc[-1]
        )


    except Exception as error:

        print(
            "Nifty benchmark unavailable:",
            error
        )

        return 0.0


# =========================================================
# SECTOR STRENGTH
# =========================================================

def calculate_sector_scores(df):

    df = df.copy()


    if "sector" not in df.columns:

        df["sector"] = "Unknown"


    df["sector"] = (
        df["sector"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


    for column in [
        "ret3m",
        "ret6m",
        "relative_strength",
    ]:

        if column not in df.columns:

            df[column] = 0.0


        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0)


    stats = (

        df
        .groupby(
            "sector",
            dropna=False
        )
        .agg(

            sector_ret3m=(
                "ret3m",
                "mean"
            ),

            sector_ret6m=(
                "ret6m",
                "mean"
            ),

            sector_rs=(
                "relative_strength",
                "mean"
            ),

            sector_count=(
                "symbol",
                "count"
            )

        )
        .reset_index()

    )


    def percentile(series):

        series = pd.to_numeric(
            series,
            errors="coerce"
        )


        if series.notna().sum() <= 1:

            return pd.Series(
                50.0,
                index=series.index
            )


        return (

            series
            .rank(
                pct=True,
                method="average"
            )
            * 100

        ).fillna(50.0)


    stats["score_3m"] = percentile(
        stats["sector_ret3m"]
    )

    stats["score_6m"] = percentile(
        stats["sector_ret6m"]
    )

    stats["score_rs"] = percentile(
        stats["sector_rs"]
    )


    stats["sector_score"] = (

        stats["score_3m"] * 0.30

        + stats["score_6m"] * 0.40

        + stats["score_rs"] * 0.30

    )


    stats.loc[
        stats["sector_count"] < 2,
        "sector_score"
    ] = 50.0


    stats["sector_score"] = (
        stats["sector_score"]
        .fillna(50.0)
        .clip(0, 100)
    )


    return (

        df
        .drop(
            columns=["sector_score"],
            errors="ignore"
        )
        .merge(
            stats[
                [
                    "sector",
                    "sector_score"
                ]
            ],
            on="sector",
            how="left"
        )

    )


# =========================================================
# PROCESS STOCKS
# =========================================================

def process_stocks(
    universe,
    data,
    nifty_return_6m
):

    rows = []


    for _, stock in universe.iterrows():

        ticker = stock["ticker"]


        try:

            close = series_for(
                data,
                ticker,
                "Close"
            )

            high = series_for(
                data,
                ticker,
                "High"
            )

            low = series_for(
                data,
                ticker,
                "Low"
            )

            volume = series_for(
                data,
                ticker,
                "Volume"
            )


            price_df = pd.concat(
                {
                    "Close": close,
                    "High": high,
                    "Low": low,
                    "Volume": volume,
                },
                axis=1
            ).dropna()


            # More robust than the old 220-day minimum.
            if len(price_df) < 260:

                continue


            ind = indicators(
                price_df
            )


            if ind is None:

                continue


            metrics = latest_metrics(
                ind
            )


            # Fix previous volume-history issue.
            if "vol20_prev" in ind.columns:

                metrics["vol20_prev"] = (
                    _safe_float(
                        ind[
                            "vol20_prev"
                        ].iloc[-1],
                        metrics.get(
                            "vol20",
                            0
                        )
                    )
                )

            else:

                metrics["vol20_prev"] = (
                    metrics.get(
                        "vol20",
                        0
                    )
                )


            # -------------------------------------------------
            # UNIVERSE DATA
            # -------------------------------------------------

            metrics.update(
                {
                    "symbol":
                        stock["symbol"],

                    "ticker":
                        ticker,

                    "index":
                        stock["index"],

                    "sector":
                        stock.get(
                            "sector",
                            "Unknown"
                        ),
                }
            )


            # -------------------------------------------------
            # IMPORTANT:
            # ACTUAL RELATIVE STRENGTH VS NIFTY
            # -------------------------------------------------

            metrics[
                "relative_strength"
            ] = (

                metrics["ret6m"]

                - nifty_return_6m

            )


            # -------------------------------------------------
            # FUNDAMENTALS
            # -------------------------------------------------

            try:

                fundamentals = (
                    get_fundamentals(
                        ticker
                    )
                )


                if fundamentals:

                    metrics.update(
                        fundamentals
                    )


            except Exception as error:

                print(
                    f"Fundamental data unavailable "
                    f"for {ticker}: {error}"
                )


            rows.append(
                metrics
            )


        except Exception as error:

            print(
                f"Skipping {ticker}: {error}"
            )


    return rows


# =========================================================
# OUTPUT COLUMNS
# =========================================================

OUTPUT_COLUMNS = [

    "rank",
    "symbol",
    "index",
    "sector",

    "price",

    "final_score",
    "early_score",

    "value_score",
    "quality_score",
    "trend_score",
    "momentum_score",
    "base_score",
    "volume_score",

    "sector_score",
    "fundamental_coverage",

    "setup",
    "chase",

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
    "volume_ratio",

    "entry_low",
    "entry_ideal",
    "entry_high",

    "stop",

    "t1",
    "t2",
    "t3",
    "t4",

    "rr_t1",
    "rr_t2",
    "rr_t3",
    "rr_t4",

]


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

def create_telegram_message(df):

    qualified = (
        df[
            df["setup"].isin(
                [
                    "BREAKOUT",
                    "PRE-BREAKOUT"
                ]
            )
        ]
        .copy()
    )


    watch = (
        df[
            df["setup"] == "WATCH"
        ]
        .copy()
    )


    lines = [

        "🚀 VALUE-MOMENTUM INSTITUTIONAL",

        "NIFTY 50 + NEXT 50",

        "",

        "🎯 ACTIONABLE SETUPS",

        "",
    ]


    # =====================================================
    # NO TRADE
    # =====================================================

    if qualified.empty:

        lines.extend(
            [

                "NO HIGH-QUALITY SETUP TODAY.",

                "",

                "Market/stock conditions do not",
                "justify a fresh trade.",

                "",

                "NO-TRADE IS A VALID SIGNAL.",

                "",
            ]
        )


    else:

        # =================================================
        # QUALIFIED STOCKS
        # =================================================

        for rank, (
            _,
            row
        ) in enumerate(
            qualified.head(8).iterrows(),
            1
        ):


            icon = (

                "🔥"

                if row["setup"] ==
                "BREAKOUT"

                else

                "⭐"

            )


            lines.extend(
                [

                    (
                        f"{icon} #{rank} "
                        f"{row['symbol']} — "
                        f"{row['setup']}"
                    ),

                    (
                        f"Score "
                        f"{row['final_score']:.0f} | "
                        f"Sector "
                        f"{row['sector_score']:.0f}"
                    ),

                    (
                        f"BUY RANGE "
                        f"₹{row['entry_low']:.2f}"
                        f"–"
                        f"₹{row['entry_high']:.2f}"
                    ),

                    (
                        f"Ideal "
                        f"₹{row['entry_ideal']:.2f}"
                        f" | SL "
                        f"₹{row['stop']:.2f}"
                    ),

                    (
                        f"T1 ₹{row['t1']:.2f}"
                        f" | T2 ₹{row['t2']:.2f}"
                    ),

                    (
                        f"T3 ₹{row['t3']:.2f}"
                        f" | T4 ₹{row['t4']:.2f}"
                    ),

                    (
                        f"R:R "
                        f"1:{row['rr_t1']:.1f}"
                        f" / "
                        f"1:{row['rr_t2']:.1f}"
                        f" / "
                        f"1:{row['rr_t3']:.1f}"
                        f" / "
                        f"1:{row['rr_t4']:.1f}"
                    ),

                    "",
                ]
            )


    # =====================================================
    # WATCH
    # =====================================================

    if not watch.empty:

        lines.extend(
            [
                "👀 WATCH",
                "",
            ]
        )


        for _, row in watch.head(5).iterrows():

            lines.append(

                (
                    f"{row['symbol']} | "
                    f"Score "
                    f"{row['final_score']:.0f} | "
                    f"Buy zone "
                    f"₹{row['entry_low']:.2f}"
                    f"–"
                    f"₹{row['entry_high']:.2f}"
                )

            )


        lines.append("")


    # =====================================================
    # FOOTER
    # =====================================================

    lines.extend(
        [

            (
                f"Qualified: "
                f"{len(qualified)}"
                f" | Watch: "
                f"{len(watch)}"
            ),

            "",

            "Quality + Value + Trend + Momentum",

            "Base + Volume + Relative Strength",

            "",

            "Fresh buying only inside the stated range.",

            "Do not chase above the range.",

        ]
    )


    message = "\n".join(
        lines
    )


    (
        OUTPUT_DIR /
        "telegram_message.txt"
    ).write_text(
        message,
        encoding="utf-8"
    )


# =========================================================
# MAIN
# =========================================================

def run():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # UNIVERSE
    # -----------------------------------------------------

    universe = get_universe()


    tickers = (
        universe["ticker"]
        .tolist()
    )


    print(
        f"Universe loaded: "
        f"{len(tickers)} stocks"
    )


    # -----------------------------------------------------
    # PRICE DATA
    # -----------------------------------------------------

    data = download_prices(
        tickers
    )


    # -----------------------------------------------------
    # NIFTY BENCHMARK
    # -----------------------------------------------------

    nifty_return_6m = (
        get_nifty_6m_return()
    )


    print(
        f"Nifty 6M return: "
        f"{nifty_return_6m * 100:.2f}%"
    )


    # -----------------------------------------------------
    # PROCESS
    # -----------------------------------------------------

    rows = process_stocks(
        universe,
        data,
        nifty_return_6m
    )


    if not rows:

        raise RuntimeError(
            "No stocks could be processed."
        )


    df = pd.DataFrame(
        rows
    )


    print(
        f"Successfully processed: "
        f"{len(df)} stocks"
    )


    # -----------------------------------------------------
    # FUNDAMENTALS
    # -----------------------------------------------------

    df = score_fundamentals(
        df
    )


    # -----------------------------------------------------
    # SECTORS
    # -----------------------------------------------------

    df = calculate_sector_scores(
        df
    )


    # -----------------------------------------------------
    # TECHNICALS
    # -----------------------------------------------------

    df = technical_scores(
        df,
        nifty_return_6m
    )


    # -----------------------------------------------------
    # RANK
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "final_score",
                "early_score",
                "momentum_raw"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    df["rank"] = (
        df.index + 1
    )


    # -----------------------------------------------------
    # SAFETY: OUTPUT COLUMNS
    # -----------------------------------------------------

    for column in OUTPUT_COLUMNS:

        if column not in df.columns:

            df[column] = np.nan


    # -----------------------------------------------------
    # MASTER 30
    # -----------------------------------------------------

    df[
        OUTPUT_COLUMNS
    ].head(30).to_csv(

        OUTPUT_DIR /
        "master_top30.csv",

        index=False
    )


    # -----------------------------------------------------
    # QUALIFIED
    # -----------------------------------------------------

    qualified = (

        df[
            df["setup"].isin(
                [
                    "BREAKOUT",
                    "PRE-BREAKOUT"
                ]
            )
        ]
        .sort_values(
            [
                "final_score",
                "early_score"
            ],
            ascending=False
        )
    )


    qualified[
        OUTPUT_COLUMNS
    ].head(15).to_csv(

        OUTPUT_DIR /
        "qualified_setups.csv",

        index=False
    )


    # -----------------------------------------------------
    # EARLY
    # -----------------------------------------------------

    early = (

        df[
            df["setup"].isin(
                [
                    "PRE-BREAKOUT",
                    "WATCH"
                ]
            )
        ]
        .sort_values(
            "early_score",
            ascending=False
        )
    )


    early[
        OUTPUT_COLUMNS
    ].head(10).to_csv(

        OUTPUT_DIR /
        "top10_early_momentum.csv",

        index=False
    )


    # -----------------------------------------------------
    # VALUE + MOMENTUM
    # -----------------------------------------------------

    vm = df[
        (df["value_score"] >= 5.5)

        &

        (df["momentum_score"] >= 13.75)

        &

        (df["trend_score"] >= 14)
    ].sort_values(
        "final_score",
        ascending=False
    )


    vm[
        OUTPUT_COLUMNS
    ].head(10).to_csv(

        OUTPUT_DIR /
        "top10_value_momentum.csv",

        index=False
    )


    # -----------------------------------------------------
    # BREAKOUT
    # -----------------------------------------------------

    breakout = (

        df[
            df["setup"] ==
            "BREAKOUT"
        ]
        .sort_values(
            "final_score",
            ascending=False
        )
    )


    breakout[
        OUTPUT_COLUMNS
    ].head(10).to_csv(

        OUTPUT_DIR /
        "top10_breakout_ready.csv",

        index=False
    )


    # -----------------------------------------------------
    # TELEGRAM
    # -----------------------------------------------------

    create_telegram_message(
        df
    )


    # -----------------------------------------------------
    # CONSOLE
    # -----------------------------------------------------

    print("")
    print(
        "======================================"
    )

    print(
        "VALUE-MOMENTUM INSTITUTIONAL SCANNER"
    )

    print(
        "======================================"
    )

    print(
        f"Stocks processed : "
        f"{len(df)}"
    )

    print(
        f"Qualified setups : "
        f"{len(qualified)}"
    )

    print(
        f"Breakouts        : "
        f"{(df['setup'] == 'BREAKOUT').sum()}"
    )

    print(
        f"Pre-breakouts    : "
        f"{(df['setup'] == 'PRE-BREAKOUT').sum()}"
    )

    print(
        f"Watch candidates : "
        f"{(df['setup'] == 'WATCH').sum()}"
    )


    if not df.empty:

        print(
            f"Top stock        : "
            f"{df.iloc[0]['symbol']}"
        )

        print(
            f"Top score        : "
            f"{df.iloc[0]['final_score']:.1f}"
        )

        print(
            f"Top setup        : "
            f"{df.iloc[0]['setup']}"
        )


    print(
        "======================================"
    )


if __name__ == "__main__":

    run()
