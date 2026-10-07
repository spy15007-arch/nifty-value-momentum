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


# =========================================================
# HELPERS
# =========================================================

def _safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# =========================================================
# SECTOR STRENGTH
# =========================================================

def calculate_sector_scores(df):

    df = df.copy()

    # -----------------------------------------------------
    # Remove any existing sector_score.
    #
    # This prevents pandas from creating:
    # sector_score_x / sector_score_y
    # -----------------------------------------------------

    if "sector_score" in df.columns:

        df = df.drop(
            columns=["sector_score"]
        )

    if "sector" not in df.columns:

        df["sector"] = "Unknown"

    df["sector"] = (
        df["sector"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Make sure required columns exist
    # -----------------------------------------------------

    for column in [
        "ret3m",
        "ret6m",
        "relative_strength",
    ]:

        if column not in df.columns:

            df[column] = 0.0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Calculate sector statistics
    # -----------------------------------------------------

    sector_stats = (
        df.groupby(
            "sector",
            dropna=False,
        )
        .agg(
            sector_ret3m=(
                "ret3m",
                "mean",
            ),

            sector_ret6m=(
                "ret6m",
                "mean",
            ),

            sector_rs=(
                "relative_strength",
                "mean",
            ),

            sector_count=(
                "symbol",
                "count",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Percentile helper
    # -----------------------------------------------------

    def percentile(series):

        series = pd.to_numeric(
            series,
            errors="coerce",
        )

        if series.notna().sum() <= 1:

            return pd.Series(
                50.0,
                index=series.index,
            )

        result = (
            series
            .rank(
                pct=True,
                method="average",
            )
            * 100
        )

        return result.fillna(50.0)

    # -----------------------------------------------------
    # Sector components
    #
    # 30% = 3-month sector strength
    # 40% = 6-month sector strength
    # 30% = relative strength
    # -----------------------------------------------------

    sector_stats["score_3m"] = percentile(
        sector_stats["sector_ret3m"]
    )

    sector_stats["score_6m"] = percentile(
        sector_stats["sector_ret6m"]
    )

    sector_stats["score_rs"] = percentile(
        sector_stats["sector_rs"]
    )

    sector_stats["sector_score"] = (

        sector_stats["score_3m"] * 0.30

        + sector_stats["score_6m"] * 0.40

        + sector_stats["score_rs"] * 0.30

    )

    # -----------------------------------------------------
    # Very small / unknown sectors
    # -----------------------------------------------------

    sector_stats.loc[
        sector_stats["sector_count"] < 2,
        "sector_score",
    ] = 50.0

    sector_stats["sector_score"] = (
        sector_stats["sector_score"]
        .fillna(50.0)
        .clip(0, 100)
    )

    # -----------------------------------------------------
    # Merge sector score back
    # -----------------------------------------------------

    sector_scores = sector_stats[
        [
            "sector",
            "sector_score",
        ]
    ].copy()

    df = df.merge(
        sector_scores,
        on="sector",
        how="left",
        suffixes=(
            "",
            "_new",
        ),
    )

    # -----------------------------------------------------
    # Safety handling in case pandas creates
    # sector_score_new
    # -----------------------------------------------------

    if "sector_score_new" in df.columns:

        df["sector_score"] = (
            df["sector_score_new"]
            .fillna(50.0)
        )

        df = df.drop(
            columns=[
                "sector_score_new"
            ]
        )

    elif "sector_score" not in df.columns:

        df["sector_score"] = 50.0

    # -----------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------

    df["sector_score"] = (
        pd.to_numeric(
            df["sector_score"],
            errors="coerce",
        )
        .fillna(50.0)
        .clip(0, 100)
    )

    return df


# =========================================================
# MAIN SCANNER
# =========================================================

def run():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # LOAD NIFTY 50 + NEXT 50
    # -----------------------------------------------------

    universe = get_universe()

    tickers = (
        universe["ticker"]
        .tolist()
    )

    print(
        f"Universe loaded: {len(tickers)} stocks"
    )

    # -----------------------------------------------------
    # DOWNLOAD PRICE DATA
    # -----------------------------------------------------

    data = download_prices(
        tickers
    )

    rows = []

    nifty_return_6m = 0.0

    # -----------------------------------------------------
    # NIFTY 50 BENCHMARK
    # -----------------------------------------------------

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

            try:

                nifty = nifty.xs(
                    "Close",
                    axis=1,
                    level=0,
                )

                nifty = nifty.iloc[
                    :,
                    0,
                ]

            except Exception:

                nifty = nifty.iloc[
                    :,
                    0,
                ]

        else:

            if "Close" in nifty.columns:

                nifty = nifty[
                    "Close"
                ]

            else:

                nifty = nifty.iloc[
                    :,
                    0,
                ]

        nifty = nifty.dropna()

        if len(nifty) > 126:

            nifty_return_6m = float(
                nifty
                .pct_change(126)
                .iloc[-1]
            )

    except Exception as error:

        print(
            "Nifty benchmark unavailable:",
            error,
        )

    # -----------------------------------------------------
    # PROCESS EACH STOCK
    # -----------------------------------------------------

    processed = 0

    for _, stock in universe.iterrows():

        ticker = stock["ticker"]

        try:

            # ---------------------------------------------
            # PRICE SERIES
            # ---------------------------------------------

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

            # ---------------------------------------------
            # MINIMUM HISTORY
            # ---------------------------------------------

            if len(price_df) < 220:

                continue

            # ---------------------------------------------
            # TECHNICAL INDICATORS
            # ---------------------------------------------

            ind = indicators(
                price_df
            )

            if ind is None:

                continue

            # ---------------------------------------------
            # LATEST METRICS
            # ---------------------------------------------

            metrics = latest_metrics(
                ind
            )

            # ---------------------------------------------
            # FIX vol20_prev
            # ---------------------------------------------

            if "vol20_prev" in ind.columns:

                metrics[
                    "vol20_prev"
                ] = _safe_float(
                    ind[
                        "vol20_prev"
                    ].iloc[-1],
                    0.0,
                )

            else:

                metrics[
                    "vol20_prev"
                ] = _safe_float(
                    metrics.get(
                        "vol20",
                        0.0,
                    ),
                    0.0,
                )

            # ---------------------------------------------
            # UNIVERSE INFORMATION
            # ---------------------------------------------

            metrics.update(
                {
                    "symbol": stock[
                        "symbol"
                    ],

                    "ticker": ticker,

                    "index": stock[
                        "index"
                    ],

                    "sector": stock.get(
                        "sector",
                        "Unknown",
                    ),
                }
            )

            # ---------------------------------------------
            # FUNDAMENTAL DATA
            # ---------------------------------------------

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

            processed += 1

            if processed % 10 == 0:

                print(
                    f"Processed {processed} stocks..."
                )

        except Exception as error:

            print(
                f"Skipping {ticker}: {error}"
            )

    # -----------------------------------------------------
    # BUILD DATAFRAME
    # -----------------------------------------------------

    if not rows:

        raise RuntimeError(
            "No stocks could be processed."
        )

    df = pd.DataFrame(
        rows
    )

    print(
        f"Successfully processed: {len(df)} stocks"
    )

    # -----------------------------------------------------
    # FUNDAMENTAL SCORING
    # -----------------------------------------------------

    df = score_fundamentals(
        df
    )

    # -----------------------------------------------------
    # SECTOR STRENGTH
    # -----------------------------------------------------

    df = calculate_sector_scores(
        df
    )

    # -----------------------------------------------------
    # TECHNICAL SCORING
    # -----------------------------------------------------

    df = technical_scores(
        df,
        nifty_return_6m,
    )

    # -----------------------------------------------------
    # ENSURE FINAL SCORE EXISTS
    # -----------------------------------------------------

    if "final_score" not in df.columns:

        df["final_score"] = (
            df["overall_score"]
            .fillna(0)
        )

    # -----------------------------------------------------
    # CLEAN SCORE COLUMNS
    # -----------------------------------------------------

    for column in [
        "final_score",
        "early_score",
        "value_score",
        "quality_score",
        "momentum_score",
        "base_score",
        "sector_score",
    ]:

        if column not in df.columns:

            df[column] = 0.0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    # -----------------------------------------------------
    # FINAL RANKING
    #
    # Score first.
    # Early setup second.
    # Momentum third.
    # NOT alphabetical.
    # -----------------------------------------------------

    df = df.sort_values(
        [
            "final_score",
            "early_score",
            "momentum_score",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    df["rank"] = (
        df.index + 1
    )

    # -----------------------------------------------------
    # OUTPUT COLUMNS
    # -----------------------------------------------------

    columns = [

        "rank",

        "symbol",
        "index",
        "sector",

        "price",

        "final_score",
        "early_score",

        "value_score",
        "quality_score",
        "momentum_score",
        "base_score",

        "sector_score",

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
        "vol20_prev",

        "entry",
        "stop",

        "t1",
        "t2",
        "t3",

        "rr_t1",
        "rr_t2",
        "rr_t3",
    ]

    # Add missing columns safely.

    for column in columns:

        if column not in df.columns:

            df[column] = np.nan

    # -----------------------------------------------------
    # MASTER TOP 30
    # -----------------------------------------------------

    master = df[
        columns
    ].head(30)

    master.to_csv(
        OUTPUT_DIR
        / "master_top30.csv",
        index=False,
    )

    # -----------------------------------------------------
    # QUALIFIED SETUPS
    #
    # ONLY:
    # BREAKOUT
    # PRE-BREAKOUT
    # -----------------------------------------------------

    qualified = df[
        df["setup"].isin(
            [
                "BREAKOUT",
                "PRE-BREAKOUT",
            ]
        )
    ].copy()

    qualified = qualified.sort_values(
        [
            "final_score",
            "early_score",
        ],
        ascending=False,
    )

    qualified[
        columns
    ].head(15).to_csv(
        OUTPUT_DIR
        / "qualified_setups.csv",
        index=False,
    )

    # -----------------------------------------------------
    # EARLY MOMENTUM
    # -----------------------------------------------------

    early = df[
        df["setup"].isin(
            [
                "PRE-BREAKOUT",
                "WATCH",
            ]
        )
    ].sort_values(
        "early_score",
        ascending=False,
    )

    early[
        columns
    ].head(10).to_csv(
        OUTPUT_DIR
        / "top10_early_momentum.csv",
        index=False,
    )

    # -----------------------------------------------------
    # VALUE + MOMENTUM
    # -----------------------------------------------------

    vm = df[
        (
            df["value_score"]
            >= 18
        )
        &
        (
            df["momentum_score"]
            >= 20
        )
    ].sort_values(
        "final_score",
        ascending=False,
    )

    vm[
        columns
    ].head(10).to_csv(
        OUTPUT_DIR
        / "top10_value_momentum.csv",
        index=False,
    )

    # -----------------------------------------------------
    # CONFIRMED BREAKOUT
    # -----------------------------------------------------

    breakout = df[
        df["setup"]
        == "BREAKOUT"
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

    # -----------------------------------------------------
    # TELEGRAM MESSAGE
    # -----------------------------------------------------

    create_telegram_message(
        df
    )

    # -----------------------------------------------------
    # CONSOLE SUMMARY
    # -----------------------------------------------------

    qualified_count = len(
        qualified
    )

    breakout_count = len(
        breakout
    )

    pre_breakout_count = len(
        df[
            df["setup"]
            == "PRE-BREAKOUT"
        ]
    )

    watch_count = len(
        df[
            df["setup"]
            == "WATCH"
        ]
    )

    print("")
    print(
        "======================================"
    )
    print(
        "VALUE-MOMENTUM SCANNER COMPLETE"
    )
    print(
        "======================================"
    )

    print(
        f"Stocks processed : {len(df)}"
    )

    print(
        f"Qualified setups : {qualified_count}"
    )

    print(
        f"Breakouts        : {breakout_count}"
    )

    print(
        f"Pre-breakouts    : {pre_breakout_count}"
    )

    print(
        f"Watch candidates : {watch_count}"
    )

    if len(df) > 0:

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


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

def create_telegram_message(df):

    # -----------------------------------------------------
    # QUALIFIED
    # -----------------------------------------------------

    qualified = df[
        df["setup"].isin(
            [
                "BREAKOUT",
                "PRE-BREAKOUT",
            ]
        )
    ].copy()

    # -----------------------------------------------------
    # WATCH
    # -----------------------------------------------------

    watch = df[
        df["setup"]
        == "WATCH"
    ].copy()

    # -----------------------------------------------------
    # MESSAGE HEADER
    # -----------------------------------------------------

    lines = [

        "🚀 VALUE-MOMENTUM DAILY",

        "NIFTY 50 + NEXT 50",

        "",

        "🔥 QUALIFIED SETUPS",

        "",
    ]

    # -----------------------------------------------------
    # QUALIFIED SECTION
    # -----------------------------------------------------

    if qualified.empty:

        lines.extend(
            [
                "NO QUALIFIED BREAKOUT/",
                "PRE-BREAKOUT SETUPS TODAY.",
                "",
                "This is a NO-TRADE signal.",
                "",
            ]
        )

    else:

        for rank, (
            _,
            row,
        ) in enumerate(
            qualified.head(10).iterrows(),
            1,
        ):

            lines.extend(
                [

                    (
                        f"#{rank} "
                        f"{row['symbol']} | "
                        f"{row['setup']}"
                    ),

                    (
                        f"Score "
                        f"{row['final_score']:.0f} | "
                        f"Early "
                        f"{row['early_score']:.0f}"
                    ),

                    (
                        f"Value "
                        f"{row['value_score']:.0f}/30 | "
                        f"Quality "
                        f"{row['quality_score']:.0f}/20"
                    ),

                    (
                        f"Momentum "
                        f"{row['momentum_score']:.0f}/35 | "
                        f"Base "
                        f"{row['base_score']:.0f}/15"
                    ),

                    (
                        f"Sector "
                        f"{row['sector_score']:.0f} | "
                        f"6M "
                        f"{row['ret6m'] * 100:.1f}%"
                    ),

                    (
                        f"Entry ₹"
                        f"{row['entry']:.2f} | "
                        f"SL ₹"
                        f"{row['stop']:.2f}"
                    ),

                    (
                        f"T1 ₹"
                        f"{row['t1']:.2f} | "
                        f"T2 ₹"
                        f"{row['t2']:.2f} | "
                        f"T3 ₹"
                        f"{row['t3']:.2f}"
                    ),

                    (
                        f"R:R "
                        f"{row['rr_t2']:.1f}"
                    ),

                    "",
                ]
            )

    # -----------------------------------------------------
    # WATCH SECTION
    # -----------------------------------------------------

    lines.extend(
        [
            "👀 WATCH — DEVELOPING",
            "",
        ]
    )

    if watch.empty:

        lines.append(
            "No watch candidates."
        )

    else:

        for rank, (
            _,
            row,
        ) in enumerate(
            watch.head(5).iterrows(),
            1,
        ):

            lines.append(
                (
                    f"{rank}. "
                    f"{row['symbol']} | "
                    f"Score "
                    f"{row['final_score']:.0f} | "
                    f"Early "
                    f"{row['early_score']:.0f} | "
                    f"Sector "
                    f"{row['sector_score']:.0f}"
                )
            )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    lines.extend(
        [
            "",
            "❌ REJECT stocks hidden.",
            "",
            "System:",
            "Value + Quality + Momentum + Base",
            "→ wait for confirmation.",
        ]
    )

    message = "\n".join(
        lines
    )

    # -----------------------------------------------------
    # SAVE TELEGRAM MESSAGE
    # -----------------------------------------------------

    (
        OUTPUT_DIR
        / "telegram_message.txt"
    ).write_text(
        message,
        encoding="utf-8",
    )


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    run()
