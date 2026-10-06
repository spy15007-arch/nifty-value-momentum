import numpy as np
import pandas as pd


# =========================================================
# HELPERS
# =========================================================

def _num(df, column, default=0.0):

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def _normalise(series, low=0, high=100):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    if series.notna().sum() <= 1:

        return pd.Series(
            50.0,
            index=series.index,
        )

    minimum = series.quantile(0.10)
    maximum = series.quantile(0.90)

    if maximum <= minimum:

        return pd.Series(
            50.0,
            index=series.index,
        )

    result = (
        (series - minimum)
        / (maximum - minimum)
    )

    return (
        result
        .clip(0, 1)
        * (high - low)
        + low
    )


def _percentile_score(
    series,
    higher_is_better=True,
):

    score = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .rank(
            pct=True,
            ascending=higher_is_better,
        )
        * 100
    )

    return score.fillna(50)


# =========================================================
# FUNDAMENTAL SCORE
# =========================================================

def score_fundamentals(df):

    df = df.copy()

    # -----------------------------------------------------
    # VALUE — 30 POINTS
    # -----------------------------------------------------

    pe = _num(
        df,
        "pe",
        np.nan,
    )

    pb = _num(
        df,
        "pb",
        np.nan,
    )

    ev_ebitda = _num(
        df,
        "ev_ebitda",
        np.nan,
    )

    earnings_yield = _num(
        df,
        "earnings_yield",
        np.nan,
    )

    value_parts = []

    if pe.notna().any():

        pe_clean = pe.where(
            (pe > 0) & (pe < 100)
        )

        value_parts.append(
            _percentile_score(
                -pe_clean
            )
        )

    if pb.notna().any():

        pb_clean = pb.where(
            (pb > 0) & (pb < 30)
        )

        value_parts.append(
            _percentile_score(
                -pb_clean
            )
        )

    if ev_ebitda.notna().any():

        ev_clean = ev_ebitda.where(
            (ev_ebitda > 0)
            & (ev_ebitda < 100)
        )

        value_parts.append(
            _percentile_score(
                -ev_clean
            )
        )

    if earnings_yield.notna().any():

        value_parts.append(
            _percentile_score(
                earnings_yield
            )
        )

    if value_parts:

        value_score = pd.concat(
            value_parts,
            axis=1,
        ).mean(axis=1)

    else:

        # If fundamental data is unavailable,
        # don't falsely call a stock "cheap".
        value_score = pd.Series(
            50.0,
            index=df.index,
        )

    df["value_score"] = (
        value_score
        .clip(0, 100)
        * 0.30
    )

    # -----------------------------------------------------
    # QUALITY — 20 POINTS
    # -----------------------------------------------------

    roe = _num(
        df,
        "roe",
        np.nan,
    )

    roce = _num(
        df,
        "roce",
        np.nan,
    )

    debt_equity = _num(
        df,
        "debt_to_equity",
        np.nan,
    )

    profit_margin = _num(
        df,
        "profit_margin",
        np.nan,
    )

    quality_parts = []

    if roe.notna().any():

        quality_parts.append(
            _percentile_score(
                roe
            )
        )

    if roce.notna().any():

        quality_parts.append(
            _percentile_score(
                roce
            )
        )

    if profit_margin.notna().any():

        quality_parts.append(
            _percentile_score(
                profit_margin
            )
        )

    if debt_equity.notna().any():

        debt_clean = debt_equity.where(
            debt_equity >= 0
        )

        quality_parts.append(
            _percentile_score(
                -debt_clean
            )
        )

    if quality_parts:

        quality_score = pd.concat(
            quality_parts,
            axis=1,
        ).mean(axis=1)

    else:

        quality_score = pd.Series(
            50.0,
            index=df.index,
        )

    df["quality_score"] = (
        quality_score
        .clip(0, 100)
        * 0.20
    )

    # -----------------------------------------------------
    # PLACEHOLDER TOTAL
    # -----------------------------------------------------

    df["fundamental_score"] = (
        df["value_score"]
        + df["quality_score"]
    )

    return df


# =========================================================
# TECHNICAL / MOMENTUM SCORE
# =========================================================

def technical_scores(
    df,
    nifty_return_6m=0.0,
):

    df = df.copy()

    # -----------------------------------------------------
    # RETURNS
    # -----------------------------------------------------

    ret1m = _num(
        df,
        "ret1m",
    )

    ret3m = _num(
        df,
        "ret3m",
    )

    ret6m = _num(
        df,
        "ret6m",
    )

    ret12m = _num(
        df,
        "ret12m",
    )

    relative_strength = (
        _num(
            df,
            "relative_strength",
        )
    )

    # Ensure relative_strength is in the output dataframe
    df["relative_strength"] = relative_strength

    # -----------------------------------------------------
    # MOMENTUM — 35 POINTS
    # -----------------------------------------------------

    m1 = _percentile_score(
        ret1m
    )

    m3 = _percentile_score(
        ret3m
    )

    m6 = _percentile_score(
        ret6m
    )

    m12 = _percentile_score(
        ret12m
    )

    rs = _percentile_score(
        relative_strength
    )

    momentum_raw = (
        m1 * 0.10
        + m3 * 0.20
        + m6 * 0.30
        + m12 * 0.15
        + rs * 0.25
    )

    df["momentum_score"] = (
        momentum_raw
        * 0.35
    )

    # -----------------------------------------------------
    # MOVING AVERAGE STRUCTURE
    # -----------------------------------------------------

    price = _num(
        df,
        "price",
    )

    sma20 = _num(
        df,
        "sma20",
    )

    sma50 = _num(
        df,
        "sma50",
    )

    sma200 = _num(
        df,
        "sma200",
    )

    trend_points = (

        (price > sma200)
        .astype(int)

        + (sma50 > sma200)
        .astype(int)

        + (sma20 > sma50)
        .astype(int)

    )

    # -----------------------------------------------------
    # DISTANCE FROM RESISTANCE
    # -----------------------------------------------------

    dist20res = _num(
        df,
        "dist20res",
    )

    dist52 = _num(
        df,
        "dist52",
    )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    vol20 = _num(
        df,
        "vol20",
    )

    vol20_prev = _num(
        df,
        "vol20_prev",
    )

    volume_ratio = (
        vol20
        / vol20_prev.replace(
            0,
            np.nan,
        )
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1.0)

    # -----------------------------------------------------
    # BASE / BREAKOUT — 15 POINTS
    # -----------------------------------------------------

    # Ideal pre-breakout:
    # close near resistance,
    # but not excessively extended.

    resistance_proximity = (
        1
        - (
            dist20res
            .abs()
            .clip(0, 0.15)
            / 0.15
        )
    ) * 100

    resistance_proximity = (
        resistance_proximity
        .clip(0, 100)
    )

    # Avoid chasing stocks already far above
    # their recent breakout zone.

    extension_penalty = (
        (
            dist20res > 0.05
        )
        .astype(int)
        * 25
    )

    volume_score = (
        (
            volume_ratio
            .clip(0.5, 2.5)
            - 0.5
        )
        / 2.0
        * 100
    ).clip(
        0,
        100,
    )

    trend_score = (
        trend_points
        / 3
        * 100
    )

    base_raw = (
        resistance_proximity * 0.45
        + volume_score * 0.20
        + trend_score * 0.20
        + _percentile_score(
            -dist52
        ) * 0.15
    )

    base_raw = (
        base_raw
        - extension_penalty
    ).clip(
        0,
        100,
    )

    df["base_score"] = (
        base_raw
        * 0.15
    )

    # -----------------------------------------------------
    # BREAKOUT SCORE
    # -----------------------------------------------------

    breakout_signal = (

        (dist20res >= 0)
        & (dist20res <= 0.03)
        & (volume_ratio >= 1.20)
        & (price > sma20)
        & (price > sma50)

    )

    df["breakout_score"] = np.where(
        breakout_signal,
        100,
        0,
    )

    # -----------------------------------------------------
    # EARLY MOMENTUM
    # -----------------------------------------------------

    early_raw = (

        momentum_raw * 0.55

        + trend_score * 0.20

        + resistance_proximity * 0.25

    )

    df["early_score"] = (
        early_raw
        .clip(0, 100)
    )

    # -----------------------------------------------------
    # BASE SCORE IN RAW 0-100 FORM
    # -----------------------------------------------------

    df["base_raw_score"] = (
        base_raw
    )

    # -----------------------------------------------------
    # SECTOR SCORE
    #
    # This is calculated later by scanner.py.
    # -----------------------------------------------------

    if "sector_score" not in df.columns:

        df["sector_score"] = 50.0

    # -----------------------------------------------------
    # FINAL SCORE
    #
    # 30 value
    # 20 quality
    # 35 momentum
    # 15 base
    #
    # Sector acts as confirmation rather
    # than an additional 105th point.
    # -----------------------------------------------------

    df["overall_score"] = (
        df["value_score"]
        + df["quality_score"]
        + df["momentum_score"]
        + df["base_score"]
    ).clip(
        0,
        100,
    )

    # Sector confirmation

    sector_penalty = np.where(
        df["sector_score"] < 35,
        5,
        0,
    )

    sector_bonus = np.where(
        df["sector_score"] >= 75,
        2,
        0,
    )

    df["final_score"] = (
        df["overall_score"]
        - sector_penalty
        + sector_bonus
    ).clip(
        0,
        100,
    )

    # -----------------------------------------------------
    # SETUP CLASSIFICATION
    # -----------------------------------------------------

    strong_trend = (
        (price > sma200)
        & (sma50 > sma200)
        & (sma20 > sma50)
    )

    close_to_resistance = (
        dist20res >= -0.03
    ) & (
        dist20res <= 0.03
    )

    not_extended = (
        dist20res <= 0.05
    )

    sector_ok = (
        df["sector_score"] >= 45
    )

    # Confirmed breakout

    breakout = (

        (df["final_score"] >= 78)

        & (df["momentum_score"] >= 23)

        & strong_trend

        & (df["breakout_score"] >= 100)

        & sector_ok

    )

    # Pre-breakout

    pre_breakout = (

        (df["final_score"] >= 72)

        & (df["momentum_score"] >= 22)

        & (df["base_raw_score"] >= 60)

        & close_to_resistance

        & not_extended

        & strong_trend

        & sector_ok

    )

    # Watch

    watch = (

        (df["final_score"] >= 65)

        & (df["momentum_score"] >= 18)

        & (df["sector_score"] >= 35)

    )

    df["setup"] = np.select(
        [
            breakout,
            pre_breakout,
            watch,
        ],
        [
            "BREAKOUT",
            "PRE-BREAKOUT",
            "WATCH",
        ],
        default="REJECT",
    )

    # -----------------------------------------------------
    # TRADE LEVELS
    # -----------------------------------------------------

    atr_pct = _num(
        df,
        "atr_pct",
        0.02,
    )

    atr_pct = (
        atr_pct
        .clip(0.005, 0.08)
    )

    # Entry

    df["entry"] = np.where(

        df["setup"] == "BREAKOUT",

        price,

        price * (
            1
            + np.maximum(
                0,
                dist20res
                .clip(-0.03, 0.03)
            )
        ),
    )

    # Stop

    df["stop"] = (
        price
        * (
            1
            - (
                atr_pct
                * 1.5
            )
        )
    )

    # Risk

    risk = (
        df["entry"]
        - df["stop"]
    ).clip(
        lower=0.01
    )

    # Targets

    df["t1"] = (
        df["entry"]
        + risk * 1.5
    )

    df["t2"] = (
        df["entry"]
        + risk * 2.0
    )

    df["t3"] = (
        df["entry"]
        + risk * 3.0
    )

    df["rr_t1"] = (
        (
            df["t1"]
            - df["entry"]
        )
        / risk
    )

    df["rr_t2"] = (
        (
            df["t2"]
            - df["entry"]
        )
        / risk
    )

    df["rr_t3"] = (
        (
            df["t3"]
            - df["entry"]
        )
        / risk
    )

    return df
