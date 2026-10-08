import numpy as np
import pandas as pd


def _num(df, column, default=np.nan):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def _rank_score(series, higher=True):
    s = pd.to_numeric(
        series,
        errors="coerce"
    )

    if s.notna().sum() <= 1:
        return pd.Series(
            50.0,
            index=s.index
        )

    return (
        s.rank(
            pct=True,
            method="average",
            ascending=higher
        ) * 100
    ).fillna(50.0)


def _clip(series, low=0, high=100):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(50.0).clip(
        low,
        high
    )


# =========================================================
# FUNDAMENTALS
# =========================================================

def score_fundamentals(df):

    df = df.copy()

    pe = _num(df, "pe")
    pb = _num(df, "pb")
    ev = _num(df, "ev_ebitda")
    earnings_yield = _num(
        df,
        "earnings_yield"
    )
    fcf_yield = _num(
        df,
        "fcf_yield"
    )

    value_parts = []

    if pe.notna().sum():
        value_parts.append(
            _rank_score(
                pe.where(
                    (pe > 0) &
                    (pe < 100)
                ),
                higher=False
            )
        )

    if pb.notna().sum():
        value_parts.append(
            _rank_score(
                pb.where(
                    (pb > 0) &
                    (pb < 30)
                ),
                higher=False
            )
        )

    if ev.notna().sum():
        value_parts.append(
            _rank_score(
                ev.where(
                    (ev > 0) &
                    (ev < 100)
                ),
                higher=False
            )
        )

    if earnings_yield.notna().sum():
        value_parts.append(
            _rank_score(
                earnings_yield,
                higher=True
            )
        )

    if fcf_yield.notna().sum():
        value_parts.append(
            _rank_score(
                fcf_yield,
                higher=True
            )
        )

    if value_parts:

        value_raw = pd.concat(
            value_parts,
            axis=1
        ).mean(axis=1)

    else:

        value_raw = pd.Series(
            50.0,
            index=df.index
        )


    # -----------------------------------------------------
    # QUALITY
    # -----------------------------------------------------

    roe = _num(df, "roe")
    roce = _num(df, "roce")
    debt_equity = _num(
        df,
        "debt_equity"
    )
    eps_growth = _num(
        df,
        "eps_growth"
    )
    revenue_growth = _num(
        df,
        "revenue_growth"
    )

    quality_parts = []

    for series in [
        roe,
        roce,
        eps_growth,
        revenue_growth
    ]:

        if series.notna().sum():

            quality_parts.append(
                _rank_score(
                    series,
                    higher=True
                )
            )

    if debt_equity.notna().sum():

        quality_parts.append(
            _rank_score(
                debt_equity.where(
                    debt_equity >= 0
                ),
                higher=False
            )
        )

    if quality_parts:

        quality_raw = pd.concat(
            quality_parts,
            axis=1
        ).mean(axis=1)

    else:

        quality_raw = pd.Series(
            50.0,
            index=df.index
        )


    # -----------------------------------------------------
    # FUNDAMENTAL DATA COVERAGE
    # -----------------------------------------------------

    coverage = pd.concat(
        [
            pe.notna(),
            pb.notna(),
            ev.notna(),
            earnings_yield.notna(),
            fcf_yield.notna(),
            roe.notna(),
            roce.notna(),
            debt_equity.notna(),
            eps_growth.notna(),
            revenue_growth.notna(),
        ],
        axis=1
    ).sum(axis=1)

    df["fundamental_coverage"] = coverage

    # New weighting:
    #
    # VALUE   = 10
    # QUALITY = 20

    df["value_score"] = (
        _clip(value_raw) * 0.10
    )

    df["quality_score"] = (
        _clip(quality_raw) * 0.20
    )

    df["fundamental_score"] = (
        df["value_score"] +
        df["quality_score"]
    )

    return df


# =========================================================
# TECHNICAL / INSTITUTIONAL SCORING
# =========================================================

def technical_scores(
    df,
    nifty_return_6m=0.0
):

    df = df.copy()

    price = _num(
        df,
        "price",
        0
    )

    sma20 = _num(
        df,
        "sma20",
        0
    )

    sma50 = _num(
        df,
        "sma50",
        0
    )

    sma200 = _num(
        df,
        "sma200",
        0
    )


    ret1m = _num(
        df,
        "ret1m",
        0
    )

    ret3m = _num(
        df,
        "ret3m",
        0
    )

    ret6m = _num(
        df,
        "ret6m",
        0
    )

    ret12m = _num(
        df,
        "ret12m",
        0
    )


    # =====================================================
    # RELATIVE STRENGTH
    # =====================================================

    df["relative_strength"] = (
        ret6m -
        float(nifty_return_6m)
    )


    # =====================================================
    # TREND — 20 POINTS
    # =====================================================

    trend_alignment = (

        (price > sma200).astype(int)

        + (sma50 > sma200).astype(int)

        + (sma20 > sma50).astype(int)

        + (price > sma20).astype(int)

    ) / 4.0 * 100


    slope20 = (
        price /
        sma20.replace(
            0,
            np.nan
        ) - 1
    ).clip(
        -0.10,
        0.10
    )


    slope50 = (
        sma20 /
        sma50.replace(
            0,
            np.nan
        ) - 1
    ).clip(
        -0.10,
        0.10
    )


    trend_slope = (

        ((slope20 + 0.10) / 0.20 * 50)

        +

        ((slope50 + 0.10) / 0.20 * 50)

    )


    trend_raw = (

        trend_alignment * 0.70

        +

        trend_slope.clip(
            0,
            100
        ) * 0.30

    )


    df["trend_raw"] = _clip(
        trend_raw
    )

    df["trend_score"] = (
        df["trend_raw"] * 0.20
    )


    # =====================================================
    # MOMENTUM + RELATIVE STRENGTH — 25 POINTS
    # =====================================================

    m1 = _rank_score(
        ret1m
    )

    m3 = _rank_score(
        ret3m
    )

    m6 = _rank_score(
        ret6m
    )

    m12 = _rank_score(
        ret12m
    )

    rs = _rank_score(
        df["relative_strength"]
    )


    momentum_raw = (

        m1 * 0.10

        + m3 * 0.20

        + m6 * 0.30

        + m12 * 0.15

        + rs * 0.25

    )


    df["momentum_raw"] = _clip(
        momentum_raw
    )

    df["momentum_score"] = (
        df["momentum_raw"] * 0.25
    )


    # =====================================================
    # STRUCTURE / BASE — 15 POINTS
    # =====================================================

    dist20 = _num(
        df,
        "dist20res",
        0.10
    )

    dist52 = _num(
        df,
        "dist52",
        0.20
    )

    atr = _num(
        df,
        "atr_pct",
        0.02
    ).clip(
        0.005,
        0.10
    )

    atr_old = _num(
        df,
        "atr_pct_20ago",
        np.nan
    )


    resistance_score = (

        100 -

        (
            dist20.abs().clip(
                0,
                0.12
            ) / 0.12 * 100
        )

    ).clip(
        0,
        100
    )


    high52_score = (

        100 -

        (
            dist52.clip(
                0,
                0.20
            ) / 0.20 * 100
        )

    ).clip(
        0,
        100
    )


    volatility_score = (

        100 -

        (
            (
                atr - 0.012
            ).clip(
                0,
                0.05
            ) / 0.05 * 100
        )

    ).clip(
        0,
        100
    )


    contraction = np.where(

        atr_old.notna(),

        (
            atr <=
            atr_old * 1.10
        ).astype(float) * 100,

        60.0

    )


    base_raw = (

        resistance_score * 0.40

        + high52_score * 0.20

        + volatility_score * 0.20

        + pd.Series(
            contraction,
            index=df.index
        ) * 0.20

    )


    df["base_raw_score"] = _clip(
        base_raw
    )

    df["base_score"] = (
        df["base_raw_score"] * 0.15
    )


    # =====================================================
    # VOLUME / ACCUMULATION — 10 POINTS
    # =====================================================

    volume = _num(
        df,
        "volume",
        0
    )

    vol20 = _num(
        df,
        "vol20",
        0
    ).replace(
        0,
        np.nan
    )


    volume_ratio = (

        volume /
        vol20

    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(
        1.0
    )


    df["volume_ratio"] = (
        volume_ratio
    )


    vol20_prev = _num(
        df,
        "vol20_prev",
        np.nan
    ).replace(
        0,
        np.nan
    )


    vol20_trend = (

        vol20 /
        vol20_prev

    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(
        1.0
    )


    df["vol20_trend"] = (
        vol20_trend
    )


    volume_strength = (

        (
            volume_ratio - 0.60
        ) / 1.40 * 100

    ).clip(
        0,
        100
    )


    volume_trend = (

        (
            vol20_trend - 0.75
        ) / 0.75 * 100

    ).clip(
        0,
        100
    )


    volume_raw = (

        volume_strength * 0.70

        +

        volume_trend * 0.30

    )


    df["volume_raw"] = _clip(
        volume_raw
    )

    df["volume_score"] = (
        df["volume_raw"] * 0.10
    )


    # =====================================================
    # FINAL 100-POINT SCORE
    # =====================================================

    df["final_score_base"] = (

        df["value_score"]

        + df["quality_score"]

        + df["trend_score"]

        + df["momentum_score"]

        + df["base_score"]

        + df["volume_score"]

    ).clip(
        0,
        100
    )


    # =====================================================
    # SECTOR GATE
    # =====================================================

    sector_score = _num(
        df,
        "sector_score",
        50
    )

    sector_ok = (
        sector_score >= 45
    )


    # =====================================================
    # HARD QUALITY GATES
    # =====================================================

    strong_trend = (

        (price > sma200)

        & (sma50 > sma200)

        & (sma20 > sma50)

        & (price > sma20)

    )


    rs_positive = (
        df["relative_strength"] > 0
    )


    momentum_ok = (
        df["momentum_raw"] >= 55
    )


    trend_ok = (
        df["trend_raw"] >= 70
    )


    fundamental_ok = (

        (df["quality_score"] >= 9)

        &

        (df["fundamental_coverage"] >= 2)

    )


    # =====================================================
    # BREAKOUT
    # =====================================================

    breakout = (

        strong_trend

        & rs_positive

        & momentum_ok

        & trend_ok

        & sector_ok

        & fundamental_ok

        & (dist20 <= 0.005)

        & (dist20 >= -0.025)

        & (volume_ratio >= 1.30)

        & (df["final_score_base"] >= 72)

    )


    # =====================================================
    # PRE-BREAKOUT
    # =====================================================

    pre_breakout = (

        strong_trend

        & rs_positive

        & momentum_ok

        & trend_ok

        & sector_ok

        & fundamental_ok

        & (dist20 >= 0.0)

        & (dist20 <= 0.03)

        & (volume_ratio <= 1.35)

        & (atr <= 0.045)

        & (df["final_score_base"] >= 68)

    )


    # =====================================================
    # WATCH
    # =====================================================

    watch = (

        (df["final_score_base"] >= 62)

        & (df["momentum_raw"] >= 50)

        & (df["trend_raw"] >= 60)

        & sector_ok

    )


    df["breakout_signal"] = (
        breakout
    )


    df["setup"] = np.select(

        [
            breakout,
            pre_breakout,
            watch
        ],

        [
            "BREAKOUT",
            "PRE-BREAKOUT",
            "WATCH"
        ],

        default="REJECT"

    )


    # =====================================================
    # SECTOR ADJUSTMENT
    # =====================================================

    sector_adjustment = np.where(

        sector_score >= 75,

        2,

        np.where(
            sector_score < 30,
            -4,
            0
        )

    )


    df["final_score"] = (

        df["final_score_base"]

        + sector_adjustment

    ).clip(
        0,
        100
    )


    # =====================================================
    # ENTRY RANGE
    # =====================================================

    resistance = _num(
        df,
        "high20",
        price
    )


    # PRE-BREAKOUT
    pre_low = (
        resistance * 0.995
    )

    pre_high = (
        resistance * 1.010
    )

    pre_ideal = (
        resistance * 1.002
    )


    # BREAKOUT
    breakout_low = (
        resistance * 0.995
    )

    breakout_high = (
        resistance * 1.015
    )

    breakout_ideal = (
        resistance * 1.005
    )


    df["entry_low"] = np.where(

        df["setup"] == "BREAKOUT",

        breakout_low,

        pre_low

    )


    df["entry_high"] = np.where(

        df["setup"] == "BREAKOUT",

        breakout_high,

        pre_high

    )


    df["entry_ideal"] = np.where(

        df["setup"] == "BREAKOUT",

        breakout_ideal,

        pre_ideal

    )


    # =====================================================
    # CHASE PROTECTION
    # =====================================================

    chase = (
        price > df["entry_high"]
    )

    df["chase"] = chase


    df.loc[
        chase &
        df["setup"].isin(
            [
                "BREAKOUT",
                "PRE-BREAKOUT"
            ]
        ),
        "setup"
    ] = "WATCH"


    # =====================================================
    # STOP LOSS
    # =====================================================

    atr_value = (
        atr * price
    )


    structural_stop = (

        resistance -

        1.25 * atr_value

    )


    atr_stop = (

        price -

        1.50 * atr_value

    )


    stop = pd.concat(
        [
            structural_stop,
            atr_stop
        ],
        axis=1
    ).min(
        axis=1
    )


    stop = stop.clip(

        lower=price * 0.90,

        upper=price * 0.995

    )


    df["stop"] = stop


    # =====================================================
    # FOUR TARGETS
    # =====================================================

    risk = (

        df["entry_ideal"]

        - df["stop"]

    ).clip(
        lower=0.01
    )


    df["t1"] = (
        df["entry_ideal"]
        + risk * 1.5
    )

    df["t2"] = (
        df["entry_ideal"]
        + risk * 2.5
    )

    df["t3"] = (
        df["entry_ideal"]
        + risk * 3.5
    )

    df["t4"] = (
        df["entry_ideal"]
        + risk * 5.0
    )


    for n in [
        1,
        2,
        3,
        4
    ]:

        df[f"rr_t{n}"] = (

            (
                df[f"t{n}"]
                - df["entry_ideal"]
            )

            / risk

        )


    # =====================================================
    # EARLY SCORE
    # =====================================================

    df["early_score"] = (

        df["momentum_raw"] * 0.45

        + df["base_raw_score"] * 0.35

        + df["trend_raw"] * 0.20

    ).clip(
        0,
        100
    )


    df["overall_score"] = (
        df["final_score"]
    )


    return df
