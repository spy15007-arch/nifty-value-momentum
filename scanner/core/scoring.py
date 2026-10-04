import numpy as np
import pandas as pd


def percentile_score(
    value,
    series,
    maximum,
):

    if value is None or pd.isna(value):
        return 0

    clean = series.dropna()

    if len(clean) < 5:
        return 0

    percentile = (clean <= value).mean()

    if maximum == 8:
        return (
            8 if percentile >= 0.90 else
            6 if percentile >= 0.75 else
            4 if percentile >= 0.50 else
            2 if percentile >= 0.25 else
            0
        )

    if maximum == 7:
        return (
            7 if percentile >= 0.90 else
            5 if percentile >= 0.75 else
            3 if percentile >= 0.50 else
            1 if percentile >= 0.25 else
            0
        )

    if maximum == 6:
        return (
            6 if percentile >= 0.90 else
            5 if percentile >= 0.75 else
            3 if percentile >= 0.50 else
            1 if percentile >= 0.25 else
            0
        )

    if maximum == 5:
        return (
            5 if percentile >= 0.90 else
            4 if percentile >= 0.75 else
            3 if percentile >= 0.50 else
            1 if percentile >= 0.25 else
            0
        )

    return 0


def inverse_percentile_score(
    value,
    series,
    maximum,
):

    if value is None or pd.isna(value):
        return 0

    clean = series.dropna()

    if len(clean) < 5:
        return 0

    percentile = (clean >= value).mean()

    if maximum == 5:

        return (
            5 if percentile >= 0.90 else
            4 if percentile >= 0.75 else
            3 if percentile >= 0.50 else
            1 if percentile >= 0.25 else
            0
        )

    return 0


def score_fundamentals(df):

    df = df.copy()

    for column in [
        "earnings_yield",
        "ps",
        "pb",
        "ev_ebitda",
        "fcf_yield",
    ]:

        if column not in df:
            df[column] = np.nan

    # VALUE

    df["earnings_yield_score"] = df[
        "earnings_yield"
    ].apply(
        lambda value:
        percentile_score(
            value,
            df["earnings_yield"],
            8,
        )
    )

    for column in [
        "ps",
        "pb",
        "ev_ebitda",
    ]:

        df[f"{column}_score"] = df[column].apply(
            lambda value:
            inverse_percentile_score(
                value,
                df[column],
                5,
            )
        )

    df["fcf_yield_score"] = df[
        "fcf_yield"
    ].apply(
        lambda value:
        percentile_score(
            value,
            df["fcf_yield"],
            7,
        )
    )

    df["value_score"] = (
        df["earnings_yield_score"]
        + df["ps_score"]
        + df["pb_score"]
        + df["ev_ebitda_score"]
        + df["fcf_yield_score"]
    ).clip(0, 30)

    # QUALITY

    def roce_score(value):

        if pd.isna(value):
            return 0

        if value > 0.25:
            return 6

        if value > 0.20:
            return 5

        if value > 0.15:
            return 4

        if value >= 0.12:
            return 2

        return -99

    def roe_score(value):

        if pd.isna(value):
            return 0

        if value > 0.25:
            return 4

        if value > 0.20:
            return 3

        if value > 0.15:
            return 2

        if value >= 0.12:
            return 1

        return 0

    def eps_growth_score(value):

        if pd.isna(value):
            return 0

        if value > 0.25:
            return 5

        if value > 0.15:
            return 4

        if value > 0.10:
            return 3

        if value >= 0.05:
            return 1

        return 0

    def debt_score(value):

        if pd.isna(value):
            return 0

        if value < 0.20:
            return 5

        if value < 0.50:
            return 4

        if value < 0.75:
            return 3

        if value <= 1.00:
            return 1

        return -99

    df["roce_score"] = df["roce"].apply(
        roce_score
    )

    df["roe_score"] = df["roe"].apply(
        roe_score
    )

    df["eps_growth_score"] = df[
        "eps_growth"
    ].apply(
        eps_growth_score
    )

    df["debt_score"] = df[
        "debt_equity"
    ].apply(
        debt_score
    )

    df["quality_score"] = (
        df["roce_score"].clip(lower=0)
        + df["roe_score"]
        + df["eps_growth_score"]
        + df["debt_score"].clip(lower=0)
    ).clip(0, 20)

    df["hard_reject"] = (
        (df["roce_score"] < 0)
        | (df["debt_score"] < 0)
    )

    return df


def technical_scores(
    df,
    nifty_return_6m,
):

    df = df.copy()

    df["relative_strength"] = (
        df["ret6m"] - nifty_return_6m
    )

    # MOMENTUM

    df["six_score"] = (
        df["ret6m"]
        .rank(pct=True)
        .apply(
            lambda x:
            7 if x >= .90 else
            6 if x >= .75 else
            4 if x >= .50 else
            2 if x >= .25 else
            0
        )
    )

    df["twelve_score"] = (
        df["ret12m"]
        .rank(pct=True)
        .apply(
            lambda x:
            6 if x >= .90 else
            5 if x >= .75 else
            3 if x >= .50 else
            1 if x >= .25 else
            0
        )
    )

    def relative_score(value):

        if value > .20:
            return 7

        if value > .10:
            return 6

        if value > .05:
            return 4

        if value >= 0:
            return 2

        return 0

    df["relative_score"] = (
        df["relative_strength"]
        .apply(relative_score)
    )

    def regime_score(row):

        if row["price"] <= row["sma200"]:
            return 0

        if (
            row["sma200"]
            > row["sma200_20ago"] * 1.01
        ):
            return 5

        return 4

    df["regime_score"] = df.apply(
        regime_score,
        axis=1,
    )

    def moving_average_score(row):

        if (
            row["sma20"]
            > row["sma50"]
            > row["sma200"]
            and row["sma20"]
            > row["sma20_20ago"]
            and row["sma50"]
            > row["sma50_20ago"]
        ):
            return 5

        if (
            row["sma20"]
            > row["sma50"]
            > row["sma200"]
        ):
            return 4

        if row["sma20"] > row["sma50"]:
            return 2

        return 0

    df["moving_average_score"] = df.apply(
        moving_average_score,
        axis=1,
    )

    df["momentum_efficiency"] = (
        df["ret6m"]
        / df["vol6m"].replace(0, np.nan)
    )

    df["efficiency_score"] = (
        df["momentum_efficiency"]
        .rank(pct=True)
        .apply(
            lambda x:
            5 if x >= .90 else
            4 if x >= .75 else
            3 if x >= .50 else
            1 if x >= .25 else
            0
        )
    )

    df["momentum_score"] = (
        df["six_score"]
        + df["twelve_score"]
        + df["relative_score"]
        + df["regime_score"]
        + df["moving_average_score"]
        + df["efficiency_score"]
    ).clip(0, 35)

    # BASE / BREAKOUT

    df["distance_52w_score"] = (
        df["dist52"].apply(
            lambda x:
            4 if x <= .03 else
            3 if x <= .05 else
            2 if x <= .08 else
            1 if x <= .12 else
            0
        )
    )

    df["volatility_contraction_score"] = (
        (
            df["atr_pct"]
            < df["atr_pct_20ago"]
        ).astype(int) * 3
    )

    df["volume_dryup_score"] = (
        (
            df["vol20"]
            < df["vol20_prev"]
        ).astype(int) * 3
    )

    df["resistance_score"] = (
        df["dist20res"].apply(
            lambda x:
            3 if x < .02 else
            2 if x < .04 else
            1 if x < .07 else
            0
        )
    )

    df["breakout_score"] = (
        (
            (df["price"] > df["high20"])
            &
            (df["volume"] > 1.5 * df["vol20"])
        ).astype(int) * 2
    )

    df["base_score"] = (
        df["distance_52w_score"]
        + df["volatility_contraction_score"]
        + df["volume_dryup_score"]
        + df["resistance_score"]
        + df["breakout_score"]
    ).clip(0, 15)

    # FINAL SCORE

    df["overall_score"] = (
        df["value_score"]
        + df["quality_score"]
        + df["momentum_score"]
        + df["base_score"]
    ).clip(0, 100)

    # EARLY MOMENTUM

    relative_strength_acceleration = (
        df["ret1m"]
        - (df["ret3m"] / 3)
    ).rank(pct=True)

    sector_proxy = (
        df["ret3m"]
        .rank(pct=True)
    )

    base_quality = (
        df["base_score"] / 15
    )

    distance_to_breakout = (
        1
        - df["dist20res"]
        .clip(lower=0, upper=.15)
        / .15
    )

    volume_accumulation = (
        df["vol20"]
        / df["vol20_prev"]
    ).clip(0, 2) / 2

    dma_acceleration = (
        (
            df["sma20"]
            / df["sma20_20ago"]
        ) - 1
    ).rank(pct=True)

    df["early_score"] = (
        sector_proxy * 20
        + relative_strength_acceleration * 20
        + base_quality * 20
        + distance_to_breakout * 15
        + volume_accumulation * 10
        + dma_acceleration * 10
        + (df["six_score"] / 7) * 5
    ).clip(0, 100)

    df.loc[
        df["hard_reject"],
        "overall_score"
    ] = 0

    return df
