import os
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Nifty Value-Momentum OS",
    page_icon="📈",
    layout="wide",
)


OUTPUT_DIR = Path("output")


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_csv(filename):

    path = OUTPUT_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception as error:

        st.error(
            f"Unable to read {filename}: {error}"
        )

        return pd.DataFrame()


def format_dataframe(df):

    if df.empty:
        return df

    display = df.copy()

    percentage_columns = [
        "ret1m",
        "ret3m",
        "ret6m",
        "ret12m",
        "relative_strength",
        "dist52",
        "dist20res",
        "atr_pct",
    ]

    for column in percentage_columns:

        if column in display.columns:

            display[column] = (
                display[column] * 100
            ).round(1)

    number_columns = [
        "price",
        "final_score",
        "overall_score",
        "early_score",
        "value_score",
        "quality_score",
        "momentum_score",
        "base_score",
        "sector_score",
    ]

    for column in number_columns:

        if column in display.columns:

            display[column] = display[column].round(1)

    return display


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title(
    "📈 NIFTY 50 + NEXT 50 "
    "VALUE-MOMENTUM OS"
)

st.caption(
    "Value + Quality + Momentum + Base/Breakout"
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header(
    "Scanner Controls"
)

if st.sidebar.button(
    "🔄 Refresh Results"
):

    st.cache_data.clear()

    st.rerun()


st.sidebar.markdown(
    """
### Score Structure

**Value:** 30

**Quality:** 20

**Momentum:** 35

**Base / Breakout:** 15

**Total:** 100

**Early Momentum:** 100
"""
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

master = load_csv(
    "master_top30.csv"
)

early = load_csv(
    "top10_early_momentum.csv"
)

value_momentum = load_csv(
    "top10_value_momentum.csv"
)

breakout = load_csv(
    "top10_breakout_ready.csv"
)


# ---------------------------------------------------------
# STATUS
# ---------------------------------------------------------

if master.empty:

    st.warning(
        "No scanner results are available yet."
    )

    st.info(
        "Run the GitHub Actions scanner first."
    )

    st.stop()


# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Top Stock",
        master.iloc[0]["symbol"],
    )


with col2:

    st.metric(
        "Top Score",
        f"{master.iloc[0]['final_score']:.0f}",
    )


with col3:

    st.metric(
        "Early Momentum",
        f"{master.iloc[0]['early_score']:.0f}",
    )


with col4:

    breakout_count = len(
        master[
            master["setup"].isin(
                [
                    "BREAKOUT",
                    "PRE-BREAKOUT",
                ]
            )
        ]
    )

    st.metric(
        "Breakout Candidates",
        breakout_count,
    )


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏆 Master Top 30",
        "🚀 Early Momentum",
        "💎 Value + Momentum",
        "🔥 Breakout Ready",
    ]
)


# ---------------------------------------------------------
# MASTER
# ---------------------------------------------------------

with tab1:

    st.subheader(
        "Master Top 30"
    )

    display = format_dataframe(
        master
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# EARLY MOMENTUM
# ---------------------------------------------------------

with tab2:

    st.subheader(
        "Top 10 Early Momentum"
    )

    st.caption(
        "Designed to identify stocks "
        "showing improving momentum before "
        "a full breakout."
    )

    display = format_dataframe(
        early
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# VALUE + MOMENTUM
# ---------------------------------------------------------

with tab3:

    st.subheader(
        "Top 10 Value + Momentum"
    )

    st.caption(
        "Stocks combining valuation, "
        "quality and price momentum."
    )

    display = format_dataframe(
        value_momentum
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# BREAKOUT
# ---------------------------------------------------------

with tab4:

    st.subheader(
        "Top 10 Breakout Ready"
    )

    st.caption(
        "Pre-breakout and confirmed breakout "
        "candidates."
    )

    display = format_dataframe(
        breakout
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# SETUP DISTRIBUTION
# ---------------------------------------------------------

st.divider()

st.subheader(
    "Setup Distribution"
)

setup_counts = (
    master["setup"]
    .value_counts()
)

st.bar_chart(
    setup_counts
)


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.divider()

st.subheader(
    "Download Results"
)

csv_data = master.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Master Top 30 CSV",
    data=csv_data,
    file_name="master_top30.csv",
    mime="text/csv",
)
