from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Nifty Value-Momentum OS",
    page_icon="📈",
    layout="wide",
)


OUTPUT_DIR = Path("output")


def load_csv(filename):

    path = OUTPUT_DIR / filename

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception:

        return pd.DataFrame()


def pct(value):

    try:

        return f"{float(value) * 100:.1f}%"

    except Exception:

        return "-"


def score(value):

    try:

        return f"{float(value):.0f}"

    except Exception:

        return "-"


st.title(
    "📈 NIFTY 50 + NEXT 50"
)

st.subheader(
    "VALUE-MOMENTUM TRADING OS"
)

st.caption(
    "Ranked Value + Quality + Momentum + "
    "Base / Breakout scanner"
)


master = load_csv(
    "master_top30.csv"
)

qualified = load_csv(
    "qualified_setups.csv"
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


if master.empty:

    st.warning(
        "No scanner results yet."
    )

    st.info(
        "Run GitHub Actions first."
    )

    st.stop()


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Qualified",
        len(qualified),
    )


with c2:

    st.metric(
        "Top Score",
        score(
            master.iloc[0][
                "final_score"
            ]
        ),
    )


with c3:

    st.metric(
        "Top Stock",
        master.iloc[0][
            "symbol"
        ],
    )


with c4:

    st.metric(
        "Breakouts",
        len(breakout),
    )


# ---------------------------------------------------------
# QUALIFIED
# ---------------------------------------------------------

st.divider()

st.header(
    "🔥 Qualified Setups"
)

if qualified.empty:

    st.info(
        "NO QUALIFIED BREAKOUT / "
        "PRE-BREAKOUT SETUPS TODAY."
    )

else:

    st.dataframe(
        qualified[
            [
                "rank",
                "symbol",
                "sector",
                "final_score",
                "setup",
                "value_score",
                "quality_score",
                "momentum_score",
                "base_score",
                "sector_score",
                "price",
                "entry",
                "stop",
                "t1",
                "t2",
                "t3",
                "rr_t2",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🚀 Early Momentum",
        "💎 Value + Momentum",
        "🔥 Breakout",
        "🏆 Master 30",
    ]
)


with tab1:

    st.subheader(
        "Early Momentum / Developing"
    )

    st.dataframe(
        early,
        use_container_width=True,
        hide_index=True,
    )


with tab2:

    st.subheader(
        "Value + Momentum"
    )

    st.dataframe(
        value_momentum,
        use_container_width=True,
        hide_index=True,
    )


with tab3:

    st.subheader(
        "Confirmed Breakout"
    )

    if breakout.empty:

        st.info(
            "No confirmed breakout today."
        )

    else:

        st.dataframe(
            breakout,
            use_container_width=True,
            hide_index=True,
        )


with tab4:

    st.subheader(
        "Master Top 30"
    )

    st.dataframe(
        master,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.divider()

csv_data = master.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    "⬇️ Download Master Top 30",
    csv_data,
    "master_top30.csv",
    "text/csv",
)
