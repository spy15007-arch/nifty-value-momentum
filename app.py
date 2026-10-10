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
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


def number(value, decimals=1):
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def show_table(df, preferred_columns=None):
    if df.empty:
        st.info("No records available in this section.")
        return
    if preferred_columns:
        columns = [c for c in preferred_columns if c in df.columns]
        if columns:
            st.dataframe(df[columns], use_container_width=True, hide_index=True)
            return
    st.dataframe(df, use_container_width=True, hide_index=True)


st.title("📈 NIFTY VALUE-MOMENTUM OS")
st.caption("Nifty 50 • Next 50 • Midcap 150 • Smallcap 250")
st.write(
    "Ranked stock research with trend, momentum, relative strength, "
    "setup status and trade planning."
)

master = load_csv("master_top30.csv")
qualified = load_csv("qualified_setups.csv")
early = load_csv("top10_early_momentum.csv")
value_momentum = load_csv("top10_value_momentum.csv")
breakout = load_csv("top10_breakout_ready.csv")
diagnostics = load_csv("qualification_diagnostics.csv")

if master.empty:
    st.warning("No scanner results are available yet.")
    st.info(
        "Run the repository's configured GitHub Actions workflow and "
        "check that it creates output/master_top30.csv."
    )
    st.stop()

st.divider()
st.subheader("Market Scanner Summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Qualified Setups", len(qualified))
with c2:
    st.metric("Top Score", number(master.iloc[0].get("final_score")))
with c3:
    st.metric("Top Stock", str(master.iloc[0].get("symbol", "-")))
with c4:
    st.metric("Breakouts", len(breakout))

st.divider()
st.header("🎯 Qualified Setups")
if qualified.empty:
    st.info(
        "No BREAKOUT or PRE-BREAKOUT setups in the latest scan. "
        "Do not force a trade when the rules are not satisfied."
    )
else:
    show_table(
        qualified,
        [
            "rank", "symbol", "index", "sector", "final_score", "setup",
            "price", "entry_low", "entry_ideal", "entry_high", "stop",
            "t1", "t2", "t3", "t4", "rr_t1", "rr_t2", "rr_t3", "rr_t4",
            "chase",
        ],
    )

st.divider()
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🚀 Early Momentum", "💎 Value + Momentum", "🔥 Breakouts",
        "🏆 Master Top 30", "🔎 Diagnostics",
    ]
)
with tab1:
    st.subheader("Early Momentum / Developing Setups")
    show_table(early)
with tab2:
    st.subheader("Value + Momentum")
    show_table(value_momentum)
with tab3:
    st.subheader("Confirmed Breakouts")
    show_table(breakout)
with tab4:
    st.subheader("Master Ranked Stocks")
    show_table(master)
with tab5:
    st.subheader("Why did stocks fail qualification?")
    if diagnostics.empty:
        st.info(
            "Diagnostics will appear here after the scanner generates "
            "output/qualification_diagnostics.csv."
        )
    else:
        if "diagnostic_reason" in diagnostics.columns:
            only_failed = st.checkbox(
                "Show only stocks that did not qualify", value=False
            )
            if only_failed:
                diagnostics = diagnostics[
                    diagnostics["diagnostic_reason"].astype(str).str.contains(
                        "not above|below|not positive|failed|more than|"
                        "fewer than|wait for|do not chase",
                        case=False,
                        regex=True,
                    )
                ]
        show_table(diagnostics)

st.divider()
st.subheader("⬇️ Download Reports")
downloads = [
    ("Master Top 30", master, "master_top30.csv"),
    ("Qualified Setups", qualified, "qualified_setups.csv"),
    ("Early Momentum", early, "top10_early_momentum.csv"),
    ("Value + Momentum", value_momentum, "top10_value_momentum.csv"),
    ("Breakout Ready", breakout, "top10_breakout_ready.csv"),
    ("Qualification Diagnostics", diagnostics, "qualification_diagnostics.csv"),
]
columns = st.columns(3)
for i, (label, frame, filename) in enumerate(downloads):
    with columns[i % 3]:
        st.download_button(
            label=f"Download {label}",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            disabled=frame.empty,
            key=f"download_{filename}",
        )

st.caption(
    "Research tool only. A listed setup is not a guaranteed trade. "
    "Check liquidity, live price, slippage and risk before entering."
)
