"""Persist signal snapshots and measure subsequent daily-close follow-through from the scan-time price.

These are research observations, not simulated fills or audited BTST/swing P&L.
"""
from pathlib import Path

import pandas as pd

from .market_data import series_for
from .early_radar import build_early_radar

OUTPUT_DIR = Path("output")
HORIZONS = (1, 3, 5, 10)
HISTORY_COLUMNS = [
    "scan_date", "ticker", "symbol", "signal_type", "signal_price",
    "score", "market_sentiment", "return_1d_pct", "return_3d_pct",
    "return_5d_pct", "return_10d_pct",
]
TRANSITION_COLUMNS = ["from_stage", "to_stage", "transition_count", "unique_symbols", "note"]


def build_stage_transition_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed stage changes; tracked candidates are a sample, not the full universe."""
    if history is None or history.empty:
        return pd.DataFrame(columns=TRANSITION_COLUMNS)
    required = {"scan_date", "symbol", "signal_type"}
    if not required.issubset(history.columns):
        return pd.DataFrame(columns=TRANSITION_COLUMNS)

    transitions = []
    clean = history.dropna(subset=["scan_date", "symbol", "signal_type"])
    for symbol, group in clean.groupby("symbol"):
        ordered = group.sort_values(["scan_date", "signal_type"])
        stages = ordered["signal_type"].astype(str).tolist()
        compact = []
        for stage in stages:
            if not compact or stage != compact[-1]:
                compact.append(stage)
        transitions.extend(
            {"symbol": str(symbol), "from_stage": left, "to_stage": right}
            for left, right in zip(compact, compact[1:])
            if left != right
        )

    if not transitions:
        return pd.DataFrame(columns=TRANSITION_COLUMNS)
    frame = pd.DataFrame(transitions)
    summary = frame.groupby(["from_stage", "to_stage"]).agg(
        transition_count=("symbol", "size"),
        unique_symbols=("symbol", "nunique"),
    ).reset_index()
    summary["note"] = (
        "Observed transitions among tracked top candidates only; not a full-universe "
        "conversion rate or trade-performance measure."
    )
    return summary[TRANSITION_COLUMNS].sort_values(
        ["from_stage", "to_stage"]
    ).reset_index(drop=True)


def update_performance_tracker(
    diagnostics: pd.DataFrame,
    price_data: pd.DataFrame,
    market_sentiment: str,
    history_path: Path = OUTPUT_DIR / "signal_history.csv",
    summary_path: Path = OUTPUT_DIR / "performance_summary.csv",
    transition_summary_path: Path = OUTPUT_DIR / "stage_transition_summary.csv",
) -> None:
    """Update forward close returns for actionable/watch and early-base signals."""
    if diagnostics is None or diagnostics.empty or price_data is None or price_data.empty:
        print("Performance tracker skipped: diagnostics or price history unavailable.")
        return

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    for col in HISTORY_COLUMNS:
        if col not in history.columns:
            history[col] = pd.NA

    radar = build_early_radar(diagnostics, limit=5)
    candidates = diagnostics[
        diagnostics["setup"].isin(["BREAKOUT", "PRE-BREAKOUT", "WATCH"])
    ].copy()
    candidates = candidates.sort_values(
        ["ranking_score", "final_score"], ascending=False
    ).drop_duplicates("symbol").head(10)
    existing_symbols = set(candidates["symbol"].astype(str))
    if not radar.empty:
        radar_only = radar[~radar["symbol"].astype(str).isin(existing_symbols)].copy()
        radar_only["setup"] = "EARLY_BASE"
        candidates = pd.concat([candidates, radar_only], ignore_index=True)

    new_rows = []
    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        symbol = str(row.get("symbol", "")).strip()
        if not ticker or not symbol:
            continue
        close = series_for(price_data, ticker, "Close")
        if close.empty:
            continue
        scan_date = pd.Timestamp(close.index[-1]).date().isoformat()
        signal_type = str(row.get("setup", "WATCH"))
        duplicate = (
            (history["scan_date"].astype(str) == scan_date)
            & (history["symbol"].astype(str) == symbol)
            & (history["signal_type"].astype(str) == signal_type)
        )
        if duplicate.any():
            continue
        score = row.get("radar_score", pd.NA)
        if pd.isna(score):
            score = row.get("final_score", row.get("early_score", 0))
        new_rows.append({
            "scan_date": scan_date, "ticker": ticker, "symbol": symbol,
            "signal_type": signal_type, "signal_price": float(close.iloc[-1]),
            "score": float(score) if pd.notna(score) else 0.0,
            "market_sentiment": market_sentiment,
            "return_1d_pct": pd.NA, "return_3d_pct": pd.NA,
            "return_5d_pct": pd.NA, "return_10d_pct": pd.NA,
        })

    if new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)

    for idx, record in history.iterrows():
        close = series_for(price_data, str(record["ticker"]), "Close")
        if close.empty:
            continue
        dates = pd.to_datetime(close.index).date
        signal_date = pd.Timestamp(record["scan_date"]).date()
        future = close.iloc[[i for i, day in enumerate(dates) if day > signal_date]]
        if future.empty:
            continue
        base = pd.to_numeric(pd.Series([record["signal_price"]]), errors="coerce").iloc[0]
        if pd.isna(base) or base <= 0:
            continue
        for horizon in HORIZONS:
            col = f"return_{horizon}d_pct"
            if len(future) >= horizon:
                history.at[idx, col] = round((float(future.iloc[horizon - 1]) / base - 1) * 100, 4)

    history = history.drop_duplicates(
        subset=["scan_date", "symbol", "signal_type"], keep="last"
    )
    history.to_csv(history_path, index=False)

    build_stage_transition_summary(history).to_csv(transition_summary_path, index=False)

    summary_rows = []
    for signal_type, group in history.groupby("signal_type"):
        for horizon in HORIZONS:
            col = f"return_{horizon}d_pct"
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            wins = values[values > 0]
            losses = values[values < 0]
            gross_profit, gross_loss = wins.sum(), abs(losses.sum())
            summary_rows.append({
                "signal_type": signal_type,
                "horizon_sessions": horizon,
                "observations": int(len(values)),
                "win_rate_pct": round(float((values > 0).mean() * 100), 2),
                "average_return_pct": round(float(values.mean()), 3),
                "median_return_pct": round(float(values.median()), 3),
                "profit_factor_close_returns": round(float(gross_profit / gross_loss), 3) if gross_loss > 0 else pd.NA,
                "note": "Scan-time price to subsequent daily closes; not simulated BTST/swing trade P&L.",
            })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Performance tracker updated: {len(history)} observations; {len(summary_rows)} summary rows.")
