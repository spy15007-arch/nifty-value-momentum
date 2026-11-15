"""Early-base radar for candidates that are not yet actionable breakouts.

This is a watchlist layer, not a buy signal. It uses the full-universe
qualification diagnostics produced by scanner.core.scanner.
"""
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("output")


def build_early_radar(diagnostics: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Select constructive bases 1-10% below 20-day resistance.

    The relaxed gates intentionally identify candidates before the strict
    PRE-BREAKOUT/ BREAKOUT rules pass. Candidates remain observation-only.
    """
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame()

    df = diagnostics.copy()
    required = {
        "symbol", "price", "early_score", "trend_raw", "momentum_raw",
        "relative_strength", "sector_score", "quality_coverage",
        "dist20res", "range10_pct", "liquidity_ok", "setup",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()

    numeric = [
        "price", "early_score", "trend_raw", "momentum_raw",
        "relative_strength", "sector_score", "quality_coverage",
        "dist20res", "range10_pct",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    liquidity = df["liquidity_ok"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    # dist20res is (prior 20-day high - price) / prior 20-day high.
    # Keep names near resistance, but still below it: 0.5%-10%.
    mask = (
        df["price"].gt(0)
        & df["early_score"].ge(60)
        & df["trend_raw"].ge(50)
        & df["momentum_raw"].ge(45)
        & df["relative_strength"].gt(0)
        & df["sector_score"].ge(40)
        & df["quality_coverage"].ge(2)
        & df["dist20res"].between(0.005, 0.10, inclusive="both")
        & df["range10_pct"].le(0.12)
        & liquidity
        & ~df["setup"].isin(["BREAKOUT", "PRE-BREAKOUT"])
    )
    candidates = df.loc[mask].copy()
    if candidates.empty:
        return candidates

    candidates["resistance"] = (
        candidates["price"] / (1.0 - candidates["dist20res"])
    )
    candidates["trigger"] = candidates["resistance"] * 1.002
    candidates = candidates.sort_values(
        ["early_score", "momentum_raw", "trend_raw"],
        ascending=False,
    )
    return candidates.head(limit).reset_index(drop=True)


def append_early_radar(message_path: Path = OUTPUT_DIR / "telegram_message.txt",
                       diagnostics_path: Path = OUTPUT_DIR / "qualification_diagnostics.csv",
                       output_path: Path = OUTPUT_DIR / "early_base_radar.csv") -> int:
    """Append a concise observation-only early-base section to Telegram text."""
    if not diagnostics_path.exists() or not message_path.exists():
        print("Early-base radar skipped: scanner output files are missing.")
        return 0

    diagnostics = pd.read_csv(diagnostics_path)
    candidates = build_early_radar(diagnostics)
    candidates.to_csv(output_path, index=False)

    lines = ["", "🌱 EARLY BASE RADAR — OBSERVE, NOT A BUY SIGNAL", ""]
    if candidates.empty:
        lines.extend([
            "No early-base candidates passed the relaxed quality and trend filters.",
            "",
        ])
    else:
        for rank, row in candidates.iterrows():
            distance = float(row["dist20res"]) * 100
            lines.append(
                f"#{rank + 1} {row['symbol']} | EARLY BASE | "
                f"Price ₹{row['price']:.2f} | "
                f"20D resistance ₹{row['resistance']:.2f} | "
                f"Trigger ₹{row['trigger']:.2f} | "
                f"{distance:.1f}% below resistance | "
                f"Early score {row['early_score']:.0f}"
            )
            lines.append("")
        lines.extend([
            "Wait for price/volume confirmation above the trigger.",
            "No entry, stop-loss or target is issued at the early-base stage.",
            "",
        ])

    with message_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Early-base radar appended: {len(candidates)} candidate(s).")
    return len(candidates)
