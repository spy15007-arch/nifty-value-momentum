"""Early-base radar: observation-only candidates before a confirmed breakout."""
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")


def _numeric(df, column, default=50.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_early_radar(diagnostics: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Rank constructive bases 0.5%-10% below 20-day resistance.

    Volume dry-up and up/down-volume are supporting indicators, not hard
    gates. The latter is a simple price-direction volume proxy, not proof
    of institutional accumulation.
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

    for col in [
        "price", "early_score", "trend_raw", "momentum_raw",
        "relative_strength", "sector_score", "quality_coverage",
        "dist20res", "range10_pct",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    liquidity = df["liquidity_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
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

    candidates["resistance"] = candidates["price"] / (1.0 - candidates["dist20res"])
    candidates["trigger"] = candidates["resistance"] * 1.002
    candidates["radar_proximity_score"] = (
        100 - candidates["dist20res"].clip(0, 0.10) * 1000
    ).clip(0, 100)
    candidates["radar_tightness_score"] = (
        100 - candidates["range10_pct"].clip(0, 0.15) / 0.15 * 100
    ).clip(0, 100)

    dryup = _numeric(candidates, "volume_dryup_ratio", 0.85)
    candidates["radar_volume_dryup_score"] = (
        100 - (dryup - 0.75).abs().clip(0, 1.0) * 100
    ).clip(0, 100)
    accumulation = _numeric(candidates, "up_down_volume_ratio", 1.0)
    candidates["radar_accumulation_score"] = (
        accumulation.clip(0, 2.0) / 2.0 * 100
    ).clip(0, 100)
    rs_score = (
        candidates["relative_strength"].clip(0, 0.20) / 0.20 * 100
    ).clip(0, 100)
    atr_contraction = _numeric(candidates, "atr_contraction_ratio", 1.0)
    candidates["radar_atr_contraction_score"] = (
        100 - ((atr_contraction - 0.70).clip(lower=0) / 0.80 * 100)
    ).clip(0, 100)

    # Reduce priority for late-stage price extension without rejecting it
    # automatically; missing columns remain neutral for older diagnostic files.
    rsi = _numeric(candidates, "rsi14", 60.0).clip(0, 100)
    extension20 = _numeric(candidates, "extension20_pct", 0.0)
    candidates["radar_extension_penalty"] = (
        (rsi - 70).clip(lower=0) * 0.75
        + (extension20 - 0.05).clip(lower=0, upper=0.10) * 150
    )

    candidates["radar_score"] = (
        candidates["early_score"] * 0.23
        + candidates["trend_raw"] * 0.14
        + candidates["momentum_raw"] * 0.14
        + candidates["radar_proximity_score"] * 0.14
        + candidates["radar_tightness_score"] * 0.09
        + candidates["radar_volume_dryup_score"] * 0.09
        + candidates["radar_accumulation_score"] * 0.05
        + rs_score * 0.05
        + candidates["radar_atr_contraction_score"] * 0.07
        - candidates["radar_extension_penalty"]
    ).clip(0, 100)

    return candidates.sort_values(
        ["radar_score", "early_score", "momentum_raw"], ascending=False
    ).head(limit).reset_index(drop=True)


def append_early_radar(
    message_path: Path = OUTPUT_DIR / "telegram_message.txt",
    diagnostics_path: Path = OUTPUT_DIR / "qualification_diagnostics.csv",
    output_path: Path = OUTPUT_DIR / "early_base_radar.csv",
) -> int:
    """Append the observation-only early-base section to Telegram text."""
    if not diagnostics_path.exists() or not message_path.exists():
        print("Early-base radar skipped: scanner output files are missing.")
        return 0

    candidates = build_early_radar(pd.read_csv(diagnostics_path))
    candidates.to_csv(output_path, index=False)
    lines = ["", "🌱 EARLY BASE RADAR — OBSERVE, NOT A BUY SIGNAL", ""]
    if candidates.empty:
        lines.extend(["No early-base candidates passed the quality and structure filters.", ""])
    else:
        for rank, row in candidates.iterrows():
            distance = float(row["dist20res"]) * 100
            dryup = row.get("volume_dryup_ratio", np.nan)
            accum = row.get("up_down_volume_ratio", np.nan)
            volume_note = f" | 5D/20D vol {float(dryup):.2f}x" if pd.notna(dryup) else ""
            accumulation_note = f" | Up/down vol {float(accum):.2f}x" if pd.notna(accum) else ""
            rsi_value = _numeric(pd.DataFrame([row]), "rsi14", 60.0).iloc[0]
            extension_value = _numeric(pd.DataFrame([row]), "extension20_pct", 0.0).iloc[0]
            lines.append(
                f"#{rank + 1} {row['symbol']} | EARLY BASE | "
                f"Price ₹{row['price']:.2f} | 20D resistance ₹{row['resistance']:.2f} | "
                f"Trigger ₹{row['trigger']:.2f} | {distance:.1f}% below resistance | "
                f"RSI {rsi_value:.0f} | {extension_value * 100:+.1f}% vs 20-DMA | "
                f"Radar {row['radar_score']:.0f}{volume_note}{accumulation_note}"
            )
            lines.append("")
        lines.extend([
            "Wait for a closing breakout with strong volume confirmation.",
            "No entry, stop-loss or target is issued at the early-base stage.",
            "",
        ])
    with message_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Early-base radar appended: {len(candidates)} candidate(s).")
    return len(candidates)
