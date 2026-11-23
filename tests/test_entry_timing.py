import unittest

import numpy as np
import pandas as pd

from scanner.core.early_radar import build_early_radar
from scanner.core.market_data import indicators
from scanner.core.scoring import technical_scores


class EntryTimingTests(unittest.TestCase):
    def test_indicators_calculate_rsi_and_moving_average_extension(self):
        index = pd.bdate_range("2025-01-01", periods=280)
        close = pd.Series(
            100 + np.linspace(0, 35, len(index)) + 2 * np.sin(np.arange(len(index)) / 4),
            index=index,
        )
        frame = pd.DataFrame(
            {
                "Close": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Volume": 100000 + (np.arange(len(index)) % 7) * 1000,
            },
            index=index,
        )
        result = indicators(frame)
        self.assertIsNotNone(result)
        self.assertIn("rsi14", result.columns)
        self.assertIn("extension20_pct", result.columns)
        self.assertTrue(result["rsi14"].dropna().between(0, 100).all())
        self.assertTrue(np.isfinite(result["extension20_pct"].dropna()).all())

    def test_prebreakout_is_surfaced_five_percent_below_resistance(self):
        rows = [
            {
                "symbol": "EARLY", "price": 100.0, "sma20": 98.0,
                "sma50": 95.0, "sma200": 90.0, "ret1m": 0.20,
                "ret3m": 0.40, "ret6m": 0.60, "ret12m": 0.80,
                "dist20res": 0.04, "dist52": 0.10, "atr_pct": 0.02,
                "atr_pct_20ago": 0.02, "range10_pct": 0.04,
                "volume": 1.0, "vol20": 1.0, "vol20_prev": 1.0,
                "high20": 100 / 0.96, "rsi14": 66.0,
                "value_score": 8.0, "quality_score": 12.0,
                "fundamental_coverage": 5, "quality_coverage": 4,
                "sector_score": 80, "avg_traded_value_20d": 50000000,
            },
            {
                "symbol": "MID", "price": 95.0, "sma20": 96.0,
                "sma50": 94.0, "sma200": 90.0, "ret1m": 0.05,
                "ret3m": 0.10, "ret6m": 0.15, "ret12m": 0.20,
                "dist20res": 0.08, "dist52": 0.15, "atr_pct": 0.025,
                "atr_pct_20ago": 0.025, "range10_pct": 0.06,
                "volume": 0.8, "vol20": 1.0, "vol20_prev": 1.0,
                "high20": 95 / 0.92, "rsi14": 54.0,
                "value_score": 8.0, "quality_score": 12.0,
                "fundamental_coverage": 5, "quality_coverage": 4,
                "sector_score": 80, "avg_traded_value_20d": 50000000,
            },
            {
                "symbol": "LATE", "price": 115.0, "sma20": 100.0,
                "sma50": 96.0, "sma200": 90.0, "ret1m": 0.18,
                "ret3m": 0.35, "ret6m": 0.52, "ret12m": 0.70,
                "dist20res": -0.01, "dist52": 0.02, "atr_pct": 0.02,
                "atr_pct_20ago": 0.02, "range10_pct": 0.05,
                "volume": 1.6, "vol20": 1.0, "vol20_prev": 1.0,
                "high20": 115 / 1.01, "rsi14": 82.0,
                "value_score": 8.0, "quality_score": 12.0,
                "fundamental_coverage": 5, "quality_coverage": 4,
                "sector_score": 80, "avg_traded_value_20d": 50000000,
            },
        ]
        result = technical_scores(pd.DataFrame(rows))
        early = result.set_index("symbol").loc["EARLY"]
        late = result.set_index("symbol").loc["LATE"]
        self.assertEqual(early["setup"], "PRE-BREAKOUT")
        self.assertFalse(bool(early["entry_ready"]))
        self.assertFalse(bool(early["late_stage"]))
        self.assertEqual(late["setup"], "WATCH")
        self.assertTrue(bool(late["late_stage"]))
        self.assertFalse(bool(late["entry_ready"]))

    def test_rsi_above_70_alone_does_not_reject_a_tight_setup(self):
        frame = pd.DataFrame([
            {
                "symbol": "STRONG", "price": 100.0, "sma20": 98.0,
                "sma50": 95.0, "sma200": 90.0, "ret1m": 0.20,
                "ret3m": 0.40, "ret6m": 0.60, "ret12m": 0.80,
                "dist20res": 0.02, "dist52": 0.10, "atr_pct": 0.02,
                "atr_pct_20ago": 0.02, "range10_pct": 0.04,
                "volume": 1.0, "vol20": 1.0, "vol20_prev": 1.0,
                "high20": 100 / 0.98, "rsi14": 74.0,
                "value_score": 8.0, "quality_score": 12.0,
                "fundamental_coverage": 5, "quality_coverage": 4,
                "sector_score": 80, "avg_traded_value_20d": 50000000,
            },
            {
                "symbol": "BASE", "price": 95.0, "sma20": 96.0,
                "sma50": 94.0, "sma200": 90.0, "ret1m": 0.05,
                "ret3m": 0.10, "ret6m": 0.15, "ret12m": 0.20,
                "dist20res": 0.08, "dist52": 0.15, "atr_pct": 0.025,
                "atr_pct_20ago": 0.025, "range10_pct": 0.06,
                "volume": 0.8, "vol20": 1.0, "vol20_prev": 1.0,
                "high20": 95 / 0.92, "rsi14": 54.0,
                "value_score": 8.0, "quality_score": 12.0,
                "fundamental_coverage": 5, "quality_coverage": 4,
                "sector_score": 80, "avg_traded_value_20d": 50000000,
            },
        ])
        result = technical_scores(frame).set_index("symbol")
        self.assertFalse(bool(result.loc["STRONG", "late_stage"]))

    def test_early_radar_ranks_less_extended_candidate_higher(self):
        frame = pd.DataFrame([
            {
                "symbol": "EARLY", "price": 100.0, "early_score": 75,
                "trend_raw": 70, "momentum_raw": 65, "relative_strength": 0.08,
                "sector_score": 65, "quality_coverage": 3, "dist20res": 0.025,
                "range10_pct": 0.04, "liquidity_ok": True, "setup": "WATCH",
                "volume_dryup_ratio": 0.75, "up_down_volume_ratio": 1.8,
                "rsi14": 62, "extension20_pct": 0.025,
            },
            {
                "symbol": "EXTENDED", "price": 100.0, "early_score": 75,
                "trend_raw": 70, "momentum_raw": 65, "relative_strength": 0.08,
                "sector_score": 65, "quality_coverage": 3, "dist20res": 0.025,
                "range10_pct": 0.04, "liquidity_ok": True, "setup": "WATCH",
                "volume_dryup_ratio": 0.75, "up_down_volume_ratio": 1.8,
                "rsi14": 82, "extension20_pct": 0.09,
            },
        ])
        result = build_early_radar(frame)
        self.assertEqual(result.iloc[0]["symbol"], "EARLY")


if __name__ == "__main__":
    unittest.main()
