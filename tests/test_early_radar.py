import unittest

import pandas as pd

from scanner.core.early_radar import build_early_radar


class EarlyRadarTests(unittest.TestCase):
    def test_finds_quality_candidate_before_strict_prebreakout(self):
        frame = pd.DataFrame([
            {
                "symbol": "ASAHIINDIA", "price": 990.0,
                "early_score": 72, "trend_raw": 62, "momentum_raw": 58,
                "relative_strength": 0.04, "sector_score": 72,
                "quality_coverage": 3, "dist20res": 0.035,
                "range10_pct": 0.07, "liquidity_ok": True, "setup": "WATCH",
            },
            {
                "symbol": "TOO_FAR", "price": 100.0,
                "early_score": 80, "trend_raw": 70, "momentum_raw": 70,
                "relative_strength": 0.10, "sector_score": 70,
                "quality_coverage": 3, "dist20res": 0.15,
                "range10_pct": 0.05, "liquidity_ok": True, "setup": "WATCH",
            },
            {
                "symbol": "WEAK_RS", "price": 100.0,
                "early_score": 80, "trend_raw": 70, "momentum_raw": 70,
                "relative_strength": -0.01, "sector_score": 70,
                "quality_coverage": 3, "dist20res": 0.03,
                "range10_pct": 0.05, "liquidity_ok": True, "setup": "WATCH",
            },
        ])

        result = build_early_radar(frame)

        self.assertEqual(result["symbol"].tolist(), ["ASAHIINDIA"])
        self.assertAlmostEqual(result.iloc[0]["resistance"], 990 / 0.965)
        self.assertGreater(result.iloc[0]["trigger"], result.iloc[0]["resistance"])

    def test_does_not_duplicate_actionable_breakouts(self):
        frame = pd.DataFrame([{
            "symbol": "ACTIONABLE", "price": 100.0,
            "early_score": 80, "trend_raw": 70, "momentum_raw": 70,
            "relative_strength": 0.10, "sector_score": 70,
            "quality_coverage": 3, "dist20res": 0.03,
            "range10_pct": 0.05, "liquidity_ok": True, "setup": "PRE-BREAKOUT",
        }])

        self.assertTrue(build_early_radar(frame).empty)


if __name__ == "__main__":
    unittest.main()
