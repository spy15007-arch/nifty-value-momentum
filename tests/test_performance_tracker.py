import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scanner.core.performance_tracker import update_performance_tracker


class PerformanceTrackerTests(unittest.TestCase):
    def test_records_signal_and_forward_close_returns(self):
        dates = pd.bdate_range("2026-01-01", periods=14)
        prices = [100 + i for i in range(14)]
        columns = pd.MultiIndex.from_product([["TEST.NS"], ["Close"]])
        price_data = pd.DataFrame(
            [[value] for value in prices], index=dates, columns=columns
        )
        diagnostics = pd.DataFrame([{
            "ticker": "TEST.NS", "symbol": "TEST", "setup": "WATCH",
            "ranking_score": 70, "final_score": 68, "early_score": 75,
            "price": 113, "trend_raw": 70, "momentum_raw": 65,
            "relative_strength": 0.05, "sector_score": 60,
            "quality_coverage": 3, "dist20res": 0.02,
            "range10_pct": 0.05, "liquidity_ok": True,
        }])
        with tempfile.TemporaryDirectory() as temp:
            history = Path(temp) / "history.csv"
            summary = Path(temp) / "summary.csv"
            update_performance_tracker(
                diagnostics, price_data, "BULLISH",
                history_path=history, summary_path=summary,
            )
            saved = pd.read_csv(history)
            self.assertEqual(len(saved), 1)
            self.assertAlmostEqual(saved.iloc[0]["return_1d_pct"], 0.88496, places=3)
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
