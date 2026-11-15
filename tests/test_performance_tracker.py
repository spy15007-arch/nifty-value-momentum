import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scanner.core.performance_tracker import update_performance_tracker


class PerformanceTrackerTests(unittest.TestCase):
    def test_updates_forward_returns_from_prior_signal_snapshot(self):
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
            pd.DataFrame([{
                "scan_date": dates[0].date().isoformat(),
                "ticker": "TEST.NS", "symbol": "TEST",
                "signal_type": "WATCH", "signal_price": 100,
                "score": 70, "market_sentiment": "BULLISH",
            }]).to_csv(history, index=False)
            update_performance_tracker(
                diagnostics, price_data, "BULLISH",
                history_path=history, summary_path=summary,
            )
            saved = pd.read_csv(history)
            prior = saved[saved["scan_date"] == dates[0].date().isoformat()].iloc[0]
            self.assertAlmostEqual(prior["return_1d_pct"], 1.0, places=3)
            self.assertAlmostEqual(prior["return_3d_pct"], 3.0, places=3)
            self.assertAlmostEqual(prior["return_5d_pct"], 5.0, places=3)
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
