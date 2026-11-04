"""A股原始价 + 前复权价加载与存储约定测试（不访问网络）。"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marketdata.sources.akshare_source import (  # noqa: E402
    _normalize_sina_frame,
    load_cn_daily_bars_raw_and_qfq,
)


class NormalizeFrameTest(unittest.TestCase):
    def test_volume_converted_to_hands(self):
        df = pd.DataFrame({
            "date": ["2026-07-17", "2026-07-20"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.8, 10.8],
            "close": [10.2, 11.2],
            "volume": [1_000_000, 2_000_000],  # 股
            "amount": [10_000_000, 22_000_000],
        })
        out = _normalize_sina_frame(df)
        self.assertEqual(out["volume"].tolist(), [10_000.0, 20_000.0])
        self.assertEqual(out["date"].iloc[0], date(2026, 7, 17))

    def test_missing_amount(self):
        df = pd.DataFrame({
            "date": ["2026-07-17"],
            "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2],
            "volume": [100_000],
        })
        out = _normalize_sina_frame(df)
        self.assertIn("amount", out.columns)
        self.assertTrue(pd.isna(out["amount"].iloc[0]))


class RawQfqMergeTest(unittest.TestCase):
    def test_raw_and_qfq_stored_separately(self):
        raw = pd.DataFrame({
            "date": [date(2020, 1, 2), date(2026, 7, 17)],
            "open": [100.0, 200.0],
            "high": [110.0, 210.0],
            "low": [99.0, 199.0],
            "close": [105.0, 205.0],
            "volume": [10_000.0, 20_000.0],
            "amount": [1e6, 4e6],
        })
        qfq = pd.DataFrame({
            "date": [date(2020, 1, 2), date(2026, 7, 17)],
            "open": [80.0, 200.0],
            "high": [88.0, 210.0],
            "low": [79.0, 199.0],
            "close": [84.0, 205.0],
            "volume": [10_000.0, 20_000.0],
            "amount": [1e6, 4e6],
        })
        with patch("marketdata.sources.akshare_source._fetch_raw_series", return_value=raw), \
             patch("marketdata.sources.akshare_source._fetch_qfq_series", return_value=qfq):
            bars = load_cn_daily_bars_raw_and_qfq("600519", date(2020, 1, 1), date(2026, 7, 18))

        self.assertEqual(len(bars), 2)
        self.assertAlmostEqual(bars[0].close, 105.0)
        self.assertAlmostEqual(bars[0].adj_close, 84.0)
        self.assertAlmostEqual(bars[0].adj_factor, 84.0 / 105.0, places=9)
        self.assertAlmostEqual(bars[1].close, 205.0)
        self.assertAlmostEqual(bars[1].adj_close, 205.0)
        self.assertEqual(bars[1].pre_close, 105.0)
        self.assertEqual(bars[0].adjustment, "qfq")

    def test_missing_qfq_falls_back_to_raw(self):
        raw = pd.DataFrame({
            "date": [date(2026, 7, 17)],
            "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2],
            "volume": [1000.0], "amount": [1e6],
        })
        with patch("marketdata.sources.akshare_source._fetch_raw_series", return_value=raw), \
             patch("marketdata.sources.akshare_source._fetch_qfq_series", return_value=None):
            bars = load_cn_daily_bars_raw_and_qfq("600519", date(2026, 7, 1), date(2026, 7, 20))
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].adj_close, bars[0].close)
        self.assertEqual(bars[0].adj_factor, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
