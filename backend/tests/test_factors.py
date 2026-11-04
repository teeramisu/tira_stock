"""Unit tests for shared factor primitives."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factors.primitives.etf_selection import build_etf_momentum_row, select_best_etf_momentum
from factors.primitives.momentum import polyfit_log_momentum_score, wpetf_momentum_score
from factors.primitives.oscillators import rsi
from factors.primitives.timing import (
    index_pullback_entry_rate,
    price_emotion_monitor,
    volume_emotion_monitor,
)
from factors.primitives.trend import bbi_price_ratio, compute_bbi


def test_rsi_bounds():
    close = pd.Series(np.linspace(10, 20, 30))
    values = rsi(close, 14)
    assert values.dropna().between(0, 100).all()


def test_bbi_ratio_bullish_when_price_above_composite_ma():
    closes = np.linspace(10, 15, 30)
    ratio = bbi_price_ratio(closes, (3, 6, 12, 24))
    assert ratio < 1.0


def test_volume_emotion_bearish_on_declining_volume():
    volume = np.array([100, 100, 100, 100, 100, 100, 80, 70, 60, 50, 40, 30, 20], dtype=float)
    assert volume_emotion_monitor(volume, ma_window=7, below_ma_days=6) == -1


def test_polyfit_momentum_prefers_uptrend():
    up = np.linspace(10, 15, 25)
    flat = np.full(25, 10.0)
    assert polyfit_log_momentum_score(up) > polyfit_log_momentum_score(flat)


def test_wpetf_score_positive_on_trend():
    rng = np.random.default_rng(1)
    up = np.linspace(10, 14, 30) + rng.normal(0, 0.2, 30)
    assert wpetf_momentum_score(up) > 0


def test_etf_momentum_row_selection():
    up = np.linspace(10, 12, 14)
    flat = np.full(14, 10.0)
    rows = [
        build_etf_momentum_row("UP", up, period_bars=13),
        build_etf_momentum_row("FLAT", flat, period_bars=13),
    ]
    rows = [row for row in rows if row is not None]
    best = select_best_etf_momentum(rows)
    assert best is not None
    assert best.symbol == "UP"


def test_pullback_rate_lower_after_drawdown():
    peak_then_drop = np.concatenate([np.linspace(10, 20, 200), np.linspace(20, 12, 50)])
    shallow = np.linspace(10, 12, 250)
    assert index_pullback_entry_rate(peak_then_drop) < index_pullback_entry_rate(shallow)


def test_compute_bbi_series_length():
    close = pd.Series(np.linspace(1, 2, 30))
    bbi = compute_bbi(close, (3, 6, 12, 24))
    assert len(bbi) == len(close)
