"""Tests for newly adapted JQ strategies and primitives."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.engine import run_backtest
from backtest.imported_strategies import (
    STRATEGY_JQ_MA_CONSOLIDATION_BREAKOUT,
    STRATEGY_JQ_BIAS_MOMENTUM_TIMING,
    STRATEGY_JQ_RSRS_WILLIAMS_TIMING,
    jq_ma_consolidation_breakout_signal,
    jq_bias_momentum_timing_signal,
    jq_rsrs_williams_timing_signal,
    list_adapted_strategies,
)
from factors.primitives.timing import (
    williams_r,
    rsrs_zscore_slope,
    bias_momentum_score,
    ma_confluence_ratio,
    breakout_above_high,
    consecutive_below_ma,
)


class TestWilliamsR:
    def test_oversold_range(self):
        h = np.array([11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
        l = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        c = np.array([9.1, 9.1, 9.1, 9.1, 9.1, 9.1])
        wr = williams_r(h, l, c, period=5)
        assert wr < -80

    def test_overbought_range(self):
        h = np.array([11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
        l = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        c = np.array([10.9, 10.9, 10.9, 10.9, 10.9, 10.9])
        wr = williams_r(h, l, c, period=5)
        assert wr > -20

    def test_insufficient_data_fallback(self):
        h = np.array([10.0, 11.0])
        l = np.array([9.0, 10.0])
        c = np.array([9.5, 10.5])
        assert williams_r(h, l, c, period=21) == -50.0

    def test_exact_oversold_threshold(self):
        h = np.array([10.0] * 25 + [12.0])
        l = np.array([8.0] * 25 + [8.0])
        c = np.array([8.5] * 25 + [8.1])
        wr = williams_r(h, l, c, period=21)
        assert wr < -90


class TestRsrsZscoreSlope:
    def test_positive_slope(self):
        z = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert rsrs_zscore_slope(z, k=5) > 0

    def test_negative_slope(self):
        z = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        assert rsrs_zscore_slope(z, k=5) < 0

    def test_short_input_returns_zero(self):
        assert rsrs_zscore_slope(np.array([1.0]), k=8) == 0.0

    def test_flat_returns_zero(self):
        z = np.array([0.5] * 10)
        assert abs(rsrs_zscore_slope(z, k=8)) < 1e-10


class TestBiasMomentumScore:
    def test_uptrend_gives_positive(self):
        c = np.linspace(10, 20, 200)
        score = bias_momentum_score(c, bias_period=90, momentum_days=20)
        assert score > 0 or abs(score) < 0.001

    def test_downtrend_gives_negative(self):
        c = np.linspace(15, 10, 150)
        score = bias_momentum_score(c, bias_period=90, momentum_days=20)
        assert score < 0

    def test_insufficient_data(self):
        c = np.linspace(10, 11, 50)
        assert bias_momentum_score(c, bias_period=90, momentum_days=20) == float("-inf")


class TestMaConfluenceRatio:
    def test_aligned_mas_low_diff(self):
        c = np.linspace(10, 10.5, 100)
        ratio = ma_confluence_ratio(c, periods=(13, 34, 55))
        assert 0 < ratio < 0.05

    def test_insufficient_data(self):
        c = np.linspace(10, 11, 30)
        assert ma_confluence_ratio(c, periods=(13, 34, 55)) == float("inf")

    def test_large_diff_returns_high_ratio(self):
        c = np.array([10.0] * 80 + [100.0] * 30)
        ratio = ma_confluence_ratio(c, periods=(13, 34, 55))
        assert ratio > 0.2


class TestBreakoutAboveHigh:
    def test_breakout_detected(self):
        c = np.array([10.0] * 32 + [10.5])
        assert breakout_above_high(c, lookback=30)

    def test_no_breakout(self):
        c = np.array([10.0] * 32 + [9.5])
        assert not breakout_above_high(c, lookback=30)

    def test_insufficient_data(self):
        assert not breakout_above_high(np.array([10.0] * 5), lookback=30)


class TestConsecutiveBelowMa:
    def test_below_ma(self):
        c = np.array([10.0] * 5 + [6.0, 6.0, 6.0, 5.0, 5.0])
        assert consecutive_below_ma(c, ma_period=5, days=3)

    def test_above_ma(self):
        c = np.array([11.0] * 20)
        assert not consecutive_below_ma(c, ma_period=5, days=3)

    def test_insufficient_data(self):
        assert not consecutive_below_ma(np.array([9.0] * 5), ma_period=13, days=3)


class TestMaConsolidationBreakoutSignal:
    def test_signal_runs_on_ohlcv(self):
        n = 120
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(0).normal(0.05, 0.4, n))
        df = pd.DataFrame(
            {"Open": base * 0.998, "High": base * 1.01, "Low": base * 0.99, "Close": base},
            index=dates,
        )
        signal = jq_ma_consolidation_breakout_signal(df)
        assert len(signal) == n
        assert signal.isin([0, 1]).all()

    def test_registered_and_runs_via_engine(self):
        ids = {s["id"] for s in list_adapted_strategies()}
        assert STRATEGY_JQ_MA_CONSOLIDATION_BREAKOUT in ids
        n = 120
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(1).normal(0.05, 0.4, n))
        df = pd.DataFrame({"Close": base}, index=dates)
        res = run_backtest(df, STRATEGY_JQ_MA_CONSOLIDATION_BREAKOUT, initial_cash=100_000.0)
        assert res.final_value > 0


class TestBiasMomentumTimingSignal:
    def test_signal_runs_on_ohlcv(self):
        n = 150
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(2).normal(0.05, 0.4, n))
        df = pd.DataFrame({"Close": base}, index=dates)
        signal = jq_bias_momentum_timing_signal(df, bias_period=60, momentum_days=15)
        assert len(signal) == n
        assert signal.isin([0, 1]).all()

    def test_registered_and_runs_via_engine(self):
        ids = {s["id"] for s in list_adapted_strategies()}
        assert STRATEGY_JQ_BIAS_MOMENTUM_TIMING in ids
        n = 150
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(3).normal(0.05, 0.4, n))
        df = pd.DataFrame({"Close": base}, index=dates)
        res = run_backtest(df, STRATEGY_JQ_BIAS_MOMENTUM_TIMING, initial_cash=100_000.0)
        assert res.final_value > 0


class TestRsrsWilliamsTimingSignal:
    def test_signal_runs_on_ohlcv(self):
        n = 700
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(4).normal(0.05, 0.8, n))
        base = np.maximum(base, 50)
        df = pd.DataFrame(
            {"Open": base * 0.995, "High": base * 1.02, "Low": base * 0.98, "Close": base},
            index=dates,
        )
        signal = jq_rsrs_williams_timing_signal(df, rsrs_n=10, rsrs_m=50)
        assert len(signal) == n
        assert signal.isin([0, 1]).all()

    def test_registered_and_runs_via_engine(self):
        ids = {s["id"] for s in list_adapted_strategies()}
        assert STRATEGY_JQ_RSRS_WILLIAMS_TIMING in ids


class TestNewPortfolioStrategies:
    def test_ma_consolidation_breakout_registered(self):
        from portfolio.strategy_registry import supported_portfolio_strategies
        assert "jq_ma_consolidation_breakout" in supported_portfolio_strategies()

    def test_simple_momentum_registered(self):
        from portfolio.strategy_registry import supported_portfolio_strategies
        assert "jq_simple_momentum" in supported_portfolio_strategies()

    def test_all_adapted_have_builders(self):
        from portfolio.strategy_registry import missing_adapted_portfolio_builders
        assert missing_adapted_portfolio_builders() == []

    def test_new_adapted_not_in_etf_pool(self):
        from portfolio.strategy_registry import ETF_POOL_STRATEGY_IDS
        assert "jq_ma_consolidation_breakout" not in ETF_POOL_STRATEGY_IDS
        assert "jq_simple_momentum" not in ETF_POOL_STRATEGY_IDS
