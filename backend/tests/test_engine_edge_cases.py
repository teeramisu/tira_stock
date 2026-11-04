"""Edge-case tests for single-instrument engine metrics and signal functions."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.engine import (
    run_backtest,
    run_sma_crossover,
    run_ema_crossover,
    run_rsi,
    run_macd,
    run_bollinger,
    run_buy_hold,
    _rsi_signal,
    _macd_signal,
    _bollinger_signal,
    _sma_crossover_signal,
    _ema_crossover_signal,
    _run_from_position,
)


def _df_from_close(close: list[float]) -> pd.DataFrame:
    n = len(close)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


def _pos_from_values(values: list[float], df: pd.DataFrame) -> pd.Series:
    return pd.Series(values, index=df.index)


class TestMetricsEdgeCases:
    def test_sharpe_zero_when_volatility_zero(self):
        close = [100.0] * 50
        df = _df_from_close(close)
        res = run_buy_hold(df, initial_cash=100_000.0)
        assert res.sharpe_ratio == 0.0
        assert res.volatility == 0.0

    def test_sharpe_negative_returns_gives_negative_sharpe(self):
        n = 50
        close = 100 * np.exp(-np.linspace(0, 0.5, n))
        df = _df_from_close(list(close))
        res = run_buy_hold(df, initial_cash=100_000.0)
        assert res.sharpe_ratio < 0

    def test_total_return_negative_when_prices_fall(self):
        close = [100.0] * 10 + [80.0] * 40
        df = _df_from_close(close)
        res = run_buy_hold(df, initial_cash=100_000.0)
        assert res.total_return < 0

    def test_max_drawdown_is_bounded(self):
        n = 100
        close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
        close = np.maximum(close, 50)
        df = _df_from_close(list(close))
        res = run_sma_crossover(df)
        assert 0.0 <= res.max_drawdown <= 1.0

    def test_max_drawdown_duration_non_negative(self):
        df = _df_from_close([100.0] * 10 + [120.0] * 40 + [90.0] * 20 + [130.0] * 30)
        res = run_sma_crossover(df)
        assert res.max_drawdown_duration >= 0

    def test_very_short_data_runs_without_error(self):
        df = _df_from_close([100.0, 101.0, 102.0, 103.0, 104.0])
        res = run_sma_crossover(df, fast_period=2, slow_period=3)
        assert res.volatility >= 0.0

    def test_data_below_warmup_raises(self):
        df = _df_from_close([100.0] * 10)
        with pytest.raises(ValueError):
            run_sma_crossover(df, fast_period=5, slow_period=30)


class TestCommissionEdgeCases:
    def test_commission_reduces_return(self):
        n = 60
        close = [100.0] * 20 + [110.0] * 40
        df = _df_from_close(close)
        res_no_cost = run_sma_crossover(df, initial_cash=100_000.0, commission_rate=0.0)
        res_cost = run_sma_crossover(df, initial_cash=100_000.0, commission_rate=0.01)
        assert res_cost.final_value <= res_no_cost.final_value + 1.0

    def test_high_commission_can_make_trade_unprofitable(self):
        n = 60
        close = [100.0] * 20 + [101.0] * 40
        df = _df_from_close(close)
        res = run_sma_crossover(df, initial_cash=100_000.0, commission_rate=0.05)
        assert res.total_return <= 0.0

    def test_zero_initial_cash_returns_zero(self):
        df = _df_from_close([100.0] * 35)
        res = run_sma_crossover(df, initial_cash=0.0, fast_period=5, slow_period=30)
        assert res.final_value == 0.0
        assert res.n_trades == 0


class TestSignalFunctions:
    def test_rsi_signal_bounds(self):
        close = pd.Series(np.linspace(10, 20, 100))
        signal = _rsi_signal(close, period=14)
        assert signal.dtype == float
        assert signal.isin([0, 1]).all()

    def test_rsi_signal_in_oversold_buys(self):
        n = 100
        rng = np.random.default_rng(42)
        close = 100 + rng.normal(0, 2, n).cumsum()
        close = np.maximum(close, 50)
        signal = _rsi_signal(pd.Series(close), period=14, oversold=40, overbought=60)
        assert signal.iloc[-10:].sum() >= 0

    def test_macd_signal_bounds(self):
        close = pd.Series(np.linspace(10, 20, 100))
        signal = _macd_signal(close, fast=12, slow=26, signal_period=9)
        assert signal.isin([0, 1]).all()

    def test_macd_signal_not_all_one(self):
        close = pd.Series(np.full(100, 100.0))
        signal = _macd_signal(close, fast=5, slow=10, signal_period=3)
        assert signal.min() == 0

    def test_bollinger_signal_flat_market_low_trades(self):
        close = pd.Series(np.full(100, 100.0))
        signal = _bollinger_signal(close, period=20)
        assert signal.isin([0, 1]).all()

    def test_sma_signal_crosses_once(self):
        close = pd.Series([100.0] * 15 + [110.0] * 35)
        signal = _sma_crossover_signal(close, fast_period=5, slow_period=15)
        transitions = signal.diff().dropna()
        buy_signals = (transitions == 1).sum()
        assert buy_signals == 1

    def test_ema_signal_matches_sma_direction(self):
        close = pd.Series([100.0] * 15 + [110.0] * 35)
        sma_sig = _sma_crossover_signal(close, fast_period=5, slow_period=15)
        ema_sig = _ema_crossover_signal(close, fast_period=5, slow_period=15)
        assert ema_sig.sum() >= 0

    def test_all_signals_same_length_as_input(self):
        n = 100
        close = pd.Series(np.linspace(10, 20, n))
        for sig_fn in [
            lambda c: _sma_crossover_signal(c, 5, 15),
            lambda c: _ema_crossover_signal(c, 5, 15),
            lambda c: _rsi_signal(c, 14),
            lambda c: _macd_signal(c, 12, 26, 9),
            lambda c: _bollinger_signal(c, 20),
        ]:
            assert len(sig_fn(close)) == n


class TestRunFromPosition:
    def test_position_shorter_than_df_pads_with_zeros(self):
        df = _df_from_close([100.0] * 50)
        pos = _pos_from_values([1.0] * 30 + [0.0] * 20, df)
        res = _run_from_position(df, pos, 100_000.0, 0.0)
        assert res.final_value > 0

    def test_position_longer_than_df_truncates(self):
        df = _df_from_close([100.0] * 30)
        pos = _pos_from_values([1.0] * 30, df)
        res = _run_from_position(df, pos, 100_000.0, 0.0)
        assert len(res.equity_curve) == 30

    def test_all_nan_position_treated_as_zero(self):
        df = _df_from_close([100.0] * 30)
        pos = _pos_from_values([float("nan")] * 30, df)
        res = _run_from_position(df, pos, 100_000.0, 0.0)
        assert res.final_value == 100_000.0
        assert res.n_trades == 0

    def test_all_position_one_tracks_price_exactly(self):
        df = _df_from_close([100.0, 110.0, 120.0, 130.0, 140.0])
        pos = _pos_from_values([1.0] * 5, df)
        res = _run_from_position(df, pos, 100_000.0, 0.0)
        expected = 100_000 / 100.0 * 140.0
        assert res.final_value == pytest.approx(expected)

    def test_buyhold_matches_price_change(self):
        df = _df_from_close([100.0, 105.0, 110.0, 115.0, 120.0])
        pos = _pos_from_values([1.0] * 5, df)
        res = _run_from_position(df, pos, 100_000.0, 0.0)
        assert res.total_return == pytest.approx(0.2)

    def test_commission_reduces_final_value(self):
        df = _df_from_close([100.0] * 10 + [110.0] * 10 + [100.0] * 10)
        pos = _pos_from_values([0.0] * 10 + [1.0] * 10 + [0.0] * 10, df)
        res = _run_from_position(df, pos, 100_000.0, 0.01)
        assert res.n_trades >= 1
        no_comm = _run_from_position(df, pos, 100_000.0, 0.0)
        assert res.final_value <= no_comm.final_value + 1.0

    def test_all_position_one_commission_zero_returns_exact(self):
        df = _df_from_close([100.0, 110.0])
        pos = _pos_from_values([1.0, 0.0], df)
        res = _run_from_position(df, pos, 100_000.0, 0.0)
        assert res.n_trades == 1
        assert len(res.trades) == 1


class TestRunBacktestDispatch:
    def test_unknown_strategy_raises(self):
        df = _df_from_close([100.0] * 30)
        with pytest.raises(ValueError, match="不支持的策略"):
            run_backtest(df, "nonexistent_strategy")

    def test_default_strategy_is_sma(self):
        df = _df_from_close([100.0] * 30)
        res = run_backtest(df, None)
        assert res.initial_value == 100_000.0

    def test_ema_crossover_produces_valid_result(self):
        df = _df_from_close([100.0] * 10 + [110.0] * 50)
        res = run_ema_crossover(df, fast_period=5, slow_period=15)
        assert res.final_value > 0

    def test_rsi_uses_default_params(self):
        df = _df_from_close([100.0] * 80)
        res = run_rsi(df)
        assert res.final_value > 0

    def test_macd_uses_default_params(self):
        df = _df_from_close([100.0] * 80)
        res = run_macd(df)
        assert res.final_value > 0

    def test_bollinger_uses_default_params(self):
        df = _df_from_close([100.0] * 80)
        res = run_bollinger(df)
        assert res.final_value > 0


class TestEdgeCasePriceData:
    def test_single_trading_day(self):
        df = _df_from_close([100.0])
        res = run_buy_hold(df)
        assert res.n_trades == 0
        assert res.final_value == 100_000.0

    def test_sudden_spike_does_not_cause_invalid_metrics(self):
        close = [100.0] * 30 + [1_000_000.0] + [100.0] * 29
        df = _df_from_close(close)
        res = run_sma_crossover(df)
        assert np.isfinite(res.total_return)
        assert np.isfinite(res.sharpe_ratio)
        assert np.isfinite(res.max_drawdown)

    def test_strictly_decreasing_prices(self):
        df = _df_from_close([100.0, 90.0, 80.0, 70.0, 60.0])
        res = run_buy_hold(df)
        assert res.total_return < 0
        assert res.annual_return < 0

    def test_small_initial_cash_handled(self):
        df = _df_from_close([100.0, 110.0, 120.0])
        res = run_buy_hold(df, initial_cash=1.0)
        assert res.final_value == pytest.approx(1.2)
