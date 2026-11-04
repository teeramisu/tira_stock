"""Tests for batch JoinQuant OHLCV adapters."""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.imported_strategies import (
    STRATEGY_JQ_ETF_MA_CROSS,
    STRATEGY_JQ_ETF_MOMENTUM_RSRS_TOP1,
    STRATEGY_JQ_ETF_MOMENTUM_VOLUME_GATE,
    STRATEGY_JQ_ETF_POLYFIT_TOP1,
    STRATEGY_JQ_ETF_VALUE_AVERAGING,
    STRATEGY_JQ_INDUSTRY_ETF_BBI,
    STRATEGY_JQ_SMALL_CAP_TOP_K,
    STRATEGY_JQ_WIDE_BASE_BBI,
    STRATEGY_JQ_WPETF_MOMENTUM,
    _adapted_source_ids,
    jq_etf_ma_cross_timing_signal,
    list_adapted_strategies,
)
from backtest.jq_bbi import bbi_price_ratio, ema5_bearish_gate, market_breadth_bullish
from factors.primitives.momentum import polyfit_log_momentum_score
from backtest.engine import run_backtest
from marketdata.data_portal import DataPortal
from marketdata.models import DailyBar, DatasetVersion, Instrument
from portfolio.jq_strategies import (
    EtfPolyfitMomentumTop1Strategy,
    JqEtfMomentumVolumeGateTop1Strategy,
    JqIndustryEtfBbiRotationStrategy,
    JqSmallCapTopKStrategy,
    JqWideBaseEtfBbiRotationStrategy,
    WpetfMomentumTop1Strategy,
)
from portfolio.models import StrategyContext
from test_portfolio_engine import _repository, _seed_jq_value_fixture


def test_polyfit_momentum_prefers_uptrend():
    up = np.linspace(10, 15, 25)
    flat = np.full(25, 10.0)
    assert polyfit_log_momentum_score(up) > polyfit_log_momentum_score(flat)


def test_adapted_strategies_cover_top30_batch():
    ids = {s["id"] for s in list_adapted_strategies()}
    assert STRATEGY_JQ_ETF_POLYFIT_TOP1 in ids
    assert STRATEGY_JQ_ETF_MOMENTUM_RSRS_TOP1 in ids
    assert STRATEGY_JQ_SMALL_CAP_TOP_K in ids
    assert STRATEGY_JQ_ETF_MA_CROSS in ids
    assert STRATEGY_JQ_INDUSTRY_ETF_BBI in ids
    assert STRATEGY_JQ_ETF_MOMENTUM_VOLUME_GATE in ids
    assert STRATEGY_JQ_WIDE_BASE_BBI in ids
    assert STRATEGY_JQ_ETF_VALUE_AVERAGING in ids
    assert STRATEGY_JQ_WPETF_MOMENTUM in ids


def test_source_id_aliases():
    by_id = {s["id"]: s for s in list_adapted_strategies()}
    rsrs_ids = set(_adapted_source_ids(by_id["jq_rsrs_timing"]))
    assert "jq-0004" in rsrs_ids
    assert "jq-0008" in rsrs_ids
    small_ids = set(_adapted_source_ids(by_id[STRATEGY_JQ_SMALL_CAP_TOP_K]))
    assert "jq-0006" in small_ids
    assert "jq-0023" in small_ids


def test_etf_polyfit_top1_picks_best_momentum(tmp_path):
    repo = _repository(tmp_path)
    _seed_jq_value_fixture(repo, days=40)
    strategy = EtfPolyfitMomentumTop1Strategy(lookback_days=10)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 2, 10))
    targets = strategy.generate_targets(
        date(2024, 2, 10),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "AAA"


def test_small_cap_top_k_picks_smallest_liquidity(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    _seed_jq_value_fixture(repo, days=40)
    strategy = JqSmallCapTopKStrategy(top_k=1, lookback_days=5, require_ma5=False)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 2, 10))
    targets = strategy.generate_targets(
        date(2024, 2, 10),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "BBB"


def _seed_industry_etf_fixture(repo, *, days: int = 100) -> None:
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="IND_A", name="Industry A", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="IND_B", name="Industry B", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="510300", name="HS300", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="510500", name="CSI500", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="511880", name="Bond ETF", exchange="SSE", list_date=date(2020, 1, 1)),
        ]
    )
    bars = []
    start = date(2024, 1, 1)
    for offset in range(days):
        trade_date = (pd.Timestamp(start) + pd.Timedelta(days=offset)).date()
        bench = 10.0 + offset * 0.12 + (0.05 if offset % 5 else -0.02)
        ind_a = 20.0 + offset * 0.20
        ind_b = 15.0 + offset * 0.02
        bond = 100.0 + offset * 0.001
        bench500 = bench * 1.01
        bars.extend(
            [
                DailyBar(market="cn", symbol="510300", trade_date=trade_date, open=bench, high=bench + 0.2, low=bench - 0.2, close=bench),
                DailyBar(market="cn", symbol="510500", trade_date=trade_date, open=bench500, high=bench500 + 0.2, low=bench500 - 0.2, close=bench500),
                DailyBar(market="cn", symbol="IND_A", trade_date=trade_date, open=ind_a, high=ind_a + 0.2, low=ind_a - 0.2, close=ind_a),
                DailyBar(market="cn", symbol="IND_B", trade_date=trade_date, open=ind_b, high=ind_b + 0.2, low=ind_b - 0.2, close=ind_b),
                DailyBar(market="cn", symbol="511880", trade_date=trade_date, open=bond, high=bond + 0.01, low=bond - 0.01, close=bond),
            ]
        )
    repo.upsert_daily_bars(bars)
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True)
    )


def test_industry_etf_bbi_picks_strongest_etf(tmp_path):
    repo = _repository(tmp_path)
    _seed_industry_etf_fixture(repo, days=100)
    strategy = JqIndustryEtfBbiRotationStrategy(
        etf_pool=("IND_A", "IND_B"),
        bond_symbol="511880",
        timing_symbol="510300",
        market_symbols=("510300", "510500"),
        bbi_periods=(5, 8, 13, 21),
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 4, 10))
    targets = strategy.generate_targets(
        date(2024, 4, 10),
        portal,
        StrategyContext(
            market="cn",
            universe=("IND_A", "IND_B", "511880"),
            dataset_version="fixture-cn",
        ),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "IND_A"
    assert targets[0].weight == pytest.approx(1.0)


def test_industry_etf_bbi_falls_back_to_bond_on_ema_bearish(tmp_path):
    repo = _repository(tmp_path)
    _seed_industry_etf_fixture(repo, days=100)
    crash = []
    for offset in range(5):
        trade_date = (pd.Timestamp("2024-04-11") + pd.Timedelta(days=offset)).date()
        px = max(5.0, 20.0 - offset * 3.0)
        crash.append(
            DailyBar(
                market="cn",
                symbol="510300",
                trade_date=trade_date,
                open=px,
                high=px + 0.1,
                low=px - 0.1,
                close=px,
            )
        )
    repo.upsert_daily_bars(crash)
    strategy = JqIndustryEtfBbiRotationStrategy(
        etf_pool=("IND_A", "IND_B"),
        bond_symbol="511880",
        timing_symbol="510300",
        market_symbols=("510300", "510500"),
        bbi_periods=(5, 8, 13, 21),
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 4, 15))
    targets = strategy.generate_targets(
        date(2024, 4, 15),
        portal,
        StrategyContext(market="cn", universe=("IND_A", "IND_B", "511880"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "511880"


def test_jq_etf_ma_cross_runs_on_ohlcv():
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(np.random.default_rng(0).normal(0.05, 0.4, n))
    df = pd.DataFrame(
        {
            "Open": base * 0.998,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
        },
        index=dates,
    )
    signal = jq_etf_ma_cross_timing_signal(df)
    assert len(signal) == n
    res = run_backtest(df, STRATEGY_JQ_ETF_MA_CROSS, initial_cash=100_000.0)
    assert res.final_value > 0


def test_jq_etf_value_averaging_timing_ignores_sma_params():
    from backtest.imported_strategies import STRATEGY_JQ_ETF_VALUE_AVERAGING_TIMING

    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(np.random.default_rng(1).normal(0.05, 0.4, n))
    df = pd.DataFrame(
        {
            "Open": base * 0.998,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
        },
        index=dates,
    )
    res = run_backtest(
        df,
        STRATEGY_JQ_ETF_VALUE_AVERAGING_TIMING,
        initial_cash=100_000.0,
        fast_period=10,
        slow_period=30,
    )
    assert res.final_value > 0


def _seed_etf_rotation_fixture(repo, *, days: int = 60) -> None:
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="AAA", name="ETF A", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="ETF B", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="510300", name="HS300", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="510500", name="CSI500", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="511880", name="Bond", exchange="SSE", list_date=date(2020, 1, 1)),
        ]
    )
    bars = []
    start = date(2024, 1, 1)
    for offset in range(days):
        trade_date = (pd.Timestamp(start) + pd.Timedelta(days=offset)).date()
        noise = 0.08 if offset % 4 == 0 else -0.03
        a = 10.0 + offset * 0.25 + noise + (offset % 7) * 0.01
        b = 10.0 + offset * 0.02
        bench = 10.0 + offset * 0.05
        vol_a = 1_000_000 + offset * 1000
        vol_bench = 2_000_000
        bars.extend(
            [
                DailyBar(market="cn", symbol="AAA", trade_date=trade_date, open=a, high=a + 0.2, low=a - 0.2, close=a, volume=vol_a),
                DailyBar(market="cn", symbol="BBB", trade_date=trade_date, open=b, high=b + 0.2, low=b - 0.2, close=b, volume=500_000),
                DailyBar(market="cn", symbol="510300", trade_date=trade_date, open=bench, high=bench + 0.2, low=bench - 0.2, close=bench, volume=vol_bench),
                DailyBar(market="cn", symbol="510500", trade_date=trade_date, open=bench, high=bench + 0.2, low=bench - 0.2, close=bench, volume=vol_bench),
                DailyBar(market="cn", symbol="511880", trade_date=trade_date, open=100, high=100.01, low=99.99, close=100, volume=100_000),
            ]
        )
    repo.upsert_daily_bars(bars)
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True)
    )


def test_jq0020_volume_gate_picks_momentum_leader(tmp_path):
    repo = _repository(tmp_path)
    _seed_etf_rotation_fixture(repo, days=60)
    strategy = JqEtfMomentumVolumeGateTop1Strategy(
        etf_pool=("AAA", "BBB"),
        bond_symbol="511880",
        period_bars=13,
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 2, 28))
    targets = strategy.generate_targets(
        date(2024, 2, 28),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB", "511880"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "AAA"


def test_wide_base_bbi_rotation_picks_leader(tmp_path):
    repo = _repository(tmp_path)
    _seed_etf_rotation_fixture(repo, days=60)
    strategy = JqWideBaseEtfBbiRotationStrategy(
        etf_pool=("AAA", "BBB"),
        breadth_symbols=("510300", "510500"),
        bbi_periods=(3, 6, 12, 24),
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 2, 28))
    targets = strategy.generate_targets(
        date(2024, 2, 28),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "AAA"


def test_wpetf_top1_picks_best(tmp_path):
    repo = _repository(tmp_path)
    _seed_etf_rotation_fixture(repo, days=60)
    strategy = WpetfMomentumTop1Strategy(lookback_days=25, etf_pool=("AAA", "BBB"), min_score=0.0)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 2, 28))
    targets = strategy.generate_targets(
        date(2024, 2, 28),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "AAA"
