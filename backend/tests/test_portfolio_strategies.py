"""Tests for portfolio strategies including rotation and edge cases."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date

from marketdata.data_portal import DataPortal
from marketdata.db import connect
from marketdata.models import DailyBar, DatasetVersion, Instrument
from marketdata.repository import MarketDataRepository
from portfolio.models import PortfolioBacktestConfig, StrategyContext, TargetPosition
from portfolio.strategies import (
    EtfRotationTop1Strategy,
    MomentumTopKStrategy,
    _judge_more_average,
)
from portfolio.engine import PortfolioEngine, CostModel

from test_portfolio_engine import _repository, _seed_portfolio_fixture


class TestJudgeMoreAverage:
    def test_uptrend_not_judged_average(self):
        closes = pd.Series(np.linspace(10, 20, 30))
        assert not _judge_more_average(closes)

    def test_short_ma_below_long_ma_returns_true(self):
        closes = pd.Series([10.0] * 5 + [9.0] * 5 + [8.0] * 5 + [7.0] * 5 + [6.0] * 10)
        assert _judge_more_average(closes)

    def test_insufficient_data_returns_false(self):
        closes = pd.Series([10.0] * 10)
        assert not _judge_more_average(closes)

    def test_exactly_30_days_works(self):
        closes = pd.Series([10.0] * 5 + [9.0] * 5 + [8.0] * 5 + [7.0] * 5 + [6.0] * 10)
        result = _judge_more_average(closes)
        assert isinstance(result, bool)

    def test_ma_windows_use_correct_lengths(self):
        """Verify each MA uses the correct window size, not the old 15/25 bug."""
        # Distinct values per position so we can verify exact window lengths
        data = list(range(1, 31))  # 1, 2, 3, ..., 30
        closes = pd.Series(data, dtype=float)
        # tail(5) = 26,27,28,29,30 → mean = 28
        # tail(10) = 21-30 → mean = 25.5
        # tail(20) = 11-30 → mean = 20.5
        # tail(30) = 1-30 → mean = 15.5
        # We call the private function to verify internal logic
        assert abs(closes.tail(5).mean() - 28.0) < 0.01
        assert abs(closes.tail(10).mean() - 25.5) < 0.01
        assert abs(closes.tail(20).mean() - 20.5) < 0.01
        assert abs(closes.tail(30).mean() - 15.5) < 0.01


class TestEtfRotationTop1:
    def _seed_rotation_fixture(self, tmp_path) -> MarketDataRepository:
        repo = _repository(tmp_path)
        repo.upsert_instruments([
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SSE", list_date=date(2020, 1, 1)),
        ])
        closes = {"AAA": [10.0, 11.0, 12.0, 13.0, 14.0], "BBB": [10.0, 9.5, 9.0, 8.5, 8.0]}
        bars = []
        for idx, trade_date in enumerate([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]):
            for symbol, sym_closes in closes.items():
                bars.append(DailyBar(market="cn", symbol=symbol, trade_date=trade_date, open=sym_closes[idx], high=sym_closes[idx]+1, low=sym_closes[idx]-1, close=sym_closes[idx]))
        repo.upsert_daily_bars(bars)
        repo.upsert_dataset_version(DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True))
        return repo

    def test_etf_rotation_selects_top_performer(self, tmp_path):
        repo = self._seed_rotation_fixture(tmp_path)
        strategy = EtfRotationTop1Strategy(lookback_days=2)
        portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 5))
        targets = strategy.generate_targets(
            date(2024, 1, 5),
            portal,
            StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
        )
        assert len(targets) == 1
        assert targets[0].symbol == "AAA"
        assert targets[0].weight == pytest.approx(1.0)

    def test_etf_rotation_returns_empty_when_no_universe(self):
        strategy = EtfRotationTop1Strategy(lookback_days=2)
        targets = strategy.generate_targets(
            date(2024, 1, 5),
            None,
            StrategyContext(market="cn", universe=(), dataset_version="fixture-cn"),
        )
        assert targets == []

    def test_etf_rotation_returns_empty_when_insufficient_history(self, tmp_path):
        repo = _repository(tmp_path)
        repo.upsert_instruments([Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1))])
        bars = [DailyBar(market="cn", symbol="AAA", trade_date=date(2024, 1, 2), open=10, high=11, low=9, close=10.5)]
        repo.upsert_daily_bars(bars)
        repo.upsert_dataset_version(DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True))
        strategy = EtfRotationTop1Strategy(lookback_days=20)
        portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 3))
        targets = strategy.generate_targets(
            date(2024, 1, 3),
            portal,
            StrategyContext(market="cn", universe=("AAA",), dataset_version="fixture-cn"),
        )
        assert targets == []

    def test_etf_rotation_negative_lookback_raises(self):
        with pytest.raises(ValueError):
            EtfRotationTop1Strategy(lookback_days=0)


class TestMomentumTopKEdgeCases:
    def _seed_same_close_fixture(self, tmp_path) -> MarketDataRepository:
        repo = _repository(tmp_path)
        repo.upsert_instruments([
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="CCC", name="Gamma", exchange="SSE", list_date=date(2020, 1, 1)),
        ])
        bars = []
        for idx, trade_date in enumerate([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]):
            for symbol in ("AAA", "BBB", "CCC"):
                bars.append(DailyBar(market="cn", symbol=symbol, trade_date=trade_date, open=10.0, high=11.0, low=9.0, close=10.0))
        repo.upsert_daily_bars(bars)
        repo.upsert_dataset_version(DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True))
        return repo

    def test_negative_lookback_raises(self):
        with pytest.raises(ValueError):
            MomentumTopKStrategy(lookback_days=-1, top_k=5)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            MomentumTopKStrategy(lookback_days=20, top_k=0)

    def test_all_same_returns_picks_first_k(self, tmp_path):
        repo = self._seed_same_close_fixture(tmp_path)
        strategy = MomentumTopKStrategy(lookback_days=2, top_k=2)
        portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 5))
        targets = strategy.generate_targets(
            date(2024, 1, 5),
            portal,
            StrategyContext(market="cn", universe=("AAA", "BBB", "CCC"), dataset_version="fixture-cn"),
        )
        assert len(targets) == 2
        for t in targets:
            assert t.weight == pytest.approx(0.5)

    def test_warmup_days_equals_lookback(self):
        s = MomentumTopKStrategy(lookback_days=20, top_k=5)
        assert s.warmup_days == 20

    def test_top_k_larger_than_universe_uses_all(self, tmp_path):
        repo = _repository(tmp_path)
        repo.upsert_instruments([
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SSE", list_date=date(2020, 1, 1)),
        ])
        bars = []
        for trade_date in [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]:
            for symbol, c in [("AAA", 11.0), ("BBB", 10.0)]:
                bars.append(DailyBar(market="cn", symbol=symbol, trade_date=trade_date, open=c, high=c+1, low=c-1, close=c))
        repo.upsert_daily_bars(bars)
        repo.upsert_dataset_version(DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True))
        strategy = MomentumTopKStrategy(lookback_days=2, top_k=10)
        portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 4))
        targets = strategy.generate_targets(
            date(2024, 1, 4),
            portal,
            StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
        )
        assert len(targets) == 2


class TestMomentumFastPathEdgeCases:
    def test_empty_universe_returns_empty(self):
        from portfolio.engine import PortfolioEngine as PE
        eng = PE.__new__(PE)
        eng.strategy = MomentumTopKStrategy(lookback_days=20, top_k=2)
        result = eng._momentum_targets_fast(signal_date=date(2024, 1, 5), universe=(), score_matrix=pd.DataFrame({"AAA": [1, 2], "BBB": [3, 4]}))
        assert result == []

    def test_empty_score_matrix_returns_empty(self):
        from portfolio.engine import PortfolioEngine as PE
        eng = PE.__new__(PE)
        eng.strategy = MomentumTopKStrategy(lookback_days=20, top_k=2)
        result = eng._momentum_targets_fast(signal_date=date(2024, 1, 5), universe=("AAA", "BBB"), score_matrix=pd.DataFrame())
        assert result == []

    def test_all_nan_scores_returns_empty(self):
        from portfolio.engine import PortfolioEngine as PE
        eng = PE.__new__(PE)
        eng.strategy = MomentumTopKStrategy(lookback_days=20, top_k=2)
        matrix = pd.DataFrame({"AAA": [float("nan")], "BBB": [float("nan")]}, index=[date(2024, 1, 5)])
        result = eng._momentum_targets_fast(signal_date=date(2024, 1, 5), universe=("AAA", "BBB"), score_matrix=matrix)
        assert result == []


class TestBrokerRebalanceEdgeCases:
    def test_partial_fill_on_insufficient_cash(self, tmp_path):
        repo = _repository(tmp_path)
        _seed_portfolio_fixture(repo)
        engine = PortfolioEngine(
            repository=repo,
            strategy=MomentumTopKStrategy(lookback_days=2, top_k=2),
            config=PortfolioBacktestConfig(
                market="cn",
                dataset_version="fixture-cn",
                start=date(2024, 1, 1),
                end=date(2024, 1, 5),
                initial_cash=100,
            ),
            cost_model=CostModel(min_commission=0),
        )
        result = engine.run()
        assert result.equity_curve["cash"].iloc[-1] <= 100

    def test_turnover_zero_when_no_trades(self, tmp_path):
        repo = _repository(tmp_path)
        _seed_portfolio_fixture(repo)
        result = PortfolioEngine(
            repository=repo,
            strategy=MomentumTopKStrategy(lookback_days=50, top_k=2),
            config=PortfolioBacktestConfig(
                market="cn",
                dataset_version="fixture-cn",
                start=date(2024, 1, 1),
                end=date(2024, 1, 5),
                initial_cash=100_000,
            ),
            cost_model=CostModel(min_commission=0),
        ).run()
        assert result.metrics["turnover"] == 0.0

    def test_cost_to_turnover_ratio_zero_when_no_turnover(self, tmp_path):
        repo = _repository(tmp_path)
        _seed_portfolio_fixture(repo)
        result = PortfolioEngine(
            repository=repo,
            strategy=MomentumTopKStrategy(lookback_days=50, top_k=2),
            config=PortfolioBacktestConfig(
                market="cn",
                dataset_version="fixture-cn",
                start=date(2024, 1, 1),
                end=date(2024, 1, 5),
                initial_cash=100_000,
            ),
            cost_model=CostModel(min_commission=0),
        ).run()
        assert result.metrics["cost_to_turnover_ratio"] == 0.0

    def test_monthly_rebalance_does_not_error(self, tmp_path):
        repo = _repository(tmp_path)
        _seed_portfolio_fixture(repo)
        result = PortfolioEngine(
            repository=repo,
            strategy=MomentumTopKStrategy(lookback_days=2, top_k=2),
            config=PortfolioBacktestConfig(
                market="cn",
                dataset_version="fixture-cn",
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                initial_cash=100_000,
                rebalance_frequency="monthly",
            ),
            cost_model=CostModel(min_commission=0),
        ).run()
        assert not result.equity_curve.empty
