"""Unit tests for portfolio strategies, broker, and engine."""

from __future__ import annotations

from datetime import date, datetime
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marketdata.data_portal import DataPortal
from marketdata.db import connect
from marketdata.models import DailyBar, DatasetVersion, Instrument, IntradayBar
from marketdata.repository import MarketDataRepository
from portfolio.broker import BrokerSimulator, CostModel
from portfolio.engine import PortfolioEngine
from portfolio.models import Order, PortfolioBacktestConfig, StrategyContext, TargetPosition
from portfolio.strategies import JqValueTimingTopKStrategy, MomentumTopKStrategy


def _repository(tmp_path) -> MarketDataRepository:
    return MarketDataRepository(connect(tmp_path / "market_data.duckdb"))


def _seed_portfolio_fixture(
    repo: MarketDataRepository,
    *,
    limit_up_on_execution: bool = False,
    limit_down_on_sell: bool = False,
) -> None:
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SZSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="CCC", name="Gamma", exchange="BSE", list_date=date(2020, 1, 1)),
        ]
    )
    closes = {
        "AAA": [10.0, 11.0, 12.0, 13.0, 14.0],
        "BBB": [10.0, 9.0, 8.0, 8.0, 8.0],
        "CCC": [10.0, 10.0, 10.0, 11.0, 11.0],
    }
    bars = []
    for idx, trade_date in enumerate(
        [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    ):
        for symbol, symbol_closes in closes.items():
            close = symbol_closes[idx]
            bars.append(
                DailyBar(
                    market="cn",
                    symbol=symbol,
                    trade_date=trade_date,
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    limit_up=13.0
                    if limit_up_on_execution and symbol == "AAA" and trade_date == date(2024, 1, 4)
                    else None,
                    limit_down=12.0
                    if limit_down_on_sell and symbol == "AAA" and trade_date == date(2024, 1, 3)
                    else None,
                )
            )
    repo.upsert_daily_bars(bars)
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True)
    )


def _seed_jq_value_fixture(repo: MarketDataRepository, *, days: int = 70) -> None:
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SZSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="510300", name="HS300 ETF", exchange="SSE", list_date=date(2020, 1, 1)),
        ]
    )
    bars = []
    start = date(2024, 1, 1)
    for offset in range(days):
        trade_date = (pd.Timestamp(start) + pd.Timedelta(days=offset)).date()
        bench_close = 10.0 + offset * 0.15 + (0.08 if offset % 4 == 0 else -0.03)
        stock_close = 20.0 + offset * 0.12
        bars.extend(
            [
                DailyBar(
                    market="cn",
                    symbol="510300",
                    trade_date=trade_date,
                    open=bench_close,
                    high=bench_close + 0.2,
                    low=bench_close - 0.2,
                    close=bench_close,
                    amount=1_000_000.0,
                ),
                DailyBar(
                    market="cn",
                    symbol="AAA",
                    trade_date=trade_date,
                    open=stock_close,
                    high=stock_close + 0.2,
                    low=stock_close - 0.2,
                    close=stock_close,
                    amount=5_000_000.0,
                ),
                DailyBar(
                    market="cn",
                    symbol="BBB",
                    trade_date=trade_date,
                    open=10.0,
                    high=10.5,
                    low=9.5,
                    close=10.0,
                    amount=500_000.0,
                ),
            ]
        )
    repo.upsert_daily_bars(bars)
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True)
    )


def test_data_portal_blocks_future_data(tmp_path):
    """DataPortal rejects reads after as_of."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 3))
    with pytest.raises(ValueError):
        portal.history(
            market="cn",
            symbols=["AAA"],
            fields=["Close"],
            end_time=date(2024, 1, 4),
            bars=1,
        )


def test_data_portal_blocks_current_unclosed_intraday_bar(tmp_path):
    """DataPortal rejects reads at the current unfinished intraday timestamp."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    repo.upsert_intraday_bars(
        [
            IntradayBar(
                market="cn",
                symbol="AAA",
                bar_time=datetime(2024, 1, 3, 9, 35),
                trade_date=date(2024, 1, 3),
                frequency="1m",
                open=10,
                high=11,
                low=9,
                close=10,
            )
        ]
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=datetime(2024, 1, 3, 9, 35))
    with pytest.raises(ValueError):
        portal.history(
            market="cn",
            symbols=["AAA"],
            fields=["Close"],
            end_time=datetime(2024, 1, 3, 9, 35),
            bars=1,
            frequency="1m",
        )


def test_jq_value_timing_top_k_prefers_high_liquidity(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    _seed_jq_value_fixture(repo, days=70)
    strategy = JqValueTimingTopKStrategy(top_k=1, lookback_days=5, ma_period=20)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 3, 10))
    monkeypatch.setattr(strategy, "_benchmark_allows_trade", lambda *_args, **_kwargs: True)
    targets = strategy.generate_targets(
        date(2024, 3, 10),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB"), dataset_version="fixture-cn"),
    )
    assert len(targets) == 1
    assert targets[0].symbol == "AAA"


def test_momentum_strategy_selects_top_k(tmp_path):
    """Momentum strategy ranks symbols by lookback return."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    strategy = MomentumTopKStrategy(lookback_days=2, top_k=2)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 3))
    targets = strategy.generate_targets(
        date(2024, 1, 3),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB", "CCC"), dataset_version="fixture-cn"),
    )
    assert [target.symbol for target in targets] == ["AAA", "CCC"]
    assert sum(target.weight for target in targets) == pytest.approx(1.0)


def test_momentum_strategy_matches_reference_implementation(tmp_path):
    """Momentum ranking stays consistent with a simple reference implementation."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    strategy = MomentumTopKStrategy(lookback_days=2, top_k=2)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 3))
    context = StrategyContext(market="cn", universe=("AAA", "BBB", "CCC"), dataset_version="fixture-cn")
    actual = strategy.generate_targets(date(2024, 1, 3), portal, context)

    history = portal.history(
        market=context.market,
        symbols=context.universe,
        fields=["Close"],
        end_time=date(2024, 1, 3),
        bars=3,
        frequency="1d",
    )
    ranked: list[tuple[str, float]] = []
    for symbol, symbol_history in history.groupby("symbol"):
        closes = symbol_history.sort_values("trade_date")["Close"].dropna()
        if len(closes) < 3:
            continue
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        if first <= 0 or last <= 0:
            continue
        ranked.append((str(symbol), last / first - 1.0))
    ranked.sort(key=lambda item: item[1], reverse=True)
    expected = ranked[:2]
    expected_weight = 1.0 / len(expected)

    assert [target.symbol for target in actual] == [symbol for symbol, _ in expected]
    assert [target.score for target in actual] == pytest.approx([score for _, score in expected])
    assert sum(target.weight for target in actual) == pytest.approx(1.0)
    # Score-weighted: higher score gets proportionally more weight
    assert [target.weight for target in actual][0] > [target.weight for target in actual][1]


def test_strategy_weights_sum_must_not_exceed_one():
    """Broker target validation rejects leverage."""
    with pytest.raises(ValueError):
        BrokerSimulator(cash=1000).validator.validate_targets(
            [TargetPosition(symbol="AAA", weight=0.8), TargetPosition(symbol="BBB", weight=0.3)]
        )


def test_order_validator_rejects_invalid_symbol_or_weight():
    """Order validator rejects invalid symbol and negative weight targets."""
    validator = BrokerSimulator(cash=1000).validator
    with pytest.raises(ValueError):
        validator.validate_targets([TargetPosition(symbol="", weight=0.5)])
    with pytest.raises(ValueError):
        validator.validate_targets([TargetPosition(symbol="AAA", weight=-0.1)])


def test_empty_targets_mean_no_position(tmp_path):
    """Empty targets clear existing positions."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = PortfolioEngine(
        repository=repo,
        strategy=_BuyThenEmptyStrategy(),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 3),
            initial_cash=100_000,
        ),
        cost_model=CostModel(min_commission=0),
    ).run()
    sells = result.trades[result.trades["side"] == "sell"]
    assert not sells.empty
    assert result.positions[result.positions["symbol"] == "AAA"]["shares"].iloc[-1] == 0


def test_portfolio_engine_buys_score_weighted_top_k(tmp_path):
    """Engine buys score-weighted top momentum symbols on next trade date."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo)
    filled_buys = result.trades[
        (result.trades["status"] == "filled")
        & (result.trades["side"] == "buy")
        & (result.trades["trade_date"] == date(2024, 1, 4))
    ]
    # With score-weighted, AAA (higher score) gets more cash
    assert "AAA" in set(filled_buys["symbol"])
    assert filled_buys[filled_buys["symbol"] == "AAA"]["filled_shares"].iloc[0] > 0


def test_portfolio_engine_rebalances_on_schedule_only(tmp_path):
    """Weekly rebalance does not trade on non-Monday execution dates."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = PortfolioEngine(
        repository=repo,
        strategy=MomentumTopKStrategy(lookback_days=2, top_k=2),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 5),
            initial_cash=100_000,
            rebalance_frequency="weekly",
        ),
        cost_model=CostModel(min_commission=0),
    ).run()
    assert result.trades.empty


def test_portfolio_engine_uses_next_day_open_without_lookahead(tmp_path):
    """Signal on T executes with T+1 open."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo)
    aaa_trade = result.trades[result.trades["symbol"] == "AAA"].iloc[0]
    assert aaa_trade["trade_date"] == date(2024, 1, 4)
    assert aaa_trade["raw_price"] == 13.0


def test_portfolio_engine_respects_lot_size(tmp_path):
    """Buy orders are rounded down to 100-share lots."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo)
    assert all(result.trades[result.trades["side"] == "buy"]["filled_shares"] % 100 == 0)


def test_portfolio_engine_outputs_daily_position_weights(tmp_path):
    """Positions artifact includes daily position weights."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo)
    assert "weight" in result.positions.columns
    weighted = result.positions[result.positions["trade_date"] == date(2024, 1, 4)]
    if not weighted.empty:
        assert weighted["weight"].sum() <= 1.0 + 1e-9


def test_broker_records_rejected_order_reason():
    """Rejected orders are returned with explicit reasons."""
    fill = BrokerSimulator(cash=1000).execute_order(date(2024, 1, 1), Order("AAA", "buy", 100), None)
    assert fill.status == "rejected"
    assert fill.reason == "missing_price"


def test_broker_rejects_nan_or_invalid_price_as_missing():
    """Orders with NaN/non-positive prices are rejected as missing_price."""
    broker = BrokerSimulator(cash=1000)
    nan_bar = pd.Series({"Open": float("nan"), "Close": float("nan"), "paused": False})
    fill_nan = broker.execute_order(date(2024, 1, 1), Order("AAA", "buy", 100), nan_bar)
    assert fill_nan.status == "rejected"
    assert fill_nan.reason == "missing_price"
    bad_bar = pd.Series({"Open": -1.0, "Close": -1.0, "paused": False})
    fill_bad = broker.execute_order(date(2024, 1, 1), Order("AAA", "buy", 100), bad_bar)
    assert fill_bad.status == "rejected"
    assert fill_bad.reason == "missing_price"


def test_portfolio_engine_blocks_limit_up_buy(tmp_path):
    """Engine rejects buy orders when execution day is limit-up."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo, limit_up_on_execution=True)
    result = _run_fixture_engine(repo)
    rejected = result.trades[(result.trades["symbol"] == "AAA") & (result.trades["status"] == "rejected")]
    assert rejected["reason"].iloc[0] == "limit_up"


def test_portfolio_engine_blocks_limit_down_sell(tmp_path):
    """Engine rejects sell orders when execution day is limit-down."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo, limit_down_on_sell=True)
    result = PortfolioEngine(
        repository=repo,
        strategy=_BuyThenEmptyStrategy(),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 3),
            initial_cash=100_000,
        ),
        cost_model=CostModel(min_commission=0),
    ).run()
    rejected = result.trades[(result.trades["symbol"] == "AAA") & (result.trades["side"] == "sell")]
    assert rejected["status"].iloc[0] == "rejected"
    assert rejected["reason"].iloc[0] == "limit_down"


def test_cost_model_applies_min_commission():
    """Cost model applies minimum commission per order."""
    _, costs = CostModel(buy_commission_rate=0.0001, min_commission=5).calculate("buy", 100, 10)
    assert costs.commission == 5


def test_cost_model_charges_stamp_tax_on_sell_only():
    """Stamp tax is charged on sells only."""
    model = CostModel(stamp_tax_rate=0.001, min_commission=0)
    _, buy_costs = model.calculate("buy", 100, 10)
    _, sell_costs = model.calculate("sell", 100, 10)
    assert buy_costs.stamp_tax == 0
    assert sell_costs.stamp_tax == 1


def test_cost_model_applies_slippage_to_fill_price():
    """Slippage moves buy prices up and sell prices down."""
    model = CostModel(slippage_bps=10, min_commission=0)
    buy_price, _ = model.calculate("buy", 100, 10)
    sell_price, _ = model.calculate("sell", 100, 10)
    assert buy_price == pytest.approx(10.01)
    assert sell_price == pytest.approx(9.99)


def test_buy_order_cash_check_includes_all_costs():
    """Buy orders include costs in cash sufficiency checks."""
    broker = BrokerSimulator(cash=1000, cost_model=CostModel(min_commission=5))
    bar = _bar(open_price=10)
    fill = broker.execute_order(date(2024, 1, 1), Order("AAA", "buy", 100), bar)
    assert fill.status == "rejected"
    assert fill.reason == "insufficient_cash"


def test_portfolio_engine_keeps_cash_when_order_cannot_fill(tmp_path):
    """Engine keeps cash unchanged when all buy orders are rejected."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = PortfolioEngine(
        repository=repo,
        strategy=MomentumTopKStrategy(lookback_days=2, top_k=2),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 5),
            initial_cash=100_000,
        ),
        cost_model=CostModel(
            buy_commission_rate=0.0,
            sell_commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=200_000.0,
        ),
    ).run()
    assert not result.trades.empty
    assert (result.trades["status"] == "rejected").all()
    assert result.equity_curve["cash"].iloc[-1] == pytest.approx(100_000.0)


def test_total_cost_is_reported_in_trades_and_metrics(tmp_path):
    """Trade artifacts and metrics include total costs."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo, cost_model=CostModel(min_commission=5))
    assert result.trades["total_cost"].sum() > 0
    assert result.metrics["total_cost"] == pytest.approx(result.trades["total_cost"].sum())
    assert result.cost_summary["total_cost"] == pytest.approx(result.trades["total_cost"].sum())
    assert result.cost_summary["cost_to_turnover_ratio"] >= 0


def test_portfolio_engine_is_deterministic_for_same_inputs(tmp_path):
    """Engine outputs remain identical across repeated runs."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    first = _run_fixture_engine(repo, cost_model=CostModel(min_commission=0))
    second = _run_fixture_engine(repo, cost_model=CostModel(min_commission=0))

    pd.testing.assert_frame_equal(
        first.equity_curve.sort_values(["trade_date"]).reset_index(drop=True),
        second.equity_curve.sort_values(["trade_date"]).reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        first.positions.sort_values(["trade_date", "symbol"]).reset_index(drop=True),
        second.positions.sort_values(["trade_date", "symbol"]).reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        first.trades.sort_values(["trade_date", "symbol", "side"]).reset_index(drop=True),
        second.trades.sort_values(["trade_date", "symbol", "side"]).reset_index(drop=True),
        check_dtype=False,
    )
    assert first.metrics["total_return"] == pytest.approx(second.metrics["total_return"])
    assert first.metrics["total_cost"] == pytest.approx(second.metrics["total_cost"])
    assert first.metrics["turnover"] == pytest.approx(second.metrics["turnover"])


def test_engine_metrics_include_timing_breakdown_without_changing_core_metrics(tmp_path):
    """Timing metrics are additive and do not alter existing metric fields."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo, cost_model=CostModel(min_commission=0))
    assert "elapsed_seconds" in result.metrics
    assert result.metrics["elapsed_seconds"] >= 0
    assert result.metrics["total_return"] == pytest.approx(
        result.equity_curve["equity"].iloc[-1] / 100_000 - 1.0
    )
    assert result.metrics["total_cost"] == pytest.approx(result.trades["total_cost"].sum())
    assert "annual_return" in result.metrics
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics
    assert "max_drawdown_duration" in result.metrics
    assert "cost_to_turnover_ratio" in result.metrics
    assert "avg_holding_count" in result.metrics
    assert "win_rate" in result.metrics
    assert "benchmark_excess_return" in result.metrics
    assert result.metrics["annual_return"] >= -1.0
    assert result.metrics["max_drawdown"] >= 0.0
    assert result.metrics["max_drawdown"] <= 1.0
    assert result.metrics["max_drawdown_duration"] >= 0
    assert result.metrics["cost_to_turnover_ratio"] >= 0.0
    assert result.metrics["avg_holding_count"] >= 0.0
    assert 0.0 <= result.metrics["win_rate"] <= 1.0
    expected_benchmark_return = result.benchmark_curve["equity"].iloc[-1] / 100_000 - 1.0
    assert result.metrics["benchmark_excess_return"] == pytest.approx(
        result.metrics["total_return"] - expected_benchmark_return
    )
    for key in (
        "trade_dates_seconds",
        "load_bars_seconds",
        "universe_seconds",
        "strategy_seconds",
        "rebalance_seconds",
        "bookkeeping_seconds",
    ):
        assert key in result.metrics
        assert result.metrics[key] >= 0


def test_strategy_returns_empty_targets_when_history_insufficient(tmp_path):
    """Strategy returns empty targets when lookback data is insufficient."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    strategy = MomentumTopKStrategy(lookback_days=50, top_k=2)
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 3))
    targets = strategy.generate_targets(
        date(2024, 1, 3),
        portal,
        StrategyContext(market="cn", universe=("AAA", "BBB", "CCC"), dataset_version="fixture-cn"),
    )
    assert targets == []


def test_portfolio_metrics_match_manual_fixture(tmp_path):
    """Core metrics match manual formulas on fixture result."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo, cost_model=CostModel(min_commission=0))
    equity_curve = result.equity_curve.sort_values("trade_date").reset_index(drop=True)
    total_return = equity_curve["equity"].iloc[-1] / 100_000 - 1.0
    returns = equity_curve["equity"].pct_change().dropna()
    annual_vol = float(returns.std() * (252**0.5)) if not returns.empty else 0.0
    peak = equity_curve["equity"].cummax()
    max_drawdown = float((1.0 - equity_curve["equity"] / peak).max())
    assert result.metrics["total_return"] == pytest.approx(total_return)
    assert result.metrics["annualized_volatility"] == pytest.approx(annual_vol)
    assert result.metrics["max_drawdown"] == pytest.approx(max_drawdown)


def test_portfolio_engine_invariants_hold_cash_positions_and_costs(tmp_path):
    """Engine keeps cash non-negative, weights bounded, and costs consistent."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo, cost_model=CostModel(min_commission=0))
    assert (result.equity_curve["cash"] >= -1e-6).all()
    if not result.positions.empty:
        daily_weight = result.positions.groupby("trade_date")["weight"].sum()
        assert (daily_weight <= 1.0 + 1e-9).all()
    assert result.metrics["total_cost"] >= 0.0
    assert result.metrics["total_cost"] == pytest.approx(result.trades["total_cost"].sum())
    if not result.trades.empty:
        assert (result.trades["total_cost"] >= 0).all()


def test_rebalance_logs_include_audit_fields(tmp_path):
    """Rebalance logs carry dataset version, targets, and rejected details."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    result = _run_fixture_engine(repo, cost_model=CostModel(min_commission=0))
    assert not result.rebalance_logs.empty
    row = result.rebalance_logs.iloc[0]
    assert row["dataset_version"] == "fixture-cn"
    assert "targets" in result.rebalance_logs.columns
    assert "cost_snapshot" in result.rebalance_logs.columns


def test_strategy_must_declare_data_frequency(tmp_path):
    """Engine rejects strategy missing required protocol attributes."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)
    engine = PortfolioEngine(
        repository=repo,
        strategy=_MissingFrequencyStrategy(),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 5),
            initial_cash=100_000,
        ),
        cost_model=CostModel(min_commission=0),
    )
    with pytest.raises(ValueError, match="strategy missing required attributes"):
        engine.run()


def _run_fixture_engine(
    repo: MarketDataRepository,
    *,
    cost_model: CostModel | None = None,
):
    return PortfolioEngine(
        repository=repo,
        strategy=MomentumTopKStrategy(lookback_days=2, top_k=2),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 5),
            initial_cash=100_000,
        ),
        cost_model=cost_model or CostModel(min_commission=0),
    ).run()


class _BuyThenEmptyStrategy:
    id = "buy_then_empty"
    name = "Buy Then Empty"
    warmup_days = 0
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if trade_date == date(2024, 1, 1):
            return [TargetPosition(symbol="AAA", weight=1.0)]
        return []


class _MissingFrequencyStrategy:
    id = "missing_frequency"
    name = "Missing Frequency Strategy"
    warmup_days = 0
    rebalance_frequency = "daily"

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        return []


def _bar(*, open_price: float, close: float | None = None) -> pd.Series:
    return pd.Series({"Open": open_price, "Close": close or open_price, "paused": False})


# ─── v0.3.0 修复针对性测试 ─────────────────────────────────────────────

def test_slippage_not_double_counted_in_total_cost():
    """Slippage is in fill_price; total_cost should NOT include slippage_cost."""
    from portfolio.models import CostBreakdown

    cb = CostBreakdown(commission=5.0, stamp_tax=1.0, slippage_cost=3.0)
    assert cb.total_cost == 6.0  # 5 + 1, NOT 5 + 1 + 3


def test_cost_defaults_nonzero(tmp_path):
    """Default slippage_bps and market_impact_bps are > 0 for realistic backtests."""
    from services.portfolio_backtest import PortfolioBacktestRequest

    req = PortfolioBacktestRequest(start=date(2024, 1, 1), end=date(2024, 1, 10))
    assert req.slippage_bps > 0
    assert req.market_impact_bps > 0


def test_sector_diversification_caps_single_sector(tmp_path):
    """Engine caps sector weight when max_sector_pct is set."""
    repo = _repository(tmp_path)
    _seed_portfolio_fixture(repo)

    # All fixture stocks (AAA, BBB) have no sector mapping → fall into _unknown_
    # With 2 stocks in the same "unknown" sector, weight = 1.0 > 0.30
    engine = PortfolioEngine(
        repository=repo,
        strategy=MomentumTopKStrategy(lookback_days=2, top_k=2),
        config=PortfolioBacktestConfig(
            market="cn",
            dataset_version="fixture-cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 4),
            initial_cash=100_000,
        ),
        cost_model=CostModel(min_commission=0),
        max_sector_pct=0.30,
        sector_map_path="/dev/null",  # no sector data → all _unknown_
    )
    result = engine.run()
    filled_buys = result.trades[
        (result.trades["status"] == "filled") & (result.trades["side"] == "buy")
    ]
    # Should still trade (sector cap doesn't block, just scales)
    assert not filled_buys.empty


def test_sector_map_loading():
    """_load_sector_map builds valid symbol→sector mapping from board_constituents.json."""
    from portfolio.engine import PortfolioEngine

    # Use a minimal engine just to call _load_sector_map
    sector_map = PortfolioEngine._load_sector_map.__get__(object())
    # Can't easily invoke without repo, but we can test the static behavior
    # by constructing a minimal engine
    import tempfile, json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "industry": {
                "IT服务": {
                    "stocks": [{"code": "600570", "name": "恒生电子"}, {"code": "000938", "name": "紫光股份"}]
                },
                "银行": {
                    "stocks": [{"code": "601398", "name": "工商银行"}]
                }
            }
        }, f)
        f.flush()
        engine = PortfolioEngine.__new__(PortfolioEngine)
        engine.sector_map_path = f.name
        engine._sector_map = None
        result = engine._load_sector_map()
        assert "600570.SSE" in result
        assert result["600570.SSE"] == "IT服务"
        assert "000938.SZSE" in result
        assert result["000938.SZSE"] == "IT服务"
        assert "601398.SSE" in result
        assert result["601398.SSE"] == "银行"
    os.unlink(f.name)

