"""Tests for per-symbol attribution artifacts."""

from __future__ import annotations

from datetime import date
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marketdata.db import connect
from marketdata.models import Instrument
from marketdata.repository import MarketDataRepository
from research.symbol_attribution import (
    build_symbol_attribution_artifact,
    build_trade_cycles,
)


def _repository(tmp_path) -> MarketDataRepository:
    return MarketDataRepository(connect(tmp_path / "market_data.duckdb"))


def _filled_trade(
    *,
    symbol: str,
    trade_date: date,
    side: str,
    shares: int,
    price: float,
    cost: float = 0.0,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "side": side,
        "filled_shares": shares,
        "fill_price": price,
        "total_cost": cost,
        "status": "filled",
    }


def test_build_trade_cycles_single_round_trip():
    trades = pd.DataFrame(
        [
            _filled_trade(symbol="AAA", trade_date=date(2024, 1, 2), side="buy", shares=100, price=10.0, cost=1.0),
            _filled_trade(symbol="AAA", trade_date=date(2024, 1, 4), side="sell", shares=100, price=12.0, cost=1.0),
        ]
    )
    cycles = build_trade_cycles(trades)["AAA"]
    assert len(cycles) == 1
    assert cycles[0]["status"] == "closed"
    assert cycles[0]["entry_date"] == "2024-01-02"
    assert cycles[0]["exit_date"] == "2024-01-04"
    assert len(cycles[0]["events"]) == 2
    assert cycles[0]["realized_pnl"] == pytest.approx(198.0)


def test_build_trade_cycles_open_position_at_end():
    trades = pd.DataFrame(
        [
            _filled_trade(symbol="BBB", trade_date=date(2024, 1, 2), side="buy", shares=50, price=20.0),
            _filled_trade(symbol="BBB", trade_date=date(2024, 1, 3), side="buy", shares=50, price=21.0),
        ]
    )
    cycles = build_trade_cycles(trades)["BBB"]
    assert len(cycles) == 1
    assert cycles[0]["status"] == "open"
    assert len(cycles[0]["events"]) == 2


def test_build_trade_cycles_multiple_round_trips():
    trades = pd.DataFrame(
        [
            _filled_trade(symbol="CCC", trade_date=date(2024, 1, 2), side="buy", shares=10, price=10.0),
            _filled_trade(symbol="CCC", trade_date=date(2024, 1, 3), side="sell", shares=10, price=11.0),
            _filled_trade(symbol="CCC", trade_date=date(2024, 1, 4), side="buy", shares=20, price=9.0),
            _filled_trade(symbol="CCC", trade_date=date(2024, 1, 5), side="sell", shares=20, price=10.0),
        ]
    )
    cycles = build_trade_cycles(trades)["CCC"]
    assert len(cycles) == 2
    assert cycles[0]["status"] == "closed"
    assert cycles[1]["status"] == "closed"


def test_build_symbol_attribution_ranks_profit_and_drawdown(tmp_path):
    repo = _repository(tmp_path)
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE"),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SZSE"),
        ]
    )
    trades = pd.DataFrame(
        [
            _filled_trade(symbol="AAA", trade_date=date(2024, 1, 2), side="buy", shares=100, price=10.0),
            _filled_trade(symbol="AAA", trade_date=date(2024, 1, 4), side="sell", shares=100, price=13.0),
            _filled_trade(symbol="BBB", trade_date=date(2024, 1, 2), side="buy", shares=100, price=10.0),
            _filled_trade(symbol="BBB", trade_date=date(2024, 1, 4), side="sell", shares=100, price=8.0),
        ]
    )
    positions = pd.DataFrame(
        [
            {"trade_date": date(2024, 1, 2), "symbol": "AAA", "shares": 100, "market_value": 1000.0, "weight": 0.5},
            {"trade_date": date(2024, 1, 3), "symbol": "AAA", "shares": 100, "market_value": 1200.0, "weight": 0.55},
            {"trade_date": date(2024, 1, 4), "symbol": "AAA", "shares": 0, "market_value": 0.0, "weight": 0.0},
            {"trade_date": date(2024, 1, 2), "symbol": "BBB", "shares": 100, "market_value": 1000.0, "weight": 0.5},
            {"trade_date": date(2024, 1, 3), "symbol": "BBB", "shares": 100, "market_value": 900.0, "weight": 0.45},
            {"trade_date": date(2024, 1, 4), "symbol": "BBB", "shares": 0, "market_value": 0.0, "weight": 0.0},
        ]
    )
    equity_curve = pd.DataFrame(
        [
            {"trade_date": date(2024, 1, 1), "equity": 10000.0},
            {"trade_date": date(2024, 1, 2), "equity": 10000.0},
            {"trade_date": date(2024, 1, 3), "equity": 11000.0},
            {"trade_date": date(2024, 1, 4), "equity": 10200.0},
        ]
    )
    artifact = build_symbol_attribution_artifact(
        trades=trades,
        positions=positions,
        equity_curve=equity_curve,
        repository=repo,
        market="cn",
        sample_start="2024-01-01",
        sample_end="2024-01-04",
    )
    assert artifact["artifact_type"] == "symbol_attribution_v1"
    assert artifact["top_profit_contributors"][0]["symbol"] == "AAA"
    assert artifact["top_profit_contributors"][0]["name"] == "Alpha"
    assert artifact["top_drawdown_contributors"][0]["drawdown_contribution"] < 0
    assert artifact["top_drawdown_contributors"][0]["drawdown_contribution"] < 0
    assert artifact["top_drawdown_contributors"][0]["cycles"]
