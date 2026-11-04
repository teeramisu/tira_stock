"""Tests for US option chain loading via yfinance adapter."""

from __future__ import annotations

from datetime import date
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marketdata.models import OptionChainSnapshot
from marketdata.sources import yfinance_source as yfs


class _FakeChain:
    calls = pd.DataFrame(
        [
            {
                "contractSymbol": "TEST240119C00001000",
                "strike": 100.0,
                "lastPrice": 2.5,
                "bid": 2.4,
                "ask": 2.6,
                "volume": 100.0,
                "openInterest": 500.0,
                "impliedVolatility": 0.35,
                "inTheMoney": True,
            }
        ]
    )
    puts = pd.DataFrame(
        [
            {
                "contractSymbol": "TEST240119P00001000",
                "strike": 100.0,
                "lastPrice": 1.2,
                "bid": 1.1,
                "ask": 1.25,
                "volume": 50.0,
                "openInterest": 200.0,
                "impliedVolatility": 0.33,
                "inTheMoney": False,
            }
        ]
    )


class _FakeTicker:
    options = ("2024-01-19",)

    def option_chain(self, _exp: str) -> _FakeChain:
        return _FakeChain()


def test_load_us_option_chain_maps_calls_and_puts(monkeypatch):
    """Loader flattens Yahoo-style calls/puts frames into snapshot rows."""

    def _fake_ticker(_symbol: str) -> _FakeTicker:
        return _FakeTicker()

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _fake_ticker)

    rows = yfs.load_us_option_chain_snapshots(
        "TEST",
        snapshot_date=date(2024, 1, 18),
        max_expirations=1,
        timeout_seconds=5.0,
    )
    assert len(rows) == 2
    calls = [r for r in rows if r.contract_right == "call"]
    puts = [r for r in rows if r.contract_right == "put"]
    assert len(calls) == 1 and len(puts) == 1
    assert calls[0].underlying == "TEST"
    assert calls[0].expiry == date(2024, 1, 19)
    assert calls[0].strike == 100.0
    assert puts[0].contract_symbol.endswith("P00001000")


def test_load_us_option_chain_empty_expirations(monkeypatch):
    """When no expirations are listed, return an empty list."""

    class _EmptyTicker:
        options = ()

        def option_chain(self, _exp: str) -> _FakeChain:
            raise AssertionError("should not be called")

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda _s: _EmptyTicker())

    assert yfs.load_us_option_chain_snapshots("ZZZ", timeout_seconds=5.0) == []


def test_repository_option_chain_roundtrip(tmp_path):
    """Option snapshots persist through DuckDB upsert."""
    from marketdata.db import connect
    from marketdata.repository import MarketDataRepository

    repo = MarketDataRepository(connect(tmp_path / "m.duckdb"))
    rows = [
        OptionChainSnapshot(
            market="us",
            snapshot_date=date(2024, 1, 18),
            underlying="TEST",
            expiry=date(2024, 1, 19),
            strike=100.0,
            contract_right="call",
            contract_symbol="TEST240119C00001000",
            last_price=1.0,
            source="fixture",
        )
    ]
    assert repo.upsert_option_chain_snapshots(rows) == 1
    n = repo.conn.execute("SELECT COUNT(*) FROM option_chain_snapshots").fetchone()[0]
    assert n == 1
