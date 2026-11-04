"""Tests for OHLCV fetch with local market DB fallback."""

from __future__ import annotations

from datetime import date

from marketdata.models import DailyBar
from marketdata.repository import MarketDataRepository
from services.data import fetch_ohlcv, to_yfinance_ticker


def _seed_aapl(repo: MarketDataRepository) -> None:
    bars = [
        DailyBar(
            market="us",
            symbol="AAPL",
            trade_date=date(2024, 1, 2),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000_000.0,
        ),
        DailyBar(
            market="us",
            symbol="AAPL",
            trade_date=date(2024, 1, 3),
            open=100.5,
            high=102.0,
            low=100.0,
            close=101.0,
            volume=1_100_000.0,
        ),
    ]
    repo.upsert_daily_bars(bars)


def test_fetch_ohlcv_prefers_local_db_for_us(tmp_path, monkeypatch):
    """US single-symbol backtests should read local DuckDB before Yahoo."""
    from marketdata.db import connect

    db_path = tmp_path / "market.duckdb"
    repo = MarketDataRepository(connect(db_path))
    _seed_aapl(repo)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))

    df = fetch_ohlcv("AAPL", start="2024-01-02", end="2024-01-03", market="us")

    assert len(df) == 2
    assert float(df.iloc[-1]["Close"]) == 101.0


def test_to_yfinance_ticker_accepts_cn_etf_codes():
    assert to_yfinance_ticker("510300", "cn") == "510300.SS"
    assert to_yfinance_ticker("159915", "cn") == "159915.SZ"
    assert to_yfinance_ticker("588000", "cn") == "588000.SS"


def test_fetch_ohlcv_prefers_local_db_for_cn_etf(tmp_path, monkeypatch):
    """CN ETF single-symbol backtests should read local DuckDB without stock-only validation."""
    from marketdata.db import connect

    db_path = tmp_path / "market.duckdb"
    repo = MarketDataRepository(connect(db_path))
    repo.upsert_daily_bars(
        [
            DailyBar(
                market="cn",
                symbol="510300",
                trade_date=date(2024, 1, 2),
                open=3.5,
                high=3.6,
                low=3.4,
                close=3.55,
                volume=1_000_000.0,
            )
        ]
    )
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))

    df = fetch_ohlcv("510300", start="2024-01-02", end="2024-01-02", market="cn")

    assert len(df) == 1
    assert float(df.iloc[-1]["Close"]) == 3.55
