"""Unit tests for market universe filtering."""

from __future__ import annotations

from datetime import date
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marketdata.db import connect
from marketdata.models import DailyBar, Instrument
from marketdata.repository import MarketDataRepository
from marketdata.universe import UniverseConfig, get_universe, us_universe_config


def _repository(tmp_path) -> MarketDataRepository:
    return MarketDataRepository(connect(tmp_path / "market_data.duckdb"))


def _seed_universe_fixture(repo: MarketDataRepository) -> None:
    repo.upsert_instruments(
        [
            Instrument(
                market="cn",
                symbol="600000",
                name="浦发银行",
                exchange="SSE",
                list_date=date(2000, 1, 1),
            ),
            Instrument(
                market="cn",
                symbol="430001",
                name="北交所样例",
                exchange="BSE",
                list_date=date(2018, 1, 1),
            ),
            Instrument(
                market="cn",
                symbol="400001",
                name="新三板样例",
                exchange="NEEQ",
                list_date=date(2018, 1, 1),
            ),
            Instrument(
                market="cn",
                symbol="000001",
                name="ST 平安",
                exchange="SZSE",
                list_date=date(2000, 1, 1),
                is_st=True,
            ),
            Instrument(
                market="cn",
                symbol="300001",
                name="新股样例",
                exchange="SZSE",
                list_date=date(2024, 1, 15),
            ),
            Instrument(
                market="cn",
                symbol="600001",
                name="停牌样例",
                exchange="SSE",
                list_date=date(2000, 1, 1),
            ),
            Instrument(
                market="cn",
                symbol="600002",
                name="涨停样例",
                exchange="SSE",
                list_date=date(2000, 1, 1),
            ),
            Instrument(
                market="us",
                symbol="AAPL",
                name="Apple",
                exchange="NASDAQ",
                currency="USD",
                list_date=date(1980, 12, 12),
            ),
            Instrument(
                market="us",
                symbol="IBM",
                name="IBM",
                exchange="NYSE",
                currency="USD",
                list_date=date(1911, 1, 1),
            ),
        ]
    )
    repo.upsert_daily_bars(
        [
            _daily_bar("600000", paused=False),
            _daily_bar("430001", paused=False),
            _daily_bar("000001", paused=False),
            _daily_bar("300001", paused=False),
            _daily_bar("600001", paused=True),
            _daily_bar("600002", paused=False, close=10, limit_up=10),
        ]
    )


def _daily_bar(
    symbol: str,
    *,
    paused: bool,
    close: float = 10,
    limit_up: float | None = None,
) -> DailyBar:
    return DailyBar(
        market="cn",
        symbol=symbol,
        trade_date=date(2024, 2, 1),
        open=10,
        high=11,
        low=9,
        close=close,
        paused=paused,
        limit_up=limit_up,
    )


def test_universe_excludes_neeq(tmp_path):
    """Default CN universe excludes NEEQ symbols."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1))
    assert "400001" not in result.symbols


def test_cn_universe_includes_bse_by_default(tmp_path):
    """Default CN universe includes BSE symbols."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1))
    assert "430001" in result.symbols


def test_us_universe_includes_unknown_exchange_metadata(tmp_path):
    """Default US universe keeps legacy UNKNOWN exchange metadata."""
    repo = _repository(tmp_path)
    repo.upsert_instruments(
        [
            Instrument(
                market="us",
                symbol="MSFT",
                name="Microsoft",
                exchange="UNKNOWN",
                currency="USD",
            )
        ]
    )
    repo.upsert_daily_bars(
        [
            DailyBar(
                market="us",
                symbol="MSFT",
                trade_date=date(2024, 2, 1),
                open=400,
                high=410,
                low=390,
                close=405,
            )
        ]
    )
    result = get_universe(repo, date(2024, 2, 1), us_universe_config())
    assert "MSFT" in result.symbols


def test_us_universe_uses_us_exchanges_without_cn_filters(tmp_path):
    """US universe uses US exchanges and does not apply CN listing filter."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1), us_universe_config())
    assert result.symbols == ["AAPL", "IBM"]


def test_universe_excludes_st_by_default(tmp_path):
    """Default CN universe excludes ST instruments."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1))
    assert "000001" not in result.symbols


def test_universe_respects_min_list_days(tmp_path):
    """Default CN universe excludes recent listings."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1), UniverseConfig(min_list_days=60))
    assert "300001" not in result.symbols


def test_universe_excludes_suspended_on_trade_date(tmp_path):
    """Default CN universe excludes symbols paused on the trade date."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1))
    assert "600001" not in result.symbols


def test_universe_returns_filter_diagnostics(tmp_path):
    """Universe result includes counts after each filter."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1))
    assert result.diagnostics["initial"] == 7
    assert result.diagnostics["after_st"] == 5
    assert result.diagnostics["final"] == 2


def test_universe_respects_custom_symbols(tmp_path):
    """Universe can start from a manually provided symbol list."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1), UniverseConfig(symbols=("430001",)))
    assert result.symbols == ["430001"]


def test_universe_excludes_limit_up_buy_candidates(tmp_path):
    """Default CN universe excludes symbols already at limit up."""
    repo = _repository(tmp_path)
    _seed_universe_fixture(repo)
    result = get_universe(repo, date(2024, 2, 1))
    assert "600002" not in result.symbols


def test_universe_reports_missing_optional_fields(tmp_path):
    """Universe diagnostics expose missing optional data counts."""
    repo = _repository(tmp_path)
    repo.upsert_instruments(
        [
            Instrument(
                market="cn",
                symbol="600003",
                name="缺字段样例",
                exchange="SSE",
            )
        ]
    )
    result = get_universe(repo, date(2024, 2, 1))
    assert result.diagnostics["missing_list_date"] == 1
    assert result.diagnostics["missing_trade_bar"] == 1

