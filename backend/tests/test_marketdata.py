"""Unit tests for the local market data warehouse."""

from __future__ import annotations

from datetime import date, datetime
import json
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marketdata.data_portal import DataPortal
from marketdata.db import connect
from marketdata.errors import InvalidMarketData, UnstableDatasetVersion
from marketdata.models import (
    DailyBar,
    DatasetVersion,
    Instrument,
    IntradayBar,
    RefreshRun,
    TradingCalendarDay,
    TradingSession,
)
from marketdata.refresh import (
    create_stable_version_from_current,
    refresh_daily_bars_with_retries,
    refresh_intraday_bars_with_retries,
    run_daily_quality_checks,
    run_intraday_quality_checks,
)
from marketdata.repository import MarketDataRepository


def _repository(tmp_path) -> MarketDataRepository:
    conn = connect(tmp_path / "market_data.duckdb")
    return MarketDataRepository(conn)


def _seed_daily_bars(repo: MarketDataRepository) -> None:
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="600000", name="浦发银行", exchange="SSE"),
            Instrument(market="cn", symbol="000001", name="平安银行", exchange="SZSE"),
            Instrument(market="us", symbol="AAPL", name="Apple", exchange="NASDAQ", currency="USD"),
        ]
    )
    repo.upsert_daily_bars(
        [
            DailyBar(
                market="cn",
                symbol="600000",
                trade_date=date(2024, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10.5,
                adj_open=20,
                adj_high=22,
                adj_low=18,
                adj_close=21,
                volume=1000,
            ),
            DailyBar(
                market="cn",
                symbol="600000",
                trade_date=date(2024, 1, 3),
                open=11,
                high=12,
                low=10,
                close=11.5,
                adj_open=22,
                adj_high=24,
                adj_low=20,
                adj_close=23,
                volume=1100,
            ),
            DailyBar(
                market="cn",
                symbol="000001",
                trade_date=date(2024, 1, 3),
                open=8,
                high=9,
                low=7,
                close=8.5,
                adj_open=8,
                adj_high=9,
                adj_low=7,
                adj_close=8.5,
                volume=1200,
            ),
        ]
    )


def test_marketdata_db_initializes_schema(tmp_path):
    """Schema initialization creates required tables and metadata."""
    repo = _repository(tmp_path)
    tables = {
        row[0]
        for row in repo.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    assert "daily_bars" in tables
    assert "dataset_versions" in tables
    schema_version = repo.conn.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert schema_version == "marketdata_v1"


def test_daily_bars_upsert_is_idempotent(tmp_path):
    """Upserting the same daily key overwrites instead of duplicating."""
    repo = _repository(tmp_path)
    bar = DailyBar(
        market="cn",
        symbol="600000",
        trade_date=date(2024, 1, 2),
        open=10,
        high=11,
        low=9,
        close=10.5,
    )
    assert repo.upsert_daily_bars([bar]) == 1
    updated = DailyBar(**{**bar.__dict__, "close": 10.8})
    assert repo.upsert_daily_bars([updated]) == 1
    count, close = repo.conn.execute(
        "SELECT COUNT(*), MAX(close) FROM daily_bars WHERE market='cn' AND symbol='600000'"
    ).fetchone()
    assert count == 1
    assert close == 10.8


def test_read_price_matrix_filters_date_range(tmp_path):
    """Repository reads adjusted daily prices for requested range."""
    repo = _repository(tmp_path)
    _seed_daily_bars(repo)
    df = repo.read_daily_bars(
        market="cn",
        symbols=["600000", "000001"],
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    assert set(df["symbol"]) == {"600000", "000001"}
    assert df[df["symbol"] == "600000"]["Close"].iloc[0] == 23


def test_intraday_bars_upsert_uses_symbol_time_frequency_key(tmp_path):
    """Intraday bars are keyed by market, symbol, bar time, and frequency."""
    repo = _repository(tmp_path)
    bar_time = datetime(2024, 1, 2, 9, 35)
    bar = IntradayBar(
        market="cn",
        symbol="600000",
        bar_time=bar_time,
        trade_date=date(2024, 1, 2),
        frequency="5m",
        open=10,
        high=11,
        low=9,
        close=10.5,
    )
    assert repo.upsert_intraday_bars([bar]) == 1
    assert repo.upsert_intraday_bars([IntradayBar(**{**bar.__dict__, "close": 10.7})]) == 1
    rows = repo.read_intraday_bars(
        market="cn",
        symbols=["600000"],
        start_time=datetime(2024, 1, 2, 9, 30),
        end_time=datetime(2024, 1, 2, 9, 40),
        frequency="5m",
    )
    assert len(rows) == 1
    assert rows["Close"].iloc[0] == 10.7


def test_data_portal_reads_daily_and_intraday_by_frequency(tmp_path):
    """DataPortal exposes daily and intraday history through one interface."""
    repo = _repository(tmp_path)
    _seed_daily_bars(repo)
    repo.upsert_intraday_bars(
        [
            IntradayBar(
                market="cn",
                symbol="600000",
                bar_time=datetime(2024, 1, 3, 9, 35),
                trade_date=date(2024, 1, 3),
                frequency="5m",
                open=11,
                high=12,
                low=10,
                close=11.5,
            )
        ]
    )
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True)
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=datetime(2024, 1, 3, 15, 0))
    daily = portal.history(
        market="cn",
        symbols=["600000"],
        fields=["Close", "Volume"],
        end_time=date(2024, 1, 3),
        bars=1,
        frequency="1d",
    )
    intraday = portal.history(
        market="cn",
        symbols=["600000"],
        fields=["Close"],
        end_time=datetime(2024, 1, 3, 9, 35),
        bars=1,
        frequency="5m",
    )
    assert daily["Close"].iloc[0] == 23
    assert intraday["Close"].iloc[0] == 11.5


def test_data_portal_rejects_non_price_fields_without_pit_modeling(tmp_path):
    """DataPortal blocks fields lacking known/effective-date modeling."""
    repo = _repository(tmp_path)
    _seed_daily_bars(repo)
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn", markets=("cn",), quality_status="ok", is_stable=True)
    )
    portal = DataPortal(repo, dataset_version="fixture-cn", as_of=date(2024, 1, 3))
    with pytest.raises(ValueError):
        portal.history(
            market="cn",
            symbols=["600000"],
            fields=["market_cap"],
            end_time=date(2024, 1, 3),
            bars=1,
            frequency="1d",
        )


def test_repository_intraday_reads_push_down_requested_fields(tmp_path):
    """Repository intraday reads only include requested projected columns."""
    repo = _repository(tmp_path)
    repo.upsert_intraday_bars(
        [
            IntradayBar(
                market="cn",
                symbol="600000",
                bar_time=datetime(2024, 1, 3, 9, 35),
                trade_date=date(2024, 1, 3),
                frequency="5m",
                open=11,
                high=12,
                low=10,
                close=11.5,
                volume=1000,
            )
        ]
    )
    df = repo.read_intraday_bars(
        market="cn",
        symbols=["600000"],
        start_time=datetime(2024, 1, 3, 9, 30),
        end_time=datetime(2024, 1, 3, 9, 40),
        frequency="5m",
        fields=["Close"],
    )
    assert "Close" in df.columns
    assert "Open" not in df.columns
    assert "Volume" not in df.columns


def test_trading_sessions_model_supports_midday_break(tmp_path):
    """Trading sessions can represent A-share morning and afternoon sessions."""
    repo = _repository(tmp_path)
    count = repo.upsert_trading_sessions(
        [
            TradingSession(
                market="cn",
                exchange="SSE",
                session_type="regular_morning",
                start_time="09:30",
                end_time="11:30",
                timezone="Asia/Shanghai",
            ),
            TradingSession(
                market="cn",
                exchange="SSE",
                session_type="regular_afternoon",
                start_time="13:00",
                end_time="15:00",
                timezone="Asia/Shanghai",
            ),
        ]
    )
    assert count == 2
    stored = repo.conn.execute("SELECT COUNT(*) FROM trading_sessions").fetchone()[0]
    assert stored == 2


def test_backtest_reads_only_stable_dataset_version(tmp_path):
    """DataPortal rejects dataset versions not marked as stable."""
    repo = _repository(tmp_path)
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="unstable", markets=("cn",), quality_status="failed", is_stable=False)
    )
    with pytest.raises(UnstableDatasetVersion):
        DataPortal(repo, dataset_version="unstable", as_of=date(2024, 1, 3))


def test_dataset_version_created_after_successful_refresh(tmp_path):
    """Refresh summary creates a stable dataset version from current rows."""
    repo = _repository(tmp_path)
    _seed_daily_bars(repo)
    version = create_stable_version_from_current(
        repo,
        source="fixture",
        markets=("cn", "us"),
        quality_summary={"status": "fixture_ok"},
    )
    loaded = repo.get_dataset_version(version.dataset_version)
    assert loaded.is_stable is True
    assert loaded.daily_bar_count == 3
    assert loaded.instrument_count == 3
    assert loaded.quality_summary["status"] == "fixture_ok"


def test_repository_rejects_duplicate_or_invalid_keys(tmp_path):
    """Repository rejects invalid OHLC rows before persistence."""
    repo = _repository(tmp_path)
    with pytest.raises(InvalidMarketData):
        repo.upsert_daily_bars(
            [
                DailyBar(
                    market="cn",
                    symbol="600000",
                    trade_date=date(2024, 1, 2),
                    open=10,
                    high=9,
                    low=11,
                    close=10,
                )
            ]
        )


def test_refresh_retries_and_checkpoint_resume(tmp_path):
    """Refresh retries transient failures and resumes from checkpoint."""
    repo = _repository(tmp_path)
    checkpoint_path = tmp_path / "cn_refresh_checkpoint.json"
    attempts = {"600000": 0, "000001": 0}

    def loader(symbol: str) -> list[DailyBar]:
        attempts[symbol] += 1
        if symbol == "600000" and attempts[symbol] == 1:
            raise RuntimeError("temporary provider error")
        return [
            DailyBar(
                market="cn",
                symbol=symbol,
                trade_date=date(2024, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10.5,
            )
        ]

    summary = refresh_daily_bars_with_retries(
        repo,
        symbols=["600000", "000001"],
        loader=loader,
        max_attempts=3,
        checkpoint_path=checkpoint_path,
    )
    assert summary.symbols_requested == 2
    assert summary.symbols_success == 2
    assert summary.symbols_failed == 0
    assert summary.bars_written == 2
    assert attempts["600000"] == 2

    calls_after_resume = {"count": 0}

    def loader_after_resume(symbol: str) -> list[DailyBar]:
        calls_after_resume["count"] += 1
        return []

    resumed = refresh_daily_bars_with_retries(
        repo,
        symbols=["600000", "000001"],
        loader=loader_after_resume,
        max_attempts=1,
        checkpoint_path=checkpoint_path,
    )
    assert resumed.symbols_success == 2
    assert resumed.symbols_failed == 0
    assert resumed.bars_written == 0
    assert calls_after_resume["count"] == 0
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert sorted(checkpoint.get("succeeded", [])) == ["000001", "600000"]
    assert checkpoint.get("failed_symbols", {}) == {}


def test_refresh_records_failed_symbols_after_max_attempts(tmp_path):
    """Refresh reports failed symbols after exhausting retries."""
    repo = _repository(tmp_path)

    def always_fail_loader(symbol: str) -> list[DailyBar]:
        raise RuntimeError(f"provider failure: {symbol}")

    summary = refresh_daily_bars_with_retries(
        repo,
        symbols=["600000"],
        loader=always_fail_loader,
        max_attempts=2,
    )
    assert summary.symbols_requested == 1
    assert summary.symbols_success == 0
    assert summary.symbols_failed == 1
    assert summary.bars_written == 0
    assert "600000" in summary.failed_symbols


def test_refresh_clears_failed_symbol_after_later_success(tmp_path):
    """Refresh checkpoint removes a symbol from failed list once recovered."""
    repo = _repository(tmp_path)
    checkpoint_path = tmp_path / "recover_checkpoint.json"
    call_count = {"600000": 0}

    def unstable_loader(symbol: str) -> list[DailyBar]:
        call_count[symbol] += 1
        if call_count[symbol] == 1:
            raise RuntimeError("temporary down")
        return [
            DailyBar(
                market="cn",
                symbol=symbol,
                trade_date=date(2024, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10.5,
            )
        ]

    first = refresh_daily_bars_with_retries(
        repo,
        symbols=["600000"],
        loader=lambda _: (_ for _ in ()).throw(RuntimeError("hard fail")),
        max_attempts=1,
        checkpoint_path=checkpoint_path,
    )
    assert first.symbols_failed == 1
    checkpoint1 = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert "600000" in checkpoint1.get("failed_symbols", {})

    second = refresh_daily_bars_with_retries(
        repo,
        symbols=["600000"],
        loader=unstable_loader,
        max_attempts=2,
        checkpoint_path=checkpoint_path,
    )
    assert second.symbols_failed == 0
    checkpoint2 = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint2.get("failed_symbols", {}) == {}


def test_refresh_run_records_success_and_failure(tmp_path):
    """Repository can persist refresh run lifecycle records."""
    repo = _repository(tmp_path)
    run_id = f"refresh_{uuid.uuid4().hex[:8]}"
    started_at = datetime.utcnow()
    repo.upsert_refresh_run(
        RefreshRun(
            id=run_id,
            dataset_version="pending",
            job_type="cn_incremental_daily",
            started_at=started_at,
            status="running",
            symbols_total=2,
        )
    )
    running = repo.conn.execute("SELECT status, symbols_total FROM refresh_runs WHERE id = ?", [run_id]).fetchone()
    assert running == ("running", 2)

    repo.upsert_refresh_run(
        RefreshRun(
            id=run_id,
            dataset_version="ds_cn_test",
            job_type="cn_incremental_daily",
            started_at=started_at,
            finished_at=datetime.utcnow(),
            status="partial_success",
            symbols_total=2,
            symbols_success=1,
            symbols_failed=1,
            error="{'000001': 'timeout'}",
        )
    )
    finished = repo.conn.execute(
        "SELECT dataset_version, status, symbols_success, symbols_failed FROM refresh_runs WHERE id = ?",
        [run_id],
    ).fetchone()
    assert finished == ("ds_cn_test", "partial_success", 1, 1)


def test_quality_check_rejects_invalid_ohlc(tmp_path):
    """Quality checks reject rows with invalid OHLC bounds."""
    repo = _repository(tmp_path)
    repo.upsert_instruments([Instrument(market="cn", symbol="600000", name="浦发银行", exchange="SSE")])
    repo.conn.execute(
        """
        INSERT INTO daily_bars (
            market, symbol, trade_date, open, high, low, close,
            adjustment, paused, source, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'qfq', false, 'fixture', now())
        """,
        ["cn", "600000", date(2024, 1, 2), 10.0, 9.0, 8.0, 10.0],
    )
    with pytest.raises(InvalidMarketData):
        run_daily_quality_checks(
            repo,
            market="cn",
            start=date(2024, 1, 1),
            end=date(2024, 1, 3),
            expected_symbols=1,
        )


def test_refresh_continues_after_one_symbol_failure(tmp_path):
    """Refresh continues writing successful symbols after one symbol failure."""
    repo = _repository(tmp_path)

    def loader(symbol: str) -> list[DailyBar]:
        if symbol == "BAD":
            raise RuntimeError("provider down")
        return [
            DailyBar(
                market="cn",
                symbol=symbol,
                trade_date=date(2024, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10.5,
            )
        ]

    summary = refresh_daily_bars_with_retries(
        repo,
        symbols=["GOOD", "BAD"],
        loader=loader,
        max_attempts=1,
    )
    stored = repo.conn.execute("SELECT COUNT(*) FROM daily_bars WHERE market = 'cn'").fetchone()[0]
    assert summary.symbols_success == 1
    assert summary.symbols_failed == 1
    assert stored == 1


def test_refresh_skips_symbols_already_up_to_date(tmp_path):
    """Refresh skips symbols that are already marked successful in checkpoint."""
    repo = _repository(tmp_path)
    checkpoint_path = tmp_path / "skip_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps({"succeeded": ["600000"], "failed_symbols": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    calls = {"count": 0}

    def loader(symbol: str) -> list[DailyBar]:
        calls["count"] += 1
        return [
            DailyBar(
                market="cn",
                symbol=symbol,
                trade_date=date(2024, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10.5,
            )
        ]

    summary = refresh_daily_bars_with_retries(
        repo,
        symbols=["600000", "000001"],
        loader=loader,
        max_attempts=1,
        checkpoint_path=checkpoint_path,
    )
    assert summary.symbols_requested == 2
    assert summary.symbols_success == 2
    assert calls["count"] == 1


def test_quality_check_reports_coverage(tmp_path):
    """Quality checks report low latest-date symbol coverage."""
    repo = _repository(tmp_path)
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE"),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SZSE"),
        ]
    )
    repo.upsert_daily_bars(
        [
            DailyBar(market="cn", symbol="AAA", trade_date=date(2024, 1, 2), open=10, high=11, low=9, close=10),
            DailyBar(market="cn", symbol="BBB", trade_date=date(2024, 1, 2), open=10, high=11, low=9, close=10),
            DailyBar(market="cn", symbol="AAA", trade_date=date(2024, 1, 3), open=10, high=11, low=9, close=10),
        ]
    )
    report = run_daily_quality_checks(
        repo,
        market="cn",
        start=date(2024, 1, 1),
        end=date(2024, 1, 3),
        expected_symbols=2,
        coverage_threshold=0.9,
    )
    assert report["latest_trade_date"] == "2024-01-03"
    assert report["covered_symbols"] == 1
    assert report["coverage_ratio"] == pytest.approx(0.5)
    assert report["coverage_warning"] is True


def test_daily_quality_check_allows_missing_volume_with_warning(tmp_path):
    """Missing volume rows are allowed but counted and flagged in the summary."""
    repo = _repository(tmp_path)
    repo.upsert_instruments([Instrument(market="cn", symbol="600000", name="浦发银行", exchange="SSE")])
    repo.conn.execute(
        """
        INSERT INTO daily_bars (
            market, symbol, trade_date, open, high, low, close,
            adjustment, paused, source, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'qfq', false, 'fixture', now())
        """,
        ["cn", "600000", date(2024, 1, 2), 10.0, 11.0, 9.0, 10.5],
    )
    report = run_daily_quality_checks(
        repo,
        market="cn",
        start=date(2024, 1, 1),
        end=date(2024, 1, 3),
        expected_symbols=1,
    )
    assert report["missing_volume"] == 1
    assert report["missing_volume_allowed"] is True
    assert report["missing_volume_recorded"] is True
    assert report["missing_volume_warning"] is True


def test_upsert_trading_calendar_is_idempotent(tmp_path):
    """Trading calendar upserts replace the same (market, trade_date) key."""
    repo = _repository(tmp_path)
    days = [
        TradingCalendarDay(market="cn", trade_date=date(2024, 1, 2), is_open=True),
        TradingCalendarDay(market="cn", trade_date=date(2024, 1, 3), is_open=False),
    ]
    assert repo.upsert_trading_calendar(days) == 2
    assert repo.upsert_trading_calendar(days) == 2
    n = repo.conn.execute("SELECT COUNT(*) FROM trading_calendar WHERE market = 'cn'").fetchone()[0]
    assert n == 2


def test_is_cn_etf_symbol_includes_star_board_etf():
    from marketdata.sources import akshare_source as mod

    assert mod.is_cn_etf_symbol("588000")
    assert mod.is_cn_etf_symbol("510300")


def test_load_cn_etf_daily_bars_falls_back_to_sina(monkeypatch):
    """When Eastmoney ETF history fails, use Sina ETF history."""
    from datetime import date

    import pandas as pd

    from marketdata.sources import akshare_source as mod

    def fail_em(*args, **kwargs):
        raise RuntimeError("eastmoney blocked")

    def fake_sina(symbol: str = "sh510300"):
        return pd.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 3)],
                "open": [3.0, 3.1],
                "high": [3.2, 3.3],
                "low": [2.9, 3.0],
                "close": [3.1, 3.2],
                "volume": [1000, 1100],
            }
        )

    monkeypatch.setattr(mod, "_call_with_timeout", lambda fetcher, **kwargs: fetcher())
    import akshare as ak

    monkeypatch.setattr(ak, "fund_etf_hist_em", fail_em)
    monkeypatch.setattr(ak, "fund_etf_hist_sina", fake_sina)
    bars = mod.load_cn_etf_daily_bars_for_symbol("510300", date(2024, 1, 1), date(2024, 1, 31))
    assert len(bars) == 2
    assert bars[0].source == "akshare_etf_sina"


def test_load_cn_instruments_for_symbols_prefers_etf_metadata(monkeypatch):
    """Explicit ETF symbol refresh should not require full A-share instrument scan."""
    from marketdata.sources import akshare_source as mod

    def fake_etf_instruments():
        return [
            Instrument(
                market="cn",
                symbol="510300",
                name="沪深300ETF",
                exchange="SSE",
                asset_type="etf",
                source="akshare_etf",
            )
        ]

    def fail_full_stock_load(*args, **kwargs):
        raise AssertionError("should not load full stock universe for ETF-only symbols")

    monkeypatch.setattr(mod, "load_cn_etf_instruments", fake_etf_instruments)
    monkeypatch.setattr(mod, "load_cn_instruments", fail_full_stock_load)
    rows = mod.load_cn_instruments_for_symbols(["510300", "159915"])
    assert [item.symbol for item in rows] == ["510300", "159915"]
    assert rows[0].asset_type == "etf"
    assert rows[0].name == "沪深300ETF"
    assert rows[1].asset_type == "etf"


def test_load_cn_trading_calendar_filters_date_range(monkeypatch):
    """AkShare calendar loader keeps only rows inside the requested inclusive range."""
    from marketdata.sources import akshare_source as mod

    def fake_call(_fetcher, **kwargs):
        import pandas as pd

        return pd.DataFrame({"trade_date": ["2023-12-30", "2024-01-02", "2024-01-15", "2025-01-01"]})

    monkeypatch.setattr(mod, "_call_with_timeout", fake_call)
    rows = mod.load_cn_trading_calendar(date(2024, 1, 1), date(2024, 1, 31))
    assert [r.trade_date for r in rows] == [date(2024, 1, 2), date(2024, 1, 15)]
    assert all(r.is_open for r in rows)


def test_quality_check_rejects_intraday_bar_outside_session(tmp_path):
    """Intraday quality checks reject bars outside configured sessions."""
    repo = _repository(tmp_path)
    repo.upsert_instruments([Instrument(market="cn", symbol="600000", name="浦发银行", exchange="SSE")])
    repo.upsert_trading_sessions(
        [
            TradingSession(
                market="cn",
                exchange="SSE",
                session_type="regular",
                start_time="09:30",
                end_time="11:30",
                timezone="Asia/Shanghai",
            )
        ]
    )
    repo.upsert_intraday_bars(
        [
            IntradayBar(
                market="cn",
                symbol="600000",
                bar_time=datetime(2024, 1, 3, 12, 30),
                trade_date=date(2024, 1, 3),
                frequency="1m",
                open=10,
                high=10.2,
                low=9.8,
                close=10.1,
            )
        ]
    )
    with pytest.raises(InvalidMarketData):
        run_intraday_quality_checks(
            repo,
            market="cn",
            start=datetime(2024, 1, 3, 0, 0),
            end=datetime(2024, 1, 4, 0, 0),
            frequency="1m",
            expected_symbols=1,
        )


def test_refresh_intraday_retries_with_checkpoint(tmp_path):
    """Intraday refresh retries once and persists checkpoint success."""
    repo = _repository(tmp_path)
    checkpoint_path = tmp_path / "intraday_checkpoint.json"
    attempts = {"AAPL": 0}

    def loader(symbol: str) -> list[IntradayBar]:
        attempts[symbol] += 1
        if attempts[symbol] == 1:
            raise RuntimeError("transient intraday error")
        return [
            IntradayBar(
                market="us",
                symbol=symbol,
                bar_time=datetime(2024, 1, 3, 9, 31),
                trade_date=date(2024, 1, 3),
                frequency="1m",
                open=100,
                high=101,
                low=99,
                close=100.5,
            )
        ]

    summary = refresh_intraday_bars_with_retries(
        repo,
        symbols=["AAPL"],
        loader=loader,
        max_attempts=2,
        checkpoint_path=checkpoint_path,
    )
    assert summary.symbols_success == 1
    assert summary.symbols_failed == 0
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["succeeded"] == ["AAPL"]

