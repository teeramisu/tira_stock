"""Tests for full-market data status helpers."""

from __future__ import annotations

from datetime import date

from marketdata.db import connect
from marketdata.models import DailyBar, DatasetVersion, Instrument
from marketdata.repository import MarketDataRepository
from services.market_status import build_market_data_status, latest_stable_dataset_version


def _seed_versions(repo: MarketDataRepository) -> None:
    repo.upsert_instruments(
        [Instrument(market="cn", symbol="AAA", name="A", exchange="SSE", list_date=date(2020, 1, 1))]
    )
    repo.upsert_daily_bars(
        [
            DailyBar(
                market="cn",
                symbol="AAA",
                trade_date=date(2024, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10,
            )
        ]
    )
    repo.upsert_dataset_version(
        DatasetVersion(
            dataset_version="old-cn",
            markets=("cn",),
            max_trade_date=date(2024, 1, 1),
            instrument_count=1,
            daily_bar_count=1,
            is_stable=True,
            quality_status="ok",
        )
    )
    repo.upsert_dataset_version(
        DatasetVersion(
            dataset_version="new-cn",
            markets=("cn",),
            max_trade_date=date(2024, 1, 2),
            instrument_count=1,
            daily_bar_count=1,
            is_stable=True,
            quality_status="ok",
        )
    )


def test_latest_stable_dataset_version_prefers_newest_trade_date(tmp_path):
    """Pick stable version with latest max_trade_date, not only latest created_at."""
    repo = MarketDataRepository(connect(tmp_path / "market.duckdb"))
    _seed_versions(repo)
    assert latest_stable_dataset_version(repo, "cn") == "new-cn"


def test_build_market_data_status_includes_recommended_versions(tmp_path):
    """Status payload exposes per-market latest stable metadata."""
    repo = MarketDataRepository(connect(tmp_path / "market.duckdb"))
    _seed_versions(repo)
    payload = build_market_data_status(repo)
    assert payload["as_of"] == "2024-01-02"
    assert payload["recommended_versions"]["cn"] == "new-cn"
    assert payload["markets"][0]["latest_stable"]["dataset_version"] == "new-cn"
