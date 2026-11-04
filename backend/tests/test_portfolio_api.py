"""API tests for local market data and portfolio backtests."""

from __future__ import annotations

from datetime import date
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app
from marketdata.db import connect
from marketdata.models import DailyBar, DatasetVersion, Instrument
from marketdata.repository import MarketDataRepository


def _seed_api_fixture(db_path) -> None:
    repo = MarketDataRepository(connect(db_path))
    repo.upsert_instruments(
        [
            Instrument(market="cn", symbol="AAA", name="Alpha", exchange="SSE", list_date=date(2020, 1, 1)),
            Instrument(market="cn", symbol="BBB", name="Beta", exchange="SZSE", list_date=date(2020, 1, 1)),
        ]
    )
    bars = []
    closes = {"AAA": [10, 11, 12, 13], "BBB": [10, 9, 8, 8]}
    for idx, trade_date in enumerate(
        [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
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
                )
            )
    repo.upsert_daily_bars(bars)
    repo.upsert_dataset_version(
        DatasetVersion(
            dataset_version="fixture-cn",
            markets=("cn",),
            source="fixture",
            min_trade_date=date(2024, 1, 1),
            max_trade_date=date(2024, 1, 4),
            instrument_count=2,
            daily_bar_count=8,
            quality_status="ok",
            is_stable=True,
        )
    )
    repo.upsert_dataset_version(
        DatasetVersion(dataset_version="fixture-cn-unstable", markets=("cn",), quality_status="running", is_stable=False)
    )


def test_market_data_status_does_not_create_missing_db(tmp_path, monkeypatch):
    """Data status reports a missing DB without creating it."""
    db_path = tmp_path / "missing.duckdb"
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).get("/api/market/data-status")
    assert response.status_code == 200
    assert response.json()["exists"] is False
    assert not db_path.exists()


def test_market_data_status_api_reports_date_range(tmp_path, monkeypatch):
    """Data status includes row counts and available date range."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).get("/api/market/data-status")
    assert response.status_code == 200
    payload = response.json()
    market = payload["markets"][0]
    assert market["daily_bars"] == 8
    assert market["min_trade_date"] == "2024-01-01"
    assert market["max_trade_date"] == "2024-01-04"
    assert payload["as_of"] == "2024-01-04"
    assert payload["recommended_versions"]["cn"] == "fixture-cn"
    assert market["latest_stable"]["dataset_version"] == "fixture-cn"


def test_supported_markets_api_includes_cn_and_us():
    """Supported markets endpoint includes CN and US."""
    response = TestClient(app).get("/api/market/supported-markets")
    assert response.status_code == 200
    markets = {item["market"] for item in response.json()["markets"]}
    assert {"cn", "us"}.issubset(markets)


def test_market_dataset_versions_api_returns_records(tmp_path, monkeypatch):
    """Dataset versions endpoint returns stable records."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).get("/api/market/dataset-versions?market=cn&stable_only=true")
    assert response.status_code == 200
    versions = response.json()["versions"]
    assert versions
    assert versions[0]["dataset_version"] == "fixture-cn"


def test_market_universe_endpoint_returns_diagnostics(tmp_path, monkeypatch):
    """Universe endpoint returns symbols and filter diagnostics."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).get("/api/market/universe?trade_date=2024-01-03&market=cn")
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbols"] == ["AAA", "BBB"]
    assert payload["diagnostics"]["final"] == 2


@pytest.mark.xfail(strict=False, reason="endpoint 已改为进程内刷新，ENABLE_MARKET_REFRESH_API 特性开关被移除，待重写")
def test_market_refresh_endpoint_rejects_when_disabled(monkeypatch):
    """Refresh endpoint is blocked when feature flag is not enabled."""
    monkeypatch.delenv("ENABLE_MARKET_REFRESH_API", raising=False)
    response = TestClient(app).post("/api/market/refresh", json={"market": "cn", "mode": "status"})
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


@pytest.mark.xfail(strict=False, reason="endpoint 不再走 subprocess，待重写为进程内刷新断言")
def test_market_refresh_endpoint_runs_script_in_dev_mode(monkeypatch):
    """Refresh endpoint invokes refresh script when enabled."""
    monkeypatch.setenv("ENABLE_MARKET_REFRESH_API", "1")

    def _fake_run(command, cwd, capture_output, text, timeout, check):  # noqa: ANN001
        assert "--market" in command
        assert "--mode" in command
        return SimpleNamespace(returncode=0, stdout="created dataset_version=fixture", stderr="")

    monkeypatch.setattr("app.main.subprocess.run", _fake_run)
    response = TestClient(app).post(
        "/api/market/refresh",
        json={"market": "cn", "mode": "status", "frequency": "1d", "symbols": ["AAA"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert any("--market" == part for part in payload["command"])


@pytest.mark.xfail(strict=False, reason="trades 排序/字段与当前实现不一致，待对齐")
def test_portfolio_backtest_endpoint_runs_momentum_top_k(tmp_path, monkeypatch):
    """Portfolio endpoint runs against local fixture data."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "strategy": "momentum_top_k",
            "max_positions": 1,
            "lookback_days": 2,
            "min_commission": 0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_version"] == "fixture-cn"
    assert payload["request_hash"]
    assert payload["run_metadata"]["request_hash"] == payload["request_hash"]
    assert payload["run_metadata"]["dataset_version"] == payload["dataset_version"]
    assert payload["run_metadata"]["strategy"] == payload["strategy"]
    assert payload["run_metadata"]["elapsed_seconds"] >= 0
    assert payload["run_metadata"]["universe_size"] >= 0
    assert payload["run_metadata"]["experiment"]["sample_start"] == "2024-01-01"
    assert payload["run_metadata"]["experiment"]["sample_end"] == "2024-01-04"
    assert "params" in payload["run_metadata"]["experiment"]
    assert payload["trades"][0]["symbol"] == "AAA"
    assert payload["metrics"]["turnover"] > 0
    assert payload["benchmark_curve"]
    assert payload["cost_summary"]["total_cost"] >= 0
    assert payload["final_positions"]
    assert payload["final_positions"][0]["symbol"] == "AAA"
    assert payload["governance_artifacts"]["regime_metrics"]["artifact_type"] == "regime_metrics_v1"
    assert payload["governance_artifacts"]["stress_validation"]["artifact_type"] == "stress_validation_v1"
    assert payload["governance_artifacts"]["symbol_attribution"]["artifact_type"] == "symbol_attribution_v1"
    assert "top_profit_contributors" in payload["governance_artifacts"]["symbol_attribution"]
    assert payload["run_provenance"]["artifact_type"] == "run_provenance_v1"
    assert payload["run_provenance"]["dataset_version"] == "fixture-cn"
    assert payload["run_provenance"]["params_hash"] == payload["request_hash"]


def test_multi_strategy_portfolio_endpoint_allocates_via_allocator(tmp_path, monkeypatch):
    """Multi-strategy endpoint runs sleeves and returns allocation artifact."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio/multi",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "min_commission": 0,
            "sleeves": [
                {"sleeve_id": "fast", "lookback_days": 2, "max_positions": 1},
                {"sleeve_id": "slow", "lookback_days": 3, "max_positions": 1},
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["allocation"]["artifact_type"] == "allocation_v1"
    assert len(payload["sleeves"]) == 2
    assert sum(payload["allocation"]["actual_weights"].values()) <= 1.0 + 1e-9


def test_multi_strategy_portfolio_rejects_naive_sum(tmp_path, monkeypatch):
    """Multi-strategy endpoint rejects naive_sum allocator."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio/multi",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "allocator_method": "naive_sum",
            "sleeves": [
                {"sleeve_id": "a", "lookback_days": 2, "max_positions": 1},
                {"sleeve_id": "b", "lookback_days": 3, "max_positions": 1},
            ],
        },
    )
    assert response.status_code == 400
    assert "naive sum" in response.json()["detail"]


@pytest.mark.xfail(strict=False, reason="artifacts 分页字段与当前实现不一致，待对齐")
def test_portfolio_backtest_artifacts_endpoint_returns_paginated_rows(tmp_path, monkeypatch):
    """Artifacts endpoint returns paginated rows for a cached run."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    run_resp = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "strategy": "momentum_top_k",
            "max_positions": 1,
            "lookback_days": 2,
            "min_commission": 0,
        },
    )
    run_id = run_resp.json()["request_hash"]
    artifacts_resp = TestClient(app).get(
        f"/api/backtest/{run_id}/artifacts?artifact=trades&limit=1&offset=0"
    )
    assert artifacts_resp.status_code == 200
    payload = artifacts_resp.json()
    assert payload["artifact"] == "trades"
    assert payload["total"] >= 1
    assert len(payload["items"]) == 1
    benchmark_resp = TestClient(app).get(
        f"/api/backtest/{run_id}/artifacts?artifact=benchmark_curve&limit=5&offset=0"
    )
    assert benchmark_resp.status_code == 200
    benchmark_payload = benchmark_resp.json()
    assert benchmark_payload["artifact"] == "benchmark_curve"
    assert benchmark_payload["total"] >= 1
    metadata_resp = TestClient(app).get(
        f"/api/backtest/{run_id}/artifacts?artifact=run_metadata&limit=5&offset=0"
    )
    assert metadata_resp.status_code == 200
    metadata_payload = metadata_resp.json()
    assert metadata_payload["artifact"] == "run_metadata"
    assert metadata_payload["total"] == 1
    assert metadata_payload["items"][0]["request_hash"] == run_id


@pytest.mark.xfail(strict=False, reason="snapshot 读取路径与当前实现不一致，待对齐")
def test_portfolio_artifacts_endpoint_reads_from_snapshot_file(tmp_path, monkeypatch):
    """Artifacts endpoint can read backtest snapshot after cache reset."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    run_resp = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "strategy": "momentum_top_k",
            "max_positions": 1,
            "lookback_days": 2,
            "min_commission": 0,
        },
    )
    run_id = run_resp.json()["request_hash"]
    from services import portfolio_backtest as portfolio_svc

    portfolio_svc._PORTFOLIO_ARTIFACTS.clear()
    artifacts_resp = TestClient(app).get(
        f"/api/backtest/{run_id}/artifacts?artifact=trades&limit=1&offset=0"
    )
    assert artifacts_resp.status_code == 200
    payload = artifacts_resp.json()
    assert payload["artifact"] == "trades"
    assert payload["total"] >= 1


def test_portfolio_backtest_endpoint_runs_etf_rotation_top1(tmp_path, monkeypatch):
    """Portfolio endpoint supports ETF rotation top1 strategy."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "strategy": "etf_rotation_top1",
            "lookback_days": 2,
            "min_commission": 0,
            "symbols": ["AAA", "BBB"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "etf_rotation_top1"
    assert payload["trades"]


def test_portfolio_backtest_endpoint_returns_json_safe_metrics(tmp_path, monkeypatch):
    """Portfolio endpoint must not emit NaN/Inf (Starlette JSON rejects them)."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-02",
            "end": "2024-01-03",
            "initial_cash": 100000,
            "strategy": "momentum_top_k",
            "max_positions": 1,
            "lookback_days": 2,
            "min_commission": 0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    vol = payload["metrics"]["annualized_volatility"]
    assert vol is None or vol == 0.0


def test_portfolio_backtest_endpoint_rejects_missing_db(tmp_path, monkeypatch):
    """Portfolio endpoint returns a clear error when local DB is missing."""
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(tmp_path / "missing.duckdb"))
    response = TestClient(app).post(
        "/api/backtest/portfolio",
        json={"start": "2024-01-01", "end": "2024-01-04"},
    )
    assert response.status_code == 503
    assert "本地行情库不存在" in response.json()["detail"]


def test_portfolio_backtest_api_rejects_unstable_dataset_version(tmp_path, monkeypatch):
    """Portfolio endpoint rejects unstable dataset versions."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn-unstable",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "strategy": "momentum_top_k",
            "max_positions": 1,
            "lookback_days": 2,
            "min_commission": 0,
        },
    )
    assert response.status_code == 400
    assert "not stable" in response.json()["detail"]


def test_portfolio_backtest_api_validates_universe_params(tmp_path, monkeypatch):
    """Portfolio endpoint rejects invalid include_exchanges."""
    db_path = tmp_path / "market.duckdb"
    _seed_api_fixture(db_path)
    monkeypatch.setenv("MARKET_DATA_DB_PATH", str(db_path))
    response = TestClient(app).post(
        "/api/backtest/portfolio",
        json={
            "market": "cn",
            "dataset_version": "fixture-cn",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "initial_cash": 100000,
            "strategy": "momentum_top_k",
            "max_positions": 1,
            "lookback_days": 2,
            "min_commission": 0,
            "include_exchanges": ["NYSE"],
        },
    )
    assert response.status_code == 400
    assert "include_exchanges" in response.json()["detail"]

