#!/usr/bin/env python
"""Synthetic performance benchmark for portfolio backtest engine."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import multiprocessing as mp
import argparse

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.marketdata.db import connect  # noqa: E402
from backend.marketdata.models import DatasetVersion  # noqa: E402
from backend.marketdata.repository import MarketDataRepository  # noqa: E402
from backend.portfolio.broker import CostModel  # noqa: E402
from backend.portfolio.engine import PortfolioEngine  # noqa: E402
from backend.portfolio.models import PortfolioBacktestConfig  # noqa: E402
from backend.portfolio.strategies import MomentumTopKStrategy  # noqa: E402


def _seed_repo(repo: MarketDataRepository, *, n_symbols: int, start: date, years: int) -> tuple[date, date]:
    days = 252 * years
    repo.conn.execute(
        f"""
        INSERT INTO instruments (
            instrument_id, market, symbol, name, exchange, asset_type, currency,
            list_date, delist_date, is_active, is_st, source, updated_at
        )
        SELECT
            'cn:' || printf('S%04d', symbol_idx),
            'cn',
            printf('S%04d', symbol_idx),
            'Sym' || CAST(symbol_idx AS VARCHAR),
            'SSE',
            'stock',
            'CNY',
            ?::DATE,
            NULL,
            TRUE,
            FALSE,
            'benchmark',
            now()
        FROM range({n_symbols}) AS t(symbol_idx)
        """,
        [start],
    )
    repo.conn.execute(
        f"""
        INSERT INTO daily_bars (
            market, symbol, trade_date, open, high, low, close, pre_close,
            adj_open, adj_high, adj_low, adj_close, adjustment, volume, amount,
            adj_factor, paused, limit_up, limit_down, source, updated_at
        )
        SELECT
            'cn',
            printf('S%04d', symbol_idx),
            (?::DATE + (day_idx || ' day')::INTERVAL)::DATE,
            px,
            px * 1.01,
            px * 0.99,
            px,
            NULL,
            px,
            px * 1.01,
            px * 0.99,
            px,
            'qfq',
            1000000,
            NULL,
            1.0,
            FALSE,
            NULL,
            NULL,
            'benchmark',
            now()
        FROM range({n_symbols}) AS s(symbol_idx)
        CROSS JOIN range({days}) AS d(day_idx)
        CROSS JOIN LATERAL (
            SELECT (10.0 + symbol_idx * 0.01 + day_idx * 0.001) AS px
        ) p
        """,
        [start],
    )
    repo.upsert_dataset_version(
        DatasetVersion(
            dataset_version="benchmark-cn",
            markets=("cn",),
            quality_status="ok",
            is_stable=True,
        )
    )
    end = repo.conn.execute("SELECT MAX(trade_date) FROM daily_bars WHERE market = 'cn'").fetchone()[0]
    return start, end


def _run_case(n_symbols: int, years: int) -> tuple[float, dict[str, float]]:
    with TemporaryDirectory() as tmp_dir:
        repo = MarketDataRepository(connect(f"{tmp_dir}/benchmark.duckdb"))
        start, end = _seed_repo(repo, n_symbols=n_symbols, start=date(2016, 1, 4), years=years)
        started = perf_counter()
        result = PortfolioEngine(
            repository=repo,
            strategy=MomentumTopKStrategy(lookback_days=20, top_k=min(20, n_symbols)),
            config=PortfolioBacktestConfig(
                market="cn",
                dataset_version="benchmark-cn",
                start=start,
                end=end,
                initial_cash=1_000_000,
            ),
            cost_model=CostModel(min_commission=0),
        ).run()
        return perf_counter() - started, result.metrics


def _run_case_worker(n_symbols: int, years: int, queue: mp.Queue) -> None:
    queue.put(_run_case(n_symbols, years))


def _run_case_with_timeout(n_symbols: int, years: int, timeout_seconds: int) -> tuple[float, dict[str, float]] | None:
    queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(target=_run_case_worker, args=(n_symbols, years, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    if not queue.empty():
        return queue.get()
    return None


def _write_outputs(
    *,
    output_path: Path | None,
    history_path: Path | None,
    payload: dict[str, object],
) -> None:
    """Persist benchmark payload as JSON and optional JSONL history."""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic portfolio benchmark and optionally persist results.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write one benchmark snapshot JSON to this path.",
    )
    parser.add_argument(
        "--history-jsonl",
        type=Path,
        default=None,
        help="Append benchmark snapshot as one JSON line to this file.",
    )
    args = parser.parse_args()
    run_at = datetime.now(timezone.utc).isoformat()
    fixture_seconds, fixture_metrics = _run_case(20, 1)
    fixture_row = {"case": "fixture_20_1y", "seconds": round(fixture_seconds, 3), "timing": fixture_metrics}
    print(fixture_row, flush=True)
    case_100_seconds, case_100_metrics = _run_case(100, 3)
    case_100_row = {"case": "stocks_100_3y", "seconds": round(case_100_seconds, 3), "timing": case_100_metrics}
    print(case_100_row, flush=True)
    case_1000_seconds, case_1000_metrics = _run_case(1000, 5)
    case_1000_row = {"case": "stocks_1000_5y", "seconds": round(case_1000_seconds, 3), "timing": case_1000_metrics}
    print(case_1000_row, flush=True)
    case_5000 = _run_case_with_timeout(5000, 5, timeout_seconds=60)
    case_5000_seconds = None if case_5000 is None else case_5000[0]
    case_5000_metrics = {} if case_5000 is None else case_5000[1]
    case_5000_row = {
        "case": "stocks_5000_5y",
        "seconds": round(case_5000_seconds, 3) if case_5000_seconds is not None else None,
        "within_60s": case_5000_seconds is not None and case_5000_seconds < 60,
        "timing": case_5000_metrics,
    }
    print(case_5000_row, flush=True)
    summary = {
        "fixture_seconds": round(fixture_seconds, 3),
        "case_100_3y_seconds": round(case_100_seconds, 3),
        "case_1000_5y_seconds": round(case_1000_seconds, 3),
        "case_5000_5y_seconds": round(case_5000_seconds, 3) if case_5000_seconds is not None else None,
    }
    print(summary)
    payload: dict[str, object] = {
        "run_at_utc": run_at,
        "cases": [fixture_row, case_100_row, case_1000_row, case_5000_row],
        "summary": summary,
    }
    _write_outputs(output_path=args.output_json, history_path=args.history_jsonl, payload=payload)


if __name__ == "__main__":
    main()
