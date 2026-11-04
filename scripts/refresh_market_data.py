#!/usr/bin/env python
"""Refresh local DuckDB market data warehouse.

This script is intentionally small for M1. It supports sample CN/US daily
refreshes and creates a stable dataset version after successful writes.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh local market data warehouse")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--mode", choices=["init", "incremental", "status"], default="status")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols for sample refresh")
    parser.add_argument("--symbols-file", default="", help="One symbol per line; overrides --symbols when set")
    parser.add_argument("--skip-instruments", action="store_true", help="Skip instrument master refresh (already loaded)")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--frequency", choices=["1d", "1m", "5m", "15m", "30m", "60m"], default="1d")
    parser.add_argument("--db-path", default="")
    parser.add_argument(
        "--us-provider",
        choices=["yfinance", "stooq", "akshare", "auto"],
        default="auto",
        help="US daily source provider (auto tries yfinance->stooq->akshare).",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Retries per symbol before fail")
    parser.add_argument("--checkpoint-path", default="", help="Checkpoint JSON path for resume")
    parser.add_argument("--full-fetch", action="store_true", help="Refresh full market symbols for the market")
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Only refresh symbols that failed in checkpoint",
    )
    return parser.parse_args()


def _date_range(args: argparse.Namespace) -> tuple[date, date]:
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=120)
    return start, end


def main() -> None:
    """Run refresh command."""
    from marketdata.db import connect
    from marketdata.refresh import (
        create_stable_version_from_current,
        load_refresh_checkpoint,
        refresh_daily_bars_with_retries,
        refresh_intraday_bars_with_retries,
        run_daily_quality_checks,
        run_intraday_quality_checks,
    )
    from marketdata.models import RefreshRun
    from marketdata.repository import MarketDataRepository
    from marketdata.sources.akshare_source import (
        load_cn_daily_bars_for_symbol,
        load_cn_instruments,
        load_cn_instruments_for_symbols,
        load_cn_trading_calendar,
    )
    from marketdata.sources.yfinance_source import (
        load_us_daily_bars,
        load_us_instruments,
        load_us_intraday_bars,
        load_us_symbols_akshare,
    )

    args = _parse_args()
    repo = MarketDataRepository(connect(args.db_path or None))
    if args.mode == "status":
        rows = repo.conn.execute(
            """
            SELECT dataset_version, markets, min_trade_date, max_trade_date, is_stable
            FROM dataset_versions
            ORDER BY created_at DESC
            LIMIT 10
            """
        ).fetchall()
        for row in rows:
            print(row)
        return

    start, end = _date_range(args)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        symbols = [
            line.strip().upper()
            for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    checkpoint_override = args.checkpoint_path.strip() or None
    checkpoint_path: str | None = None
    refresh_run_id = f"refresh_{uuid.uuid4().hex[:12]}"
    run_started_at = datetime.utcnow()
    run_job_type = f"{args.market}_{args.mode}_{args.frequency}"
    try:
        if args.market == "cn":
            if args.frequency != "1d":
                raise ValueError("CN intraday refresh is not implemented yet, please use --frequency 1d")
            calendar_rows = load_cn_trading_calendar(start, end)
            if calendar_rows:
                cal_written = repo.upsert_trading_calendar(calendar_rows)
                print(f"trading_calendar upserted rows={cal_written} market=cn range={start}..{end}")
            else:
                print(f"WARN: 交易日历为空，跳过 upsert（行情刷新继续） range={start}..{end}")
            if symbols:
                instruments = load_cn_instruments_for_symbols(symbols)
            else:
                instrument_limit = None if args.full_fetch else args.limit
                instruments = load_cn_instruments(limit=instrument_limit)
            if not symbols:
                symbols = (
                    [item.symbol for item in instruments]
                    if args.full_fetch
                    else [item.symbol for item in instruments[: args.limit]]
                )
            repo.upsert_instruments(instruments)
            checkpoint_path = checkpoint_override or (
                str(PROJECT_ROOT / "data" / "market" / "checkpoints" / f"{args.market}_{start}_{end}.json")
                if args.full_fetch
                else None
            )
            if args.retry_failed_only:
                if not checkpoint_path:
                    raise ValueError("--retry-failed-only requires --checkpoint-path or --full-fetch")
                checkpoint = load_refresh_checkpoint(checkpoint_path)
                failed_only = sorted(checkpoint.get("failed_symbols", {}).keys())
                if not failed_only:
                    print("no failed symbols in checkpoint, nothing to refresh")
                    return
                symbols = failed_only
            source = "akshare"
        else:
            if args.full_fetch and not symbols:
                try:
                    symbols = load_us_symbols_akshare()
                except RuntimeError:
                    symbols = []
                if not symbols:
                    rows = repo.conn.execute(
                        "SELECT DISTINCT symbol FROM instruments WHERE market = 'us' ORDER BY symbol"
                    ).fetchall()
                    symbols = [str(row[0]).upper() for row in rows]
                    if symbols:
                        print(f"US full-fetch: akshare spot unavailable, using {len(symbols)} symbols from local DB")
                if not symbols:
                    raise RuntimeError("US full-fetch found no symbols from akshare spot endpoint or local DB")
            symbols = symbols or ["AAPL", "MSFT"]
            checkpoint_path = checkpoint_override or (
                str(PROJECT_ROOT / "data" / "market" / "checkpoints" / f"{args.market}_{start}_{end}.json")
                if args.full_fetch
                else None
            )
            if args.retry_failed_only:
                if not checkpoint_path:
                    raise ValueError("--retry-failed-only requires --checkpoint-path or --full-fetch")
                checkpoint = load_refresh_checkpoint(checkpoint_path)
                failed_only = sorted(checkpoint.get("failed_symbols", {}).keys())
                if not failed_only:
                    print("no failed symbols in checkpoint, nothing to refresh")
                    return
                symbols = failed_only
            if not args.skip_instruments:
                repo.upsert_instruments(load_us_instruments(symbols))
            source = args.us_provider if args.us_provider != "auto" else "us-auto"

        repo.upsert_refresh_run(
            RefreshRun(
                id=refresh_run_id,
                dataset_version="pending",
                job_type=run_job_type,
                started_at=run_started_at,
                status="running",
                symbols_total=len(symbols),
            )
        )
        if args.frequency == "1d":
            if args.market == "cn":
                summary = refresh_daily_bars_with_retries(
                    repo,
                    symbols=symbols,
                    loader=lambda symbol: load_cn_daily_bars_for_symbol(symbol, start, end),
                    max_attempts=args.max_attempts,
                    checkpoint_path=checkpoint_path,
                )
            else:
                summary = refresh_daily_bars_with_retries(
                    repo,
                    symbols=symbols,
                    loader=lambda symbol: load_us_daily_bars([symbol], start, end, provider=args.us_provider),
                    max_attempts=args.max_attempts,
                    checkpoint_path=checkpoint_path,
                )
            quality_report = run_daily_quality_checks(
                repo,
                market=args.market,
                start=start,
                end=end,
                expected_symbols=len(symbols),
            )
        else:
            day = start
            intraday_bars = 0
            intraday_success = 0
            intraday_requested = 0
            intraday_failed: dict[str, str] = {}
            while day <= end:
                day_start = datetime.combine(day, time.min)
                day_end = datetime.combine(day + timedelta(days=1), time.min)
                day_checkpoint = None
                if checkpoint_path:
                    day_checkpoint = f"{checkpoint_path}.{args.frequency}.{day.isoformat()}"
                day_summary = refresh_intraday_bars_with_retries(
                    repo,
                    symbols=symbols,
                    loader=lambda symbol, s=day_start, e=day_end: load_us_intraday_bars(
                        [symbol], start_time=s, end_time=e, frequency=args.frequency
                    ),
                    max_attempts=args.max_attempts,
                    checkpoint_path=day_checkpoint,
                )
                intraday_bars += day_summary.bars_written
                intraday_success += day_summary.symbols_success
                intraday_requested += day_summary.symbols_requested
                intraday_failed.update(day_summary.failed_symbols)
                day += timedelta(days=1)
            summary = day_summary
            quality_report = run_intraday_quality_checks(
                repo,
                market=args.market,
                start=datetime.combine(start, time.min),
                end=datetime.combine(end + timedelta(days=1), time.min),
                frequency=args.frequency,
                expected_symbols=len(symbols),
            )
            summary = type(day_summary)(
                symbols_requested=intraday_requested,
                symbols_success=intraday_success,
                symbols_failed=len(intraday_failed),
                bars_written=intraday_bars,
                failed_symbols=intraday_failed,
            )
        version = create_stable_version_from_current(
            repo,
            source=source,
            markets=(args.market,),
            quality_summary={
                "symbols_requested": summary.symbols_requested,
                "symbols_success": summary.symbols_success,
                "symbols_failed": summary.symbols_failed,
                "bars_written": summary.bars_written,
                "failed_symbols": summary.failed_symbols,
                "full_fetch": bool(args.full_fetch),
                "checkpoint_path": checkpoint_path,
                "frequency": args.frequency,
                "daily_quality": quality_report,
            },
        )
        repo.upsert_refresh_run(
            RefreshRun(
                id=refresh_run_id,
                dataset_version=version.dataset_version,
                job_type=run_job_type,
                started_at=run_started_at,
                finished_at=datetime.utcnow(),
                status="success" if summary.symbols_failed == 0 else "partial_success",
                symbols_total=summary.symbols_requested,
                symbols_success=summary.symbols_success,
                symbols_failed=summary.symbols_failed,
                error=None if not summary.failed_symbols else str(summary.failed_symbols),
            )
        )
        print(
            f"created dataset_version={version.dataset_version} "
            f"bars={summary.bars_written} success={summary.symbols_success}/{summary.symbols_requested} "
            f"failed={summary.symbols_failed} refresh_run_id={refresh_run_id}"
        )

        # ── 写入更新日志 ──
        try:
            from services.update_log import log_update
            market_name = "A股" if args.market == "cn" else "美股"
            log_update(
                market=args.market,
                data_type=f"{args.market}_daily",
                source_used=source,
                summary=f"{market_name}{len(symbols)}只标的日线数据更新至{end}，新增{summary.bars_written}条记录",
                symbols=symbols[:20],  # 最多记录前20个
                records_added=summary.bars_written,
                date_range={"start": str(start), "end": str(end)},
                details=f"模式={args.mode}, 频率={args.frequency}, 成功={summary.symbols_success}, 失败={summary.symbols_failed}, 版本={version.dataset_version}",
                source_chain=[f"{source}(success)"],
            )
        except Exception as log_exc:
            print(f"Warning: failed to write update log: {log_exc}")
    except Exception as exc:
        repo.upsert_refresh_run(
            RefreshRun(
                id=refresh_run_id,
                dataset_version="failed",
                job_type=run_job_type,
                started_at=run_started_at,
                finished_at=datetime.utcnow(),
                status="failed",
                symbols_total=len(symbols),
                symbols_success=0,
                symbols_failed=len(symbols),
                error=str(exc),
            )
        )
        raise


if __name__ == "__main__":
    main()

