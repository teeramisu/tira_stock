#!/usr/bin/env python
"""Refresh US equity option chains into local DuckDB (Yahoo Finance snapshots).

Full-universe pulls require hours or days: Yahoo rate-limits aggressively. Use
``--checkpoint``, ``--sleep``, ``--limit``, and resume with ``--retry-failed-only``.

Examples::

    # Sample underlyings (default limit 20)
    python scripts/refresh_us_options.py --universe cli --symbols AAPL,MSFT

    # All symbols listed in local DuckDB ``instruments`` (market=us)
    python scripts/refresh_us_options.py --universe duckdb --checkpoint data/market/checkpoints/us_options.json

    # Full US equity list from AkShare spot (very large; throttle heavily)
    python scripts/refresh_us_options.py --universe akshare --sleep 2.0 --checkpoint data/market/checkpoints/us_options.json

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh US option chains into DuckDB")
    parser.add_argument("--db-path", default="", help="DuckDB path; default from MARKET_DATA_DB_PATH or project default")
    parser.add_argument(
        "--universe",
        choices=["cli", "duckdb", "akshare"],
        default="cli",
        help="Where to discover US equity underlyings",
    )
    parser.add_argument("--symbols", default="", help="Comma-separated tickers (universe=cli)")
    parser.add_argument("--symbol-file", default="", help="One ticker per line (universe=cli)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of underlyings after offset (0 = no cap; recommended with --checkpoint)",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip first N symbols after sort")
    parser.add_argument(
        "--max-expirations",
        type=int,
        default=0,
        help="Max expiry dates per underlying (0 = all listed)",
    )
    parser.add_argument("--snapshot-date", default="", help="Snapshot date YYYY-MM-DD (default: UTC today)")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between underlyings")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-underlying yfinance timeout seconds")
    parser.add_argument("--checkpoint", default="", help="JSON checkpoint path for resume")
    parser.add_argument("--retry-failed-only", action="store_true", help="Only retry symbols in checkpoint failed_symbols")
    parser.add_argument("--max-attempts", type=int, default=2, help="Retries per underlying on transient errors")
    return parser.parse_args()


def _snapshot_date(raw: str) -> date:
    if raw.strip():
        return date.fromisoformat(raw.strip())
    return datetime.utcnow().date()


def _load_checkpoint(path: str) -> dict:
    p = Path(path)
    if not path or not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_checkpoint(path: str, succeeded: list[str], failed_symbols: dict[str, str]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"succeeded": succeeded, "failed_symbols": failed_symbols}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.universe == "cli":
        symbols: list[str] = []
        if args.symbols.strip():
            symbols.extend(s.strip().upper() for s in args.symbols.split(",") if s.strip())
        if args.symbol_file.strip():
            text = Path(args.symbol_file.strip()).read_text(encoding="utf-8")
            for line in text.splitlines():
                token = line.strip().upper()
                if token and not token.startswith("#"):
                    symbols.append(token)
        if not symbols:
            raise ValueError("universe=cli requires --symbols and/or --symbol-file")
        return sorted(set(symbols))
    if args.universe == "duckdb":
        from marketdata.db import connect
        from marketdata.repository import MarketDataRepository

        repo = MarketDataRepository(connect(args.db_path or None))
        df = repo.conn.execute(
            "SELECT DISTINCT symbol FROM instruments WHERE market = 'us' ORDER BY symbol"
        ).fetchdf()
        if df is None or df.empty:
            return []
        return [str(s).strip().upper() for s in df["symbol"].tolist()]
    if args.universe == "akshare":
        from marketdata.sources.akshare_source import load_us_symbols_akshare

        # Always fetch full symbol list once; apply ``--limit`` / ``--offset`` in ``main``.
        return load_us_symbols_akshare(limit=None)
    raise ValueError(f"unsupported universe: {args.universe}")


def main() -> None:
    args = _parse_args()
    snap = _snapshot_date(args.snapshot_date)
    max_exp = args.max_expirations if args.max_expirations > 0 else None

    from marketdata.db import connect
    from marketdata.repository import MarketDataRepository
    from marketdata.sources.yfinance_source import load_us_option_chain_snapshots

    symbols = _resolve_symbols(args)
    symbols = sorted(set(symbols))
    if args.offset > 0:
        symbols = symbols[args.offset :]
    if args.limit > 0:
        symbols = symbols[: args.limit]

    checkpoint_path = args.checkpoint.strip()
    cp = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    succeeded = sorted(set(cp.get("succeeded", [])))
    failed_map: dict[str, str] = dict(cp.get("failed_symbols", {}))

    if args.retry_failed_only:
        if not failed_map:
            print("retry-failed-only: no failed_symbols in checkpoint, exiting")
            return
        symbols = sorted(failed_map.keys())
        failed_map = {}

    repo = MarketDataRepository(connect(args.db_path or None))
    total_rows = 0
    for idx, sym in enumerate(symbols):
        if sym in succeeded:
            continue
        attempt = 0
        last_err = ""
        while attempt < args.max_attempts:
            attempt += 1
            try:
                rows = load_us_option_chain_snapshots(
                    sym,
                    snapshot_date=snap,
                    max_expirations=max_exp,
                    timeout_seconds=args.timeout,
                )
                n = repo.upsert_option_chain_snapshots(rows)
                total_rows += n
                succeeded.append(sym)
                succeeded = sorted(set(succeeded))
                failed_map.pop(sym, None)
                print(f"[{idx + 1}/{len(symbols)}] {sym} rows={n} expiries_ok")
                _save_checkpoint(checkpoint_path, succeeded, failed_map)
                break
            except Exception as exc:  # noqa: BLE001 - record provider failures
                last_err = str(exc)
                if attempt >= args.max_attempts:
                    failed_map[sym] = last_err
                    print(f"[{idx + 1}/{len(symbols)}] {sym} FAILED: {last_err}")
                    _save_checkpoint(checkpoint_path, succeeded, failed_map)
                else:
                    time.sleep(min(5.0 * attempt, 30.0))

        if args.sleep > 0 and idx + 1 < len(symbols):
            time.sleep(args.sleep)

    print(
        f"done snapshot_date={snap.isoformat()} underlyings_ok={len(succeeded)} "
        f"failed={len(failed_map)} total_option_rows_written={total_rows}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
