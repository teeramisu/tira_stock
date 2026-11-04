#!/usr/bin/env python
"""Repair US instrument exchange metadata in the local DuckDB warehouse."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair US instrument exchange metadata")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Only repair first N symbols (0=all)")
    parser.add_argument("--only-unknown", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    from marketdata.db import connect
    from marketdata.models import Instrument
    from marketdata.repository import MarketDataRepository
    from marketdata.sources.yfinance_source import (
        _load_us_exchange_map_akshare,
        _load_us_exchange_map_yfinance,
    )

    args = _parse_args()
    repo = MarketDataRepository(connect(args.db_path or None))
    where_sql = "WHERE market = 'us'"
    if args.only_unknown:
        where_sql += " AND exchange = 'UNKNOWN'"
    rows = repo.conn.execute(
        f"SELECT symbol FROM instruments {where_sql} ORDER BY symbol"
    ).fetchall()
    symbols = [str(row[0]).upper() for row in rows]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        print("no US instruments to repair")
        return

    print(f"repairing symbols={len(symbols)} batch_size={args.batch_size}")
    exchange_map = _load_us_exchange_map_akshare(symbols)
    missing = [symbol for symbol in symbols if symbol not in exchange_map]
    for start in range(0, len(missing), args.batch_size):
        chunk = missing[start : start + args.batch_size]
        exchange_map.update(_load_us_exchange_map_yfinance(chunk, chunk_size=args.batch_size))
        repaired = sum(1 for symbol in chunk if exchange_map.get(symbol, "UNKNOWN") != "UNKNOWN")
        print(f"batch {start // args.batch_size + 1}: repaired={repaired}/{len(chunk)}")

    instruments = [
        Instrument(
            market="us",
            symbol=symbol,
            name=symbol,
            exchange=exchange_map.get(symbol, "UNKNOWN"),
            currency="USD",
            source="yfinance",
        )
        for symbol in symbols
        if exchange_map.get(symbol, "UNKNOWN") != "UNKNOWN"
    ]
    if not instruments:
        print("no exchange metadata resolved")
        return
    repo.upsert_instruments(instruments)
    unknown_left = repo.conn.execute(
        "SELECT COUNT(*) FROM instruments WHERE market='us' AND exchange='UNKNOWN'"
    ).fetchone()[0]
    print(f"upserted={len(instruments)} unknown_left={unknown_left}")


if __name__ == "__main__":
    main()
