#!/usr/bin/env python3
"""刷新国家统计局 / 人民银行宏观数据。

用法:
    python scripts/refresh_macro_data.py
    python scripts/refresh_macro_data.py --sources akshare,nbs,pbc
    python scripts/refresh_macro_data.py --no-export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from macro import MacroStore, refresh  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh NBS/PBOC macro data")
    parser.add_argument("--sources", default="akshare,cn_extra,nbs,pbc,usjp,worldbank")
    parser.add_argument("--db", default="", help="macro.duckdb 路径")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--export-dir", default="")
    args = parser.parse_args()

    store = MacroStore(args.db or None)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    result = refresh(
        sources=sources,
        store=store,
        export=not args.no_export,
        export_dir=args.export_dir or None,
    )
    print(f"run_id={result['run_id']} status={result['status']}")
    print(f"series={result['series']} observations={result['observations']} raw_tables={result['raw_tables']}")
    print(f"db={result['db']}")
    if result["exports"]:
        for k, v in result["exports"].items():
            print(f"export {k}: {v}")
    if result["errors"]:
        print("errors:")
        for err in result["errors"]:
            print(" -", err)
    status = store.status()
    print("status:", status)
    latest = store.latest()
    print("\nlatest:")
    for row in latest[:20]:
        print(f"  {row['series_id']:<32} {row['period']}  {row['value']:>14.4f} {row['unit']}")
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
