#!/usr/bin/env python3
"""校验 A 股原始价/前复权价修复结果，并隔离失败代码的断层区间。

用法:
    python scripts/verify_cn_data_quality.py
    python scripts/verify_cn_data_quality.py --quarantine   # 删除失败代码 2026-07-17 之后的行
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import duckdb  # noqa: E402

DB = ROOT / "data" / "market" / "market_data.duckdb"
CHECKPOINT = ROOT / "data" / "market" / "checkpoints" / "repair_cn_adjust_20261026.json"
SEAM_START = date(2026, 7, 18)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarantine", action="store_true", help="删除失败代码断层区间数据")
    parser.add_argument("--date", default="2026-07-20", help="检查的断层日期")
    args = parser.parse_args()

    failed: list[str] = []
    if CHECKPOINT.exists():
        try:
            state = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            failed = sorted(state.get("failed", {}).keys())
        except Exception:
            pass

    con = duckdb.connect(str(DB))
    try:
        summary = con.execute("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT symbol) AS symbols,
                   MIN(trade_date) AS min_date,
                   MAX(trade_date) AS max_date,
                   SUM(CASE WHEN adj_factor IS NULL THEN 1 ELSE 0 END) AS missing_factor
            FROM daily_bars WHERE market='cn'
        """).fetchone()
        print(f"CN rows={summary[0]} symbols={summary[1]} range={summary[2]}~{summary[3]} "
              f"missing_adj_factor={summary[4]}")

        # 断层日横截面收益
        seam = con.execute("""
            WITH prices AS (
                SELECT symbol,
                       MAX(CASE WHEN trade_date = DATE '2026-07-17' THEN close END) AS p0,
                       MAX(CASE WHEN trade_date = CAST(? AS DATE) THEN close END) AS p1,
                       MAX(CASE WHEN trade_date = DATE '2026-07-17' THEN adj_close END) AS a0,
                       MAX(CASE WHEN trade_date = CAST(? AS DATE) THEN adj_close END) AS a1
                FROM daily_bars WHERE market='cn' AND trade_date IN (DATE '2026-07-17', CAST(? AS DATE))
                GROUP BY symbol
            )
            SELECT COUNT(*) FILTER (WHERE p0 IS NOT NULL AND p1 IS NOT NULL) AS paired,
                   AVG(p1 / p0 - 1) AS mean_raw_ret,
                   MIN(p1 / p0 - 1) AS min_raw_ret,
                   MAX(p1 / p0 - 1) AS max_raw_ret,
                   COUNT(*) FILTER (WHERE p0 IS NOT NULL AND p1 IS NOT NULL AND ABS(p1 / p0 - 1) > 0.35) AS outliers
            FROM prices
        """, [args.date, args.date, args.date]).fetchone()
        print(f"seam {args.date}: paired={seam[0]} mean_raw={seam[1]:+.4f} "
              f"min={seam[2]:+.4f} max={seam[3]:+.4f} outliers>35%={seam[4]}")

        # 近期异常日收益（原始价）
        bad = con.execute("""
            WITH r AS (
                SELECT symbol, trade_date, close,
                       LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev
                FROM daily_bars
                WHERE market='cn' AND trade_date BETWEEN DATE '2026-07-01' AND DATE '2026-10-26'
            )
            SELECT symbol, trade_date, close / prev - 1 AS ret
            FROM r WHERE prev IS NOT NULL AND prev > 0 AND ABS(close / prev - 1) > 0.35
            ORDER BY ABS(close / prev - 1) DESC LIMIT 20
        """).fetchall()
        print(f"extreme raw returns >35% in 2026-07..10: {len(bad)}")
        for row in bad[:5]:
            print("  ", row)

        # 失败代码中仍有断层后数据的
        if failed:
            placeholders = ",".join(["?"] * len(failed))
            count = con.execute(
                f"SELECT COUNT(*) FROM daily_bars WHERE market='cn' AND symbol IN ({placeholders}) "
                f"AND trade_date >= CAST(? AS DATE)",
                [*failed, SEAM_START],
            ).fetchone()[0]
            print(f"failed symbols={len(failed)}, rows after {SEAM_START}: {count}")
            if args.quarantine and count:
                # 分批删除，避免单条 SQL 参数过多
                for i in range(0, len(failed), 200):
                    chunk = failed[i:i + 200]
                    ph = ",".join(["?"] * len(chunk))
                    con.execute(
                        f"DELETE FROM daily_bars WHERE market='cn' AND symbol IN ({ph}) "
                        f"AND trade_date >= CAST(? AS DATE)",
                        [*chunk, SEAM_START],
                    )
                print(f"quarantined {count} rows for {len(failed)} failed symbols")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
