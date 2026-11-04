#!/usr/bin/env python3
"""Track local market data refresh progress (CN / US, checkpoints, DuckDB).

Examples:
  python scripts/track_market_refresh_progress.py
  python scripts/track_market_refresh_progress.py --market cn
  python scripts/track_market_refresh_progress.py --market us --json
  python scripts/track_market_refresh_progress.py --watch 30
  python scripts/track_market_refresh_progress.py --checkpoint data/market/checkpoints/cn_2016-01-01_2026-05-07.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DEFAULT_DB = PROJECT_ROOT / "data" / "market" / "market_data.duckdb"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "market" / "checkpoints"


@dataclass(frozen=True)
class CnBoardRule:
    """CN board label and SQL CASE branch."""

    label: str
    sql_predicate: str


CN_BOARD_RULES: tuple[CnBoardRule, ...] = (
    CnBoardRule("科创板", "i.symbol LIKE '688%'"),
    CnBoardRule("创业板", "i.symbol LIKE '300%'"),
    CnBoardRule("北交所", "i.symbol LIKE '920%' OR i.symbol LIKE '8%' OR i.symbol LIKE '4%'"),
    CnBoardRule("上证主板", "i.exchange = 'SSE' AND i.symbol LIKE '6%' AND i.symbol NOT LIKE '688%'"),
    CnBoardRule("深证主板", "i.exchange = 'SZSE' AND (i.symbol LIKE '000%' OR i.symbol LIKE '001%')"),
    CnBoardRule("中小板(深)", "i.exchange = 'SZSE' AND i.symbol LIKE '002%'"),
    CnBoardRule("深主板新段", "i.exchange = 'SZSE' AND i.symbol LIKE '003%'"),
)


def _parse_checkpoint_name(path: Path) -> dict[str, str]:
    """Parse ``{market}_{start}_{end}.json`` from filename."""
    match = re.match(r"^(cn|us)_(.+)_(\d{4}-\d{2}-\d{2})\.json$", path.name)
    if not match:
        return {"market": "?", "start": "", "end": ""}
    return {"market": match.group(1), "start": match.group(2), "end": match.group(3)}


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load refresh checkpoint JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    succeeded = payload.get("succeeded") or []
    failed = payload.get("failed_symbols") or {}
    return {
        "path": str(path),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "succeeded": set(str(s) for s in succeeded),
        "failed": dict(failed),
    }


def cn_board_case_sql() -> str:
    """Build SQL CASE for A-share board labels (uses ``i.`` column prefix)."""
    parts = [f"WHEN {rule.sql_predicate} THEN '{rule.label}'" for rule in CN_BOARD_RULES]
    return "CASE\n  " + "\n  ".join(parts) + "\n  ELSE '其他'\nEND"


def query_db_summary(conn: Any, *, market: str) -> dict[str, Any]:
    """Aggregate DuckDB coverage for one market."""
    inst_row = conn.execute(
        """
        SELECT COUNT(*), MIN(list_date), MAX(list_date)
        FROM instruments WHERE market = ?
        """,
        [market],
    ).fetchone()
    bar_row = conn.execute(
        """
        SELECT
            COUNT(DISTINCT symbol),
            MIN(trade_date),
            MAX(trade_date),
            COUNT(*)
        FROM daily_bars WHERE market = ?
        """,
        [market],
    ).fetchone()
    exchange_rows = conn.execute(
        """
        SELECT exchange, COUNT(*) FROM instruments WHERE market = ? GROUP BY exchange ORDER BY 2 DESC
        """,
        [market],
    ).fetchall()
    symbols_with_bars = {
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily_bars WHERE market = ?",
            [market],
        ).fetchall()
    }
    all_symbols = {
        str(r[0])
        for r in conn.execute(
            "SELECT symbol FROM instruments WHERE market = ?",
            [market],
        ).fetchall()
    }
    missing_in_db = sorted(all_symbols - symbols_with_bars)
    return {
        "instruments": int(inst_row[0] or 0),
        "symbols_with_daily_bars": int(bar_row[0] or 0),
        "bar_min_date": str(bar_row[1]) if bar_row[1] else None,
        "bar_max_date": str(bar_row[2]) if bar_row[2] else None,
        "daily_bar_rows": int(bar_row[3] or 0),
        "exchanges": {str(ex): int(n) for ex, n in exchange_rows},
        "missing_daily_bars": missing_in_db,
        "missing_daily_bars_count": len(missing_in_db),
    }


def query_cn_board_breakdown(conn: Any) -> list[dict[str, Any]]:
    """Per-board instrument vs daily-bar symbol counts."""
    board_expr = cn_board_case_sql()
    rows = conn.execute(
        f"""
        WITH tagged AS (
            SELECT
                i.symbol,
                {board_expr} AS board,
                CASE WHEN b.symbol IS NOT NULL THEN 1 ELSE 0 END AS has_bars
            FROM instruments AS i
            LEFT JOIN (
                SELECT DISTINCT symbol FROM daily_bars WHERE market = 'cn'
            ) AS b ON b.symbol = i.symbol
            WHERE i.market = 'cn'
        )
        SELECT
            board,
            COUNT(*) AS instruments,
            SUM(has_bars) AS with_daily_bars,
            COUNT(*) - SUM(has_bars) AS missing_daily_bars
        FROM tagged
        GROUP BY board
        ORDER BY instruments DESC
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for board, inst_n, with_bars, missing in rows:
        inst_n = int(inst_n)
        with_bars = int(with_bars)
        out.append(
            {
                "board": str(board),
                "instruments": inst_n,
                "with_daily_bars": with_bars,
                "missing_daily_bars": int(missing),
                "coverage_pct": round(100.0 * with_bars / inst_n, 2) if inst_n else 100.0,
            }
        )
    return out


def list_checkpoints(directory: Path, *, market: str | None) -> list[Path]:
    """List checkpoint JSON files, newest mtime first."""
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if market:
        paths = [p for p in paths if p.name.startswith(f"{market}_")]
    return paths


def merge_checkpoint_progress(
    db_summary: dict[str, Any],
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare checkpoint succeeded/failed vs DB instrument universe."""
    total_inst = db_summary["instruments"]
    if checkpoint is None:
        return {
            "checkpoint": None,
            "refresh_complete_pct": None,
            "status": "no_checkpoint",
        }
    succeeded = checkpoint["succeeded"]
    failed = checkpoint["failed"]
    attempted = len(succeeded) + len(failed)
    with_bars = db_summary["symbols_with_daily_bars"]

    return {
        "checkpoint": {
            "path": checkpoint["path"],
            "succeeded": checkpoint["succeeded_count"],
            "failed": checkpoint["failed_count"],
            "attempted": attempted,
        },
        "attempted_vs_instruments": {
            "instruments_in_db": total_inst,
            "attempted": attempted,
            "gap": total_inst - attempted,
        },
        "failed_sample": dict(list(failed.items())[:5]),
        "refresh_complete_pct": round(100.0 * len(succeeded) / attempted, 2) if attempted else None,
        "db_bar_coverage_pct": round(100.0 * with_bars / total_inst, 2) if total_inst else None,
        "status": _status_label(
            failed_count=len(failed),
            missing_bars=db_summary["missing_daily_bars_count"],
            attempted=attempted,
            instruments=total_inst,
        ),
    }


def _status_label(
    *,
    failed_count: int,
    missing_bars: int,
    attempted: int,
    instruments: int,
) -> str:
    if instruments == 0:
        return "empty_db"
    if missing_bars == 0 and failed_count == 0:
        return "complete"
    if missing_bars == 0 and failed_count > 0:
        return "db_complete_checkpoint_has_failures"
    if failed_count > 0:
        return "has_failures"
    if missing_bars > 0:
        return "missing_bars"
    return "in_progress"


def build_report(
    *,
    db_path: Path,
    checkpoint_path: Path | None,
    market_filter: str | None,
) -> dict[str, Any]:
    """Build full progress report from DB + checkpoints."""
    from marketdata.db import connect

    conn = connect(db_path)
    markets = ["cn", "us"] if market_filter is None else [market_filter]
    checkpoint_dir = checkpoint_path.parent if checkpoint_path else DEFAULT_CHECKPOINT_DIR

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(db_path),
        "markets": {},
    }

    for market in markets:
        db_summary = query_db_summary(conn, market=market)
        cp_path = checkpoint_path
        if cp_path is None:
            candidates = list_checkpoints(checkpoint_dir, market=market)
            cp_path = candidates[0] if candidates else None
        cp_data = load_checkpoint(cp_path) if cp_path and cp_path.exists() else None
        progress = merge_checkpoint_progress(db_summary, cp_data)
        if cp_data:
            symbols_with_bars = set(
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT symbol FROM daily_bars WHERE market = ?",
                    [market],
                ).fetchall()
            )
            progress["succeeded_missing_bars_count"] = len(cp_data["succeeded"] - symbols_with_bars)

        entry: dict[str, Any] = {
            "database": db_summary,
            "checkpoint_progress": progress,
            "checkpoints_available": [
                {
                    **_parse_checkpoint_name(p),
                    "path": str(p),
                    "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                }
                for p in list_checkpoints(checkpoint_dir, market=market)[:8]
            ],
        }
        if market == "cn":
            entry["cn_boards"] = query_cn_board_breakdown(conn)
        report["markets"][market] = entry

    versions = conn.execute(
        """
        SELECT dataset_version, markets, min_trade_date, max_trade_date, is_stable, quality_status
        FROM dataset_versions
        ORDER BY created_at DESC
        LIMIT 5
        """
    ).fetchall()
    report["dataset_versions"] = [
        {
            "dataset_version": row[0],
            "markets": row[1],
            "min_trade_date": str(row[2]) if row[2] else None,
            "max_trade_date": str(row[3]) if row[3] else None,
            "is_stable": bool(row[4]),
            "quality_status": row[5],
        }
        for row in versions
    ]
    return report


def print_human_report(report: dict[str, Any]) -> None:
    """Print a readable progress summary."""
    print(f"生成时间: {report['generated_at']}")
    print(f"数据库: {report['db_path']}")
    print()

    for market, data in report["markets"].items():
        db = data["database"]
        prog = data["checkpoint_progress"]
        title = "A股" if market == "cn" else "美股"
        print(f"{'=' * 60}")
        print(f"{title} ({market})")
        print(f"{'=' * 60}")
        print(
            f"  instruments: {db['instruments']:,}  |  "
            f"有日线: {db['symbols_with_daily_bars']:,}  |  "
            f"缺日线: {db['missing_daily_bars_count']:,}"
        )
        if db["bar_min_date"]:
            print(
                f"  日线区间: {db['bar_min_date']} ~ {db['bar_max_date']}  "
                f"({db['daily_bar_rows']:,} 行)"
            )
        if db["exchanges"]:
            print(f"  交易所: {db['exchanges']}")

        cp = prog.get("checkpoint")
        if cp:
            print(
                f"  checkpoint: 成功 {cp['succeeded']:,} / 失败 {cp['failed']:,} "
                f"(尝试 {cp['attempted']:,})"
            )
            print(f"  路径: {cp['path']}")
            if prog.get("failed_sample"):
                print("  失败样例:")
                for sym, reason in prog["failed_sample"].items():
                    print(f"    - {sym}: {reason[:80]}")
        else:
            print("  checkpoint: 无")

        status = prog.get("status", "?")
        status_zh = {
            "complete": "已完成",
            "db_complete_checkpoint_has_failures": "库内日线已齐，checkpoint 仍有失败记录（可重试）",
            "has_failures": "有失败标的",
            "missing_bars": "有标的缺日线",
            "in_progress": "进行中",
            "no_checkpoint": "无 checkpoint",
            "empty_db": "库为空",
        }.get(status, status)
        print(f"  状态: {status_zh}")
        if prog.get("db_bar_coverage_pct") is not None:
            print(f"  库内日线覆盖率: {prog['db_bar_coverage_pct']}%")

        if market == "cn" and data.get("cn_boards"):
            print()
            print("  板块覆盖 (instruments / 有日线 / 覆盖率):")
            for row in data["cn_boards"]:
                print(
                    f"    {row['board']:10}  {row['instruments']:5} / "
                    f"{row['with_daily_bars']:5} / {row['coverage_pct']:6.2f}%"
                )
        print()

    if report.get("dataset_versions"):
        print("最近 dataset_versions:")
        for v in report["dataset_versions"]:
            stable = "stable" if v["is_stable"] else "draft"
            print(
                f"  - {v['dataset_version']} [{stable}] "
                f"{v['min_trade_date']} ~ {v['max_trade_date']} ({v['quality_status']})"
            )


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser(description="Track market data refresh progress")
    parser.add_argument("--db-path", default=str(DEFAULT_DB), help="DuckDB path")
    parser.add_argument("--market", choices=["cn", "us", "all"], default="all")
    parser.add_argument("--checkpoint", default="", help="Specific checkpoint JSON path")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--watch", type=int, default=0, metavar="SEC", help="Refresh every N seconds")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"数据库不存在: {db_path}", file=sys.stderr)
        sys.exit(1)

    market_filter = None if args.market == "all" else args.market
    checkpoint_path = Path(args.checkpoint) if args.checkpoint.strip() else None

    def _run_once() -> dict[str, Any]:
        return build_report(
            db_path=db_path,
            checkpoint_path=checkpoint_path,
            market_filter=market_filter,
        )

    if args.watch > 0:
        try:
            while True:
                if args.json:
                    print(json.dumps(_run_once(), ensure_ascii=False, indent=2))
                else:
                    print("\033[2J\033[H", end="")
                    print_human_report(_run_once())
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n已停止 watch。")
            return

    report = _run_once()
    if args.json:
        # Convert sets if any slipped in (none expected in final report)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_human_report(report)


if __name__ == "__main__":
    main()
