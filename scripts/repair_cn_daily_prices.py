#!/usr/bin/env python3
"""修复 A 股复权断层：重新拉取全历史「原始价 + 前复权价」。

存储约定
- open/high/low/close = 未复权真实成交价
- adj_open/.../adj_close = 前复权价（统一以最新交易日为基准）
- adj_factor = adj_close / close

用法:
    # 先试跑 3 只，不写库
    python scripts/repair_cn_daily_prices.py --symbols 600519,688195,300750 --dry-run

    # 全量修复（建议停掉服务；会占用 DuckDB 写锁）
    python scripts/repair_cn_daily_prices.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from marketdata.db import connect  # noqa: E402
from marketdata.refresh import create_stable_version_from_current  # noqa: E402
from marketdata.repository import MarketDataRepository  # noqa: E402
from marketdata.sources.akshare_source import load_cn_daily_bars_raw_and_qfq  # noqa: E402

CHECKPOINT = ROOT / "data" / "market" / "checkpoints" / "repair_cn_adjust_20261026.json"


def _validate(bars: list) -> None:
    if len(bars) < 20:
        raise ValueError(f"too few bars: {len(bars)}")
    dates = [b.trade_date for b in bars]
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate dates")
    for b in bars:
        if min(b.open, b.high, b.low, b.close) <= 0:
            raise ValueError(f"non-positive raw price on {b.trade_date}")
        if b.adj_close is None or b.adj_close <= 0:
            raise ValueError(f"non-positive adj_close on {b.trade_date}")
        if b.high < max(b.open, b.close) - 1e-6 or b.low > min(b.open, b.close) + 1e-6:
            raise ValueError(f"invalid OHLC on {b.trade_date}")
    # 原始价允许拆股/送转跳变；复权价允许复牌跳空，但不能出现 3 倍级错误。
    # 同时对已知的 2026-07-20 断层日做单独检查。
    seam = date(2026, 7, 20)
    for i in range(1, len(bars)):
        prev, cur = bars[i - 1], bars[i]
        if not prev.adj_close or prev.adj_close <= 0:
            continue
        ret = cur.adj_close / prev.adj_close - 1
        if abs(ret) > 0.85:
            raise ValueError(f"qfq price jump {ret:.1%} on {cur.trade_date}")
        if cur.trade_date == seam and abs(ret) > 0.25:
            raise ValueError(f"qfq seam jump {ret:.1%} on {cur.trade_date}")
    factors = [b.adj_factor for b in bars if b.adj_factor]
    if len(factors) < len(bars) * 0.9:
        raise ValueError("adj_factor coverage < 90%")


def _fetch(symbol: str, start: date, end: date, attempts: int = 2):
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            bars = load_cn_daily_bars_raw_and_qfq(symbol, start, end)
            _validate(bars)
            return symbol, bars, None
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < attempts:
                time.sleep(0.4 * attempt)
    return symbol, None, last_err


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"succeeded": [], "failed": {}, "updated_at": ""}


def _save_checkpoint(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair CN raw + qfq daily prices")
    parser.add_argument("--symbols", default="", help="逗号分隔，仅修复指定代码")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 只（调试）")
    parser.add_argument("--dry-run", action="store_true", help="只抓取校验，不写库")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--retry-failed", action="store_true", help="只重试 checkpoint 中失败代码")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    checkpoint_path = Path(args.checkpoint)
    state = _load_checkpoint(checkpoint_path)
    succeeded = set(state.get("succeeded", []))
    failed: dict[str, str] = dict(state.get("failed", {}))

    repo = MarketDataRepository(connect())
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = [
            row[0]
            for row in repo.conn.execute(
                "SELECT DISTINCT symbol FROM daily_bars WHERE market='cn' ORDER BY symbol"
            ).fetchall()
        ]
    if args.retry_failed:
        symbols = sorted(failed.keys())
    elif not args.symbols:
        symbols = [s for s in symbols if s not in succeeded]
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"待修复 {len(symbols)} 只（已完成 {len(succeeded)}，失败 {len(failed)}）")
    if args.dry_run:
        for symbol in symbols:
            sym, bars, err = _fetch(symbol, start, end)
            if err:
                print(f"  {sym} FAIL {err[:120]}")
            else:
                p0 = bars[0]
                p1 = bars[-1]
                print(f"  {sym} OK {len(bars)} bars {p0.trade_date}~{p1.trade_date} "
                      f"raw={p1.close:.2f} qfq={p1.adj_close:.2f} factor={p1.adj_factor}")
        return 0

    started = time.time()
    written = 0
    errors = 0
    done = 0
    lock = __import__("threading").Lock()

    def write_one(symbol: str, bars: list) -> None:
        nonlocal written
        with lock:
            try:
                repo.conn.execute("BEGIN")
                repo.conn.execute(
                    "DELETE FROM daily_bars WHERE market='cn' AND symbol=?", [symbol]
                )
                repo.upsert_daily_bars(bars)
                repo.conn.execute("COMMIT")
                written += len(bars)
            except Exception:
                repo.conn.execute("ROLLBACK")
                raise

    iterator = iter(symbols)
    inflight: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        def submit_next() -> bool:
            try:
                symbol = next(iterator)
            except StopIteration:
                return False
            inflight[executor.submit(_fetch, symbol, start, end)] = symbol
            return True

        for _ in range(max(1, args.workers) * 2):
            if not submit_next():
                break

        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for future in finished:
                symbol = inflight.pop(future)
                sym, bars, err = future.result()
                done += 1
                if err:
                    errors += 1
                    failed[sym] = err[:300]
                    print(f"[{done}/{len(symbols)}] {sym} FAIL {err[:100]}")
                else:
                    try:
                        write_one(sym, bars)
                        succeeded.add(sym)
                        failed.pop(sym, None)
                        print(f"[{done}/{len(symbols)}] {sym} OK {len(bars)} bars")
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        failed[sym] = f"write: {exc}"[:300]
                        print(f"[{done}/{len(symbols)}] {sym} WRITE FAIL {exc}")
                if done % 10 == 0 or done == len(symbols):
                    state["succeeded"] = sorted(succeeded)
                    state["failed"] = failed
                    _save_checkpoint(checkpoint_path, state)
                if done % 50 == 0:
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 1e-9)
                    eta = (len(symbols) - done) / max(rate, 1e-9)
                    print(f"  progress {done}/{len(symbols)}  {rate:.2f}/s  ETA {eta/60:.1f} min")
                submit_next()

    state["succeeded"] = sorted(succeeded)
    state["failed"] = failed
    _save_checkpoint(checkpoint_path, state)

    print(f"完成：成功 {len(succeeded)}，失败 {len(failed)}，写入 {written} 条，耗时 {(time.time()-started)/60:.1f} min")
    version = create_stable_version_from_current(
        repo,
        source="akshare_raw+qfq-repair",
        markets=("cn",),
        quality_summary={
            "repair": True,
            "symbols_succeeded": len(succeeded),
            "symbols_failed": len(failed),
            "bars_written": written,
            "date_range": [str(start), str(end)],
        },
    )
    print(f"新数据集版本: {version.dataset_version}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
