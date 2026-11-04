#!/usr/bin/env python3
"""A股增量刷新（原始价 + 前复权，按上次复权基准对齐）。

流程
1. 读取每只股票在库中的最后交易日与最后 adj_close；
2. 抓取「最后交易日 - 重叠窗口 ~ 今天」的 raw+qfq；
3. 用重叠日把新 qfq 缩放到旧基准：scale = old_adj_close / new_adj_close(重叠日)；
4. 只追加最后交易日之后的新行，raw 保持真实成交价，adj_factor = adj_close/close。

这样可以避免每次刷新因 qfq 基准变化产生断层。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from marketdata.db import connect  # noqa: E402
from marketdata.refresh import create_stable_version_from_current  # noqa: E402
from marketdata.repository import MarketDataRepository  # noqa: E402
from marketdata.sources.akshare_source import (  # noqa: E402
    load_cn_daily_bars_raw_and_qfq,
    load_cn_trading_calendar,
)

CHECKPOINT = ROOT / "data" / "market" / "checkpoints" / "refresh_cn_raw_qfq_20261102.json"
REPAIR_STATE = ROOT / "data" / "market" / "checkpoints" / "repair_cn_adjust_20261026.json"


def _fetch(symbol: str, last_date: date, last_adj: float, end: date, attempts: int = 2):
    start = max(date(2016, 1, 1), last_date - timedelta(days=30))
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            bars = load_cn_daily_bars_raw_and_qfq(symbol, start, end)
            overlap = next((b for b in bars if b.trade_date == last_date), None)
            scale = 1.0
            if overlap is not None and overlap.adj_close and last_adj:
                scale = float(last_adj) / float(overlap.adj_close)
            if not (0.5 <= scale <= 2.0):
                raise ValueError(f"suspicious qfq scale {scale:.4f}")
            scaled = []
            for b in bars:
                if b.adj_open is not None and b.adj_close is not None:
                    adj_close = b.adj_close * scale
                    b = replace(
                        b,
                        adj_open=b.adj_open * scale,
                        adj_high=b.adj_high * scale,
                        adj_low=b.adj_low * scale,
                        adj_close=adj_close,
                        adj_factor=round(adj_close / b.close, 10) if b.close > 0 else None,
                    )
                scaled.append(b)
            new_bars = [b for b in scaled if b.trade_date > last_date]
            for b in new_bars:
                if b.close <= 0 or b.adj_close is None or b.adj_close <= 0:
                    raise ValueError(f"bad price on {b.trade_date}")
                if b.high < max(b.open, b.close) - 1e-6 or b.low > min(b.open, b.close) + 1e-6:
                    raise ValueError(f"invalid OHLC on {b.trade_date}")
            return symbol, new_bars, None, scale
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < attempts:
                time.sleep(0.4 * attempt)
    return symbol, None, last_err, None


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"succeeded": [], "failed": {}, "updated_at": ""}


def _save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="A股 raw+qfq 增量刷新")
    parser.add_argument("--end", default="")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    checkpoint_path = Path(args.checkpoint)
    state = _load_state(checkpoint_path)
    succeeded = set(state.get("succeeded", []))
    failed: dict[str, str] = dict(state.get("failed", {}))

    repo = MarketDataRepository(connect())

    # 只刷新已成功修复的代码（避免把旧 qfq 当 raw 的失败代码再次混入）
    repair_ok: set[str] = set()
    if REPAIR_STATE.exists():
        try:
            repair_ok = set(json.loads(REPAIR_STATE.read_text(encoding="utf-8")).get("succeeded", []))
        except Exception:
            repair_ok = set()

    rows = repo.conn.execute(
        """
        WITH ranked AS (
            SELECT symbol, trade_date, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
            FROM daily_bars WHERE market='cn'
        )
        SELECT symbol, trade_date, adj_close FROM ranked WHERE rn = 1
        """
    ).fetchall()
    latest = {str(s): (d, a) for s, d, a in rows}

    symbols = sorted(
        s for s, (d, a) in latest.items()
        if (not repair_ok or s in repair_ok) and d < end and a is not None
    )
    if args.symbols:
        wanted = {x.strip() for x in args.symbols.split(",") if x.strip()}
        symbols = [s for s in symbols if s in wanted]
    if args.retry_failed:
        symbols = sorted(failed.keys())
    else:
        symbols = [s for s in symbols if s not in succeeded]
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"待增量刷新 {len(symbols)} 只（目标 {end}，已成功 {len(succeeded)}，失败 {len(failed)}）")
    if not symbols:
        return 0

    started = time.time()
    written = 0
    done = 0
    inflight: dict[Any, str] = {}
    iterator = iter(symbols)

    def submit_next(executor) -> bool:
        try:
            symbol = next(iterator)
        except StopIteration:
            return False
        d, a = latest.get(symbol, (None, None))
        if d is None:
            return submit_next(executor)
        inflight[executor.submit(_fetch, symbol, d, a, end)] = symbol
        return True

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for _ in range(max(1, args.workers) * 2):
            if not submit_next(executor):
                break
        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for future in finished:
                symbol = inflight.pop(future)
                sym, bars, err, scale = future.result()
                done += 1
                if err:
                    failed[sym] = err[:300]
                    print(f"[{done}/{len(symbols)}] {sym} FAIL {err[:100]}")
                elif not bars:
                    succeeded.add(sym)
                    failed.pop(sym, None)
                    print(f"[{done}/{len(symbols)}] {sym} no new bars (scale={scale:.4f})")
                else:
                    try:
                        repo.upsert_daily_bars(bars)
                        written += len(bars)
                        succeeded.add(sym)
                        failed.pop(sym, None)
                        print(f"[{done}/{len(symbols)}] {sym} +{len(bars)} bars "
                              f"{bars[0].trade_date}~{bars[-1].trade_date} scale={scale:.4f}")
                    except Exception as exc:  # noqa: BLE001
                        failed[sym] = f"write: {exc}"[:300]
                        print(f"[{done}/{len(symbols)}] {sym} WRITE FAIL {exc}")
                if done % 10 == 0 or done == len(symbols):
                    state["succeeded"] = sorted(succeeded)
                    state["failed"] = failed
                    _save_state(checkpoint_path, state)
                if done % 50 == 0:
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 1e-9)
                    eta = (len(symbols) - done) / max(rate, 1e-9)
                    print(f"  progress {done}/{len(symbols)} {rate:.2f}/s ETA {eta/60:.1f} min")
                submit_next(executor)

    state["succeeded"] = sorted(succeeded)
    state["failed"] = failed
    _save_state(checkpoint_path, state)

    try:
        cal = load_cn_trading_calendar(end - timedelta(days=10), end)
        if cal:
            n = repo.upsert_trading_calendar(cal)
            print(f"trading_calendar upserted {n} rows")
    except Exception as exc:  # noqa: BLE001
        print(f"calendar update skipped: {exc}")

    print(f"完成：写入 {written} 条，成功 {len(succeeded)}，失败 {len(failed)}，"
          f"耗时 {(time.time()-started)/60:.1f} min")
    version = create_stable_version_from_current(
        repo,
        source="akshare_raw+qfq-incremental",
        markets=("cn",),
        quality_summary={
            "incremental": True,
            "target_end": str(end),
            "bars_written": written,
            "symbols_succeeded": len(succeeded),
            "symbols_failed": len(failed),
        },
    )
    print(f"新数据集版本: {version.dataset_version}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
