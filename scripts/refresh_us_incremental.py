#!/usr/bin/env python3
"""美股增量刷新（原始价 + 前复权价，按上次复权基准对齐）。

- 股票池：data/market/us_liquid_universe.txt（ADV >= $1M）
- 数据源：akshare stock_us_hist，自动探测 105(NASDAQ)/106(NYSE)/107(AMEX/ETF)
- 存储：close/open/high/low=未复权；adj_*=前复权；adj_factor=adj_close/close
- 增量：从库内最后交易日往前 30 天取重叠，scale 对齐旧 adj 基准，只追加新行
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from marketdata.db import connect  # noqa: E402
from marketdata.refresh import create_stable_version_from_current  # noqa: E402
from marketdata.repository import MarketDataRepository  # noqa: E402
from marketdata.models import DailyBar  # noqa: E402

UNIVERSE = ROOT / "data" / "market" / "us_liquid_universe.txt"
CHECKPOINT = ROOT / "data" / "market" / "checkpoints" / "refresh_us_raw_qfq_20261102.json"
PREFIXES = ("105", "106", "107")
_PREFIX_CACHE: dict[str, str] = {}


def _normalize(df) -> pd.DataFrame:
    colmap = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount",
    }
    out = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    return out


def _fetch_series(symbol: str, start: date, end: date, adjust: str):
    import akshare as ak

    candidates = []
    cached = _PREFIX_CACHE.get(symbol)
    if cached:
        candidates = [cached]
    candidates += [p for p in PREFIXES if p not in candidates]
    for prefix in candidates:
        try:
            df = ak.stock_us_hist(
                symbol=f"{prefix}.{symbol}",
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust=adjust,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        frame = _normalize(df)
        if frame.empty:
            continue
        _PREFIX_CACHE[symbol] = prefix
        return frame
    return None


def _fetch(symbol: str, last_date: date, last_adj: float, end: date, attempts: int = 2):
    start = max(date(2016, 1, 1), last_date - timedelta(days=30))
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            raw = _fetch_series(symbol, start, end, "")
            if raw is None:
                raise RuntimeError("no raw data")
            qfq = _fetch_series(symbol, start, end, "qfq")
            qmap = {r["date"]: r for r in qfq.to_dict("records")} if qfq is not None else {}
            bars: list[DailyBar] = []
            prev_close = None
            for row in raw.to_dict("records"):
                d = row["date"]
                adj = qmap.get(d, row)
                raw_close = float(row["close"])
                adj_close = float(adj.get("close", raw_close))
                bars.append(DailyBar(
                    market="us",
                    symbol=symbol,
                    trade_date=d,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=raw_close,
                    pre_close=prev_close,
                    adj_open=float(adj.get("open", row["open"])),
                    adj_high=float(adj.get("high", row["high"])),
                    adj_low=float(adj.get("low", row["low"])),
                    adj_close=adj_close,
                    adjustment="qfq",
                    volume=float(row["volume"]) if row.get("volume") is not None else None,
                    amount=float(row["amount"]) if row.get("amount") is not None else None,
                    adj_factor=round(adj_close / raw_close, 10) if raw_close > 0 else None,
                    source="akshare_raw+qfq",
                ))
                prev_close = raw_close
            overlap = next((b for b in bars if b.trade_date == last_date), None)
            scale = 1.0
            if overlap is not None and overlap.adj_close and last_adj:
                scale = float(last_adj) / float(overlap.adj_close)
            if not (0.3 <= scale <= 3.0):
                raise ValueError(f"suspicious qfq scale {scale:.4f}")
            scaled = []
            for b in bars:
                if b.adj_open is not None and b.adj_close is not None:
                    ac = b.adj_close * scale
                    b = replace(
                        b,
                        adj_open=b.adj_open * scale,
                        adj_high=b.adj_high * scale,
                        adj_low=b.adj_low * scale,
                        adj_close=ac,
                        adj_factor=round(ac / b.close, 10) if b.close > 0 else None,
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
    parser = argparse.ArgumentParser(description="US raw+qfq incremental refresh")
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

    universe = [line.strip().upper() for line in UNIVERSE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.symbols:
        wanted = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
        universe = [s for s in universe if s in wanted]
    if args.retry_failed:
        symbols = sorted(failed.keys())
    else:
        symbols = [s for s in universe if s not in succeeded]

    repo = MarketDataRepository(connect())
    rows = repo.conn.execute(
        """
        WITH ranked AS (
            SELECT symbol, trade_date, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
            FROM daily_bars WHERE market='us'
        )
        SELECT symbol, trade_date, adj_close FROM ranked WHERE rn = 1
        """
    ).fetchall()
    latest = {str(s): (d, a) for s, d, a in rows}
    symbols = [s for s in symbols if latest.get(s, (None, None))[0] is not None and latest[s][0] < end]
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"待刷新 {len(symbols)} 只（目标 {end}，已成功 {len(succeeded)}，失败 {len(failed)}）")
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
        d, a = latest[symbol]
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
                    print(f"[{done}/{len(symbols)}] {sym} no new bars (scale={scale})")
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
                if done % 100 == 0:
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 1e-9)
                    eta = (len(symbols) - done) / max(rate, 1e-9)
                    print(f"  progress {done}/{len(symbols)} {rate:.2f}/s ETA {eta/60:.1f} min")
                submit_next(executor)

    state["succeeded"] = sorted(succeeded)
    state["failed"] = failed
    _save_state(checkpoint_path, state)
    print(f"完成：写入 {written} 条，成功 {len(succeeded)}，失败 {len(failed)}，"
          f"耗时 {(time.time()-started)/60:.1f} min")
    version = create_stable_version_from_current(
        repo,
        source="akshare_raw+qfq-incremental",
        markets=("us",),
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
