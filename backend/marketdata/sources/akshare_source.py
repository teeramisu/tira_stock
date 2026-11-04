"""AkShare adapter for A-share sample market data refreshes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date, timedelta
import os
from typing import Iterable

import pandas as pd

from ..models import DailyBar, Instrument, TradingCalendarDay


DEFAULT_AKSHARE_TIMEOUT_SECONDS = 30


def _call_with_timeout(fetcher, *, timeout_seconds: int = DEFAULT_AKSHARE_TIMEOUT_SECONDS):
    """Run a provider call with timeout protection."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fetcher)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise RuntimeError(f"akshare call timeout after {timeout_seconds}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def is_cn_etf_symbol(symbol: str) -> bool:
    """Heuristic: on-exchange CN ETF / fund codes (not exhaustive for all LOF)."""
    code = str(symbol).zfill(6)
    if code.startswith(
        (
            "510",
            "511",
            "512",
            "513",
            "514",
            "515",
            "516",
            "517",
            "518",
            "560",
            "561",
            "562",
            "563",
            "588",  # 科创板 ETF，如 588000
        )
    ):
        return True
    if code.startswith(("159", "16")):
        return True
    return False


def cn_etf_sina_symbol(symbol: str) -> str:
    """Map numeric ETF code to Sina quote symbol (``sh510300`` / ``sz159915``)."""
    normalized = str(symbol).zfill(6)
    prefix = "sh" if normalized.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{normalized}"


def load_cn_etf_instruments() -> list[Instrument]:
    """Load on-exchange ETF metadata from AkShare ``fund_etf_spot_em``."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share ETF refresh") from exc

    df = _call_with_timeout(ak.fund_etf_spot_em)
    code_col = "代码" if "代码" in df.columns else "code"
    name_col = "名称" if "名称" in df.columns else "name"
    if code_col not in df.columns or name_col not in df.columns:
        raise RuntimeError("Unexpected AkShare ETF spot schema")
    instruments: list[Instrument] = []
    for row in df.to_dict(orient="records"):
        symbol = str(row[code_col]).zfill(6)
        exchange = "SSE" if symbol.startswith(("5", "6")) else "SZSE"
        instruments.append(
            Instrument(
                market="cn",
                symbol=symbol,
                name=str(row[name_col]),
                exchange=exchange,
                asset_type="etf",
                currency="CNY",
                is_st=False,
                source="akshare_etf",
            )
        )
    return instruments


def load_cn_instruments(limit: int | None = None, *, include_etfs: bool = True) -> list[Instrument]:
    """Load A-share stocks and optionally on-exchange ETFs from AkShare.

    Args:
        limit: Optional maximum number of instruments for sample refreshes.
        include_etfs: When True, merge ETF master records (asset_type=etf).

    Returns:
        Instrument records.

    Raises:
        RuntimeError: If AkShare is not installed or its response schema changes.
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share refresh") from exc

    try:
        df = _call_with_timeout(ak.stock_info_a_code_name)
    except Exception:
        df = _load_cn_instruments_without_bse(ak)
    required = {"code", "name"}
    if not required.issubset(df.columns):
        raise RuntimeError("Unexpected AkShare instrument schema")
    if limit is not None:
        df = df.head(limit)
    instruments: list[Instrument] = []
    for row in df.to_dict(orient="records"):
        symbol = str(row["code"]).zfill(6)
        exchange = "SSE" if symbol.startswith(("6", "9")) else "BSE" if symbol.startswith(("8", "4")) else "SZSE"
        instruments.append(
            Instrument(
                market="cn",
                symbol=symbol,
                name=str(row["name"]),
                exchange=exchange,
                asset_type="stock",
                currency="CNY",
                is_st="ST" in str(row["name"]).upper(),
                source="akshare",
            )
        )
    if include_etfs:
        by_symbol = {item.symbol: item for item in instruments}
        for etf in load_cn_etf_instruments():
            by_symbol[etf.symbol] = etf
        instruments = list(by_symbol.values())
    return instruments


def _minimal_cn_instrument(symbol: str, *, asset_type: str = "stock") -> Instrument:
    """Build a placeholder instrument when master lookup misses a symbol."""
    normalized = str(symbol).zfill(6)
    exchange = "SSE" if normalized.startswith(("5", "6", "9")) else "SZSE"
    return Instrument(
        market="cn",
        symbol=normalized,
        name=normalized,
        exchange=exchange,
        asset_type=asset_type,
        currency="CNY",
        is_st=False,
        source="akshare_etf" if asset_type == "etf" else "akshare",
    )


def load_cn_instruments_for_symbols(symbols: Iterable[str]) -> list[Instrument]:
    """Load instrument metadata for an explicit symbol list without full-market scan."""
    normalized = [str(symbol).zfill(6) for symbol in symbols if str(symbol).strip()]
    if not normalized:
        return []
    wanted = set(normalized)
    etf_wanted = {symbol for symbol in wanted if is_cn_etf_symbol(symbol)}
    stock_wanted = wanted - etf_wanted
    out: dict[str, Instrument] = {}
    if stock_wanted:
        for item in load_cn_instruments(limit=None, include_etfs=False):
            if item.symbol in stock_wanted:
                out[item.symbol] = item
        for symbol in stock_wanted:
            out.setdefault(symbol, _minimal_cn_instrument(symbol, asset_type="stock"))
    if etf_wanted:
        for item in load_cn_etf_instruments():
            if item.symbol in etf_wanted:
                out[item.symbol] = item
        for symbol in etf_wanted:
            out.setdefault(symbol, _minimal_cn_instrument(symbol, asset_type="etf"))
    return [out[symbol] for symbol in normalized if symbol in out]


def _normalize_instrument_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize varying code/name column names."""
    col_map = {
        "证券代码": "code",
        "A股代码": "code",
        "股票代码": "code",
        "code": "code",
        "证券简称": "name",
        "A股简称": "name",
        "股票简称": "name",
        "name": "name",
    }
    out = df.rename(columns={name: col_map.get(name, name) for name in df.columns})
    required = {"code", "name"}
    if not required.issubset(out.columns):
        raise RuntimeError("Unexpected AkShare instrument schema")
    return out[list(required)].copy()


def _load_cn_instruments_without_bse(ak_module) -> pd.DataFrame:
    """Fallback instrument loader that avoids BSE endpoint dependency."""
    frames: list[pd.DataFrame] = []
    if hasattr(ak_module, "stock_info_sh_name_code"):
        frames.append(_normalize_instrument_columns(_call_with_timeout(ak_module.stock_info_sh_name_code)))
    if hasattr(ak_module, "stock_info_sz_name_code"):
        frames.append(_normalize_instrument_columns(_call_with_timeout(ak_module.stock_info_sz_name_code)))
    if not frames:
        raise RuntimeError("No fallback CN instrument loaders available")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["code"])


def load_cn_daily_bars(symbols: Iterable[str], start: date, end: date) -> list[DailyBar]:
    """Load A-share daily bars from AkShare.

    Args:
        symbols: A-share symbols.
        start: Start date.
        end: End date.

    Returns:
        Daily bar records.
    """
    bars: list[DailyBar] = []
    for symbol in symbols:
        bars.extend(load_cn_daily_bars_for_symbol(symbol, start, end))
    return bars


def load_cn_daily_bars_for_symbol(symbol: str, start: date, end: date) -> list[DailyBar]:
    """Load one CN symbol daily bars (stock or on-exchange ETF)."""
    normalized_symbol = str(symbol).zfill(6)
    if is_cn_etf_symbol(normalized_symbol):
        return load_cn_etf_daily_bars_for_symbol(normalized_symbol, start, end)
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share refresh") from exc

    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    df = _fetch_cn_daily_with_fallbacks(ak, normalized_symbol, start_s, end_s)
    if df is None or df.empty:
        return []
    return _akshare_daily_frame_to_bars(normalized_symbol, df)


def load_cn_etf_daily_bars_for_symbol(symbol: str, start: date, end: date) -> list[DailyBar]:
    """Load ETF daily bars via Eastmoney, with Sina fallback when EM is unreachable."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share ETF refresh") from exc

    normalized_symbol = str(symbol).zfill(6)
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    errors: list[str] = []

    try:
        df = _call_with_timeout(
            lambda: ak.fund_etf_hist_em(
                symbol=normalized_symbol,
                period="daily",
                start_date=start_s,
                end_date=end_s,
                adjust="qfq",
            )
        )
        if df is not None and not df.empty:
            return _akshare_daily_frame_to_bars(normalized_symbol, df, source="akshare_etf")
        errors.append("fund_etf_hist_em: empty")
    except Exception as exc:  # noqa: BLE001 - collect interface failures for diagnostics.
        errors.append(f"fund_etf_hist_em: {exc}")

    try:
        df = _call_with_timeout(lambda: ak.fund_etf_hist_sina(symbol=cn_etf_sina_symbol(normalized_symbol)))
        if df is None or df.empty:
            errors.append("fund_etf_hist_sina: empty")
        else:
            trade_dates = pd.to_datetime(df["date"], errors="coerce").dt.date
            mask = (trade_dates >= start) & (trade_dates <= end)
            filtered = df.loc[mask]
            if filtered.empty:
                errors.append("fund_etf_hist_sina: empty in range")
            else:
                return _akshare_daily_frame_to_bars(
                    normalized_symbol,
                    filtered,
                    source="akshare_etf_sina",
                )
    except Exception as exc:  # noqa: BLE001 - collect interface failures for diagnostics.
        errors.append(f"fund_etf_hist_sina: {exc}")

    if errors:
        raise RuntimeError(
            f"all akshare cn etf daily interfaces failed for {normalized_symbol}; {' | '.join(errors)}"
        )
    return []


def _sina_symbol(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith("920") or s.startswith(("4", "8")):
        return f"bj{s}"
    if s.startswith(("6", "9")):
        return f"sh{s}"
    return f"sz{s}"


def _normalize_sina_frame(df: pd.DataFrame, *, volume_divisor: float = 100.0) -> pd.DataFrame:
    """Sina 返回的成交量单位是股，统一换算成手（与东财存量数据一致）。"""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    for col in ("volume", "amount"):
        if col not in out.columns:
            out[col] = None
    if volume_divisor:
        out["volume"] = out["volume"] / volume_divisor
    return out[["date", "open", "high", "low", "close", "volume", "amount"]].sort_values("date")


def _fetch_sina_stock(symbol: str, start: date, end: date, *, adjust: str) -> pd.DataFrame | None:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share repair") from exc
    try:
        df = _call_with_timeout(
            lambda: ak.stock_zh_a_daily(
                symbol=_sina_symbol(symbol),
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust=adjust,
            )
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        out = _normalize_sina_frame(df)
    except Exception:
        return None
    return out if not out.empty else None


def _fetch_sina_etf_raw(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share repair") from exc
    try:
        df = _call_with_timeout(lambda: ak.fund_etf_hist_sina(symbol=_sina_symbol(symbol)))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        out = _normalize_sina_frame(df)
    except Exception:
        return None
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out if not out.empty else None


def _fetch_tencent_series(
    symbol: str,
    start: date,
    end: date,
    *,
    adjust: str,
) -> pd.DataFrame | None:
    """Tencent 备用：单次最多约 640 根，按日期向前分页；成交量单位已是手。"""
    import requests

    fq = adjust if adjust in ("qfq", "hfq") else ""
    key = f"{fq}day" if fq else "day"
    sina_symbol = _sina_symbol(symbol)
    frames: list[pd.DataFrame] = []
    cursor_end = end
    for _ in range(20):
        params = {
            "param": f"{sina_symbol},day,{start.isoformat()},{cursor_end.isoformat()},1000,{fq}"
        }
        try:
            resp = requests.get(
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            resp.raise_for_status()
            node = (resp.json().get("data") or {}).get(sina_symbol) or {}
        except Exception:
            break
        rows = node.get(key) or node.get("day") or []
        if not rows:
            break
        rows = [r[:6] for r in rows if isinstance(r, (list, tuple)) and len(r) >= 6]
        if not rows:
            break
        frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        for col in ("open", "close", "high", "low", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["amount"] = None
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        frames.append(frame[["date", "open", "high", "low", "close", "volume", "amount"]])
        earliest = min(frame["date"])
        if earliest <= start or len(rows) < 640:
            break
        next_end = earliest - timedelta(days=1)
        if next_end >= cursor_end:
            break
        cursor_end = next_end
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    return out


def _first_frame(*frames: pd.DataFrame | None) -> pd.DataFrame | None:
    for frame in frames:
        if frame is not None and not frame.empty:
            return frame
    return None


def _fetch_raw_series(symbol: str, start: date, end: date, *, is_etf: bool) -> pd.DataFrame | None:
    primary = (
        _fetch_sina_etf_raw(symbol, start, end)
        if is_etf
        else _fetch_sina_stock(symbol, start, end, adjust="")
    )
    return _first_frame(primary, _fetch_tencent_series(symbol, start, end, adjust=""))


def _fetch_qfq_series(symbol: str, start: date, end: date, *, is_etf: bool) -> pd.DataFrame | None:
    if is_etf:
        return _fetch_tencent_series(symbol, start, end, adjust="qfq")
    return _first_frame(
        _fetch_sina_stock(symbol, start, end, adjust="qfq"),
        _fetch_tencent_series(symbol, start, end, adjust="qfq"),
    )


def load_cn_daily_bars_raw_and_qfq(symbol: str, start: date, end: date) -> list[DailyBar]:
    """同时加载原始价与前复权价。

    - ``open/high/low/close`` 保存**未复权**真实成交价；
    - ``adj_open/.../adj_close`` 保存**前复权**价（统一以最新日期为基准）；
    - ``adj_factor = adj_close / close``，用于审计与复权还原。
    成交量统一为“手”，与东财存量数据一致。
    """
    normalized_symbol = str(symbol).zfill(6)
    is_etf = is_cn_etf_symbol(normalized_symbol)
    raw = _fetch_raw_series(normalized_symbol, start, end, is_etf=is_etf)
    qfq = _fetch_qfq_series(normalized_symbol, start, end, is_etf=is_etf)
    if raw is None and qfq is None:
        raise RuntimeError(f"no raw/qfq data for {normalized_symbol}")
    if raw is None:
        raw = qfq
    if qfq is None:
        qfq = raw

    qfq_map = {row["date"]: row for row in qfq.to_dict(orient="records")}
    bars: list[DailyBar] = []
    prev_raw_close: float | None = None
    for row in raw.to_dict(orient="records"):
        d = row["date"]
        adj = qfq_map.get(d, row)
        raw_close = float(row["close"])
        qfq_close = float(adj.get("close", raw_close))
        factor = (qfq_close / raw_close) if raw_close > 0 else None
        bars.append(
            DailyBar(
                market="cn",
                symbol=normalized_symbol,
                trade_date=d,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=raw_close,
                pre_close=prev_raw_close,
                adj_open=float(adj.get("open", row["open"])),
                adj_high=float(adj.get("high", row["high"])),
                adj_low=float(adj.get("low", row["low"])),
                adj_close=qfq_close,
                adjustment="qfq",
                volume=float(row["volume"]) if row.get("volume") is not None else None,
                amount=float(row["amount"]) if row.get("amount") is not None else None,
                adj_factor=round(factor, 10) if factor is not None else None,
                paused=False,
                source="sina_raw+qfq" if not is_etf else "sina_etf_raw+tencent_qfq",
            )
        )
        prev_raw_close = raw_close
    return bars


def _fetch_cn_daily_with_fallbacks(ak_module, symbol: str, start_s: str, end_s: str) -> pd.DataFrame | None:
    """Fetch A-share daily data via multiple AkShare interfaces, with Tencent fallback."""
    errors: list[str] = []
    call_plan = [
        (
            "stock_zh_a_hist_qfq",
            lambda: ak_module.stock_zh_a_hist(
                symbol=symbol, period="daily", start_date=start_s, end_date=end_s, adjust="qfq"
            ),
        ),
        (
            "stock_zh_a_hist_no_adjust",
            lambda: ak_module.stock_zh_a_hist(
                symbol=symbol, period="daily", start_date=start_s, end_date=end_s, adjust=""
            ),
        ),
    ]
    if hasattr(ak_module, "stock_zh_a_hist_tx"):
        call_plan.append(
            (
                "stock_zh_a_hist_tx",
                lambda: ak_module.stock_zh_a_hist_tx(symbol=symbol, start_date=start_s, end_date=end_s),
            )
        )
    if hasattr(ak_module, "stock_zh_a_daily"):
        call_plan.append(
            (
                "stock_zh_a_daily",
                lambda: ak_module.stock_zh_a_daily(symbol=symbol, start_date=start_s, end_date=end_s),
            )
        )

    for interface_name, fetcher in call_plan:
        try:
            df = _call_with_timeout(fetcher)
            if df is None or df.empty:
                errors.append(f"{interface_name}: empty")
                continue
            return _normalize_daily_columns(df)
        except Exception as exc:  # noqa: BLE001 - collect interface failures for diagnostics.
            errors.append(f"{interface_name}: {exc}")

    # Tencent finance fallback
    try:
        df = _fetch_cn_daily_tencent(symbol, start_s, end_s)
        if df is not None and not df.empty:
            return df
        errors.append("tencent: empty")
    except Exception as exc:
        errors.append(f"tencent: {exc}")

    raise RuntimeError(f"all akshare cn daily interfaces failed for {symbol}; {' | '.join(errors)}")


def _fetch_cn_daily_tencent(symbol: str, start_s: str, end_s: str) -> pd.DataFrame | None:
    """Fetch A-share daily data from Tencent Finance API.

    Tencent API returns kline data as [date, open, close, high, low, volume].
    """
    import requests

    # Determine exchange prefix
    if symbol.startswith(("6", "9")):
        tc_symbol = f"sh{symbol}"
    else:
        tc_symbol = f"sz{symbol}"

    # Convert date format: "20260703" -> "2026-07-03"
    start_fmt = f"{start_s[:4]}-{start_s[4:6]}-{start_s[6:]}"
    end_fmt = f"{end_s[:4]}-{end_s[4:6]}-{end_s[6:]}"

    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{tc_symbol},day,{start_fmt},{end_fmt},500,qfq"}
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "data" not in data:
        return None

    stock_data = data["data"].get(tc_symbol, {})
    # Try qfqday first (forward-adjusted), then day (unadjusted)
    klines = stock_data.get("qfqday") or stock_data.get("day") or []
    if not klines:
        return None

    rows = []
    for k in klines:
        if len(k) < 6:
            continue
        rows.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]) if k[5] else None,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize varying AkShare column naming conventions."""
    col_map = {
        "日期": "date",
        "date": "date",
        "Date": "date",
        "开盘": "open",
        "open": "open",
        "Open": "open",
        "收盘": "close",
        "close": "close",
        "Close": "close",
        "最高": "high",
        "high": "high",
        "High": "high",
        "最低": "low",
        "low": "low",
        "Low": "low",
        "成交量": "volume",
        "volume": "volume",
        "Volume": "volume",
        "成交额": "amount",
        "amount": "amount",
        "Amount": "amount",
    }
    out = df.rename(columns={name: col_map.get(name, name) for name in df.columns})
    required = {"date", "open", "close", "high", "low"}
    if not required.issubset(out.columns):
        raise RuntimeError("Unexpected AkShare daily schema after normalization")
    return out


def _akshare_daily_frame_to_bars(symbol: str, df: pd.DataFrame, *, source: str = "akshare") -> list[DailyBar]:
    """Convert AkShare historical daily DataFrame to DailyBar records."""
    out = _normalize_daily_columns(df)
    bars = []
    for row in out.to_dict(orient="records"):
        trade_date = pd.Timestamp(row["date"]).date()
        bars.append(
            DailyBar(
                market="cn",
                symbol=symbol,
                trade_date=trade_date,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adj_open=float(row["open"]),
                adj_high=float(row["high"]),
                adj_low=float(row["low"]),
                adj_close=float(row["close"]),
                adjustment="qfq",
                volume=float(row["volume"]) if row.get("volume") is not None else None,
                amount=float(row["amount"]) if row.get("amount") is not None else None,
                source=source,
            )
        )
    return bars


def load_cn_trading_calendar(start: date, end: date) -> list[TradingCalendarDay]:
    """Load A-share trading days in ``[start, end]`` (inclusive) from AkShare.

    优先新浪交易日历；新浪偶发被代理重置，失败时改用东方财富上证指数日线日期。
    仅返回真实交易日（``is_open=True``）。

    Args:
        start: Calendar range start.
        end: Calendar range end.

    Returns:
        ``TradingCalendarDay`` rows with ``market='cn'`` and ``is_open=True``.

    Raises:
        RuntimeError: If AkShare is missing or the response schema changes.
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for A-share calendar refresh") from exc

    # 新浪接口走代理时容易被 reset，临时移除代理环境变量直连
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    saved_proxy = {k: os.environ.pop(k, None) for k in proxy_keys}
    try:
        df = _call_with_timeout(ak.tool_trade_date_hist_sina)
    except Exception:
        df = None
    finally:
        for key, value in saved_proxy.items():
            if value is not None:
                os.environ[key] = value

    trade_dates: list[date] = []
    if df is not None and not df.empty and "trade_date" in df.columns:
        ts = pd.to_datetime(df["trade_date"], errors="coerce")
        trade_dates = [d for d in ts.dt.date.tolist() if d is not None and start <= d <= end]

    if not trade_dates:
        # 备用：上证指数日线（东方财富），其日期即交易日
        try:
            index_df = _call_with_timeout(lambda: ak.stock_zh_index_daily_em(symbol="sh000001"))
            if index_df is not None and not index_df.empty:
                col = "date" if "date" in index_df.columns else index_df.columns[0]
                ts = pd.to_datetime(index_df[col], errors="coerce")
                trade_dates = [d for d in ts.dt.date.tolist() if d is not None and start <= d <= end]
        except Exception:
            trade_dates = []

    if not trade_dates:
        return []

    unique_days = sorted(set(trade_dates))
    return [TradingCalendarDay(market="cn", trade_date=d, is_open=True) for d in unique_days]
