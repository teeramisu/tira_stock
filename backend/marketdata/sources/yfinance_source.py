"""yfinance adapter for small US market data samples."""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd

from ..models import DailyBar, Instrument, IntradayBar, OptionChainSnapshot

_US_YF_EXCHANGE_MAP = {
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "NYSE": "NYSE",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "ASE": "AMEX",
    "AMX": "AMEX",
    "AMEX": "AMEX",
}


def normalize_us_exchange(raw: str | None) -> str:
    """Map provider exchange codes to NYSE/NASDAQ/AMEX."""
    if not raw:
        return "UNKNOWN"
    token = str(raw).strip().upper()
    if token in {"NYSE", "NASDAQ", "AMEX"}:
        return token
    mapped = _US_YF_EXCHANGE_MAP.get(token)
    if mapped:
        return mapped
    if "NASDAQ" in token:
        return "NASDAQ"
    if "NYSE" in token or "NEW YORK" in token:
        return "NYSE"
    if "AMEX" in token or "AMERICAN" in token:
        return "AMEX"
    return "UNKNOWN"


def _load_us_exchange_map_akshare(symbols: list[str]) -> dict[str, str]:
    """Build symbol->exchange map from akshare US spot table when available."""
    if not symbols:
        return {}
    try:
        import akshare as ak
    except ImportError:
        return {}
    try:
        df = ak.stock_us_spot()
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    symbol_col = _guess_symbol_column(df)
    if symbol_col is None:
        return {}
    exchange_col = None
    for preferred in ("exchange", "Exchange", "市场", "market", "上市地点"):
        if preferred in df.columns:
            exchange_col = preferred
            break
    wanted = {symbol.upper() for symbol in symbols}
    out: dict[str, str] = {}
    for row in df.to_dict(orient="records"):
        raw_symbol = str(row.get(symbol_col, "")).strip().upper()
        if "." in raw_symbol:
            raw_symbol = raw_symbol.split(".")[-1]
        if raw_symbol not in wanted:
            continue
        raw_exchange = str(row.get(exchange_col, "")).strip() if exchange_col else ""
        out[raw_symbol] = normalize_us_exchange(raw_exchange)
    return out


def _load_us_exchange_map_yfinance(symbols: list[str], *, chunk_size: int = 50) -> dict[str, str]:
    """Resolve US exchanges from yfinance metadata in batches."""
    if not symbols:
        return {}
    try:
        import yfinance as yf
    except ImportError:
        return {}
    out: dict[str, str] = {}
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start : start + chunk_size]
        try:
            tickers = yf.Tickers(" ".join(chunk))
        except Exception:
            continue
        for symbol in chunk:
            try:
                info = tickers.tickers[symbol].info or {}
            except Exception:
                info = {}
            raw_exchange = info.get("exchange") or info.get("fullExchangeName")
            out[symbol] = normalize_us_exchange(str(raw_exchange) if raw_exchange else None)
    return out


def load_us_instruments(symbols: Iterable[str]) -> list[Instrument]:
    """Build US instrument metadata for requested symbols."""
    symbol_list = sorted({str(symbol).upper() for symbol in symbols})
    if not symbol_list:
        return []

    exchange_by_symbol = _load_us_exchange_map_akshare(symbol_list)
    missing = [symbol for symbol in symbol_list if symbol not in exchange_by_symbol]
    if missing and len(missing) <= 500:
        exchange_by_symbol.update(_load_us_exchange_map_yfinance(missing))

    instruments: list[Instrument] = []
    for symbol in symbol_list:
        instruments.append(
            Instrument(
                market="us",
                symbol=symbol,
                name=symbol,
                exchange=exchange_by_symbol.get(symbol, "UNKNOWN"),
                currency="USD",
                source="yfinance",
            )
        )
    return instruments


def load_us_symbols_akshare(*, limit: int | None = None) -> list[str]:
    """Load US symbols from akshare spot endpoint."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for US symbol discovery") from exc
    try:
        df = ak.stock_us_spot()
    except Exception as exc:  # noqa: BLE001 - caller decides fallback strategy.
        raise RuntimeError("failed to load US symbols from akshare.stock_us_spot") from exc
    if df is None or df.empty:
        return []
    symbol_col = _guess_symbol_column(df)
    if symbol_col is None:
        return []
    out: list[str] = []
    for raw in df[symbol_col].astype(str).tolist():
        token = raw.strip().upper()
        if "." in token:
            token = token.split(".")[-1]
        if re.match(r"^[A-Z][A-Z0-9.\-]{0,12}$", token):
            out.append(token)
    deduped = sorted(set(out))
    if limit is not None and limit > 0:
        return deduped[:limit]
    return deduped


def _guess_symbol_column(df: pd.DataFrame) -> str | None:
    """Guess symbol-like column from provider DataFrame."""
    for preferred in ("symbol", "Symbol", "代码", "code", "ticker", "Ticker"):
        if preferred in df.columns:
            return preferred
    for col in df.columns:
        sample = df[col].astype(str).head(200)
        if sample.empty:
            continue
        score = sum(1 for val in sample if re.match(r"^(\d+\.)?[A-Za-z][A-Za-z0-9.\-]{0,12}$", val.strip()))
        if score >= max(5, int(len(sample) * 0.2)):
            return str(col)
    return None


def load_us_daily_bars(
    symbols: Iterable[str],
    start: date,
    end: date,
    *,
    provider: str = "yfinance",
) -> list[DailyBar]:
    """Load US daily bars from provider (yfinance/stooq/auto)."""
    provider_norm = provider.strip().lower()
    if provider_norm not in {"yfinance", "stooq", "akshare", "auto"}:
        raise ValueError(f"Unsupported US provider: {provider}")
    if provider_norm == "stooq":
        bars: list[DailyBar] = []
        for symbol in symbols:
            bars.extend(load_us_daily_bars_stooq(str(symbol).upper(), start, end))
        return bars
    if provider_norm == "akshare":
        bars = []
        for symbol in symbols:
            bars.extend(load_us_daily_bars_akshare(str(symbol).upper(), start, end))
        return bars
    if provider_norm == "auto":
        bars = []
        for symbol in symbols:
            symbol_norm = str(symbol).upper()
            fetched = _load_us_daily_bars_yfinance(symbol_norm, start, end)
            if not fetched:
                fetched = load_us_daily_bars_stooq(symbol_norm, start, end)
            if not fetched:
                fetched = load_us_daily_bars_akshare(symbol_norm, start, end)
            bars.extend(fetched)
        return bars
    bars = []
    for symbol in symbols:
        bars.extend(_load_us_daily_bars_yfinance(str(symbol).upper(), start, end))
    return bars


def _load_us_daily_bars_yfinance(symbol: str, start: date, end: date) -> list[DailyBar]:
    """Load one symbol from yfinance; return empty list on provider failure."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for US refresh") from exc
    try:
        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=20,
        )
    except Exception:  # noqa: BLE001 - provider/network failures should fallback upstream.
        return []
    if df is None or df.empty:
        return []
    return _yfinance_daily_frame_to_bars(symbol, df)


def _yfinance_daily_frame_to_bars(symbol: str, df: pd.DataFrame) -> list[DailyBar]:
    """Convert yfinance daily DataFrame to DailyBar records.

    Handles adj_close correctly:
    - yFinance Adj Close accounts for both splits and dividends
    - For splits: all OHLC should be adjusted equally (current approach is correct)
    - For dividends: only Close is affected (Open/High/Low should NOT be adjusted)
    - Since we can't distinguish split vs dividend from Adj Close alone,
      we store raw OHLC for fills and adj_close only for return calculations.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    bars = []
    for row in df.reset_index().to_dict(orient="records"):
        trade_date = pd.Timestamp(row.get("Date") or row.get("index")).date()
        adj_close = row.get("Adj Close", row.get("Close"))
        adj_factor = float(adj_close) / float(row["Close"]) if float(row["Close"]) else 1.0
        bars.append(
            DailyBar(
                market="us",
                symbol=symbol,
                trade_date=trade_date,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                # adj_open/high/low use raw values (dividends only adjust Close)
                adj_open=float(row["Open"]),
                adj_high=float(row["High"]),
                adj_low=float(row["Low"]),
                adj_close=float(adj_close),
                adjustment="adj_close",
                adj_factor=adj_factor,
                source="yfinance",
            )
        )
    return bars


def load_us_daily_bars_stooq(symbol: str, start: date, end: date) -> list[DailyBar]:
    """Load one US symbol daily bars from Stooq CSV endpoint."""
    params = urlencode({"s": f"{symbol.lower()}.us", "i": "d"})
    url = f"https://stooq.com/q/d/l/?{params}"
    try:
        df = pd.read_csv(url)
    except Exception:  # noqa: BLE001 - network/provider failures handled by caller.
        return []
    if df is None or df.empty:
        return []
    if "Date" not in df.columns:
        return []
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]
    if df.empty:
        return []
    mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
    df = df[mask]
    if df.empty:
        return []
    bars: list[DailyBar] = []
    for row in df.to_dict(orient="records"):
        close = float(row["Close"])
        bars.append(
            DailyBar(
                market="us",
                symbol=symbol,
                trade_date=pd.Timestamp(row["Date"]).date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=close,
                adj_open=float(row["Open"]),
                adj_high=float(row["High"]),
                adj_low=float(row["Low"]),
                adj_close=close,
                adjustment="none",
                volume=float(row["Volume"]) if row.get("Volume") is not None else None,
                adj_factor=1.0,
                source="stooq",
            )
        )
    return bars


def load_us_daily_bars_akshare(symbol: str, start: date, end: date) -> list[DailyBar]:
    """Load one US symbol daily bars from akshare.stock_us_daily."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for US akshare provider") from exc
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="")
    except Exception:  # noqa: BLE001 - provider/network failures handled by caller.
        return []
    if df is None or df.empty:
        return []
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()]
    mask = (frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)
    frame = frame.loc[mask]
    if frame.empty:
        return []
    bars: list[DailyBar] = []
    for row in frame.to_dict(orient="records"):
        close = float(row["close"])
        bars.append(
            DailyBar(
                market="us",
                symbol=symbol,
                trade_date=pd.Timestamp(row["date"]).date(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=close,
                adj_open=float(row["open"]),
                adj_high=float(row["high"]),
                adj_low=float(row["low"]),
                adj_close=close,
                adjustment="none",
                volume=float(row["volume"]) if row.get("volume") is not None else None,
                adj_factor=1.0,
                source="akshare",
            )
        )
    return bars


def load_us_option_chain_snapshots(
    underlying: str,
    *,
    snapshot_date: date | None = None,
    max_expirations: int | None = None,
    timeout_seconds: float = 45.0,
) -> list[OptionChainSnapshot]:
    """Load full listed option chain for one US underlying via Yahoo Finance.

    Yahoo returns *current* chain snapshots for each expiry; historical
    chain-as-of a past date is not supported on the free endpoint.

    Args:
        underlying: Equity ticker, e.g. ``AAPL``.
        snapshot_date: Logical snapshot date (typically ``date.today()``).
        max_expirations: Optional cap on number of expiry dates (nearest first).
        timeout_seconds: Wall-clock bound for the entire underlying fetch.

    Returns:
        Flattened ``OptionChainSnapshot`` rows for calls and puts.

    Raises:
        RuntimeError: If yfinance is missing or the fetch times out.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for US options refresh") from exc

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    snap = snapshot_date or date.today()
    symbol = str(underlying).upper()

    def _fetch() -> list[OptionChainSnapshot]:
        ticker = yf.Ticker(symbol)
        expirations = list(ticker.options or ())
        if not expirations:
            return []
        if max_expirations is not None and max_expirations > 0:
            expirations = expirations[:max_expirations]
        rows: list[OptionChainSnapshot] = []
        for exp in expirations:
            chain = ticker.option_chain(exp)
            exp_date = pd.Timestamp(exp).date()
            rows.extend(_option_chain_side_to_snapshots(symbol, exp_date, "call", chain.calls, snap))
            rows.extend(_option_chain_side_to_snapshots(symbol, exp_date, "put", chain.puts, snap))
        return rows

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise RuntimeError(f"yfinance option chain timeout after {timeout_seconds}s for {symbol}") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _option_chain_side_to_snapshots(
    underlying: str,
    expiry: date,
    contract_right: str,
    frame: pd.DataFrame,
    snapshot_date: date,
) -> list[OptionChainSnapshot]:
    """Map one calls or puts DataFrame to snapshot rows."""
    if frame is None or frame.empty:
        return []
    rows: list[OptionChainSnapshot] = []
    contract_col = "contractSymbol" if "contractSymbol" in frame.columns else None
    if contract_col is None:
        for preferred in ("contract_symbol", "symbol"):
            if preferred in frame.columns:
                contract_col = preferred
                break
    if contract_col is None:
        return []

    def _f(row: dict[str, object], key: str) -> float | None:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _b(row: dict[str, object], key: str) -> bool | None:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return bool(val)

    for raw in frame.to_dict(orient="records"):
        cid = str(raw.get(contract_col, "")).strip()
        if not cid:
            continue
        strike = raw.get("strike")
        if strike is None or (isinstance(strike, float) and pd.isna(strike)):
            continue
        rows.append(
            OptionChainSnapshot(
                market="us",
                snapshot_date=snapshot_date,
                underlying=underlying,
                expiry=expiry,
                strike=float(strike),
                contract_right=contract_right,
                contract_symbol=cid,
                last_price=_f(raw, "lastPrice"),
                bid=_f(raw, "bid"),
                ask=_f(raw, "ask"),
                volume=_f(raw, "volume"),
                open_interest=_f(raw, "openInterest"),
                implied_volatility=_f(raw, "impliedVolatility"),
                in_the_money=_b(raw, "inTheMoney"),
                currency="USD",
                source="yfinance",
            )
        )
    return rows


def load_us_intraday_bars(
    symbols: Iterable[str],
    *,
    start_time: datetime,
    end_time: datetime,
    frequency: str,
) -> list[IntradayBar]:
    """Load US intraday bars from yfinance for requested symbols."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for US refresh") from exc
    if frequency not in {"1m", "5m", "15m", "30m", "60m"}:
        raise ValueError(f"Unsupported intraday frequency: {frequency}")

    bars: list[IntradayBar] = []
    for symbol in symbols:
        df = yf.download(
            str(symbol).upper(),
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
            interval=frequency,
            timeout=20,
        )
        if df is None or df.empty:
            continue
        bars.extend(_yfinance_intraday_frame_to_bars(str(symbol).upper(), df, frequency=frequency))
    return bars


def _yfinance_intraday_frame_to_bars(symbol: str, df: pd.DataFrame, *, frequency: str) -> list[IntradayBar]:
    """Convert yfinance intraday DataFrame to IntradayBar records."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    bars = []
    for row in df.reset_index().to_dict(orient="records"):
        bar_time = pd.Timestamp(row.get("Datetime") or row.get("Date") or row.get("index")).to_pydatetime()
        bars.append(
            IntradayBar(
                market="us",
                symbol=symbol,
                bar_time=bar_time,
                trade_date=bar_time.date(),
                frequency=frequency,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]) if row.get("Volume") is not None else None,
                source="yfinance",
            )
        )
    return bars

