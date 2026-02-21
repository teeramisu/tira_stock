"""
备用数据源：当 Yahoo (yfinance) 限流或失败时可选使用（仅美股）。

已接入（按 fallback 顺序）：
- Alpha Vantage：ALPHAVANTAGE_API_KEY，免费约 5 次/分钟、500 次/天。
  申请：https://www.alphavantage.co/support/#api-key
- Stooq：无需 Key，pandas_datareader，pip install pandas-datareader。
- Financial Modeling Prep (FMP)：FMP_API_KEY，免费档有每日限额。
  申请：https://site.financialmodelingprep.com/developer/docs/dashboard

更多可选数据源见 docs/DATA_SOURCES.md。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def _fetch_ohlcv_alphavantage(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "1y",
) -> Optional[pd.DataFrame]:
    """
    通过 Alpha Vantage TIME_SERIES_DAILY 拉取美股日线。
    免费版 outputsize=compact 仅返回最近约 100 个交易日；无 start/end 参数，需在本地按日期切片。
    返回与 fetch_ohlcv 一致的 DataFrame（索引为日期，列含 Open/High/Low/Close/Volume），失败返回 None。
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        return None
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None
    try:
        import requests
    except ImportError:
        return None
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY"
        "&symbol=" + symbol +
        "&outputsize=compact"
        "&apikey=" + api_key
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return None
    series = j.get("Time Series (Daily)")
    if not series or not isinstance(series, dict):
        return None
    rows = []
    for date_str, v in series.items():
        if not isinstance(v, dict):
            continue
        try:
            rows.append({
                "Date": date_str,
                "Open": float(v.get("1. open", 0)),
                "High": float(v.get("2. high", 0)),
                "Low": float(v.get("3. low", 0)),
                "Close": float(v.get("4. close", 0)),
                "Volume": int(float(v.get("5. volume", 0))),
            })
        except (TypeError, ValueError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if start or end:
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        return None
    return df


def _fetch_ohlcv_stooq(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "1y",
) -> Optional[pd.DataFrame]:
    """
    通过 pandas_datareader 的 Stooq 数据源拉取美股日线（无需 API Key）。
    Stooq 美股代码格式为 SYMBOL.US，如 AAPL.US。
    返回与 fetch_ohlcv 一致的 DataFrame，失败返回 None。
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None
    try:
        import pandas_datareader as pdr
    except ImportError:
        return None
    # Stooq 美股：后缀 .US
    stooq_symbol = symbol if symbol.endswith(".US") else f"{symbol}.US"
    end_dt = datetime.now()
    if end:
        try:
            end_dt = datetime.strptime(end[:10], "%Y-%m-%d")
        except ValueError:
            pass
    if start:
        try:
            start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
        except ValueError:
            start_dt = end_dt - timedelta(days=365)
    else:
        period = (period or "1y").strip().lower()
        days = 365
        if period.endswith("y"):
            try:
                days = int(period.replace("y", "")) * 365
            except ValueError:
                pass
        elif period.endswith("mo"):
            try:
                days = int(period.replace("mo", "")) * 30
            except ValueError:
                pass
        start_dt = end_dt - timedelta(days=min(days, 365 * 10))
    try:
        df = pdr.DataReader(stooq_symbol, "stooq", start=start_dt, end=end_dt)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        return None
    return df


def _fetch_ohlcv_fmp(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "1y",
) -> Optional[pd.DataFrame]:
    """
    通过 Financial Modeling Prep API 拉取美股日线。
    需设置环境变量 FMP_API_KEY，免费档有请求限制。
    接口：historical-price-full，返回与 fetch_ohlcv 一致的 DataFrame。
    """
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        return None
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None
    try:
        import requests
    except ImportError:
        return None
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception:
        return None
    hist = j.get("historical")
    if not hist or not isinstance(hist, list):
        return None
    rows = []
    for row in hist:
        if not isinstance(row, dict):
            continue
        try:
            rows.append({
                "Date": row.get("date"),
                "Open": float(row.get("open", 0)),
                "High": float(row.get("high", 0)),
                "Low": float(row.get("low", 0)),
                "Close": float(row.get("close", 0)),
                "Volume": int(float(row.get("volume", 0))),
            })
        except (TypeError, ValueError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        return None
    return df
