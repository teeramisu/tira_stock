"""美股 / A股 / 港股历史数据：通过 yfinance 拉取。"""
from __future__ import annotations

import re
import time
import pandas as pd
from typing import Optional

# 每次重试前等待（秒），避免连续请求触发限流
_FETCH_DELAY = 2
# 明确遇到限流时等待（秒）再重试，Yahoo 常需较长时间恢复
_RATE_LIMIT_WAIT = 45
# 限流后最多再重试次数（总尝试 = 1 + 此值）
_RATE_LIMIT_RETRIES = 2

# 市场: us=美股, cn=A股, hk=港股
MARKET_US = "us"
MARKET_CN = "cn"
MARKET_HK = "hk"


def to_yfinance_ticker(symbol: str, market: str) -> str:
    """
    将 代码+市场 转为 yfinance 使用的 ticker。
    - 美股(us): 原样，如 AAPL
    - A股(cn): 沪 6/5/9 开头 -> .SS，深 0/3 开头 -> .SZ，如 600519.SS、000001.SZ
    - 港股(hk): 后缀 .HK，如 0700.HK、9988.HK
    """
    s = (symbol or "").strip().upper()
    m = (market or MARKET_US).strip().lower()
    if not s:
        raise ValueError("股票代码不能为空")
    if m == MARKET_US:
        return s
    if m == MARKET_HK:
        if s.endswith(".HK"):
            return s
        return s + ".HK"
    if m == MARKET_CN:
        if ".SS" in s or ".SZ" in s or ".SH" in s:
            s = re.sub(r"\.(SS|SZ|SH)$", "", s, flags=re.IGNORECASE)
        digits = re.sub(r"\D", "", s)
        if not digits:
            raise ValueError("A股代码需为数字，如 600519、000001")
        first = digits[0]
        if first in "69":
            return digits + ".SS"
        if first in "03":
            return digits + ".SZ"
        raise ValueError("A股代码：沪市 6/9 开头，深市 0/3 开头，如 600519、000001")
    return s


def _normalize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    """统一列名并确保有 Close。"""
    if data is None or data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in data.columns:
            for x in data.columns:
                if x and str(x).lower() == c.lower():
                    data = data.rename(columns={x: c})
                    break
    return data


def fetch_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "1y",
    market: Optional[str] = "us",
) -> pd.DataFrame:
    """
    拉取 OHLCV。支持美股、A股、港股。
    symbol: 美股如 AAPL/SPY；A股如 600519/000001；港股如 0700/9988。
    market: us/cn/hk。start/end 格式 YYYY-MM-DD；若不传则用 period。
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("请安装 yfinance: pip install yfinance")

    ticker = to_yfinance_ticker(symbol, market or MARKET_US)
    data = None
    last_err: Optional[Exception] = None

    def _is_rate_limit(e: Exception) -> bool:
        if "Rate limited" in str(e) or "Too Many Requests" in str(e):
            return True
        try:
            from yfinance.exceptions import YFRateLimitError
            return type(e) is YFRateLimitError
        except ImportError:
            return False

    def _download() -> Optional[pd.DataFrame]:
        if start and end:
            return yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=15,
            )
        return yf.download(
            ticker,
            period=period or "1y",
            progress=False,
            auto_adjust=True,
            threads=False,
            timeout=15,
        )

    # 方式一：yf.download；遇限流则等待后重试，最多 _RATE_LIMIT_RETRIES 次
    for attempt in range(1 + _RATE_LIMIT_RETRIES):
        try:
            data = _download()
            if data is not None and not data.empty:
                break
        except Exception as e:
            last_err = e
            if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES:
                time.sleep(_RATE_LIMIT_WAIT)
                continue
            if attempt == 0:
                time.sleep(_FETCH_DELAY)
                continue
            break
        if (data is None or data.empty) and attempt == 0:
            time.sleep(_FETCH_DELAY)

    # 方式二：Ticker().history 备用；遇限流同样等待后重试
    if data is None or data.empty:
        for attempt in range(1 + _RATE_LIMIT_RETRIES):
            try:
                obj = yf.Ticker(ticker)
                if start and end:
                    data = obj.history(start=start, end=end, auto_adjust=True)
                else:
                    data = obj.history(period=period or "1y", auto_adjust=True)
                if data is not None and not data.empty:
                    break
            except Exception as e:
                last_err = e
                if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES:
                    time.sleep(_RATE_LIMIT_WAIT)
                    continue
                break

    if data is None or data.empty:
        msg = (
            f"未获取到 {symbol}（{ticker}）的历史数据，请检查代码与市场、日期范围。"
            "若在境内访问美股数据，可能需检查网络或稍后重试。"
        )
        if last_err:
            msg += f" 详情: {last_err}"
        raise ValueError(msg)

    data = _normalize_ohlcv_columns(data)
    if "Close" not in data.columns:
        raise ValueError("下载的数据中缺少 Close 列")
    return data.sort_index()
