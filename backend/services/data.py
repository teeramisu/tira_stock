"""
美股 / A股 / 港股历史数据：通过 yfinance 拉取。

【限流与并发】
- FastAPI 多线程处理请求，可能同时有多路调用 fetch_ohlcv（例如前端同时请求新闻+回测）。
- Yahoo 对同一 IP 有严格限流，多路并发或请求过密会触发 YFRateLimitError。
- 本模块通过以下方式控制：
  1. 内存缓存：同一 (ticker,start,end,period) 在 _CACHE_TTL 秒内直接返回，不请求 Yahoo。
  2. 磁盘缓存：同一天内同一请求从 .ohlcv_cache 读盘，不请求 Yahoo。
  3. 请求间隔 _MIN_REQUEST_INTERVAL：任意两次「开始请求」至少间隔 N 秒。
  4. 下载锁 _download_lock：同一时刻只允许一个线程执行 yf.download / Ticker().history，
     避免「间隔够了但多线程同时发起」导致的并发请求。
  5. 遇限流时等待 _RATE_LIMIT_WAIT 秒后重试，最多 _RATE_LIMIT_RETRIES 次。
  6. 备用数据源：Yahoo 仍失败且为美股时，依次尝试 Alpha Vantage（需环境变量 ALPHAVANTAGE_API_KEY）、
     Stooq（需 pip install pandas-datareader，无需 Key）。见 services/data_sources.py。

【Debug】
- 将 _DEBUG_OHLCV 设为 True 可在控制台看到：memory_cache HIT / disk_cache HIT / throttle wait / download attempt / rate limited wait / FAIL。
- 控制台里的 "1 Failed download: ['AAPL']" 是 yfinance 库内部打印的，不是本模块；本模块用 [OHLCV] 前缀区分。
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from datetime import date
from pathlib import Path

import pandas as pd
from typing import Optional, Tuple
import os

proxy = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

# ---------- 路径与缓存目录 ----------
_THIS_DIR = Path(__file__).resolve().parent
# 本地磁盘缓存目录（同一天内再次请求直接读盘，不请求 Yahoo）
_OHLCV_CACHE_DIR = _THIS_DIR.parent / ".ohlcv_cache"

# ---------- 限流与重试参数 ----------
# 任意两次「开始发起 Yahoo 请求」的最小间隔（秒）
_MIN_REQUEST_INTERVAL = 10
# 遇限流时按次数递增等待（秒）：第 1 次等 120s，第 2 次等 180s，给 Yahoo 更长时间恢复
_RATE_LIMIT_WAIT_BACKOFF = [120, 180]
# 限流后最多再重试几次（总尝试 = 1 + 此值）
_RATE_LIMIT_RETRIES = 2
# 内存缓存 TTL（秒），同一 cache_key 在此时间内直接返回
_CACHE_TTL = 600

# ---------- 调试：设为 True 时在控制台打印缓存命中/下载/限流等待 ----------
_DEBUG_OHLCV = True
_log = logging.getLogger(__name__)

# 抑制 yfinance 库的 INFO 输出，减少控制台 "1 Failed download: [...]" 刷屏（该信息来自 yfinance 内部）
try:
    logging.getLogger("yfinance").setLevel(logging.WARNING)
except Exception:
    pass

# ---------- 全局状态与锁 ----------
# 上次「开始请求」的时间（monotonic），用于 _MIN_REQUEST_INTERVAL 计算
_last_request_time = 0.0
# 锁：在「等待间隔 + 更新 _last_request_time」时使用，保证间隔计算正确
_request_lock = threading.Lock()
# 锁：整个「调用 yf.download 或 Ticker().history」期间持有，保证同一时刻只有一个 Yahoo 请求
_download_lock = threading.Lock()
# 内存缓存: cache_key -> (DataFrame, 过期时间 monotonic)
_ohlcv_cache: dict[Tuple[str, Optional[str], Optional[str], Optional[str]], Tuple[pd.DataFrame, float]] = {}
_cache_lock = threading.Lock()

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


def _disk_cache_path(cache_key: Tuple[str, Optional[str], Optional[str], Optional[str]]) -> Path:
    """缓存键 -> 本地文件路径（用 hash 避免非法文件名）。"""
    key_str = repr(cache_key)
    name = hashlib.sha256(key_str.encode()).hexdigest()[:32] + ".pkl"
    return _OHLCV_CACHE_DIR / name


def _load_disk_cache_if_today(cache_key: Tuple[str, Optional[str], Optional[str], Optional[str]]) -> Optional[pd.DataFrame]:
    """若存在且为当天写入的缓存则加载并返回，否则返回 None。"""
    path = _disk_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if date.fromtimestamp(mtime) != date.today():
            return None
        df = pd.read_pickle(path)
        if df is not None and not df.empty and "Close" in df.columns:
            return df
    except Exception:
        pass
    return None


def _save_disk_cache(cache_key: Tuple[str, Optional[str], Optional[str], Optional[str]], data: pd.DataFrame) -> None:
    """将数据写入本地缓存（当天有效）。"""
    try:
        _OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _disk_cache_path(cache_key)
        data.to_pickle(path)
    except Exception:
        pass


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
    cache_key: Tuple[str, Optional[str], Optional[str], Optional[str]] = (ticker, start, end, period)

    # ---------- 1. 内存缓存：命中则直接返回，不访问 Yahoo ----------
    with _cache_lock:
        if cache_key in _ohlcv_cache:
            cached_df, expiry = _ohlcv_cache[cache_key]
            if time.monotonic() < expiry:
                if _DEBUG_OHLCV:
                    _log.info("[OHLCV] memory_cache HIT %s", cache_key)
                return cached_df.copy()
            del _ohlcv_cache[cache_key]

    # ---------- 2. 磁盘缓存：同一天内写入的同一请求直接读盘 ----------
    disk_df = _load_disk_cache_if_today(cache_key)
    if disk_df is not None:
        if _DEBUG_OHLCV:
            _log.info("[OHLCV] disk_cache HIT %s", cache_key)
        with _cache_lock:
            _ohlcv_cache[cache_key] = (disk_df.copy(), time.monotonic() + _CACHE_TTL)
        return disk_df.copy()

    data = None
    last_err: Optional[Exception] = None

    # 等待与上次「开始请求」至少间隔 _MIN_REQUEST_INTERVAL 秒（多线程下由 _request_lock 串行化）
    def _wait_interval() -> None:
        global _last_request_time
        with _request_lock:
            now = time.monotonic()
            wait = _MIN_REQUEST_INTERVAL - (now - _last_request_time)
            if wait > 0:
                if _DEBUG_OHLCV:
                    _log.info("[OHLCV] throttle wait %.1fs (ticker=%s)", wait, ticker)
                time.sleep(wait)
            _last_request_time = time.monotonic()

    def _is_rate_limit(e: Exception) -> bool:
        if "Rate limited" in str(e) or "Too Many Requests" in str(e):
            return True
        try:
            from yfinance.exceptions import YFRateLimitError
            return type(e) is YFRateLimitError
        except ImportError:
            return False

    def _do_download() -> Optional[pd.DataFrame]:
        """实际调用 yf.download。调用方必须在持有 _download_lock 时执行，保证全局单线程访问 Yahoo。"""
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

    # ---------- 3. 方式一：yf.download；先等间隔再抢下载锁，避免多线程并发请求 Yahoo ----------
    for attempt in range(1 + _RATE_LIMIT_RETRIES):
        _wait_interval()
        if _DEBUG_OHLCV:
            _log.info("[OHLCV] download attempt %s/3 (yf.download) ticker=%s", attempt + 1, ticker)
        try:
            with _download_lock:
                data = _do_download()
            if data is not None and not data.empty:
                if _DEBUG_OHLCV:
                    _log.info("[OHLCV] download OK (yf.download) ticker=%s", ticker)
                break
        except Exception as e:
            last_err = e
            if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES:
                wait_sec = _RATE_LIMIT_WAIT_BACKOFF[min(attempt, len(_RATE_LIMIT_WAIT_BACKOFF) - 1)]
                if _DEBUG_OHLCV:
                    _log.warning("[OHLCV] rate limited, wait %ss then retry (attempt %s) ticker=%s", wait_sec, attempt + 1, ticker)
                time.sleep(wait_sec)
                continue
            if attempt == 0:
                time.sleep(2)
                continue
            break
        if (data is None or data.empty) and attempt == 0:
            time.sleep(2)

    # ---------- 4. 方式二：Ticker().history 备用；同样「间隔 + 下载锁」+ 限流重试 ----------
    if data is None or data.empty:
        for attempt in range(1 + _RATE_LIMIT_RETRIES):
            _wait_interval()
            if _DEBUG_OHLCV:
                _log.info("[OHLCV] download attempt %s/3 (Ticker.history) ticker=%s", attempt + 1, ticker)
            try:
                with _download_lock:
                    obj = yf.Ticker(ticker)
                    if start and end:
                        data = obj.history(start=start, end=end, auto_adjust=True)
                    else:
                        data = obj.history(period=period or "1y", auto_adjust=True)
                if data is not None and not data.empty:
                    if _DEBUG_OHLCV:
                        _log.info("[OHLCV] download OK (Ticker.history) ticker=%s", ticker)
                    break
            except Exception as e:
                last_err = e
                if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES:
                    wait_sec = _RATE_LIMIT_WAIT_BACKOFF[min(attempt, len(_RATE_LIMIT_WAIT_BACKOFF) - 1)]
                    if _DEBUG_OHLCV:
                        _log.warning("[OHLCV] rate limited (history), wait %ss then retry (attempt %s) ticker=%s", wait_sec, attempt + 1, ticker)
                    time.sleep(wait_sec)
                    continue
                break

    # ---------- 5. 备用数据源：Yahoo 失败且为美股时，尝试 Alpha Vantage（需 Key）或 Stooq（无需 Key） ----------
    if (data is None or data.empty) and (market or MARKET_US) == MARKET_US:
        try:
            from .data_sources import _fetch_ohlcv_alphavantage, _fetch_ohlcv_stooq, _fetch_ohlcv_fmp
        except ImportError:
            pass
        else:
            for name, fetcher in [
                ("Alpha Vantage", _fetch_ohlcv_alphavantage),
                ("Stooq", _fetch_ohlcv_stooq),
                ("FMP", _fetch_ohlcv_fmp),
            ]:
                try:
                    alt_df = fetcher(symbol, start=start, end=end, period=period)
                    if alt_df is not None and not alt_df.empty and "Close" in alt_df.columns:
                        data = _normalize_ohlcv_columns(alt_df)
                        if data is not None and not data.empty:
                            if _DEBUG_OHLCV:
                                _log.info("[OHLCV] fallback %s OK ticker=%s", name, ticker)
                            break
                except Exception:
                    continue

    # ---------- 6. 仍无数据：可能被限流或网络问题，抛错便于上层提示 ----------
    if data is None or data.empty:
        if _DEBUG_OHLCV:
            _log.warning("[OHLCV] FAIL no data ticker=%s last_err=%s", ticker, last_err)
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
    out = data.sort_index()
    with _cache_lock:
        _ohlcv_cache[cache_key] = (out.copy(), time.monotonic() + _CACHE_TTL)
    _save_disk_cache(cache_key, out)
    if _DEBUG_OHLCV:
        _log.info("[OHLCV] saved to memory + disk cache ticker=%s", ticker)
    return out
