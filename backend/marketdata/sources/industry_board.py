"""Eastmoney / THS industry board data via AkShare.

Tries Eastmoney first; falls back to THS when Eastmoney is unreachable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .akshare_source import _call_with_timeout

_BOARD_COLUMN_MAP = {
    "排名": "rank",
    "板块名称": "name",
    "板块代码": "board_code",
    "最新价": "last_price",
    "涨跌额": "change_amount",
    "涨跌幅": "change_pct",
    "总市值": "market_cap",
    "换手率": "turnover_rate",
    "上涨家数": "up_count",
    "下跌家数": "down_count",
    "领涨股票": "leader_stock",
    "领涨股票-涨跌幅": "leader_change_pct",
}

_THS_INDUSTRY_COLUMN_MAP = {
    "序号": "rank",
    "板块": "name",
    "涨跌幅": "change_pct",
    "上涨家数": "up_count",
    "下跌家数": "down_count",
    "领涨股": "leader_stock",
    "领涨股-涨跌幅": "leader_change_pct",
    "领涨股-最新价": "leader_last_price",
    "净流入": "net_inflow",
}

_HIST_WEEK_TRADING_DAYS = 5
_HIST_MONTH_TRADING_DAYS = 22
_HIST_QUARTER_TRADING_DAYS = 66

_ths_code_cache: dict[str, dict[str, str]] = {}

# Local cache file for board constituents (code+name mapping)
_BOARD_CONSTITUENTS_CACHE_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "board_constituents.json"
)
_board_constituents_cache: dict | None = None
_BOARD_CACHE_TTL_DAYS = 7  # cache considered fresh for 7 days


def _load_board_constituents_cache(force_reload: bool = False) -> dict:
    """Load the local board constituents cache file.

    Returns empty dict if cache file doesn't exist.
    Cache is reloaded from disk after TTL or when force_reload=True.
    """
    global _board_constituents_cache
    if _board_constituents_cache is not None and not force_reload:
        return _board_constituents_cache
    if _BOARD_CONSTITUENTS_CACHE_FILE.exists():
        import json
        try:
            with open(_BOARD_CONSTITUENTS_CACHE_FILE, "r", encoding="utf-8") as f:
                _board_constituents_cache = json.load(f)
            return _board_constituents_cache
        except Exception:
            pass
    _board_constituents_cache = {}
    return _board_constituents_cache


def _is_board_cache_fresh(board_type: str, symbol: str) -> bool:
    """Check if a cached board entry is still within TTL."""
    cache = _load_board_constituents_cache()
    entry = cache.get(board_type, {}).get(symbol)
    if not entry or not entry.get("updated_at"):
        return False
    try:
        updated = datetime.strptime(entry["updated_at"], "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - updated).days < _BOARD_CACHE_TTL_DAYS
    except (ValueError, TypeError):
        return False


def invalidate_board_cache() -> None:
    """Force reload of board constituents cache from disk on next access."""
    global _board_constituents_cache
    _board_constituents_cache = None


def _cn_symbol_to_sina(code: str) -> str:
    """Convert 6-digit A-share code to Sina symbol (sh/sz prefix)."""
    code = code.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _batch_fetch_sina_realtime(codes: list[str]) -> dict[str, dict[str, Any]]:
    """Batch fetch real-time quotes from Sina API for a list of A-share codes.

    Returns: {code: {price, change_pct, change_amount, volume, market_cap, ...}}
    """
    if not codes:
        return {}
    import requests as req

    sina_symbols = [_cn_symbol_to_sina(c) for c in codes]
    # Sina supports batch query, but limit to ~50 per request
    result: dict[str, dict[str, Any]] = {}
    batch_size = 50
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    for i in range(0, len(sina_symbols), batch_size):
        batch = sina_symbols[i : i + batch_size]
        url = f"https://hq.sinajs.cn/list={','.join(batch)}"
        try:
            resp = req.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            for line in resp.text.strip().split("\n"):
                line = line.strip()
                if not line or '="' not in line:
                    continue
                # var hq_str_sh600519="贵州茅台,1849.01,..."
                var_part, data_part = line.split('=', 1)
                sina_sym = var_part.split("_")[-1]
                code = sina_sym[2:]  # strip sh/sz prefix
                fields = data_part.strip('";\n').split(",")
                if len(fields) < 10:
                    continue
                try:
                    prev_close = float(fields[2]) if fields[2] else 0
                    current = float(fields[3]) if fields[3] else 0
                    high = float(fields[4]) if fields[4] else 0
                    low = float(fields[5]) if fields[5] else 0
                    volume = float(fields[8]) if fields[8] else 0
                    amount = float(fields[9]) if fields[9] else 0
                    change_amt = round(current - prev_close, 4) if prev_close else None
                    change_pct = round(change_amt / prev_close * 100, 2) if prev_close and prev_close != 0 else None
                    result[code] = {
                        "price": current or None,
                        "change_pct": change_pct,
                        "change_amount": change_amt,
                        "volume": volume,
                        "market_cap": None,
                        "turnover_rate": None,
                        "pe_ratio": None,
                        "pb_ratio": None,
                    }
                except (ValueError, IndexError):
                    continue
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Xueqiu (雪球) real-time quotes — richer data (PE/PB/market_cap/turnover)
# ---------------------------------------------------------------------------
_xq_session: Any = None  # requests.Session with cookies
_xq_session_init_time: float = 0
_XQ_SESSION_TTL = 1800  # 30 minutes


def _get_xq_session():
    """Get or create a Xueqiu session with valid cookies."""
    import time
    import requests as req
    global _xq_session, _xq_session_init_time
    now = time.time()
    if _xq_session is not None and (now - _xq_session_init_time) < _XQ_SESSION_TTL:
        return _xq_session
    try:
        s = req.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        r = s.get("https://xueqiu.com/hq", timeout=10)
        if r.status_code == 200 and "xq_a_token" in s.cookies.get_dict():
            _xq_session = s
            _xq_session_init_time = now
            return s
    except Exception:
        pass
    _xq_session = None
    return None


def _cn_code_to_xq(code: str) -> str:
    """Convert 6-digit A-share code to Xueqiu symbol (SH/SZ prefix)."""
    code = code.zfill(6)
    if code.startswith(("6", "9")):
        return f"SH{code}"
    return f"SZ{code}"


def _batch_fetch_xq_realtime(codes: list[str]) -> dict[str, dict[str, Any]]:
    """Batch fetch real-time quotes from Xueqiu batch/quote API.

    Returns: {code: {price, change_pct, volume, market_cap, pe_ratio, pb_ratio, turnover_rate}}
    Richer than Sina: includes PE, PB, market_cap, turnover_rate.
    """
    if not codes:
        return {}
    s = _get_xq_session()
    if s is None:
        return {}

    xq_symbols = [_cn_code_to_xq(c) for c in codes]
    result: dict[str, dict[str, Any]] = {}
    batch_size = 50

    for i in range(0, len(xq_symbols), batch_size):
        batch = xq_symbols[i : i + batch_size]
        try:
            r = s.get(
                "https://stock.xueqiu.com/v5/stock/batch/quote.json",
                params={"symbol": ",".join(batch), "market": "CN"},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            data = r.json().get("data", {})
            for item in data.get("items", []):
                q = item.get("quote", {})
                symbol = q.get("symbol", "")
                code = symbol.replace("SH", "").replace("SZ", "")
                if not code:
                    continue
                result[code] = {
                    "price": q.get("current"),
                    "change_pct": q.get("percent"),
                    "change_amount": q.get("chg"),
                    "volume": q.get("volume"),
                    "market_cap": q.get("market_capital"),
                    "turnover_rate": q.get("turnover_rate"),
                    "pe_ratio": q.get("pe_ttm"),
                    "pb_ratio": q.get("pb_ttm"),
                }
        except Exception:
            continue
    return result


def _batch_fetch_realtime(codes: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch real-time quotes — Xueqiu for rich data, Sina to fill gaps.

    Strategy: always fetch both sources, merge them so that:
    - Xueqiu data preferred (has market_cap, turnover_rate)
    - Sina fills coverage gaps (Xueqiu misses some 科创板/北交所 stocks)
    """
    # Fetch from both sources
    xq_data = _batch_fetch_xq_realtime(codes)
    sina_data = _batch_fetch_sina_realtime(codes)

    if not xq_data and not sina_data:
        return {}
    if not xq_data:
        return sina_data
    if not sina_data:
        return xq_data

    # Merge: Sina as base (wider coverage), XQ overwrites where available (richer data)
    merged: dict[str, dict[str, Any]] = {}
    for code in codes:
        if code in sina_data:
            merged[code] = sina_data[code]
        if code in xq_data:
            if code in merged:
                # Overlay XQ fields onto Sina base
                merged[code].update({k: v for k, v in xq_data[code].items() if v is not None})
            else:
                merged[code] = xq_data[code]
    return merged


def _get_ths_board_code_map(board_type: str) -> dict[str, str]:
    """Return name -> THS code mapping for industry or concept boards."""
    if board_type in _ths_code_cache:
        return _ths_code_cache[board_type]
    try:
        import akshare as ak
    except ImportError:
        return {}
    func = ak.stock_board_concept_name_ths if board_type == "concept" else ak.stock_board_industry_name_ths
    try:
        df = _call_with_timeout(func, timeout_seconds=10)
        if df is not None and not df.empty:
            mapping = {}
            for row in df.to_dict(orient="records"):
                name = str(row.get("name") or "").strip()
                code = str(row.get("code") or "").strip()
                if name:
                    mapping[name] = code
            _ths_code_cache[board_type] = mapping
            return mapping
    except Exception:
        pass
    return {}


def _fetch_ths_board_constituents(ths_code: str, board_type: str = "industry") -> list[dict[str, Any]]:
    """Scrape THS board detail page for constituent stocks.

    Handles THS pages that lack proper <thead> by parsing stock links directly.
    """
    import re
    import requests as req

    if board_type == "concept":
        base_url = f"https://q.10jqka.com.cn/gn/detail/code/{ths_code}"
        full_page_size = 10
    else:
        base_url = f"https://q.10jqka.com.cn/thshy/detail/code/{ths_code}"
        full_page_size = 20

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    items: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    max_pages = 10

    # Pattern: two consecutive links per stock:
    #   <a href="...stockpage.10jqka.com.cn/600519/"...>600519</a>  (code)
    #   <a href="...stockpage.10jqka.com.cn/600519"...>贵州茅台</a> (name)
    code_link = re.compile(r'stockpage\.10jqka\.com\.cn/(\d{6})/?[^>]*>(\d{6})</a>', re.IGNORECASE)
    name_link = re.compile(r'stockpage\.10jqka\.com\.cn/(\d{6})[^>]*>([^<\d][^<]*)</a>', re.IGNORECASE)

    for page in range(1, max_pages + 1):
        url = f"{base_url}/page/{page}/"
        try:
            r = req.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                break

            # Method 1: try pd.read_html with header detection
            parsed_via_pandas = False
            try:
                dfs = pd.read_html(r.text)
                for df in dfs:
                    if "代码" in df.columns and len(df) >= 5:
                        for row in df.to_dict(orient="records"):
                            code = str(row.get("代码") or "").strip()
                            name = str(row.get("名称") or "").strip()
                            if not code or code in seen_codes:
                                continue
                            seen_codes.add(code)
                            items.append({"code": code, "name": name or code})
                        parsed_via_pandas = True
                        break
            except Exception:
                pass

            if not parsed_via_pandas:
                # Method 2: regex extraction from stock page links
                code_map = {}
                for match_code, match_text in code_link.findall(r.text):
                    code_map[match_code] = match_text.strip()
                name_map = {}
                for match_code, match_name in name_link.findall(r.text):
                    name_map[match_code] = match_name.strip()
                count = 0
                for code in code_map:
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)
                    name = name_map.get(code, code)
                    items.append({"code": code, "name": name})
                    count += 1
                if count < full_page_size - 5:
                    break
        except Exception:
            break

    return items


def _parse_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_pct(value: Any) -> float | None:
    """Parse a percentage value; auto-correct locale issues (>100 -> divide by 100)."""
    result = _parse_number(value)
    if result is not None and abs(result) > 100:
        result = result / 100
    return result


def _parse_pct_or_none(value: Any) -> float | None:
    """Same as _parse_pct but allows returning None for \"--\" etc."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text == "-" or text == "--":
        return None
    try:
        val = float(text)
        if abs(val) > 100:
            val = val / 100
        return val
    except ValueError:
        return None


def _normalize_board_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    renamed = df.rename(columns={name: _BOARD_COLUMN_MAP.get(name, name) for name in df.columns})
    rows: list[dict[str, Any]] = []
    for raw in renamed.to_dict(orient="records"):
        change_pct = _parse_pct(raw.get("change_pct"))
        if change_pct is None:
            continue
        rows.append(
            {
                "rank": int(_parse_number(raw.get("rank")) or 0),
                "name": str(raw.get("name") or "").strip(),
                "board_code": str(raw.get("board_code") or "").strip(),
                "last_price": _parse_number(raw.get("last_price")),
                "change_amount": _parse_number(raw.get("change_amount")),
                "change_pct": change_pct,
                "market_cap": _parse_number(raw.get("market_cap")),
                "turnover_rate": _parse_number(raw.get("turnover_rate")),
                "up_count": int(_parse_number(raw.get("up_count")) or 0),
                "down_count": int(_parse_number(raw.get("down_count")) or 0),
                "leader_stock": str(raw.get("leader_stock") or "").strip(),
                "leader_change_pct": _parse_pct_or_none(raw.get("leader_change_pct")),
            }
        )
    return rows


def _normalize_ths_industry_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    renamed = df.rename(columns={name: _THS_INDUSTRY_COLUMN_MAP.get(name, name) for name in df.columns})
    board_codes = _get_ths_board_code_map("industry")
    rows: list[dict[str, Any]] = []
    for raw in renamed.to_dict(orient="records"):
        change_pct = _parse_pct(raw.get("change_pct"))
        if change_pct is None:
            continue
        name = str(raw.get("name") or "").strip()
        rows.append(
            {
                "rank": int(_parse_number(raw.get("rank")) or 0),
                "name": name,
                "board_code": board_codes.get(name, ""),
                "last_price": None,
                "change_amount": None,
                "change_pct": change_pct,
                "market_cap": None,
                "turnover_rate": None,
                "up_count": int(_parse_number(raw.get("up_count")) or 0),
                "down_count": int(_parse_number(raw.get("down_count")) or 0),
                "leader_stock": str(raw.get("leader_stock") or "").strip(),
                "leader_change_pct": _parse_pct_or_none(raw.get("leader_change_pct")),
                "source": "ths",
            }
        )
    return rows


def _normalize_ths_concept_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(df.to_dict(orient="records")):
        name = str(raw.get("概念名称") or raw.get("name") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "rank": idx + 1,
                "name": name,
                "board_code": str(raw.get("code") or "").strip(),
                "last_price": None,
                "change_amount": None,
                "change_pct": None,
                "market_cap": None,
                "turnover_rate": None,
                "up_count": 0,
                "down_count": 0,
                "leader_stock": str(raw.get("龙头股") or "").strip(),
                "leader_change_pct": None,
                "source": "ths",
                "event": str(raw.get("驱动事件") or "").strip(),
            }
        )
    return rows


def _load_em_data(fetcher) -> list[dict[str, Any]] | None:
    """Try loading Eastmoney data; returns None on failure."""
    try:
        df = _call_with_timeout(fetcher)
        if df is None or df.empty:
            return None
        return _normalize_board_df(df)
    except Exception:
        return None


def load_cn_industry_board_spot() -> list[dict[str, Any]]:
    """Load A-share industry board snapshot (EM with THS fallback)."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for industry board refresh") from exc

    result = _load_em_data(ak.stock_board_industry_name_em)
    if result is not None:
        return result

    try:
        df = _call_with_timeout(ak.stock_board_industry_summary_ths)
        return _normalize_ths_industry_df(df)
    except Exception:
        return _load_em_data(ak.stock_board_industry_name_em)


def load_cn_concept_board_spot() -> list[dict[str, Any]]:
    """Load A-share concept board snapshot (EM with THS fallback)."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for concept board refresh") from exc

    result = _load_em_data(ak.stock_board_concept_name_em)
    if result is not None:
        return result

    try:
        df = _call_with_timeout(ak.stock_board_concept_summary_ths)
        return _normalize_ths_concept_df(df)
    except Exception:
        return _load_em_data(ak.stock_board_concept_name_em)


def _get_board_hist_func(board_type: str):
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for board history") from exc

    if board_type == "concept":
        return ak.stock_board_concept_hist_em, ak.stock_board_concept_index_ths
    return ak.stock_board_industry_hist_em, ak.stock_board_industry_index_ths


def get_board_historical_returns(
    board_name: str,
    board_type: str = "industry",
) -> dict[str, float | None]:
    """Fetch historical OHLCV data for a board and compute multi-period returns.

    Returns:
        dict with keys ``week``, ``month``, ``quarter`` — values in percent.
    """
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=120)).strftime("%Y%m%d")

    em_func, ths_func = _get_board_hist_func(board_type)

    df = None
    for hist_func in [em_func, ths_func]:
        try:
            df = _call_with_timeout(
                lambda fn=hist_func: fn(symbol=board_name, period="日k", start_date=start_date, end_date=end_date),
                timeout_seconds=15,
            )
            if df is not None and not df.empty:
                break
        except Exception:
            continue

    if df is None or df.empty:
        if board_type == "industry":
            return {"week": None, "month": None, "quarter": None}
        return {"week": None, "month": None, "quarter": None}

    close_col = "收盘" if "收盘" in df.columns else ("收盘价" if "收盘价" in df.columns else None)
    if close_col is None:
        return {"week": None, "month": None, "quarter": None}

    closes = pd.to_numeric(df[close_col], errors="coerce").dropna().values
    n = len(closes)
    if n == 0:
        return {"week": None, "month": None, "quarter": None}

    latest = float(closes[-1])

    def _ret(idx: int) -> float | None:
        if n <= idx or closes[idx] == 0:
            return None
        return round((latest / float(closes[idx]) - 1) * 100, 2)

    return {
        "week": _ret(max(0, n - 1 - _HIST_WEEK_TRADING_DAYS)),
        "month": _ret(max(0, n - 1 - _HIST_MONTH_TRADING_DAYS)),
        "quarter": _ret(max(0, n - 1 - _HIST_QUARTER_TRADING_DAYS)),
    }


_CONSTITUENT_COL_MAP = {
    "代码": "code",
    "名称": "name",
    "code": "code",
    "name": "name",
    "最新价": "price",
    "涨跌幅": "change_pct",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
    "总市值": "market_cap",
    "流通市值": "float_market_cap",
    "市盈率-动态": "pe_ratio",
    "市净率": "pb_ratio",
    "60日涨跌幅": "change_pct_60d",
    "年初至今涨跌幅": "change_pct_ytd",
}


def _parse_constituent_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single constituent stock row into a normalized dict."""
    code = str(raw.get("code") or "").strip()
    if not code:
        return None
    name = str(raw.get("name") or "").strip()
    price = _parse_number(raw.get("price"))
    change_pct = _parse_pct(raw.get("change_pct"))
    change_amount = _parse_number(raw.get("change_amount"))
    turnover_rate = _parse_number(raw.get("turnover_rate"))
    market_cap = _parse_number(raw.get("market_cap"))
    float_market_cap = _parse_number(raw.get("float_market_cap"))
    pe_ratio = _parse_number(raw.get("pe_ratio"))
    pb_ratio = _parse_number(raw.get("pb_ratio"))
    change_pct_60d = _parse_pct(raw.get("change_pct_60d"))
    change_pct_ytd = _parse_pct(raw.get("change_pct_ytd"))
    return {
        "code": code,
        "name": name or code,
        "price": price,
        "change_pct": change_pct,
        "change_amount": change_amount,
        "turnover_rate": turnover_rate,
        "market_cap": market_cap,
        "float_market_cap": float_market_cap,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "change_pct_60d": change_pct_60d,
        "change_pct_ytd": change_pct_ytd,
    }


def get_board_constituents(
    symbol: str,
    board_type: str = "industry",
) -> list[dict[str, Any]]:
    """Fetch constituent stocks of a board (by name or board_code).

    Priority: local cache (fast) -> Eastmoney -> THS scraping.
    Always enriches with Sina real-time quotes for price data.
    """
    # --- Layer 1: Local cache (instant, no network) ---
    cache = _load_board_constituents_cache()
    type_cache = cache.get(board_type, {})
    cached_entry = type_cache.get(symbol)
    if cached_entry and cached_entry.get("stocks") and _is_board_cache_fresh(board_type, symbol):
        items = list(cached_entry["stocks"])  # shallow copy
        # Enrich with real-time quotes (Xueqiu first, Sina fallback)
        codes = [item["code"] for item in items if item.get("code")]
        if codes:
            quotes = _batch_fetch_realtime(codes)
            for item in items:
                code = item.get("code", "")
                if code in quotes:
                    item.update({k: v for k, v in quotes[code].items() if v is not None})
        return items

    # --- Layer 2: Eastmoney API ---
    try:
        import akshare as ak
    except ImportError:
        ak = None

    if ak is not None:
        cons_func = (
            ak.stock_board_concept_cons_em
            if board_type == "concept"
            else ak.stock_board_industry_cons_em
        )
        try:
            df = _call_with_timeout(lambda fn=cons_func: fn(symbol=symbol), timeout_seconds=15)
            if df is not None and not df.empty:
                renamed = df.rename(columns={c: _CONSTITUENT_COL_MAP.get(c, c) for c in df.columns})
                items = []
                for raw in renamed.to_dict(orient="records"):
                    parsed = _parse_constituent_row(raw)
                    if parsed:
                        items.append(parsed)
                if items:
                    return items
        except Exception:
            pass

    # --- Layer 3: THS scraping + Sina enrichment ---
    board_codes = _get_ths_board_code_map(board_type)
    ths_code = board_codes.get(symbol, "")
    if not ths_code:
        for name, code in board_codes.items():
            if symbol in name or name in symbol:
                ths_code = code
                break

    if ths_code:
        items = _fetch_ths_board_constituents(ths_code, board_type)
        if items:
            codes = [item["code"] for item in items if item.get("code")]
            if codes:
                quotes = _batch_fetch_realtime(codes)
                for item in items:
                    code = item.get("code", "")
                    if code in quotes:
                        item.update({k: v for k, v in quotes[code].items() if v is not None})
            return items

    return []


def enrich_boards_with_historical(
    boards: list[dict[str, Any]],
    board_type: str = "industry",
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    """Enrich a list of board dicts with multi-period historical returns.

    Fetches historical data for each board in parallel and adds
    ``change_pct_1w``, ``change_pct_1m``, ``change_pct_3m`` fields.
    """
    if not boards:
        return boards

    def _fetch(idx: int, name: str) -> tuple[int, dict[str, float | None]]:
        try:
            return idx, get_board_historical_returns(name, board_type)
        except Exception:
            return idx, {"week": None, "month": None, "quarter": None}

    results: dict[int, dict[str, float | None]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch, i, b.get("name", "")): i
            for i, b in enumerate(boards)
        }
        for future in as_completed(futures):
            try:
                idx, ret = future.result(timeout=20)
                results[idx] = ret
            except Exception:
                pass

    for i, board in enumerate(boards):
        hist = results.get(i, {"week": None, "month": None, "quarter": None})
        board["change_pct_1w"] = hist.get("week")
        board["change_pct_1m"] = hist.get("month")
        board["change_pct_3m"] = hist.get("quarter")

    return boards
