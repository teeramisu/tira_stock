"""Stock symbol search with pinyin initial support.

Supports:
- Code prefix match: "600" -> "600519 贵州茅台"
- Name substring: "茅台" -> "600519 贵州茅台"
- Pinyin initials: "gzmt" -> "600519 贵州茅台"
"""
from __future__ import annotations

import threading
import time
from typing import Any

from marketdata.db import DEFAULT_MARKET_DB_PATH, connect as connect_market_db
from marketdata.repository import MarketDataRepository

_CACHE_TTL = 600  # 10 minutes
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_cache_lock = threading.Lock()

# Pinyin initials cache: symbol -> pinyin initials string
_pinyin_cache: dict[str, str] = {}
_pinyin_lock = threading.Lock()


def _get_pinyin_initials(text: str) -> str:
    """Get the first letter of each character's pinyin.

    Example: "贵州茅台" -> "gzmt"
    """
    if text in _pinyin_cache:
        return _pinyin_cache[text]
    try:
        from pypinyin import lazy_pinyin, Style
        initials = "".join(p[0] for p in lazy_pinyin(text, style=Style.FIRST_LETTER))
    except ImportError:
        initials = ""
    _pinyin_cache[text] = initials
    return initials


def _load_instruments(market: str | None = None) -> list[dict[str, Any]]:
    """Load instruments from DuckDB with caching."""
    cache_key = market or "all"
    now = time.monotonic()

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and now < cached[0]:
            return cached[1]

    try:
        from pathlib import Path
        import os
        db_path = Path(os.environ.get("MARKET_DATA_DB_PATH", str(DEFAULT_MARKET_DB_PATH)))
        if not db_path.exists():
            return []
        conn = connect_market_db(db_path)
        repo = MarketDataRepository(conn)

        if market:
            df = repo.conn.execute(
                "SELECT symbol, name, exchange, market, asset_type FROM instruments "
                "WHERE market = ? AND is_active = true ORDER BY symbol LIMIT 10000",
                [market],
            ).fetchdf()
        else:
            df = repo.conn.execute(
                "SELECT symbol, name, exchange, market, asset_type FROM instruments "
                "WHERE is_active = true ORDER BY symbol LIMIT 20000"
            ).fetchdf()
        conn.close()
    except Exception:
        return []

    items = []
    for _, row in df.iterrows():
        item = {
            "symbol": str(row.get("symbol", "")),
            "name": str(row.get("name", "")),
            "exchange": str(row.get("exchange", "")),
            "market": str(row.get("market", "")),
            "asset_type": str(row.get("asset_type", "stock")),
        }
        # Pre-compute pinyin initials for Chinese names
        if item["market"] == "cn" and item["name"]:
            item["pinyin"] = _get_pinyin_initials(item["name"])
        items.append(item)

    with _cache_lock:
        _cache[cache_key] = (time.monotonic() + _CACHE_TTL, items)

    return items


def search_stocks(
    query: str,
    market: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search stocks by code prefix, name substring, or pinyin initials.

    Args:
        query: Search query (code, name, or pinyin initials).
        market: Optional market filter (cn/us/hk).
        limit: Maximum results to return.

    Returns:
        List of matching instruments with symbol, name, market, exchange.
    """
    if not query or not query.strip():
        return []

    q = query.strip().lower()
    safe_limit = max(1, min(limit, 50))
    instruments = _load_instruments(market)

    results: list[dict[str, Any]] = []
    # Score-based ranking: exact code match > code prefix > pinyin match > name match
    for item in instruments:
        symbol_lower = item["symbol"].lower()
        name_lower = item["name"].lower()
        pinyin = item.get("pinyin", "")

        score = 0
        if symbol_lower == q:
            score = 100  # exact code match
        elif symbol_lower.startswith(q):
            score = 80  # code prefix match
        elif pinyin and pinyin.startswith(q):
            score = 60  # pinyin initials match
        elif pinyin and q in pinyin:
            score = 40  # pinyin substring
        elif name_lower and q in name_lower:
            score = 20  # name substring
        else:
            continue

        results.append({**item, "_score": score})

    results.sort(key=lambda x: (-x["_score"], x["symbol"]))
    return [
        {k: v for k, v in r.items() if k != "_score"}
        for r in results[:safe_limit]
    ]
