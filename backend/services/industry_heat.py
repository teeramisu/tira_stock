"""Hot rising industry / concept boards for the market insight page."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

from marketdata.sources.industry_board import (
    enrich_boards_with_historical,
    load_cn_concept_board_spot,
    load_cn_industry_board_spot,
)

_CACHE_TTL_SECONDS = 300
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()
_enriched_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_enriched_cache_lock = threading.Lock()


def _load_board_rows(board_type: str) -> list[dict[str, Any]]:
    if board_type == "concept":
        return load_cn_concept_board_spot()
    return load_cn_industry_board_spot()


def get_hot_boards(
    *,
    board_type: str = "industry",
    limit: int = 30,
    min_change_pct: float = 0.0,
    force_refresh: bool = False,
    enrich_historical: bool = False,
) -> dict[str, Any]:
    """Return boards sorted by same-day change percent (descending).

    Args:
        enrich_historical: When True, fetch 1-week/1-month/3-month returns
            for the top boards via historical data. Cached separately (10 min).
    """
    normalized_type = (board_type or "industry").strip().lower()
    if normalized_type not in {"industry", "concept"}:
        raise ValueError("board_type 仅支持 industry 或 concept")
    safe_limit = max(1, min(int(limit), 100))
    cache_key = f"{normalized_type}:{safe_limit}:{min_change_pct:.4f}"
    enrich_key = f"enriched:{normalized_type}:{safe_limit}:{min_change_pct:.4f}"

    if not force_refresh:
        if enrich_historical:
            with _enriched_cache_lock:
                cached = _enriched_cache.get(enrich_key)
                if cached and time.monotonic() < cached[0]:
                    return cached[1]
        else:
            with _cache_lock:
                cached = _cache.get(cache_key)
                if cached and time.monotonic() < cached[0]:
                    return cached[1]

    rows = _load_board_rows(normalized_type)
    is_ths_fallback = any(row.get("source") == "ths" for row in rows[:1])

    if is_ths_fallback and normalized_type == "concept":
        top_items = rows[:safe_limit]
        rising = top_items
        avg_top10 = 0.0
    else:
        rising = [row for row in rows if (row.get("change_pct") or 0.0) > min_change_pct]
        rising.sort(key=lambda row: row.get("change_pct") or -999.0, reverse=True)
        top_items = rising[:safe_limit]
        avg_top10 = (
            sum(item.get("change_pct") or 0.0 for item in rising[:10]) / min(10, len(rising))
            if rising
            else 0.0
        )

    if enrich_historical:
        top_items = enrich_boards_with_historical(top_items, normalized_type)

    data_source = "ths" if is_ths_fallback else ("akshare_em_concept" if normalized_type == "concept" else "akshare_em_industry")
    payload: dict[str, Any] = {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "board_type": normalized_type,
        "source": data_source,
        "total_boards": len(rows),
        "rising_count": len(rising),
        "avg_top10_change_pct": round(avg_top10, 2),
        "items": top_items,
    }
    if enrich_historical:
        with _enriched_cache_lock:
            _enriched_cache[enrich_key] = (time.monotonic() + _CACHE_TTL_SECONDS * 2, payload)
    else:
        with _cache_lock:
            _cache[cache_key] = (time.monotonic() + _CACHE_TTL_SECONDS, payload)
    return payload
