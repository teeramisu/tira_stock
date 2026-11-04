"""
JoinQuant 精选策略库：加载已导入的原始策略、评分和介绍。

这些策略大多依赖聚宽 API，当前先作为可检索策略资产纳入系统；
后续可以挑选高分策略逐个改写为本项目的 signal(df) 形式。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from backtest.imported_strategies import _adapted_source_ids, list_adapted_strategies


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_PATH = PROJECT_ROOT / "data" / "joinquant_strategy_library.json"


@lru_cache(maxsize=1)
def load_strategy_library() -> dict[str, Any]:
    """读取策略库 JSON，并缓存到进程内。"""
    if not LIBRARY_PATH.exists():
        return {"count": 0, "strategies": [], "score_rubric": {}}
    with LIBRARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_strategy_library(
    *,
    q: Optional[str] = None,
    collection: Optional[str] = None,
    tag: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """返回过滤后的策略摘要列表。"""
    payload = load_strategy_library()
    adapted_by_source = _adapted_by_source()
    strategies = payload.get("strategies", [])
    q_norm = (q or "").strip().lower()
    collection_norm = (collection or "").strip()
    tag_norm = (tag or "").strip()

    def matched(item: dict[str, Any]) -> bool:
        if collection_norm and item.get("collection") != collection_norm:
            return False
        if tag_norm and tag_norm not in item.get("tags", []):
            return False
        if min_score is not None and float(item.get("score") or 0) < min_score:
            return False
        if q_norm:
            haystack = " ".join(
                str(item.get(k, ""))
                for k in ("title", "intro", "relative_path", "source_excerpt")
            ).lower()
            if q_norm not in haystack:
                return False
        return True

    filtered = [item for item in strategies if matched(item)]
    total = len(filtered)
    offset = max(offset, 0)
    limit = max(1, min(limit, 200))
    page = filtered[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "strategies": [_summary_item(item, adapted_by_source.get(item.get("id"))) for item in page],
        "score_rubric": payload.get("score_rubric", {}),
    }


def get_strategy_library_item(strategy_id: str, *, include_source: bool = False) -> dict[str, Any]:
    """按 jq-0001 这类 id 获取策略详情。"""
    payload = load_strategy_library()
    adapted_by_source = _adapted_by_source()
    for item in payload.get("strategies", []):
        if item.get("id") == strategy_id:
            result = dict(item)
            adapted = adapted_by_source.get(item.get("id"))
            if adapted:
                result["engine_status"] = "adapted_backtest_ready"
                result["adapted_strategy"] = adapted
                result["migration_artifact"] = _build_joinquant_migration_artifact(item, adapted)
                result["engine_note"] = "该策略已抽取 OHLCV 可复现部分，可在 /api/backtest 中直接选择运行。"
            if include_source:
                result["source"] = read_strategy_source(item)
            return result
    raise KeyError(strategy_id)


def get_strategy_library_facets() -> dict[str, Any]:
    """返回集合和标签聚合，方便前端筛选。"""
    payload = load_strategy_library()
    collections: dict[str, int] = {}
    tags: dict[str, int] = {}
    for item in payload.get("strategies", []):
        collection = item.get("collection") or "未分类"
        collections[collection] = collections.get(collection, 0) + 1
        for tag in item.get("tags", []):
            tags[tag] = tags.get(tag, 0) + 1
    adapted_count = len(_adapted_by_source())
    return {
        "count": payload.get("count", 0),
        "adapted_count": adapted_count,
        "collections": [
            {"name": name, "count": count}
            for name, count in sorted(collections.items(), key=lambda x: (-x[1], x[0]))
        ],
        "tags": [
            {"name": name, "count": count}
            for name, count in sorted(tags.items(), key=lambda x: (-x[1], x[0]))
        ],
        "score_rubric": payload.get("score_rubric", {}),
    }


def read_strategy_source(item: dict[str, Any]) -> str:
    """读取已复制进项目 data 目录的原始策略文件。"""
    rel = item.get("source_path")
    if not rel:
        return ""
    path = PROJECT_ROOT / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _adapted_by_source() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in list_adapted_strategies():
        for sid in _adapted_source_ids(item):
            existing = out.get(sid)
            if existing is None:
                out[sid] = item
                continue
            prefer_new = item.get("run_preset", {}).get("mode") == "portfolio"
            prefer_existing = existing.get("run_preset", {}).get("mode") == "portfolio"
            if prefer_new and not prefer_existing:
                out[sid] = item
    return out


def _summary_item(item: dict[str, Any], adapted: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    summary = {
        "id": item.get("id"),
        "rank": item.get("rank"),
        "title": item.get("title"),
        "collection": item.get("collection"),
        "score": item.get("score"),
        "scores": item.get("scores"),
        "confidence": item.get("confidence"),
        "tags": item.get("tags", []),
        "metrics": item.get("metrics", {}),
        "intro": item.get("intro"),
        "cautions": item.get("cautions", []),
        "engine_status": item.get("engine_status"),
        "source_path": item.get("source_path"),
    }
    if adapted:
        summary["engine_status"] = "adapted_backtest_ready"
        summary["adapted_strategy"] = adapted
        summary["migration_artifact"] = _build_joinquant_migration_artifact(item, adapted)
    return summary


def _build_joinquant_migration_artifact(item: dict[str, Any], adapted: dict[str, Any]) -> dict[str, Any]:
    """Build migration transparency artifact for one adapted strategy."""
    return {
        "artifact_type": "joinquant_migration_v1",
        "source_strategy": {
            "id": item.get("id"),
            "title": item.get("title"),
            "source_path": item.get("source_path"),
        },
        "adapted_strategy": {
            "id": adapted.get("id"),
            "name": adapted.get("name"),
            "source_strategy_id": adapted.get("source_strategy_id"),
        },
        "reproducible_scope": adapted.get("reproducible_scope"),
        "non_reproducible_diff": adapted.get("non_reproducible_diff"),
    }
