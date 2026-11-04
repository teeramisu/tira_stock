"""更新日志系统：记录每次数据刷新的详细信息。

提供两种输出格式：
1. JSON 结构化日志（agent 可读）— data/update_log.json
2. 摘要生成（用于 HTML 展示或 agent 上下文注入）
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 日志文件路径
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_LOG_FILE = _LOG_DIR / "update_log.json"
_MAX_ENTRIES = 500  # 保留最近 500 条记录


def _ensure_log_file() -> dict[str, Any]:
    """确保日志文件存在并返回内容。"""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _LOG_FILE.exists():
        data = {"entries": [], "metadata": {"created_at": datetime.now().isoformat(timespec="seconds")}}
        _save_log(data)
        return data
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {"entries": [], "metadata": {"created_at": datetime.now().isoformat(timespec="seconds")}}
        _save_log(data)
        return data


def _save_log(data: dict[str, Any]) -> None:
    """保存日志到文件。"""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_update(
    market: str,
    data_type: str,
    source_used: str,
    summary: str,
    symbols: Optional[list[str]] = None,
    records_added: int = 0,
    date_range: Optional[dict[str, str]] = None,
    details: str = "",
    source_chain: Optional[list[str]] = None,
) -> dict[str, Any]:
    """写入一条更新日志。

    Args:
        market: 市场 (us/cn/hk)
        data_type: 数据类型 (us_daily/cn_daily/etf_valuation/...)
        source_used: 实际使用的数据源 ID
        summary: 一句话摘要（人类可读）
        symbols: 涉及的标的列表
        records_added: 新增记录数
        date_range: 数据日期范围 {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
        details: 详细说明（agent 可读，包含错误信息、降级原因等）
        source_chain: 数据源调用链，如 ["yfinance(failed:rate_limit)", "akshare(success)"]
    """
    data = _ensure_log_file()

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "market": market,
        "data_type": data_type,
        "source_used": source_used,
        "summary": summary,
        "symbols": symbols or [],
        "records_added": records_added,
        "date_range": date_range or {},
        "details": details,
        "source_chain": source_chain or [],
    }

    data["entries"].insert(0, entry)  # 最新的在前面
    # 截断旧记录
    if len(data["entries"]) > _MAX_ENTRIES:
        data["entries"] = data["entries"][:_MAX_ENTRIES]
    data["metadata"] = data.get("metadata", {})
    data["metadata"]["last_updated"] = datetime.now().isoformat(timespec="seconds")
    data["metadata"]["total_entries"] = len(data["entries"])

    _save_log(data)
    logger.info("Update logged: %s", summary)
    return entry


def get_update_log(
    market: Optional[str] = None,
    data_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """读取更新日志，支持按市场和类型筛选。"""
    data = _ensure_log_file()
    entries = data.get("entries", [])

    if market:
        entries = [e for e in entries if e.get("market") == market]
    if data_type:
        entries = [e for e in entries if e.get("data_type") == data_type]

    return entries[:limit]


def get_update_summary() -> dict[str, Any]:
    """生成压缩摘要，适合注入 agent 上下文。"""
    data = _ensure_log_file()
    entries = data.get("entries", [])

    # 按市场+类型分组统计
    groups: dict[str, list] = {}
    for e in entries:
        key = f"{e.get('market', '?')}/{e.get('data_type', '?')}"
        groups.setdefault(key, []).append(e)

    summary_lines = []
    for key, group_entries in groups.items():
        latest = group_entries[0]
        count = len(group_entries)
        summary_lines.append(
            f"{key}: {count}次更新, 最近={latest.get('timestamp', '?')[:10]}, "
            f"源={latest.get('source_used', '?')}, "
            f"摘要: {latest.get('summary', '')}"
        )

    return {
        "total_entries": len(entries),
        "last_updated": data.get("metadata", {}).get("last_updated"),
        "by_category": summary_lines,
    }
