"""Tests for hot industry board service."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import industry_heat as mod


def test_get_hot_boards_sorts_by_change_pct(monkeypatch):
    sample = [
        {"name": "半导体", "change_pct": 3.2, "rank": 2},
        {"name": "银行", "change_pct": -1.0, "rank": 1},
        {"name": "光学元件", "change_pct": 7.5, "rank": 3},
    ]

    monkeypatch.setattr(mod, "_load_board_rows", lambda board_type: sample)
    mod._cache.clear()
    mod._enriched_cache.clear()

    payload = mod.get_hot_boards(board_type="industry", limit=2, force_refresh=True)

    assert payload["rising_count"] == 2
    assert [item["name"] for item in payload["items"]] == ["光学元件", "半导体"]


def test_get_hot_boards_rejects_unknown_board_type():
    try:
        mod.get_hot_boards(board_type="unknown")
    except ValueError as exc:
        assert "industry" in str(exc)
    else:
        raise AssertionError("expected ValueError")
