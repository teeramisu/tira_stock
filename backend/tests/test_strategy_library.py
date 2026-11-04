"""
策略库服务测试：验证导入的 JoinQuant 策略评分、筛选和源码读取。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytestmark = pytest.mark.requires_data

from services.strategy_library import (
    get_strategy_library_facets,
    get_strategy_library_item,
    list_strategy_library,
)


def test_strategy_library_loads_ranked_items():
    result = list_strategy_library(limit=5)
    assert result["total"] >= 600
    assert len(result["strategies"]) == 5
    first = result["strategies"][0]
    assert first["id"].startswith("jq-")
    assert first["score"] > 0
    assert first["intro"]


def test_strategy_library_filters_by_tag_and_score():
    result = list_strategy_library(tag="ETF/指数", min_score=50, limit=20)
    assert result["total"] > 0
    assert all("ETF/指数" in item["tags"] for item in result["strategies"])
    assert all(item["score"] >= 50 for item in result["strategies"])


def test_strategy_library_detail_can_include_source():
    first = list_strategy_library(limit=1)["strategies"][0]
    detail = get_strategy_library_item(first["id"], include_source=True)
    assert detail["id"] == first["id"]
    assert "source" in detail
    assert len(detail["source"]) > 100


def test_strategy_library_facets():
    facets = get_strategy_library_facets()
    assert facets["count"] >= 600
    assert facets["adapted_count"] >= 4
    assert facets["collections"]
    assert facets["tags"]


def test_strategy_library_marks_jq0007_industry_bbi():
    detail = get_strategy_library_item("jq-0007")
    assert detail["engine_status"] == "adapted_backtest_ready"
    assert detail["adapted_strategy"]["id"] == "jq_industry_etf_bbi_rotation"


def test_strategy_library_marks_jq0019_portfolio_adaptation():
    detail = get_strategy_library_item("jq-0019")
    assert detail["engine_status"] == "adapted_backtest_ready"
    assert detail["adapted_strategy"]["id"] == "jq_etf_polyfit_momentum_top1"


def test_strategy_library_marks_jq0006_small_cap():
    detail = get_strategy_library_item("jq-0006")
    assert detail["engine_status"] == "adapted_backtest_ready"
    assert detail["adapted_strategy"]["id"] == "jq_small_cap_top_k"


def test_strategy_library_marks_jq0026_ma_cross():
    detail = get_strategy_library_item("jq-0026")
    assert detail["engine_status"] == "adapted_backtest_ready"
    assert detail["adapted_strategy"]["id"] == "jq_etf_ma_cross_timing"


def test_strategy_library_marks_jq0002_portfolio_adaptation():
    detail = get_strategy_library_item("jq-0002")
    assert detail["engine_status"] == "adapted_backtest_ready"
    assert detail["adapted_strategy"]["id"] == "jq_value_timing_top_k"
    assert detail["adapted_strategy"]["run_preset"]["mode"] == "portfolio"


def test_strategy_library_marks_adapted_sources():
    detail = get_strategy_library_item("jq-0001")
    assert detail["engine_status"] == "adapted_backtest_ready"
    assert detail["adapted_strategy"]["id"] == "jq_wide_base_etf_bbi_rotation"
    assert detail["adapted_strategy"]["source_strategy_id"] == "jq-0001"
    assert detail["adapted_strategy"]["reproducible_scope"]
    assert detail["adapted_strategy"]["non_reproducible_diff"]
    assert detail["migration_artifact"]["artifact_type"] == "joinquant_migration_v1"
    assert detail["migration_artifact"]["source_strategy"]["id"] == "jq-0001"
    assert detail["migration_artifact"]["adapted_strategy"]["id"] == "jq_wide_base_etf_bbi_rotation"


if __name__ == "__main__":
    test_strategy_library_loads_ranked_items()
    test_strategy_library_filters_by_tag_and_score()
    test_strategy_library_detail_can_include_source()
    test_strategy_library_facets()
    test_strategy_library_marks_adapted_sources()
    print("All strategy library tests passed.")
