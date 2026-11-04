"""测试公共配置。

CI 是干净 checkout，没有本地 `data/`（行情库、策略库、星球数据，约 11GB）。
依赖这些数据集的测试统一打 `requires_data` 标记，这里用「哨兵文件」判断
数据是否真的可用——只看目录存在与否不够（很多模块会顺手 mkdir），
否则又一次变成「本地绿、CI 红」。
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

# 任一存在就认为本地数据集可用
SENTINELS = (
    DATA_ROOT / "market" / "market_data.duckdb",      # 行情库
    DATA_ROOT / "joinquant_strategy_library.json",    # 策略库
    DATA_ROOT / "board_constituents.json",            # 板块成分
    DATA_ROOT / "zsxq_stocks_scored.json",            # 星球评分
)


def data_available() -> bool:
    return any(p.exists() for p in SENTINELS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if data_available():
        return
    skip = pytest.mark.skip(reason="缺少本地 data/ 数据集（干净 checkout / CI 环境）")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)
