"""数据库配置：从环境变量读取，未配置时禁用写库。"""
from __future__ import annotations

import os

# 例: postgresql+asyncpg://user:pass@localhost:5432/stock_backtest
DATABASE_URL: str | None = os.getenv("DATABASE_URL")

def is_configured() -> bool:
    return bool(DATABASE_URL and DATABASE_URL.startswith("postgresql"))
