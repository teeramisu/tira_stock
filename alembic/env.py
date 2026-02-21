"""Alembic 环境：从 backend.database 读取模型与 URL。"""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.config import DATABASE_URL
from backend.database.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 使用环境变量或默认占位
target_metadata = Base.metadata
db_url = DATABASE_URL or "postgresql+asyncpg://localhost/stock_backtest"
if db_url.startswith("postgresql://"):
    db_url = "postgresql+asyncpg://" + db_url.split("://", 1)[1]
config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """离线模式：仅生成 SQL，不连库。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
