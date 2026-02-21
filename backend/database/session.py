"""异步数据库会话与连接池。"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import DATABASE_URL
from .models import Base

# 仅当配置了 DATABASE_URL 时创建 engine
engine = None
async_session_factory = None

if DATABASE_URL:
    # 确保是 async 驱动
    url = DATABASE_URL
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url.split("://", 1)[1]
    engine = create_async_engine(
        url,
        pool_size=20,
        max_overflow=10,
        echo=False,
    )
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：注入异步 Session。未配置数据库时返回 None 会由调用方处理。"""
    if not async_session_factory:
        yield None
        return
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """创建所有表（开发/测试用；生产建议用 Alembic 迁移）。"""
    if not engine:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
