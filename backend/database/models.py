"""数据库表模型（万级用户：用户 + 回测记录）。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"), onupdate=datetime.utcnow)

    backtest_runs: Mapped[list["BacktestRun"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    anonymous_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(8), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))

    user: Mapped[User | None] = relationship(back_populates="backtest_runs")

    __table_args__ = (
        Index("ix_backtest_runs_user_created", "user_id", "created_at"),
        Index("ix_backtest_runs_anon_created", "anonymous_id", "created_at"),
    )
