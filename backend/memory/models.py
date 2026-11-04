"""Memory models — Pydantic schemas for agent signals, decisions, and snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SignalRecord(BaseModel):
    """A single agent signal output."""

    id: str = Field(default_factory=lambda: f"sig_{datetime.utcnow().timestamp()}")
    agent_name: str = Field(..., description="e.g. sentiment, industry, strategy")
    signal_type: str = Field(default="score", description="score / vote / alert")
    score: float | None = Field(default=None, description="0-100 when signal_type=score")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    ticker: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, description="分类标签，用于分类检索")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str | None = Field(default=None)


class DecisionRecord(BaseModel):
    """A final trading decision."""

    id: str = Field(default_factory=lambda: f"dec_{datetime.utcnow().timestamp()}")
    run_id: str = Field(..., description="Orchestrator batch run id")
    agent_name: str = Field(default="strategy")
    signal_ids: list[str] = Field(default_factory=list)
    decision: str = Field(default="pass", description="buy / sell / hold / pass")
    score: float | None = Field(default=None, description="Aggregated score 0-100")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ticker: str | None = Field(default=None)
    target_position: float | None = Field(default=None)
    stop_loss: float | None = Field(default=None)
    take_profit: float | None = Field(default=None)
    reasoning: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
    performance_after: float | None = Field(default=None, description="Filled later via reflection")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class MarketSnapshot(BaseModel):
    """A point-in-time market state for agent context replay."""

    id: str = Field(default_factory=lambda: f"snap_{datetime.utcnow().timestamp()}")
    trade_date: str = Field(..., description="YYYY-MM-DD")
    market: str = Field(default="cn")
    regime: str = Field(default="unknown", description="bull / bear / range / volatile")
    vix_level: float | None = Field(default=None)
    snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
