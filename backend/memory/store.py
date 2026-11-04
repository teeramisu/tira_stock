"""Memory Store — DuckDB persistence for agent signals and decisions.

Implements a composite scoring retrieval inspired by CrewAI:
    score = semantic_weight * similarity + recency_weight * decay + importance_weight * importance
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from marketdata.db import DEFAULT_MARKET_DB_PATH

from .models import DecisionRecord, MarketSnapshot, SignalRecord

_AGENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_signals (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    score REAL,
    confidence REAL NOT NULL DEFAULT 0.5,
    reasoning TEXT,
    ticker TEXT,
    metadata JSON,
    tags JSON,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    expires_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    signal_ids JSON,
    decision TEXT NOT NULL DEFAULT 'pass',
    score REAL,
    confidence REAL NOT NULL DEFAULT 0.5,
    ticker TEXT,
    target_position REAL,
    stop_loss REAL,
    take_profit REAL,
    reasoning TEXT,
    metadata JSON,
    performance_after REAL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id TEXT PRIMARY KEY,
    trade_date DATE NOT NULL,
    market TEXT NOT NULL DEFAULT 'cn',
    regime TEXT NOT NULL DEFAULT 'unknown',
    vix_level REAL,
    snapshot JSON,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);
"""


_SIGNAL_COLS = "id, agent_name, signal_type, score, confidence, reasoning, ticker, metadata, tags, created_at, expires_at"


class MemoryStore:
    """DuckDB-backed store with composite retrieval scoring."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        recency_weight: float = 0.6,
        importance_weight: float = 0.1,
        recency_half_life_days: float = 7.0,
    ):
        path = Path(db_path) if db_path else DEFAULT_MARKET_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(path))
        self._conn.execute(_AGENT_SCHEMA_SQL)
        # Migrate: add tags column if missing (for existing databases)
        try:
            self._conn.execute("ALTER TABLE agent_signals ADD COLUMN tags JSON")
        except Exception:
            pass  # column already exists
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.recency_half_life = recency_half_life_days

    # ---- Signals ----

    def save_signal(self, signal: SignalRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO agent_signals
               (id, agent_name, signal_type, score, confidence, reasoning, ticker, metadata, tags, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                signal.id, signal.agent_name, signal.signal_type,
                signal.score, signal.confidence, signal.reasoning,
                signal.ticker, json.dumps(signal.metadata, ensure_ascii=False),
                json.dumps(signal.tags or [], ensure_ascii=False),
                signal.created_at, signal.expires_at,
            ],
        )

    def query_signals(
        self,
        *,
        agent_name: str | None = None,
        ticker: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[SignalRecord]:
        conditions = []
        params: list[Any] = []
        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._conn.execute(
            f"SELECT {_SIGNAL_COLS} FROM agent_signals {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [_row_to_signal(r) for r in rows]

    def recall_signals(
        self,
        query: str,
        *,
        agent_name: str | None = None,
        tag: str | None = None,
        limit: int = 10,
    ) -> list[SignalRecord]:
        """Composite-score retrieval.

        Since DuckDB has no built-in embedding, we use a simple keyword-match
        as a lightweight semantic proxy, combined with recency decay and
        importance scoring.
        """
        conditions = []
        params: list[Any] = []
        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if tag:
            conditions.append("tags::TEXT ILIKE ?")
            params.append(f"%{tag}%")
        conditions.append("(reasoning ILIKE ? OR ticker ILIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])
        where = "WHERE " + " AND ".join(conditions)

        rows = self._conn.execute(
            f"SELECT {_SIGNAL_COLS} FROM agent_signals {where} ORDER BY created_at DESC LIMIT ?",
            params + [max(limit * 3, 30)],
        ).fetchall()

        if not rows:
            return []

        now = datetime.now()
        scored: list[tuple[float, SignalRecord]] = []
        for row in rows:
            rec = _row_to_signal(row)
            try:
                ts = datetime.fromisoformat(rec.created_at)
                days_ago = (now - ts).total_seconds() / 86400
            except Exception:
                days_ago = 30.0
            recency_decay = math.exp(-days_ago / max(self.recency_half_life, 1.0))
            importance = rec.confidence or 0.5
            composite = self.recency_weight * recency_decay + self.importance_weight * importance
            scored.append((composite, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def latest_signal(self, agent_name: str) -> SignalRecord | None:
        rows = self.query_signals(agent_name=agent_name, limit=1)
        return rows[0] if rows else None

    # ---- Decisions ----

    def save_decision(self, decision: DecisionRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO agent_decisions
               (id, run_id, agent_name, signal_ids, decision, score, confidence,
                ticker, target_position, stop_loss, take_profit, reasoning, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                decision.id, decision.run_id, decision.agent_name,
                json.dumps(decision.signal_ids), decision.decision,
                decision.score, decision.confidence, decision.ticker,
                decision.target_position, decision.stop_loss, decision.take_profit,
                decision.reasoning, json.dumps(decision.metadata, ensure_ascii=False),
                decision.created_at,
            ],
        )

    def query_decisions(self, *, run_id: str | None = None, ticker: str | None = None, since: str | None = None, limit: int = 20) -> list[DecisionRecord]:
        conditions = []
        params: list[Any] = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM agent_decisions {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [_row_to_decision(r) for r in rows]

    def update_decision_performance(self, decision_id: str, performance: float) -> None:
        self._conn.execute(
            "UPDATE agent_decisions SET performance_after = ? WHERE id = ?",
            [performance, decision_id],
        )

    # ---- Snapshots ----

    def save_snapshot(self, snapshot: MarketSnapshot) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO market_snapshots
               (id, trade_date, market, regime, vix_level, snapshot, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                snapshot.id, snapshot.trade_date, snapshot.market,
                snapshot.regime, snapshot.vix_level,
                json.dumps(snapshot.snapshot, ensure_ascii=False),
                snapshot.created_at,
            ],
        )

    def latest_snapshot(self, market: str = "cn") -> MarketSnapshot | None:
        rows = self._conn.execute(
            "SELECT * FROM market_snapshots WHERE market = ? ORDER BY trade_date DESC LIMIT 1",
            [market],
        ).fetchall()
        return _row_to_snapshot(rows[0]) if rows else None

    def cleanup_expired(self) -> int:
        """Delete expired signals (expires_at < now()). Returns count deleted."""
        now = datetime.now().isoformat()
        result = self._conn.execute(
            "DELETE FROM agent_signals WHERE expires_at IS NOT NULL AND expires_at < ?",
            [now],
        )
        return result.fetchone()[0] if result else 0


def _row_to_signal(row: tuple) -> SignalRecord:
    cols = ["id", "agent_name", "signal_type", "score", "confidence", "reasoning",
            "ticker", "metadata", "tags", "created_at", "expires_at"]
    d = dict(zip(cols, row))
    d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else (d["metadata"] or {})
    d["tags"] = json.loads(d["tags"]) if isinstance(d["tags"], str) else (d["tags"] or [])
    d["created_at"] = _to_iso(d.get("created_at"))
    d["expires_at"] = _to_iso(d.get("expires_at"))
    return SignalRecord(**d)


def _row_to_decision(row: tuple) -> DecisionRecord:
    cols = ["id", "run_id", "agent_name", "signal_ids", "decision", "score",
            "confidence", "ticker", "target_position", "stop_loss", "take_profit",
            "reasoning", "metadata", "performance_after", "created_at"]
    d = dict(zip(cols, row))
    raw_ids = d.pop("signal_ids", "[]")
    d["signal_ids"] = json.loads(raw_ids) if isinstance(raw_ids, str) else (raw_ids or [])
    d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else (d["metadata"] or {})
    d["created_at"] = _to_iso(d.get("created_at"))
    return DecisionRecord(**d)


def _row_to_snapshot(row: tuple) -> MarketSnapshot:
    cols = ["id", "trade_date", "market", "regime", "vix_level", "snapshot", "created_at"]
    d = dict(zip(cols, row))
    d["trade_date"] = str(d["trade_date"])
    if isinstance(d["trade_date"], str) and " " in d["trade_date"]:
        d["trade_date"] = d["trade_date"].split(" ")[0]
    d["snapshot"] = json.loads(d["snapshot"]) if isinstance(d["snapshot"], str) else (d["snapshot"] or {})
    d["created_at"] = _to_iso(d.get("created_at"))
    return MarketSnapshot(**d)


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
