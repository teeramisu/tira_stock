"""Discovery Loop 实验账本：候选、实验、失败记忆与血缘。

使用 SQLite（``data/discovery/ledger.db``），与 DuckDB 行情库解耦，
因此行情刷新占用写锁时账本仍可读写。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "data" / "discovery" / "ledger.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    parent_ids TEXT NOT NULL DEFAULT '[]',
    genome_json TEXT NOT NULL,
    archive_cell TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT 'generated',
    promotion_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    candidate_id TEXT,
    genome_hash TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT '',
    params_json TEXT NOT NULL DEFAULT '{}',
    dataset_version TEXT NOT NULL DEFAULT '',
    split_id TEXT NOT NULL DEFAULT '',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    dsr REAL,
    pbo REAL,
    fdr REAL,
    status TEXT NOT NULL DEFAULT 'completed',
    failure_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    hypothesis TEXT NOT NULL,
    failure_mode TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_candidate ON experiments(candidate_id);
CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_exp_dsr ON experiments(dsr DESC);
CREATE INDEX IF NOT EXISTS idx_cand_stage ON candidates(stage);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ExperimentLedger:
    """线程安全的实验账本（每次操作独立连接，WAL 模式）。"""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.path = Path(db_path) if db_path else DEFAULT_LEDGER_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._session() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ candidates
    def record_candidate(
        self,
        genome: Dict[str, Any],
        *,
        parent_ids: Optional[Iterable[str]] = None,
        archive_cell: str = "",
        stage: str = "generated",
        promotion_status: str = "pending",
        candidate_id: Optional[str] = None,
    ) -> str:
        cid = candidate_id or _hash(genome)
        with self._lock, self._session() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO candidates "
                "(candidate_id, parent_ids, genome_json, archive_cell, stage, promotion_status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    json.dumps(list(parent_ids or []), ensure_ascii=False),
                    json.dumps(genome, ensure_ascii=False, sort_keys=True),
                    archive_cell,
                    stage,
                    promotion_status,
                    _now(),
                ),
            )
        return cid

    def update_candidate_status(
        self,
        candidate_id: str,
        *,
        stage: Optional[str] = None,
        promotion_status: Optional[str] = None,
        archive_cell: Optional[str] = None,
    ) -> bool:
        fields, values = [], []
        if stage is not None:
            fields.append("stage = ?")
            values.append(stage)
        if promotion_status is not None:
            fields.append("promotion_status = ?")
            values.append(promotion_status)
        if archive_cell is not None:
            fields.append("archive_cell = ?")
            values.append(archive_cell)
        if not fields:
            return False
        values.append(candidate_id)
        with self._lock, self._session() as conn:
            cur = conn.execute(f"UPDATE candidates SET {', '.join(fields)} WHERE candidate_id = ?", values)
            changed = cur.rowcount
        return changed > 0

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
            row = dict(row) if row else None
        return self._candidate_row(row) if row else None

    def list_candidates(
        self,
        *,
        stage: Optional[str] = None,
        promotion_status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM candidates WHERE 1=1"
        params: List[Any] = []
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        if promotion_status:
            query += " AND promotion_status = ?"
            params.append(promotion_status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._session() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        return [self._candidate_row(r) for r in rows]

    # ------------------------------------------------------------ experiments
    def record_experiment(
        self,
        *,
        genome_hash: str,
        strategy: str = "",
        params: Optional[Dict[str, Any]] = None,
        dataset_version: str = "",
        split_id: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        dsr: Optional[float] = None,
        pbo: Optional[float] = None,
        fdr: Optional[float] = None,
        status: str = "completed",
        failure_reason: str = "",
        candidate_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        payload = {
            "genome_hash": genome_hash,
            "strategy": strategy,
            "params": params or {},
            "dataset_version": dataset_version,
            "split_id": split_id,
        }
        eid = experiment_id or _hash(payload)
        with self._lock, self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments "
                "(experiment_id, candidate_id, genome_hash, strategy, params_json, dataset_version, "
                " split_id, metrics_json, dsr, pbo, fdr, status, failure_reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    eid,
                    candidate_id,
                    genome_hash,
                    strategy,
                    json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
                    dataset_version,
                    split_id,
                    json.dumps(metrics or {}, ensure_ascii=False),
                    dsr,
                    pbo,
                    fdr,
                    status,
                    failure_reason,
                    _now(),
                ),
            )
        return eid

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
            row = dict(row) if row else None
        return self._experiment_row(row) if row else None

    def list_experiments(
        self,
        *,
        strategy: Optional[str] = None,
        status: Optional[str] = None,
        min_dsr: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params: List[Any] = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if status:
            query += " AND status = ?"
            params.append(status)
        if min_dsr is not None:
            query += " AND dsr IS NOT NULL AND dsr >= ?"
            params.append(float(min_dsr))
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._session() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        return [self._experiment_row(r) for r in rows]

    def leaderboard(self, *, metric: str = "dsr", limit: int = 20) -> List[Dict[str, Any]]:
        if metric not in {"dsr", "pbo", "fdr"}:
            raise ValueError("metric must be dsr/pbo/fdr")
        order = "DESC" if metric == "dsr" else "ASC"
        query = (
            f"SELECT * FROM experiments WHERE {metric} IS NOT NULL "
            f"ORDER BY {metric} {order}, created_at DESC LIMIT ?"
        )
        with self._session() as conn:
            rows = [dict(r) for r in conn.execute(query, (int(limit),)).fetchall()]
        return [self._experiment_row(r) for r in rows]

    # ------------------------------------------------------------ lessons
    def record_lesson(
        self,
        *,
        hypothesis: str,
        failure_mode: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        lesson_id: Optional[str] = None,
    ) -> str:
        lid = lesson_id or _hash({"hypothesis": hypothesis, "failure_mode": failure_mode})
        with self._lock, self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lessons (lesson_id, hypothesis, failure_mode, evidence_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (lid, hypothesis, failure_mode, json.dumps(evidence or {}, ensure_ascii=False), _now()),
            )
        return lid

    def list_lessons(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM lessons ORDER BY created_at DESC LIMIT ?", (int(limit),)
                ).fetchall()
            ]
        return [
            {
                "lesson_id": r["lesson_id"],
                "hypothesis": r["hypothesis"],
                "failure_mode": r["failure_mode"],
                "evidence": json.loads(r["evidence_json"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------ summary
    def summary(self) -> Dict[str, Any]:
        with self._session() as conn:
            counts = {
                "candidates": conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
                "experiments": conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
                "lessons": conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
            }
            by_status = {
                row[0]: row[1]
                for row in conn.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall()
            }
            by_stage = {
                row[0]: row[1]
                for row in conn.execute("SELECT stage, COUNT(*) FROM candidates GROUP BY stage").fetchall()
            }
            best = conn.execute(
                "SELECT experiment_id, strategy, dsr, pbo FROM experiments "
                "WHERE dsr IS NOT NULL ORDER BY dsr DESC LIMIT 1"
            ).fetchone()
        return {
            "path": str(self.path),
            **counts,
            "experiments_by_status": by_status,
            "candidates_by_stage": by_stage,
            "best_experiment": dict(best) if best else None,
        }

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _candidate_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "candidate_id": row["candidate_id"],
            "parent_ids": json.loads(row["parent_ids"] or "[]"),
            "genome": json.loads(row["genome_json"] or "{}"),
            "archive_cell": row["archive_cell"],
            "stage": row["stage"],
            "promotion_status": row["promotion_status"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _experiment_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "candidate_id": row["candidate_id"],
            "genome_hash": row["genome_hash"],
            "strategy": row["strategy"],
            "params": json.loads(row["params_json"] or "{}"),
            "dataset_version": row["dataset_version"],
            "split_id": row["split_id"],
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "dsr": row["dsr"],
            "pbo": row["pbo"],
            "fdr": row["fdr"],
            "status": row["status"],
            "failure_reason": row["failure_reason"],
            "created_at": row["created_at"],
        }
