"""Metrics Store — 训练和评估指标持久化.

SQLite存储所有训练运行和评估的指标历史，支持：
- 训练指标记录（loss, reward, 耗时）
- 评估指标记录（Sharpe, Alpha, MaxDD）
- 趋势查询（最近N天指标变化）
- 运行状态追踪
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rl.metrics")

_METRICS_DB = Path(__file__).resolve().parents[2] / "data" / "models" / "metrics.db"


@dataclass
class TrainingRunMetrics:
    """单次训练运行指标."""
    run_id: str
    version_id: str
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    total_steps: int = 0
    n_seeds: int = 1
    best_seed: int = 0
    best_sharpe: float = 0.0
    best_return: float = 0.0
    status: str = "running"  # running | completed | failed
    error: str = ""
    config_snapshot: dict = None

    def __post_init__(self):
        if self.config_snapshot is None:
            self.config_snapshot = {}


@dataclass
class EvaluationMetrics:
    """模型评估指标."""
    eval_id: str
    version_id: str
    eval_type: str  # "test_set" | "walk_forward" | "quick_eval"
    created_at: str
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    positive_alpha_rate: float = 0.0
    n_windows: int = 0
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class MetricsStore:
    """指标存储 — SQLite持久化."""

    def __init__(self, db_path: Path = _METRICS_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    version_id TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    n_seeds INTEGER DEFAULT 1,
                    best_seed INTEGER DEFAULT 0,
                    best_sharpe REAL DEFAULT 0,
                    best_return REAL DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    error TEXT DEFAULT '',
                    config_snapshot TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    eval_id TEXT PRIMARY KEY,
                    version_id TEXT,
                    eval_type TEXT,
                    created_at TEXT NOT NULL,
                    sharpe REAL DEFAULT 0,
                    annual_return REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    alpha REAL DEFAULT 0,
                    positive_alpha_rate REAL DEFAULT 0,
                    n_windows INTEGER DEFAULT 0,
                    details TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_status (
                    date TEXT PRIMARY KEY,
                    training_triggered INTEGER DEFAULT 0,
                    training_succeeded INTEGER DEFAULT 0,
                    model_promoted INTEGER DEFAULT 0,
                    data_refreshed INTEGER DEFAULT 0,
                    new_data_days INTEGER DEFAULT 0,
                    error TEXT DEFAULT '',
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_status ON training_runs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evals_version ON evaluations(version_id)
            """)
            conn.commit()

    # ---- Training Runs ----

    def start_training_run(
        self,
        run_id: str,
        version_id: str = "",
        total_steps: int = 0,
        n_seeds: int = 1,
        config: dict = None,
    ) -> TrainingRunMetrics:
        """记录训练开始."""
        run = TrainingRunMetrics(
            run_id=run_id,
            version_id=version_id,
            started_at=datetime.now().isoformat(),
            total_steps=total_steps,
            n_seeds=n_seeds,
            status="running",
            config_snapshot=config or {},
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO training_runs
                   (run_id, version_id, started_at, total_steps, n_seeds, status, config_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id, run.version_id, run.started_at,
                    run.total_steps, run.n_seeds, run.status,
                    json.dumps(run.config_snapshot),
                ),
            )
            conn.commit()
        logger.info(f"Training run started: {run_id}")
        return run

    def finish_training_run(
        self,
        run_id: str,
        status: str = "completed",
        best_seed: int = 0,
        best_sharpe: float = 0.0,
        best_return: float = 0.0,
        version_id: str = "",
        error: str = "",
    ):
        """记录训练完成."""
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            # Get start time to compute duration
            row = conn.execute(
                "SELECT started_at FROM training_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            duration = 0.0
            if row:
                start = datetime.fromisoformat(row[0])
                duration = (now - start).total_seconds()

            conn.execute(
                """UPDATE training_runs SET
                   finished_at = ?, duration_seconds = ?, status = ?,
                   best_seed = ?, best_sharpe = ?, best_return = ?,
                   version_id = ?, error = ?
                   WHERE run_id = ?""",
                (
                    now.isoformat(), duration, status,
                    best_seed, best_sharpe, best_return,
                    version_id, error, run_id,
                ),
            )
            conn.commit()
        logger.info(f"Training run {run_id}: {status} (Sharpe={best_sharpe:.2f}, {duration:.0f}s)")

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        """获取最近训练运行."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Evaluations ----

    def record_evaluation(
        self,
        version_id: str,
        eval_type: str,
        sharpe: float = 0.0,
        annual_return: float = 0.0,
        max_drawdown: float = 0.0,
        alpha: float = 0.0,
        positive_alpha_rate: float = 0.0,
        n_windows: int = 0,
        details: dict = None,
    ) -> str:
        """记录评估结果."""
        eval_id = f"eval_{version_id}_{eval_type}_{datetime.now().strftime('%H%M%S')}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO evaluations
                   (eval_id, version_id, eval_type, created_at, sharpe, annual_return,
                    max_drawdown, alpha, positive_alpha_rate, n_windows, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eval_id, version_id, eval_type, datetime.now().isoformat(),
                    sharpe, annual_return, max_drawdown, alpha,
                    positive_alpha_rate, n_windows, json.dumps(details or {}),
                ),
            )
            conn.commit()
        return eval_id

    def get_evaluations(self, version_id: str) -> list[dict]:
        """获取指定版本的所有评估."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE version_id = ? ORDER BY created_at DESC",
                (version_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Daily Status ----

    def record_daily_status(
        self,
        date_str: Optional[str] = None,
        training_triggered: bool = False,
        training_succeeded: bool = False,
        model_promoted: bool = False,
        data_refreshed: bool = False,
        new_data_days: int = 0,
        error: str = "",
        notes: str = "",
    ):
        """记录每日状态."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_status
                   (date, training_triggered, training_succeeded, model_promoted,
                    data_refreshed, new_data_days, error, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date_str,
                    int(training_triggered), int(training_succeeded),
                    int(model_promoted), int(data_refreshed),
                    new_data_days, error, notes,
                ),
            )
            conn.commit()

    def get_daily_status(self, days: int = 30) -> list[dict]:
        """获取最近N天状态."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM daily_status WHERE date >= ? ORDER BY date DESC",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Trends ----

    def get_sharpe_trend(self, days: int = 30) -> list[tuple[str, float]]:
        """获取Sharpe趋势 (date, sharpe)."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT created_at, best_sharpe FROM training_runs
                   WHERE status = 'completed' AND started_at >= ?
                   ORDER BY started_at""",
                (cutoff,),
            ).fetchall()
        return [(r[0][:10], r[1]) for r in rows]

    def get_last_successful_run(self) -> Optional[dict]:
        """获取最近一次成功训练."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM training_runs WHERE status = 'completed' ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_stats_summary(self) -> dict:
        """获取统计摘要."""
        with sqlite3.connect(self.db_path) as conn:
            total_runs = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            successful = conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE status = 'completed'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE status = 'failed'"
            ).fetchone()[0]
            avg_duration = conn.execute(
                "SELECT AVG(duration_seconds) FROM training_runs WHERE status = 'completed'"
            ).fetchone()[0] or 0
            best_sharpe = conn.execute(
                "SELECT MAX(best_sharpe) FROM training_runs WHERE status = 'completed'"
            ).fetchone()[0] or 0

        return {
            "total_runs": total_runs,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / max(total_runs, 1),
            "avg_duration_seconds": avg_duration,
            "best_sharpe_ever": best_sharpe,
        }
