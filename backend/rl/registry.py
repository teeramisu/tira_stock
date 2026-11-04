"""Model Registry — 模型版本管理与晋升逻辑.

管理RL-MACS模型的生命周期：
- 版本注册（candidate → production → archived）
- 指标追踪（Sharpe, Alpha, MaxDD）
- 自动晋升（候选优于生产时晋升）
- 模型文件管理（production/candidates/archive目录）
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rl.registry")

# Default paths
_REGISTRY_DB = Path(__file__).resolve().parents[2] / "data" / "models" / "registry.db"
_MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class ModelMetrics:
    """模型评估指标."""
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    positive_alpha_rate: float = 0.0
    win_rate: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMetrics":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class ModelVersion:
    """模型版本记录."""
    version_id: str
    created_at: str
    model_path: str
    config: dict = field(default_factory=dict)
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    status: str = "candidate"  # candidate | production | archived
    parent_version: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        return d


class ModelRegistry:
    """模型注册中心 — SQLite存储 + 文件管理."""

    def __init__(
        self,
        db_path: Path = _REGISTRY_DB,
        models_dir: Path = _MODELS_DIR,
    ):
        self.db_path = db_path
        self.models_dir = models_dir
        self.production_dir = models_dir / "production"
        self.candidates_dir = models_dir / "candidates"
        self.archive_dir = models_dir / "archive"

        # Ensure directories exist
        for d in [self.production_dir, self.candidates_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    config TEXT DEFAULT '{}',
                    metrics TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'candidate',
                    parent_version TEXT,
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON model_versions(status)
            """)
            conn.commit()

    def _generate_version_id(self) -> str:
        """Generate version ID: v{major}.{date}.{seq}."""
        today = datetime.now().strftime("%Y%m%d")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM model_versions WHERE version_id LIKE ?",
                (f"v%.{today}.%",),
            ).fetchone()
            seq = (row[0] if row else 0) + 1
        return f"v6.{today}.{seq:02d}"

    def register(
        self,
        model_path: str | Path,
        config: dict,
        metrics: ModelMetrics,
        notes: str = "",
        parent_version: Optional[str] = None,
    ) -> ModelVersion:
        """注册新模型版本.

        Args:
            model_path: 模型文件路径 (.zip)
            config: 训练配置快照
            metrics: 评估指标
            notes: 备注
            parent_version: 父版本ID

        Returns:
            ModelVersion 记录
        """
        version_id = self._generate_version_id()
        model_path = Path(model_path)

        # Copy model to candidates directory
        dest = self.candidates_dir / f"{version_id}.zip"
        if model_path.exists():
            shutil.copy2(model_path, dest)
            stored_path = str(dest)
        else:
            stored_path = str(model_path)

        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            model_path=stored_path,
            config=config,
            metrics=metrics,
            status="candidate",
            parent_version=parent_version,
            notes=notes,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO model_versions
                   (version_id, created_at, model_path, config, metrics, status, parent_version, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    version.version_id,
                    version.created_at,
                    version.model_path,
                    json.dumps(version.config),
                    json.dumps(version.metrics.to_dict()),
                    version.status,
                    version.parent_version,
                    version.notes,
                ),
            )
            conn.commit()

        logger.info(f"Registered model: {version_id} (Sharpe={metrics.sharpe:.2f})")
        return version

    def promote(self, version_id: str) -> bool:
        """晋升候选模型为生产模型.

        - 备份当前生产模型到archive
        - 复制候选模型到production
        - 更新状态
        """
        version = self.get_version(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False

        # Archive current production model
        current_prod = self.get_production()
        if current_prod:
            self._archive_version(current_prod)

        # Copy to production
        src = Path(version.model_path)
        prod_path = self.production_dir / "rl_macs_production.zip"
        if src.exists():
            shutil.copy2(src, prod_path)
        else:
            logger.warning(f"Model file not found: {src}")

        # Also save as versioned production copy
        versioned_prod = self.production_dir / f"{version_id}.zip"
        if src.exists():
            shutil.copy2(src, versioned_prod)

        # Update status in DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE model_versions SET status = 'production' WHERE version_id = ?",
                (version_id,),
            )
            conn.commit()

        logger.info(f"Promoted {version_id} to production")
        return True

    def _archive_version(self, version: ModelVersion):
        """Archive a version (move file + update status)."""
        src = Path(version.model_path)
        if src.exists() and src.parent == self.production_dir:
            dest = self.archive_dir / src.name
            shutil.move(str(src), str(dest))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE model_versions SET status = 'archived' WHERE version_id = ?",
                (version.version_id,),
            )
            conn.commit()
        logger.info(f"Archived {version.version_id}")

    def get_production(self) -> Optional[ModelVersion]:
        """获取当前生产模型."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE status = 'production' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return self._row_to_version(row)

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """获取指定版本."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_version(row)

    def get_candidates(self, limit: int = 10) -> list[ModelVersion]:
        """获取最近候选模型."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE status = 'candidate' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_history(self, limit: int = 30) -> list[ModelVersion]:
        """获取版本历史."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def should_promote(
        self,
        candidate: ModelMetrics,
        production: Optional[ModelMetrics] = None,
    ) -> bool:
        """判断是否应该晋升.

        晋升条件（保守策略）：
        1. 无生产模型 → 直接晋升
        2. Sharpe > production * 0.9（不能比生产差太多）
        3. MaxDD < 10%（风险控制）
        4. 正收益率 > 0
        """
        # No production model yet → promote
        if production is None:
            return candidate.sharpe > 0 and candidate.max_drawdown < 0.15

        # Risk control: MaxDD must be reasonable
        if candidate.max_drawdown > 0.10:
            logger.info(f"  Reject: MaxDD {candidate.max_drawdown:.2%} > 10%")
            return False

        # Must be profitable
        if candidate.annual_return <= 0:
            logger.info(f"  Reject: negative return {candidate.annual_return:.2%}")
            return False

        # Sharpe must be at least 90% of production
        threshold = production.sharpe * 0.9
        if candidate.sharpe < threshold:
            logger.info(
                f"  Reject: Sharpe {candidate.sharpe:.2f} < threshold {threshold:.2f}"
            )
            return False

        return True

    def get_production_model_path(self) -> Optional[Path]:
        """获取生产模型文件路径（用于推理）."""
        prod_path = self.production_dir / "rl_macs_production.zip"
        if prod_path.exists():
            return prod_path
        # Fallback: check DB
        version = self.get_production()
        if version:
            p = Path(version.model_path)
            if p.exists():
                return p
        return None

    def _row_to_version(self, row: tuple) -> ModelVersion:
        """Convert DB row to ModelVersion."""
        return ModelVersion(
            version_id=row[0],
            created_at=row[1],
            model_path=row[2],
            config=json.loads(row[3]) if row[3] else {},
            metrics=ModelMetrics.from_dict(json.loads(row[4]) if row[4] else {}),
            status=row[5],
            parent_version=row[6],
            notes=row[7] if row[7] else "",
        )

    def cleanup_old_candidates(self, keep: int = 10):
        """清理旧候选模型（保留最近N个）."""
        candidates = self.get_candidates(limit=100)
        if len(candidates) <= keep:
            return
        to_remove = candidates[keep:]
        for v in to_remove:
            src = Path(v.model_path)
            if src.exists():
                src.unlink()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM model_versions WHERE version_id = ?",
                    (v.version_id,),
                )
                conn.commit()
        logger.info(f"Cleaned up {len(to_remove)} old candidates")
