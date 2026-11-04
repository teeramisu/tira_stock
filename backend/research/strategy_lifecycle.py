"""Strategy lifecycle management: state machine + degradation detection.

States:
  research → validated → production → degraded → retired
     ↑                                          │
     └────────── retrain/update ←───────────────┘

Degradation rules:
- Rolling 20-day sharpe < 0 AND max_drawdown > 15% → degraded
- 3 consecutive degraded periods → retired

Usage:
    from research.strategy_lifecycle import StrategyLifecycleManager
    mgr = StrategyLifecycleManager()
    mgr.record_run("momentum_top_k", metrics={"sharpe_ratio": -0.5, ...})
    state = mgr.get_state("momentum_top_k")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LIFECYCLE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "strategy_lifecycle.json"


class StrategyState(str, Enum):
    """Strategy lifecycle states."""

    RESEARCH = "research"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEGRADED = "degraded"
    RETIRED = "retired"


# Valid state transitions
_VALID_TRANSITIONS: dict[StrategyState, set[StrategyState]] = {
    StrategyState.RESEARCH: {StrategyState.VALIDATED},
    StrategyState.VALIDATED: {StrategyState.PRODUCTION, StrategyState.RESEARCH},
    StrategyState.PRODUCTION: {StrategyState.DEGRADED, StrategyState.VALIDATED},
    StrategyState.DEGRADED: {StrategyState.RETIRED, StrategyState.VALIDATED, StrategyState.RESEARCH},
    StrategyState.RETIRED: {StrategyState.RESEARCH},
}


@dataclass
class StrategyScorecard:
    """One backtest run's performance record."""

    run_date: str
    regime: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRecord:
    """Full lifecycle record for one strategy."""

    strategy_id: str
    state: StrategyState = StrategyState.RESEARCH
    scorecards: list[StrategyScorecard] = field(default_factory=list)
    degraded_count: int = 0  # consecutive degraded periods
    last_updated: str = ""
    promoted_at: str = ""
    retired_at: str = ""
    notes: str = ""


class StrategyLifecycleManager:
    """Manage strategy lifecycle state transitions and degradation detection."""

    def __init__(self, db_path: Path = LIFECYCLE_DB_PATH):
        self.db_path = db_path
        self._records: dict[str, StrategyRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load records from JSON file."""
        if not self.db_path.exists():
            return
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            for sid, rec in data.items():
                self._records[sid] = StrategyRecord(
                    strategy_id=sid,
                    state=StrategyState(rec.get("state", "research")),
                    scorecards=[
                        StrategyScorecard(**sc) for sc in rec.get("scorecards", [])
                    ],
                    degraded_count=rec.get("degraded_count", 0),
                    last_updated=rec.get("last_updated", ""),
                    promoted_at=rec.get("promoted_at", ""),
                    retired_at=rec.get("retired_at", ""),
                    notes=rec.get("notes", ""),
                )
        except Exception as e:
            logger.warning("Failed to load lifecycle DB: %s", e)

    def _save(self) -> None:
        """Persist records to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for sid, rec in self._records.items():
            data[sid] = {
                "strategy_id": rec.strategy_id,
                "state": rec.state.value,
                "scorecards": [
                    {
                        "run_date": sc.run_date,
                        "regime": sc.regime,
                        "total_return": sc.total_return,
                        "sharpe_ratio": sc.sharpe_ratio,
                        "max_drawdown": sc.max_drawdown,
                        "win_rate": sc.win_rate,
                        "trade_count": sc.trade_count,
                        "params": sc.params,
                    }
                    for sc in rec.scorecards
                ],
                "degraded_count": rec.degraded_count,
                "last_updated": rec.last_updated,
                "promoted_at": rec.promoted_at,
                "retired_at": rec.retired_at,
                "notes": rec.notes,
            }
        self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_record(self, strategy_id: str) -> StrategyRecord:
        """Get or create a strategy record."""
        if strategy_id not in self._records:
            self._records[strategy_id] = StrategyRecord(strategy_id=strategy_id)
        return self._records[strategy_id]

    def get_state(self, strategy_id: str) -> StrategyState:
        """Get current state of a strategy."""
        return self.get_record(strategy_id).state

    def record_run(
        self,
        strategy_id: str,
        *,
        metrics: dict[str, Any],
        regime: str = "UNKNOWN",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a backtest run and check for degradation.

        Args:
            strategy_id: Strategy identifier.
            metrics: Backtest metrics (sharpe_ratio, max_drawdown, etc.).
            regime: Current market regime.
            params: Strategy parameters used.

        Returns:
            Dict with state, degradation info, and recommendation.
        """
        rec = self.get_record(strategy_id)
        now = datetime.now().isoformat()

        scorecard = StrategyScorecard(
            run_date=date.today().isoformat(),
            regime=regime,
            total_return=float(metrics.get("total_return", 0)),
            sharpe_ratio=float(metrics.get("sharpe_ratio", 0)),
            max_drawdown=float(metrics.get("max_drawdown", 0)),
            win_rate=float(metrics.get("win_rate", 0)),
            trade_count=int(metrics.get("trade_count", 0)),
            params=params or {},
        )
        rec.scorecards.append(scorecard)
        rec.last_updated = now

        # Check degradation
        is_degraded = self._check_degradation(rec)
        result: dict[str, Any] = {
            "strategy_id": strategy_id,
            "state": rec.state.value,
            "is_degraded": is_degraded,
            "degraded_count": rec.degraded_count,
        }

        if is_degraded:
            rec.degraded_count += 1
            if rec.state == StrategyState.PRODUCTION:
                self._transition(rec, StrategyState.DEGRADED)
                result["state"] = rec.state.value
                result["action"] = "策略降级：检测到性能衰退"
            elif rec.state == StrategyState.DEGRADED:
                if rec.degraded_count >= 3:
                    self._transition(rec, StrategyState.RETIRED)
                    result["state"] = rec.state.value
                    result["action"] = "策略退役：连续3个周期衰退"
                else:
                    result["action"] = f"持续衰退 ({rec.degraded_count}/3)"
        else:
            rec.degraded_count = 0
            if rec.state == StrategyState.DEGRADED:
                self._transition(rec, StrategyState.VALIDATED)
                result["state"] = rec.state.value
                result["action"] = "策略恢复：性能改善"

        # Auto-promote from research after 3 good runs
        if rec.state == StrategyState.RESEARCH and len(rec.scorecards) >= 3:
            recent = rec.scorecards[-3:]
            avg_sharpe = sum(sc.sharpe_ratio for sc in recent) / 3
            if avg_sharpe > 0.5:
                self._transition(rec, StrategyState.VALIDATED)
                result["state"] = rec.state.value
                result["action"] = f"策略验证通过 (avg_sharpe={avg_sharpe:.2f})"

        self._save()
        return result

    def _check_degradation(self, rec: StrategyRecord) -> bool:
        """Check if latest run indicates degradation.

        Rules:
        - Rolling sharpe < 0 AND max_drawdown > 15%
        """
        if not rec.scorecards:
            return False
        latest = rec.scorecards[-1]
        # Degraded if negative sharpe AND significant drawdown
        return latest.sharpe_ratio < 0 and abs(latest.max_drawdown) > 0.15

    def _transition(self, rec: StrategyRecord, new_state: StrategyState) -> bool:
        """Attempt state transition, return True if successful."""
        valid = _VALID_TRANSITIONS.get(rec.state, set())
        if new_state not in valid:
            logger.warning(
                "Invalid transition: %s → %s for %s",
                rec.state.value,
                new_state.value,
                rec.strategy_id,
            )
            return False

        old_state = rec.state
        rec.state = new_state
        now = datetime.now().isoformat()

        if new_state == StrategyState.VALIDATED:
            rec.promoted_at = now
        elif new_state == StrategyState.RETIRED:
            rec.retired_at = now

        logger.info(
            "Strategy %s: %s → %s",
            rec.strategy_id,
            old_state.value,
            new_state.value,
        )
        return True

    def promote(self, strategy_id: str, target: StrategyState) -> bool:
        """Manually promote a strategy to a new state."""
        rec = self.get_record(strategy_id)
        success = self._transition(rec, target)
        if success:
            self._save()
        return success

    def get_active_strategies(self) -> list[str]:
        """Return strategies in PRODUCTION or VALIDATED state."""
        return [
            sid
            for sid, rec in self._records.items()
            if rec.state in (StrategyState.PRODUCTION, StrategyState.VALIDATED)
        ]

    def get_degraded_strategies(self) -> list[str]:
        """Return strategies in DEGRADED state."""
        return [
            sid
            for sid, rec in self._records.items()
            if rec.state == StrategyState.DEGRADED
        ]

    def get_retired_strategies(self) -> list[str]:
        """Return strategies in RETIRED state."""
        return [
            sid
            for sid, rec in self._records.items()
            if rec.state == StrategyState.RETIRED
        ]

    def summary(self) -> dict[str, Any]:
        """Return summary of all strategy states."""
        states: dict[str, list[str]] = {s.value: [] for s in StrategyState}
        for sid, rec in self._records.items():
            states[rec.state.value].append(sid)
        return {
            "total_strategies": len(self._records),
            "by_state": {k: v for k, v in states.items() if v},
            "recent_runs": sum(len(r.scorecards) for r in self._records.values()),
        }
