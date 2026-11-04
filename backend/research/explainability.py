"""Explainability artifacts for research/backtest decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionExplainabilityArtifact:
    """Per-period explainability payload."""

    trade_date: str
    select_reasons: dict[str, str]
    factor_breakdown: dict[str, float]
    filter_reason_stats: dict[str, int]
    return_decomposition: dict[str, float] = field(
        default_factory=lambda: {
            "selection": 0.0,
            "timing": 0.0,
            "cost": 0.0,
            "cash_drag": 0.0,
        }
    )
