"""Multi-strategy portfolio capital allocation (allocator-required)."""

from __future__ import annotations

from typing import Any

from .governance import (
    AllocatorMethod,
    PortfolioRiskBudget,
    build_allocation_artifact,
)

NAIVE_SUM_METHOD = "naive_sum"


def allocate_multi_strategy(
    *,
    strategy_scores: dict[str, float],
    budget: PortfolioRiskBudget,
    method: str = "equal_weight",
    strategy_metrics: dict[str, dict[str, float]] | None = None,
    regime: str = "UNKNOWN",
) -> dict[str, Any]:
    """Allocate capital across strategy sleeves; naive sum is forbidden."""
    if method == NAIVE_SUM_METHOD:
        raise ValueError("multi-strategy portfolio must use allocator; naive sum is forbidden")
    allocator_method: AllocatorMethod = "equal_weight"
    if method in ("equal_weight", "risk_parity", "momentum_weighted", "regime_adaptive", "kelly"):
        allocator_method = method  # type: ignore[assignment]
    else:
        raise ValueError(f"unsupported allocator method: {method}")
    return build_allocation_artifact(
        strategy_scores=strategy_scores,
        budget=budget,
        method=allocator_method,
        strategy_metrics=strategy_metrics,
        regime=regime,
    )


def combine_strategy_sleeve_metrics(
    sleeve_metrics: dict[str, dict[str, float]],
    *,
    budget: PortfolioRiskBudget,
    method: str = "equal_weight",
    score_key: str = "total_return",
    regime: str = "UNKNOWN",
    regime_by_sleeve: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build allocation artifact from per-sleeve backtest metrics."""
    strategy_scores = {
        sleeve_id: float(metrics.get(score_key, 0.0)) for sleeve_id, metrics in sleeve_metrics.items()
    }
    return allocate_multi_strategy(
        strategy_scores=strategy_scores,
        budget=budget,
        method=method,
        strategy_metrics=sleeve_metrics,
        regime=regime,
    )
