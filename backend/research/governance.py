"""Strategy governance, risk controls, and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import date
from math import isfinite
from typing import Any, Literal

import pandas as pd


def _safe_std(series: pd.Series) -> float:
    """Return sample std or 0.0 when there are fewer than two observations."""
    if len(series) < 2:
        return 0.0
    value = float(series.std())
    return value if isfinite(value) else 0.0


StrategyState = Literal["idea", "research", "validated", "paper_trading_ready", "deprecated"]
AllocatorMethod = Literal["equal_weight", "risk_parity", "momentum_weighted", "regime_adaptive", "kelly"]


@dataclass(frozen=True)
class ValidationEvidence:
    """Evidence required for strategy promotion checks."""

    out_of_sample_ok: bool
    stress_ok: bool
    cost_sensitivity_ok: bool
    capacity_ok: bool


@dataclass(frozen=True)
class StrategyMetadata:
    """Governance metadata for one strategy."""

    strategy_id: str
    hypothesis: str
    economic_rationale: str
    suitable_regime: str
    failure_mode: str
    risk_notes: str
    model_risk_score: float = 0.0
    state: StrategyState = "idea"


def can_promote_to_validated(
    *,
    out_of_sample_ok: bool,
    stress_ok: bool,
    cost_sensitivity_ok: bool,
    capacity_ok: bool,
    model_risk_score: float,
    max_model_risk: float = 0.7,
) -> bool:
    """Return whether strategy passes validation gate."""
    if model_risk_score > max_model_risk:
        return False
    return all([out_of_sample_ok, stress_ok, cost_sensitivity_ok, capacity_ok])


def validate_strategy_promotion(
    metadata: StrategyMetadata,
    evidence: ValidationEvidence,
    *,
    target_state: StrategyState,
    max_model_risk: float = 0.7,
) -> tuple[bool, list[str]]:
    """Validate whether metadata can be promoted to target state."""
    if target_state != "validated":
        return True, []
    reasons: list[str] = []
    if metadata.model_risk_score > max_model_risk:
        reasons.append("high_model_risk")
    if not can_promote_to_validated(
        out_of_sample_ok=evidence.out_of_sample_ok,
        stress_ok=evidence.stress_ok,
        cost_sensitivity_ok=evidence.cost_sensitivity_ok,
        capacity_ok=evidence.capacity_ok,
        model_risk_score=metadata.model_risk_score,
        max_model_risk=max_model_risk,
    ):
        if not evidence.out_of_sample_ok:
            reasons.append("missing_out_of_sample")
        if not evidence.stress_ok:
            reasons.append("stress_test_failed")
        if not evidence.cost_sensitivity_ok:
            reasons.append("cost_sensitivity_failed")
        if not evidence.capacity_ok:
            reasons.append("capacity_failed")
    return len(reasons) == 0, reasons


@dataclass(frozen=True)
class RegimeWindow:
    """Manual regime label window."""

    label: str
    start: date
    end: date


def slice_metrics_by_regime(equity_curve: pd.DataFrame, regimes: tuple[RegimeWindow, ...]) -> dict[str, dict[str, float]]:
    """Slice return/drawdown/volatility by configured regimes."""
    if equity_curve.empty:
        return {}
    frame = equity_curve.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame = frame.sort_values("trade_date")
    results: dict[str, dict[str, float]] = {}
    for regime in regimes:
        chunk = frame[(frame["trade_date"] >= regime.start) & (frame["trade_date"] <= regime.end)]
        if chunk.empty:
            results[regime.label] = _empty_regime_metrics(frame)
            continue
        returns = chunk["equity"].pct_change().dropna()
        total_return = float(chunk["equity"].iloc[-1] / chunk["equity"].iloc[0] - 1.0)
        volatility = _safe_std(returns)
        running_peak = chunk["equity"].cummax()
        drawdown = 1.0 - chunk["equity"] / running_peak
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
        metrics: dict[str, float] = {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "win_rate": win_rate,
        }
        if "turnover" in chunk.columns:
            metrics["turnover"] = float(chunk["turnover"].sum())
        if "cost" in chunk.columns:
            metrics["cost"] = float(chunk["cost"].sum())
        results[regime.label] = metrics
    return results


def _empty_regime_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics = {"total_return": 0.0, "max_drawdown": 0.0, "volatility": 0.0, "win_rate": 0.0}
    if "turnover" in frame.columns:
        metrics["turnover"] = 0.0
    if "cost" in frame.columns:
        metrics["cost"] = 0.0
    return metrics


def build_regime_metrics_artifact(
    equity_curve: pd.DataFrame,
    regimes: tuple[RegimeWindow, ...],
    *,
    market: str = "cn",
) -> dict[str, Any]:
    """Build regime metrics artifact for validation persistence."""
    return {
        "artifact_type": "regime_metrics_v1",
        "market": market,
        "by_regime": slice_metrics_by_regime(equity_curve, regimes),
    }


@dataclass(frozen=True)
class StressScenario:
    """Stress scenario for validation artifact generation."""

    name: str
    cost_multiplier: float = 1.0
    liquidity_haircut: float = 0.0
    volatility_multiplier: float = 1.0
    correlation_shock: float = 0.0


def apply_stress_scenario(
    *,
    base_return: float,
    base_cost: float,
    scenario: StressScenario,
) -> tuple[dict[str, float], dict[str, object]]:
    """Apply a simplified stress transform and return diagnostics."""
    stressed_cost = base_cost * max(0.0, scenario.cost_multiplier)
    liquidity_penalty = max(0.0, min(1.0, scenario.liquidity_haircut))
    volatility_penalty = max(0.0, scenario.volatility_multiplier - 1.0) * 0.1
    correlation_penalty = max(0.0, scenario.correlation_shock) * 0.1
    stressed_return = base_return * (1.0 - liquidity_penalty) - volatility_penalty - correlation_penalty
    diagnostics = {
        "scenario": scenario.name,
        "failed": stressed_return < base_return,
        "assumptions": {
            "cost_multiplier": scenario.cost_multiplier,
            "liquidity_haircut": scenario.liquidity_haircut,
            "volatility_multiplier": scenario.volatility_multiplier,
            "correlation_shock": scenario.correlation_shock,
        },
    }
    return {"total_return": stressed_return, "total_cost": stressed_cost}, diagnostics


def build_stress_validation_artifact(
    *,
    base_return: float,
    base_cost: float,
    scenarios: tuple[StressScenario, ...],
) -> dict[str, Any]:
    """Build validation artifact with stress diagnostics."""
    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, object]] = []
    for scenario in scenarios:
        metrics, diag = apply_stress_scenario(
            base_return=base_return,
            base_cost=base_cost,
            scenario=scenario,
        )
        results.append({"scenario": scenario.name, "metrics": metrics})
        diagnostics.append(diag)
    return {
        "artifact_type": "stress_validation_v1",
        "base": {"total_return": base_return, "total_cost": base_cost},
        "scenarios": results,
        "diagnostics": diagnostics,
        "passed": all(not diag.get("failed") for diag in diagnostics),
    }


@dataclass(frozen=True)
class PortfolioRiskBudget:
    """Risk constraints for capital allocation."""

    max_single_weight: float = 0.6
    max_gross_exposure: float = 1.0


@dataclass(frozen=True)
class AllocationResult:
    """Allocator output with diagnostics for artifact persistence."""

    weights: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)


def allocate_capital(
    *,
    strategy_scores: dict[str, float],
    budget: PortfolioRiskBudget,
    method: AllocatorMethod = "equal_weight",
    strategy_metrics: dict[str, dict[str, float]] | None = None,
    regime: str = "UNKNOWN",
) -> AllocationResult:
    """Allocate capital with risk-budget constraints.

    Supports multiple allocation methods by delegating to MetaAllocator.
    strategy_metrics should contain per-strategy volatility, sharpe, etc.
    """
    if not strategy_scores:
        return AllocationResult(weights={}, diagnostics={"gross_exposure": 0.0})

    if method == "equal_weight":
        n = len(strategy_scores)
        raw_weight = 1.0 / n
        capped = min(raw_weight, budget.max_single_weight)
        weights = {sid: capped for sid in strategy_scores}
    else:
        # Delegate to MetaAllocator for advanced methods
        try:
            from research.meta_allocator import MetaAllocator, SleeveMetrics

            sleeves = {}
            for sid in strategy_scores:
                m = (strategy_metrics or {}).get(sid, {})
                sleeves[sid] = SleeveMetrics(
                    strategy_id=sid,
                    total_return=float(m.get("total_return", strategy_scores.get(sid, 0.0))),
                    sharpe_ratio=float(m.get("sharpe_ratio", 0.0)),
                    max_drawdown=float(m.get("max_drawdown", 0.0)),
                    volatility=float(m.get("annualized_volatility", m.get("volatility", 0.2))),
                    win_rate=float(m.get("win_rate", 0.5)),
                    trade_count=int(m.get("trade_count", 0)),
                )
            allocator = MetaAllocator()
            result = allocator.allocate(
                sleeves,
                method=method,
                regime=regime,
                max_single_weight=budget.max_single_weight,
                min_weight=0.0,
            )
            weights = dict(result.weights)
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.warning("MetaAllocator %s failed (%s), falling back to equal_weight", method, e)
            n = len(strategy_scores)
            raw_weight = 1.0 / n
            capped = min(raw_weight, budget.max_single_weight)
            weights = {sid: capped for sid in strategy_scores}

    gross = float(sum(weights.values()))
    if gross > budget.max_gross_exposure and gross > 0:
        scale = budget.max_gross_exposure / gross
        weights = {sid: weight * scale for sid, weight in weights.items()}
        gross = float(sum(weights.values()))
    return AllocationResult(
        weights=weights,
        diagnostics={
            "gross_exposure": gross,
            "max_weight": max(weights.values()) if weights else 0.0,
            "method": method,
        },
    )


def build_allocation_artifact(
    *,
    strategy_scores: dict[str, float],
    budget: PortfolioRiskBudget,
    method: AllocatorMethod = "equal_weight",
    strategy_metrics: dict[str, dict[str, float]] | None = None,
    regime: str = "UNKNOWN",
) -> dict[str, Any]:
    """Build allocation artifact with inputs, constraints, and weights."""
    result = allocate_capital(
        strategy_scores=strategy_scores,
        budget=budget,
        method=method,
        strategy_metrics=strategy_metrics,
        regime=regime,
    )
    return {
        "artifact_type": "allocation_v1",
        "method": method,
        "inputs": {"strategy_scores": strategy_scores},
        "constraints": {
            "max_single_weight": budget.max_single_weight,
            "max_gross_exposure": budget.max_gross_exposure,
        },
        "target_weights": result.weights,
        "actual_weights": result.weights,
        "diagnostics": result.diagnostics,
    }


@dataclass(frozen=True)
class DecisionAuditArtifact:
    """Single-date decision audit payload."""

    trade_date: date
    filter_reasons: dict[str, str]
    order_reasons: dict[str, str]

    def has_complete_reasons(self) -> bool:
        """Return True when both filter and order reasons are present."""
        return bool(self.filter_reasons) and bool(self.order_reasons)


def strategy_logic_depends_on_broker_mode(outputs_by_mode: dict[str, tuple[str, ...]]) -> bool:
    """Detect whether strategy output changes across broker modes."""
    if not outputs_by_mode:
        return False
    first = next(iter(outputs_by_mode.values()))
    return any(result != first for result in outputs_by_mode.values())


def build_validation_gate_artifact(
    metadata: StrategyMetadata,
    evidence: ValidationEvidence,
    *,
    target_state: StrategyState = "validated",
) -> dict[str, Any]:
    """Build promotion gate artifact for strategy governance."""
    allowed, reasons = validate_strategy_promotion(metadata, evidence, target_state=target_state)
    return {
        "artifact_type": "validation_gate_v1",
        "strategy_id": metadata.strategy_id,
        "target_state": target_state,
        "allowed": allowed,
        "reasons": reasons,
        "evidence": {
            "out_of_sample_ok": evidence.out_of_sample_ok,
            "stress_ok": evidence.stress_ok,
            "cost_sensitivity_ok": evidence.cost_sensitivity_ok,
            "capacity_ok": evidence.capacity_ok,
        },
    }

