"""Research platform modules."""

from .governance import (
    AllocationResult,
    DecisionAuditArtifact,
    PortfolioRiskBudget,
    RegimeWindow,
    StrategyMetadata,
    StressScenario,
    ValidationEvidence,
    allocate_capital,
    apply_stress_scenario,
    build_allocation_artifact,
    build_regime_metrics_artifact,
    build_stress_validation_artifact,
    build_validation_gate_artifact,
    can_promote_to_validated,
    slice_metrics_by_regime,
    strategy_logic_depends_on_broker_mode,
    validate_strategy_promotion,
)
from .regime_labels import RegimeLabelsConfig, load_regime_labels

__all__ = [
    "AllocationResult",
    "DecisionAuditArtifact",
    "PortfolioRiskBudget",
    "RegimeLabelsConfig",
    "RegimeWindow",
    "StrategyMetadata",
    "StressScenario",
    "ValidationEvidence",
    "allocate_capital",
    "apply_stress_scenario",
    "build_allocation_artifact",
    "build_regime_metrics_artifact",
    "build_stress_validation_artifact",
    "build_validation_gate_artifact",
    "can_promote_to_validated",
    "load_regime_labels",
    "slice_metrics_by_regime",
    "strategy_logic_depends_on_broker_mode",
    "validate_strategy_promotion",
]
