"""Central registry for portfolio backtest strategies."""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from backtest.imported_strategies import ADAPTED_PORTFOLIO_STRATEGIES
from portfolio.jq_strategies import (
    EtfMomentumRsrsTop1Strategy,
    EtfPolyfitMomentumTop1Strategy,
    JqEtfMomentumVolumeGateTop1Strategy,
    JqEtfValueAveragingStrategy,
    JqIndustryEtfBbiRotationStrategy,
    JqMaConsolidationBreakoutStrategy,
    JqSimpleMomentumStrategy,
    JqSmallCapTopKStrategy,
    JqWideBaseEtfBbiRotationStrategy,
    WpetfMomentumTop1Strategy,
)
from portfolio.strategies import (
    EtfRotationTop1Strategy,
    JqValueTimingTopKStrategy,
    MeanReversionTopKStrategy,
    MomentumTopKStrategy,
    MultiFactorTopKStrategy,
    RiskAdjustedMomentumStrategy,
)


class _PortfolioRequest(Protocol):
    strategy: str
    lookback_days: int
    max_positions: int


StrategyBuilder = Callable[[_PortfolioRequest], Any]

PORTFOLIO_STRATEGY_BUILDERS: dict[str, StrategyBuilder] = {
    "momentum_top_k": lambda req: MomentumTopKStrategy(
        lookback_days=req.lookback_days,
        top_k=req.max_positions,
    ),
    "etf_rotation_top1": lambda req: EtfRotationTop1Strategy(lookback_days=req.lookback_days),
    "jq_value_timing_top_k": lambda req: JqValueTimingTopKStrategy(
        top_k=req.max_positions,
        lookback_days=req.lookback_days,
    ),
    "jq_etf_polyfit_momentum_top1": lambda req: EtfPolyfitMomentumTop1Strategy(
        lookback_days=req.lookback_days,
    ),
    "jq_etf_momentum_rsrs_top1": lambda req: EtfMomentumRsrsTop1Strategy(
        lookback_days=req.lookback_days,
    ),
    "jq_small_cap_top_k": lambda req: JqSmallCapTopKStrategy(
        top_k=req.max_positions,
        lookback_days=req.lookback_days,
    ),
    "jq_industry_etf_bbi_rotation": lambda req: JqIndustryEtfBbiRotationStrategy(),
    "jq_etf_momentum_volume_gate_top1": lambda req: JqEtfMomentumVolumeGateTop1Strategy(
        period_bars=req.lookback_days,
    ),
    "jq_wide_base_etf_bbi_rotation": lambda req: JqWideBaseEtfBbiRotationStrategy(),
    "jq_etf_value_averaging": lambda req: JqEtfValueAveragingStrategy(),
    "jq_wpetf_momentum_top1": lambda req: WpetfMomentumTop1Strategy(lookback_days=req.lookback_days),
    "jq_ma_consolidation_breakout": lambda req: JqMaConsolidationBreakoutStrategy(
        max_positions=req.max_positions,
    ),
    "jq_simple_momentum": lambda req: JqSimpleMomentumStrategy(
        lookback_days=req.lookback_days,
        top_k=req.max_positions,
    ),
    "mean_reversion_top_k": lambda req: MeanReversionTopKStrategy(
        lookback_days=req.lookback_days,
        top_k=req.max_positions,
    ),
    "multi_factor_top_k": lambda req: MultiFactorTopKStrategy(
        lookback_days=req.lookback_days,
        top_k=req.max_positions,
    ),
    "risk_adj_momentum_top_k": lambda req: RiskAdjustedMomentumStrategy(
        lookback_days=req.lookback_days,
        top_k=req.max_positions,
    ),
}

WEEKLY_WHEN_DAILY = frozenset(
    {
        "jq_value_timing_top_k",
        "jq_small_cap_top_k",
        "jq_industry_etf_bbi_rotation",
        "mean_reversion_top_k",
        "multi_factor_top_k",
    }
)

# Strategies that must run on an ETF pool, not the full A-share stock universe.
ETF_POOL_STRATEGY_IDS = frozenset(
    {
        "etf_rotation_top1",
        "jq_etf_polyfit_momentum_top1",
        "jq_etf_momentum_rsrs_top1",
        "jq_industry_etf_bbi_rotation",
        "jq_etf_momentum_volume_gate_top1",
        "jq_wide_base_etf_bbi_rotation",
        "jq_etf_value_averaging",
        "jq_wpetf_momentum_top1",
    }
)

DEFAULT_ETF_ROTATION_SYMBOLS: tuple[str, ...] = (
    "510300",
    "510500",
    "159915",
    "513100",
    "518880",
    "510180",
)


def default_symbols_for_strategy(strategy_id: str) -> tuple[str, ...]:
    """Return preset ETF/symbol pool from adapted strategy metadata."""
    meta = ADAPTED_PORTFOLIO_STRATEGIES.get(strategy_id)
    if meta:
        preset = meta.get("run_preset") or {}
        raw = preset.get("symbols")
        if isinstance(raw, str) and raw.strip():
            return tuple(part.strip() for part in raw.split(",") if part.strip())
    if strategy_id == "etf_rotation_top1":
        return DEFAULT_ETF_ROTATION_SYMBOLS
    return ()


def resolve_portfolio_symbols(strategy_id: str, symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Resolve universe symbols; ETF strategies fall back to preset pool when empty."""
    explicit = tuple(str(s).strip() for s in symbols if str(s).strip())
    if explicit:
        return explicit
    key = (strategy_id or "").strip()
    if key in ETF_POOL_STRATEGY_IDS:
        defaults = default_symbols_for_strategy(key)
        if defaults:
            return defaults
    return ()


def is_etf_pool_strategy(strategy_id: str) -> bool:
    return (strategy_id or "").strip() in ETF_POOL_STRATEGY_IDS


def supported_portfolio_strategies() -> frozenset[str]:
    """Return strategy ids accepted by the portfolio backtest API."""
    return frozenset(PORTFOLIO_STRATEGY_BUILDERS.keys())


def _apply_optimized_params(req: _PortfolioRequest) -> None:
    """Override request params with walk-forward optimized values when available."""
    try:
        from research.walk_forward import load_optimized_params

        optimized = load_optimized_params(req.strategy)
        if not optimized:
            return
        # Only override params that exist on the request
        for k, v in optimized.items():
            attr = k
            if k == "top_k":
                attr = "max_positions"
            if hasattr(req, attr) and v is not None:
                try:
                    setattr(req, attr, type(getattr(req, attr))(v))
                    logger.info("WF param override %s.%s = %s", req.strategy, attr, v)
                except (TypeError, ValueError):
                    pass
    except Exception:
        logger.debug("No WF params for %s", req.strategy)


def build_portfolio_strategy(req: _PortfolioRequest) -> Any:
    """Instantiate a portfolio strategy from a backtest request.

    When walk-forward optimized params are available (from a previous
    /api/optimize/walkforward run), they automatically override defaults.
    """
    key = (req.strategy or "").strip()
    builder = PORTFOLIO_STRATEGY_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"暂不支持组合策略: {key}")
    _apply_optimized_params(req)
    return builder(req)


def resolve_rebalance_frequency(strategy_id: str, rebalance_frequency: str) -> str:
    """Apply strategy-specific rebalance defaults."""
    if strategy_id in WEEKLY_WHEN_DAILY and rebalance_frequency == "daily":
        return "weekly"
    return rebalance_frequency


# Strategy metadata: regime affinity + risk level
# Each strategy declares which regimes it performs best in.
STRATEGY_REGIME_AFFINITY: dict[str, dict[str, object]] = {
    "momentum_top_k":          {"best_regimes": ["BULL"],       "risk": "high",   "type": "stock"},
    "multi_factor_top_k":      {"best_regimes": ["BULL"],       "risk": "medium", "type": "stock"},
    "risk_adj_momentum_top_k": {"best_regimes": ["BULL"],       "risk": "medium", "type": "stock"},
    "mean_reversion_top_k":    {"best_regimes": ["RANGE"],      "risk": "high",   "type": "stock"},
    "jq_value_timing_top_k":   {"best_regimes": ["BULL"],       "risk": "medium", "type": "stock"},
    "jq_small_cap_top_k":      {"best_regimes": ["RANGE"],      "risk": "high",   "type": "stock"},
    "jq_ma_consolidation_breakout": {"best_regimes": ["RANGE"], "risk": "medium", "type": "stock"},
    "jq_etf_momentum_rsrs_top1":    {"best_regimes": ["BULL", "BEAR"], "risk": "low", "type": "etf"},
    "jq_wide_base_etf_bbi_rotation": {"best_regimes": ["RANGE", "BEAR"], "risk": "low", "type": "etf"},
    "jq_industry_etf_bbi_rotation":  {"best_regimes": ["RANGE", "BULL"], "risk": "medium", "type": "etf"},
    "jq_etf_value_averaging":        {"best_regimes": ["BEAR", "CRASH"], "risk": "low", "type": "etf"},
    "jq_etf_polyfit_momentum_top1":  {"best_regimes": ["BULL"],   "risk": "medium", "type": "etf"},
    "jq_wpetf_momentum_top1":        {"best_regimes": ["BULL"],   "risk": "medium", "type": "etf"},
    "etf_rotation_top1":             {"best_regimes": ["BULL"],   "risk": "medium", "type": "etf"},
}


def strategies_for_regime(regime: str) -> list[str]:
    """Return strategy IDs recommended for the given regime, sorted by affinity."""
    matched = [
        sid for sid, meta in STRATEGY_REGIME_AFFINITY.items()
        if regime in meta.get("best_regimes", [])
    ]
    # Sort: low-risk first in BEAR/CRASH, high-risk first in BULL
    if regime in ("BEAR", "CRASH"):
        matched.sort(key=lambda s: STRATEGY_REGIME_AFFINITY[s].get("risk", "high"))
    else:
        matched.sort(key=lambda s: {"high": 0, "medium": 1, "low": 2}.get(
            str(STRATEGY_REGIME_AFFINITY[s].get("risk", "medium")), 1
        ))
    return matched


def missing_adapted_portfolio_builders() -> list[str]:
    """Adapted portfolio strategies without a runtime builder (should stay empty)."""
    missing: list[str] = []
    for strategy_id in ADAPTED_PORTFOLIO_STRATEGIES:
        if strategy_id not in PORTFOLIO_STRATEGY_BUILDERS:
            missing.append(strategy_id)
    return missing
