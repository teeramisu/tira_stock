"""Meta-allocator: intelligent capital allocation across strategy sleeves.

Allocation methods:
1. Equal Weight:        baseline, 1/N allocation
2. Risk Parity:         allocate inversely proportional to volatility
3. Momentum-Weighted:   weight by recent risk-adjusted return (Sharpe)
4. Regime-Adaptive:     combine regime affinity + momentum + risk parity
5. Kelly Criterion:     fractional Kelly based on win rate and payoff ratio

Usage:
    from research.meta_allocator import MetaAllocator
    alloc = MetaAllocator()
    result = alloc.allocate(sleeve_metrics, method="regime_adaptive", regime="BULL")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SleeveMetrics:
    """Performance metrics for one strategy sleeve."""

    strategy_id: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0  # annualized
    win_rate: float = 0.0
    trade_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    regime: str = "UNKNOWN"


@dataclass
class AllocationResult:
    """Result of meta-allocation."""

    method: str
    weights: dict[str, float]  # strategy_id → weight (0-1)
    expected_sharpe: float
    expected_return: float
    diversification_ratio: float  # > 1 means diversification benefit
    risk_contributions: dict[str, float]
    reasoning: list[str]


class MetaAllocator:
    """Intelligent capital allocation across strategy sleeves."""

    def allocate(
        self,
        sleeves: dict[str, SleeveMetrics],
        *,
        method: str = "regime_adaptive",
        regime: str = "UNKNOWN",
        max_single_weight: float = 0.40,
        min_weight: float = 0.05,
    ) -> AllocationResult:
        """Allocate capital across strategy sleeves.

        Args:
            sleeves: Dict of strategy_id → SleeveMetrics.
            method: Allocation method.
            regime: Current market regime for regime-adaptive method.
            max_single_weight: Maximum weight for any single strategy.
            min_weight: Minimum weight for included strategies.

        Returns:
            AllocationResult with weights and diagnostics.
        """
        if not sleeves:
            return AllocationResult(
                method=method,
                weights={},
                expected_sharpe=0,
                expected_return=0,
                diversification_ratio=1.0,
                risk_contributions={},
                reasoning=["无策略可分配"],
            )

        methods = {
            "equal_weight": self._equal_weight,
            "risk_parity": self._risk_parity,
            "momentum_weighted": self._momentum_weighted,
            "regime_adaptive": self._regime_adaptive,
            "kelly": self._kelly_criterion,
        }
        fn = methods.get(method, self._regime_adaptive)
        raw_weights = fn(sleeves, regime=regime)

        # Apply constraints: clip to [min_weight, max_single_weight]
        weights = self._apply_constraints(
            raw_weights, min_w=min_weight, max_w=max_single_weight
        )

        # Compute diagnostics
        risk_contribs = self._compute_risk_contributions(weights, sleeves)
        exp_sharpe = self._estimate_portfolio_sharpe(weights, sleeves)
        exp_return = sum(w * sleeves[sid].total_return for sid, w in weights.items() if sid in sleeves)
        div_ratio = self._diversification_ratio(weights, sleeves)

        reasoning = self._generate_reasoning(method, weights, sleeves, regime, exp_sharpe)

        return AllocationResult(
            method=method,
            weights=weights,
            expected_sharpe=exp_sharpe,
            expected_return=exp_return,
            diversification_ratio=div_ratio,
            risk_contributions=risk_contribs,
            reasoning=reasoning,
        )

    def _equal_weight(self, sleeves: dict[str, SleeveMetrics], **_: Any) -> dict[str, float]:
        """1/N allocation."""
        n = len(sleeves)
        return {sid: 1.0 / n for sid in sleeves}

    def _risk_parity(self, sleeves: dict[str, SleeveMetrics], **_: Any) -> dict[str, float]:
        """Inverse-volatility weighting (risk parity approximation)."""
        inv_vols = {}
        for sid, m in sleeves.items():
            vol = max(abs(m.volatility), 0.01)  # floor at 1%
            inv_vols[sid] = 1.0 / vol

        total = sum(inv_vols.values())
        return {sid: v / total for sid, v in inv_vols.items()}

    def _momentum_weighted(self, sleeves: dict[str, SleeveMetrics], **_: Any) -> dict[str, float]:
        """Weight by recent Sharpe ratio (risk-adjusted momentum)."""
        scores = {}
        for sid, m in sleeves.items():
            # Use Sharpe but floor at 0 to avoid negative weights
            scores[sid] = max(m.sharpe_ratio, 0.0) + 0.1  # small base weight

        total = sum(scores.values())
        if total < 1e-8:
            return self._equal_weight(sleeves)
        return {sid: s / total for sid, s in scores.items()}

    def _regime_adaptive(
        self, sleeves: dict[str, SleeveMetrics], *, regime: str = "UNKNOWN", **_: Any
    ) -> dict[str, float]:
        """Combine regime affinity, momentum, and risk parity."""
        from portfolio.strategy_registry import STRATEGY_REGIME_AFFINITY

        scores: dict[str, float] = {}
        for sid, m in sleeves.items():
            # Component 1: Regime affinity bonus (0.5 or 1.5)
            affinity = STRATEGY_REGIME_AFFINITY.get(sid, {})
            best_regimes = affinity.get("best_regimes", [])
            regime_bonus = 1.5 if regime in best_regimes else 0.5

            # Component 2: Sharpe momentum (0-2 range)
            sharpe_score = max(m.sharpe_ratio + 1.0, 0.0)

            # Component 3: Risk parity (inverse vol)
            vol = max(abs(m.volatility), 0.01)
            risk_parity_score = 1.0 / vol

            # Component 4: Drawdown penalty
            dd_penalty = max(0, 1.0 - abs(m.max_drawdown) * 2)

            # Combined score: weighted geometric mean
            scores[sid] = (
                regime_bonus * 0.35
                + sharpe_score * 0.30
                + risk_parity_score * 0.20
                + dd_penalty * 0.15
            )

        total = sum(scores.values())
        if total < 1e-8:
            return self._equal_weight(sleeves)
        return {sid: s / total for sid, s in scores.items()}

    def _kelly_criterion(self, sleeves: dict[str, SleeveMetrics], **_: Any) -> dict[str, float]:
        """Fractional Kelly criterion (half-Kelly for safety)."""
        kelly_weights = {}
        for sid, m in sleeves.items():
            if m.avg_loss == 0 or m.trade_count < 5:
                kelly_weights[sid] = 0.1
                continue
            # Kelly formula: f* = (p * b - q) / b
            # where p = win_rate, b = avg_win/avg_loss, q = 1 - p
            p = max(min(m.win_rate, 0.99), 0.01)
            q = 1.0 - p
            b = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else 1.0
            kelly_f = max((p * b - q) / b, 0.0)
            # Half-Kelly for safety
            kelly_weights[sid] = kelly_f * 0.5 + 0.05  # floor

        total = sum(kelly_weights.values())
        if total < 1e-8:
            return self._equal_weight(sleeves)
        return {sid: w / total for sid, w in kelly_weights.items()}

    def _apply_constraints(
        self, weights: dict[str, float], *, min_w: float, max_w: float
    ) -> dict[str, float]:
        """Clip weights and re-normalize."""
        clipped = {sid: max(min_w, min(max_w, w)) for sid, w in weights.items()}
        total = sum(clipped.values())
        if total > 0:
            clipped = {sid: w / total for sid, w in clipped.items()}
        return clipped

    def _compute_risk_contributions(
        self, weights: dict[str, float], sleeves: dict[str, SleeveMetrics]
    ) -> dict[str, float]:
        """Compute each strategy's contribution to portfolio risk."""
        total_risk = sum(
            w * sleeves.get(sid, SleeveMetrics(strategy_id=sid)).volatility
            for sid, w in weights.items()
        )
        if total_risk < 1e-8:
            return {sid: 1.0 / len(weights) for sid in weights}
        return {
            sid: (w * sleeves.get(sid, SleeveMetrics(strategy_id=sid)).volatility) / total_risk
            for sid, w in weights.items()
        }

    def _estimate_portfolio_sharpe(
        self, weights: dict[str, float], sleeves: dict[str, SleeveMetrics]
    ) -> float:
        """Estimate portfolio Sharpe (simplified, assumes low correlation)."""
        weighted_return = sum(
            w * sleeves.get(sid, SleeveMetrics(strategy_id=sid)).total_return
            for sid, w in weights.items()
        )
        weighted_vol = np.sqrt(
            sum(
                (w * sleeves.get(sid, SleeveMetrics(strategy_id=sid)).volatility) ** 2
                for sid, w in weights.items()
            )
        )
        if weighted_vol < 1e-8:
            return 0.0
        return float(weighted_return / weighted_vol)

    def _diversification_ratio(
        self, weights: dict[str, float], sleeves: dict[str, SleeveMetrics]
    ) -> float:
        """Compute diversification ratio (weighted avg vol / portfolio vol)."""
        weighted_avg_vol = sum(
            w * sleeves.get(sid, SleeveMetrics(strategy_id=sid)).volatility
            for sid, w in weights.items()
        )
        port_vol = np.sqrt(
            sum(
                (w * sleeves.get(sid, SleeveMetrics(strategy_id=sid)).volatility) ** 2
                for sid, w in weights.items()
            )
        )
        if port_vol < 1e-8:
            return 1.0
        return float(weighted_avg_vol / port_vol)

    def _generate_reasoning(
        self,
        method: str,
        weights: dict[str, float],
        sleeves: dict[str, SleeveMetrics],
        regime: str,
        exp_sharpe: float,
    ) -> list[str]:
        """Generate human-readable allocation reasoning."""
        reasons = [f"分配方法: {method}", f"当前Regime: {regime}"]

        # Top weighted strategies
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        top3 = sorted_w[:3]
        reasons.append(
            "最大权重: " + ", ".join(f"{sid}({w:.0%})" for sid, w in top3)
        )

        # Risk warnings
        for sid, m in sleeves.items():
            if abs(m.max_drawdown) > 0.20 and weights.get(sid, 0) > 0.15:
                reasons.append(f"⚠ {sid} 回撤较大({m.max_drawdown:.1%})但权重较高")

        if exp_sharpe < 0:
            reasons.append("⚠ 预期夏普为负，建议降低仓位或切换策略")

        return reasons
