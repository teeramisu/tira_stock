"""Portfolio constructor, risk and execution interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


class AlphaModel(Protocol):
    def score(self, factors: pd.DataFrame) -> pd.Series: ...


class RiskModel(Protocol):
    def risk_penalty(self, factors: pd.DataFrame) -> pd.Series: ...


class ExecutionModel(Protocol):
    def estimate_cost(self, target_weights: pd.Series, liquidity: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class ConstructorConstraints:
    max_positions: int = 10
    max_weight: float = 0.2
    max_gross_exposure: float = 1.0
    min_daily_amount: float = 0.0
    max_participation_rate: float = 0.2


@dataclass
class PortfolioConstructor:
    constraints: ConstructorConstraints = field(default_factory=ConstructorConstraints)

    def construct(
        self,
        scored: pd.DataFrame,
        *,
        capital: float,
        adv_col: str = "adv20",
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Construct constrained weights from scored symbols."""
        out = scored.sort_values("score", ascending=False).copy()
        diagnostics = {"input": len(out), "filtered_liquidity": 0, "filtered_weight_cap": 0}
        if adv_col in out.columns and self.constraints.min_daily_amount > 0:
            before = len(out)
            out = out[out[adv_col].fillna(0.0) >= self.constraints.min_daily_amount]
            diagnostics["filtered_liquidity"] = before - len(out)
        out = out.head(self.constraints.max_positions).copy()
        if out.empty:
            out["target_weight"] = []
            return out, diagnostics
        equal_weight = min(self.constraints.max_weight, self.constraints.max_gross_exposure / len(out))
        out["target_weight"] = equal_weight
        diagnostics["filtered_weight_cap"] = int((out["target_weight"] >= self.constraints.max_weight).sum())
        if adv_col in out.columns and capital > 0:
            out["participation_rate"] = (out["target_weight"] * capital) / out[adv_col].replace(0, pd.NA)
            out["participation_rate"] = out["participation_rate"].fillna(float("inf"))
            out["participation_rate"] = out["participation_rate"].clip(upper=self.constraints.max_participation_rate)
        return out, diagnostics


def capacity_report(scored: pd.DataFrame, *, capital: float, constructor: PortfolioConstructor) -> dict[str, float]:
    """Compute simple capacity diagnostics for one rebalance."""
    built, _ = constructor.construct(scored, capital=capital)
    if built.empty:
        return {
            "capital": capital,
            "unfilled_ratio": 1.0,
            "turnover": 0.0,
            "cost_ratio": 0.0,
            "capacity_sensitivity": 0.0,
        }
    unfilled = float((built.get("participation_rate", pd.Series(dtype=float)) >= constructor.constraints.max_participation_rate).mean())
    turnover = float(built["target_weight"].abs().sum())
    cost_ratio = float((built.get("participation_rate", pd.Series(0.0, index=built.index))).mean() * 0.001)
    sensitivity = cost_ratio * (capital / max(capital, 1.0))
    return {
        "capital": capital,
        "unfilled_ratio": unfilled,
        "turnover": turnover,
        "cost_ratio": cost_ratio,
        "capacity_sensitivity": sensitivity,
    }
