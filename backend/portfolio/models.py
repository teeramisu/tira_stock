"""Typed models for portfolio backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd


OrderSide = Literal["buy", "sell"]
FillStatus = Literal["filled", "rejected"]


@dataclass(frozen=True)
class TargetPosition:
    """Target portfolio weight emitted by a portfolio strategy."""

    symbol: str
    weight: float
    score: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class StrategyContext:
    """Runtime context visible to portfolio strategies."""

    market: str
    universe: tuple[str, ...]
    dataset_version: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    """Explicit order generated from target weights."""

    symbol: str
    side: OrderSide
    shares: int
    reason: str = ""
    target_weight: float | None = None


@dataclass(frozen=True)
class CostBreakdown:
    """Per-fill cost details."""

    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    slippage_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Return total explicit trading costs (slippage already in fill_price)."""
        return self.commission + self.stamp_tax + self.transfer_fee


@dataclass(frozen=True)
class Fill:
    """Order execution result."""

    trade_date: date
    symbol: str
    side: OrderSide
    requested_shares: int
    filled_shares: int
    raw_price: float
    fill_price: float
    status: FillStatus
    reason: str = ""
    costs: CostBreakdown = field(default_factory=CostBreakdown)

    @property
    def gross_amount(self) -> float:
        """Return traded cash amount before explicit fees."""
        return self.filled_shares * self.fill_price


@dataclass
class Position:
    """Open position."""

    symbol: str
    shares: int = 0


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """Portfolio backtest configuration."""

    market: str
    dataset_version: str
    start: date
    end: date
    initial_cash: float = 1_000_000.0
    rebalance_frequency: str = "daily"
    lot_size: int = 100


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Portfolio backtest artifacts."""

    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    rebalance_logs: pd.DataFrame
    filter_diagnostics: pd.DataFrame
    benchmark_curve: pd.DataFrame
    cost_summary: dict[str, float]
    metrics: dict[str, float]

