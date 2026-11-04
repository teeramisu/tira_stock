"""Broker, order validation, and trading cost simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from typing import Protocol

import pandas as pd

from .models import CostBreakdown, Fill, Order, Position, TargetPosition


@dataclass(frozen=True)
class CostModel:
    """Trading cost model with explicit configurable parameters."""

    buy_commission_rate: float = 0.0003
    sell_commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.0
    min_commission: float = 5.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    # 差异化滑点：按市值分 3 档
    slippage_large_bps: float = 2.0   # 市值 > 500亿
    slippage_medium_bps: float = 5.0  # 50亿 ~ 500亿
    slippage_small_bps: float = 10.0  # < 50亿

    def _effective_slippage_bps(self, symbol_meta: dict | None = None) -> float:
        """根据 symbol_meta 中的市值返回差异化滑点，无数据时回退到默认值。"""
        if symbol_meta is None:
            return self.slippage_bps + self.market_impact_bps
        cap = symbol_meta.get("market_cap")
        if cap is None:
            return self.slippage_bps + self.market_impact_bps
        if cap > 500e8:
            return self.slippage_large_bps
        if cap > 50e8:
            return self.slippage_medium_bps
        return self.slippage_small_bps

    def fill_price(self, side: str, raw_price: float, symbol_meta: dict | None = None) -> float:
        """Return side-adjusted fill price after slippage and impact."""
        bps = self._effective_slippage_bps(symbol_meta) / 10_000
        if side == "buy":
            return raw_price * (1 + bps)
        return raw_price * (1 - bps)

    def calculate(self, side: str, shares: int, raw_price: float, symbol_meta: dict | None = None) -> tuple[float, CostBreakdown]:
        """Calculate fill price and per-order costs."""
        fill_price = self.fill_price(side, raw_price, symbol_meta=symbol_meta)
        amount = shares * fill_price
        commission_rate = self.buy_commission_rate if side == "buy" else self.sell_commission_rate
        commission = max(amount * commission_rate, self.min_commission) if shares > 0 else 0.0
        stamp_tax = amount * self.stamp_tax_rate if side == "sell" else 0.0
        transfer_fee = amount * self.transfer_fee_rate
        slippage_cost = abs(fill_price - raw_price) * shares
        return fill_price, CostBreakdown(
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            slippage_cost=slippage_cost,
        )


class OrderValidator:
    """Validate market-rule constraints before execution."""

    def validate(self, order: Order, bar: pd.Series | None) -> str | None:
        """Return rejection reason or None when order can trade."""
        if order.shares <= 0:
            return "non_positive_shares"
        if bar is None:
            return "missing_price"
        raw_open = bar.get("Open")
        raw_close = bar.get("Close")
        has_valid_open = pd.notna(raw_open) and math.isfinite(float(raw_open)) and float(raw_open) > 0
        has_valid_close = pd.notna(raw_close) and math.isfinite(float(raw_close)) and float(raw_close) > 0
        if not (has_valid_open or has_valid_close):
            return "missing_price"
        if bool(bar.get("paused", False)):
            return "suspended"
        close = bar.get("Close")
        if order.side == "buy" and pd.notna(bar.get("limit_up")) and close >= bar.get("limit_up"):
            return "limit_up"
        if order.side == "sell" and pd.notna(bar.get("limit_down")) and close <= bar.get("limit_down"):
            return "limit_down"
        return None

    @staticmethod
    def validate_targets(targets: list[TargetPosition]) -> None:
        """Validate strategy target weights."""
        total_weight = sum(target.weight for target in targets)
        if total_weight > 1.0 + 1e-9:
            raise ValueError("target weights must not exceed 1.0")
        for target in targets:
            if target.weight < 0:
                raise ValueError("target weight must be non-negative")
            if not target.symbol:
                raise ValueError("target symbol is required")


class Broker(Protocol):
    """Common broker protocol for backtest/paper/live modes."""

    cash: float
    positions: dict[str, Position]

    def market_value(self, bars_by_symbol: dict[str, pd.Series]) -> float:
        """Return total equity using supplied bars."""

    def rebalance(
        self,
        *,
        trade_date: date,
        targets: list[TargetPosition],
        bars_by_symbol: dict[str, pd.Series],
    ) -> list[Fill]:
        """Rebalance current holdings to target weights."""


@dataclass
class BrokerSimulator:
    """Cash and position accounting for portfolio backtests."""

    cash: float
    cost_model: CostModel = field(default_factory=CostModel)
    validator: OrderValidator = field(default_factory=OrderValidator)
    lot_size: int = 100
    positions: dict[str, Position] = field(default_factory=dict)
    _last_known_prices: dict[str, float] = field(default_factory=dict, repr=False)

    def market_value(self, bars_by_symbol: dict[str, pd.Series]) -> float:
        """Return total equity using supplied bars, falling back to last known prices."""
        value = self.cash
        for symbol, position in self.positions.items():
            bar = bars_by_symbol.get(symbol)
            price = None
            if bar is not None:
                price = self._extract_price(bar)
            if price is None:
                price = self._last_known_prices.get(symbol)
            if price is not None and price > 0:
                self._last_known_prices[symbol] = price
                value += position.shares * price
        return value

    def rebalance(
        self,
        *,
        trade_date: date,
        targets: list[TargetPosition],
        bars_by_symbol: dict[str, pd.Series],
    ) -> list[Fill]:
        """Rebalance current holdings to target weights."""
        self.validator.validate_targets(targets)
        portfolio_value = self.market_value(bars_by_symbol)
        target_weights = {target.symbol: target.weight for target in targets}
        target_reasons = {target.symbol: target.reason for target in targets}
        symbols = set(self.positions) | set(target_weights)
        orders: list[Order] = []
        for symbol in symbols:
            bar = bars_by_symbol.get(symbol)
            price = self._extract_price(bar)
            if price is None:
                price = self._last_known_prices.get(symbol, 0.0)
            if price is not None and price > 0:
                self._last_known_prices[symbol] = price
            current_shares = self.positions.get(symbol, Position(symbol)).shares
            current_value = current_shares * price
            target_value = portfolio_value * target_weights.get(symbol, 0.0)
            share_delta = int((target_value - current_value) // price) if price > 0 else 0
            if share_delta > 0:
                buy_shares = (share_delta // self.lot_size) * self.lot_size
                if buy_shares > 0:
                    orders.append(
                        Order(symbol=symbol, side="buy", shares=buy_shares, reason=target_reasons.get(symbol, ""))
                    )
            elif share_delta < 0:
                sell_shares = self._sell_order_shares(
                    requested=min(abs(share_delta), current_shares),
                    current_shares=current_shares,
                    full_exit=target_weights.get(symbol, 0.0) <= 0,
                )
                if sell_shares > 0:
                    orders.append(
                        Order(symbol=symbol, side="sell", shares=sell_shares, reason=target_reasons.get(symbol, ""))
                    )

        sell_orders = [order for order in orders if order.side == "sell"]
        buy_orders = [order for order in orders if order.side == "buy"]
        fills: list[Fill] = []
        for order in [*sell_orders, *buy_orders]:
            fills.append(self.execute_order(trade_date, order, bars_by_symbol.get(order.symbol)))
        return fills

    def execute_order(self, trade_date: date, order: Order, bar: pd.Series | None) -> Fill:
        """Execute a single order or return a rejected fill."""
        reason = self.validator.validate(order, bar)
        raw_price = self._extract_price(bar) or 0.0
        if reason is not None:
            return Fill(
                trade_date=trade_date,
                symbol=order.symbol,
                side=order.side,
                requested_shares=order.shares,
                filled_shares=0,
                raw_price=raw_price,
                fill_price=raw_price,
                status="rejected",
                reason=reason,
            )

        fill_price, costs = self.cost_model.calculate(order.side, order.shares, raw_price)
        gross = order.shares * fill_price
        if order.side == "buy":
            cash_required = gross + costs.commission + costs.stamp_tax + costs.transfer_fee
            if cash_required > self.cash:
                affordable = self._affordable_buy_shares(raw_price, order.shares)
                if affordable <= 0:
                    return Fill(
                        trade_date=trade_date,
                        symbol=order.symbol,
                        side=order.side,
                        requested_shares=order.shares,
                        filled_shares=0,
                        raw_price=raw_price,
                        fill_price=fill_price,
                        status="rejected",
                        reason="insufficient_cash",
                    )
                order = Order(order.symbol, order.side, affordable, order.reason, order.target_weight)
                fill_price, costs = self.cost_model.calculate(order.side, order.shares, raw_price)
                gross = order.shares * fill_price
                cash_required = gross + costs.commission + costs.stamp_tax + costs.transfer_fee
            self.cash -= cash_required
            position = self.positions.setdefault(order.symbol, Position(order.symbol))
            position.shares += order.shares
        else:
            position = self.positions.get(order.symbol, Position(order.symbol))
            sell_shares = self._normalize_sell_fill(order.shares, position.shares)
            if sell_shares <= 0:
                return Fill(
                    trade_date=trade_date,
                    symbol=order.symbol,
                    side=order.side,
                    requested_shares=order.shares,
                    filled_shares=0,
                    raw_price=raw_price,
                    fill_price=fill_price,
                    status="rejected",
                    reason="invalid_lot_size",
                )
            fill_price, costs = self.cost_model.calculate(order.side, sell_shares, raw_price)
            gross = sell_shares * fill_price
            self.cash += gross - costs.commission - costs.stamp_tax - costs.transfer_fee
            position.shares -= sell_shares
            self.positions[order.symbol] = position
            order = Order(order.symbol, order.side, sell_shares, order.reason, order.target_weight)

        return Fill(
            trade_date=trade_date,
            symbol=order.symbol,
            side=order.side,
            requested_shares=order.shares,
            filled_shares=order.shares,
            raw_price=raw_price,
            fill_price=fill_price,
            status="filled",
            costs=costs,
            reason=order.reason,
        )

    @staticmethod
    def _extract_price(bar: pd.Series | None) -> float | None:
        """Extract tradable price from a bar, returning None when unavailable."""
        if bar is None:
            return None
        raw = bar.get("Open")
        if pd.isna(raw):
            raw = bar.get("Close")
        if pd.isna(raw):
            return None
        price = float(raw)
        if not math.isfinite(price) or price <= 0:
            return None
        return price

    def _sell_order_shares(self, *, requested: int, current_shares: int, full_exit: bool) -> int:
        """A-share partial sells must be whole lots; full exit may clear odd lots."""
        requested = min(requested, current_shares)
        if requested <= 0:
            return 0
        if self.lot_size <= 1:
            return requested
        if full_exit:
            return current_shares
        return (requested // self.lot_size) * self.lot_size

    def _normalize_sell_fill(self, requested: int, current_shares: int) -> int:
        """Clamp sell fill to lot rules at execution time."""
        requested = min(requested, current_shares)
        if requested <= 0:
            return 0
        if self.lot_size <= 1:
            return requested
        if requested == current_shares:
            return requested
        return (requested // self.lot_size) * self.lot_size

    def _affordable_buy_shares(self, raw_price: float, requested_shares: int) -> int:
        shares = (requested_shares // self.lot_size) * self.lot_size
        while shares > 0:
            fill_price, costs = self.cost_model.calculate("buy", shares, raw_price)
            cash_required = shares * fill_price + costs.commission + costs.transfer_fee
            if cash_required <= self.cash:
                return shares
            shares -= self.lot_size
        return 0


@dataclass
class BacktestBroker(BrokerSimulator):
    """Backtest-mode broker implementation."""


@dataclass
class PaperBroker(BrokerSimulator):
    """Paper-trading broker placeholder sharing same accounting contract."""


@dataclass
class LiveBroker(BrokerSimulator):
    """Live-trading broker placeholder sharing same accounting contract."""

