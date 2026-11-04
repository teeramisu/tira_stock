"""Reward function design for RL-MACS trading environment.

Multi-component reward that balances:
- Risk-adjusted return (primary signal)
- Transaction cost penalty (discourage over-trading)
- Drawdown penalty (risk management)
- Concentration penalty (diversification incentive)

Reference: MARS (AAAI 2026) reward design with risk-aware components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Reward function hyperparameters."""

    # Component weights
    alpha: float = 1.0  # risk-adjusted return weight
    beta: float = 0.1  # transaction cost penalty weight
    gamma_dd: float = 2.0  # drawdown penalty weight
    delta: float = 0.5  # concentration penalty weight

    # Phase 4: Anti-overfitting penalties
    turnover_penalty: float = 0.2  # penalize daily turnover > 20%
    single_stock_penalty: float = 0.3  # penalize single position > 15%
    turnover_threshold: float = 0.20  # 20% daily turnover threshold
    single_stock_limit: float = 0.15  # 15% single position limit

    # Risk parameters
    risk_free_rate: float = 0.0  # daily risk-free rate (annualized / 252)
    dd_threshold: float = 0.10  # drawdown threshold before penalty kicks in
    vol_window: int = 30  # rolling volatility window for risk adjustment

    # Reward scaling (for stable RL training)
    reward_scale: float = 10.0  # multiply final reward
    reward_clip: float = 5.0  # clip reward to [-clip, +clip]

    # Sharpe-style risk adjustment
    use_sharpe_reward: bool = True  # divide return by rolling vol


def compute_reward(
    portfolio_return: float,
    transaction_cost: float,
    current_drawdown: float,
    positions: np.ndarray,
    rolling_vol: float = 0.0,
    config: RewardConfig = RewardConfig(),
    prev_positions: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute multi-component trading reward.

    Args:
        portfolio_return: (V_{t+1} - V_t) / V_t, single-step portfolio return.
        transaction_cost: Total transaction cost as fraction of portfolio value.
        current_drawdown: Current drawdown from peak (positive number, e.g. 0.05 = 5% dd).
        positions: Current position weights array (sums to ~1.0 with cash).
        rolling_vol: Rolling volatility estimate (annualized). If 0, skip vol adjustment.
        config: Reward configuration.
        prev_positions: Previous positions for turnover calculation.

    Returns:
        Dict with 'total' reward and individual components for logging.
    """
    # Component 1: Risk-adjusted return
    if config.use_sharpe_reward and rolling_vol > 1e-8:
        daily_vol = rolling_vol / np.sqrt(252)
        risk_adjusted_return = (portfolio_return - config.risk_free_rate) / max(daily_vol, 1e-8)
    else:
        risk_adjusted_return = portfolio_return - config.risk_free_rate

    return_component = config.alpha * risk_adjusted_return

    # Component 2: Transaction cost penalty
    cost_component = -config.beta * transaction_cost

    # Component 3: Drawdown penalty (quadratic beyond threshold)
    excess_dd = max(0.0, current_drawdown - config.dd_threshold)
    dd_component = -config.gamma_dd * (excess_dd ** 2)

    # Component 4: Concentration penalty (HHI-based)
    concentration = _compute_hhi(positions)
    n_assets = max(len(positions), 1)
    min_hhi = 1.0 / n_assets  # perfectly diversified
    concentration_excess = max(0.0, concentration - min_hhi)
    conc_component = -config.delta * concentration_excess

    # Component 5: Turnover penalty (Phase 4)
    turnover_component = 0.0
    if prev_positions is not None and len(prev_positions) == len(positions):
        turnover = np.sum(np.abs(positions - prev_positions)) / 2.0
        excess_turnover = max(0.0, turnover - config.turnover_threshold)
        turnover_component = -config.turnover_penalty * excess_turnover

    # Component 6: Single-stock concentration penalty (Phase 4)
    single_stock_component = 0.0
    if len(positions) > 0:
        max_position = np.max(np.abs(positions))
        excess_position = max(0.0, max_position - config.single_stock_limit)
        single_stock_component = -config.single_stock_penalty * excess_position

    # Total reward
    total = (return_component + cost_component + dd_component + conc_component
             + turnover_component + single_stock_component)

    # Scale and clip for stable training
    total = total * config.reward_scale
    total = float(np.clip(total, -config.reward_clip, config.reward_clip))

    return {
        "total": total,
        "return_component": float(return_component),
        "cost_component": float(cost_component),
        "drawdown_component": float(dd_component),
        "concentration_component": float(conc_component),
        "turnover_component": float(turnover_component),
        "single_stock_component": float(single_stock_component),
        "raw_return": float(portfolio_return),
        "transaction_cost": float(transaction_cost),
        "current_drawdown": float(current_drawdown),
    }


def _compute_hhi(weights: np.ndarray) -> float:
    """Compute Herfindahl-Hirschman Index for position concentration.

    HHI = sum(w_i^2), ranges from 1/N (diversified) to 1.0 (concentrated).
    Only considers non-cash positions.
    """
    if len(weights) == 0:
        return 0.0
    # Filter out near-zero positions
    active = weights[np.abs(weights) > 1e-6]
    if len(active) == 0:
        return 0.0
    # Normalize to sum to 1
    total = np.sum(np.abs(active))
    if total < 1e-8:
        return 0.0
    normalized = np.abs(active) / total
    return float(np.sum(normalized ** 2))


def compute_episode_reward(
    returns: list[float],
    costs: list[float],
    drawdowns: list[float],
    config: RewardConfig = RewardConfig(),
) -> float:
    """Compute cumulative discounted episode reward (for evaluation).

    Args:
        returns: List of per-step portfolio returns.
        costs: List of per-step transaction costs.
        drawdowns: List of per-step drawdown values.

    Returns:
        Total discounted reward for the episode.
    """
    total = 0.0
    discount = 1.0
    gamma = 0.99

    for r, c, dd in zip(returns, costs, drawdowns):
        reward_dict = compute_reward(
            portfolio_return=r,
            transaction_cost=c,
            current_drawdown=dd,
            positions=np.array([]),  # simplified for episode eval
            config=config,
        )
        total += discount * reward_dict["total"]
        discount *= gamma

    return total
