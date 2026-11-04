"""TradingEnv — Gymnasium-compatible trading environment for RL-MACS.

State space combines:
- Market-level features (10 dims from ml/features.py)
- Regime one-hot encoding (4 dims)
- Per-asset returns/volatility (N_assets dims)
- Agent opinion vectors (4 agents × K dims each)
- Account state (3 dims: cash ratio, total position, unrealized PnL)

Action space:
- Continuous Box(-1, 1) of shape (N_assets + 1,)
- First N_assets: target position change per asset
- Last dim: cash ratio adjustment

Compatible with Stable-Baselines3 PPO/SAC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import RewardConfig, compute_reward
from .risk_overlay import RiskOverlay, RiskOverlayConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Trading environment configuration."""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    n_market_features: int = 10  # From ml/features.py
    n_regime_classes: int = 4  # BULL/RANGE/BEAR/CRASH
    agent_opinion_dims: int = 26  # Total opinion vector dim (8+6+6+6)
    account_state_dims: int = 3  # cash_ratio, total_position, unrealized_pnl
    max_episode_length: int = 252  # One trading year
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    risk_config: RiskOverlayConfig = field(default_factory=RiskOverlayConfig)
    normalize_obs: bool = True  # Normalize observations


class TradingEnv(gym.Env):
    """Gymnasium trading environment for multi-agent RL.

    This environment simulates portfolio trading with:
    - Multiple assets with historical price data
    - Agent opinion vectors as part of observation
    - Risk overlay constraints on actions
    - Multi-component reward function

    Usage:
        env = TradingEnv(
            price_data=prices_dict,
            market_features=features_array,
            regime_labels=regime_array,
            agent_opinions=opinions_array,
            config=EnvConfig(),
        )
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: dict[str, np.ndarray],
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        agent_opinions: np.ndarray | None = None,
        config: EnvConfig = EnvConfig(),
    ):
        """Initialize trading environment.

        Args:
            price_data: Dict of {ticker: close_prices_array}. All arrays same length.
            market_features: (T, 10) market-level feature array.
            regime_labels: (T,) integer regime labels (0=BULL, 1=RANGE, 2=BEAR, 3=CRASH).
            agent_opinions: (T, agent_opinion_dims) pre-computed agent opinions.
                           If None, zeros are used (for warm-up training).
            config: Environment configuration.
        """
        super().__init__()
        self.config = config
        self.tickers = sorted(price_data.keys())
        self.n_assets = len(self.tickers)

        # Price matrix: (T, N_assets)
        self.prices = np.column_stack([price_data[t] for t in self.tickers])
        self.n_steps_total = len(self.prices)

        # Market features: (T, 10)
        self.market_features = market_features
        assert len(self.market_features) >= self.n_steps_total

        # Regime labels: (T,)
        self.regime_labels = regime_labels
        assert len(self.regime_labels) >= self.n_steps_total

        # Agent opinions: (T, opinion_dim)
        if agent_opinions is not None:
            self.agent_opinions = agent_opinions
        else:
            self.agent_opinions = np.zeros(
                (self.n_steps_total, config.agent_opinion_dims), dtype=np.float32
            )

        # Pre-compute daily returns for all assets
        self.daily_returns = np.zeros_like(self.prices)
        self.daily_returns[1:] = self.prices[1:] / self.prices[:-1] - 1.0

        # Pre-compute rolling volatility (20-day)
        self.rolling_vol = self._compute_rolling_vol(window=20)

        # Define spaces
        obs_dim = (
            config.n_market_features  # 10 market features
            + config.n_regime_classes  # 4 regime one-hot
            + self.n_assets  # per-asset returns
            + self.n_assets  # per-asset volatility
            + config.agent_opinion_dims  # agent opinions
            + config.account_state_dims  # account state
        )
        self.obs_dim = obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # Risk overlay
        self.risk_overlay = RiskOverlay(config=config.risk_config)

        # Episode state
        self._current_step = 0
        self._episode_start = 0
        self._cash = config.initial_cash
        self._positions = np.zeros(self.n_assets)  # position weights
        self._portfolio_value = config.initial_cash
        self._peak_value = config.initial_cash
        self._prev_positions = np.zeros(self.n_assets)
        self._episode_returns: list[float] = []
        self._episode_costs: list[float] = []

        # Running statistics for normalization
        self._obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self._obs_std = np.ones(obs_dim, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to start of episode.

        Optionally randomize start point for training diversity.
        """
        super().reset(seed=seed)

        # Random start point (leave room for full episode)
        max_start = max(0, self.n_steps_total - self.config.max_episode_length - 60)
        if max_start > 0 and self.training:
            self._episode_start = self.np_random.integers(60, max_start)
        else:
            self._episode_start = 60  # Need lookback for features

        self._current_step = self._episode_start
        self._cash = self.config.initial_cash
        self._positions = np.zeros(self.n_assets)
        self._portfolio_value = self.config.initial_cash
        self._peak_value = self.config.initial_cash
        self._prev_positions = np.zeros(self.n_assets)
        self._episode_returns = []
        self._episode_costs = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one trading step.

        Args:
            action: (N_assets + 1,) action vector from RL agent.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # Apply risk overlay
        regime_idx = int(self.regime_labels[self._current_step])
        regime_name = ["BULL", "RANGE", "BEAR", "CRASH"][regime_idx]

        risk_result = self.risk_overlay.apply(
            action=action,
            prev_positions=self._prev_positions.copy(),
            regime=regime_name,
        )
        safe_action = risk_result.adjusted_action

        # Execute trades
        asset_action = safe_action[: self.n_assets]
        new_positions = self._prev_positions + asset_action
        new_positions = np.clip(new_positions, 0.0, 1.0)  # long-only, no leverage

        # Calculate transaction costs
        turnover = np.sum(np.abs(new_positions - self._prev_positions))
        transaction_cost = turnover * (self.config.commission_rate + self.config.slippage)

        # Advance to next step
        self._current_step += 1
        if self._current_step >= self.n_steps_total:
            # Episode ended (ran out of data)
            obs = self._get_observation_at(self._current_step - 1)
            return obs, 0.0, True, False, self._get_info()

        # Calculate portfolio return based on asset returns
        step_returns = self.daily_returns[self._current_step]
        portfolio_return = float(np.sum(new_positions * step_returns))

        # Update portfolio value
        prev_value = self._portfolio_value
        self._portfolio_value *= (1.0 + portfolio_return - transaction_cost)
        self._cash = self._portfolio_value * (1.0 - np.sum(new_positions))

        # Update peak and drawdown
        self._peak_value = max(self._peak_value, self._portfolio_value)
        current_drawdown = (
            (self._peak_value - self._portfolio_value) / self._peak_value
            if self._peak_value > 0
            else 0.0
        )

        # Compute reward
        vol = float(self.rolling_vol[self._current_step]) if self._current_step < len(self.rolling_vol) else 0.0
        reward_dict = compute_reward(
            portfolio_return=portfolio_return,
            transaction_cost=transaction_cost,
            current_drawdown=current_drawdown,
            positions=new_positions,
            rolling_vol=vol,
            config=self.config.reward_config,
        )
        reward = reward_dict["total"]

        # Update state
        self._prev_positions = new_positions.copy()
        self._positions = new_positions.copy()
        self._episode_returns.append(portfolio_return)
        self._episode_costs.append(transaction_cost)

        # Check termination
        episode_length = self._current_step - self._episode_start
        terminated = episode_length >= self.config.max_episode_length
        truncated = False

        # Early termination: portfolio lost >50%
        if self._portfolio_value < self.config.initial_cash * 0.5:
            terminated = True

        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_dict
        info["constraints_applied"] = risk_result.constraints_applied

        return obs, reward, terminated, truncated, info

    # ---- Properties ----

    @property
    def training(self) -> bool:
        """Whether env is in training mode (random start)."""
        return getattr(self, "_training", True)

    @training.setter
    def training(self, value: bool) -> None:
        self._training = value

    # ---- Internal methods ----

    def _get_observation(self) -> np.ndarray:
        """Get observation at current step."""
        return self._get_observation_at(self._current_step)

    def _get_observation_at(self, step: int) -> np.ndarray:
        """Construct observation vector at given step.

        TEMPORAL CONTRACT: Only accesses data[0..step]. Never touches data[step+1:].
        This ensures no look-ahead bias in the observation.
        """
        step = min(step, self.n_steps_total - 1)
        cfg = self.config

        # 1. Market features (10 dims)
        mkt_feat = self.market_features[step].astype(np.float32)

        # 2. Regime one-hot (4 dims)
        regime_idx = int(self.regime_labels[step])
        regime_onehot = np.zeros(cfg.n_regime_classes, dtype=np.float32)
        regime_onehot[regime_idx] = 1.0

        # 3. Per-asset recent returns (N dims) - last 5 day return
        lookback = min(5, step)
        if lookback > 0 and step - lookback >= 0:
            asset_returns = (
                self.prices[step] / self.prices[step - lookback] - 1.0
            ).astype(np.float32)
        else:
            asset_returns = np.zeros(self.n_assets, dtype=np.float32)

        # 4. Per-asset volatility (N dims)
        asset_vol = np.zeros(self.n_assets, dtype=np.float32)
        if step >= 20:
            recent_returns = self.daily_returns[step - 20 : step]
            asset_vol = recent_returns.std(axis=0).astype(np.float32) * np.sqrt(252)

        # 5. Agent opinions (opinion_dim dims)
        opinions = self.agent_opinions[step].astype(np.float32)

        # 6. Account state (3 dims)
        total_invested = float(np.sum(self._positions))
        cash_ratio = 1.0 - total_invested
        unrealized_pnl = (
            (self._portfolio_value - self.config.initial_cash) / self.config.initial_cash
        )
        account_state = np.array(
            [cash_ratio, total_invested, unrealized_pnl], dtype=np.float32
        )

        # Concatenate all
        obs = np.concatenate([
            mkt_feat,
            regime_onehot,
            asset_returns,
            asset_vol,
            opinions,
            account_state,
        ])

        # Normalize if configured
        if cfg.normalize_obs:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)

        return obs.astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Get diagnostic info."""
        return {
            "step": self._current_step,
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "positions": self._positions.copy(),
            "peak_value": self._peak_value,
            "drawdown": (self._peak_value - self._portfolio_value) / max(self._peak_value, 1e-8),
            "total_return": (self._portfolio_value - self.config.initial_cash) / self.config.initial_cash,
            "n_trades": len(self._episode_returns),
            "tickers": self.tickers,
        }

    def _compute_rolling_vol(self, window: int = 20) -> np.ndarray:
        """Compute portfolio-level rolling volatility."""
        # Equal-weight portfolio returns
        if self.n_assets > 0:
            port_returns = self.daily_returns.mean(axis=1)
        else:
            port_returns = np.zeros(self.n_steps_total)

        vol = np.zeros(self.n_steps_total)
        for i in range(window, self.n_steps_total):
            vol[i] = port_returns[i - window : i].std() * np.sqrt(252)
        return vol

    def update_observation_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Update running observation statistics for normalization."""
        self._obs_mean = mean.astype(np.float32)
        self._obs_std = std.astype(np.float32)

    def get_episode_summary(self) -> dict[str, Any]:
        """Get summary statistics for the completed episode."""
        if not self._episode_returns:
            return {"total_return": 0.0, "n_steps": 0}

        returns = np.array(self._episode_returns)
        total_return = (self._portfolio_value - self.config.initial_cash) / self.config.initial_cash
        annual_factor = 252.0 / max(len(returns), 1)
        annual_return = (1 + total_return) ** annual_factor - 1 if total_return > -1 else -1.0
        vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
        sharpe = annual_return / vol if vol > 1e-8 else 0.0

        # Max drawdown
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (peak - cum_returns) / peak
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": vol,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "n_steps": len(returns),
            "total_costs": float(sum(self._episode_costs)),
            "final_value": self._portfolio_value,
        }
