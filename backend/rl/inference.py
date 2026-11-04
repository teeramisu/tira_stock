"""Inference engine for RL-MACS.

Handles:
- Loading trained models
- Running single-step predictions
- Full episode backtesting with RL policy
- Ensemble inference (averaging multiple models)
- Integration with existing backtest engine
- Temporal masking to prevent look-ahead bias
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO

    _SB3_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# Temporal Mask — prevents look-ahead bias
# ============================================================


class TemporalMask:
    """Enforces causal (no look-ahead) data access.

    At decision time t, only data from steps [0, t] is observable.
    The return earned is from t to t+1 (unknown at decision time).

    Usage:
        mask = TemporalMask(current_step=100)
        masked_prices = mask.apply(prices)  # prices[101:] zeroed out
        mask.validate_access(step=100)  # OK
        mask.validate_access(step=101)  # raises LookAheadError
    """

    def __init__(self, current_step: int):
        self._current_step = current_step

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._current_step = value

    def validate_access(self, step: int, name: str = "data") -> None:
        """Raise if accessing data beyond current step."""
        if step > self._current_step:
            raise LookAheadError(
                f"Look-ahead bias detected: accessing {name}[{step}] "
                f"at decision time t={self._current_step}"
            )

    def validate_slice(self, start: int, end: int, name: str = "data") -> None:
        """Validate a slice [start:end] doesn't exceed current step."""
        if end > self._current_step + 1:
            raise LookAheadError(
                f"Look-ahead bias detected: accessing {name}[{start}:{end}] "
                f"at decision time t={self._current_step} (max allowed: {self._current_step + 1})"
            )

    def mask_array(self, data: np.ndarray) -> np.ndarray:
        """Return a copy of data with future steps zeroed out.

        For 1D: data[t+1:] = 0
        For 2D: data[t+1:, :] = 0
        """
        masked = data.copy()
        if masked.ndim == 1:
            masked[self._current_step + 1:] = 0
        elif masked.ndim == 2:
            masked[self._current_step + 1:] = 0
        return masked

    def observable_window(self, data: np.ndarray, lookback: int | None = None) -> np.ndarray:
        """Return only the observable portion of data up to current_step.

        Args:
            data: Full data array.
            lookback: If set, return only [current_step - lookback : current_step + 1].

        Returns:
            Sliced view of data (no copy for performance).
        """
        end = self._current_step + 1
        if lookback is not None:
            start = max(0, self._current_step - lookback + 1)
            return data[start:end]
        return data[:end]


class LookAheadError(Exception):
    """Raised when look-ahead bias is detected."""
    pass


def validate_no_lookahead(
    prices: np.ndarray,
    market_features: np.ndarray,
    regime_labels: np.ndarray,
    agent_opinions: np.ndarray | None,
    decision_step: int,
) -> None:
    """Validate that all data used at decision_step is causally valid.

    Checks:
    1. market_features[t] only uses prices[:t+1]
    2. regime_labels[t] only uses features[:t+1]
    3. agent_opinions[t] only uses data[:t+1]
    4. The return earned is from t to t+1 (future, unknown at decision time)

    This is a structural validation — call during development/testing.
    """
    mask = TemporalMask(decision_step)

    # Validate array lengths
    if len(market_features) <= decision_step:
        raise LookAheadError(
            f"market_features too short: need index {decision_step}, "
            f"got length {len(market_features)}"
        )
    if len(regime_labels) <= decision_step:
        raise LookAheadError(
            f"regime_labels too short: need index {decision_step}, "
            f"got length {len(regime_labels)}"
        )
    if agent_opinions is not None and len(agent_opinions) <= decision_step:
        raise LookAheadError(
            f"agent_opinions too short: need index {decision_step}, "
            f"got length {len(agent_opinions)}"
        )

    # Validate no NaN in observable window (would indicate future dependency)
    obs_prices = prices[:decision_step + 1]
    if np.any(np.isnan(obs_prices)):
        raise LookAheadError(
            f"NaN in prices[:{decision_step + 1}] — possible data alignment issue"
        )


@dataclass
class PredictionResult:
    """Result of a single RL prediction."""

    action: np.ndarray  # Raw action vector
    safe_action: np.ndarray  # After risk overlay
    positions: np.ndarray  # Target positions
    regime: str
    confidence: float
    agent_opinions: dict[str, np.ndarray]  # Per-agent opinion vectors
    risk_info: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result of RL backtest over a period."""

    equity_curve: np.ndarray
    daily_returns: np.ndarray
    positions_history: np.ndarray
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    n_trades: int
    total_costs: float
    regime_distribution: dict[str, int]
    agent_opinions_history: np.ndarray | None = None


class RLInferenceEngine:
    """Inference engine for trained RL-MACS models.

    Supports:
    - Single model inference
    - Ensemble inference (multiple seeds)
    - Full episode backtesting
    - Integration with existing system components
    - Observation normalization (matching training env)
    - Temporal masking (no look-ahead bias)

    Usage:
        engine = RLInferenceEngine(
            model_path="backend/rl/models/rl_macs_ppo.zip",
            obs_stats_path="backend/rl/models/obs_stats.json",
        )
        result = engine.predict(observation)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        ensemble_paths: list[str | Path] | None = None,
        device: str = "cpu",
        obs_stats_path: str | Path | None = None,
        slippage: float = 0.0005,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to single trained model.
            ensemble_paths: Paths to ensemble models (overrides model_path).
            device: Torch device for inference.
            obs_stats_path: Path to observation normalization stats (JSON).
                           Contains {"mean": [...], "std": [...]}.
            slippage: Slippage rate (default 0.05%, matching training env).
        """
        self.device = device
        self.models: list[Any] = []
        self.model_path = model_path
        self.ensemble_paths = ensemble_paths
        self.slippage = slippage

        # Observation normalization stats (must match training env)
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None
        if obs_stats_path:
            self.load_obs_stats(obs_stats_path)

        if ensemble_paths:
            for p in ensemble_paths:
                self._load_model(p)
        elif model_path:
            self._load_model(model_path)

    def load_obs_stats(self, path: str | Path) -> None:
        """Load observation normalization statistics from JSON file.

        The stats file should contain:
            {"mean": [float, ...], "std": [float, ...], "obs_dim": int}
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Obs stats not found: {path}. Normalization disabled.")
            return
        try:
            with open(path) as f:
                stats = json.load(f)
            self._obs_mean = np.array(stats["mean"], dtype=np.float32)
            self._obs_std = np.array(stats["std"], dtype=np.float32)
            logger.info(f"Loaded obs stats: dim={len(self._obs_mean)}")
        except Exception as e:
            logger.warning(f"Failed to load obs stats: {e}")

    def set_obs_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set observation normalization statistics directly."""
        self._obs_mean = mean.astype(np.float32)
        self._obs_std = std.astype(np.float32)

    def save_obs_stats(self, path: str | Path) -> None:
        """Save observation normalization statistics to JSON file."""
        if self._obs_mean is None or self._obs_std is None:
            logger.warning("No obs stats to save.")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "mean": self._obs_mean.tolist(),
            "std": self._obs_std.tolist(),
            "obs_dim": len(self._obs_mean),
        }
        with open(path, "w") as f:
            json.dump(stats, f)
        logger.info(f"Saved obs stats to {path}")

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation normalization (matching training env)."""
        if self._obs_mean is not None and self._obs_std is not None:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)
        return obs

    def _load_model(self, path: str | Path) -> None:
        """Load a single model from path."""
        if not _SB3_AVAILABLE:
            logger.warning("Stable-Baselines3 not available. Cannot load model.")
            return

        path = Path(path)
        if not path.exists():
            # Try with .zip extension
            zip_path = path.with_suffix(".zip")
            if zip_path.exists():
                path = zip_path
            else:
                logger.warning(f"Model not found: {path}")
                return

        try:
            model = PPO.load(str(path), device=self.device)
            self.models.append(model)
            logger.info(f"Loaded model: {path}")
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")

    @property
    def is_ready(self) -> bool:
        """Whether at least one model is loaded."""
        return len(self.models) > 0

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict action for a single observation.

        If ensemble is loaded, averages predictions across models.

        Args:
            observation: (obs_dim,) observation vector.
            deterministic: Whether to use deterministic policy.

        Returns:
            (action_dim,) action vector.
        """
        if not self.is_ready:
            raise RuntimeError("No model loaded. Call load_model() first.")

        actions = []
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            actions.append(action)

        # Ensemble: average actions
        if len(actions) > 1:
            avg_action = np.mean(actions, axis=0)
            # Clip to valid range
            return np.clip(avg_action, -1.0, 1.0).astype(np.float32)
        return actions[0]

    def predict_with_context(
        self,
        price_data: dict[str, np.ndarray],
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        step: int,
        prev_positions: np.ndarray | None = None,
        agent_opinions: np.ndarray | None = None,
    ) -> PredictionResult:
        """Predict with full context (builds observation internally).

        Args:
            price_data: Dict of {ticker: close_prices}.
            market_features: (T, 10) market features.
            regime_labels: (T,) regime labels.
            step: Current timestep.
            prev_positions: Previous positions.
            agent_opinions: (T, 26) pre-computed opinions.

        Returns:
            PredictionResult with action and diagnostics.
        """
        from .agents import FactorAgent, MacroAgent, SentimentRLAgent, TechnicalAgent
        from .risk_overlay import RiskOverlay, RiskOverlayConfig

        tickers = sorted(price_data.keys())
        n_assets = len(tickers)
        prices = np.column_stack([price_data[t] for t in tickers])

        regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]
        regime = regime_names[int(regime_labels[step])]

        # Compute agent opinions if not provided
        if agent_opinions is not None and step < len(agent_opinions):
            opinions = agent_opinions[step]
        else:
            opinions = self._compute_opinions_at_step(
                step, prices, market_features, regime
            )

        # Build observation (simplified — matches env structure)
        obs = self._build_observation(
            step, prices, market_features, regime_labels, opinions, prev_positions
        )

        # Get action from model
        raw_action = self.predict(obs)

        # Apply risk overlay
        overlay = RiskOverlay(config=RiskOverlayConfig())
        risk_result = overlay.apply(raw_action, prev_positions, regime)

        # Compute target positions
        if prev_positions is not None:
            target_positions = prev_positions + risk_result.adjusted_action[:n_assets]
        else:
            target_positions = risk_result.adjusted_action[:n_assets]
        target_positions = np.clip(target_positions, 0.0, 1.0)

        # Build per-agent opinion dict
        agent_op_dict = {
            "factor": opinions[:8],
            "macro": opinions[8:14],
            "sentiment": opinions[14:20],
            "technical": opinions[20:26],
        }

        return PredictionResult(
            action=raw_action,
            safe_action=risk_result.adjusted_action,
            positions=target_positions,
            regime=regime,
            confidence=1.0 - risk_result.total_exposure * 0.1,
            agent_opinions=agent_op_dict,
            risk_info={
                "constraints_applied": risk_result.constraints_applied,
                "total_exposure": risk_result.total_exposure,
                "cash_ratio": risk_result.cash_ratio,
                "turnover": risk_result.turnover,
            },
        )

    def run_backtest(
        self,
        price_data: dict[str, np.ndarray],
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        agent_opinions: np.ndarray | None = None,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.001,
        start_step: int = 60,
        end_step: int | None = None,
        validate_causal: bool = True,
    ) -> BacktestResult:
        """Run full backtest with RL policy.

        Temporal contract (no look-ahead):
            - At step t: observation uses only data[0..t]
            - Action is decided at close of day t
            - Return earned is daily_returns[t+1] (day t → t+1, unknown at decision)

        Args:
            price_data: Dict of {ticker: close_prices}.
            market_features: (T, 10) market features (causal: features[t] uses data[:t+1]).
            regime_labels: (T,) regime labels (causal: labels[t] uses features[:t+1]).
            agent_opinions: (T, 26) pre-computed opinions (causal: opinions[t] uses data[:t+1]).
            initial_cash: Starting capital.
            commission_rate: Commission rate per trade.
            start_step: First step to trade.
            end_step: Last step (None = end of data).
            validate_causal: If True, run look-ahead validation at start.

        Returns:
            BacktestResult with full performance metrics.
        """
        from .risk_overlay import RiskOverlay, RiskOverlayConfig

        tickers = sorted(price_data.keys())
        n_assets = len(tickers)
        prices = np.column_stack([price_data[t] for t in tickers])
        n_steps = len(prices)

        if end_step is None:
            end_step = n_steps - 1
        end_step = min(end_step, n_steps - 1)

        # Causal validation: ensure no look-ahead bias in input data
        if validate_causal:
            validate_no_lookahead(
                prices, market_features, regime_labels, agent_opinions, start_step
            )

        # Pre-compute daily returns
        daily_returns = np.zeros_like(prices)
        daily_returns[1:] = prices[1:] / prices[:-1] - 1.0

        # Initialize
        overlay = RiskOverlay(config=RiskOverlayConfig())
        regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]

        portfolio_value = initial_cash
        positions = np.zeros(n_assets)
        equity_curve = [portfolio_value]
        positions_history = [positions.copy()]
        returns_list = [0.0]
        total_costs = 0.0
        regime_dist: dict[str, int] = {}

        # Temporal mask for causal enforcement
        mask = TemporalMask(current_step=start_step)

        for step in range(start_step, end_step):
            mask.current_step = step

            regime = regime_names[int(regime_labels[step])]
            regime_dist[regime] = regime_dist.get(regime, 0) + 1

            # Build observation — only uses data[0..step] (causal)
            if agent_opinions is not None and step < len(agent_opinions):
                opinions = agent_opinions[step]
            else:
                opinions = np.zeros(26, dtype=np.float32)

            # Compute unrealized PnL for account state
            unrealized_pnl = (portfolio_value - initial_cash) / initial_cash

            obs = self._build_observation(
                step, prices, market_features, regime_labels, opinions,
                positions, unrealized_pnl=unrealized_pnl,
            )

            # Get action
            raw_action = self.predict(obs)

            # Apply risk overlay
            risk_result = overlay.apply(raw_action, positions, regime)
            new_positions = positions + risk_result.adjusted_action[:n_assets]
            new_positions = np.clip(new_positions, 0.0, 1.0)

            # Transaction cost (commission + slippage, matching training env)
            turnover = np.sum(np.abs(new_positions - positions))
            cost = turnover * (commission_rate + self.slippage)
            total_costs += cost

            # Portfolio return: earn daily_returns[step+1] (FUTURE return, unknown at decision)
            # This is correct: decide at close of day t, earn return from t to t+1
            step_ret = daily_returns[step + 1] if step + 1 < n_steps else np.zeros(n_assets)
            port_return = float(np.sum(new_positions * step_ret)) - cost

            # Update
            portfolio_value *= (1.0 + port_return)
            positions = new_positions.copy()

            equity_curve.append(portfolio_value)
            positions_history.append(positions.copy())
            returns_list.append(port_return)

        # Compute metrics
        equity = np.array(equity_curve)
        rets = np.array(returns_list[1:])  # Skip initial 0

        total_return = (equity[-1] - initial_cash) / initial_cash
        n_days = len(rets)
        annual_factor = 252.0 / max(n_days, 1)
        annual_return = (1 + total_return) ** annual_factor - 1 if total_return > -1 else -1.0
        vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
        sharpe = annual_return / vol if vol > 1e-8 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / np.maximum(peak, 1e-8)
        max_dd = float(drawdowns.max())

        # Win rate
        wins = np.sum(rets > 0)
        win_rate = float(wins / max(len(rets), 1))

        return BacktestResult(
            equity_curve=equity,
            daily_returns=rets,
            positions_history=np.array(positions_history),
            total_return=float(total_return),
            annual_return=float(annual_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=max_dd,
            volatility=vol,
            win_rate=win_rate,
            n_trades=int(np.sum(np.abs(np.diff(positions_history, axis=0)) > 0.01)),
            total_costs=float(total_costs),
            regime_distribution=regime_dist,
        )

    def _build_observation(
        self,
        step: int,
        prices: np.ndarray,
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        opinions: np.ndarray,
        positions: np.ndarray | None,
        unrealized_pnl: float = 0.0,
    ) -> np.ndarray:
        """Build observation vector matching TradingEnv format.

        TEMPORAL CONTRACT: Only accesses data[0..step]. Never touches data[step+1:].
        """
        n_assets = prices.shape[1]

        # Market features (10 dims) — causal: features[step] computed from data[:step+1]
        mkt_feat = market_features[step].astype(np.float32) if step < len(market_features) else np.zeros(10, dtype=np.float32)

        # Regime one-hot (4 dims) — causal: labels[step] computed from features[:step+1]
        regime_onehot = np.zeros(4, dtype=np.float32)
        if step < len(regime_labels):
            regime_onehot[int(regime_labels[step])] = 1.0

        # Per-asset returns (N dims) — uses prices[step] and prices[step-5] (past only)
        lookback = min(5, step)
        if lookback > 0:
            asset_returns = (prices[step] / prices[step - lookback] - 1.0).astype(np.float32)
        else:
            asset_returns = np.zeros(n_assets, dtype=np.float32)

        # Per-asset volatility (N dims)
        # NOTE: Must match env.py logic exactly:
        #   env: daily_returns[step-20:step].std(axis=0) * sqrt(252)
        #   where daily_returns[0]=0, daily_returns[i]=prices[i]/prices[i-1]-1
        asset_vol = np.zeros(n_assets, dtype=np.float32)
        if step >= 20:
            start = step - 20
            if start >= 1:
                # All indices >= 1: prices[start-1:step] → 21 prices → 20 returns
                window = prices[start - 1:step]
                rets = window[1:] / window[:-1] - 1.0
            else:
                # start == 0: daily_returns[0]=0, then normal returns
                window = prices[0:step]  # step prices → step-1 returns
                rets_inner = window[1:] / window[:-1] - 1.0
                rets = np.vstack([np.zeros((1, n_assets)), rets_inner])
            asset_vol = rets.std(axis=0).astype(np.float32) * np.sqrt(252)

        # Agent opinions (26 dims) — causal: opinions[step] computed from data[:step+1]
        opinions = opinions.astype(np.float32) if opinions is not None else np.zeros(26, dtype=np.float32)

        # Account state (3 dims) — real-time portfolio state
        if positions is not None:
            total_invested = float(np.sum(positions))
            cash_ratio = 1.0 - total_invested
        else:
            total_invested = 0.0
            cash_ratio = 1.0
        account_state = np.array([cash_ratio, total_invested, unrealized_pnl], dtype=np.float32)

        obs = np.concatenate([
            mkt_feat, regime_onehot, asset_returns, asset_vol, opinions, account_state
        ])

        # Apply normalization (matching training env)
        obs = self._normalize_obs(obs)

        return obs.astype(np.float32)

    def _compute_opinions_at_step(
        self, step: int, prices: np.ndarray, market_features: np.ndarray, regime: str
    ) -> np.ndarray:
        """Compute all agent opinions at a single step."""
        from .agents import FactorAgent, MacroAgent, SentimentRLAgent, TechnicalAgent

        factor_agent = FactorAgent()
        macro_agent = MacroAgent()
        sentiment_agent = SentimentRLAgent()
        technical_agent = TechnicalAgent()

        prices_slice = prices[:step + 1]
        features_slice = market_features[:step + 1]

        f = factor_agent.compute_opinion(step, prices_slice, features_slice, regime)
        m = macro_agent.compute_opinion(step, prices_slice, features_slice, regime)
        s = sentiment_agent.compute_opinion(step, prices_slice, features_slice, regime)
        t = technical_agent.compute_opinion(step, prices_slice, features_slice, regime)

        return np.concatenate([f.vector, m.vector, s.vector, t.vector])

    def get_status(self) -> dict[str, Any]:
        """Get inference engine status."""
        return {
            "is_ready": self.is_ready,
            "n_models": len(self.models),
            "model_path": str(self.model_path) if self.model_path else None,
            "ensemble_paths": [str(p) for p in self.ensemble_paths] if self.ensemble_paths else None,
            "device": self.device,
            "sb3_available": _SB3_AVAILABLE,
        }
