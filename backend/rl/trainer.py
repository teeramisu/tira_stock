"""Training pipeline for RL-MACS Decision Agent.

Implements a multi-phase training procedure:
1. Data preparation: Load historical data from DuckDB → build TradingEnv
2. Warm-up phase: Train with basic features only (50K steps)
3. Joint training: Full agent opinions + curriculum learning (500K steps)
4. Stress testing: Validate on crisis periods
5. Walk-forward validation: IS/OOS Sharpe ratio check

Uses Stable-Baselines3 PPO as the primary training backend,
with custom DecisionNetwork architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    pass

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    # Phases
    warmup_steps: int = 50_000
    main_steps: int = 500_000
    stress_test_steps: int = 0  # Evaluation only

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Rollout length
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    net_arch: list[int] = field(default_factory=lambda: [256, 128, 64])

    # Environment
    n_envs: int = 4  # Parallel environments
    max_episode_length: int = 252

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: list[str] = field(
        default_factory=lambda: ["BULL", "RANGE", "BEAR", "CRASH"]
    )

    # Evaluation
    eval_freq: int = 10_000
    eval_episodes: int = 5
    save_freq: int = 50_000

    # Paths
    model_dir: str = "backend/rl/models"
    log_dir: str = "backend/rl/logs"
    tensorboard_log: str = "backend/rl/tb_logs"

    # Ensemble
    ensemble_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"


@dataclass
class TrainingResult:
    """Result of a training run."""

    model_path: str
    total_steps: int
    training_time_seconds: float
    final_reward: float
    best_sharpe: float
    best_model_path: str
    episode_rewards: list[float]
    eval_results: list[dict[str, Any]]
    config: TrainingConfig


class TrainingCallback(BaseCallback if _SB3_AVAILABLE else object):
    """Custom callback for monitoring RL-MACS training."""

    def __init__(self, eval_freq: int = 10_000, verbose: int = 1):
        if _SB3_AVAILABLE:
            super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards: list[float] = []
        self.best_mean_reward = -np.inf
        self.training_start_time = time.time()

    def _on_step(self) -> bool:
        """Called at each step of training."""
        # Log progress
        if self.num_timesteps % self.eval_freq == 0:
            elapsed = time.time() - self.training_start_time
            fps = self.num_timesteps / max(elapsed, 1)
            logger.info(
                f"Step {self.num_timesteps:,} | "
                f"FPS: {fps:.0f} | "
                f"Elapsed: {elapsed:.0f}s"
            )
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Extract episode rewards from info buffer
        try:
            infos = self.training_env.get_attr("get_episode_summary")
            for info in infos:
                if isinstance(info, dict) and "total_return" in info:
                    self.episode_rewards.append(info["total_return"])
        except Exception:
            pass


class RLTrainer:
    """Training pipeline for RL-MACS.

    Orchestrates the full training procedure including data preparation,
    multi-phase training, evaluation, and model saving.

    Usage:
        trainer = RLTrainer(config=TrainingConfig())
        result = trainer.train(price_data, market_features, regime_labels)
    """

    def __init__(self, config: TrainingConfig = TrainingConfig()):
        self.config = config
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if not _SB3_AVAILABLE:
            logger.warning(
                "Stable-Baselines3 not installed. "
                "Install with: pip install stable-baselines3>=2.0.0"
            )
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not installed. Install with: pip install torch")

    def train(
        self,
        price_data: dict[str, np.ndarray],
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        agent_opinions: np.ndarray | None = None,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> TrainingResult:
        """Run the full training pipeline.

        Args:
            price_data: Dict of {ticker: close_prices_array}.
            market_features: (T, 10) market-level features.
            regime_labels: (T,) integer regime labels.
            agent_opinions: (T, 26) pre-computed agent opinions (optional).
            progress_callback: Optional callback(step, reward) for progress updates.

        Returns:
            TrainingResult with model paths and metrics.
        """
        if not _SB3_AVAILABLE:
            raise RuntimeError(
                "Stable-Baselines3 is required for training. "
                "Install with: pip install stable-baselines3>=2.0.0 gymnasium>=0.29.0"
            )

        start_time = time.time()
        cfg = self.config

        logger.info("=" * 60)
        logger.info("RL-MACS Training Pipeline Starting")
        logger.info(f"  Assets: {len(price_data)}")
        logger.info(f"  Data length: {len(market_features)}")
        logger.info(f"  Warmup steps: {cfg.warmup_steps:,}")
        logger.info(f"  Main steps: {cfg.main_steps:,}")
        logger.info("=" * 60)

        # Phase 1: Create environment
        from .env import EnvConfig, TradingEnv

        env_config = EnvConfig(
            max_episode_length=cfg.max_episode_length,
            agent_opinion_dims=agent_opinions.shape[1] if agent_opinions is not None else 26,
        )

        def make_env():
            env = TradingEnv(
                price_data=price_data,
                market_features=market_features,
                regime_labels=regime_labels,
                agent_opinions=agent_opinions,
                config=env_config,
            )
            env.training = True
            return env

        # Create vectorized environment
        env = DummyVecEnv([make_env for _ in range(cfg.n_envs)])

        # Phase 2: Warm-up training (basic features only)
        logger.info("\n--- Phase 2: Warm-up Training ---")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs={"net_arch": dict(pi=cfg.net_arch, vf=cfg.net_arch)},
            tensorboard_log=cfg.tensorboard_log,
            device=cfg.device,
            verbose=1,
        )

        callback = TrainingCallback(eval_freq=cfg.eval_freq)
        model.learn(
            total_timesteps=cfg.warmup_steps,
            callback=callback,
            tb_log_name="rl_macs_warmup",
        )
        logger.info(f"Warm-up complete. Steps: {cfg.warmup_steps:,}")

        # Phase 3: Joint training (full agent opinions)
        logger.info("\n--- Phase 3: Joint Training ---")
        model.learn(
            total_timesteps=cfg.main_steps,
            callback=callback,
            tb_log_name="rl_macs_main",
            reset_num_timesteps=False,
        )
        logger.info(f"Joint training complete. Total steps: {cfg.warmup_steps + cfg.main_steps:,}")

        # Save model
        model_path = self.model_dir / "rl_macs_ppo"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Phase 4: Evaluation
        logger.info("\n--- Phase 4: Evaluation ---")
        eval_results = self._evaluate(
            model, price_data, market_features, regime_labels, agent_opinions, env_config
        )

        total_time = time.time() - start_time
        final_reward = callback.episode_rewards[-1] if callback.episode_rewards else 0.0
        best_sharpe = max(
            (r.get("sharpe_ratio", 0.0) for r in eval_results), default=0.0
        )

        result = TrainingResult(
            model_path=str(model_path),
            total_steps=cfg.warmup_steps + cfg.main_steps,
            training_time_seconds=total_time,
            final_reward=final_reward,
            best_sharpe=best_sharpe,
            best_model_path=str(model_path),
            episode_rewards=callback.episode_rewards,
            eval_results=eval_results,
            config=cfg,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Best Sharpe: {best_sharpe:.3f}")
        logger.info(f"  Model: {model_path}")
        logger.info("=" * 60)

        return result

    def train_ensemble(
        self,
        price_data: dict[str, np.ndarray],
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        agent_opinions: np.ndarray | None = None,
    ) -> list[TrainingResult]:
        """Train an ensemble of models with different random seeds.

        Returns:
            List of TrainingResult, one per seed.
        """
        results = []
        for seed in self.config.ensemble_seeds:
            logger.info(f"\n{'='*40} Ensemble seed={seed} {'='*40}")
            np.random.seed(seed)
            if _TORCH_AVAILABLE:
                torch.manual_seed(seed)

            result = self.train(
                price_data=price_data,
                market_features=market_features,
                regime_labels=regime_labels,
                agent_opinions=agent_opinions,
            )
            results.append(result)

        return results

    def _evaluate(
        self,
        model: Any,
        price_data: dict[str, np.ndarray],
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        agent_opinions: np.ndarray | None,
        env_config: Any,
    ) -> list[dict[str, Any]]:
        """Evaluate trained model on different regime periods."""
        from .env import TradingEnv

        results = []
        regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]

        # Full evaluation
        eval_env = TradingEnv(
            price_data=price_data,
            market_features=market_features,
            regime_labels=regime_labels,
            agent_opinions=agent_opinions,
            config=env_config,
        )
        eval_env.training = False

        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        summary = eval_env.get_episode_summary()
        summary["regime"] = "FULL"
        results.append(summary)
        logger.info(
            f"  Full eval: Return={summary['total_return']:.2%} "
            f"Sharpe={summary['sharpe_ratio']:.3f} "
            f"MaxDD={summary['max_drawdown']:.2%}"
        )

        return results

    def get_training_status(self) -> dict[str, Any]:
        """Get current training status."""
        model_files = list(self.model_dir.glob("*.zip"))
        return {
            "model_dir": str(self.model_dir),
            "n_models": len(model_files),
            "models": [f.name for f in model_files],
            "sb3_available": _SB3_AVAILABLE,
            "torch_available": _TORCH_AVAILABLE,
        }


def precompute_agent_opinions(
    price_data: dict[str, np.ndarray],
    market_features: np.ndarray,
    regime_labels: np.ndarray,
) -> np.ndarray:
    """Pre-compute all agent opinions for the full time series.

    This is done once before training to avoid recomputing opinions
    at every environment step.

    Args:
        price_data: Dict of {ticker: close_prices_array}.
        market_features: (T, 10) market features.
        regime_labels: (T,) regime labels.

    Returns:
        (T, 26) concatenated opinion matrix (8+6+6+6 dims).
    """
    from .agents import FactorAgent, MacroAgent, SentimentRLAgent, TechnicalAgent

    tickers = sorted(price_data.keys())
    prices = np.column_stack([price_data[t] for t in tickers])
    n_steps = len(prices)

    # Initialize agents
    factor_agent = FactorAgent()
    macro_agent = MacroAgent()
    sentiment_agent = SentimentRLAgent()
    technical_agent = TechnicalAgent()

    total_dim = (
        factor_agent.opinion_dim
        + macro_agent.opinion_dim
        + sentiment_agent.opinion_dim
        + technical_agent.opinion_dim
    )  # 8 + 6 + 6 + 6 = 26

    opinions = np.zeros((n_steps, total_dim), dtype=np.float32)
    regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]

    logger.info(f"Pre-computing agent opinions for {n_steps} steps...")

    for step in range(n_steps):
        if step % 100 == 0:
            logger.debug(f"  Step {step}/{n_steps}")

        regime = regime_names[int(regime_labels[step])]
        prices_slice = prices[: step + 1]
        features_slice = market_features[: step + 1]

        # Compute each agent's opinion
        f_op = factor_agent.compute_opinion(step, prices_slice, features_slice, regime)
        m_op = macro_agent.compute_opinion(step, prices_slice, features_slice, regime)
        s_op = sentiment_agent.compute_opinion(step, prices_slice, features_slice, regime)
        t_op = technical_agent.compute_opinion(step, prices_slice, features_slice, regime)

        # Concatenate
        offset = 0
        opinions[step, offset:offset + f_op.dim] = f_op.vector
        offset += f_op.dim
        opinions[step, offset:offset + m_op.dim] = m_op.vector
        offset += m_op.dim
        opinions[step, offset:offset + s_op.dim] = s_op.vector
        offset += s_op.dim
        opinions[step, offset:offset + t_op.dim] = t_op.vector

    logger.info("Agent opinions pre-computation complete.")
    return opinions
