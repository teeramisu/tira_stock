"""BaseRLAgent — abstract base class for RL-MACS information agents.

Each information agent:
1. Receives market data (prices, features, regime) for a given timestep
2. Produces a normalized opinion vector in [-1, 1]^K
3. Does NOT make trading decisions — only provides structured signals

The Decision Agent (RL policy) learns how to weight and combine these opinions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AgentOpinion:
    """Structured output from an information agent.

    Attributes:
        vector: Normalized opinion vector in [-1, 1]^K.
        confidence: Agent's self-assessed confidence (0-1).
        name: Agent identifier.
        metadata: Optional diagnostic info.
    """

    vector: np.ndarray
    confidence: float
    name: str
    metadata: dict[str, Any] | None = None

    @property
    def dim(self) -> int:
        return len(self.vector)


class BaseRLAgent(ABC):
    """Abstract base class for RL-MACS information agents.

    Subclasses must implement:
    - ``opinion_dim``: property returning the opinion vector dimension K
    - ``compute_opinion()`: compute opinion vector for a given timestep

    Usage:
        agent = FactorAgent(price_data, ...)
        opinion = agent.compute_opinion(step=100, regime="BULL")
        # opinion.vector is shape (K,) in [-1, 1]
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def opinion_dim(self) -> int:
        """Dimension of the opinion vector this agent produces."""
        ...

    @abstractmethod
    def compute_opinion(
        self,
        step: int,
        prices: np.ndarray,
        market_features: np.ndarray,
        regime: str = "RANGE",
        **kwargs: Any,
    ) -> AgentOpinion:
        """Compute opinion vector for the given timestep.

        Args:
            step: Current timestep index.
            prices: (T, N_assets) price matrix up to current step.
            market_features: (T, 10) market-level features.
            regime: Current regime label (BULL/RANGE/BEAR/CRASH).
            **kwargs: Additional context (e.g., ticker-specific data).

        Returns:
            AgentOpinion with normalized vector in [-1, 1]^K.
        """
        ...

    def compute_opinion_batch(
        self,
        steps: np.ndarray,
        prices: np.ndarray,
        market_features: np.ndarray,
        regime_labels: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute opinions for multiple timesteps (for pre-computation).

        Args:
            steps: Array of timestep indices.
            prices: Full (T, N_assets) price matrix.
            market_features: Full (T, 10) feature matrix.
            regime_labels: (T,) regime label array.

        Returns:
            (len(steps), opinion_dim) opinion matrix.
        """
        regime_names = ["BULL", "RANGE", "BEAR", "CRASH"]
        opinions = np.zeros((len(steps), self.opinion_dim), dtype=np.float32)

        for i, step in enumerate(steps):
            regime = regime_names[int(regime_labels[step])]
            opinion = self.compute_opinion(
                step=int(step),
                prices=prices[: step + 1],
                market_features=market_features[: step + 1],
                regime=regime,
                **kwargs,
            )
            opinions[i] = opinion.vector

        return opinions

    @staticmethod
    def _normalize_to_range(values: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
        """Normalize values to [low, high] range using tanh-like squashing.

        Handles NaN/inf gracefully.
        """
        values = np.nan_to_num(values, nan=0.0, posinf=3.0, neginf=-3.0)
        # Use tanh for smooth squashing to [-1, 1]
        normalized = np.tanh(values)
        # Scale to target range
        mid = (low + high) / 2
        half_range = (high - low) / 2
        return (mid + half_range * normalized).astype(np.float32)

    @staticmethod
    def _rank_normalize(values: np.ndarray) -> np.ndarray:
        """Rank-normalize values to [-1, 1] (cross-sectional).

        Converts raw values to percentile ranks, then maps to [-1, 1].
        """
        n = len(values)
        if n <= 1:
            return np.zeros_like(values, dtype=np.float32)
        # Rank (0 to n-1)
        order = values.argsort().argsort().astype(np.float32)
        # Map to [-1, 1]
        return (2.0 * order / (n - 1) - 1.0).astype(np.float32)

    @staticmethod
    def _safe_return(prices: np.ndarray, lookback: int, step: int) -> float:
        """Safely compute return over lookback period."""
        if step < lookback or step >= len(prices):
            return 0.0
        prev = prices[step - lookback]
        curr = prices[step]
        if prev <= 0:
            return 0.0
        return float(curr / prev - 1.0)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} dim={self.opinion_dim}>"
