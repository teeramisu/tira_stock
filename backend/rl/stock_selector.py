"""Stock Selector — Cross-sectional factor scoring for full-market mode.

Selects top-K stocks from the full universe (1000+) based on multi-factor scoring.
Rebalances periodically (e.g., weekly/monthly) to reduce turnover.

Two modes:
- Small mode: No selection needed, use all assets directly (≤20)
- Full mode: Score → Rank → Select top-K → Feed to RL portfolio constructor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SelectorConfig:
    """Stock selection configuration."""

    # Selection parameters
    top_k: int = 30  # Number of stocks to select
    rebalance_freq: int = 5  # Rebalance every N days (5 = weekly)

    # Factor weights for scoring
    momentum_weight: float = 0.3  # 20-day momentum
    volatility_weight: float = 0.2  # Inverse volatility (prefer lower vol)
    volume_weight: float = 0.2  # Liquidity (average volume rank)
    mean_reversion_weight: float = 0.15  # Distance from MA (contrarian)
    trend_weight: float = 0.15  # Trend strength (ADX-like)

    # Filters
    min_volume_rank: float = 0.1  # Exclude bottom 10% by volume
    max_correlation: float = 0.85  # Exclude highly correlated pairs
    min_price: float = 5.0  # Exclude penny stocks

    # Sector diversification
    max_per_sector: int = 5  # Max stocks from same sector
    use_sector_diversification: bool = True


class StockSelector:
    """Cross-sectional stock selection using multi-factor scoring.

    At each rebalance point, scores all stocks in the universe and
    selects the top-K based on composite factor score.

    Factors:
    1. Momentum: 20-day return rank (higher = better)
    2. Volatility: Inverse 20-day vol rank (lower vol = better)
    3. Liquidity: Volume rank (higher = better)
    4. Mean Reversion: Distance below 60-day MA (more oversold = better)
    5. Trend: Price above MA20 > MA60 alignment (stronger = better)
    """

    def __init__(self, config: SelectorConfig = SelectorConfig()):
        self.config = config
        self._last_selection_step = -999
        self._current_selection: np.ndarray | None = None  # Indices of selected stocks
        self._current_scores: np.ndarray | None = None

    def should_rebalance(self, step: int) -> bool:
        """Check if we should rebalance at this step."""
        return (step - self._last_selection_step) >= self.config.rebalance_freq

    def select(
        self,
        prices: np.ndarray,
        step: int,
        volumes: np.ndarray | None = None,
        sector_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select top-K stocks at given step.

        Args:
            prices: (T, N) full price history up to current step.
            step: Current time step (only use data[:step+1]).
            volumes: (T, N) volume data (optional).
            sector_labels: (N,) sector ID per stock (optional).

        Returns:
            Array of selected stock indices (length = top_k).
        """
        n_assets = prices.shape[1]

        # If universe is small enough, select all
        if n_assets <= self.config.top_k:
            self._current_selection = np.arange(n_assets)
            self._last_selection_step = step
            return self._current_selection

        # Check if we can reuse previous selection
        if not self.should_rebalance(step) and self._current_selection is not None:
            return self._current_selection

        # Compute factor scores (TEMPORAL: only use data up to step)
        scores = self._compute_scores(prices, step, volumes)

        # Apply filters
        valid_mask = self._apply_filters(prices, step, volumes, scores)

        # Set invalid stocks to -inf score
        scores[~valid_mask] = -np.inf

        # Sector diversification
        if self.config.use_sector_diversification and sector_labels is not None:
            selected = self._diversified_selection(scores, sector_labels)
        else:
            # Simple top-K by score
            selected = np.argsort(scores)[-self.config.top_k:]

        self._current_selection = selected
        self._current_scores = scores
        self._last_selection_step = step

        return selected

    def _compute_scores(
        self,
        prices: np.ndarray,
        step: int,
        volumes: np.ndarray | None,
    ) -> np.ndarray:
        """Compute composite factor scores for all stocks at step."""
        n_assets = prices.shape[1]
        scores = np.zeros(n_assets, dtype=np.float32)

        # Need at least 60 days of history
        if step < 60:
            return np.ones(n_assets, dtype=np.float32)  # Uniform if insufficient data

        # Factor 1: Momentum (20-day return)
        if step >= 20:
            ret_20 = prices[step] / prices[step - 20] - 1.0
            momentum_rank = self._rank_normalize(ret_20)
            scores += self.config.momentum_weight * momentum_rank

        # Factor 2: Inverse Volatility (lower vol = higher score)
        if step >= 20:
            returns = prices[step - 20:step + 1]
            daily_rets = returns[1:] / returns[:-1] - 1.0
            vol = np.nanstd(daily_rets, axis=0)
            vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.median(vol[vol > 0]) if np.any(vol > 0) else 0.02)
            inv_vol_rank = self._rank_normalize(-vol)  # Negative: lower vol = higher rank
            scores += self.config.volatility_weight * inv_vol_rank

        # Factor 3: Liquidity (volume rank)
        if volumes is not None and step >= 20:
            avg_vol = np.nanmean(volumes[step - 20:step + 1], axis=0)
            vol_rank = self._rank_normalize(avg_vol)
            scores += self.config.volume_weight * vol_rank
        else:
            # Without volume data, use price * abs(return) as proxy
            if step >= 5:
                activity = np.nanmean(np.abs(prices[step - 5:step + 1] / prices[step - 6:step] - 1.0), axis=0)
                activity_rank = self._rank_normalize(activity)
                scores += self.config.volume_weight * activity_rank

        # Factor 4: Mean Reversion (distance below 60-day MA)
        if step >= 60:
            ma60 = np.nanmean(prices[step - 60:step + 1], axis=0)
            distance = (prices[step] - ma60) / (ma60 + 1e-8)
            # Contrarian: stocks below MA get higher score
            mr_rank = self._rank_normalize(-distance)
            scores += self.config.mean_reversion_weight * mr_rank

        # Factor 5: Trend Strength (MA20 > MA60 alignment)
        if step >= 60:
            ma20 = np.nanmean(prices[step - 20:step + 1], axis=0)
            ma60 = np.nanmean(prices[step - 60:step + 1], axis=0)
            trend_signal = (ma20 - ma60) / (ma60 + 1e-8)
            trend_rank = self._rank_normalize(trend_signal)
            scores += self.config.trend_weight * trend_rank

        return scores

    def _apply_filters(
        self,
        prices: np.ndarray,
        step: int,
        volumes: np.ndarray | None,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Apply hard filters to exclude invalid stocks."""
        n_assets = prices.shape[1]
        valid = np.ones(n_assets, dtype=bool)

        # Filter: minimum price
        valid &= prices[step] >= self.config.min_price

        # Filter: must have valid price
        valid &= np.isfinite(prices[step]) & (prices[step] > 0)

        # Filter: volume rank (if available)
        if volumes is not None and step >= 20:
            avg_vol = np.nanmean(volumes[step - 20:step + 1], axis=0)
            vol_percentile = self._rank_normalize(avg_vol)
            valid &= vol_percentile >= self.config.min_volume_rank

        # Filter: sufficient history (no NaN in last 20 days)
        if step >= 20:
            recent = prices[step - 20:step + 1]
            has_nan = np.any(~np.isfinite(recent), axis=0)
            valid &= ~has_nan

        return valid

    def _diversified_selection(
        self,
        scores: np.ndarray,
        sector_labels: np.ndarray,
    ) -> np.ndarray:
        """Select top-K with sector diversification constraint."""
        sorted_indices = np.argsort(scores)[::-1]  # Highest score first
        selected = []
        sector_counts: dict[int, int] = {}

        for idx in sorted_indices:
            if len(selected) >= self.config.top_k:
                break
            sector = int(sector_labels[idx])
            count = sector_counts.get(sector, 0)
            if count < self.config.max_per_sector:
                selected.append(idx)
                sector_counts[sector] = count + 1

        # If not enough selected (strict sector limit), relax and fill
        if len(selected) < self.config.top_k:
            for idx in sorted_indices:
                if len(selected) >= self.config.top_k:
                    break
                if idx not in selected:
                    selected.append(idx)

        return np.array(selected, dtype=int)

    @staticmethod
    def _rank_normalize(values: np.ndarray) -> np.ndarray:
        """Convert values to percentile ranks [0, 1]."""
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.zeros_like(values)

        ranks = np.zeros_like(values)
        valid_values = values[valid]
        # Rank within valid values
        order = np.argsort(valid_values)
        ranked = np.zeros_like(valid_values)
        ranked[order] = np.linspace(0, 1, len(valid_values))
        ranks[valid] = ranked

        return ranks

    def get_selection_info(self) -> dict:
        """Return info about current selection for logging."""
        if self._current_selection is None:
            return {"selected": [], "scores": []}
        return {
            "selected": self._current_selection.tolist(),
            "n_selected": len(self._current_selection),
            "last_rebalance": self._last_selection_step,
        }
