"""FactorAgent — produces multi-factor opinion vectors for RL-MACS.

Computes 8 factor dimensions from price/volume data:
0. Momentum (20-day return rank)
1. Value (mean-reversion distance from MA)
2. Quality (volatility-adjusted return stability)
3. Volatility (inverse realized vol rank)
4. Liquidity (volume trend)
5. Size (price level as proxy for market cap rank)
6. Reversal (5-day oversold bounce signal)
7. Factor Momentum (rolling IC of momentum factor)

All values normalized to [-1, 1] via cross-sectional rank or tanh squashing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_rl_agent import AgentOpinion, BaseRLAgent


class FactorAgent(BaseRLAgent):
    """Multi-factor signal agent producing 8-dimensional opinion vectors.

    This agent extracts classical quantitative factors from price data
    and outputs a cross-sectionally normalized factor exposure vector.

    The factors are designed to be complementary:
    - Momentum + Reversal capture different time-horizon effects
    - Value + Quality provide fundamental-like signals from price data
    - Volatility + Liquidity capture risk/tradability dimensions
    """

    def __init__(self, lookback_momentum: int = 20, lookback_short: int = 5):
        super().__init__(
            name="factor",
            description="Multi-factor signal agent (momentum, value, quality, vol, liquidity, size, reversal, factor_mom)",
        )
        self.lookback_momentum = lookback_momentum
        self.lookback_short = lookback_short

    @property
    def opinion_dim(self) -> int:
        return 8

    def compute_opinion(
        self,
        step: int,
        prices: np.ndarray,
        market_features: np.ndarray,
        regime: str = "RANGE",
        **kwargs: Any,
    ) -> AgentOpinion:
        """Compute 8-dim factor opinion vector.

        Args:
            step: Current timestep.
            prices: (T, N_assets) price matrix.
            market_features: (T, 10) market features.
            regime: Current regime.

        Returns:
            AgentOpinion with 8-dim vector.
        """
        n_assets = prices.shape[1] if prices.ndim > 1 else 1
        vector = np.zeros(self.opinion_dim, dtype=np.float32)

        if step < self.lookback_momentum or n_assets == 0:
            return AgentOpinion(
                vector=vector, confidence=0.0, name=self.name,
                metadata={"reason": "insufficient_data"},
            )

        # Ensure 2D
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        current_prices = prices[step]
        valid_mask = current_prices > 0

        if valid_mask.sum() == 0:
            return AgentOpinion(vector=vector, confidence=0.0, name=self.name)

        # --- Factor 0: Momentum (20-day return, cross-sectional rank) ---
        mom_returns = np.zeros(n_assets)
        for i in range(n_assets):
            if valid_mask[i] and prices[step - self.lookback_momentum, i] > 0:
                mom_returns[i] = current_prices[i] / prices[step - self.lookback_momentum, i] - 1.0
        vector[0] = float(np.mean(self._rank_normalize(mom_returns[valid_mask])))

        # --- Factor 1: Value (distance from 60-day MA, mean-reversion signal) ---
        ma_window = min(60, step)
        if ma_window >= 20:
            ma60 = prices[step - ma_window:step + 1].mean(axis=0)
            value_dist = np.zeros(n_assets)
            for i in range(n_assets):
                if valid_mask[i] and ma60[i] > 0:
                    # Negative distance = undervalued (buy signal)
                    value_dist[i] = -(current_prices[i] / ma60[i] - 1.0)
            vector[1] = float(np.mean(self._normalize_to_range(value_dist[valid_mask] * 3)))

        # --- Factor 2: Quality (return stability = Sharpe-like over 20 days) ---
        if step >= 20:
            daily_rets = prices[step - 19:step + 1] / prices[step - 20:step] - 1.0
            daily_rets = np.nan_to_num(daily_rets, nan=0.0)
            mean_ret = daily_rets.mean(axis=0)
            std_ret = daily_rets.std(axis=0) + 1e-8
            quality = mean_ret / std_ret  # Information ratio proxy
            vector[2] = float(np.mean(self._normalize_to_range(quality[valid_mask])))

        # --- Factor 3: Volatility (inverse vol rank — low vol preferred) ---
        if step >= 20:
            vol_20d = daily_rets.std(axis=0) * np.sqrt(252)
            # Inverse: low vol → positive signal
            inv_vol = -vol_20d
            vector[3] = float(np.mean(self._rank_normalize(inv_vol[valid_mask])))

        # --- Factor 4: Liquidity (volume trend proxy via price volatility change) ---
        if step >= 10:
            recent_vol = prices[step - 4:step + 1].std(axis=0)
            older_vol = prices[step - 9:step - 4].std(axis=0) + 1e-8
            vol_change = (recent_vol - older_vol) / older_vol
            # Increasing liquidity (vol) is mildly positive
            vector[4] = float(np.mean(self._normalize_to_range(vol_change[valid_mask])))

        # --- Factor 5: Size (price level as market cap proxy — small cap premium) ---
        if valid_mask.sum() > 1:
            log_prices = np.log(current_prices[valid_mask] + 1e-8)
            # Inverse rank: smaller price → positive (small-cap premium)
            size_signal = -self._rank_normalize(log_prices)
            vector[5] = float(np.mean(size_signal))

        # --- Factor 6: Reversal (5-day oversold bounce) ---
        if step >= self.lookback_short:
            short_ret = np.zeros(n_assets)
            for i in range(n_assets):
                if valid_mask[i] and prices[step - self.lookback_short, i] > 0:
                    short_ret[i] = current_prices[i] / prices[step - self.lookback_short, i] - 1.0
            # Negative recent return → reversal buy signal
            reversal_signal = -short_ret
            vector[6] = float(np.mean(self._normalize_to_range(reversal_signal[valid_mask] * 5)))

        # --- Factor 7: Factor Momentum (is momentum working recently?) ---
        if step >= 40:
            # Compare momentum factor performance: recent vs older
            recent_mom = np.zeros(n_assets)
            older_mom = np.zeros(n_assets)
            for i in range(n_assets):
                if valid_mask[i]:
                    if prices[step - 10, i] > 0 and prices[step - 20, i] > 0:
                        recent_mom[i] = prices[step, i] / prices[step - 10, i] - 1.0
                    if prices[step - 20, i] > 0 and prices[step - 30, i] > 0:
                        older_mom[i] = prices[step - 10, i] / prices[step - 20, i] - 1.0
            # If recent momentum > older momentum, momentum is working
            mom_accel = recent_mom - older_mom
            vector[7] = float(np.mean(self._normalize_to_range(mom_accel[valid_mask] * 5)))

        # Confidence based on data availability
        confidence = min(1.0, step / 60.0) * min(1.0, valid_mask.sum() / max(n_assets * 0.5, 1))

        return AgentOpinion(
            vector=vector,
            confidence=float(confidence),
            name=self.name,
            metadata={
                "n_valid_assets": int(valid_mask.sum()),
                "regime": regime,
                "step": step,
            },
        )
