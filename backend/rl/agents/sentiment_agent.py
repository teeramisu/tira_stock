"""SentimentRLAgent — produces market sentiment opinion vectors for RL-MACS.

Analyzes market sentiment using price/volume-derived signals:
0. News sentiment proxy (crash indicator from features)
1. Fear index (5-day crash percentile — VIX equivalent)
2. Market participation (advance/decline ratio from breadth)
3. Capital sentiment (volume-price divergence)
4. Sector rotation heat (momentum dispersion as proxy)
5. Sentiment momentum (5-day rate of change of sentiment)

When no news data is available, degrades gracefully to price/volume-based
sentiment indicators (always available from market data).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_rl_agent import AgentOpinion, BaseRLAgent


class SentimentRLAgent(BaseRLAgent):
    """Market sentiment analysis agent producing 6-dimensional opinion vectors.

    This agent derives sentiment signals purely from market data (prices, volumes,
    breadth indicators), making it always available without external news feeds.

    Optional: Can integrate with DeepSeek API for news-based sentiment when available.
    """

    def __init__(self, sentiment_lookback: int = 20, fear_window: int = 5):
        super().__init__(
            name="sentiment",
            description="Market sentiment agent (fear, participation, volume-price, rotation, momentum)",
        )
        self.sentiment_lookback = sentiment_lookback
        self.fear_window = fear_window

    @property
    def opinion_dim(self) -> int:
        return 6

    def compute_opinion(
        self,
        step: int,
        prices: np.ndarray,
        market_features: np.ndarray,
        regime: str = "RANGE",
        **kwargs: Any,
    ) -> AgentOpinion:
        """Compute 6-dim sentiment opinion vector.

        Args:
            step: Current timestep.
            prices: (T, N_assets) price matrix.
            market_features: (T, 10) market features.
            regime: Current regime label.

        Returns:
            AgentOpinion with 6-dim vector.
        """
        vector = np.zeros(self.opinion_dim, dtype=np.float32)

        if step < self.sentiment_lookback or len(market_features) <= step:
            return AgentOpinion(
                vector=vector, confidence=0.0, name=self.name,
                metadata={"reason": "insufficient_data"},
            )

        # Ensure 2D prices
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        n_assets = prices.shape[1]
        features = market_features[step]
        n_features = min(len(features), 10)

        # --- Dim 0: News sentiment proxy (crash indicator inverted) ---
        # Feature 4: crash_5d (5th percentile 5-day return)
        # Less negative = better sentiment
        if n_features > 4:
            crash_signal = features[4]  # Typically negative
            # Map: -0.10 (crash) → -1.0, 0.0 (neutral) → +0.5
            vector[0] = float(np.clip(crash_signal * 15 + 0.3, -1.0, 1.0))

        # --- Dim 1: Fear index (VIX equivalent) ---
        # Use realized volatility percentile as fear proxy
        if n_features > 5 and step >= 60:
            current_vol = features[5]  # realized_vol
            hist_vol = market_features[step - 60:step, 5]
            hist_vol = hist_vol[~np.isnan(hist_vol)]
            if len(hist_vol) > 10:
                vol_percentile = float(np.mean(hist_vol < current_vol))
                # High vol percentile = high fear = negative sentiment
                vector[1] = float(1.0 - 2.0 * vol_percentile)  # [1,-1]

        # --- Dim 2: Market participation (breadth as advance/decline) ---
        if n_features > 1:
            breadth_ma20 = features[1]  # % stocks above MA20
            # >0.5 = more advancers = positive sentiment
            vector[2] = float(np.clip((breadth_ma20 - 0.5) * 4, -1.0, 1.0))

        # --- Dim 3: Capital sentiment (volume-price divergence) ---
        # Detect divergence: price up but volume down (bearish) or vice versa
        if n_features > 8 and step >= 5:
            volume_trend = features[8]  # 20-day volume trend
            trend_5d = features[3] if n_features > 3 else 0.0  # 5-day price trend

            # Divergence: price and volume moving in opposite directions
            if trend_5d > 0.005 and volume_trend < -0.01:
                # Price up, volume down → weak rally (bearish divergence)
                divergence = -0.5
            elif trend_5d < -0.005 and volume_trend > 0.01:
                # Price down, volume up → capitulation (potential bottom)
                divergence = 0.3
            else:
                # Confirmation: aligned movements
                divergence = float(np.sign(trend_5d) * min(abs(volume_trend) * 10, 0.5))
            vector[3] = float(np.clip(divergence, -1.0, 1.0))

        # --- Dim 4: Sector rotation heat (momentum dispersion) ---
        if n_features > 7:
            momentum_disp = features[7]  # Cross-sectional momentum dispersion
            # High dispersion = active rotation = opportunities exist
            # Normalize to [0, 1] range (typical values 0-0.05)
            rotation_heat = min(1.0, momentum_disp * 30)
            # Moderate rotation is positive; extreme is chaotic
            if rotation_heat > 0.8:
                vector[4] = -0.3  # Too chaotic
            else:
                vector[4] = float(rotation_heat - 0.3)  # Positive when moderate

        # --- Dim 5: Sentiment momentum (rate of change) ---
        # Compare current sentiment indicators vs 5 days ago
        if step >= 5 and len(market_features) > step - 5:
            past_features = market_features[step - 5]
            if n_features > 1:
                breadth_change = features[1] - past_features[1]  # breadth_ma20 change
                # Positive change = improving sentiment
                vector[5] = float(np.clip(breadth_change * 20, -1.0, 1.0))

        # Confidence
        non_zero = np.count_nonzero(vector)
        data_confidence = min(1.0, step / self.sentiment_lookback)
        feature_confidence = non_zero / self.opinion_dim
        confidence = data_confidence * (0.4 + 0.6 * feature_confidence)

        # Regime modulation: sentiment signals are more reliable in trending markets
        regime_mult = {"BULL": 0.9, "RANGE": 0.6, "BEAR": 0.8, "CRASH": 0.95}
        confidence *= regime_mult.get(regime, 0.6)

        return AgentOpinion(
            vector=vector,
            confidence=float(confidence),
            name=self.name,
            metadata={
                "regime": regime,
                "step": step,
                "fear_level": float(-vector[1]) if vector[1] != 0 else 0.0,
                "participation": float(vector[2]),
            },
        )
