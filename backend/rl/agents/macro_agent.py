"""MacroAgent — produces macro-economic opinion vectors for RL-MACS.

Analyzes market-level macro environment using:
0. Regime strength (confidence from RegimeDetector)
1. Trend persistence (multi-horizon return consistency)
2. Volatility regime (vol-of-vol percentile)
3. Breadth momentum (market breadth acceleration)
4. Cross-market correlation (diversification signal)
5. Liquidity environment (volume trend)

Integrates with existing ml/features.py 10-dim market features and
research/regime_detector.py regime signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_rl_agent import AgentOpinion, BaseRLAgent


class MacroAgent(BaseRLAgent):
    """Macro-economic analysis agent producing 6-dimensional opinion vectors.

    This agent interprets market-level features to assess the macro environment.
    It uses the 10 market features from ml/features.py:
        0: breadth_ma60, 1: breadth_ma20, 2: trend_20d, 3: trend_5d,
        4: crash_5d, 5: realized_vol, 6: vol_of_vol, 7: momentum_disp,
        8: volume_trend, 9: breadth_mom
    """

    # Regime confidence mapping
    REGIME_BASE_CONFIDENCE: dict[str, float] = {
        "BULL": 0.8,
        "RANGE": 0.5,
        "BEAR": 0.7,
        "CRASH": 0.9,
    }

    def __init__(self, vol_lookback: int = 60):
        super().__init__(
            name="macro",
            description="Macro-economic environment agent (regime, trend, vol, breadth, cross-market, liquidity)",
        )
        self.vol_lookback = vol_lookback

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
        """Compute 6-dim macro opinion vector.

        Args:
            step: Current timestep.
            prices: (T, N_assets) price matrix.
            market_features: (T, 10) market-level features.
            regime: Current regime label.

        Returns:
            AgentOpinion with 6-dim vector.
        """
        vector = np.zeros(self.opinion_dim, dtype=np.float32)

        if step < 20 or len(market_features) <= step:
            return AgentOpinion(
                vector=vector, confidence=0.0, name=self.name,
                metadata={"reason": "insufficient_data"},
            )

        features = market_features[step]  # (10,)
        n_features = min(len(features), 10)

        # --- Dim 0: Regime strength ---
        # Map regime to directional signal: BULL=+1, RANGE=0, BEAR=-0.5, CRASH=-1
        regime_signal = {"BULL": 0.8, "RANGE": 0.0, "BEAR": -0.6, "CRASH": -1.0}
        base_signal = regime_signal.get(regime, 0.0)
        # Modulate by breadth (feature 0: breadth_ma60)
        if n_features > 0:
            breadth = features[0]  # 0-1 scale
            # Strengthen signal when breadth confirms
            breadth_confirmation = (breadth - 0.5) * 0.4  # [-0.2, +0.2]
            vector[0] = float(np.clip(base_signal + breadth_confirmation, -1.0, 1.0))
        else:
            vector[0] = base_signal

        # --- Dim 1: Trend persistence (multi-horizon consistency) ---
        if n_features >= 4:
            trend_20d = features[2]  # 20-day median return
            trend_5d = features[3]  # 5-day median return
            # Consistent trends (same sign, both positive) → strong signal
            if trend_20d > 0 and trend_5d > 0:
                persistence = min(1.0, (trend_20d + trend_5d) * 10)
            elif trend_20d < 0 and trend_5d < 0:
                persistence = max(-1.0, (trend_20d + trend_5d) * 10)
            else:
                # Mixed signals → weak
                persistence = (trend_5d - trend_20d) * 5  # deceleration
            vector[1] = float(np.clip(persistence, -1.0, 1.0))

        # --- Dim 2: Volatility regime (vol-of-vol percentile) ---
        if n_features >= 7 and step >= self.vol_lookback:
            current_vov = features[6]  # vol_of_vol
            # Compute percentile over lookback
            hist_vov = market_features[step - self.vol_lookback:step, 6]
            hist_vov = hist_vov[~np.isnan(hist_vov)]
            if len(hist_vov) > 5:
                percentile = float(np.mean(hist_vov < current_vov))
                # High vol-of-vol → negative (uncertain environment)
                vector[2] = float(1.0 - 2.0 * percentile)  # Map [0,1] → [1,-1]
            else:
                vector[2] = 0.0

        # --- Dim 3: Breadth momentum (acceleration) ---
        if n_features >= 10:
            breadth_mom = features[9]  # breadth_mom: 5-day breadth change
            # Positive breadth momentum = improving market internals
            vector[3] = float(np.clip(breadth_mom * 10, -1.0, 1.0))

        # --- Dim 4: Cross-market / diversification signal ---
        if n_features >= 8:
            momentum_disp = features[7]  # Cross-sectional momentum dispersion
            # High dispersion = more opportunity for stock picking
            # Normalize: typical range is [0, 0.1]
            disp_signal = momentum_disp * 20  # Scale up
            vector[4] = float(np.clip(disp_signal - 0.5, -1.0, 1.0))

        # --- Dim 5: Liquidity environment (volume trend) ---
        if n_features >= 9:
            volume_trend = features[8]  # 20-day volume linear trend
            # Positive volume trend = increasing participation = bullish
            vector[5] = float(np.clip(volume_trend * 5, -1.0, 1.0))

        # Confidence: higher when more features are available and non-zero
        non_zero = np.count_nonzero(vector)
        data_confidence = min(1.0, step / 60.0)
        feature_confidence = non_zero / self.opinion_dim
        confidence = data_confidence * (0.5 + 0.5 * feature_confidence)

        # Regime affects confidence
        confidence *= self.REGIME_BASE_CONFIDENCE.get(regime, 0.5)

        return AgentOpinion(
            vector=vector,
            confidence=float(confidence),
            name=self.name,
            metadata={
                "regime": regime,
                "step": step,
                "breadth": float(features[0]) if n_features > 0 else 0.0,
                "trend_20d": float(features[2]) if n_features > 2 else 0.0,
            },
        )
