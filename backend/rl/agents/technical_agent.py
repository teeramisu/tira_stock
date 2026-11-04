"""TechnicalAgent — produces technical analysis opinion vectors for RL-MACS.

Computes 6 technical indicator dimensions from price data:
0. Trend strength (ADX equivalent via directional movement)
1. Moving average alignment score
2. RSI position (oversold/overbought)
3. MACD momentum signal
4. Bollinger Band position
5. Volume confirmation degree

All values normalized to [-1, 1].
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_rl_agent import AgentOpinion, BaseRLAgent


class TechnicalAgent(BaseRLAgent):
    """Technical analysis agent producing 6-dimensional opinion vectors.

    Computes classical technical indicators from OHLCV data and outputs
    a normalized signal vector. Works with close prices only (no need for
    full OHLCV in the RL context).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        super().__init__(
            name="technical",
            description="Technical analysis agent (trend, MA alignment, RSI, MACD, Bollinger, volume)",
        )
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std

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
        """Compute 6-dim technical opinion vector.

        Args:
            step: Current timestep.
            prices: (T, N_assets) price matrix.
            market_features: (T, 10) market features.
            regime: Current regime label.

        Returns:
            AgentOpinion with 6-dim vector.
        """
        vector = np.zeros(self.opinion_dim, dtype=np.float32)

        min_lookback = max(self.macd_slow + self.macd_signal, self.bb_period, 30)
        if step < min_lookback:
            return AgentOpinion(
                vector=vector, confidence=0.0, name=self.name,
                metadata={"reason": "insufficient_data"},
            )

        # Ensure 2D
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        n_assets = prices.shape[1]
        valid_mask = prices[step] > 0

        if valid_mask.sum() == 0:
            return AgentOpinion(vector=vector, confidence=0.0, name=self.name)

        # Compute per-asset signals, then average cross-sectionally
        trend_scores = []
        ma_scores = []
        rsi_scores = []
        macd_scores = []
        bb_scores = []
        vol_scores = []

        for i in range(n_assets):
            if not valid_mask[i]:
                continue

            close = prices[:step + 1, i]
            if len(close) < min_lookback or close[-1] <= 0:
                continue

            # --- Dim 0: Trend strength (ADX-like) ---
            trend = self._compute_trend_strength(close)
            trend_scores.append(trend)

            # --- Dim 1: MA alignment ---
            ma_align = self._compute_ma_alignment(close)
            ma_scores.append(ma_align)

            # --- Dim 2: RSI ---
            rsi = self._compute_rsi(close, self.rsi_period)
            # Map RSI [0,100] → [-1, 1]: 30→-0.4, 50→0, 70→+0.4
            rsi_signal = (rsi - 50) / 50.0
            rsi_scores.append(rsi_signal)

            # --- Dim 3: MACD ---
            macd_sig = self._compute_macd_signal(close)
            macd_scores.append(macd_sig)

            # --- Dim 4: Bollinger Band position ---
            bb_pos = self._compute_bb_position(close, self.bb_period, self.bb_std)
            bb_scores.append(bb_pos)

            # --- Dim 5: Volume confirmation (price change vs vol change) ---
            vol_conf = self._compute_volume_confirmation(close)
            vol_scores.append(vol_conf)

        # Average across assets
        if trend_scores:
            vector[0] = float(np.clip(np.mean(trend_scores), -1.0, 1.0))
        if ma_scores:
            vector[1] = float(np.clip(np.mean(ma_scores), -1.0, 1.0))
        if rsi_scores:
            vector[2] = float(np.clip(np.mean(rsi_scores), -1.0, 1.0))
        if macd_scores:
            vector[3] = float(np.clip(np.mean(macd_scores), -1.0, 1.0))
        if bb_scores:
            vector[4] = float(np.clip(np.mean(bb_scores), -1.0, 1.0))
        if vol_scores:
            vector[5] = float(np.clip(np.mean(vol_scores), -1.0, 1.0))

        # Confidence
        n_valid = len(trend_scores)
        confidence = min(1.0, n_valid / max(n_assets * 0.3, 1)) * min(1.0, step / min_lookback)

        return AgentOpinion(
            vector=vector,
            confidence=float(confidence),
            name=self.name,
            metadata={
                "n_valid_assets": n_valid,
                "regime": regime,
                "step": step,
                "avg_rsi": float(np.mean(rsi_scores) * 50 + 50) if rsi_scores else 50.0,
            },
        )

    # ---- Technical indicator computations ----

    def _compute_trend_strength(self, close: np.ndarray, period: int = 14) -> float:
        """Compute ADX-like trend strength indicator.

        Returns value in [-1, 1]: positive = uptrend, negative = downtrend.
        Magnitude = trend strength.
        """
        if len(close) < period + 1:
            return 0.0

        # Simple directional movement using price changes
        changes = np.diff(close[-period - 1:])
        up_moves = np.maximum(changes, 0)
        down_moves = np.maximum(-changes, 0)

        plus_dm = np.sum(up_moves)
        minus_dm = np.sum(down_moves)
        total_dm = plus_dm + minus_dm

        if total_dm < 1e-8:
            return 0.0

        # Directional index
        plus_di = plus_dm / total_dm
        minus_di = minus_dm / total_dm

        # DX = |DI+ - DI-| / (DI+ + DI-)
        dx = abs(plus_di - minus_di)

        # Direction: positive if uptrend
        direction = np.sign(plus_di - minus_di)

        return float(direction * dx)

    def _compute_ma_alignment(self, close: np.ndarray) -> float:
        """Compute moving average alignment score.

        Perfect bullish alignment (MA5 > MA10 > MA20 > MA60) → +1
        Perfect bearish alignment → -1
        Mixed → 0
        """
        if len(close) < 60:
            if len(close) >= 20:
                ma5 = close[-5:].mean()
                ma10 = close[-10:].mean()
                ma20 = close[-20:].mean()
                score = 0
                if ma5 > ma10:
                    score += 1
                else:
                    score -= 1
                if ma10 > ma20:
                    score += 1
                else:
                    score -= 1
                return score / 2.0
            return 0.0

        ma5 = close[-5:].mean()
        ma10 = close[-10:].mean()
        ma20 = close[-20:].mean()
        ma60 = close[-60:].mean()

        # Count bullish alignments
        score = 0
        pairs = [(ma5, ma10), (ma10, ma20), (ma20, ma60)]
        for fast, slow in pairs:
            if fast > slow:
                score += 1
            else:
                score -= 1

        # Normalize to [-1, 1]
        return score / 3.0

    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Compute RSI value (0-100)."""
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close[-period - 1:])
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss < 1e-8:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    def _compute_macd_signal(self, close: np.ndarray) -> float:
        """Compute MACD histogram signal normalized to [-1, 1]."""
        if len(close) < self.macd_slow + self.macd_signal:
            return 0.0

        # EMA computation
        ema_fast = self._ema(close, self.macd_fast)
        ema_slow = self._ema(close, self.macd_slow)

        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.macd_signal)

        # Histogram (last value)
        histogram = macd_line[-1] - signal_line[-1]

        # Normalize by price level
        price_level = close[-1]
        if price_level > 0:
            normalized = histogram / price_level * 100  # Percentage
        else:
            normalized = 0.0

        return float(np.clip(normalized * 10, -1.0, 1.0))

    def _compute_bb_position(self, close: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
        """Compute Bollinger Band position.

        Returns: -1 at lower band, 0 at middle, +1 at upper band.
        """
        if len(close) < period:
            return 0.0

        window = close[-period:]
        mid = np.mean(window)
        std = np.std(window)

        if std < 1e-8:
            return 0.0

        upper = mid + num_std * std
        lower = mid - num_std * std
        band_width = upper - lower

        if band_width < 1e-8:
            return 0.0

        # Position within bands: [lower, upper] → [-1, 1]
        position = (close[-1] - mid) / (band_width / 2)
        return float(np.clip(position, -1.0, 1.0))

    def _compute_volume_confirmation(self, close: np.ndarray) -> float:
        """Compute volume-price confirmation using price volatility as volume proxy.

        Positive: price rising with increasing "volume" (volatility expansion)
        Negative: price rising with decreasing "volume" (weak rally)
        """
        if len(close) < 10:
            return 0.0

        # Use absolute price changes as volume proxy
        abs_changes = np.abs(np.diff(close))
        recent_activity = np.mean(abs_changes[-5:])
        older_activity = np.mean(abs_changes[-10:-5]) + 1e-8

        activity_ratio = recent_activity / older_activity

        # Price direction
        price_change = (close[-1] - close[-5]) / close[-5] if close[-5] > 0 else 0.0

        # Confirmation: activity increasing in direction of price move
        if price_change > 0 and activity_ratio > 1.1:
            return min(1.0, (activity_ratio - 1.0) * 2)  # Bullish confirmation
        elif price_change < 0 and activity_ratio > 1.1:
            return max(-1.0, -(activity_ratio - 1.0) * 2)  # Bearish confirmation
        elif price_change > 0 and activity_ratio < 0.9:
            return -0.3  # Weak rally (bearish divergence)
        else:
            return 0.0

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
