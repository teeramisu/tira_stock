"""Automatic market regime detection using breadth, trend, and volatility signals.

Regime labels:
- BULL:    broad uptrend, most stocks above MA60
- RANGE:   sideways, mixed signals
- BEAR:    broad downtrend, few stocks above MA60
- CRASH:   acute panic, 5-day median return < -4%

Design:
- Uses the full A-share universe (or a representative sample) as the signal source.
- Computes 4 features daily, then classifies into one of 4 regimes.
- Returns a per-day regime series suitable for overlay on equity curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration for regime detection thresholds."""

    ma_period: int = 60
    trend_period: int = 20
    crash_period: int = 5

    # Breadth thresholds
    bull_breadth: float = 0.55  # > 55% stocks above MA60 → bullish
    bear_breadth: float = 0.35  # < 35% stocks above MA60 → bearish

    # Trend thresholds (median return over trend_period)
    bull_trend: float = 0.00  # > 0% → bullish
    bear_trend: float = -0.03  # < -3% → bearish

    # Crash guard (median return over crash_period)
    crash_threshold: float = -0.04  # < -4% → crash

    # Minimum stocks needed for reliable breadth calculation
    min_stocks: int = 50


@dataclass(frozen=True)
class RegimeSignal:
    """One day's regime detection result."""

    trade_date: date
    regime: str  # BULL / RANGE / BEAR / CRASH
    breadth: float  # % stocks above MA60
    trend_20d: float  # median 20-day return
    crash_5d: float  # median 5-day return
    realized_vol: float  # 20-day realized volatility (annualized)
    confidence: float  # 0-1 confidence score


@dataclass
class RegimeDetector:
    """Detect market regime from a close-price matrix.

    Usage:
        detector = RegimeDetector(config=RegimeConfig())
        daily_regimes = detector.detect_all(close_matrix)
    """

    config: RegimeConfig = field(default_factory=RegimeConfig)

    def detect_all(
        self,
        close_matrix: pd.DataFrame,
        *,
        market: str = "cn",
    ) -> list[RegimeSignal]:
        """Compute regime for every date in the close matrix.

        Args:
            close_matrix: DataFrame with trade_date index and symbol columns,
                          values are adjusted close prices.
            market: Market code for threshold tuning.

        Returns:
            List of RegimeSignal, one per date where enough data exists.
        """
        if close_matrix.empty:
            return []

        cfg = self.config
        frame = close_matrix.sort_index()
        dates = frame.index.tolist()
        symbols = frame.columns.tolist()

        # Pre-compute MA60 for all symbols (rolling mean)
        ma60 = frame.rolling(window=cfg.ma_period, min_periods=cfg.ma_period).mean()

        # Pre-compute daily returns for volatility
        daily_returns = frame.pct_change()

        signals: list[RegimeSignal] = []

        for i, d in enumerate(dates):
            if i < cfg.ma_period:
                continue  # Not enough history for MA60

            # --- Feature 1: Breadth (% stocks above MA60) ---
            row_close = frame.loc[d]
            row_ma = ma60.loc[d]
            valid_mask = row_close.notna() & row_ma.notna() & (row_ma > 0)
            valid_close = row_close[valid_mask]
            valid_ma = row_ma[valid_mask]

            if len(valid_close) < cfg.min_stocks:
                continue

            above_ma60 = (valid_close > valid_ma).sum()
            breadth = float(above_ma60 / len(valid_close))

            # --- Feature 2: Trend (median return over trend_period) ---
            if i < cfg.trend_period:
                trend_20d = 0.0
            else:
                d_past = dates[i - cfg.trend_period]
                past_close = frame.loc[d_past]
                valid_mask_trend = (
                    row_close.notna() & past_close.notna() & (past_close > 0)
                )
                returns_trend = (
                    row_close[valid_mask_trend] / past_close[valid_mask_trend] - 1.0
                )
                trend_20d = float(returns_trend.median()) if len(returns_trend) > 0 else 0.0

            # --- Feature 3: Crash (median return over crash_period) ---
            if i < cfg.crash_period:
                crash_5d = 0.0
            else:
                d_crash = dates[i - cfg.crash_period]
                past_close_crash = frame.loc[d_crash]
                valid_mask_crash = (
                    row_close.notna() & past_close_crash.notna() & (past_close_crash > 0)
                )
                returns_crash = (
                    row_close[valid_mask_crash]
                    / past_close_crash[valid_mask_crash]
                    - 1.0
                )
                crash_5d = float(returns_crash.median()) if len(returns_crash) > 0 else 0.0

            # --- Feature 4: Realized volatility (20-day annualized, T-1 only) ---
            if i >= cfg.trend_period:
                vol_slice = daily_returns.iloc[i - cfg.trend_period : i]
                avg_vol = float(vol_slice.std().mean()) if not vol_slice.empty else 0.0
                realized_vol = float(avg_vol * np.sqrt(252)) if np.isfinite(avg_vol) else 0.0
            else:
                realized_vol = 0.0

            # --- Classification ---
            regime, confidence = self._classify(
                breadth=breadth,
                trend_20d=trend_20d,
                crash_5d=crash_5d,
                realized_vol=realized_vol,
                market=market,
            )

            signals.append(
                RegimeSignal(
                    trade_date=d if isinstance(d, date) else d.date(),
                    regime=regime,
                    breadth=breadth,
                    trend_20d=trend_20d,
                    crash_5d=crash_5d,
                    realized_vol=realized_vol,
                    confidence=confidence,
                )
            )

        return signals

    def _classify(
        self,
        *,
        breadth: float,
        trend_20d: float,
        crash_5d: float,
        realized_vol: float,
        market: str,
    ) -> tuple[str, float]:
        """Classify regime and compute confidence score.

        Returns:
            (regime_label, confidence_0_to_1)
        """
        cfg = self.config

        # Priority: CRASH > BEAR > BULL > RANGE
        # Crash is the highest priority — acute danger overrides everything
        if crash_5d < cfg.crash_threshold:
            # Confidence: how far below crash threshold
            crash_depth = min(1.0, abs(crash_5d) / abs(cfg.crash_threshold))
            return "CRASH", min(1.0, 0.6 + crash_depth * 0.4)

        # Bear: broad downtrend
        is_bear = breadth < cfg.bear_breadth and trend_20d < cfg.bear_trend
        if is_bear:
            bear_strength = (
                0.5 * (1.0 - breadth / cfg.bear_breadth)
                + 0.5 * min(1.0, abs(trend_20d) / abs(cfg.bear_trend))
            )
            return "BEAR", min(1.0, 0.5 + bear_strength * 0.5)

        # Bull: broad uptrend
        is_bull = breadth > cfg.bull_breadth and trend_20d > cfg.bull_trend
        if is_bull:
            bull_strength = (
                0.5 * (breadth - cfg.bull_breadth) / (1.0 - cfg.bull_breadth)
                + 0.5 * min(1.0, trend_20d / 0.05)
            )
            return "BULL", min(1.0, 0.5 + bull_strength * 0.5)

        # Range: everything else
        # Confidence: how close to the boundary (0.5 breadth, 0 trend)
        dist_from_boundary = abs(breadth - 0.5) + abs(trend_20d)
        range_conf = max(0.3, 1.0 - dist_from_boundary)
        return "RANGE", min(1.0, range_conf)

    def detect_latest(
        self,
        close_matrix: pd.DataFrame,
        *,
        market: str = "cn",
    ) -> RegimeSignal | None:
        """Return only the most recent regime signal."""
        signals = self.detect_all(close_matrix, market=market)
        return signals[-1] if signals else None

    def regime_summary(
        self,
        signals: list[RegimeSignal],
    ) -> dict[str, Any]:
        """Summarize regime distribution over a period.

        Returns:
            Dict with regime counts, percentages, and recent trend.
        """
        if not signals:
            return {"total_days": 0, "regimes": {}}

        total = len(signals)
        regime_counts: dict[str, int] = {}
        for s in signals:
            regime_counts[s.regime] = regime_counts.get(s.regime, 0) + 1

        # Recent regime (last 5 days)
        recent = signals[-5:] if len(signals) >= 5 else signals
        recent_regimes = [s.regime for s in recent]

        # Dominant regime
        dominant = max(regime_counts, key=regime_counts.get)  # type: ignore[arg-type]

        # Latest signal
        latest = signals[-1]

        return {
            "total_days": total,
            "regimes": {
                r: {"days": c, "pct": round(c / total * 100, 1)}
                for r, c in sorted(regime_counts.items())
            },
            "dominant_regime": dominant,
            "dominant_pct": round(regime_counts[dominant] / total * 100, 1),
            "recent_regimes": recent_regimes,
            "latest": {
                "trade_date": latest.trade_date.isoformat(),
                "regime": latest.regime,
                "breadth": round(latest.breadth, 4),
                "trend_20d": round(latest.trend_20d, 4),
                "crash_5d": round(latest.crash_5d, 4),
                "confidence": round(latest.confidence, 2),
            },
        }


# --- Strategy-Regime Routing Table ---

# Maps regime → position scale (0.0 = fully cash, 1.0 = full position)
# Applied as a multiplier on strategy-generated target weights.
REGIME_POSITION_SCALE: dict[str, float] = {
    "BULL": 1.0,    # Full position
    "RANGE": 0.6,   # 60% position
    "BEAR": 0.3,    # 30% position (defensive)
    "CRASH": 0.0,   # Full cash
}

# Maps regime → recommended strategy families
REGIME_STRATEGY_AFFINITY: dict[str, list[str]] = {
    "BULL": [
        "momentum_top_k",
        "multi_factor_top_k",
        "jq_value_timing_top_k",
        "risk_adj_momentum_top_k",
    ],
    "RANGE": [
        "mean_reversion_top_k",
        "jq_small_cap_top_k",
        "jq_ma_consolidation_breakout",
    ],
    "BEAR": [
        "jq_etf_momentum_rsrs_top1",
        "jq_wide_base_etf_bbi_rotation",
        "jq_etf_value_averaging",
    ],
    "CRASH": [
        "jq_etf_value_averaging",  # DCA into crash
    ],
}

# Strategies that are robust across all regimes (no scaling applied)
REGIME_IMMUNE_STRATEGIES: frozenset[str] = frozenset({
    "jq_etf_value_averaging",  # DCA is regime-agnostic by design
})


def position_scale_for_regime(
    regime: str,
    strategy_id: str,
) -> float:
    """Return the position weight multiplier for a given regime and strategy.

    Some strategies are immune to regime scaling (e.g., DCA strategies).
    """
    if strategy_id in REGIME_IMMUNE_STRATEGIES:
        return 1.0
    return REGIME_POSITION_SCALE.get(regime, 1.0)


def regime_recommendation(regime: str) -> dict[str, Any]:
    """Return a human-readable recommendation for the given regime."""
    recommendations = {
        "BULL": {
            "action": "积极做多",
            "position": "80-100%",
            "strategy_hint": "优选动量/多因子策略，追逐强势股",
            "risk_hint": "注意止盈，设置移动止损",
        },
        "RANGE": {
            "action": "谨慎操作",
            "position": "40-60%",
            "strategy_hint": "优选均值回归/小盘策略，捕捉震荡收益",
            "risk_hint": "降低仓位，避免追高",
        },
        "BEAR": {
            "action": "防守为主",
            "position": "10-30%",
            "strategy_hint": "切换ETF轮动/避险策略，远离个股",
            "risk_hint": "控制回撤优先，可配置黄金/债券ETF",
        },
        "CRASH": {
            "action": "现金为王",
            "position": "0%",
            "strategy_hint": "清仓观望，可启动定投抄底",
            "risk_hint": "不要抄底！等待市场企稳信号",
        },
    }
    rec = recommendations.get(regime, recommendations["RANGE"])
    rec["regime"] = regime
    return rec
