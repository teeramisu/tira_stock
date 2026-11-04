"""Reusable signal primitives (factors) for backtest and research."""

from factors.primitives.breadth import market_breadth_bullish
from factors.primitives.momentum import polyfit_log_momentum_score, wpetf_momentum_score
from factors.primitives.oscillators import rsi
from factors.primitives.timing import (
    compute_value_timing_gate_series,
    price_emotion_monitor,
    rsrs_ma_timing_allows,
    value_timing_gate_at,
    volume_emotion_monitor,
)
from factors.primitives.trend import bbi_price_ratio, compute_bbi, ema5_bearish_gate

__all__ = [
    "bbi_price_ratio",
    "compute_bbi",
    "compute_value_timing_gate_series",
    "ema5_bearish_gate",
    "market_breadth_bullish",
    "polyfit_log_momentum_score",
    "price_emotion_monitor",
    "rsi",
    "rsrs_ma_timing_allows",
    "value_timing_gate_at",
    "volume_emotion_monitor",
    "wpetf_momentum_score",
]
