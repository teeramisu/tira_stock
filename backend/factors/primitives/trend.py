"""Trend / composite MA primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bbi(close: pd.Series, periods: tuple[int, ...]) -> pd.Series:
    """BBI = average of rolling means over ``periods``."""
    close = close.astype(float)
    parts = [close.rolling(p, min_periods=p).mean() for p in periods]
    if not parts:
        raise ValueError("periods must not be empty")
    return sum(parts) / len(parts)


def bbi_price_ratio(closes: np.ndarray, periods: tuple[int, ...] = (21, 34, 55, 89)) -> float:
    """Return BBI/price; values below 1.0 indicate price above BBI (bullish)."""
    if closes.size < max(periods):
        return float("inf")
    series = pd.Series(closes.astype(float))
    if series.iloc[-1] <= 0:
        return float("inf")
    bbi = compute_bbi(series, periods).iloc[-1]
    if np.isnan(bbi):
        return float("inf")
    return float(bbi / series.iloc[-1])


def ema5_bearish_gate(closes: np.ndarray) -> bool:
    """True when EMA5 gate blocks risk assets (close < EMA5 declining)."""
    if closes.size < 6:
        return True
    series = pd.Series(closes.astype(float))
    ema5 = series.ewm(span=5, adjust=False).mean()
    ema_close = float(series.iloc[-1])
    ema_5 = float(ema5.iloc[-1])
    ema_5_prev = float(ema5.iloc[-2])
    return bool(ema_close < ema_5 < ema_5_prev)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average; leading values are NaN."""
    if values.size < window:
        return np.full(values.size, np.nan)
    series = pd.Series(values.astype(float))
    return series.rolling(window, min_periods=window).mean().to_numpy()
