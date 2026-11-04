"""Oscillator primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder-style RSI on a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)
