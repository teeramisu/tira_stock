"""Momentum scoring primitives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def polyfit_log_momentum_score(closes: np.ndarray, *, annualize_days: int = 250) -> float:
    """Score = annualized log-linear slope × R²."""
    if closes.size < 3:
        return float("-inf")
    values = closes.astype(float)
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        return float("-inf")
    y = np.log(values)
    x = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    annualized = float(np.exp(slope) ** annualize_days - 1.0)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("-inf")
    r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return annualized * r_squared


def _wpetf_polynomial(x: float) -> float:
    return -1.09e-7 * x**4 + 1.81e-5 * x**3 - 0.00109 * x**2 + 0.0292 * x + 0.283


def wpetf_momentum_score(
    closes: np.ndarray,
    *,
    m_days: int = 25,
    annualize_days: int = 250,
) -> float:
    """jq-0009 WPETF_Strategy combined momentum score."""
    if closes.size < max(m_days, 25):
        return float("-inf")
    values = closes.astype(float)
    if np.any(values <= 0):
        return float("-inf")
    y = np.log(values)
    x = np.arange(y.size, dtype=float)
    weights = np.exp(np.linspace(-1, 0, num=y.size))

    slope, _ = np.polyfit(x[-25:], y[-25:], 1)
    slope_s, _ = np.polyfit(x[-15:], y[-15:], 1)
    curve2, _, _ = np.polyfit(x[-25:], y[-25:], 2, w=weights[-25:])

    y_smooth = np.convolve(y[-25:], np.ones(5) / 5, mode="valid")
    x_smooth = np.arange(len(y_smooth), dtype=float)
    curve2_smooth, _, _ = np.polyfit(x_smooth, y_smooth, 2, w=weights[-len(y_smooth) :])

    ma20 = pd.Series(values).rolling(20, min_periods=20).mean()
    recent_close = float(values[-1])
    recent_ma = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else recent_close

    changes = np.diff(y[-10:])
    gains = changes[changes > 0].sum()
    losses = -changes[changes < 0].sum()
    if losses == 0:
        losses = 1e-6
    rsi = 100 - (100 / (1 + gains / losses))

    if (recent_close > recent_ma) and ((curve2_smooth < -0.0003) or (curve2 < -0.0006)):
        slope_adjust = 0.0
    elif (recent_close < recent_ma) and ((curve2_smooth > 0.0003) or (curve2 > 0.0006)):
        slope_adjust = slope + 0.005
    else:
        slope_adjust = slope

    annualized = math.pow(math.exp(slope_adjust), annualize_days) - 1
    annualized_s = math.pow(math.exp(slope_s), annualize_days) - 1

    y_fit = np.polyval([slope, np.polyfit(x[-25:], y[-25:], 1)[1]], x[-25:])
    _, intercept = np.polyfit(x[-25:], y[-25:], 1)
    y_fit = slope * x[-25:] + intercept
    ss_res = float(np.sum((y[-25:] - y_fit) ** 2))
    ss_tot = float(np.sum((y[-25:] - np.mean(y[-25:])) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0

    slope_s, intercept_s = np.polyfit(x[-15:], y[-15:], 1)
    y_fit_s = slope_s * x[-15:] + intercept_s
    ss_res_s = float(np.sum((y[-15:] - y_fit_s) ** 2))
    ss_tot_s = float(np.sum((y[-15:] - np.mean(y[-15:])) ** 2))
    r_squared_s = 1 - ss_res_s / ss_tot_s if ss_tot_s else 0.0

    combined = annualized * r_squared
    combined_s = annualized_s * r_squared_s
    if r_squared_s >= 0.8 and combined_s > combined:
        combined = combined_s

    if rsi > 95:
        combined = 0.0
    if rsi < 10:
        combined = combined + _wpetf_polynomial(float(rsi)) / 8.0
    if combined >= 25:
        combined = 0.0
    return float(combined)
