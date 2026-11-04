"""Timing gates and emotion monitors."""

from __future__ import annotations

import numpy as np
import pandas as pd

import pandas as pd

from factors.primitives.oscillators import rsi
from factors.primitives.trend import rolling_mean

RISK_NORMAL = 2
RISK_WARNING = 1


def compute_value_timing_gate_series(
    close: pd.Series,
    *,
    ma_period: int = 250,
    ma_high: float = 2.5,
    ma_low: float = 0.30,
    ma_recover_low: float = 0.35,
    ma_recover_high: float = 0.70,
) -> pd.Series:
    """Return 0/1 gate series aligned with ``close`` (uses prior-bar indicators)."""
    close = close.astype(float)
    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    ma_rate = close / ma.replace(0, np.nan)
    rsi15 = rsi(close, 15)
    rsi60 = rsi(close, 60)
    rsi90 = rsi(close, 90)

    gate = pd.Series(0.0, index=close.index)
    status = RISK_NORMAL
    for i in range(len(close)):
        if i == 0:
            continue
        rate = ma_rate.iloc[i - 1]
        if np.isnan(rate) or rate <= 0:
            gate.iloc[i] = gate.iloc[i - 1]
            continue
        if status == RISK_NORMAL:
            if rate > ma_high or rate < ma_low:
                status = RISK_WARNING
        elif status == RISK_WARNING:
            if ma_recover_low <= rate <= ma_recover_high:
                status = RISK_NORMAL
        r15 = rsi15.iloc[i - 1]
        r60 = rsi60.iloc[i - 1]
        r90 = rsi90.iloc[i - 1]
        if status == RISK_WARNING:
            could_trade = (
                55 < r15 < 90
                and 50 < r90 < 90
                and not (np.isnan(r15) or np.isnan(r90))
            )
        else:
            could_trade = 50 < r60 < 99 and not np.isnan(r60)
        gate.iloc[i] = 1.0 if could_trade else 0.0
    return gate.fillna(0).astype(int)


def value_timing_gate_at(
    close: pd.Series,
    as_of_index: int,
    *,
    ma_period: int = 250,
    ma_high: float = 2.5,
    ma_low: float = 0.30,
    ma_recover_low: float = 0.35,
    ma_recover_high: float = 0.70,
) -> bool:
    """Evaluate value-timing gate at a specific bar index."""
    if as_of_index <= 0 or as_of_index >= len(close):
        return False
    subset = close.iloc[: as_of_index + 1]
    gate = compute_value_timing_gate_series(
        subset,
        ma_period=ma_period,
        ma_high=ma_high,
        ma_low=ma_low,
        ma_recover_low=ma_recover_low,
        ma_recover_high=ma_recover_high,
    )
    return bool(gate.iloc[-1])


def rsrs_ma_timing_allows(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    n: int = 18,
    m: int = 120,
    threshold: float = 0.7,
    mean_day: int = 20,
    mean_diff_day: int = 3,
) -> bool:
    """RSRS zscore×R² gate with short vs long MA window mean comparison."""
    if close.size < mean_day + mean_diff_day or high.size < n or low.size < n:
        return False
    close_vals = close.astype(float)
    today_ma = float(np.mean(close_vals[-mean_day:]))
    before_ma = float(np.mean(close_vals[:mean_day]))

    x = low[-n:].astype(float)
    y = high[-n:].astype(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return False
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot else 0.0

    slopes: list[float] = []
    need = n + m
    if high.size < need:
        return False
    for i in range(m):
        seg_x = low[i : i + n].astype(float)
        seg_y = high[i : i + n].astype(float)
        if np.std(seg_x) == 0 or np.std(seg_y) == 0:
            slopes.append(0.0)
            continue
        seg_slope, _ = np.polyfit(seg_x, seg_y, 1)
        slopes.append(float(seg_slope))
    slope_arr = np.asarray(slopes, dtype=float)
    std = float(np.std(slope_arr))
    if std <= 0:
        return False
    zscore = (float(slope) - float(np.mean(slope_arr))) / std
    rsrs_score = zscore * r2
    return bool(rsrs_score > threshold and today_ma > before_ma)


def volume_emotion_monitor(
    volume: np.ndarray,
    *,
    ma_window: int = 7,
    below_ma_days: int = 6,
) -> int:
    """
    jq-0020 EmotionMonitor: 1 bullish, 0 neutral, -1 bearish (volume below MA streak).
    """
    if volume.size < ma_window + max(below_ma_days, 3):
        return 0
    vol = volume.astype(float)
    vma = rolling_mean(vol, ma_window)
    if not np.isfinite(vma[-1]) or vma[-1] <= 0:
        return 0
    diff = vol - vma
    if diff[-1] >= 0:
        return 1 if bool(np.all(diff[-3:] >= 0)) else 0
    return -1 if bool(np.all(diff[-below_ma_days:] < 0)) else 0


def price_emotion_monitor(
    close: np.ndarray,
    *,
    ma_window: int = 9,
    below_ma_days: int = 8,
    crash_threshold_pct: float = -1.1,
) -> int:
    """
    jq-0020 EmotionMonitorNewZZ500 on price: 1/0/-1, -2 on intraday crash proxy.
    """
    if close.size < ma_window + max(below_ma_days, 3):
        return 0
    prices = close.astype(float)
    pma = rolling_mean(prices, ma_window)
    if not np.isfinite(pma[-1]) or pma[-1] <= 0:
        return 0
    diff = prices - pma
    if diff[-1] >= 0:
        ret_val = 1 if bool(np.all(diff[-3:] >= 0)) else 0
    else:
        ret_val = -1 if bool(np.all(diff[-below_ma_days:] < 0)) else 0
    if prices.size >= 2 and prices[-2] > 0:
        day_ret_pct = (prices[-1] / prices[-2] - 1.0) * 100.0
        if day_ret_pct < crash_threshold_pct:
            return -2
    return ret_val


def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, *, period: int = 21) -> float:
    """Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100.

    Values near -100 indicate oversold (buy signal); near 0 indicate overbought.
    """
    if high.size < period or low.size < period or close.size < period:
        return -50.0
    hh = float(np.max(high[-period:]))
    ll = float(np.min(low[-period:]))
    c = float(close[-1])
    if hh == ll:
        return -50.0
    return (hh - c) / (hh - ll) * -100.0


def rsrs_zscore_slope(zscores: np.ndarray, *, k: int = 8) -> float:
    """Slope of RSRS z-score over last K periods (momentum of RSRS itself).

    Negative slope + positive zscore = sell signal (RSRS turning down).
    """
    if zscores.size < 2:
        return 0.0
    y = zscores[-k:].astype(float)
    x = np.arange(y.size, dtype=float)
    if y.size < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def bias_momentum_score(
    closes: np.ndarray,
    *,
    bias_period: int = 90,
    momentum_days: int = 20,
    scale: float = 1.0,
) -> float:
    """乖离动量: polyfit of (close/MA_bias) normalized over momentum_days.

    From jq-0010 V2.1: bias = close / MA(bias_period), then score =
    polyfit of (bias/bias[0]) over momentum_days. Higher = stronger
    deviation momentum.
    """
    if closes.size < bias_period + momentum_days:
        return float("-inf")
    c = closes.astype(float)
    ma = pd.Series(c).rolling(bias_period, min_periods=bias_period).mean().to_numpy()
    bias = c[-momentum_days:] / ma[-momentum_days:]
    if np.any(bias <= 0) or not np.all(np.isfinite(bias)):
        return float("-inf")
    norm = bias / bias[0]
    x = np.arange(momentum_days, dtype=float)
    slope, _ = np.polyfit(x, norm, 1)
    return float(slope) * scale


def ma_confluence_ratio(closes: np.ndarray, periods: tuple[int, ...] = (13, 34, 55)) -> float:
    """Maximum pairwise distance between MAs as a fraction of price.

    Returns max(|MA_i - MA_j| / close for all i,j). Lower = more confluent.
    Used in 均线黏合 (MA consolidation) detection.
    """
    if closes.size < max(periods):
        return float("inf")
    c = closes.astype(float)
    price = float(c[-1])
    if price <= 0:
        return float("inf")
    mas = [float(pd.Series(c).rolling(p, min_periods=p).mean().iloc[-1]) for p in periods]
    mas = [m for m in mas if np.isfinite(m)]
    if len(mas) < 2:
        return float("inf")
    max_diff = max(abs(mas[i] - mas[j]) for i in range(len(mas)) for j in range(i + 1, len(mas)))
    return max_diff / price


def breakout_above_high(closes: np.ndarray, *, lookback: int = 30) -> bool:
    """True when close breaks above the highest close in lookback period."""
    if closes.size < lookback + 1:
        return False
    c = closes.astype(float)
    prev_high = float(np.max(c[-(lookback + 1):-1]))
    return bool(c[-1] > prev_high)


def consecutive_below_ma(closes: np.ndarray, *, ma_period: int = 13, days: int = 3) -> bool:
    """True when close has been below MA for N consecutive days."""
    if closes.size < ma_period + days:
        return False
    c = closes.astype(float)
    ma = pd.Series(c).rolling(ma_period, min_periods=ma_period).mean().to_numpy()
    if c.size < days or ma.size < days:
        return False
    return bool(np.all(c[-days:] < ma[-days:]))


def index_pullback_entry_rate(closes: np.ndarray, *, lookback: int = 375) -> float:
    """
    jq-0003 max_increase ``rate1``: 1 - drawdown_from_peak / rally_range.

    Lower values mean deeper pullback from recent peak (entry-friendly when < 0.318).
    """
    if closes.size < 60:
        return 1.0
    window = closes.astype(float)[-lookback:]
    peak_idx = int(np.argmax(window))
    peak = float(window[peak_idx])
    if peak <= 0:
        return 1.0
    after_peak = window[peak_idx:]
    trough_after = float(np.min(after_peak)) if after_peak.size else peak
    pre_peak = window[: peak_idx + 1]
    rally_low = float(np.min(pre_peak)) if pre_peak.size else peak
    growth = peak - rally_low
    if growth <= 0:
        return 1.0
    drawdown = peak - trough_after
    return float(1.0 - drawdown / growth)
