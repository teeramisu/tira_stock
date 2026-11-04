"""Market-level feature engineering for Transformer input.

Extracts cross-sectional and time-series features from daily bars to create
a feature matrix suitable for sequence modeling.

Features (per day, market-wide):
1. breadth_ma60:   % stocks above MA60
2. breadth_ma20:   % stocks above MA20
3. trend_20d:      median 20-day return
4. trend_5d:       median 5-day return
5. crash_5d:       5th percentile 5-day return (tail risk)
6. realized_vol:   20-day realized volatility (annualized)
7. vol_of_vol:     volatility of volatility (vol regime)
8. momentum_disp:  cross-sectional momentum dispersion (std of returns)
9. volume_trend:   20-day volume trend (linear regression slope)
10. breadth_mom:   breadth change over 5 days (acceleration)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd


# Feature names in order (must match model input dimension)
FEATURE_NAMES: tuple[str, ...] = (
    "breadth_ma60",
    "breadth_ma20",
    "trend_20d",
    "trend_5d",
    "crash_5d",
    "realized_vol",
    "vol_of_vol",
    "momentum_disp",
    "volume_trend",
    "breadth_mom",
)

# Regime label encoding (must match regime_detector.py)
REGIME_LABELS: tuple[str, ...] = ("BULL", "RANGE", "BEAR", "CRASH")
REGIME_TO_IDX: dict[str, int] = {r: i for i, r in enumerate(REGIME_LABELS)}


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction windows."""

    ma_long: int = 60
    ma_short: int = 20
    trend_long: int = 20
    trend_short: int = 5
    vol_window: int = 20
    vol_of_vol_window: int = 10
    min_stocks: int = 50
    # Forward horizons for labels
    forward_horizons: tuple[int, ...] = (5, 10, 20)


@dataclass
class FeatureMatrix:
    """Container for extracted features and labels."""

    dates: list[date]
    features: np.ndarray  # shape: (T, n_features)
    labels_return: dict[int, np.ndarray]  # horizon -> shape: (T,)
    labels_regime: np.ndarray  # shape: (T,) — forward regime (0-3)
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def n_samples(self) -> int:
        return len(self.dates)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


def build_feature_matrix(
    close_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame | None = None,
    regime_by_date: dict[date, str] | None = None,
    *,
    config: FeatureConfig = FeatureConfig(),
) -> FeatureMatrix:
    """Build feature matrix from close and volume matrices.

    Args:
        close_matrix: DataFrame with trade_date index, symbol columns, adj_close values.
        volume_matrix: Same shape as close_matrix, volume values (optional).
        regime_by_date: Pre-computed regime labels (optional, will compute if None).
        config: Feature extraction configuration.

    Returns:
        FeatureMatrix with aligned dates, features, and forward labels.
    """
    if close_matrix.empty:
        raise ValueError("close_matrix is empty")

    frame = close_matrix.sort_index()
    vol_frame = volume_matrix.sort_index() if volume_matrix is not None else None
    dates = frame.index.tolist()
    n = len(dates)

    # Pre-compute rolling MAs
    ma_long = frame.rolling(window=config.ma_long, min_periods=config.ma_long).mean()
    ma_short = frame.rolling(window=config.ma_short, min_periods=config.ma_short).mean()

    # Pre-compute daily returns
    daily_ret = frame.pct_change()

    # Feature arrays
    feat = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)

    for i in range(n):
        d = dates[i]
        row_close = frame.iloc[i]

        # --- F1: Breadth above MA60 ---
        if i >= config.ma_long:
            row_ma = ma_long.iloc[i]
            valid = row_close.notna() & row_ma.notna() & (row_ma > 0)
            if valid.sum() >= config.min_stocks:
                feat[i, 0] = (row_close[valid] > row_ma[valid]).mean()

        # --- F2: Breadth above MA20 ---
        if i >= config.ma_short:
            row_ma_s = ma_short.iloc[i]
            valid = row_close.notna() & row_ma_s.notna() & (row_ma_s > 0)
            if valid.sum() >= config.min_stocks:
                feat[i, 1] = (row_close[valid] > row_ma_s[valid]).mean()

        # --- F3: Trend 20d (median return) ---
        if i >= config.trend_long:
            past = frame.iloc[i - config.trend_long]
            valid = row_close.notna() & past.notna() & (past > 0)
            rets = row_close[valid] / past[valid] - 1.0
            if len(rets) > config.min_stocks:
                feat[i, 2] = float(rets.median())

        # --- F4: Trend 5d ---
        if i >= config.trend_short:
            past = frame.iloc[i - config.trend_short]
            valid = row_close.notna() & past.notna() & (past > 0)
            rets = row_close[valid] / past[valid] - 1.0
            if len(rets) > config.min_stocks:
                feat[i, 3] = float(rets.median())

        # --- F5: Crash 5d (5th percentile return) ---
        if i >= config.trend_short:
            past = frame.iloc[i - config.trend_short]
            valid = row_close.notna() & past.notna() & (past > 0)
            rets = row_close[valid] / past[valid] - 1.0
            if len(rets) > config.min_stocks:
                feat[i, 4] = float(rets.quantile(0.05))

        # --- F6: Realized vol (20d annualized, T-1 only, no forward look) ---
        if i >= config.vol_window:
            vol_slice = daily_ret.iloc[i - config.vol_window : i]
            port_ret = vol_slice.mean(axis=1)  # equal-weight daily return
            if port_ret.notna().sum() > 10:
                feat[i, 5] = float(port_ret.std() * np.sqrt(252))

        # --- F7: Vol of vol (volatility of realized vol) ---
        if i >= config.vol_window + config.vol_of_vol_window:
            vols = []
            for j in range(i - config.vol_of_vol_window + 1, i + 1):
                if not np.isnan(feat[j, 5]):
                    vols.append(feat[j, 5])
            if len(vols) > 3:
                feat[i, 6] = float(np.std(vols))

        # --- F8: Momentum dispersion (std of 20d returns across stocks) ---
        if i >= config.trend_long:
            past = frame.iloc[i - config.trend_long]
            valid = row_close.notna() & past.notna() & (past > 0)
            rets = row_close[valid] / past[valid] - 1.0
            if len(rets) > config.min_stocks:
                feat[i, 7] = float(rets.std())

        # --- F9: Volume trend (20d linear regression slope) ---
        if vol_frame is not None and i >= config.vol_window:
            vol_slice = vol_frame.iloc[i - config.vol_window + 1 : i + 1]
            total_vol = vol_slice.sum(axis=1)
            if total_vol.notna().sum() > 10:
                x = np.arange(len(total_vol), dtype=np.float64)
                y = total_vol.values.astype(np.float64)
                mask = np.isfinite(y)
                if mask.sum() > 10:
                    slope = np.polyfit(x[mask], y[mask], 1)[0]
                    # Normalize by mean volume
                    mean_vol = y[mask].mean()
                    if mean_vol > 0:
                        feat[i, 8] = float(slope / mean_vol)

        # --- F10: Breadth momentum (breadth change over 5 days) ---
        if i >= config.ma_long + config.trend_short:
            cur_breadth = feat[i, 0]
            past_breadth = feat[i - config.trend_short, 0]
            if not np.isnan(cur_breadth) and not np.isnan(past_breadth):
                feat[i, 9] = cur_breadth - past_breadth

    # --- Build labels ---
    labels_return: dict[int, np.ndarray] = {}
    for h in config.forward_horizons:
        lbl = np.full(n, np.nan, dtype=np.float32)
        for i in range(n - h):
            cur = frame.iloc[i].mean()  # equal-weight portfolio close
            fwd = frame.iloc[i + h].mean()
            if np.isfinite(cur) and cur > 0 and np.isfinite(fwd):
                lbl[i] = float(fwd / cur - 1.0)
        labels_return[h] = lbl

    # Forward regime labels
    labels_regime = np.full(n, -1, dtype=np.int32)
    if regime_by_date:
        for i in range(n):
            d = dates[i]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            elif hasattr(d, "date"):
                d = d.date()
            regime = regime_by_date.get(d)
            if regime and regime in REGIME_TO_IDX:
                labels_regime[i] = REGIME_TO_IDX[regime]

    # Trim NaN rows from start (warmup period)
    # Find first valid row
    valid_mask = ~np.isnan(feat[:, 0])  # breadth_ma60 as proxy
    first_valid = 0
    for i in range(n):
        if valid_mask[i]:
            first_valid = i
            break

    return FeatureMatrix(
        dates=[d if isinstance(d, date) else d.date() for d in dates[first_valid:]],
        features=feat[first_valid:],
        labels_return={h: lbl[first_valid:] for h, lbl in labels_return.items()},
        labels_regime=labels_regime[first_valid:],
    )


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features across time. Returns (normalized, mean, std).

    When mean/std are provided, applies them instead of computing from data.
    This avoids future-data leakage when normalizing validation/test folds.
    """
    if mean is None or std is None:
        mean = np.nanmean(features, axis=0, keepdims=True)
        std = np.nanstd(features, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
    normalized = (features - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized, mean, std


def create_sequences(
    fm: FeatureMatrix,
    *,
    lookback: int = 60,
    forward_horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[date]]:
    """Create sliding window sequences for Transformer training.

    Args:
        fm: FeatureMatrix from build_feature_matrix.
        lookback: Number of past days for input sequence.
        forward_horizon: Which forward return to use as primary label.

    Returns:
        (X, y_return, y_regime, dates) where:
        - X: shape (N, lookback, n_features) — raw (unnormalized) features
        - y_return: shape (N,) — forward return
        - y_regime: shape (N,) — forward regime class
        - dates: list of prediction dates

    Note:
        Normalization is NOT applied here to avoid future-data leakage.
        Callers (e.g. trainer) must normalize per-fold using normalize_features().
    """
    n = fm.n_samples
    if n < lookback + forward_horizon + 1:
        raise ValueError(f"Not enough samples: {n} < {lookback + forward_horizon + 1}")

    features = fm.features
    X_list, y_ret_list, y_reg_list, date_list = [], [], [], []

    return_labels = fm.labels_return.get(forward_horizon, fm.labels_return.get(5))
    if return_labels is None:
        raise ValueError(f"No return labels for horizon {forward_horizon}")

    for i in range(lookback, n - forward_horizon):
        # Check labels are valid
        if np.isnan(return_labels[i]) or fm.labels_regime[i] < 0:
            continue

        seq = features[i - lookback : i]  # (lookback, n_features) — raw features
        X_list.append(seq)
        y_ret_list.append(return_labels[i])
        y_reg_list.append(fm.labels_regime[i])
        date_list.append(fm.dates[i])

    if not X_list:
        raise ValueError("No valid sequences after filtering")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_ret_list, dtype=np.float32),
        np.array(y_reg_list, dtype=np.int64),
        date_list,
    )
