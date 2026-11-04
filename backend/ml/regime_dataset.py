"""Regime dataset builder — using FUTURE data as ground truth.

Key insight: we have full historical data, so we can look FORWARD from each date
and label it based on what actually happened in the future.

Labels (using future N-day forward returns):
  - BULL:  forward return > +5%  (strong upside)
  - BEAR:  forward return < -5%  (strong downside)
  - CRASH: forward return < -10% (panic)
  - RANGE: everything else

Turning point labels:
  - 1 if the regime at T+N differs from current regime at T
  - 0 otherwise

Output: (features, labels) where labels are derived from actual future outcomes,
         not from rule-based classifications of the present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LabelStrategy = Literal["forward_return", "forward_regime"]

REGIME_LABELS = ("BULL", "RANGE", "BEAR", "CRASH")
REGIME_TO_IDX = {r: i for i, r in enumerate(REGIME_LABELS)}


@dataclass
class RegimeDataset:
    """Container for regime-labeled feature dataset.

    Labels are derived from FUTURE data (forward N-day returns/regime).
    This is the ground truth — what actually happened, not what rules predicted.
    """

    features: np.ndarray  # (T, n_features)
    labels: np.ndarray  # (T,) regime: 0=BULL..3=CRASH, turning_point: 0/1
    turning_points: np.ndarray  # (T,) — 1 if regime changes within forward horizon
    dates: list[date]
    label_source: str
    feature_names: list[str]
    forward_horizon: int  # N days used for forward labels
    class_distribution: dict[str, int]

    @property
    def n_samples(self) -> int:
        return len(self.dates)

    @property
    def n_features(self) -> int:
        return self.features.shape[1] if self.features.ndim > 1 else 0


class RegimeDatasetBuilder:
    """Build regime datasets using FUTURE data as ground truth labels.

    Since we have full history, we can look forward from each date T
    and label it by what actually happened in the next N days.
    This gives us REAL regime labels, not rule-based estimates.
    """

    def __init__(self, market: str = "cn", forward_horizon: int = 20):
        self.market = market
        self.forward_horizon = forward_horizon
        self._feature_cache: dict[str, np.ndarray] = {}
        self._date_cache: list[date] = []

    def load_close_matrix(self, repo: Any) -> pd.DataFrame:
        """Load close price matrix for forward return computation."""
        rows = repo.conn.execute(
            "SELECT trade_date, symbol, COALESCE(adj_close, close) AS close "
            "FROM daily_bars WHERE market = ? ORDER BY trade_date",
            [self.market],
        ).fetchdf()
        rows["trade_date"] = pd.to_datetime(rows["trade_date"])
        pivot = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")
        pivot.index = [d.date() if hasattr(d, 'date') else d for d in pivot.index]
        return pivot

    def load_features(self, repo: Any) -> tuple[np.ndarray, list[date], list[str]]:
        if self._feature_cache:
            return self._feature_cache["feat"], self._date_cache, list(self._feature_cache.get("names", []))
        from ml.features import build_feature_matrix, FEATURE_NAMES, FeatureConfig
        rows = repo.conn.execute(
            "SELECT trade_date, symbol, COALESCE(adj_close, close) AS close, volume "
            "FROM daily_bars WHERE market = ? ORDER BY trade_date, symbol",
            [self.market],
        ).fetchdf()
        rows["trade_date"] = pd.to_datetime(rows["trade_date"])
        close_pivot = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")
        vol_pivot = rows.pivot_table(index="trade_date", columns="symbol", values="volume", aggfunc="first")
        cfg = FeatureConfig(min_stocks=30)
        fm = build_feature_matrix(close_pivot, vol_pivot, config=cfg)
        features = np.nan_to_num(fm.features, nan=0.0)
        mean, std = np.nanmean(features, axis=0, keepdims=True), np.nanstd(features, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        features = (features - mean) / std
        self._feature_cache = {"feat": features, "names": list(FEATURE_NAMES)}
        self._date_cache = fm.dates
        return features, fm.dates, list(FEATURE_NAMES)

    def build(
        self,
        repo: Any,
        *,
        strategy: LabelStrategy = "forward_return",
        start: date | None = None,
        end: date | None = None,
    ) -> RegimeDataset:
        """Build regime dataset using future data as ground truth.

        Args:
            repo: MarketDataRepository.
            strategy:
                - "forward_return": label by actual forward N-day return magnitude
                - "forward_regime": label by the regime that dominates the next N days
            start/end: Optional date range filter.

        Returns:
            RegimeDataset with forward-looking labels.
        """
        features, dates, fnames = self.load_features(repo)
        close_matrix = self.load_close_matrix(repo)
        H = self.forward_horizon

        if strategy == "forward_return":
            labels, turning = self._label_by_forward_return(close_matrix, dates, H)
        elif strategy == "forward_regime":
            labels, turning = self._label_by_forward_regime(close_matrix, dates, H)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if start or end:
            mask = np.ones(len(dates), dtype=bool)
            if start:
                mask &= np.array([d >= start for d in dates])
            if end:
                mask &= np.array([d <= end for d in dates])
            features = features[mask]
            labels = labels[mask]
            turning = turning[mask]
            dates = [d for i, d in enumerate(dates) if mask[i]]

        dist = {name: int((labels == idx).sum()) for idx, name in enumerate(REGIME_LABELS)}

        return RegimeDataset(
            features=features, labels=labels, turning_points=turning,
            dates=dates, label_source=strategy, feature_names=fnames,
            forward_horizon=H, class_distribution=dist,
        )

    def _label_by_forward_return(
        self, close_matrix: pd.DataFrame, dates: list[date], H: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Label each date by actual forward H-day market return.

        This is the TRUE regime label: what actually happened in the future.

        BULL  → forward return > +5%
        CRASH → forward return < -10%
        BEAR  → forward return < -5%
        RANGE → everything else
        """
        n = len(dates)
        labels = np.full(n, REGIME_TO_IDX["RANGE"], dtype=np.int32)
        turning = np.zeros(n, dtype=np.int32)
        date_index = {d: i for i, d in enumerate(close_matrix.index)}

        for i, d in enumerate(dates):
            if d not in date_index:
                continue
            idx = date_index[d]
            if idx + H >= len(close_matrix):
                continue
            today_close = close_matrix.iloc[idx].mean()
            fwd_close = close_matrix.iloc[idx + H].mean()
            if not (np.isfinite(today_close) and np.isfinite(fwd_close) and today_close > 0):
                continue
            fwd_ret = fwd_close / today_close - 1.0

            if fwd_ret < -0.10:
                labels[i] = REGIME_TO_IDX["CRASH"]
            elif fwd_ret < -0.05:
                labels[i] = REGIME_TO_IDX["BEAR"]
            elif fwd_ret > 0.05:
                labels[i] = REGIME_TO_IDX["BULL"]
            else:
                labels[i] = REGIME_TO_IDX["RANGE"]

        # Turning point: regime changes between T and T+H
        for i in range(len(labels) - H):
            if labels[i] != labels[i + H]:
                turning[i] = 1

        return labels, turning

    def _label_by_forward_regime(
        self, close_matrix: pd.DataFrame, dates: list[date], H: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Label each date by the MAJORITY regime in the forward H-day window.

        Uses the rule-based RegimeDetector on each day in the forward window,
        then takes the most common regime as the label for today.
        """
        from research.regime_detector import RegimeDetector, RegimeConfig

        detector = RegimeDetector(config=RegimeConfig())
        signals = detector.detect_all(close_matrix, market=self.market)
        signal_map = {s.trade_date: s.regime for s in signals}

        n = len(dates)
        labels = np.full(n, REGIME_TO_IDX["RANGE"], dtype=np.int32)
        turning = np.zeros(n, dtype=np.int32)
        date_to_idx = {d: i for i, d in enumerate(dates)}

        for i, d in enumerate(dates):
            if i + H >= n:
                continue
            # Look forward H days, collect regimes
            fwd_regimes = []
            for j in range(i, min(i + H, n)):
                r = signal_map.get(dates[j])
                if r and r in REGIME_TO_IDX:
                    fwd_regimes.append(REGIME_TO_IDX[r])
            if fwd_regimes:
                labels[i] = max(set(fwd_regimes), key=fwd_regimes.count)

        # Turning point
        for i in range(len(labels) - H):
            if labels[i] != labels[i + H]:
                turning[i] = 1

        return labels, turning


def build_regime_sequences(
    ds: RegimeDataset,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[date]]:
    """Create sliding window sequences for regime Transformer training.

    Returns:
        (X, y_regime, y_turning, train_mask, dates) where:
        - X: (N, lookback, n_features)
        - y_regime: (N,) forward-looking regime label (0-3)
        - y_turning: (N,) 1 = turning point ahead, 0 = stable
        - train_mask: (N,) True for valid labels
        - dates: list of prediction dates
    """
    n = ds.n_samples
    if n < lookback + 5:
        raise ValueError(f"Not enough samples: {n} < {lookback + 5}")

    X_list, y_reg_list, y_turn_list, date_list = [], [], [], []
    for i in range(lookback, n):
        X_list.append(ds.features[i - lookback : i])
        y_reg_list.append(ds.labels[i])
        y_turn_list.append(ds.turning_points[i])
        date_list.append(ds.dates[i])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_reg_list, dtype=np.int64),
        np.array(y_turn_list, dtype=np.int64),
        np.array(y_reg_list) >= 0,
        date_list,
    )
