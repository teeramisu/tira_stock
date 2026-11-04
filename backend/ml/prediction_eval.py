"""Prediction evaluation framework — ground truth vs model predictions.

Tracks every prediction made by the Transformer and compares it against
actual outcomes, building a labeled dataset for accuracy measurement.

Key metrics:
  - Return predictions: MAE, RMSE, hit_rate (direction correctness)
  - Regime predictions: confusion matrix, precision/recall per class
  - Baselines: always-predict-mean, last-observed, random

Output: a persistent SQLite/JSON dataset that grows over time,
allowing measurement of model accuracy decay.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EVAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "prediction_eval"


@dataclass
class PredictionRecord:
    """One prediction vs actual outcome."""

    prediction_date: str  # when the prediction was made
    target_date: str  # the date being predicted for
    horizon_days: int  # 5, 10, or 20
    predicted_return: float
    actual_return: float
    predicted_regime: int  # 0-3
    actual_regime: int  # 0-3 (based on what actually happened)
    regime_turning_point: int  # 1 if regime changed during horizon
    model_version: str = ""
    absolute_error: float = 0.0  # computed
    direction_correct: bool = False  # computed


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics for a set of predictions."""

    n_predictions: int
    return_mae: float
    return_rmse: float
    return_hit_rate: float  # % of times direction was correct
    regime_accuracy: float
    regime_confusion_matrix: list[list[int]]  # 4x4
    class_precision: dict[str, float]
    class_recall: dict[str, float]
    baseline_mae: float  # "always predict mean" baseline
    baseline_hit_rate: float  # "always predict last direction" baseline
    horizon_days: int
    period_start: str
    period_end: str


class PredictionTracker:
    """Tracks predictions vs actuals, stores results, computes metrics.

    Ground truth data source: since we have full history, after a prediction
    is made we can look forward to see what actually happened.
    """

    def __init__(self, data_dir: Path = EVAL_DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[PredictionRecord] = []
        self._load()

    def _load(self) -> None:
        path = self.data_dir / "predictions.jsonl"
        if not path.exists():
            return
        try:
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                self._records.append(PredictionRecord(**data))
        except Exception as e:
            logger.warning("Failed to load prediction records: %s", e)

    def _save(self) -> None:
        path = self.data_dir / "predictions.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for r in self._records[-1:]:  # append only latest
                f.write(json.dumps({
                    "prediction_date": r.prediction_date,
                    "target_date": r.target_date,
                    "horizon_days": r.horizon_days,
                    "predicted_return": r.predicted_return,
                    "actual_return": r.actual_return,
                    "predicted_regime": r.predicted_regime,
                    "actual_regime": r.actual_regime,
                    "regime_turning_point": r.regime_turning_point,
                    "model_version": r.model_version,
                    "absolute_error": r.absolute_error,
                    "direction_correct": r.direction_correct,
                }) + "\n")

    def record_prediction(
        self,
        prediction_date: date,
        predicted_return: float,
        predicted_regime: int,
        *,
        horizon_days: int = 5,
        model_version: str = "",
    ) -> PredictionRecord | None:
        """Record a prediction and look forward to get the ground truth.

        This uses the close price matrix to find the actual forward return
        and actual regime. Since we have full history, we can always
        get the ground truth for past predictions.
        """
        # Try to get actual outcome from market data
        try:
            from services.portfolio_backtest import market_repository
            repo = market_repository(require_exists=True)

            # Get close matrix for actual return computation
            import pandas as pd
            rows = repo.conn.execute(
                "SELECT trade_date, symbol, COALESCE(adj_close, close) AS close "
                "FROM daily_bars WHERE market = 'cn' ORDER BY trade_date, symbol"
            ).fetchdf()
            rows["trade_date"] = pd.to_datetime(rows["trade_date"])
            pivot = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")

            # Find prediction and target dates in the index
            pred_str = prediction_date.isoformat()
            if pred_str not in pivot.index:
                logger.warning("Prediction date %s not in market data", pred_str)
                return None

            idx = list(pivot.index).index(pred_str)
            target_idx = idx + horizon_days
            if target_idx >= len(pivot):
                logger.warning("Target date beyond market data range")
                return None

            target_date = pivot.index[target_idx]
            target_date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)

            # Actual forward return
            today_close = float(pivot.iloc[idx].mean())
            fwd_close = float(pivot.iloc[target_idx].mean())
            if not (np.isfinite(today_close) and np.isfinite(fwd_close) and today_close > 0):
                return None
            actual_return = fwd_close / today_close - 1.0

            # Actual regime based on forward return
            if actual_return < -0.10:
                actual_regime = 3  # CRASH
            elif actual_return < -0.05:
                actual_regime = 2  # BEAR
            elif actual_return > 0.05:
                actual_regime = 0  # BULL
            else:
                actual_regime = 1  # RANGE

            # Turning point: compare regime at prediction date vs target date
            # Use forward-looking actual_regime at both points
            pred_regime_actual = actual_regime  # This is the target's regime

            # Get regime at prediction date (from same forward-looking logic)
            if idx > 0:
                past_idx = max(0, idx - horizon_days)
                past_close = float(pivot.iloc[past_idx].mean())
                if np.isfinite(past_close) and past_close > 0:
                    past_return = today_close / past_close - 1.0
                    if past_return < -0.10:
                        pred_point_regime = 3
                    elif past_return < -0.05:
                        pred_point_regime = 2
                    elif past_return > 0.05:
                        pred_point_regime = 0
                    else:
                        pred_point_regime = 1
                else:
                    pred_point_regime = actual_regime
            else:
                pred_point_regime = actual_regime

            turning_point = 1 if pred_point_regime != actual_regime else 0

            record = PredictionRecord(
                prediction_date=pred_str,
                target_date=target_date_str,
                horizon_days=horizon_days,
                predicted_return=predicted_return,
                actual_return=round(actual_return, 6),
                predicted_regime=predicted_regime,
                actual_regime=actual_regime,
                regime_turning_point=turning_point,
                model_version=model_version,
                absolute_error=abs(predicted_return - actual_return),
                direction_correct=(predicted_return * actual_return) > 0,
            )

            self._records.append(record)
            self._save()
            return record

        except Exception as e:
            logger.error("Failed to record prediction: %s", e)
            return None

    def get_metrics(
        self,
        *,
        horizon_days: int | None = None,
        last_n: int | None = None,
    ) -> EvaluationMetrics:
        """Compute evaluation metrics over stored predictions."""
        records = self._records
        if horizon_days:
            records = [r for r in records if r.horizon_days == horizon_days]
        if last_n and last_n < len(records):
            records = records[-last_n:]

        if not records:
            return EvaluationMetrics(
                n_predictions=0, return_mae=0, return_rmse=0, return_hit_rate=0,
                regime_accuracy=0, regime_confusion_matrix=[[0]*4]*4,
                class_precision={}, class_recall={},
                baseline_mae=0, baseline_hit_rate=0,
                horizon_days=horizon_days or 0,
                period_start="", period_end="",
            )

        # Return metrics
        errors = np.array([r.absolute_error for r in records])
        preds = np.array([r.predicted_return for r in records])
        actuals = np.array([r.actual_return for r in records])

        return_mae = float(np.mean(errors))
        return_rmse = float(np.sqrt(np.mean(errors**2)))
        return_hit_rate = float(np.mean([r.direction_correct for r in records]))

        # Regime metrics
        n_classes = 4
        conf_matrix = [[0] * n_classes for _ in range(n_classes)]
        for r in records:
            if 0 <= r.actual_regime < n_classes and 0 <= r.predicted_regime < n_classes:
                conf_matrix[r.actual_regime][r.predicted_regime] += 1

        correct_regime = sum(conf_matrix[i][i] for i in range(n_classes))
        total_regime = sum(sum(row) for row in conf_matrix)
        regime_accuracy = correct_regime / total_regime if total_regime > 0 else 0

        # Per-class precision/recall
        class_names = ["BULL", "RANGE", "BEAR", "CRASH"]
        precision = {}
        recall = {}
        for i in range(n_classes):
            tp = conf_matrix[i][i]
            fp = sum(conf_matrix[j][i] for j in range(n_classes)) - tp
            fn = sum(conf_matrix[i][j] for j in range(n_classes)) - tp
            precision[class_names[i]] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[class_names[i]] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Baselines
        mean_return = float(np.mean(actuals))
        baseline_mae = float(np.mean(np.abs(preds - mean_return)))

        last_direction = np.sign(actuals[:-1]) if len(actuals) > 1 else np.array([0])
        actual_direction = np.sign(actuals[1:]) if len(actuals) > 1 else np.array([0])
        baseline_hit_rate = float(np.mean(last_direction == actual_direction)) if len(last_direction) > 0 else 0

        return EvaluationMetrics(
            n_predictions=len(records),
            return_mae=round(return_mae, 4),
            return_rmse=round(return_rmse, 4),
            return_hit_rate=round(return_hit_rate, 4),
            regime_accuracy=round(regime_accuracy, 4),
            regime_confusion_matrix=conf_matrix,
            class_precision=precision,
            class_recall=recall,
            baseline_mae=round(baseline_mae, 4),
            baseline_hit_rate=round(baseline_hit_rate, 4),
            horizon_days=horizon_days or 0,
            period_start=records[0].prediction_date,
            period_end=records[-1].prediction_date,
        )

    def summary(self) -> dict[str, Any]:
        """Return human-readable summary of prediction performance."""
        metrics_5d = self.get_metrics(horizon_days=5)
        metrics_10d = self.get_metrics(horizon_days=10)
        metrics_20d = self.get_metrics(horizon_days=20)

        def fmt(m: EvaluationMetrics) -> dict:
            return {
                "n": m.n_predictions,
                "mae": m.return_mae,
                "rmse": m.return_rmse,
                "hit_rate": m.return_hit_rate,
                "regime_acc": m.regime_accuracy,
                "baseline_mae": m.baseline_mae,
                "vs_baseline": f"{(m.baseline_mae - m.return_mae) / m.baseline_mae * 100:+.1f}%" if m.baseline_mae > 0 else "N/A",
            }

        return {
            "total_predictions": len(self._records),
            "horizon_5d": fmt(metrics_5d),
            "horizon_10d": fmt(metrics_10d),
            "horizon_20d": fmt(metrics_20d),
            "regime_confusion_5d": {
                "matrix": metrics_5d.regime_confusion_matrix,
                "accuracy": metrics_5d.regime_accuracy,
                "precision": metrics_5d.class_precision,
                "recall": metrics_5d.class_recall,
            },
        }


def generate_ground_truth_batch(
    repo: Any,
    *,
    market: str = "cn",
    start: date | None = None,
    end: date | None = None,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    """Generate a batch of ground truth labels for all dates in range.

    This creates the definitive "what actually happened" dataset:
    For each date T, what was the forward H-day return?
    What was the actual regime (by forward return magnitude)?

    This can be used to:
    1. Train the Transformer with real labels
    2. Evaluate any model's predictions
    3. Compute baselines

    Returns:
        Dict with ground truth records.
    """
    import pandas as pd

    rows = repo.conn.execute(
        "SELECT trade_date, symbol, COALESCE(adj_close, close) AS close "
        "FROM daily_bars WHERE market = ? ORDER BY trade_date",
        [market],
    ).fetchdf()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"])
    pivot = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")
    pivot.index = [d.date() if hasattr(d, "date") else d for d in pivot.index]

    if start:
        pivot = pivot[pivot.index >= start]
    if end:
        pivot = pivot[pivot.index <= end]

    records = []
    dates_list = list(pivot.index)
    for h in horizons:
        for i, d in enumerate(dates_list):
            if i + h >= len(dates_list):
                continue
            today_close = float(pivot.iloc[i].mean())
            fwd_close = float(pivot.iloc[i + h].mean())
            if not (np.isfinite(today_close) and np.isfinite(fwd_close) and today_close > 0):
                continue
            fwd_ret = fwd_close / today_close - 1.0

            if fwd_ret < -0.10:
                regime = 3
            elif fwd_ret < -0.05:
                regime = 2
            elif fwd_ret > 0.05:
                regime = 0
            else:
                regime = 1

            records.append({
                "date": d.isoformat(),
                "horizon": h,
                "forward_return": round(fwd_ret, 6),
                "regime": regime,
                "regime_name": ["BULL", "RANGE", "BEAR", "CRASH"][regime],
            })

    return {
        "n_records": len(records),
        "horizons": list(horizons),
        "date_range": {"start": str(dates_list[0]), "end": str(dates_list[-1])},
        "records": records,
    }
