"""Walk-forward optimization with IS/OOS validation.

Prevents overfitting by:
1. Splitting data into train/test windows
2. Optimizing parameters on train set
3. Evaluating on unseen test set
4. Computing IS/OOS ratio (< 0.7 = overfitting warning)

Usage:
    from research.walk_forward import WalkForwardOptimizer
    opt = WalkForwardOptimizer(repo, market="cn")
    results = opt.optimize("momentum_top_k", param_grid={"lookback_days": [10, 20, 30]})
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from marketdata.repository import MarketDataRepository

# Walk-forward optimized params cache directory
WF_PARAMS_DIR = Path(__file__).resolve().parents[2] / "data" / "walk_forward_params"

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward optimization configuration."""

    train_months: int = 24  # Training window (months)
    test_months: int = 6  # Test window (months)
    step_months: int = 6  # Step size for rolling
    min_sharpe_oos: float = 0.0  # Minimum OOS sharpe to accept
    is_oos_ratio_threshold: float = 0.7  # IS/OOS ratio warning threshold


@dataclass
class WalkForwardResult:
    """Result of one walk-forward split."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: dict[str, Any]
    is_sharpe: float  # In-sample sharpe
    oos_sharpe: float  # Out-of-sample sharpe
    is_oos_ratio: float
    is_return: float
    oos_return: float
    is_max_dd: float
    oos_max_dd: float
    all_param_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationSummary:
    """Summary of full walk-forward optimization."""

    strategy_id: str
    splits: list[WalkForwardResult]
    best_overall_params: dict[str, Any]
    avg_oos_sharpe: float
    avg_oos_return: float
    avg_is_oos_ratio: float
    parameter_stability: dict[str, float]  # param -> std/mean (lower = more stable)
    overfitting_warnings: list[str]
    recommendation: str


def _generate_splits(
    start: date, end: date, config: WalkForwardConfig
) -> list[tuple[date, date, date, date]]:
    """Generate train/test date splits."""
    splits = []
    current = start
    while True:
        train_end = current + timedelta(days=config.train_months * 30)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=config.test_months * 30)
        if test_end > end:
            break
        splits.append((current, train_end, test_start, test_end))
        current += timedelta(days=config.step_months * 30)
    return splits


class WalkForwardOptimizer:
    """Walk-forward parameter optimization with OOS validation."""

    def __init__(
        self,
        repo: MarketDataRepository,
        *,
        market: str = "cn",
        config: WalkForwardConfig = WalkForwardConfig(),
    ):
        self.repo = repo
        self.market = market
        self.config = config

    def optimize(
        self,
        strategy_id: str,
        param_grid: dict[str, list[Any]],
        *,
        start: date | None = None,
        end: date | None = None,
        run_backtest_fn: Callable | None = None,
    ) -> OptimizationSummary:
        """Run walk-forward optimization.

        Args:
            strategy_id: Strategy to optimize.
            param_grid: Dict of param_name -> list of values to try.
            start: Backtest start date (default: 3 years before end).
            end: Backtest end date (default: today).
            run_backtest_fn: Custom backtest function. Signature:
                fn(strategy_id, params, start, end) -> metrics_dict

        Returns:
            OptimizationSummary with best params and diagnostics.
        """
        if end is None:
            end = date.today()
        if start is None:
            start = end - timedelta(days=365 * 3)

        splits = _generate_splits(start, end, self.config)
        if not splits:
            return OptimizationSummary(
                strategy_id=strategy_id,
                splits=[],
                best_overall_params={},
                avg_oos_sharpe=0,
                avg_oos_return=0,
                avg_is_oos_ratio=0,
                parameter_stability={},
                overfitting_warnings=["No valid splits generated"],
                recommendation="数据不足，无法进行滚动优化",
            )

        logger.info(
            "Walk-forward: %d splits for %s (%s → %s)",
            len(splits),
            strategy_id,
            start,
            end,
        )

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combos = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*param_values)
        ]

        split_results: list[WalkForwardResult] = []
        warnings: list[str] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            logger.info(
                "Split %d/%d: train=%s→%s, test=%s→%s",
                i + 1,
                len(splits),
                train_start,
                train_end,
                test_start,
                test_end,
            )

            # Evaluate all param combos on train set
            train_results = []
            for params in param_combos:
                metrics = self._run_backtest(
                    strategy_id, params, train_start, train_end, run_backtest_fn
                )
                if metrics:
                    train_results.append({"params": params, **metrics})

            if not train_results:
                warnings.append(f"Split {i+1}: no valid train results")
                continue

            # Find best params on train set
            best_train = max(train_results, key=lambda x: x.get("sharpe_ratio", 0))
            best_params = best_train["params"]
            is_sharpe = best_train.get("sharpe_ratio", 0)
            is_return = best_train.get("total_return", 0)
            is_max_dd = best_train.get("max_drawdown", 0)

            # Evaluate best params on test set
            oos_metrics = self._run_backtest(
                strategy_id, best_params, test_start, test_end, run_backtest_fn
            )
            oos_sharpe = oos_metrics.get("sharpe_ratio", 0) if oos_metrics else 0
            oos_return = oos_metrics.get("total_return", 0) if oos_metrics else 0
            oos_max_dd = oos_metrics.get("max_drawdown", 0) if oos_metrics else 0

            # IS/OOS ratio
            if abs(is_sharpe) > 0.01:
                is_oos_ratio = oos_sharpe / is_sharpe
            else:
                is_oos_ratio = 0

            if is_oos_ratio < self.config.is_oos_ratio_threshold and is_sharpe > 0:
                warnings.append(
                    f"Split {i+1}: IS/OOS ratio {is_oos_ratio:.2f} < {self.config.is_oos_ratio_threshold} (overfitting?)"
                )

            split_results.append(
                WalkForwardResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params=best_params,
                    is_sharpe=is_sharpe,
                    oos_sharpe=oos_sharpe,
                    is_oos_ratio=is_oos_ratio,
                    is_return=is_return,
                    oos_return=oos_return,
                    is_max_dd=is_max_dd,
                    oos_max_dd=oos_max_dd,
                    all_param_results=train_results,
                )
            )

        # Aggregate results
        if not split_results:
            return OptimizationSummary(
                strategy_id=strategy_id,
                splits=[],
                best_overall_params={},
                avg_oos_sharpe=0,
                avg_oos_return=0,
                avg_is_oos_ratio=0,
                parameter_stability={},
                overfitting_warnings=warnings,
                recommendation="无有效优化结果",
            )

        avg_oos_sharpe = np.mean([s.oos_sharpe for s in split_results])
        avg_oos_return = np.mean([s.oos_return for s in split_results])
        avg_is_oos_ratio = np.mean([s.is_oos_ratio for s in split_results])

        # Parameter stability: how consistent are best params across splits?
        param_stability = self._compute_param_stability(split_results)

        # Find most frequent best params
        best_overall = self._find_consensus_params(split_results)

        # Save best params to cache (for consumption by strategy builder)
        self._save_best_params(strategy_id, best_overall)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            avg_oos_sharpe, avg_is_oos_ratio, param_stability, warnings
        )

        return OptimizationSummary(
            strategy_id=strategy_id,
            splits=split_results,
            best_overall_params=best_overall,
            avg_oos_sharpe=float(avg_oos_sharpe),
            avg_oos_return=float(avg_oos_return),
            avg_is_oos_ratio=float(avg_is_oos_ratio),
            parameter_stability=param_stability,
            overfitting_warnings=warnings,
            recommendation=recommendation,
        )

    def _run_backtest(
        self,
        strategy_id: str,
        params: dict[str, Any],
        start: date,
        end: date,
        run_backtest_fn: Callable | None,
    ) -> dict[str, Any] | None:
        """Run a single backtest with given params."""
        if run_backtest_fn:
            try:
                return run_backtest_fn(strategy_id, params, start, end)
            except Exception as e:
                logger.warning("Backtest failed: %s", e)
                return None

        # Default: use portfolio backtest
        try:
            from services.portfolio_backtest import (
                PortfolioBacktestRequest,
                execute_portfolio_backtest,
            )

            req = PortfolioBacktestRequest(
                market=self.market,
                start=start,
                end=end,
                strategy=strategy_id,
                max_positions=int(params.get("max_positions", 10)),
                lookback_days=int(params.get("lookback_days", 20)),
                initial_cash=100_000,
                min_commission=0,
            )
            result = execute_portfolio_backtest(req)
            return result.get("metrics", {})
        except Exception as e:
            logger.warning("Backtest failed for %s: %s", params, e)
            return None

    def _compute_param_stability(
        self, splits: list[WalkForwardResult]
    ) -> dict[str, float]:
        """Compute parameter stability (coefficient of variation)."""
        if not splits:
            return {}

        param_values: dict[str, list[float]] = {}
        for split in splits:
            for k, v in split.best_params.items():
                if isinstance(v, (int, float)):
                    param_values.setdefault(k, []).append(float(v))

        stability = {}
        for param, values in param_values.items():
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values)
                # Coefficient of variation (lower = more stable)
                stability[param] = float(std / mean) if mean > 0 else float("inf")
            else:
                stability[param] = 0.0
        return stability

    def _find_consensus_params(
        self, splits: list[WalkForwardResult]
    ) -> dict[str, Any]:
        """Find most frequently selected parameter values."""
        if not splits:
            return {}

        from collections import Counter

        param_counts: dict[str, Counter] = {}
        for split in splits:
            for k, v in split.best_params.items():
                param_counts.setdefault(k, Counter())[v] += 1

        consensus = {}
        for param, counter in param_counts.items():
            consensus[param] = counter.most_common(1)[0][0]
        return consensus

    def _generate_recommendation(
        self,
        avg_oos_sharpe: float,
        avg_is_oos_ratio: float,
        param_stability: dict[str, float],
        warnings: list[str],
    ) -> str:
        """Generate human-readable recommendation."""
        parts = []

        if avg_oos_sharpe > 1.0:
            parts.append("✓ OOS夏普优秀(>1.0)")
        elif avg_oos_sharpe > 0.5:
            parts.append("✓ OOS夏普良好(>0.5)")
        elif avg_oos_sharpe > 0:
            parts.append("△ OOS夏普一般(0~0.5)")
        else:
            parts.append("✗ OOS夏普为负，策略无效")

        if avg_is_oos_ratio < 0.5:
            parts.append("⚠ 严重过拟合(IS/OOS<0.5)")
        elif avg_is_oos_ratio < 0.7:
            parts.append("⚠ 轻微过拟合(IS/OOS<0.7)")
        else:
            parts.append("✓ 无明显过拟合")

        unstable = [p for p, v in param_stability.items() if v > 0.5]
        if unstable:
            parts.append(f"⚠ 参数不稳定: {', '.join(unstable)}")
        else:
            parts.append("✓ 参数稳定")

        return " | ".join(parts)

    def _save_best_params(self, strategy_id: str, params: dict[str, Any]) -> None:
        """Persist best optimized params to disk for strategy builder consumption."""
        if not params:
            return
        WF_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
        path = WF_PARAMS_DIR / f"{strategy_id}.json"
        payload = {
            "strategy_id": strategy_id,
            "params": params,
            "updated_at": date.today().isoformat(),
            "source": "walk_forward",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved optimized params for %s → %s: %s", strategy_id, path, params)


def load_optimized_params(strategy_id: str) -> dict[str, Any] | None:
    """Load walk-forward optimized params for a strategy, if available.

    Called by strategy builder to override default parameters.
    Returns dict of param_name → value, or None if no cache exists.
    """
    path = WF_PARAMS_DIR / f"{strategy_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data.get("params", {})
        logger.debug("Loaded optimized params for %s: %s", strategy_id, params)
        return params
    except Exception as e:
        logger.warning("Failed to load optimized params for %s: %s", strategy_id, e)
        return None
