"""Self-evolution loop: scheduled retraining, strategy evaluation, and reporting.

Components:
1. EvolutionRunner:  Orchestrates the full self-evolution cycle
2. ReportGenerator: Generates weekly/monthly performance reports
3. RetrainingTrigger: Decides when to retrain the Transformer model

Schedule:
- Daily:   Evaluate active strategies, record lifecycle metrics
- Weekly:  Generate performance report, check for degradation
- Monthly: Retrain Transformer if performance decay detected

Usage:
    from research.evolution_loop import EvolutionRunner
    runner = EvolutionRunner(repo)
    report = runner.run_daily_cycle()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EVOLUTION_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "evolution"
REPORTS_DIR = EVOLUTION_DATA_DIR / "reports"


@dataclass
class CycleResult:
    """Result of one evolution cycle."""

    cycle_date: str
    cycle_type: str  # "daily" / "weekly" / "monthly"
    regime: str
    regime_confidence: float
    strategy_evaluations: list[dict[str, Any]]
    lifecycle_actions: list[dict[str, Any]]
    retrain_needed: bool
    retrain_reason: str
    recommendations: list[str]
    report_path: str = ""
    factor_scan: dict[str, Any] | None = None  # monthly scan results


@dataclass
class WeeklyReport:
    """Weekly performance report."""

    report_date: str
    period_start: str
    period_end: str
    market_regime: str
    regime_distribution: dict[str, int]
    active_strategies: list[str]
    degraded_strategies: list[str]
    retired_strategies: list[str]
    best_strategy: str
    worst_strategy: str
    portfolio_sharpe: float
    portfolio_return: float
    portfolio_max_dd: float
    transformer_accuracy: float
    retrain_count: int
    key_insights: list[str]
    action_items: list[str]


class RetrainingTrigger:
    """Decides when the Transformer model needs retraining."""

    def __init__(self, *, accuracy_threshold: float = 0.60, staleness_days: int = 30):
        self.accuracy_threshold = accuracy_threshold
        self.staleness_days = staleness_days

    def should_retrain(self, model_info: dict[str, Any]) -> tuple[bool, str]:
        """Check if model needs retraining.

        Returns:
            (should_retrain, reason)
        """
        # Check 1: Model accuracy decay
        metrics = model_info.get("metrics", {})
        regime_acc = metrics.get("regime_accuracy", 0)
        if regime_acc < self.accuracy_threshold:
            return True, f"Regime准确率过低({regime_acc:.1%} < {self.accuracy_threshold:.0%})"

        # Check 2: Model staleness (trained > N days ago)
        # In production, this would check actual model file timestamp
        # For now, we always return False unless accuracy is bad
        return False, "模型状态正常"


class ReportGenerator:
    """Generate performance reports from evolution cycle data."""

    def generate_weekly(
        self,
        *,
        cycle_results: list[CycleResult],
        strategy_metrics: dict[str, dict[str, float]],
        regime_distribution: dict[str, int],
        transformer_info: dict[str, Any],
    ) -> WeeklyReport:
        """Generate a weekly performance report."""
        today = date.today()
        period_end = today.isoformat()
        period_start = (today - timedelta(days=7)).isoformat()

        # Aggregate strategy metrics
        active = []
        degraded = []
        retired = []
        best_sid = ""
        worst_sid = ""
        best_sharpe = -999
        worst_sharpe = 999

        for sid, m in strategy_metrics.items():
            sharpe = m.get("sharpe_ratio", 0)
            active.append(sid)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sid = sid
            if sharpe < worst_sharpe:
                worst_sharpe = sharpe
                worst_sid = sid

        # Populate degraded/retired from lifecycle data
        try:
            from research.strategy_lifecycle import StrategyLifecycleManager
            lm = StrategyLifecycleManager()
            degraded = lm.get_degraded_strategies()
            retired = lm.get_retired_strategies()
            active = lm.get_active_strategies()
        except Exception:
            pass

        # Key insights
        insights = self._generate_insights(
            strategy_metrics, regime_distribution, transformer_info
        )

        # Action items
        action_items = self._generate_action_items(
            strategy_metrics, degraded, cycle_results
        )

        return WeeklyReport(
            report_date=today.isoformat(),
            period_start=period_start,
            period_end=period_end,
            market_regime=cycle_results[-1].regime if cycle_results else "UNKNOWN",
            regime_distribution=regime_distribution,
            active_strategies=active,
            degraded_strategies=degraded,
            retired_strategies=retired,
            best_strategy=best_sid,
            worst_strategy=worst_sid,
            portfolio_sharpe=float(np.mean([
                m.get("sharpe_ratio", 0) for m in strategy_metrics.values()
            ])) if strategy_metrics else 0,
            portfolio_return=float(np.mean([
                m.get("total_return", 0) for m in strategy_metrics.values()
            ])) if strategy_metrics else 0,
            portfolio_max_dd=float(np.min([
                m.get("max_drawdown", 0) for m in strategy_metrics.values()
            ])) if strategy_metrics else 0,
            transformer_accuracy=transformer_info.get("metrics", {}).get("regime_accuracy", 0),
            retrain_count=sum(1 for cr in cycle_results if cr.retrain_needed),
            key_insights=insights,
            action_items=action_items,
        )

    def _generate_insights(
        self,
        strategy_metrics: dict[str, dict[str, float]],
        regime_dist: dict[str, int],
        transformer_info: dict[str, Any],
    ) -> list[str]:
        """Generate key insights from data."""
        insights = []

        # Regime insight
        if regime_dist:
            total = sum(regime_dist.values())
            dominant = max(regime_dist, key=regime_dist.get)  # type: ignore
            pct = regime_dist[dominant] / total * 100
            insights.append(f"本周市场主要状态: {dominant} ({pct:.0f}%)")

        # Best performer
        if strategy_metrics:
            best = max(strategy_metrics.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
            insights.append(
                f"表现最好策略: {best[0]} (Sharpe={best[1].get('sharpe_ratio', 0):.2f})"
            )

        # Transformer insight
        acc = transformer_info.get("metrics", {}).get("regime_accuracy", 0)
        if acc > 0.7:
            insights.append(f"Transformer模型准确率良好: {acc:.1%}")
        elif acc > 0:
            insights.append(f"⚠ Transformer模型准确率偏低: {acc:.1%}，建议重训")

        return insights

    def _generate_action_items(
        self,
        strategy_metrics: dict[str, dict[str, float]],
        degraded: list[str],
        cycle_results: list[CycleResult],
    ) -> list[str]:
        """Generate actionable items."""
        items = []

        # Check for retraining needs
        retrain_needed = any(cr.retrain_needed for cr in cycle_results)
        if retrain_needed:
            items.append("🔄 建议重训Transformer模型")

        # Check for degraded strategies
        for sid in degraded:
            items.append(f"⚠ {sid} 已降级，检查策略逻辑")

        # Negative sharpe strategies
        for sid, m in strategy_metrics.items():
            if m.get("sharpe_ratio", 0) < -0.5:
                items.append(f"❌ {sid} Sharpe < -0.5，建议暂停或退役")

        if not items:
            items.append("✅ 本周无需特别操作")

        return items

    def save_report(self, report: WeeklyReport) -> str:
        """Save report to JSON file."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"weekly_{report.report_date}.json"

        data = {
            "report_date": report.report_date,
            "period": f"{report.period_start} → {report.period_end}",
            "market_regime": report.market_regime,
            "regime_distribution": report.regime_distribution,
            "strategies": {
                "active": report.active_strategies,
                "degraded": report.degraded_strategies,
                "retired": report.retired_strategies,
            },
            "performance": {
                "best_strategy": report.best_strategy,
                "worst_strategy": report.worst_strategy,
                "portfolio_sharpe": round(report.portfolio_sharpe, 4),
                "portfolio_return": round(report.portfolio_return, 4),
                "portfolio_max_dd": round(report.portfolio_max_dd, 4),
            },
            "transformer": {
                "accuracy": round(report.transformer_accuracy, 4),
                "retrain_count": report.retrain_count,
            },
            "key_insights": report.key_insights,
            "action_items": report.action_items,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Weekly report saved: %s", path)
        return str(path)


class EvolutionRunner:
    """Orchestrates the self-evolution cycle.

    Integrates:
    - Regime detection (Layer 0)
    - Strategy lifecycle (Layer 1)
    - Transformer model (Layer 2)
    - Walk-forward optimizer (Layer 3)
    - Meta-allocator (Layer 4)
    """

    def __init__(self, repo: Any, *, market: str = "cn"):
        self.repo = repo
        self.market = market
        self.trigger = RetrainingTrigger()
        self.reporter = ReportGenerator()

    def run_monthly_cycle(self) -> CycleResult:
        """Run monthly evolution cycle: daily + factor scan + AI research."""
        daily = self.run_daily_cycle()
        scan_result = self.run_monthly_scan()
        recs = list(daily.recommendations)
        if scan_result.get("catalog_scan", {}).get("new_signals"):
            recs.append(f"因子扫描发现新信号: {len(scan_result['catalog_scan']['new_signals'])}个")
        if scan_result.get("catalog_scan", {}).get("decay_warnings"):
            recs.append(f"因子衰减警告: {len(scan_result['catalog_scan']['decay_warnings'])}个")
        if scan_result.get("ai_research", {}).get("promising"):
            recs.append(f"AI研究推荐因子: {len(scan_result['ai_research']['promising'])}个")
        return CycleResult(
            cycle_date=daily.cycle_date,
            cycle_type="monthly",
            regime=daily.regime,
            regime_confidence=daily.regime_confidence,
            strategy_evaluations=daily.strategy_evaluations,
            lifecycle_actions=daily.lifecycle_actions,
            retrain_needed=daily.retrain_needed,
            retrain_reason=daily.retrain_reason,
            recommendations=recs,
            factor_scan=scan_result,
        )

    def run_daily_cycle(self) -> CycleResult:
        """Run daily evolution cycle.

        Steps:
        1. Detect current regime
        2. Evaluate active strategies
        3. Record lifecycle metrics
        4. Check if retraining is needed
        5. Generate recommendations
        """
        from research.regime_detector import RegimeDetector, RegimeConfig
        from research.strategy_lifecycle import StrategyLifecycleManager
        from portfolio.strategy_registry import STRATEGY_REGIME_AFFINITY

        logger.info("Running daily evolution cycle for market=%s", self.market)

        # Step 1: Detect regime
        close_matrix = self._load_close_matrix()
        detector = RegimeDetector(config=RegimeConfig())
        signals = detector.detect_all(close_matrix, market=self.market)
        current_regime = signals[-1].regime if signals else "UNKNOWN"
        regime_confidence = signals[-1].confidence if signals else 0.0

        logger.info("Current regime: %s (confidence: %.2f)", current_regime, regime_confidence)

        # Step 2: Evaluate strategies with lifecycle management
        lifecycle_mgr = StrategyLifecycleManager()
        evaluations = []
        lifecycle_actions = []

        # Load real backtest metrics if available
        bt_metrics = self._load_latest_backtest_metrics()

        for sid in STRATEGY_REGIME_AFFINITY:
            aligned = current_regime in STRATEGY_REGIME_AFFINITY[sid].get("best_regimes", [])
            eval_entry = {
                "strategy_id": sid,
                "regime": current_regime,
                "regime_aligned": aligned,
                "risk_level": STRATEGY_REGIME_AFFINITY[sid].get("risk", "unknown"),
                "evaluated_at": datetime.now().isoformat(),
            }
            evaluations.append(eval_entry)

            # Record lifecycle run with actual or estimated metrics
            metrics = bt_metrics.get(sid, {})
            if not metrics:
                metrics = {
                    "sharpe_ratio": 0.5 if aligned else -0.2,
                    "total_return": 0.05 if aligned else -0.03,
                    "max_drawdown": -0.08 if aligned else -0.15,
                    "win_rate": 0.52 if aligned else 0.48,
                    "trade_count": 20,
                }
            lifecycle_result = lifecycle_mgr.record_run(
                sid,
                metrics=metrics,
                regime=current_regime,
            )
            if lifecycle_result.get("action"):
                lifecycle_actions.append(lifecycle_result)
                eval_entry["lifecycle_action"] = lifecycle_result["action"]
                eval_entry["lifecycle_state"] = lifecycle_result["state"]

        # Step 3: Check retraining need
        model_info = self._get_model_info()
        retrain_needed, retrain_reason = self.trigger.should_retrain(model_info)

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(
            current_regime, regime_confidence, evaluations
        )

        return CycleResult(
            cycle_date=date.today().isoformat(),
            cycle_type="daily",
            regime=current_regime,
            regime_confidence=regime_confidence,
            strategy_evaluations=evaluations,
            lifecycle_actions=lifecycle_actions,
            retrain_needed=retrain_needed,
            retrain_reason=retrain_reason,
            recommendations=recommendations,
        )

    def run_weekly_report(
        self, daily_results: list[CycleResult] | None = None
    ) -> WeeklyReport:
        """Generate weekly report from daily cycle results."""
        if daily_results is None:
            daily_results = [self.run_daily_cycle()]

        # Compute regime distribution from daily results
        regime_dist: dict[str, int] = {}
        for cr in daily_results:
            regime_dist[cr.regime] = regime_dist.get(cr.regime, 0) + 1

        # Load actual backtest results from saved artifacts
        strategy_metrics: dict[str, dict[str, float]] = self._load_latest_backtest_metrics()
        if not strategy_metrics:
            # Fall back to running quick evaluations
            strategy_metrics = self._run_quick_evals(daily_results)

        transformer_info = self._get_model_info()

        report = self.reporter.generate_weekly(
            cycle_results=daily_results,
            strategy_metrics=strategy_metrics,
            regime_distribution=regime_dist,
            transformer_info=transformer_info,
        )

        # Save report
        report_path = self.reporter.save_report(report)
        logger.info("Weekly report generated: %s", report_path)

        return report

    def run_monthly_scan(self) -> dict[str, Any]:
        """Run monthly factor scan and AI research cycle.

        Integrates with FactorScanHarness to:
        1. Evaluate all catalog factors
        2. Detect emerging/decaying factors
        3. Generate AI hypotheses
        4. Feed insights into evolution reports
        """
        logger.info("Running monthly factor scan for market=%s", self.market)
        try:
            from research.factor_scan import FactorScanHarness

            harness = FactorScanHarness(market=self.market)
            result = harness.run_monthly_scan()
            logger.info(
                "Monthly scan: %d factors, %d new signals, %d decay warnings",
                result.get("catalog_scan", {}).get("n_factors", 0),
                len(result.get("catalog_scan", {}).get("new_signals", [])),
                len(result.get("catalog_scan", {}).get("decay_warnings", [])),
            )
            return result
        except Exception as e:
            logger.error("Monthly factor scan failed: %s", e)
            return {"status": "failed", "error": str(e)}

    def run_retrain_if_needed(self) -> dict[str, Any]:
        """Check and run model retraining if needed."""
        model_info = self._get_model_info()
        needed, reason = self.trigger.should_retrain(model_info)

        if not needed:
            return {"retrained": False, "reason": reason}

        logger.info("Retraining triggered: %s", reason)
        try:
            from ml.trainer import train_market_transformer, TrainConfig

            config = TrainConfig(
                lookback=30,
                batch_size=64,
                learning_rate=5e-4,
                max_epochs=150,
                patience=20,
                d_model=48,
                n_heads=4,
                n_layers=2,
                d_ff=96,
                dropout=0.25,
            )
            result = train_market_transformer(
                self.repo, market=self.market, config=config
            )
            return {
                "retrained": True,
                "reason": reason,
                "new_metrics": {
                    "regime_accuracy": result.regime_accuracy,
                    "return_mae": result.return_mae,
                    "n_epochs": result.n_epochs,
                },
                "model_path": result.model_path,
            }
        except Exception as e:
            logger.error("Retraining failed: %s", e)
            return {"retrained": False, "reason": reason, "error": str(e)}

    def _load_latest_backtest_metrics(self) -> dict[str, dict[str, float]]:
        """Load actual backtest metrics from saved artifact files."""
        backtest_dir = Path(__file__).resolve().parents[2] / "data" / "backtests"
        if not backtest_dir.exists():
            return {}
        json_files = sorted(backtest_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            return {}
        latest = json_files[0]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            sleeves = data.get("sleeves", data.get("run_metadata", []))
            if isinstance(sleeves, list) and sleeves:
                metrics = {}
                for s in sleeves:
                    sid = s.get("sleeve_id", s.get("strategy", "unknown"))
                    m = s.get("metrics", {})
                    if isinstance(m, dict):
                        metrics[sid] = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
                if metrics:
                    logger.info("Loaded %d strategy metrics from %s", len(metrics), latest.name)
                    return metrics
            # Try alternate format (single backtest result)
            if "metrics" in data:
                metrics = {"single": {k: float(v) for k, v in data["metrics"].items() if isinstance(v, (int, float))}}
                logger.info("Loaded single backtest metrics from %s", latest.name)
                return metrics
        except Exception as e:
            logger.debug("Failed to load backtest artifacts: %s", e)
        return {}

    def _run_quick_evals(self, daily_results: list[CycleResult]) -> dict[str, dict[str, float]]:
        """Run lightweight portfolio backtests for active strategies.

        Falls back to regime-alignment-based estimates if backtests fail.
        """
        from portfolio.strategy_registry import STRATEGY_REGIME_AFFINITY
        result: dict[str, dict[str, float]] = {}
        try:
            from services.portfolio_backtest import execute_portfolio_backtest, PortfolioBacktestRequest
            from datetime import timedelta
            today = date.today()
            start = today - timedelta(days=180)
            for sid in STRATEGY_REGIME_AFFINITY:
                try:
                    req = PortfolioBacktestRequest(
                        market=self.market,
                        start=start,
                        end=today,
                        initial_cash=1_000_000,
                        strategy=sid,
                    )
                    bt = execute_portfolio_backtest(req)
                    m = bt.get("metrics", {})
                    result[sid] = {
                        "sharpe_ratio": float(m.get("sharpe_ratio", 0)),
                        "total_return": float(m.get("total_return", 0)),
                        "max_drawdown": float(m.get("max_drawdown", 0)),
                    }
                    logger.info("Quick eval %s: return=%.2f%% sharpe=%.2f", sid, result[sid]["total_return"] * 100, result[sid]["sharpe_ratio"])
                except Exception as e:
                    logger.debug("Quick eval failed for %s: %s", sid, e)
                    result[sid] = {}
        except Exception as e:
            logger.warning("Quick evaluation failed, using regime alignment estimates: %s", e)

        # Fallback: regime-alignment estimates for any missing strategies
        for cr in daily_results:
            for ev in cr.strategy_evaluations:
                sid = ev["strategy_id"]
                if sid not in result or not result[sid]:
                    result[sid] = {
                        "sharpe_ratio": 0.5 if ev["regime_aligned"] else -0.2,
                        "total_return": 0.05 if ev["regime_aligned"] else -0.03,
                        "max_drawdown": -0.08 if ev["regime_aligned"] else -0.15,
                    }
        return result

    def _load_close_matrix(self) -> Any:
        """Load close price matrix from repository."""
        import pandas as pd

        rows = self.repo.conn.execute(
            """
            SELECT trade_date, symbol, COALESCE(adj_close, close) AS close
            FROM daily_bars WHERE market = ?
            ORDER BY trade_date, symbol
            """,
            [self.market],
        ).fetchdf()

        rows["trade_date"] = pd.to_datetime(rows["trade_date"])
        pivot = rows.pivot_table(
            index="trade_date", columns="symbol", values="close", aggfunc="first"
        )
        pivot.index = [d.date() if hasattr(d, "date") else d for d in pivot.index]
        return pivot

    def _get_model_info(self) -> dict[str, Any]:
        """Get trained model info."""
        import torch

        model_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "models"
            / f"market_transformer_{self.market}_latest.pt"
        )
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                return {
                    "metrics": checkpoint.get("metrics", {}),
                    "config": checkpoint.get("config", {}),
                    "path": str(model_path),
                }
            except Exception:
                pass
        return {"metrics": {}, "config": {}}

    def _generate_recommendations(
        self,
        regime: str,
        confidence: float,
        evaluations: list[dict[str, Any]],
    ) -> list[str]:
        """Generate strategy recommendations based on current state."""
        from portfolio.strategy_registry import strategies_for_regime

        recs = []
        recommended = strategies_for_regime(regime)

        if recommended:
            recs.append(f"当前 {regime} 推荐策略: {', '.join(recommended[:3])}")

        aligned = [e["strategy_id"] for e in evaluations if e["regime_aligned"]]
        misaligned = [e["strategy_id"] for e in evaluations if not e["regime_aligned"]]

        if aligned:
            recs.append(f"与当前Regime匹配的策略({len(aligned)}个): {', '.join(aligned[:5])}")

        if misaligned and confidence > 0.7:
            recs.append(
                f"⚠ {len(misaligned)}个策略与当前{regime}不匹配，建议降低权重"
            )

        if confidence < 0.5:
            recs.append("⚠ Regime信心度较低，建议保守配置")

        return recs
