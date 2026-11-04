"""Factor scanning harness: periodic factor evaluation, tracking, and decay detection.

Components:
1. FactorScanner:   Runs FactorBacktestEngine on a schedule, saves results
2. FactorTracker:   Tracks IC/Rank IC over time, detects decay/emergence
3. FactorScanReport: Weekly/monthly factor performance summary

Usage:
    from research.factor_scan import FactorScanHarness
    harness = FactorScanHarness()
    scan_result = harness.run_scan(market="cn")
    print(scan_result["new_signals"])
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FACTOR_SCAN_DIR = Path(__file__).resolve().parents[2] / "data" / "factor_scan"


@dataclass
class FactorScanRecord:
    """One scan's result for a single factor."""

    scan_date: str
    factor_name: str
    ic: float
    rank_ic: float
    ic_ir: float
    rank_ic_ir: float
    spread: float
    n_obs: int
    category: str = ""
    time_horizon: str = ""


@dataclass
class FactorTrend:
    """Factor performance trend over recent scans."""

    factor_name: str
    current_rank_ic: float
    mean_rank_ic: float
    std_rank_ic: float
    z_score: float  # (current - mean) / std — positive = emerging
    trend_direction: str  # "emerging", "stable", "decaying"
    n_scans: int
    best_rank_ic: float
    worst_rank_ic: float


class FactorTracker:
    """Tracks factor performance across multiple scans, detects decay/emergence."""

    def __init__(self, scan_dir: str | Path = FACTOR_SCAN_DIR):
        self.scan_dir = Path(scan_dir)
        self.scan_dir.mkdir(parents=True, exist_ok=True)

    def save_scan(self, records: list[FactorScanRecord]) -> str:
        """Persist one scan's results as a dated JSON file."""
        scan_date = date.today().isoformat()
        path = self.scan_dir / f"scan_{scan_date}.json"
        data = {
            "scan_date": scan_date,
            "n_factors": len(records),
            "records": [{
                "factor_name": r.factor_name,
                "ic": r.ic,
                "rank_ic": r.rank_ic,
                "ic_ir": r.ic_ir,
                "rank_ic_ir": r.rank_ic_ir,
                "spread": r.spread,
                "n_obs": r.n_obs,
                "category": r.category,
                "time_horizon": r.time_horizon,
            } for r in records],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Factor scan saved: %s (%d factors)", path, len(records))
        return str(path)

    def load_history(self, n_days: int = 90) -> dict[str, list[FactorScanRecord]]:
        """Load last N days of scan results, grouped by factor name."""
        history: dict[str, list[FactorScanRecord]] = defaultdict(list)
        cutoff = date.today() - timedelta(days=n_days)
        for path in sorted(self.scan_dir.glob("scan_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                scan_date_str = data.get("scan_date", "")
                if scan_date_str < cutoff.isoformat():
                    continue
                for r in data.get("records", []):
                    rec = FactorScanRecord(
                        scan_date=scan_date_str,
                        factor_name=r["factor_name"],
                        ic=r.get("ic", 0),
                        rank_ic=r.get("rank_ic", 0),
                        ic_ir=r.get("ic_ir", 0),
                        rank_ic_ir=r.get("rank_ic_ir", 0),
                        spread=r.get("spread", 0),
                        n_obs=r.get("n_obs", 0),
                        category=r.get("category", ""),
                        time_horizon=r.get("time_horizon", ""),
                    )
                    history[rec.factor_name].append(rec)
            except Exception as e:
                logger.debug("Skipping scan file %s: %s", path.name, e)
        return dict(history)

    def compute_trends(
        self, history: dict[str, list[FactorScanRecord]], min_scans: int = 3
    ) -> list[FactorTrend]:
        """Compute emergence/decay trends for all factors."""
        trends = []
        for name, records in history.items():
            if len(records) < min_scans:
                continue
            rank_ics = [r.rank_ic for r in records[-10:]]  # last 10 scans
            if len(rank_ics) < min_scans:
                continue
            current = rank_ics[-1]
            mean = float(np.mean(rank_ics))
            std = float(np.std(rank_ics)) or 1e-6
            z = (current - mean) / std

            if z > 1.0 and current > 0.02:
                direction = "emerging"
            elif z < -1.0 and current < 0:
                direction = "decaying"
            else:
                direction = "stable"

            trends.append(FactorTrend(
                factor_name=name,
                current_rank_ic=current,
                mean_rank_ic=mean,
                std_rank_ic=std,
                z_score=z,
                trend_direction=direction,
                n_scans=len(records),
                best_rank_ic=max(rank_ics),
                worst_rank_ic=min(rank_ics),
            ))
        trends.sort(key=lambda t: -t.z_score if t.trend_direction == "emerging" else t.z_score if t.trend_direction == "decaying" else 0)
        return trends


class FactorScanHarness:
    """Orchestrates periodic factor scanning, tracking, and reporting.

    Integrates with the evolution loop to provide factor-level insights
    alongside strategy-level evaluations.
    """

    def __init__(self, market: str = "cn"):
        self.market = market
        self.tracker = FactorTracker()

    def run_scan(
        self,
        *,
        forward_period: int = 5,
        top_n: int = 200,
        lookback_days: int = 500,
    ) -> dict[str, Any]:
        """Run a full factor scan: load data → compute factors → evaluate → track.

        Returns dict with scan results, new signals, and decay warnings.
        """
        try:
            from factors.backtest import FactorBacktestEngine, DB_PATH

            engine = FactorBacktestEngine(
                market=self.market,
                sample_size=min(top_n, 500),
                recent_days=lookback_days,
            )
            logger.info("Factor scan: loading data (market=%s, top=%d)", self.market, top_n)
            engine.load_data()

            logger.info("Factor scan: computing all catalog factors...")
            engine.compute_all_factors()

            logger.info("Factor scan: evaluating IC (forward=%dd)...", forward_period)
            results = engine.evaluate_all(forward_period=forward_period)

            records = []
            for r in results:
                records.append(FactorScanRecord(
                    scan_date=date.today().isoformat(),
                    factor_name=r.get("name", "unknown"),
                    ic=r.get("ic", 0),
                    rank_ic=r.get("rank_ic", 0),
                    ic_ir=r.get("ic_ir", 0),
                    rank_ic_ir=r.get("rank_ic_ir", 0),
                    spread=r.get("spread", 0),
                    n_obs=r.get("n_obs", 0),
                ))

            # Save scan results
            self.tracker.save_scan(records)

            # Load history and compute trends
            history = self.tracker.load_history()
            trends = self.tracker.compute_trends(history)

            # Identify new signals and decay warnings
            new_signals = [
                {"factor_name": t.factor_name, "rank_ic": round(t.current_rank_ic, 4), "z_score": round(t.z_score, 2)}
                for t in trends if t.trend_direction == "emerging"
            ]
            decay_warnings = [
                {"factor_name": t.factor_name, "rank_ic": round(t.current_rank_ic, 4), "z_score": round(t.z_score, 2)}
                for t in trends if t.trend_direction == "decaying"
            ]

            # Top factors by current rank_ic
            top_factors = sorted(
                [r for r in records if abs(r.rank_ic) > 0.02],
                key=lambda r: -abs(r.rank_ic),
            )[:20]

            # Compute factor→strategy recommendations
            top_factor_dicts = [
                {"name": r.factor_name, "rank_ic": r.rank_ic, "ic_ir": r.rank_ic_ir, "spread": r.spread}
                for r in top_factors
            ]
            strategy_recs = self._compute_strategy_recommendations(
                history=history, trends=trends, top_factors=top_factor_dicts,
            )

            return {
                "status": "completed",
                "market": self.market,
                "scan_date": date.today().isoformat(),
                "n_factors_scanned": len(records),
                "n_factors_tracked": len(trends),
                "new_signals": new_signals,
                "decay_warnings": decay_warnings,
                "top_factors": [
                    {"name": f.factor_name, "rank_ic": f.rank_ic, "ic_ir": f.ic_ir, "spread": f.spread}
                    for f in top_factors
                ],
                "strategy_recommendations": strategy_recs,
                "insights": self._generate_insights(new_signals, decay_warnings, top_factor_dicts, strategy_recs),
            }

        except Exception as e:
            logger.error("Factor scan failed: %s", e)
            return {"status": "failed", "error": str(e)}

    def _compute_strategy_recommendations(
        self,
        history: dict[str, list[FactorScanRecord]],
        trends: list[FactorTrend],
        top_factors: list[dict],
        *,
        min_rank_ic: float = 0.03,
        min_consecutive_scans: int = 3,
    ) -> list[dict]:
        """Generate strategy recommendations from sustained strong factors.

        A factor becomes strategy-ready when:
        1. Rank IC > min_rank_ic for min_consecutive_scans consecutive scans
        2. OR it's in the top 5 by absolute rank_ic in the current scan
        """
        recs = []

        # Check sustained performers
        for name, records in history.items():
            if len(records) < min_consecutive_scans:
                continue
            recent = records[-min_consecutive_scans:]
            if all(abs(r.rank_ic) > min_rank_ic for r in recent):
                avg_ic = float(np.mean([r.rank_ic for r in recent]))
                direction = "long" if avg_ic > 0 else "short"
                # Map factor name to potential strategy
                suggested_strategy = self._map_factor_to_strategy(name)
                recs.append({
                    "factor_name": name,
                    "avg_rank_ic": round(avg_ic, 4),
                    "consecutive_strong": min_consecutive_scans,
                    "direction": direction,
                    "suggested_strategy": suggested_strategy,
                    "reason": f"连续{min_consecutive_scans}期Rank IC>{min_rank_ic}",
                    "source": "sustained_performance",
                })

        # Check current top performers
        for f in top_factors[:5]:
            name = f["name"]
            if not any(r["factor_name"] == name for r in recs):
                suggested_strategy = self._map_factor_to_strategy(name)
                recs.append({
                    "factor_name": name,
                    "avg_rank_ic": f["rank_ic"],
                    "consecutive_strong": 1,
                    "direction": "long" if f["rank_ic"] > 0 else "short",
                    "suggested_strategy": suggested_strategy,
                    "reason": f"当期Rank IC={f['rank_ic']:+.4f}，位列前5",
                    "source": "top_performer",
                })

        return recs

    @staticmethod
    def _map_factor_to_strategy(factor_name: str) -> str:
        """Map a factor name to the most suitable strategy type."""
        name_lower = factor_name.lower()
        if "momentum" in name_lower or "ret_" in name_lower or "trend" in name_lower or "ma_" in name_lower:
            return "momentum_top_k"
        if "reversal" in name_lower or "rev_" in name_lower or "mr_" in name_lower:
            return "mean_reversion_top_k"
        if "vol" in name_lower or "volatility" in name_lower or "skew" in name_lower or "std" in name_lower:
            return "risk_adj_momentum_top_k"
        if "rsi" in name_lower or "bb_" in name_lower or "macd" in name_lower:
            return "multi_factor_top_k"
        if "value" in name_lower or "stability" in name_lower or "consistency" in name_lower:
            return "jq_value_timing_top_k"
        if "small" in name_lower or "cap" in name_lower:
            return "jq_small_cap_top_k"
        return "multi_factor_top_k"

    def _generate_insights(
        self,
        new_signals: list[dict],
        decay_warnings: list[dict],
        top_factors: list[dict],
        strategy_recs: list[dict] | None = None,
    ) -> list[str]:
        """Generate human-readable insights from scan results."""
        insights = []
        if new_signals:
            names = ", ".join(s["factor_name"] for s in new_signals[:5])
            insights.append(f"发现 {len(new_signals)} 个新兴因子: {names}")
        if decay_warnings:
            names = ", ".join(s["factor_name"] for s in decay_warnings[:5])
            insights.append(f"⚠ {len(decay_warnings)} 个因子正在衰减: {names}")
        if top_factors:
            best = top_factors[0]
            insights.append(f"最强因子: {best['name']} (Rank IC={best['rank_ic']:+.4f})")
        if strategy_recs:
            rec_names = [r["factor_name"] + "→" + r["suggested_strategy"] for r in strategy_recs[:5]]
            insights.append(f"策略推荐: {', '.join(rec_names)}")
        if not insights:
            insights.append("因子扫描完成，无显著新信号或衰减")
        return insights

    def run_ai_research_cycle(
        self,
        *,
        regime: str = "UNKNOWN",
        n_hypotheses: int = 5,
        run_backtest: bool = True,
    ) -> dict[str, Any]:
        """Run AI factor research and integrate results into scan history.

        Generates new factor hypotheses, backtests them, and if any show
        strong IC, they are surfaced as new signals.
        """
        try:
            from research.ai_agent import AIResearchAgent

            agent = AIResearchAgent()
            report = agent.run_full_research_cycle(
                regime=regime,
                market=self.market,
                n_hypotheses=n_hypotheses,
                run_backtest=run_backtest,
            )
            result = {
                "status": "completed",
                "hypotheses_generated": report.hypotheses_generated,
                "promising_factors": report.promising_factors,
                "insights": report.insights,
                "recommendations": report.recommendations,
            }
            return result
        except Exception as e:
            logger.error("AI research cycle failed: %s", e)
            return {"status": "failed", "error": str(e)}

    def run_monthly_scan(
        self,
    ) -> dict[str, Any]:
        """Run comprehensive monthly scan: full factor catalog + AI research."""
        logger.info("Starting monthly factor scan for market=%s", self.market)

        # Step 1: Run the full catalog scan
        scan_result = self.run_scan()

        # Step 2: Run AI research
        from research.regime_detector import RegimeDetector, RegimeConfig
        import pandas as pd

        try:
            from services.portfolio_backtest import market_repository

            repo = market_repository(require_exists=True)
            rows = repo.conn.execute(
                "SELECT trade_date, symbol, COALESCE(adj_close, close) AS close "
                "FROM daily_bars WHERE market = ? ORDER BY trade_date LIMIT 50000",
                [self.market],
            ).fetchdf()
            rows["trade_date"] = pd.to_datetime(rows["trade_date"])
            pivot = rows.pivot_table(
                index="trade_date", columns="symbol", values="close", aggfunc="first"
            )
            detector = RegimeDetector(config=RegimeConfig())
            signals = detector.detect_all(pivot, market=self.market)
            regime = signals[-1].regime if signals else "UNKNOWN"
        except Exception:
            regime = "UNKNOWN"

        ai_result = self.run_ai_research_cycle(regime=regime)

        # Combine
        return {
            "scan_date": date.today().isoformat(),
            "market": self.market,
            "catalog_scan": {
                "n_factors": scan_result.get("n_factors_scanned", 0),
                "new_signals": scan_result.get("new_signals", []),
                "decay_warnings": scan_result.get("decay_warnings", []),
                "top_factors": scan_result.get("top_factors", []),
            },
            "ai_research": {
                "hypotheses": ai_result.get("hypotheses_generated", 0),
                "promising": ai_result.get("promising_factors", []),
            },
            "insights": (
                scan_result.get("insights", [])
                + ai_result.get("insights", [])
            ),
            "recommendations": ai_result.get("recommendations", []),
        }
