"""Attach governance artifacts to portfolio backtest outputs."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from .governance import (
    StressScenario,
    build_regime_metrics_artifact,
    build_stress_validation_artifact,
)
from .regime_labels import load_regime_labels

RunMode = Literal["backtest", "paper", "live"]

DEFAULT_STRESS_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario(name="cost_up", cost_multiplier=1.5),
    StressScenario(name="liquidity_down", liquidity_haircut=0.15),
    StressScenario(name="volatility_up", volatility_multiplier=2.0),
    StressScenario(name="correlation_up", correlation_shock=0.5),
)


def enrich_equity_curve_for_regime(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Add daily turnover and cost columns for regime slicing."""
    if equity_curve.empty:
        return equity_curve
    frame = equity_curve.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if trades.empty:
        frame["turnover"] = 0.0
        frame["cost"] = 0.0
        return frame
    trade_frame = trades.copy()
    trade_frame["trade_date"] = pd.to_datetime(trade_frame["trade_date"])
    trade_frame["notional"] = trade_frame["filled_shares"] * trade_frame["fill_price"]
    daily = trade_frame.groupby("trade_date", as_index=False).agg(
        turnover=("notional", "sum"),
        cost=("total_cost", "sum"),
    )
    merged = frame.merge(daily, on="trade_date", how="left")
    merged["turnover"] = merged["turnover"].fillna(0.0)
    merged["cost"] = merged["cost"].fillna(0.0)
    return merged


def build_run_provenance_artifact(
    *,
    mode: RunMode,
    dataset_version: str,
    code_version: str,
    params_hash: str,
    strategy: str,
    sample_start: str,
    sample_end: str,
) -> dict[str, Any]:
    """Bind run outputs to data/code/parameter versions (backtest and paper/live)."""
    return {
        "artifact_type": "run_provenance_v1",
        "mode": mode,
        "dataset_version": dataset_version,
        "code_version": code_version,
        "params_hash": params_hash,
        "strategy": strategy,
        "sample_start": sample_start,
        "sample_end": sample_end,
    }


def build_portfolio_governance_artifacts(
    *,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    market: str,
    metrics: dict[str, float],
    cost_summary: dict[str, float],
    mode: RunMode = "backtest",
    dataset_version: str,
    code_version: str,
    params_hash: str,
    strategy: str,
    sample_start: str,
    sample_end: str,
    stress_scenarios: tuple[StressScenario, ...] = DEFAULT_STRESS_SCENARIOS,
) -> dict[str, Any]:
    """Build regime, stress, and provenance artifacts for one portfolio run."""
    enriched = enrich_equity_curve_for_regime(equity_curve, trades)
    regimes = load_regime_labels().regimes if market == "cn" else ()
    regime_metrics = build_regime_metrics_artifact(enriched, regimes, market=market)
    stress_validation = build_stress_validation_artifact(
        base_return=float(metrics.get("total_return", 0.0)),
        base_cost=float(cost_summary.get("total_cost", metrics.get("total_cost", 0.0))),
        scenarios=stress_scenarios,
    )
    run_provenance = build_run_provenance_artifact(
        mode=mode,
        dataset_version=dataset_version,
        code_version=code_version,
        params_hash=params_hash,
        strategy=strategy,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    return {
        "regime_metrics": regime_metrics,
        "stress_validation": stress_validation,
        "run_provenance": run_provenance,
    }
