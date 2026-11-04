"""Multi-strategy portfolio backtest orchestration."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from datetime import date
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field

from research.governance import PortfolioRiskBudget
from research.multi_strategy import NAIVE_SUM_METHOD, combine_strategy_sleeve_metrics
from services.portfolio_backtest import (
    PortfolioBacktestRequest,
    _cache_portfolio_artifacts,
    execute_portfolio_backtest,
    persist_portfolio_artifacts,
)


class StrategySleeveSpec(BaseModel):
    """One strategy sleeve in a multi-strategy portfolio."""

    sleeve_id: str = Field(..., min_length=1, max_length=64)
    strategy: str = "momentum_top_k"
    lookback_days: int = Field(20, ge=1, le=500)
    max_positions: int = Field(10, ge=1, le=500)


class MultiStrategyPortfolioRequest(BaseModel):
    """Run multiple strategy sleeves and allocate capital via Allocator."""

    market: str = "cn"
    dataset_version: str | None = None
    start: date
    end: date
    initial_cash: float = Field(1_000_000.0, ge=100)
    sleeves: list[StrategySleeveSpec] = Field(..., min_length=2, max_length=10)
    rebalance_frequency: str = "daily"
    symbols: list[str] = Field(default_factory=list)
    exclude_st: bool = True
    exclude_neeq: bool = True
    exclude_suspended: bool = True
    min_list_days: int = Field(60, ge=0)
    max_single_weight: float = Field(0.6, gt=0, le=1.0)
    max_gross_exposure: float = Field(1.0, gt=0, le=2.0)
    allocator_method: str = "equal_weight"
    min_commission: float = Field(5.0, ge=0)


def _multi_request_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def execute_multi_strategy_portfolio(
    req: MultiStrategyPortfolioRequest,
    *,
    on_progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Run each sleeve backtest, then allocate capital (no naive weight sum)."""
    if req.allocator_method == NAIVE_SUM_METHOD:
        raise ValueError("multi-strategy portfolio must use allocator; naive sum is forbidden")
    if req.end <= req.start:
        raise ValueError("end 必须晚于 start")

    sleeve_results: list[dict[str, Any]] = []
    sleeve_metrics: dict[str, dict[str, float]] = {}
    dataset_version: str | None = None
    sleeve_count = len(req.sleeves)

    for sleeve_index, sleeve in enumerate(req.sleeves):
        if on_progress:
            on_progress(
                {
                    "phase": "sleeve",
                    "pct": sleeve_index / max(sleeve_count, 1),
                    "message": f"回测策略 sleeve {sleeve_index + 1}/{sleeve_count}（{sleeve.sleeve_id}）",
                    "current": sleeve_index + 1,
                    "total": sleeve_count,
                }
            )

        def _sleeve_progress(
            event: dict[str, object],
            *,
            _index: int = sleeve_index,
            _sleeve_id: str = sleeve.sleeve_id,
        ) -> None:
            if on_progress is None:
                return
            span = 1.0 / max(sleeve_count, 1)
            base = _index * span
            inner_pct = float(event.get("pct", 0.0))
            on_progress(
                {
                    **event,
                    "phase": str(event.get("phase", "sleeve")),
                    "pct": round(min(0.98, base + inner_pct * span), 4),
                    "message": f"[{_sleeve_id}] {event.get('message', '')}",
                    "current": event.get("current"),
                    "total": event.get("total"),
                }
            )

        # Allocate capital evenly across sleeves (initial equal split)
        sleeve_cash = req.initial_cash / max(sleeve_count, 1)
        sub_req = PortfolioBacktestRequest(
            market=req.market,
            dataset_version=req.dataset_version,
            start=req.start,
            end=req.end,
            initial_cash=sleeve_cash,
            strategy=sleeve.strategy,
            rebalance_frequency=req.rebalance_frequency,
            symbols=req.symbols,
            max_positions=sleeve.max_positions,
            lookback_days=sleeve.lookback_days,
            exclude_st=req.exclude_st,
            exclude_neeq=req.exclude_neeq,
            exclude_suspended=req.exclude_suspended,
            min_list_days=req.min_list_days,
            min_commission=req.min_commission,
        )
        result = execute_portfolio_backtest(sub_req, on_progress=_sleeve_progress if on_progress else None)
        dataset_version = dataset_version or str(result["dataset_version"])
        metrics = {k: float(v) for k, v in result["metrics"].items() if isinstance(v, (int, float))}
        sleeve_metrics[sleeve.sleeve_id] = metrics
        sleeve_results.append(
            {
                "sleeve_id": sleeve.sleeve_id,
                "strategy": sleeve.strategy,
                "request_hash": result["request_hash"],
                "metrics": result["metrics"],
            }
        )

    budget = PortfolioRiskBudget(
        max_single_weight=req.max_single_weight,
        max_gross_exposure=req.max_gross_exposure,
    )
    allocation_artifact = combine_strategy_sleeve_metrics(
        sleeve_metrics,
        budget=budget,
        method=req.allocator_method,
        regime="UNKNOWN",
        regime_by_sleeve=None,
    )
    if on_progress:
        on_progress({"phase": "allocate", "pct": 0.99, "message": "分配多策略权重…"})

    payload = req.model_dump(mode="json")
    payload["sleeve_metrics"] = sleeve_metrics
    run_id = _multi_request_hash(payload)

    artifacts: dict[str, object] = {
        "allocation": allocation_artifact,
        "sleeves": sleeve_results,
        "run_metadata": [
            {
                "request_hash": run_id,
                "dataset_version": dataset_version,
                "allocator_method": req.allocator_method,
                "sleeve_count": len(req.sleeves),
            }
        ],
    }
    _cache_portfolio_artifacts(run_id, artifacts)
    persist_portfolio_artifacts(run_id, artifacts)

    return {
        "request_hash": run_id,
        "dataset_version": dataset_version,
        "market": req.market,
        "allocation": allocation_artifact,
        "sleeves": sleeve_results,
        "allocator_method": req.allocator_method,
    }


def iter_multi_strategy_portfolio_stream(req: MultiStrategyPortfolioRequest) -> Iterator[str]:
    """Yield NDJSON progress/result events for streaming multi-strategy backtests."""
    event_queue: queue.Queue[tuple[str, object]] = queue.Queue()

    def on_progress(event: dict[str, object]) -> None:
        event_queue.put(("progress", event))

    def worker() -> None:
        try:
            payload = execute_multi_strategy_portfolio(req, on_progress=on_progress)
            event_queue.put(("result", payload))
        except Exception as exc:
            event_queue.put(("error", exc))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        kind, payload = event_queue.get()
        if kind == "progress":
            yield json.dumps({"type": "progress", **payload}, ensure_ascii=False) + "\n"
            continue
        if kind == "error":
            yield json.dumps({"type": "error", "detail": str(payload)}, ensure_ascii=False) + "\n"
            return
        if kind == "result":
            yield json.dumps({"type": "result", "data": payload}, ensure_ascii=False, default=str) + "\n"
            return
