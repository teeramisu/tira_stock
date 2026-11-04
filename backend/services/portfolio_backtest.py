"""Portfolio backtest orchestration for API and scripts (no FastAPI imports)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from marketdata.db import DEFAULT_MARKET_DB_PATH, connect as connect_market_db
from marketdata.errors import MarketDataNotFound
from marketdata.repository import MarketDataRepository
from marketdata.universe import UniverseConfig, effective_include_exchanges
from marketdata.versioning import require_stable_dataset_version
from portfolio.broker import CostModel
from portfolio.engine import PortfolioEngine
from portfolio.models import PortfolioBacktestConfig
from portfolio.strategy_registry import (
    build_portfolio_strategy,
    is_etf_pool_strategy,
    missing_adapted_portfolio_builders,
    resolve_portfolio_symbols,
    resolve_rebalance_frequency,
    supported_portfolio_strategies,
)
from research.portfolio_governance import build_portfolio_governance_artifacts
from research.symbol_attribution import build_symbol_attribution_artifact

logger = logging.getLogger(__name__)

_MAX_CACHED_ARTIFACT_RUNS = 100
_PORTFOLIO_ARTIFACTS: dict[str, dict[str, object]] = {}
_PORTFOLIO_SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "backtests"


class PortfolioBacktestRequest(BaseModel):
    """全市场组合回测请求。"""

    market: str = Field("cn", description="市场：cn/us")
    dataset_version: str | None = Field(None, description="稳定数据版本；为空时使用最新 stable 版本")
    start: date
    end: date
    initial_cash: float = Field(1_000_000.0, ge=100)
    strategy: str = Field("momentum_top_k")
    rebalance_frequency: str = Field("daily")
    symbols: list[str] = Field(default_factory=list)
    max_positions: int = Field(10, ge=1, le=500)
    lookback_days: int = Field(20, ge=1, le=500)
    include_exchanges: list[str] = Field(default_factory=list)
    exclude_st: bool = True
    exclude_neeq: bool = True
    exclude_suspended: bool = True
    min_list_days: int = Field(60, ge=0)
    exclude_limit_up_buy: bool = True
    exclude_limit_down_sell: bool = True
    buy_commission_rate: float = Field(0.0003, ge=0, le=0.1)
    sell_commission_rate: float = Field(0.0003, ge=0, le=0.1)
    stamp_tax_rate: float = Field(0.0005, ge=0, le=0.1)
    transfer_fee_rate: float = Field(0.0, ge=0, le=0.1)
    min_commission: float = Field(5.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0, le=1000)
    market_impact_bps: float = Field(1.0, ge=0, le=1000)


def market_db_path() -> Path:
    """Return configured market data DB path."""
    import os

    return Path(os.environ.get("MARKET_DATA_DB_PATH", str(DEFAULT_MARKET_DB_PATH)))


def market_repository(*, require_exists: bool = True) -> MarketDataRepository:
    """Open the local market data repository."""
    db_path = market_db_path()
    if require_exists and not db_path.exists():
        raise MarketDataNotFound(f"本地行情库不存在: {db_path}")
    return MarketDataRepository(connect_market_db(db_path))


def latest_stable_dataset_version(repo: MarketDataRepository, market: str) -> str:
    """Return latest stable full-market dataset version for a market."""
    from services.market_status import latest_stable_dataset_version as _latest

    return _latest(repo, market)


def allowed_exchanges(market: str) -> tuple[str, ...]:
    """Return allowed exchanges by market."""
    if market == "us":
        return ("NYSE", "NASDAQ", "AMEX")
    if market == "cn":
        return ("SSE", "SZSE", "BSE")
    raise ValueError(f"不支持的 market: {market}")


def validate_include_exchanges(market: str, exchanges: tuple[str, ...]) -> tuple[str, ...]:
    """Validate include_exchanges against market config."""
    allowed = allowed_exchanges(market)
    if not exchanges:
        return allowed
    invalid = sorted(set(exchanges) - set(allowed))
    if invalid:
        raise ValueError(f"include_exchanges 包含不支持交易所: {','.join(invalid)}")
    return exchanges


def compute_portfolio_request_hash(payload: dict[str, Any]) -> str:
    """Return stable request hash for artifact tracking."""
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _cache_portfolio_artifacts(run_id: str, artifacts: dict[str, object]) -> None:
    """Cache portfolio artifacts for lightweight artifact pagination API."""
    _PORTFOLIO_ARTIFACTS[run_id] = artifacts
    if len(_PORTFOLIO_ARTIFACTS) <= _MAX_CACHED_ARTIFACT_RUNS:
        return
    oldest_run_id = next(iter(_PORTFOLIO_ARTIFACTS))
    _PORTFOLIO_ARTIFACTS.pop(oldest_run_id, None)


def persist_portfolio_artifacts(run_id: str, artifacts: dict[str, object]) -> None:
    """Persist backtest artifacts as snapshot json."""
    _PORTFOLIO_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = _PORTFOLIO_SNAPSHOT_DIR / f"{run_id}.json"
    snapshot_path.write_text(
        json.dumps(artifacts, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_portfolio_artifacts(run_id: str) -> dict[str, object] | None:
    """Load artifacts from memory cache or snapshot file."""
    payload = _PORTFOLIO_ARTIFACTS.get(run_id)
    if payload is not None:
        return payload
    snapshot_path = _PORTFOLIO_SNAPSHOT_DIR / f"{run_id}.json"
    if not snapshot_path.exists():
        return None
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_safe_value(value: Any) -> Any:
    """Recursively coerce NaN/Inf and numpy scalars into JSON-safe values."""
    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if pd.isna(value):
        return None
    return value


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    if df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if "date" in col or "time" in col:
            out[col] = out[col].astype(str)
    if "created_at" in out.columns:
        out["created_at"] = out["created_at"].astype(str)
    return [_json_safe_value(record) for record in out.to_dict(orient="records")]


def _latest_positions(positions_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return end-of-backtest positions."""
    if positions_df.empty:
        return []
    last_trade_date = positions_df["trade_date"].max()
    latest = positions_df[positions_df["trade_date"] == last_trade_date].copy()
    latest = latest[latest["shares"] > 0]
    if latest.empty:
        return []
    latest = latest.sort_values("market_value", ascending=False)
    return _df_records(latest)


def _latest_universe_size(filter_diagnostics_df: pd.DataFrame) -> int:
    """Return latest universe size from filter diagnostics."""
    if filter_diagnostics_df.empty or "final" not in filter_diagnostics_df.columns:
        return 0
    latest_row = filter_diagnostics_df.sort_values("trade_date").iloc[-1]
    return int(latest_row.get("final", 0))


def _validate_symbols_have_bars(
    repo: MarketDataRepository,
    market: str,
    symbols: tuple[str, ...],
    start: date,
    end: date,
) -> None:
    """Fail fast when ETF/stock symbols have no local daily bars in range."""
    if not symbols:
        return
    missing: list[str] = []
    for symbol in symbols:
        row = repo.conn.execute(
            """
            SELECT COUNT(*) FROM daily_bars
            WHERE market = ? AND symbol = ? AND trade_date BETWEEN ? AND ?
            """,
            [market, symbol, start, end],
        ).fetchone()
        if not row or int(row[0]) <= 0:
            missing.append(symbol)
    if not missing:
        return
    sample = ",".join(missing[:8])
    suffix = "..." if len(missing) > 8 else ""
    raise ValueError(
        f"本地行情库缺少标的日线数据: {sample}{suffix}。"
        f" ETF 策略请先刷新: python scripts/refresh_market_data.py "
        f"--market cn --symbols {sample} --start {start} --end {end}"
    )


def execute_portfolio_backtest(
    req: PortfolioBacktestRequest,
    *,
    on_progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Run portfolio backtest and return API-shaped payload (metrics + artifacts fields).

    Raises:
        ValueError: Invalid parameters or unsupported strategy.
        MarketDataError: Domain errors from data version or universe constraints.
        MarketDataNotFound: Missing DB file or dataset version.
    """
    if req.end <= req.start:
        raise ValueError("end 必须晚于 start")
    missing_builders = missing_adapted_portfolio_builders()
    if missing_builders:
        logger.warning("adapted portfolio strategies missing builders: %s", ",".join(missing_builders))
    if req.strategy not in supported_portfolio_strategies():
        raise ValueError(f"暂不支持组合策略: {req.strategy}")

    resolved_symbols = resolve_portfolio_symbols(req.strategy, req.symbols)
    if is_etf_pool_strategy(req.strategy) and not resolved_symbols:
        raise ValueError(
            f"策略 {req.strategy} 需要 ETF 标的池；请在「自定义标的池」填写代码，"
            "或检查策略 preset 是否配置 symbols。"
        )

    repo = market_repository(require_exists=True)
    _validate_symbols_have_bars(repo, req.market, resolved_symbols, req.start, req.end)
    dataset_version = req.dataset_version or latest_stable_dataset_version(repo, req.market)
    request_payload = req.model_dump(mode="json")
    request_payload["dataset_version"] = dataset_version
    req_hash = compute_portfolio_request_hash(request_payload)

    require_stable_dataset_version(repo, dataset_version)
    exchanges = effective_include_exchanges(
        req.market,
        validate_include_exchanges(req.market, tuple(req.include_exchanges)),
    )
    strategy = build_portfolio_strategy(req)
    rebalance_frequency = resolve_rebalance_frequency(req.strategy, req.rebalance_frequency)

    result = PortfolioEngine(
        repository=repo,
        strategy=strategy,
        config=PortfolioBacktestConfig(
            market=req.market,
            dataset_version=dataset_version,
            start=req.start,
            end=req.end,
            initial_cash=req.initial_cash,
            rebalance_frequency=rebalance_frequency,
            lot_size=1 if req.market == "us" else 100,
        ),
        universe_config=UniverseConfig(
            market=req.market,
            include_exchanges=exchanges,
            symbols=resolved_symbols,
            exclude_neeq=req.exclude_neeq,
            exclude_st=req.exclude_st,
            exclude_suspended=req.exclude_suspended,
            min_list_days=req.min_list_days,
            exclude_limit_up_buy=req.exclude_limit_up_buy,
            exclude_limit_down_sell=req.exclude_limit_down_sell,
        ),
        cost_model=CostModel(
            buy_commission_rate=req.buy_commission_rate,
            sell_commission_rate=req.sell_commission_rate,
            stamp_tax_rate=req.stamp_tax_rate,
            transfer_fee_rate=req.transfer_fee_rate,
            min_commission=req.min_commission,
            slippage_bps=req.slippage_bps,
            market_impact_bps=req.market_impact_bps,
        ),
    ).run(on_progress=on_progress)

    if on_progress:
        on_progress(
            {
                "phase": "governance",
                "pct": 0.92,
                "message": "生成治理指标与标的归因…",
            }
        )
    governance_artifacts = build_portfolio_governance_artifacts(
        equity_curve=result.equity_curve,
        trades=result.trades,
        market=req.market,
        metrics=result.metrics,
        cost_summary=result.cost_summary,
        mode="backtest",
        dataset_version=dataset_version,
        code_version=os.getenv("BACKTEST_CODE_VERSION", "dev"),
        params_hash=req_hash,
        strategy=req.strategy,
        sample_start=req.start.isoformat(),
        sample_end=req.end.isoformat(),
    )
    governance_artifacts["symbol_attribution"] = build_symbol_attribution_artifact(
        trades=result.trades,
        positions=result.positions,
        equity_curve=result.equity_curve,
        repository=repo,
        market=req.market,
        sample_start=req.start.isoformat(),
        sample_end=req.end.isoformat(),
    )
    artifacts: dict[str, object] = {
        "equity_curve": _df_records(result.equity_curve),
        "benchmark_curve": _df_records(result.benchmark_curve),
        "cost_summary": result.cost_summary,
        "final_positions": _latest_positions(result.positions),
        "trades": _df_records(result.trades),
        "rebalance_logs": _df_records(result.rebalance_logs),
        "filter_diagnostics": _df_records(result.filter_diagnostics),
        "governance": governance_artifacts,
    }
    run_metadata = {
        "request_hash": req_hash,
        "dataset_version": dataset_version,
        "strategy": req.strategy,
        "elapsed_seconds": float(result.metrics.get("elapsed_seconds", 0.0)),
        "universe_size": _latest_universe_size(result.filter_diagnostics),
        "experiment": {
            "params": {
                k: v
                for k, v in request_payload.items()
                if k not in {"dataset_version"}
            },
            "sample_start": req.start.isoformat(),
            "sample_end": req.end.isoformat(),
            "code_version": os.getenv("BACKTEST_CODE_VERSION", "dev"),
            "params_hash": req_hash,
        },
        "run_provenance": governance_artifacts["run_provenance"],
    }
    artifacts["run_metadata"] = [run_metadata]
    logger.info(
        "portfolio_backtest_completed request_hash=%s dataset_version=%s strategy=%s "
        "elapsed_seconds=%.6f universe_size=%d",
        run_metadata["request_hash"],
        run_metadata["dataset_version"],
        run_metadata["strategy"],
        run_metadata["elapsed_seconds"],
        run_metadata["universe_size"],
    )
    _cache_portfolio_artifacts(req_hash, artifacts)
    persist_portfolio_artifacts(req_hash, artifacts)

    if on_progress:
        on_progress({"phase": "done", "pct": 0.98, "message": "回测完成，整理响应…"})

    return _json_safe_value(
        {
            "request_hash": req_hash,
            "dataset_version": dataset_version,
            "strategy": req.strategy,
            "run_metadata": run_metadata,
            "metrics": result.metrics,
            "equity_curve": artifacts["equity_curve"],
            "benchmark_curve": artifacts["benchmark_curve"],
            "cost_summary": artifacts["cost_summary"],
            "final_positions": artifacts["final_positions"],
            "trades": artifacts["trades"],
            "rebalance_logs": artifacts["rebalance_logs"],
            "filter_diagnostics": artifacts["filter_diagnostics"],
            "governance_artifacts": governance_artifacts,
            "run_provenance": governance_artifacts["run_provenance"],
        }
    )


def iter_portfolio_backtest_stream(req: PortfolioBacktestRequest) -> Iterator[str]:
    """Yield NDJSON progress/result events for streaming portfolio backtests."""
    event_queue: queue.Queue[tuple[str, object]] = queue.Queue()

    def on_progress(event: dict[str, object]) -> None:
        event_queue.put(("progress", event))

    def worker() -> None:
        try:
            payload = execute_portfolio_backtest(req, on_progress=on_progress)
            event_queue.put(("result", payload))
        except Exception as exc:
            event_queue.put(("error", exc))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        kind, payload = event_queue.get()
        if kind == "progress":
            line = json.dumps({"type": "progress", **payload}, ensure_ascii=False)
            yield line + "\n"
            continue
        if kind == "error":
            detail = str(payload)
            yield json.dumps({"type": "error", "detail": detail}, ensure_ascii=False) + "\n"
            return
        if kind == "result":
            yield json.dumps({"type": "result", "data": payload}, ensure_ascii=False, default=str) + "\n"
            return
