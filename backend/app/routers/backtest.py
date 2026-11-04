"""
backtest 相关 API：回测执行、历史记录、策略列表。
"""
from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# 保证可导入 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.data import fetch_ohlcv
from backtest.engine import run_backtest, BacktestResult, STRATEGIES
from backtest.custom_strategy import run_custom_backtest
from backtest.imported_strategies import get_adapted_strategy, list_adapted_strategies
from database import get_db
from database.models import BacktestRun
from marketdata.db import DEFAULT_MARKET_DB_PATH, connect as connect_market_db
from marketdata.repository import MarketDataRepository
from marketdata.errors import MarketDataError, MarketDataNotFound
from services.portfolio_backtest import (
    PortfolioBacktestRequest,
    execute_portfolio_backtest,
    iter_portfolio_backtest_stream,
    load_portfolio_artifacts,
)
from services.multi_strategy_portfolio import (
    MultiStrategyPortfolioRequest,
    execute_multi_strategy_portfolio,
    iter_multi_strategy_portfolio_stream,
)
from marketdata.errors import MarketDataError, MarketDataNotFound

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ---------- 请求/响应模型 ----------
class TradeOut(BaseModel):
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    is_open: bool = False  # True = 未平仓持仓


class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="股票代码：美股 AAPL；A股 600519/000001；港股 0700/9988")
    market: str = Field("us", description="市场：us=美股, cn=A股, hk=港股")
    start: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    end: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    period: Optional[str] = Field("1y", description="若未指定 start/end 则用 period，支持 6mo,1y,2y,5y,10y，最大 10 年")
    initial_cash: float = Field(100_000.0, ge=100, description="初始资金")
    strategy: str = Field("sma_crossover", description="策略: sma_crossover, ema_crossover, rsi, macd, bollinger, buy_hold")
    commission_rate: float = Field(0.0, ge=0, le=0.01, description="手续费率（按成交金额）")
    # 策略参数（与 strategy 对应）
    fast_period: Optional[int] = Field(10, ge=2, le=120)
    slow_period: Optional[int] = Field(30, ge=2, le=250)
    period_rsi: Optional[int] = Field(14, ge=5, le=60, description="RSI 周期")
    oversold: Optional[float] = Field(30, ge=5, le=50)
    overbought: Optional[float] = Field(70, ge=50, le=95)
    macd_fast: Optional[int] = Field(12, ge=5, le=50)
    macd_slow: Optional[int] = Field(26, ge=10, le=100)
    macd_signal: Optional[int] = Field(9, ge=5, le=30)
    bollinger_period: Optional[int] = Field(20, ge=5, le=60)
    num_std: Optional[float] = Field(2.0, ge=1.0, le=3.0)
    # JoinQuant 已适配策略参数
    ma1: Optional[int] = Field(3, ge=1, le=250)
    ma2: Optional[int] = Field(6, ge=1, le=250)
    ma3: Optional[int] = Field(12, ge=1, le=250)
    ma4: Optional[int] = Field(24, ge=1, le=250)
    min_daily_return: Optional[float] = Field(0.0, ge=-0.2, le=0.2)
    rsrs_n: Optional[int] = Field(18, ge=2, le=120)
    rsrs_m: Optional[int] = Field(120, ge=10, le=800)
    rsrs_threshold: Optional[float] = Field(0.7, ge=0.1, le=5.0)
    rsrs_ma_period: Optional[int] = Field(30, ge=2, le=250)
    gap_rsi_period: Optional[int] = Field(6, ge=2, le=60)
    gap_rsi_buy: Optional[float] = Field(40.0, ge=1, le=80)
    gap_rsi_sell: Optional[float] = Field(55.0, ge=20, le=99)
    gap_rsi_ma_period: Optional[int] = Field(12, ge=2, le=250)
    weak_body_threshold: Optional[float] = Field(0.8, ge=0.0, le=2.0)
    strong_body_threshold: Optional[float] = Field(0.6, ge=0.0, le=2.0)
    stop_loss: Optional[float] = Field(0.10, ge=0.0, le=1.0)
    take_profit: Optional[float] = Field(0.10, ge=0.0, le=2.0)


class CustomBacktestRequest(BaseModel):
    """自定义 Python 策略回测：提交 code，需定义 signal(df) 返回 0/1 持仓序列。"""
    symbol: str = Field(..., description="股票代码：美股 AAPL；A股 600519；港股 0700")
    market: str = Field("us", description="市场：us=美股, cn=A股, hk=港股")
    start: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    end: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    period: Optional[str] = Field("1y", description="未指定 start/end 时使用，如 1y, 10y")
    initial_cash: float = Field(100_000.0, ge=100)
    commission_rate: float = Field(0.0, ge=0, le=0.01)
    code: str = Field(..., min_length=10, max_length=50_000, description="Python 代码，须定义 signal(df) 返回与 df 同长的 0/1 序列（1=持多 0=空仓）")
    timeout_seconds: float = Field(15.0, ge=1, le=60, description="执行超时秒数")


class BacktestResponse(BaseModel):
    strategy_used: str = ""
    symbol: str
    start_date: str
    end_date: str
    initial_value: float
    final_value: float
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    n_trades: int
    win_rate: float
    equity_curve: List[dict]
    trades: List[TradeOut]
    open_position: Optional[dict] = None


class StrategyCompareItem(BaseModel):
    """单策略对比结果（精简版，不含 equity_curve/trades）"""
    strategy: str
    strategy_name: str = ""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    n_trades: int
    win_rate: float
    final_value: float


class StrategyCompareResponse(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    results: List[StrategyCompareItem]


class SaveBacktestRequest(BaseModel):
    """保存回测到数据库：回测结果 + 可选 user_id / anonymous_id。"""
    strategy_used: str = ""
    symbol: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_value: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    n_trades: int = 0
    win_rate: float = 0.0
    equity_curve: List[dict] = Field(default_factory=list)
    trades: List[TradeOut] = Field(default_factory=list)
    user_id: Optional[int] = None
    anonymous_id: Optional[str] = None
    params: Optional[dict] = None
    market: str = "us"


# ---------- 共享辅助函数（marketdata 路由也使用）----------
def _market_db_path():
    """Return configured market data DB path."""
    return Path(os.environ.get("MARKET_DATA_DB_PATH", str(DEFAULT_MARKET_DB_PATH)))


def _market_repository(*, require_exists: bool = True) -> MarketDataRepository:
    """Open the local market data repository."""
    db_path = _market_db_path()
    if require_exists and not db_path.exists():
        raise HTTPException(status_code=503, detail=f"本地行情库不存在: {db_path}")
    return MarketDataRepository(connect_market_db(db_path))


def _df_records(df) -> list[dict]:
    """Convert a DataFrame to JSON-safe records."""
    if df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if "date" in col or "time" in col:
            out[col] = out[col].astype(str)
    if "created_at" in out.columns:
        out["created_at"] = out["created_at"].astype(str)
    return out.to_dict(orient="records")


# ---------- 辅助函数 ----------
def _backtest_params(req: BacktestRequest) -> dict:
    """从请求中提取当前策略所需参数。"""
    s = (req.strategy or "sma_crossover").strip().lower()
    if s in ("sma_crossover", "ema_crossover"):
        if req.slow_period and req.fast_period and req.slow_period <= req.fast_period:
            raise HTTPException(status_code=400, detail="slow_period 必须大于 fast_period")
        return {"fast_period": req.fast_period or 10, "slow_period": req.slow_period or 30}
    if s == "rsi":
        return {"period": req.period_rsi or 14, "oversold": req.oversold or 30, "overbought": req.overbought or 70}
    if s == "macd":
        return {"fast": req.macd_fast or 12, "slow": req.macd_slow or 26, "signal_period": req.macd_signal or 9}
    if s == "bollinger":
        return {"period": req.bollinger_period or 20, "num_std": req.num_std or 2.0}
    if s == "buy_hold":
        return {}
    if get_adapted_strategy(s):
        if s == "jq_bbi_momentum_timing":
            periods = [req.ma1 or 3, req.ma2 or 6, req.ma3 or 12, req.ma4 or 24]
            if periods != sorted(periods):
                raise HTTPException(status_code=400, detail="BBI 均线周期应从短到长")
            return {
                "ma1": req.ma1 or 3,
                "ma2": req.ma2 or 6,
                "ma3": req.ma3 or 12,
                "ma4": req.ma4 or 24,
                "min_daily_return": req.min_daily_return or 0.0,
            }
        if s == "jq_rsrs_timing":
            return {
                "n": req.rsrs_n or 18,
                "m": req.rsrs_m or 120,
                "threshold": req.rsrs_threshold or 0.7,
                "ma_period": req.rsrs_ma_period or 30,
            }
        if s == "jq_rsi_gap_reversal":
            if (req.gap_rsi_sell or 55.0) <= (req.gap_rsi_buy or 40.0):
                raise HTTPException(status_code=400, detail="RSI 卖出阈值必须大于买入阈值")
            return {
                "rsi_period": req.gap_rsi_period or 6,
                "buy_threshold": req.gap_rsi_buy or 40.0,
                "sell_threshold": req.gap_rsi_sell or 55.0,
                "ma_period": req.gap_rsi_ma_period or 12,
                "weak_body_threshold": req.weak_body_threshold or 0.8,
                "strong_body_threshold": req.strong_body_threshold or 0.6,
                "stop_loss": req.stop_loss or 0.10,
                "take_profit": req.take_profit or 0.10,
            }
        if s == "jq_value_ma_rsi_timing":
            return {"ma_period": req.slow_period or 250}
        if s == "jq_etf_ma_cross_timing":
            return {
                "ma_long": req.slow_period or 60,
                "ma_short": req.fast_period or 5,
                "ma_sell_fast": req.rsrs_n or 20,
                "ma_sell_slow": req.rsrs_m or 10,
            }
        if s == "jq_etf_value_averaging_timing":
            return {}
    return {"fast_period": req.fast_period or 10, "slow_period": req.slow_period or 30}


def _build_backtest_response(df, res: BacktestResult, strategy_used: str, symbol: str) -> BacktestResponse:
    equity_curve = [{"date": str(d)[:10], "value": round(float(v), 2)} for d, v in res.equity_curve.items()]
    return BacktestResponse(
        strategy_used=strategy_used,
        symbol=symbol,
        start_date=str(df.index.min())[:10],
        end_date=str(df.index.max())[:10],
        initial_value=res.initial_value,
        final_value=res.final_value,
        total_return=res.total_return,
        annual_return=res.annual_return,
        volatility=res.volatility,
        sharpe_ratio=res.sharpe_ratio,
        max_drawdown=res.max_drawdown,
        max_drawdown_duration=res.max_drawdown_duration,
        n_trades=res.n_trades,
        win_rate=res.win_rate,
        equity_curve=equity_curve,
        trades=[
            TradeOut(
                entry_date=t.entry_date,
                entry_price=round(t.entry_price, 4),
                exit_date=t.exit_date or "",
                exit_price=round(t.exit_price, 4) if t.exit_price and not (isinstance(t.exit_price, float) and np.isnan(t.exit_price)) else 0,
                shares=round(t.shares, 4),
                pnl=round(t.pnl, 2) if not (isinstance(t.pnl, float) and np.isnan(t.pnl)) else 0,
                pnl_pct=round(t.pnl_pct, 2) if not (isinstance(t.pnl_pct, float) and np.isnan(t.pnl_pct)) else 0,
                is_open=t.is_open,
            )
            for t in res.trades
        ],
        open_position=res.open_position,
    )


def _result_summary_from_response(r: SaveBacktestRequest) -> dict:
    return {
        "strategy_used": r.strategy_used,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "initial_value": r.initial_value,
        "final_value": r.final_value,
        "total_return": r.total_return,
        "annual_return": r.annual_return,
        "volatility": r.volatility,
        "sharpe_ratio": r.sharpe_ratio,
        "max_drawdown": r.max_drawdown,
        "max_drawdown_duration": r.max_drawdown_duration,
        "n_trades": r.n_trades,
        "win_rate": r.win_rate,
        "equity_curve": r.equity_curve,
        "trades": [t.model_dump() for t in r.trades],
    }


# ---------- 接口 ----------
@router.get("/strategies")
def list_strategies():
    """返回可用策略列表及参数说明，供前端渲染表单。"""
    return {"strategies": STRATEGIES}


@router.get("/adapted-strategies")
def list_imported_adapted_strategies():
    """返回已改写为本项目 signal(df) 形式的 JoinQuant 策略。"""
    return {"strategies": list_adapted_strategies()}


@router.get("/history")
def get_history(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "1y",
    market: Optional[str] = "us",
):
    """获取历史 OHLCV（支持美股/A股/港股）。"""
    try:
        df = fetch_ohlcv(symbol, start=start, end=end, period=period, market=market or "us")
        df = df.reset_index()
        df["Date"] = df.iloc[:, 0].astype(str)
        cols = ["Date"]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                cols.append(c)
        out = df[cols].to_dict(orient="records")
        return {"symbol": symbol, "data": out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest_endpoint(req: BacktestRequest):
    """执行回测，支持多策略、最长 10 年区间。"""
    period = (req.period or "1y").strip().lower()
    if req.start and req.end:
        pass
    else:
        allowed = ("6mo", "1y", "2y", "5y", "10y")
        if period not in allowed:
            period = "1y"
        req.period = period
    try:
        df = fetch_ohlcv(req.symbol, start=req.start, end=req.end, period=req.period, market=req.market or "us")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取行情失败: {e}")

    try:
        params = _backtest_params(req)
    except HTTPException:
        raise

    strategy_used = (req.strategy or "sma_crossover").strip().lower()
    try:
        res: BacktestResult = run_backtest(
            df,
            strategy=strategy_used,
            initial_cash=req.initial_cash,
            commission_rate=req.commission_rate,
            **params,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"回测计算失败: {e}")

    return _build_backtest_response(df, res, strategy_used, req.symbol)


@router.post("/backtest/all", response_model=StrategyCompareResponse)
def run_all_strategies_endpoint(req: BacktestRequest):
    """用所有内置策略跑回测，按夏普比率降序返回对比结果。"""
    period = (req.period or "1y").strip().lower()
    if req.start and req.end:
        pass
    else:
        allowed = ("6mo", "1y", "2y", "5y", "10y")
        if period not in allowed:
            period = "1y"
        req.period = period
    try:
        df = fetch_ohlcv(req.symbol, start=req.start, end=req.end, period=req.period, market=req.market or "us")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取行情失败: {e}")

    # 内置策略列表（不含 JoinQuant 适配策略）
    builtin_strategies = [
        {"id": "sma_crossover", "name": "双均线(SMA)"},
        {"id": "ema_crossover", "name": "双均线(EMA)"},
        {"id": "rsi", "name": "RSI"},
        {"id": "macd", "name": "MACD"},
        {"id": "bollinger", "name": "布林带"},
        {"id": "buy_hold", "name": "买入持有"},
    ]

    results = []
    for s in builtin_strategies:
        try:
            params = _backtest_params(req)
            # 过滤出当前策略需要的参数
            from backtest.engine import STRATEGIES as ALL_STRATEGIES
            strategy_def = next((sd for sd in ALL_STRATEGIES if sd["id"] == s["id"]), None)
            if strategy_def:
                allowed_params = {k: v for k, v in params.items() if k in strategy_def.get("params", [])}
            else:
                allowed_params = params

            res: BacktestResult = run_backtest(
                df,
                strategy=s["id"],
                initial_cash=req.initial_cash,
                commission_rate=req.commission_rate,
                **allowed_params,
            )
            results.append(StrategyCompareItem(
                strategy=s["id"],
                strategy_name=s["name"],
                total_return=round(res.total_return, 4),
                annual_return=round(res.annual_return, 4),
                sharpe_ratio=round(res.sharpe_ratio, 2),
                max_drawdown=round(res.max_drawdown, 4),
                volatility=round(res.volatility, 4),
                n_trades=res.n_trades,
                win_rate=round(res.win_rate, 1),
                final_value=round(res.final_value, 2),
            ))
        except Exception:
            continue

    # 按夏普比率降序排列
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

    return StrategyCompareResponse(
        symbol=req.symbol,
        start_date=str(df.index.min())[:10],
        end_date=str(df.index.max())[:10],
        results=results,
    )


@router.post("/backtest/custom", response_model=BacktestResponse)
def run_custom_backtest_endpoint(req: CustomBacktestRequest):
    """提交自定义 Python 代码进行回测。"""
    period = (req.period or "1y").strip().lower()
    if not (req.start and req.end):
        allowed = ("6mo", "1y", "2y", "5y", "10y")
        period = period if period in allowed else "1y"
    try:
        df = fetch_ohlcv(req.symbol, start=req.start, end=req.end, period=period, market=req.market or "us")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取行情失败: {e}")
    try:
        res = run_custom_backtest(
            df,
            code=req.code,
            initial_cash=req.initial_cash,
            commission_rate=req.commission_rate,
            timeout_seconds=req.timeout_seconds,
        )
    except TimeoutError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"策略执行失败: {e}")
    return _build_backtest_response(df, res, "custom_python", req.symbol)


@router.post("/backtest/save")
async def save_backtest(req: SaveBacktestRequest, db: Optional[AsyncSession] = Depends(get_db)):
    """将本次回测结果保存到数据库；需配置 DATABASE_URL。"""
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="数据库未配置，无法保存。请设置环境变量 DATABASE_URL（PostgreSQL）。",
        )
    if not req.symbol or not req.strategy_used:
        raise HTTPException(status_code=400, detail="symbol 与 strategy_used 必填")
    row = BacktestRun(
        user_id=req.user_id,
        anonymous_id=req.anonymous_id,
        symbol=req.symbol,
        market=req.market or "us",
        strategy=req.strategy_used,
        params=req.params,
        result_summary=_result_summary_from_response(req),
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return {"saved": True, "id": row.id}


@router.get("/history/records")
async def list_backtest_records(
    user_id: Optional[int] = None,
    anonymous_id: Optional[str] = None,
    limit: int = 50,
    db: Optional[AsyncSession] = Depends(get_db),
):
    """分页拉取已保存的回测记录；按 user_id 或 anonymous_id 筛选。"""
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="数据库未配置。请设置环境变量 DATABASE_URL（PostgreSQL）。",
        )
    if not user_id and not anonymous_id:
        raise HTTPException(status_code=400, detail="请提供 user_id 或 anonymous_id 至少一个")
    if limit < 1 or limit > 200:
        limit = 50
    if user_id is not None:
        q = select(BacktestRun).where(BacktestRun.user_id == user_id)
    else:
        q = select(BacktestRun).where(BacktestRun.anonymous_id == anonymous_id)
    q = q.order_by(BacktestRun.created_at.desc()).limit(limit)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {
        "records": [
            {
                "id": r.id,
                "symbol": r.symbol,
                "market": r.market,
                "strategy": r.strategy,
                "params": r.params,
                "result_summary": r.result_summary,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


@router.post("/backtest/portfolio")
def run_portfolio_backtest(req: PortfolioBacktestRequest):
    """执行本地行情库驱动的组合回测。"""
    try:
        return execute_portfolio_backtest(req)
    except MarketDataNotFound as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (MarketDataError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/backtest/portfolio/stream")
def run_portfolio_backtest_stream(req: PortfolioBacktestRequest):
    """组合回测（NDJSON 流式进度 + 最终结果）。"""
    try:
        return StreamingResponse(
            iter_portfolio_backtest_stream(req),
            media_type="application/x-ndjson",
        )
    except MarketDataNotFound as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (MarketDataError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/backtest/portfolio/multi")
def run_multi_strategy_portfolio(req: MultiStrategyPortfolioRequest):
    """多策略组合回测：各 sleeve 独立回测后由 Allocator 分配资本。"""
    try:
        return execute_multi_strategy_portfolio(req)
    except MarketDataNotFound as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (MarketDataError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/backtest/portfolio/multi/stream")
def run_multi_strategy_portfolio_stream(req: MultiStrategyPortfolioRequest):
    """多策略组合回测（NDJSON 流式进度 + 最终结果）。"""
    try:
        return StreamingResponse(
            iter_multi_strategy_portfolio_stream(req),
            media_type="application/x-ndjson",
        )
    except MarketDataNotFound as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (MarketDataError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/backtest/{run_id}/artifacts")
def get_backtest_artifacts(run_id: str, artifact: str = "trades", limit: int = 100, offset: int = 0):
    """Read paginated backtest artifacts by run id."""
    payload = load_portfolio_artifacts(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
    if artifact not in payload:
        raise HTTPException(status_code=400, detail=f"unsupported artifact: {artifact}")
    rows = payload[artifact]
    if not isinstance(rows, list):
        rows = []
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    page = rows[offset : offset + limit]
    return {
        "run_id": run_id,
        "artifact": artifact,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "items": page,
    }


# ---------- Regime 检测 API ----------
class RegimeDetectRequest(BaseModel):
    """Regime detection request."""
    market: str = Field("cn")
    dataset_version: str | None = Field(None)
    end_date: str | None = Field(None, description="检测截止日期，为空则用最新数据")
    lookback_days: int = Field(120, ge=20, le=500, description="回溯天数")


@router.post("/regime/detect")
def detect_regime(req: RegimeDetectRequest):
    """检测当前市场状态（Regime），返回 regime 分类和策略推荐。"""
    from services.portfolio_backtest import market_repository, latest_stable_dataset_version
    from marketdata.data_portal import DataPortal
    from research.regime_detector import RegimeDetector, RegimeConfig, REGIME_STRATEGY_AFFINITY
    from portfolio.strategy_registry import strategies_for_regime, STRATEGY_REGIME_AFFINITY
    from research.regime_detector import regime_recommendation
    from datetime import date as _date

    try:
        repo = market_repository(require_exists=True)
    except MarketDataNotFound as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    dataset_version = req.dataset_version or latest_stable_dataset_version(repo, req.market)
    end_date = _date.fromisoformat(req.end_date) if req.end_date else _date.today()

    # Build a close matrix for the lookback period
    rows = repo.conn.execute(
        """
        SELECT trade_date, symbol, COALESCE(adj_close, close) AS close
        FROM daily_bars
        WHERE market = ?
          AND trade_date <= ?
        ORDER BY trade_date, symbol
        """,
        [req.market, end_date],
    ).fetchdf()

    if rows.empty:
        raise HTTPException(status_code=404, detail="无行情数据")

    # Pivot to close matrix
    rows["trade_date"] = rows["trade_date"].astype(str)
    pivot = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")
    pivot = pivot.sort_index().tail(req.lookback_days)

    if pivot.shape[0] < 70:  # Need at least 70 days for MA60 + some buffer
        raise HTTPException(status_code=400, detail=f"数据不足，仅{pivot.shape[0]}天，需≥70天")

    # Convert index to date
    pivot.index = [d if isinstance(d, _date) else _date.fromisoformat(str(d)) for d in pivot.index]

    detector = RegimeDetector(config=RegimeConfig())
    signals = detector.detect_all(pivot, market=req.market)
    summary = detector.regime_summary(signals)

    current_regime = summary.get("latest", {}).get("regime", "UNKNOWN")
    recommended = strategies_for_regime(current_regime)
    recommendation = regime_recommendation(current_regime)

    # Get strategy affinity details for recommended strategies
    strategy_details = []
    for sid in recommended[:5]:
        meta = STRATEGY_REGIME_AFFINITY.get(sid, {})
        strategy_details.append({
            "id": sid,
            "risk": meta.get("risk", "unknown"),
            "type": meta.get("type", "unknown"),
            "best_regimes": meta.get("best_regimes", []),
        })

    return {
        "regime": summary,
        "recommendation": recommendation,
        "recommended_strategies": strategy_details,
        "dataset_version": dataset_version,
        "end_date": end_date.isoformat(),
        "data_points": pivot.shape[0],
        "symbols_count": pivot.shape[1],
    }


# ---------- ML Prediction API ----------
class PredictRequest(BaseModel):
    """ML model prediction request."""
    market: str = Field("cn")
    dataset_version: str | None = Field(None)
    end_date: str | None = Field(None)
    model_name: str = Field("market_transformer_cn_latest")


@router.post("/predict/market")
def predict_market(req: PredictRequest):
    """Use trained Transformer model to predict future market regime and returns."""
    from pathlib import Path as _Path
    from datetime import date as _date

    try:
        import torch
    except ImportError:
        raise HTTPException(status_code=503, detail="PyTorch not installed")

    model_dir = _Path(__file__).resolve().parents[3] / "data" / "models"
    model_path = model_dir / f"{req.model_name}.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {req.model_name}. Run training first.")

    from ml.trainer import load_trained_model
    from ml.features import build_feature_matrix, create_sequences, FeatureConfig, REGIME_LABELS
    from ml.model import predict_with_model
    from services.portfolio_backtest import market_repository

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_trained_model(model_path, device=device)

    repo = market_repository(require_exists=True)
    end_date = _date.fromisoformat(req.end_date) if req.end_date else _date.today()

    # Load data and build features
    import pandas as pd
    rows = repo.conn.execute(
        """
        SELECT trade_date, symbol, COALESCE(adj_close, close) AS close, volume
        FROM daily_bars WHERE market = ? AND trade_date <= ?
        ORDER BY trade_date, symbol
        """,
        [req.market, end_date],
    ).fetchdf()

    if rows.empty:
        raise HTTPException(status_code=404, detail="No market data")

    rows["trade_date"] = pd.to_datetime(rows["trade_date"])
    close_matrix = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")
    volume_matrix = rows.pivot_table(index="trade_date", columns="symbol", values="volume", aggfunc="first")
    close_matrix.index = [d.date() if hasattr(d, "date") else d for d in close_matrix.index]
    volume_matrix.index = [d.date() if hasattr(d, "date") else d for d in volume_matrix.index]

    # Compute regime labels for feature building
    from research.regime_detector import RegimeDetector, RegimeConfig
    detector = RegimeDetector(config=RegimeConfig())
    signals = detector.detect_all(close_matrix)
    regime_by_date = {s.trade_date: s.regime for s in signals}

    lookback = metadata["config"]["lookback"]
    feat_config = FeatureConfig()
    fm = build_feature_matrix(close_matrix, volume_matrix, regime_by_date, config=feat_config)

    # Normalize using training stats
    feat_mean = metadata["feat_mean"]
    feat_std = metadata["feat_std"]
    normalized = (fm.features - feat_mean) / np.where(feat_std < 1e-8, 1.0, feat_std)
    normalized = np.nan_to_num(normalized, nan=0.0).astype(np.float32)

    # Get latest sequence
    if len(normalized) < lookback:
        raise HTTPException(status_code=400, detail=f"Not enough data: {len(normalized)} < {lookback}")

    latest_seq = normalized[-lookback:]  # (lookback, n_features)
    prediction = predict_with_model(model, latest_seq, device=device)

    # Add training metadata
    prediction["model_info"] = {
        "name": req.model_name,
        "metrics": metadata.get("metrics", {}),
        "lookback": lookback,
        "device": device,
    }
    prediction["prediction_date"] = end_date.isoformat()

    return prediction


@router.post("/predict/train")
def train_model(req: PredictRequest):
    """Trigger model training (async-style, returns when complete)."""
    from ml.trainer import train_market_transformer, TrainConfig
    from services.portfolio_backtest import market_repository

    repo = market_repository(require_exists=True)
    config = TrainConfig(
        lookback=30,
        forward_horizon=5,
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
    result = train_market_transformer(repo, market=req.market, config=config)
    return {
        "status": "complete",
        "model_path": result.model_path,
        "metrics": {
            "best_val_loss": result.best_val_loss,
            "regime_accuracy": result.regime_accuracy,
            "return_mae": result.return_mae,
            "n_epochs": result.n_epochs,
            "n_train_samples": result.n_train_samples,
        },
        "latest_prediction": result.predictions,
    }


# ========== Self-Evolution System APIs ==========

# --- Layer 1: Strategy Lifecycle ---

@router.get("/lifecycle/summary")
def lifecycle_summary():
    """Get strategy lifecycle summary."""
    from research.strategy_lifecycle import StrategyLifecycleManager
    mgr = StrategyLifecycleManager()
    return mgr.summary()


@router.get("/lifecycle/{strategy_id}")
def lifecycle_get(strategy_id: str):
    """Get lifecycle record for a specific strategy."""
    from research.strategy_lifecycle import StrategyLifecycleManager
    mgr = StrategyLifecycleManager()
    rec = mgr.get_record(strategy_id)
    return {
        "strategy_id": rec.strategy_id,
        "state": rec.state.value,
        "degraded_count": rec.degraded_count,
        "total_runs": len(rec.scorecards),
        "last_updated": rec.last_updated,
        "promoted_at": rec.promoted_at,
        "retired_at": rec.retired_at,
    }


# --- Layer 3: Walk-Forward Optimizer ---

class WalkForwardRequest(BaseModel):
    strategy_id: str
    market: str = "cn"
    param_grid: dict[str, list] = Field(default_factory=lambda: {
        "lookback_days": [10, 20, 30],
        "max_positions": [5, 10, 15],
    })
    train_months: int = 24
    test_months: int = 6


@router.post("/optimize/walkforward")
def walkforward_optimize(req: WalkForwardRequest):
    """Run walk-forward optimization for a strategy."""
    from research.walk_forward import WalkForwardOptimizer, WalkForwardConfig
    from services.portfolio_backtest import market_repository

    repo = market_repository(require_exists=True)
    config = WalkForwardConfig(
        train_months=req.train_months,
        test_months=req.test_months,
    )
    opt = WalkForwardOptimizer(repo, market=req.market, config=config)
    summary = opt.optimize(req.strategy_id, req.param_grid)

    return {
        "strategy_id": summary.strategy_id,
        "best_params": summary.best_overall_params,
        "avg_oos_sharpe": round(summary.avg_oos_sharpe, 4),
        "avg_oos_return": round(summary.avg_oos_return, 4),
        "avg_is_oos_ratio": round(summary.avg_is_oos_ratio, 4),
        "parameter_stability": {k: round(v, 4) for k, v in summary.parameter_stability.items()},
        "overfitting_warnings": summary.overfitting_warnings,
        "recommendation": summary.recommendation,
        "n_splits": len(summary.splits),
    }


# --- Layer 4: Meta-Allocator ---

class AllocateRequest(BaseModel):
    method: str = Field("regime_adaptive", description="equal_weight|risk_parity|momentum_weighted|regime_adaptive|kelly")
    regime: str = Field("UNKNOWN")
    strategies: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="strategy_id → {total_return, sharpe_ratio, max_drawdown, volatility, win_rate, trade_count}"
    )
    max_single_weight: float = Field(0.40, ge=0.1, le=1.0)


@router.post("/allocate")
def meta_allocate(req: AllocateRequest):
    """Intelligent capital allocation across strategy sleeves."""
    from research.meta_allocator import MetaAllocator, SleeveMetrics

    alloc = MetaAllocator()
    sleeves = {}
    for sid, m in req.strategies.items():
        sleeves[sid] = SleeveMetrics(
            strategy_id=sid,
            total_return=m.get("total_return", 0),
            sharpe_ratio=m.get("sharpe_ratio", 0),
            max_drawdown=m.get("max_drawdown", 0),
            volatility=m.get("volatility", 0.15),
            win_rate=m.get("win_rate", 0.5),
            trade_count=int(m.get("trade_count", 0)),
            avg_win=m.get("avg_win", 0.02),
            avg_loss=m.get("avg_loss", -0.01),
        )

    result = alloc.allocate(
        sleeves,
        method=req.method,
        regime=req.regime,
        max_single_weight=req.max_single_weight,
    )

    return {
        "method": result.method,
        "weights": {k: round(v, 4) for k, v in result.weights.items()},
        "expected_sharpe": round(result.expected_sharpe, 4),
        "expected_return": round(result.expected_return, 4),
        "diversification_ratio": round(result.diversification_ratio, 4),
        "risk_contributions": {k: round(v, 4) for k, v in result.risk_contributions.items()},
        "reasoning": result.reasoning,
    }


# --- Layer 5: Self-Evolution Loop ---

@router.post("/evolution/daily")
def evolution_daily():
    """Run daily evolution cycle."""
    from research.evolution_loop import EvolutionRunner
    from services.portfolio_backtest import market_repository

    repo = market_repository(require_exists=True)
    runner = EvolutionRunner(repo, market="cn")
    result = runner.run_daily_cycle()

    return {
        "cycle_date": result.cycle_date,
        "regime": result.regime,
        "regime_confidence": round(result.regime_confidence, 2),
        "strategy_evaluations": result.strategy_evaluations[:10],
        "retrain_needed": result.retrain_needed,
        "retrain_reason": result.retrain_reason,
        "recommendations": result.recommendations,
    }


@router.post("/evolution/weekly")
def evolution_weekly():
    """Generate weekly evolution report."""
    from research.evolution_loop import EvolutionRunner
    from services.portfolio_backtest import market_repository

    repo = market_repository(require_exists=True)
    runner = EvolutionRunner(repo, market="cn")
    report = runner.run_weekly_report()

    return {
        "report_date": report.report_date,
        "period": f"{report.period_start} → {report.period_end}",
        "market_regime": report.market_regime,
        "regime_distribution": report.regime_distribution,
        "active_strategies": report.active_strategies,
        "degraded_strategies": report.degraded_strategies,
        "best_strategy": report.best_strategy,
        "worst_strategy": report.worst_strategy,
        "performance": {
            "sharpe": round(report.portfolio_sharpe, 4),
            "return": round(report.portfolio_return, 4),
            "max_dd": round(report.portfolio_max_dd, 4),
        },
        "transformer_accuracy": round(report.transformer_accuracy, 4),
        "key_insights": report.key_insights,
        "action_items": report.action_items,
    }


@router.post("/evolution/retrain")
def evolution_retrain():
    """Trigger model retraining if needed."""
    from research.evolution_loop import EvolutionRunner
    from services.portfolio_backtest import market_repository

    repo = market_repository(require_exists=True)
    runner = EvolutionRunner(repo, market="cn")
    result = runner.run_retrain_if_needed()
    return result


# --- Layer 6: AI Research Agent ---

class ResearchGenerateRequest(BaseModel):
    regime: str = Field("UNKNOWN")
    market: str = Field("cn")
    n_hypotheses: int = Field(5, ge=1, le=20)
    focus_area: str = Field("")


@router.post("/research/generate")
def research_generate(req: ResearchGenerateRequest):
    """Generate factor hypotheses using AI research agent."""
    from research.ai_agent import AIResearchAgent

    agent = AIResearchAgent()
    hypotheses = agent.generate_hypotheses(
        regime=req.regime,
        market=req.market,
        n_hypotheses=req.n_hypotheses,
        focus_area=req.focus_area,
    )

    return {
        "count": len(hypotheses),
        "hypotheses": [
            {
                "id": h.hypothesis_id,
                "name": h.name,
                "description": h.description,
                "rationale": h.rationale,
                "formula": h.formula,
                "expected_regime": h.expected_regime,
                "risk_factors": h.risk_factors,
                "implementation_code": h.implementation_code,
                "confidence": h.confidence,
                "source": h.source,
            }
            for h in hypotheses
        ],
    }


@router.post("/research/cycle")
def research_cycle():
    """Run full AI research cycle: generate → evaluate → report."""
    from research.ai_agent import AIResearchAgent
    from research.regime_detector import RegimeDetector, RegimeConfig
    from services.portfolio_backtest import market_repository

    # Detect current regime
    repo = market_repository(require_exists=True)
    rows = repo.conn.execute(
        "SELECT trade_date, symbol, COALESCE(adj_close, close) AS close FROM daily_bars WHERE market = 'cn' ORDER BY trade_date, symbol"
    ).fetchdf()
    regime = "UNKNOWN"
    if not rows.empty:
        import pandas as pd
        rows["trade_date"] = pd.to_datetime(rows["trade_date"])
        pivot = rows.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="first")
        pivot.index = [d.date() if hasattr(d, "date") else d for d in pivot.index]
        detector = RegimeDetector(config=RegimeConfig())
        signals = detector.detect_all(pivot)
        if signals:
            regime = signals[-1].regime

    agent = AIResearchAgent()
    report = agent.run_full_research_cycle(regime=regime, market="cn", n_hypotheses=5)

    return {
        "report_date": report.report_date,
        "regime": report.regime,
        "hypotheses_generated": report.hypotheses_generated,
        "promising_factors": report.promising_factors,
        "rejected_factors": report.rejected_factors,
        "insights": report.insights,
        "recommendations": report.recommendations,
    }


@router.get("/regime/dataset")
def regime_dataset_info():
    """Return regime dataset info: how labels are built from FUTURE data.

    Unlike the rule-based RegimeDetector which classifies TODAY's state,
    the ML dataset looks FORWARD from each date and labels it based on
    what actually happened in the next N days — this is the TRUE ground truth.
    """
    from ml.regime_dataset import REGIME_LABELS

    return {
        "current_method": "rule_based (RegimeDetector)",
        "ml_label_strategy": "forward_looking",
        "description": "数据集标签来自未来真实数据：对每个日期T，计算T+20日的实际市场收益率作为真值。"
                       "这比规则检测更准确，因为它基于已经发生的事实，而不是今天的估计。",
        "label_strategies": [
            {"id": "forward_return", "name": "真实收益率标签",
             "desc": "用未来20日实际收益率打标签：>+5%→BULL, <-5%→BEAR, <-10%→CRASH, 其余→RANGE"},
            {"id": "forward_regime", "name": "未来Regime多数标签",
             "desc": "用未来20日每天规则Regime的众数作为标签"},
        ],
        "model": {
            "type": "MarketTransformer (regime classification head)",
            "features": ["breadth_ma60", "breadth_ma20", "trend_20d", "trend_5d", "crash_5d",
                          "realized_vol", "vol_of_vol", "momentum_disp", "volume_trend", "breadth_mom"],
            "lookback_days": 60,
            "forward_horizon_days": 20,
            "classes": list(REGIME_LABELS),
        },
        "example": {
            "question": "今天 T 是什么 Regime？",
            "rule_based_answer": "RegimeDetector 看今天的宽度+趋势 → 规则判断",
            "ml_answer": "看 T+20 日的实际收益率 → 如果涨了8% → BULL（真值）",
        },
    }


@router.post("/research/factor/scan")
def factor_scan():
    """Run a full catalog factor scan → compute IC → detect trends → recommend strategies."""
    from research.factor_scan import FactorScanHarness
    import time

    harness = FactorScanHarness(market="cn")
    result = harness.run_monthly_scan()
    return {
        "status": result.get("status", "failed"),
        "scan_date": result.get("scan_date", ""),
        "n_factors_scanned": result.get("catalog_scan", {}).get("n_factors", 0),
        "top_factors": result.get("catalog_scan", {}).get("top_factors", [])[:10],
        "new_signals": result.get("catalog_scan", {}).get("new_signals", []),
        "decay_warnings": result.get("catalog_scan", {}).get("decay_warnings", []),
        "strategy_recommendations": result.get("catalog_scan", {}).get("strategy_recommendations", [])[:10],
        "ai_hypotheses": result.get("ai_research", {}).get("promising", []),
        "insights": result.get("insights", []),
    }


@router.get("/research/factor/history")
def factor_history():
    """Return factor scan history with emergence/decay trends."""
    from research.factor_scan import FactorTracker

    tracker = FactorTracker()
    history = tracker.load_history(n_days=180)
    trends = tracker.compute_trends(history)
    return {
        "n_factors_tracked": len(trends),
        "emerging": [
            {"name": t.factor_name, "rank_ic": round(t.current_rank_ic, 4), "z_score": round(t.z_score, 2)}
            for t in trends if t.trend_direction == "emerging"
        ][:20],
        "decaying": [
            {"name": t.factor_name, "rank_ic": round(t.current_rank_ic, 4), "z_score": round(t.z_score, 2)}
            for t in trends if t.trend_direction == "decaying"
        ][:20],
        "stable_top": [
            {"name": t.factor_name, "rank_ic": round(t.current_rank_ic, 4), "mean_rank_ic": round(t.mean_rank_ic, 4)}
            for t in sorted(trends, key=lambda x: -abs(x.current_rank_ic))[:10]
        ],
    }


@router.get("/research/factor/recommendations")
def factor_recommendations():
    """Return factor-derived strategy recommendations."""
    from datetime import date as _today
    from research.factor_scan import FactorTracker, FactorScanHarness

    tracker = FactorTracker()
    history = tracker.load_history(n_days=180)
    trends = tracker.compute_trends(history)
    harness = FactorScanHarness(market="cn")
    recs = harness._compute_strategy_recommendations(
        history=history, trends=trends,
        top_factors=[
            {"name": t.factor_name, "rank_ic": t.current_rank_ic, "ic_ir": t.rank_ic_ir, "spread": 0}
            for t in trends
        ],
    )
    return {
        "recommendations": recs[:10],
        "generated_at": _today.today().isoformat(),
    }


@router.get("/research/factor/scoreboard")
def factor_scoreboard():
    """Unified scoreboard: all hypotheses + catalog factors with IC scores.

    Returns a single sorted list: best-performing factors first,
    with source (ai_hypothesis / catalog_scan) and recommendation status.
    """
    from research.factor_scan import FactorTracker, FactorScanHarness
    from research.ai_agent import RESEARCH_LOG_DIR
    import os

    entries = []

    # 1. Load AI hypotheses (with backtest results)
    hypo_path = RESEARCH_LOG_DIR / "hypotheses.json"
    if hypo_path.exists():
        try:
            hypos = json.loads(hypo_path.read_text(encoding="utf-8"))
            for h in hypos[-50:]:
                eval_res = h.get("eval_result", {})
                bt = eval_res.get("backtest", {}) if isinstance(eval_res, dict) else {}
                entry = {
                    "name": h.get("name", "?"),
                    "description": h.get("description", "")[:80],
                    "source": "ai_hypothesis",
                    "status": "hypothesis",
                    "rank_ic": bt.get("rank_ic_mean", 0),
                    "ic_ir": bt.get("rank_ic_ir", 0),
                    "spread": bt.get("spread", 0),
                    "score": eval_res.get("overall_score", 0) if isinstance(eval_res, dict) else 0,
                    "recommendation": eval_res.get("recommendation", "") if isinstance(eval_res, dict) else "",
                    "formula": h.get("formula", "")[:120],
                    "expected_regime": h.get("expected_regime", []),
                }
                entries.append(entry)
        except Exception:
            pass

    # 2. Load scan history for catalog factors
    tracker = FactorTracker()
    history = tracker.load_history(n_days=90)
    for name, records in history.items():
        if not records:
            continue
        latest = records[-1]
        entries.append({
            "name": name,
            "description": "",
            "source": "catalog_scan",
            "status": "scanned",
            "rank_ic": latest.rank_ic,
            "ic_ir": latest.rank_ic_ir,
            "spread": latest.spread,
            "score": abs(latest.rank_ic),
            "recommendation": "⭐" if abs(latest.rank_ic) > 0.03 else "" if abs(latest.rank_ic) > 0.02 else "",
            "formula": "",
            "expected_regime": [],
        })

    # Sort: by absolute rank_ic descending
    entries.sort(key=lambda e: -abs(e.get("rank_ic", 0)))
    return {
        "total": len(entries),
        "entries": entries[:100],
    }


@router.post("/research/factor/promote")
def factor_promote(req: ResearchGenerateRequest):
    """Promote a factor hypothesis to catalog entry + run IC backtest.

    Takes a factor name and regime, finds or generates the hypothesis,
    computes IC using FactorBacktestEngine, and returns the score.
    If score > 0.02, marks as 'promotable'.
    """
    from research.ai_agent import AIResearchAgent, RESEARCH_LOG_DIR
    from research.factor_scan import FactorScanHarness

    # Generate or find the hypothesis
    agent = AIResearchAgent()
    hypotheses = agent.generate_hypotheses(
        regime=req.regime, market=req.market,
        n_hypotheses=req.n_hypotheses,
        focus_area=req.focus_area,
    )

    results = []
    for h in hypotheses:
        # Backtest the hypothesis
        bt = agent.backtest_hypothesis(h, market=req.market)
        rank_ic = bt.get("rank_ic_mean", 0) if bt.get("status") == "completed" else 0

        status = "promotable" if abs(rank_ic) > 0.02 else "weak"
        results.append({
            "name": h.name,
            "description": h.description,
            "formula": h.formula,
            "rank_ic": round(rank_ic, 4),
            "status": status,
            "recommendation": "🔥 可加入策略" if status == "promotable" else "IC不足，建议改进",
        })

    # Save hypotheses with scores
    agent._save_hypotheses(hypotheses)

    return {
        "generated": len(results),
        "promotable": sum(1 for r in results if r["status"] == "promotable"),
        "results": results,
    }


@router.get("/truth/slices")
def truth_slices(symbol: str = "", horizon: int = 21, start: str = "", end: str = ""):
    """Return ground truth data for a single stock, sliced by regime periods.

    Loads the stock's close prices from DuckDB, computes forward-looking
    regime labels (what actually happened in the next N days), segments
    by regime, and returns summary stats for each slice.

    Args:
        symbol: Stock symbol like "000300.SSE" or "AAPL". Empty = use 510300.
        horizon: Forward horizon days for regime labeling (default 21).
        start: Start date filter (YYYY-MM-DD), empty = full history.
        end: End date filter (YYYY-MM-DD), empty = latest available.
    """
    from services.portfolio_backtest import market_repository

    if not symbol:
        symbol = "510300"

    try:
        repo = market_repository(require_exists=True)

        # Find the symbol (try with and without exchange suffix)
        candidates = repo.conn.execute(
            "SELECT DISTINCT symbol FROM daily_bars WHERE market = 'cn' AND symbol LIKE ? ORDER BY symbol",
            [f"%{symbol}%"],
        ).fetchall()
        if not candidates:
            return {"status": "error", "message": f"标的 {symbol} 未找到"}

        full_symbol = candidates[0][0]

        # Fetch OHLC for K-line display
        import pandas as pd
        ohlc_query = "SELECT trade_date, COALESCE(adj_open, open) AS o, high, low, COALESCE(adj_close, close) AS c " \
                     "FROM daily_bars WHERE market = 'cn' AND symbol = ?"
        ohlc_params = [full_symbol]
        if start:
            ohlc_query += " AND trade_date >= ?"
            ohlc_params.append(start)
        if end:
            ohlc_query += " AND trade_date <= ?"
            ohlc_params.append(end)
        ohlc_query += " ORDER BY trade_date"
        ohlc = repo.conn.execute(ohlc_query, ohlc_params).fetchdf()
        ohlc["trade_date"] = pd.to_datetime(ohlc["trade_date"]).dt.strftime("%Y-%m-%d")

        # Fetch Chinese name
        name_row = repo.conn.execute(
            "SELECT name FROM instruments WHERE market = 'cn' AND symbol = ?", [full_symbol]
        ).fetchone()
        chinese_name = name_row[0] if name_row else full_symbol

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
        from ml.slicer import slice_from_db, to_api_payload

        result = slice_from_db(repo, full_symbol, horizon=horizon, start=start or None, end=end or None)
        payload = to_api_payload(result)
        payload["status"] = "ok"
        payload["name_cn"] = chinese_name
        payload["ohlc"] = ohlc.to_dict(orient="records")
        return payload

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/truth/bull-stocks")
def truth_bull_stocks(horizon: int = 21, top_n: int = 30, direction: str = "bull"):
    """Find stocks currently in BULL or BEAR regime.

    For each stock, computes the most recent complete forward return
    (from T-horizon to T).

    Args:
        horizon: Forward horizon days.
        top_n: Max results to return.
        direction: "bull" (ret > +5%) or "bear" (ret < -5%).
    """
    from services.portfolio_backtest import market_repository
    import pandas as pd

    try:
        repo = market_repository(require_exists=True)
        # Get all active CN stocks (exclude delisted/ST)
        symbols = repo.conn.execute(
            "SELECT symbol, name FROM instruments WHERE market = 'cn' AND is_active = TRUE "
            "AND (name NOT LIKE '%退%' OR name IS NULL) "
            "AND (name NOT LIKE 'ST%' OR name IS NULL) "
            "AND (name NOT LIKE '*ST%' OR name IS NULL) "
            "ORDER BY symbol"
        ).fetchall()

        results = []
        for sym, name in symbols:
            try:
                rows = repo.conn.execute(
                    "SELECT trade_date, COALESCE(adj_close, close) AS c FROM daily_bars "
                    "WHERE market = 'cn' AND symbol = ? ORDER BY trade_date",
                    [sym],
                ).fetchdf()
                if len(rows) < horizon + 5:
                    continue
                # Most recent date with a COMPLETE forward window = T-horizon
                # forward_return = price[T] / price[T-horizon] - 1
                # This is the regime at date (T-horizon), not at T
                latest_close = float(rows["c"].iloc[-1])
                past_close = float(rows["c"].iloc[-horizon - 1])
                if latest_close <= 0 or past_close <= 0:
                    continue
                fwd_ret = latest_close / past_close - 1.0
                is_bull = fwd_ret > 0.05
                is_bear = fwd_ret < -0.05
                is_crash = fwd_ret < -0.10
                if (direction == "bull" and is_bull) or (direction == "bear" and is_bear) or (direction == "crash" and is_crash):
                    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
                    from ml.labels import soft_regime_probs, hard_regime
                    probs = soft_regime_probs(fwd_ret)
                    # Mini close prices for sparkline (last 60 days)
                    mini_closes = [round(float(c), 2) for c in rows["c"].iloc[-min(60, len(rows)):].tolist()]
                    mini_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in rows["trade_date"].iloc[-min(60, len(rows)):].tolist()]
                    # Regime colors per day for the same window
                    mini_regimes = []
                    for j in range(len(mini_closes) - horizon):
                        if j + horizon < len(mini_closes):
                            ret_j = mini_closes[j + horizon] / mini_closes[j] - 1.0
                            hr = hard_regime(ret_j)
                            mini_regimes.append(hr)
                        else:
                            mini_regimes.append(1)  # RANGE
                    results.append({
                        "symbol": sym,
                        "name": name or sym,
                        "forward_return_21d": round(fwd_ret, 4),
                        "bull_probability": round(probs[0], 4),
                        "bear_probability": round(probs[2] + probs[3], 4),
                        "crash_probability": round(probs[3], 4),
                        "regime": "BULL" if is_bull else "BEAR",
                        "latest_date": str(rows["trade_date"].iloc[-1].date()) if hasattr(rows["trade_date"].iloc[-1], "date") else str(rows["trade_date"].iloc[-1]),
                        "mini_closes": mini_closes,
                        "mini_dates": mini_dates,
                        "mini_regimes": mini_regimes,
                    })
            except Exception:
                continue

        label_map = {"bull": ("BULL", "> +5%"), "bear": ("BEAR", "< -5%"), "crash": ("CRASH", "< -10%")}
        label, threshold = label_map.get(direction, ("BULL", "> +5%"))
        results.sort(key=lambda x: -x["forward_return_21d"] if direction == "bull" else x["forward_return_21d"])
        return {
            "status": "ok",
            "horizon_days": horizon,
            "direction": direction,
            f"n_{direction}_stocks": len(results),
            "stocks": results[:top_n],
            "note": f"基于最近完整 forward 窗口（T-21 ~ T）的实际涨幅 {threshold}",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/truth/industry-stocks")
def truth_industry_stocks(industry: str = "", horizon: int = 21):
    """Return stocks in a given industry with their regime data.

    Args:
        industry: Industry name (e.g., "半导体", "元件").
        horizon: Forward horizon days.
    """
    from services.portfolio_backtest import market_repository

    if not industry:
        return {"status": "error", "message": "industry parameter is required"}

    try:
        repo = market_repository(require_exists=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
        from ml.industry import load_sector_map
        from ml.labels import soft_regime_probs, hard_regime

        sector_map = load_sector_map()
        symbols = [sym for sym, ind in sector_map.items() if ind == industry]
        if not symbols:
            return {"status": "error", "message": f"行业 {industry} 未找到"}

        stocks = []
        for sym in symbols[:50]:  # limit to 50
            try:
                rows = repo.conn.execute(
                    "SELECT trade_date, symbol, COALESCE(adj_close, close) AS c, "
                    "COALESCE(adj_open, open) AS o, high, low "
                    "FROM daily_bars WHERE market = 'cn' AND symbol = ? ORDER BY trade_date",
                    [sym],
                ).fetchdf()
                if len(rows) < horizon + 5:
                    continue
                latest_close = float(rows["c"].iloc[-1])
                past_close = float(rows["c"].iloc[-horizon - 1])
                if latest_close <= 0 or past_close <= 0:
                    continue
                fwd_ret = latest_close / past_close - 1.0
                probs = soft_regime_probs(fwd_ret)
                r = int(np.argmax(probs))

                # Get name
                name_row = repo.conn.execute(
                    "SELECT name FROM instruments WHERE market = 'cn' AND symbol = ?", [sym]
                ).fetchone()
                name = name_row[0] if name_row else sym

                mini_closes = [round(float(c), 2) for c in rows["c"].iloc[-60:].tolist()]
                stocks.append({
                    "symbol": sym,
                    "name": name,
                    "forward_return": round(fwd_ret, 4),
                    "regime": ["BULL", "RANGE", "BEAR", "CRASH"][r],
                    "bull_prob": round(probs[0], 4),
                    "bear_prob": round(probs[2] + probs[3], 4),
                    "crash_prob": round(probs[3], 4),
                    "mini_closes": mini_closes,
                })
            except Exception:
                continue

        stocks.sort(key=lambda x: -x["forward_return"])
        return {
            "status": "ok",
            "industry": industry,
            "n_stocks": len(stocks),
            "stocks": stocks,
            "regime_colors": {
                "BULL": "#1f8b4c", "RANGE": "#b8a894",
                "BEAR": "#cc5544", "CRASH": "#b33a3a",
            },
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/research/factor/to-strategy")
def factor_to_strategy(name: str = ""):
    """Promote a high-IC factor to a tradable strategy.

    Steps:
    1. Find or compute the factor's IC
    2. Map to best strategy type (momentum_top_k / mean_reversion_top_k / etc.)
    3. Run a quick backtest with the factor as ranking signal
    4. If IC > 0.02, register in the strategy registry with a unique ID
    5. Return the new strategy ID and backtest results

    Args:
        name: Factor name from catalog (e.g., "ret_20d", "ma_cross_10_30").
    """
    from services.portfolio_backtest import market_repository
    from factors.catalog import FACTOR_CATALOG, FactorSpec, FactorTime, FactorMarket, FactorType
    from factors.backtest import FactorBacktestEngine
    from portfolio.strategy_registry import PORTFOLIO_STRATEGY_BUILDERS
    import datetime

    if not name:
        return {"status": "error", "message": "factor name is required"}

    try:
        repo = market_repository(require_exists=True)

        # 1. Find the factor in catalog
        spec = None
        for s in FACTOR_CATALOG:
            if s.name == name:
                spec = s
                break
        if spec is None:
            return {"status": "error", "message": f"因子 {name} 未在 catalog 中找到"}

        # 2. Compute IC
        engine = FactorBacktestEngine(
            market="cn", sample_size=200, recent_days=300,
            existing_conn=repo.conn,
        )
        engine.load_data()
        factor_df = engine.compute_factor(spec)
        if factor_df is not None:
            engine.factor_values[spec.name] = factor_df
        eval_result = engine.evaluate_factor(name, forward_period=5)
        rank_ic = eval_result.get("rank_ic", 0)

        # 3. Map to strategy type
        from research.factor_scan import FactorScanHarness
        harness = FactorScanHarness(market="cn")
        suggested_strategy = harness._map_factor_to_strategy(name)

        # 4. If IC is strong enough, create a strategy entry
        is_promotable = bool(abs(rank_ic) > 0.02)
        strategy_id = f"auto_{name}" if is_promotable else None

        result = {
            "factor_name": name,
            "rank_ic": round(rank_ic, 4),
            "ic_ir": round(eval_result.get("ic_ir", 0), 2),
            "spread": round(eval_result.get("spread", 0), 4),
            "suggested_strategy": suggested_strategy,
            "is_promotable": is_promotable,
            "strategy_id": strategy_id,
            "n_samples": eval_result.get("n_obs", 0),
        }

        # Record experiment
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
        from ml.experiment_tracker import get_tracker
        get_tracker().record(
            "strategy_promote",
            params={"factor_name": name, "suggested_strategy": suggested_strategy},
            metrics={"rank_ic": round(rank_ic, 4), "ic_ir": round(eval_result.get("ic_ir", 0), 2)},
            tags=[name, suggested_strategy, "auto_generated" if is_promotable else "rejected"],
        )

        if is_promotable:
            import json
            auto_dir = Path(__file__).resolve().parents[2] / "data" / "auto_strategies"
            auto_dir.mkdir(parents=True, exist_ok=True)
            path = auto_dir / f"{strategy_id}.json"
            payload = {
                "strategy_id": strategy_id,
                "factor_name": name,
                "base_strategy": suggested_strategy,
                "rank_ic": round(rank_ic, 4),
                "created_at": datetime.datetime.now().isoformat(),
                "status": "active",
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

            # Trigger model retraining with reward mode
            retrain_result = {}
            try:
                from ml.trainer import train_market_transformer, TrainConfig
                retrain_config = TrainConfig(
                    lookback=30, batch_size=64, max_epochs=50, patience=10,
                    d_model=48, n_heads=4, n_layers=2,
                    reward_mode=True,
                )
                rt = train_market_transformer(repo, market="cn", config=retrain_config)
                retrain_result = {
                    "retrained": True,
                    "new_regime_acc": round(rt.regime_accuracy, 4),
                    "new_return_mae": round(rt.return_mae, 4),
                }
            except Exception as e:
                retrain_result = {"retrained": False, "error": str(e)}

            result["message"] = f"✅ 策略 {strategy_id} 已注册 + 模型重训完成" if retrain_result.get("retrained") else f"✅ 策略 {strategy_id} 已注册"
            result["retrain"] = retrain_result
        else:
            result["message"] = f"IC={rank_ic:.4f} 不足 0.02，暂不自动注册"

        return {"status": "ok", **result}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@router.get("/research/factor/correlation")
def factor_correlation(top_n: int = 20):
    """Compute Spearman correlation matrix between top factors.

    Identifies factor crowding: highly correlated factors (>0.7) are flagged,
    and factor clusters are returned so users can pick one per cluster.
    """
    from services.portfolio_backtest import market_repository

    try:
        repo = market_repository(require_exists=True)
        from factors.backtest import FactorBacktestEngine

        engine = FactorBacktestEngine(
            market="cn", sample_size=200, recent_days=300,
            existing_conn=repo.conn,
        )
        engine.load_data()
        engine.compute_all_factors()
        engine.evaluate_all(forward_period=5)

        result = engine.factor_correlation_matrix(top_n=min(top_n, 15))
        return {"status": "ok", **result}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@router.get("/truth/industry-analysis")
def truth_industry_analysis(horizon: int = 21, min_stocks: int = 3):
    """Analyze regime distribution per industry/sector.

    For each industry (from board_constituents.json), counts the number of
    stocks in BULL/BEAR/RANGE/CRASH based on the most recent complete forward window.
    """
    from services.portfolio_backtest import market_repository

    try:
        repo = market_repository(require_exists=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
        from ml.industry import analyze

        results = analyze(repo, horizon=horizon, min_stocks=min_stocks)
        return {
            "status": "ok",
            "horizon_days": horizon,
            "n_industries": len(results),
            "industries": results[:100],
            "regime_colors": {
                "BULL": "#1f8b4c", "RANGE": "#b8a894",
                "BEAR": "#cc5544", "CRASH": "#b33a3a",
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/evolution/auto-cycle")
def auto_cycle():
    """Run one complete self-driving evolution cycle.

    Steps:
    1. Scan all catalog factors → IC/Rank IC
    2. Build correlation matrix → identify unique factor clusters
    3. Promote top uncorrelated factors to strategies
    4. Retrain Transformer with reward monitoring
    5. Log everything to experiment tracker

    Designed to be triggered by crontab (daily) or manually.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
    from ml.auto_loop import run_auto_cycle

    try:
        result = run_auto_cycle(
            market="cn",
            top_n_factors=10,
            min_ic_threshold=0.025,
            max_factors_per_cycle=3,
            retrain_model=True,
        )
        return {"status": "ok", **result}
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@router.post("/evolution/factor-mining-cycle")
def factor_mining_cycle():
    """Run one batch of automated factor mining.

    Generates new factor hypotheses via DeepSeek + rule templates,
    computes IC/Rank IC for each, stores in factor_hypotheses.jsonl.
    Target: 2000 factors. Run multiple times to accumulate.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
    from ml.factor_mining_loop import run_mining_cycle
    try:
        result = run_mining_cycle(target_total=2000, batch_size=10)
        return {"status": "ok", **result}
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@router.get("/experiments")
def list_experiments(experiment_type: str = "", tag: str = "", limit: int = 20):
    """Query experiment tracking records."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
    from ml.experiment_tracker import get_tracker
    tracker = get_tracker()
    records = tracker.query(
        experiment_type=experiment_type or None,
        tag=tag or None,
        limit=min(limit, 100),
    )
    return {
        "total": len(records),
        "records": [{
            "id": r.experiment_id, "type": r.experiment_type,
            "time": r.timestamp, "params": r.params,
            "metrics": r.metrics, "tags": r.tags,
        } for r in records],
    }

@router.get("/data/refresh/status")
def data_refresh_status():
    """Return refresh status for all external data sources."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
    from pg_ml.refresh_pipeline import get_manager
    mgr = get_manager()
    return mgr.status()


@router.post("/data/refresh/all")
def data_refresh_all():
    """Refresh all stale external data sources."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
    from pg_ml.refresh_pipeline import get_manager
    mgr = get_manager()
    return mgr.refresh_all(force=False)


@router.get("/data/refresh/view/{source_name}")
def data_refresh_view(source_name: str):
    """View actual data for a specific source."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / ".." / "prediction_ground_truth"))
    from pg_ml.refresh_pipeline import get_manager
    import pandas as pd
    import json

    mgr = get_manager()
    for s in mgr.sources:
        if s.name == source_name:
            try:
                df = s.refresh_fn()
                if df is not None:
                    if hasattr(df, 'to_dict'):
                        rows = df.reset_index().head(500).to_dict(orient='records')
                        return {"status": "ok", "rows": rows, "n_rows": len(df),
                                "columns": list(df.columns)}
                    else:
                        return {"status": "ok", "data": str(df)[:5000]}
            except Exception as e:
                return {"status": "error", "message": str(e)}
    return {"status": "error", "message": f"Source '{source_name}' not found"}
