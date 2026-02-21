"""
美股回测系统 API：前后端分离，仅提供 REST 接口。
"""
from __future__ import annotations

import sys
import os

# 保证可导入 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.data import fetch_ohlcv
from services.etf import evaluate_etfs
from services.etf_llm import get_etf_5y_summary, run_llm_score
from services.etf_valuation import get_undervalued_etfs
from services.news import get_us_stock_news, get_cn_a_share_news, get_futures_news, get_hk_stock_news
from database import get_db, is_configured
from database.models import BacktestRun
from backtest.engine import run_backtest, BacktestResult, Trade, STRATEGIES
from backtest.custom_strategy import run_custom_backtest

app = FastAPI(
    title="美股回测系统 API",
    description="提供历史行情与双均线策略回测接口",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端单页：/ 返回 index.html，避免覆盖 /api
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")


@app.get("/")
def serve_index():
    if os.path.isfile(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"message": "美股回测 API 运行中。请将 frontend/index.html 放到前端服务器或访问 /api/health"}


# ---------- 请求/响应模型 ----------
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


class TradeOut(BaseModel):
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float


class BacktestResponse(BaseModel):
    strategy_used: str = ""  # 实际使用的策略，便于核对是否与选择一致
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
    equity_curve: List[dict]  # [{"date": "2020-01-01", "value": 100000}, ...]
    trades: List[TradeOut]


class SaveBacktestRequest(BaseModel):
    """保存回测到数据库：回测结果 + 可选 user_id / anonymous_id。"""
    # 与 BacktestResponse 一致，用于入库
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
    # 可选：登录用户 id 或匿名设备 id
    user_id: Optional[int] = None
    anonymous_id: Optional[str] = None
    # 可选：策略参数快照
    params: Optional[dict] = None
    market: str = "us"


class EtfCompareRequest(BaseModel):
    """ETF 评估：多标的买入持有对比。"""
    symbols: List[str] = Field(..., min_length=2, max_length=20, description="标的代码列表，如 SPY, QQQ, VOO")
    start: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    end: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    period: Optional[str] = Field("5y", description="未指定 start/end 时使用，如 1y, 5y, 10y")
    market: str = Field("us", description="市场：us/cn/hk")
    initial_cash: float = Field(100_000.0, ge=100)


# ---------- 接口 ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "stock-backtest"}


@app.get("/api/strategies")
def list_strategies():
    """返回可用策略列表及参数说明，供前端渲染表单。"""
    return {"strategies": STRATEGIES}


@app.get("/api/history")
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
        # 返回日期与 OHLCV，便于前端画 K 线或折线
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
    return {"fast_period": req.fast_period or 10, "slow_period": req.slow_period or 30}


@app.post("/api/backtest", response_model=BacktestResponse)
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
                exit_date=t.exit_date,
                exit_price=round(t.exit_price, 4) if t.exit_price else 0,
                shares=round(t.shares, 4),
                pnl=round(t.pnl, 2),
                pnl_pct=round(t.pnl_pct, 2),
            )
            for t in res.trades
        ],
    )


@app.post("/api/etf/compare")
def etf_compare_endpoint(req: EtfCompareRequest):
    """多 ETF/标的买入持有对比评估：同一区间下收益、波动、夏普、最大回撤及归一化净值曲线。"""
    period = (req.period or "5y").strip().lower()
    if not (req.start and req.end):
        allowed = ("6mo", "1y", "2y", "5y", "10y")
        period = period if period in allowed else "5y"
    try:
        etfs = evaluate_etfs(
            symbols=req.symbols,
            start=req.start,
            end=req.end,
            period=period,
            market=req.market or "us",
            initial_cash=req.initial_cash,
        )
        return {"period": period, "market": req.market, "etfs": etfs}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ETF 评估失败: {e}")


@app.post("/api/backtest/custom", response_model=BacktestResponse)
def run_custom_backtest_endpoint(req: CustomBacktestRequest):
    """
    提交自定义 Python 代码进行回测。
    代码必须定义函数 signal(df)，参数 df 为 DataFrame（含 Open/High/Low/Close/Volume），
    返回与 df 同长度的 0/1 序列（1=持多，0=空仓）。仅允许使用 pandas、numpy 及安全内置函数，执行限时 15 秒。
    """
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


@app.get("/api/news/us")
def us_stock_news():
    """美股市场新闻：聚合 WSJ / CNBC 等主流媒体 RSS。"""
    try:
        items = get_us_stock_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取美股新闻失败: {e}")


@app.get("/api/news/cn")
def cn_a_share_news():
    """A 股相关新闻：来源于上交所英文官网指数新闻。"""
    try:
        items = get_cn_a_share_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取 A 股新闻失败: {e}")


@app.get("/api/news/futures")
def futures_news():
    """期货市场新闻：优先东方财富，备选新浪财经。"""
    try:
        items = get_futures_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取期货新闻失败: {e}")


@app.get("/api/news/hk")
def hk_stock_news():
    """港股市场新闻：优先东方财富，备选新浪财经。"""
    try:
        items = get_hk_stock_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取港股新闻失败: {e}")


@app.get("/api/etf/undervalued")
def undervalued_etfs(top_n: int = 10):
    """美股相对低估 Top N ETF：基于 P/E、P/B、股息率打分并给出理由。"""
    if top_n < 1 or top_n > 20:
        top_n = 10
    try:
        items = get_undervalued_etfs(top_n=top_n)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取低估 ETF 失败: {e}")


class EtfLlmScoreRequest(BaseModel):
    """单只 ETF 大模型评分请求（五年数据 + Falcon-H1R-7B）。"""
    symbol: str = Field(..., description="标的代码，如 SPY、QQQ")
    market: str = Field("us", description="市场：us/cn/hk")


@app.post("/api/etf/llm-score")
def etf_llm_score_endpoint(req: EtfLlmScoreRequest):
    """获取单只 ETF 的五年表现摘要，并用大模型 tiiuae/Falcon-H1R-7B 输出 0-100 分及理由。"""
    try:
        summary = get_etf_5y_summary(req.symbol, market=req.market or "us")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        score_result = run_llm_score(req.symbol, summary)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "symbol": req.symbol,
        "market": req.market,
        "summary": summary,
        "score": score_result["score"],
        "reason": score_result["reason"],
    }


# ---------- 数据库：保存回测 / 历史记录（需配置 DATABASE_URL） ----------
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


@app.post("/api/backtest/save")
async def save_backtest(req: SaveBacktestRequest, db: Optional[AsyncSession] = Depends(get_db)):
    """将本次回测结果保存到数据库；需配置 DATABASE_URL。可选传 user_id 或 anonymous_id。"""
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


@app.get("/api/history/records")
async def list_backtest_records(
    user_id: Optional[int] = None,
    anonymous_id: Optional[str] = None,
    limit: int = 50,
    db: Optional[AsyncSession] = Depends(get_db),
):
    """分页拉取已保存的回测记录；按 user_id 或 anonymous_id 筛选，需配置 DATABASE_URL。"""
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
