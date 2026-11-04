"""
ETF 相关 API：对比评估、低估 ETF、持仓、LLM 评分。
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# 保证可导入 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.etf import evaluate_etfs
from services.etf_llm import get_etf_5y_summary, run_llm_score
from services.etf_valuation import get_undervalued_etfs, get_etf_holdings

router = APIRouter(prefix="/api")


# ---------- 请求模型 ----------
class EtfCompareRequest(BaseModel):
    """ETF 评估：多标的买入持有对比。"""
    symbols: List[str] = Field(..., min_length=2, max_length=20, description="标的代码列表，如 SPY, QQQ, VOO")
    start: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    end: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    period: Optional[str] = Field("5y", description="未指定 start/end 时使用，如 1y, 5y, 10y")
    market: str = Field("us", description="市场：us/cn/hk")
    initial_cash: float = Field(100_000.0, ge=100)


class EtfLlmScoreRequest(BaseModel):
    """单只 ETF 大模型评分请求（五年数据 + Falcon-H1R-7B）。"""
    symbol: str = Field(..., description="标的代码，如 SPY、QQQ")
    market: str = Field("us", description="市场：us/cn/hk")


# ---------- 接口 ----------
@router.post("/etf/compare")
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


@router.get("/etf/undervalued")
def undervalued_etfs(top_n: int = 10):
    """美股相对低估 Top N ETF：基于 P/E、P/B、股息率打分并给出理由。"""
    if top_n < 1 or top_n > 20:
        top_n = 10
    try:
        items = get_undervalued_etfs(top_n=top_n)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取低估 ETF 失败: {e}")


@router.get("/etf/holdings")
def etf_holdings(symbol: str, top_n: int = 20):
    """获取 ETF 的 Top N 持仓（代码、名称、权重）。"""
    if not symbol:
        raise HTTPException(status_code=400, detail="缺少 symbol 参数")
    if top_n < 1 or top_n > 50:
        top_n = 20
    try:
        holdings = get_etf_holdings(symbol.upper(), top_n=top_n)
        return {"symbol": symbol.upper(), "holdings": holdings}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取 ETF 持仓失败: {e}")


@router.post("/etf/llm-score")
def etf_llm_score_endpoint(req: EtfLlmScoreRequest):
    """获取单只 ETF 的五年表现摘要，并用大模型输出 0-100 分及理由。"""
    try:
        summary = get_etf_5y_summary(req.symbol, market=req.market or "us")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败: {type(e).__name__}: {e}")
    try:
        score_result = run_llm_score(req.symbol, summary)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型评分失败: {type(e).__name__}: {e}")
    return {
        "symbol": req.symbol,
        "market": req.market,
        "summary": summary,
        "score": score_result["score"],
        "reason": score_result["reason"],
    }
