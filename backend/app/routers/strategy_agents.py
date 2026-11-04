"""
策略库、组合策略、Agent 分析、记忆查询 API。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException

# 保证可导入 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.strategy_library import (
    get_strategy_library_facets,
    get_strategy_library_item,
    list_strategy_library,
)
from portfolio.strategy_registry import supported_portfolio_strategies

router = APIRouter(prefix="/api")


# ---------- Agent 懒加载状态 ----------
_agent_registry = None
_agent_orchestrator = None
_agent_store = None
_agent_lock = threading.Lock()


def _get_agent_pipeline():
    """Lazy-init the agent pipeline (registry + orchestrator)."""
    global _agent_registry, _agent_orchestrator, _agent_store  # noqa: PLW0603
    if _agent_registry is not None:
        return _agent_registry, _agent_orchestrator, _agent_store

    with _agent_lock:
        if _agent_registry is not None:
            return _agent_registry, _agent_orchestrator, _agent_store

        from agents.base import BaseAgent  # noqa: F401
        from agents.registry import AgentRegistry
        from agents.orchestrator import Orchestrator
        from agents.sentiment import SentimentAgent
        from agents.strategy import StrategyAgent
        from agents.risk import RiskAgent
        from memory.store import MemoryStore
        from memory.session import SessionMemory
        from tools.specific import IndustryTool, MarketDataTool, NewsTool

        store = MemoryStore()
        session = SessionMemory()

        tools = {
            "news": NewsTool(),
            "industry": IndustryTool(),
            "market_data": MarketDataTool(),
        }

        registry = AgentRegistry()
        registry.register(SentimentAgent(tools=tools, memory=store, session=session))
        registry.register(StrategyAgent(tools=tools, memory=store, session=session))
        registry.register(RiskAgent(tools=tools, memory=store, session=session))

        orch = Orchestrator(registry, store, session)

        _agent_registry = registry
        _agent_orchestrator = orch
        _agent_store = store
        return registry, orch, store


# ---------- 策略库 ----------
@router.get("/strategy-library")
def list_imported_strategy_library(
    q: Optional[str] = None,
    collection: Optional[str] = None,
    tag: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
):
    """查询已导入的 JoinQuant 策略库，包含评分、介绍和源码路径。"""
    return list_strategy_library(
        q=q,
        collection=collection,
        tag=tag,
        min_score=min_score,
        limit=limit,
        offset=offset,
    )


@router.get("/strategy-library/facets")
def strategy_library_facets():
    """返回策略库集合、标签和评分口径。"""
    return get_strategy_library_facets()


@router.get("/strategy-library/{strategy_id}")
def imported_strategy_detail(strategy_id: str, include_source: bool = False):
    """读取单个导入策略详情；include_source=true 时返回原始策略源码。"""
    try:
        return get_strategy_library_item(strategy_id, include_source=include_source)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"策略不存在: {strategy_id}")


# ---------- 组合策略 ----------
@router.get("/portfolio-strategies")
def list_portfolio_strategies():
    """返回组合回测 API 当前支持的 strategy id 列表。"""
    return {"strategies": sorted(supported_portfolio_strategies())}


# ---------- Agent ----------
@router.post("/agents/run")
def run_agents(ticker: str | None = None):
    """运行全部分析 Agent，返回信号和最终决策。"""
    try:
        registry, orch, store = _get_agent_pipeline()
        signals, decision = orch.run(ticker=ticker, force=True)

        result: dict = {
            "signals": {
                name: [s.model_dump() for s in sigs]
                for name, sigs in signals.items()
            },
            "decision": decision.model_dump() if decision else None,
            "agents": registry.list_names(),
        }

        orch.take_snapshot()
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Agent analysis failed: {exc}") from exc


@router.post("/agents/run/{agent_name}")
def run_single_agent(agent_name: str, ticker: str | None = None):
    """运行单个 Agent。"""
    try:
        registry, orch, store = _get_agent_pipeline()
        agent = registry.get(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")
        signal = orch.run_single(agent_name, ticker=ticker)
        return {"signal": signal.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Agent {agent_name} failed: {exc}") from exc


@router.get("/agents/status")
def agents_status():
    """获取所有 Agent 的最新信号摘要。"""
    try:
        registry, orch, store = _get_agent_pipeline()
        status: dict = {}
        for name in registry.list_names():
            last = store.latest_signal(name)
            entry = {
                "score": last.score if last else None,
                "confidence": last.confidence if last else None,
                "reasoning": (last.reasoning[:100] + "...") if last and len(last.reasoning) > 100 else (last.reasoning if last else ""),
                "updated": last.created_at if last else None,
            }
            # 暴露卖出指数和计算步骤
            if last and last.metadata:
                si = last.metadata.get("sell_index")
                if si is not None:
                    entry["sell_index"] = si
                cs = last.metadata.get("calc_steps")
                if cs:
                    entry["calc_steps"] = cs
            status[name] = entry
        last_decision = store.query_decisions(limit=1)
        from services.llm_client import is_api_available
        return {
            "agents": status,
            "last_decision": last_decision[0].model_dump() if last_decision else None,
            "deepseek_configured": is_api_available(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Status query failed: {exc}") from exc


# ---------- 记忆查询 ----------
@router.get("/memory/signals")
def query_signals(agent_name: str | None = None, limit: int = 20, since: str | None = None):
    """查询历史信号记录。"""
    try:
        _, _, store = _get_agent_pipeline()
        signals = store.query_signals(agent_name=agent_name, since=since, limit=limit)
        return {"count": len(signals), "signals": [s.model_dump() for s in signals]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Signal query failed: {exc}") from exc


@router.get("/memory/decisions")
def query_decisions(limit: int = 20):
    """查询历史决策记录。"""
    try:
        _, _, store = _get_agent_pipeline()
        decisions = store.query_decisions(limit=limit)
        return {"count": len(decisions), "decisions": [d.model_dump() for d in decisions]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decision query failed: {exc}") from exc
