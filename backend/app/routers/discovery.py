"""Discovery Loop 只读 API：实验账本、候选、失败记忆。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from discovery.ledger import ExperimentLedger

router = APIRouter(prefix="/api")
_ledger: Optional[ExperimentLedger] = None


def _get_ledger() -> ExperimentLedger:
    global _ledger
    if _ledger is None:
        _ledger = ExperimentLedger()
    return _ledger


@router.get("/discovery/summary")
def discovery_summary():
    """账本总览：候选/实验/失败记忆数量与最佳实验。"""
    try:
        return _get_ledger().summary()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取账本失败: {exc}") from exc


@router.get("/discovery/experiments")
def list_discovery_experiments(
    strategy: Optional[str] = None,
    status: Optional[str] = None,
    min_dsr: Optional[float] = None,
    limit: int = 100,
):
    """实验列表（按时间倒序），可按策略/状态/DSR 过滤。"""
    if limit < 1 or limit > 1000:
        limit = 100
    try:
        return {"items": _get_ledger().list_experiments(
            strategy=strategy, status=status, min_dsr=min_dsr, limit=limit
        )}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取实验失败: {exc}") from exc


@router.get("/discovery/leaderboard")
def discovery_leaderboard(metric: str = "dsr", limit: int = 20):
    """按 DSR / PBO / FDR 排序的实验榜。"""
    if metric not in {"dsr", "pbo", "fdr"}:
        raise HTTPException(status_code=400, detail="metric 仅支持 dsr/pbo/fdr")
    if limit < 1 or limit > 200:
        limit = 20
    try:
        return {"metric": metric, "items": _get_ledger().leaderboard(metric=metric, limit=limit)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取排行榜失败: {exc}") from exc


@router.get("/discovery/candidates")
def list_discovery_candidates(
    stage: Optional[str] = None,
    promotion_status: Optional[str] = None,
    limit: int = 100,
):
    """候选基因组列表。"""
    if limit < 1 or limit > 1000:
        limit = 100
    try:
        return {"items": _get_ledger().list_candidates(
            stage=stage, promotion_status=promotion_status, limit=limit
        )}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取候选失败: {exc}") from exc


@router.get("/discovery/candidates/{candidate_id}")
def get_discovery_candidate(candidate_id: str):
    """单个候选的 Genome、血缘与实验记录。"""
    ledger = _get_ledger()
    candidate = ledger.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="候选不存在")
    experiments = [
        e for e in ledger.list_experiments(limit=200)
        if e.get("candidate_id") == candidate_id
    ]
    return {"candidate": candidate, "experiments": experiments}


@router.get("/discovery/lessons")
def list_discovery_lessons(limit: int = 100):
    """失败记忆：供下一轮假设生成检索。"""
    if limit < 1 or limit > 1000:
        limit = 100
    try:
        return {"items": _get_ledger().list_lessons(limit=limit)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取失败记忆失败: {exc}") from exc
