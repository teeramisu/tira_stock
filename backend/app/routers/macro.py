"""宏观数据 API：NBS/PBOC 指标查询与刷新。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from macro import MacroStore, refresh as refresh_macro

router = APIRouter(prefix="/api")
_store: Optional[MacroStore] = None


def _get_store() -> MacroStore:
    global _store
    if _store is None:
        _store = MacroStore()
    return _store


@router.get("/macro/indicators")
def macro_indicators():
    """指标目录与覆盖区间。"""
    try:
        return {"items": _get_store().list_indicators()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取宏观指标失败: {exc}") from exc


@router.get("/macro/series/{series_id}")
def macro_series(series_id: str, start: Optional[str] = None, end: Optional[str] = None):
    """单指标时间序列。"""
    try:
        items = _get_store().query_series(series_id, start=start, end=end)
        return {"series_id": series_id, "count": len(items), "items": items}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取宏观序列失败: {exc}") from exc


@router.get("/macro/latest")
def macro_latest():
    """各指标最新值。"""
    try:
        return {"items": _get_store().latest()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取宏观最新值失败: {exc}") from exc


@router.get("/macro/wide")
def macro_wide(frequency: str = "monthly"):
    """宽表：period × indicator。"""
    if frequency not in {"monthly", "quarterly"}:
        raise HTTPException(status_code=400, detail="frequency 仅支持 monthly/quarterly")
    try:
        df = _get_store().wide(frequency)
        if df.empty:
            return {"frequency": frequency, "columns": [], "rows": []}
        df = df.reset_index().rename(columns={"index": "period"})
        # 宽表天然含 NaN，FastAPI 默认 JSON 不接受 NaN，转成 null
        import json as _json
        rows = _json.loads(df.to_json(orient="records", force_ascii=False))
        return {"frequency": frequency, "columns": list(df.columns), "rows": rows}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取宏观宽表失败: {exc}") from exc


@router.get("/macro/dashboard")
def macro_dashboard(years: int = 3, countries: str = ""):
    """中美日宏观看板：最新值、变化、同比与近 N 年序列。"""
    if years < 1 or years > 10:
        years = 3
    country_list = [c.strip().lower() for c in countries.split(",") if c.strip()]
    try:
        return _get_store().dashboard(years=years, countries=country_list or None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取宏观看板失败: {exc}") from exc


@router.get("/macro/regions/indicators")
def macro_region_indicators():
    """分省指标目录。"""
    try:
        return {"items": _get_store().region_indicators()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取分省指标失败: {exc}") from exc


@router.get("/macro/regions/{indicator_id}")
def macro_region_table(indicator_id: str, history_years: int = 5):
    """分省对比：最新值、同比变化、排名与排名变化。"""
    if history_years < 2 or history_years > 10:
        history_years = 5
    try:
        return _get_store().region_table(indicator_id, history_years=history_years)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取分省数据失败: {exc}") from exc


@router.get("/macro/status")
def macro_status():
    """宏观库状态与最近一次刷新。"""
    try:
        return _get_store().status()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取宏观库状态失败: {exc}") from exc


@router.post("/macro/refresh")
def macro_refresh(sources: str = "akshare,cn_extra,nbs,pbc,usjp,worldbank", export: bool = True):
    """刷新宏观数据（同步执行，约数秒）。"""
    try:
        result = refresh_macro(
            sources=[s.strip() for s in sources.split(",") if s.strip()],
            store=_get_store(),
            export=export,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"刷新宏观数据失败: {exc}") from exc
