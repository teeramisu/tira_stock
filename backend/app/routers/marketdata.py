"""
行情数据、新闻、推荐相关 API。
"""
from __future__ import annotations

import os
import sys
import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# 保证可导入 backend 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.news import get_us_stock_news, get_cn_a_share_news, get_futures_news, get_hk_stock_news
from services.stock_recommendation import load_scored_data, get_top_stocks_by_sector, score_all_recommendations
from marketdata.db import DEFAULT_MARKET_DB_PATH, connect as connect_market_db
from marketdata.errors import MarketDataError, MarketDataNotFound
from marketdata.repository import MarketDataRepository
from marketdata.universe import UniverseConfig, effective_include_exchanges, get_universe
from marketdata.registry import get_registry
from services.industry_heat import get_hot_boards
from services.market_status import build_market_data_status
from services.stock_search import search_stocks
from services.update_log import get_update_log, get_update_summary
from services.refresh_service import refresh_data, get_supported_data_types

# 共享辅助函数从 backtest 路由导入
from app.routers.backtest import _market_db_path, _market_repository, _df_records

router = APIRouter(prefix="/api")


# ---------- 请求模型 ----------
class MarketRefreshRequest(BaseModel):
    """Development-only request for local market refresh endpoint."""

    market: str = Field("cn", description="市场：cn/us")
    mode: str = Field("incremental", description="刷新模式：init/incremental/status")
    frequency: str = Field("1d", description="频率：1d/1m/5m/15m/30m/60m")
    symbols: List[str] = Field(default_factory=list)
    start: Optional[date] = None
    end: Optional[date] = None
    limit: int = Field(20, ge=1, le=10000)
    max_attempts: int = Field(3, ge=1, le=10)
    full_fetch: bool = False
    retry_failed_only: bool = False
    checkpoint_path: Optional[str] = None


# ---------- 后台刷新任务状态 ----------
_refresh_tasks: dict[str, dict] = {}
_refresh_tasks_lock = threading.Lock()
_refresh_logger = logging.getLogger("market_refresh")


def _run_refresh_in_process(task_id: str, market: str):
    """在服务进程内直接执行数据刷新，避免 subprocess 导致 DuckDB 锁冲突。"""
    from marketdata.db import connect as connect_market_db
    from marketdata.refresh import (
        create_stable_version_from_current,
        refresh_daily_bars_with_retries,
        run_daily_quality_checks,
    )
    from marketdata.models import RefreshRun
    from marketdata.repository import MarketDataRepository
    from marketdata.sources.akshare_source import (
        load_cn_daily_bars_for_symbol,
        load_cn_instruments,
        load_cn_trading_calendar,
    )
    from marketdata.sources.yfinance_source import (
        load_us_daily_bars,
        load_us_instruments,
        load_us_symbols_akshare,
    )
    from services.update_log import log_update

    start_time = datetime.now()
    end_date = date.today()
    start_date = end_date - __import__('datetime').timedelta(days=120)
    refresh_run_id = f"refresh_{task_id}"
    log_lines: list[str] = []

    def _log(msg: str):
        _refresh_logger.info("[%s] %s", task_id, msg)
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        with _refresh_tasks_lock:
            _refresh_tasks[task_id]["log"] = list(log_lines)

    with _refresh_tasks_lock:
        _refresh_tasks[task_id] = {
            "status": "running", "market": market,
            "started": start_time.isoformat(), "log": [],
        }

    conn = None
    try:
        _log(f"开始刷新 {market.upper()} 数据 ({start_date} ~ {end_date})")
        conn = connect_market_db()
        repo = MarketDataRepository(conn)

        if market == "cn":
            _log("加载 A 股交易日历...")
            calendar_rows = load_cn_trading_calendar(start_date, end_date)
            repo.upsert_trading_calendar(calendar_rows)
            _log(f"交易日历更新 {len(calendar_rows)} 条")

            _log("加载 A 股股票列表...")
            instruments = load_cn_instruments(limit=100)
            repo.upsert_instruments(instruments)
            symbols = [item.symbol for item in instruments[:100]]
            _log(f"股票列表: {len(symbols)} 只")

            source = "akshare"
            loader = lambda sym: load_cn_daily_bars_for_symbol(sym, start_date, end_date)
        else:
            _log("加载美股股票列表...")
            # 优先从本地 DB 获取（快速可靠），akshare 作为补充
            rows = repo.conn.execute(
                "SELECT DISTINCT symbol FROM instruments WHERE market = 'us' ORDER BY symbol"
            ).fetchall()
            symbols = [str(r[0]).upper() for r in rows]
            if symbols:
                _log(f"从本地 DB 获取到 {len(symbols)} 只美股")
            else:
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]
                _log(f"本地 DB 为空，使用默认股票列表: {symbols}")
            symbols = symbols[:200]
            # 检查哪些已有 instrument 记录，只更新缺失的
            existing = set()
            try:
                existing_rows = repo.conn.execute(
                    "SELECT symbol FROM instruments WHERE market = 'us' AND symbol IN ({})".format(
                        ','.join(f"'{s}'" for s in symbols)
                    )
                ).fetchall()
                existing = {str(r[0]).upper() for r in existing_rows}
            except Exception:
                pass
            new_symbols = [s for s in symbols if s.upper() not in existing]
            if new_symbols:
                _log(f"更新 {len(new_symbols)} 只新股 instruments...")
                repo.upsert_instruments(load_us_instruments(new_symbols))
            else:
                _log(f"全部 {len(symbols)} 只已有 instrument，跳过更新")

            source = "us-auto"
            loader = lambda sym: load_us_daily_bars([sym], start_date, end_date, provider="auto")

        repo.upsert_refresh_run(RefreshRun(
            id=refresh_run_id, dataset_version="pending",
            job_type=f"{market}_incremental_1d",
            started_at=start_time, status="running",
            symbols_total=len(symbols),
        ))

        _log(f"开始拉取 {len(symbols)} 只 {market.upper()} 日线数据...")
        summary = refresh_daily_bars_with_retries(
            repo, symbols=symbols, loader=loader,
            max_attempts=3, checkpoint_path=None,
        )
        _log(f"日线拉取完成: 写入 {summary.bars_written} 条, "
             f"成功 {summary.symbols_success}/{summary.symbols_requested}, "
             f"失败 {summary.symbols_failed}")

        if summary.failed_symbols:
            failed_list = list(summary.failed_symbols.keys())[:10]
            _log(f"失败标的: {', '.join(failed_list)}"
                 + (f" 等{len(summary.failed_symbols)}只" if len(summary.failed_symbols) > 10 else ""))

        _log("运行数据质量检查...")
        quality_report = run_daily_quality_checks(
            repo, market=market, start=start_date, end=end_date,
            expected_symbols=len(symbols),
        )
        _log(f"质量检查完成: {quality_report}")

        _log("创建数据版本...")
        version = create_stable_version_from_current(
            repo, source=source, markets=(market,),
            quality_summary={
                "symbols_requested": summary.symbols_requested,
                "symbols_success": summary.symbols_success,
                "symbols_failed": summary.symbols_failed,
                "bars_written": summary.bars_written,
                "failed_symbols": summary.failed_symbols,
                "trigger": "button_click",
            },
        )
        _log(f"数据版本: {version.dataset_version}")

        status = "success" if summary.symbols_failed == 0 else "partial_success"
        repo.upsert_refresh_run(RefreshRun(
            id=refresh_run_id, dataset_version=version.dataset_version,
            job_type=f"{market}_incremental_1d",
            started_at=start_time, finished_at=datetime.utcnow(),
            status=status,
            symbols_total=summary.symbols_requested,
            symbols_success=summary.symbols_success,
            symbols_failed=summary.symbols_failed,
            error=None if not summary.failed_symbols else str(summary.failed_symbols),
        ))

        # 写入更新日志
        market_name = "A股" if market == "cn" else "美股"
        log_update(
            market=market, data_type=f"{market}_daily",
            source_used=source,
            summary=f"{market_name}{len(symbols)}只标的日线数据更新至{end_date}，新增{summary.bars_written}条记录",
            symbols=symbols[:20], records_added=summary.bars_written,
            date_range={"start": str(start_date), "end": str(end_date)},
            details=f"模式=incremental(按键触发), 成功={summary.symbols_success}, 失败={summary.symbols_failed}, 版本={version.dataset_version}",
            source_chain=[f"{source}(success)"],
        )
        _log(f"✅ {market_name}刷新完成: {summary.bars_written} 条新数据, 版本 {version.dataset_version}")

        end_time = datetime.now()
        with _refresh_tasks_lock:
            _refresh_tasks[task_id] = {
                "status": "done", "market": market,
                "started": start_time.isoformat(),
                "finished": end_time.isoformat(),
                "bars_written": summary.bars_written,
                "symbols_success": summary.symbols_success,
                "symbols_failed": summary.symbols_failed,
                "failed_symbols": dict(list(summary.failed_symbols.items())[:10]),
                "log": log_lines,
            }

    except Exception as exc:
        end_time = datetime.now()
        error_msg = f"{type(exc).__name__}: {exc}"
        _log(f"❌ 刷新失败: {error_msg}")

        # 记录错误日志
        try:
            market_name = "A股" if market == "cn" else "美股"
            log_update(
                market=market, data_type=f"{market}_daily",
                source_used="none",
                summary=f"{market_name}刷新失败: {error_msg}",
                details=f"错误类型={type(exc).__name__}, 详情={exc}, 触发方式=按键",
                source_chain=["failed"],
            )
        except Exception:
            pass

        with _refresh_tasks_lock:
            _refresh_tasks[task_id] = {
                "status": "failed", "market": market,
                "started": start_time.isoformat(),
                "finished": end_time.isoformat(),
                "error": error_msg,
                "log": log_lines,
            }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------- 辅助函数 ----------
def _allowed_exchanges(market: str) -> tuple[str, ...]:
    """Return allowed exchanges by market."""
    if market == "us":
        return ("NYSE", "NASDAQ", "AMEX")
    if market == "cn":
        return ("SSE", "SZSE", "BSE")
    raise HTTPException(status_code=400, detail=f"不支持的 market: {market}")


def _validate_include_exchanges(market: str, exchanges: tuple[str, ...]) -> tuple[str, ...]:
    """Validate include_exchanges against market config."""
    allowed = _allowed_exchanges(market)
    if not exchanges:
        return allowed
    invalid = sorted(set(exchanges) - set(allowed))
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"include_exchanges 包含不支持交易所: {','.join(invalid)}",
        )
    return exchanges


# ---------- 接口 ----------
@router.get("/market/supported-markets")
def supported_markets():
    """返回本地行情仓库支持的市场。"""
    return {
        "markets": [
            {"market": "cn", "name": "A 股", "exchanges": ["SSE", "SZSE", "BSE"]},
            {"market": "us", "name": "美股", "exchanges": ["NYSE", "NASDAQ", "AMEX"]},
        ]
    }


@router.get("/market/search")
def market_search(q: str = "", market: Optional[str] = None, limit: int = 10):
    """搜索股票：支持代码前缀、名称关键字、拼音首字母。"""
    if not q or not q.strip():
        return {"results": []}
    try:
        results = search_stocks(q.strip(), market=market, limit=min(limit, 50))
        return {"query": q.strip(), "count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"搜索失败: {exc}") from exc


@router.get("/market/data-status")
def market_data_status():
    """返回本地行情库状态，不触发实时下载。"""
    db_path = _market_db_path()
    if not db_path.exists():
        return {"exists": False, "db_path": str(db_path), "markets": []}
    try:
        repo = _market_repository(require_exists=True)
    except Exception as exc:
        return {
            "exists": True,
            "db_path": str(db_path),
            "markets": [],
            "warning": f"行情库当前被占用，暂时无法读取状态: {exc}",
        }
    has_daily_bars = repo.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'daily_bars'
        """
    ).fetchone()[0]
    if not has_daily_bars:
        return {
            "exists": True,
            "db_path": str(db_path),
            "markets": [],
            "warning": "daily_bars 表不存在，可能尚未初始化行情数据。",
        }
    try:
        payload = build_market_data_status(repo)
    except Exception as exc:
        return {
            "exists": True,
            "db_path": str(db_path),
            "markets": [],
            "warning": f"读取 daily_bars 失败: {exc}",
        }
    return {"exists": True, "db_path": str(db_path), **payload}


@router.get("/market/data-sources")
def market_data_sources(check_health: bool = False):
    """返回所有注册的数据源状态及 fallback 链。check_health=true 时执行实时检测。"""
    reg = get_registry()
    if check_health:
        return reg.check_all()
    return {"sources": reg.to_dict()}


@router.get("/market/update-log")
def market_update_log(market: Optional[str] = None, data_type: Optional[str] = None, limit: int = 50):
    """返回数据更新日志。"""
    entries = get_update_log(market=market, data_type=data_type, limit=limit)
    return {"entries": entries, "total": len(entries)}


@router.get("/market/update-summary")
def market_update_summary():
    """返回更新日志压缩摘要（适合 agent 上下文）。"""
    return get_update_summary()


@router.get("/market/refreshable-types")
def refreshable_types():
    """返回所有支持刷新的数据类型及其源信息。"""
    return {"types": get_supported_data_types()}


@router.post("/market/refresh-data")
async def refresh_market_data_v2(req: dict):
    """Registry 驱动的数据刷新，按 fallback 链依次尝试。"""
    data_type = req.get("data_type", "")
    if not data_type:
        raise HTTPException(status_code=400, detail="缺少 data_type 参数")

    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: refresh_data(
            data_type,
            start=req.get("start"),
            end=req.get("end"),
            symbols=req.get("symbols"),
            limit=req.get("limit", 20),
            extra=req.get("extra"),
        ),
    )
    return result


@router.get("/market/dataset-versions")
def dataset_versions(market: Optional[str] = None, stable_only: bool = True, limit: int = 20):
    """返回数据版本列表。"""
    repo = _market_repository(require_exists=True)
    clauses = []
    params: list[object] = []
    if stable_only:
        clauses.append("is_stable = true")
    if market:
        clauses.append("contains(markets, ?)")
        params.append(market)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    df = repo.conn.execute(
        f"""
        SELECT
            dataset_version, schema_version, markets, source, created_at,
            min_trade_date, max_trade_date, instrument_count, daily_bar_count,
            intraday_bar_count, quality_status, is_stable
        FROM dataset_versions
        {where_sql}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchdf()
    return {"versions": _df_records(df)}


@router.get("/market/universe")
def market_universe(
    trade_date: date,
    market: str = "cn",
    symbols: Optional[str] = None,
    include_exchanges: Optional[str] = None,
    exclude_st: bool = True,
    exclude_neeq: bool = True,
    exclude_suspended: bool = True,
    min_list_days: int = 60,
):
    """查询指定交易日股票池，不触发实时下载。"""
    repo = _market_repository(require_exists=True)
    parsed_exchanges = tuple(include_exchanges.split(",")) if include_exchanges else ()
    exchanges = effective_include_exchanges(market, _validate_include_exchanges(market, parsed_exchanges))
    config = UniverseConfig(
        market=market,
        include_exchanges=exchanges,
        symbols=tuple(symbols.split(",")) if symbols else (),
        exclude_neeq=exclude_neeq,
        exclude_st=exclude_st,
        exclude_suspended=exclude_suspended,
        min_list_days=min_list_days,
    )
    result = get_universe(repo, trade_date, config)
    return {"symbols": result.symbols, "count": len(result.symbols), "diagnostics": result.diagnostics}


@router.post("/market/refresh")
def refresh_market_data(req: MarketRefreshRequest):
    """进程内同步刷新行情数据（无 subprocess，无 DuckDB 锁冲突）。"""
    if req.mode not in {"init", "incremental", "status"}:
        raise HTTPException(status_code=400, detail=f"unsupported mode: {req.mode}")

    if req.mode == "status":
        repo = _market_repository(require_exists=True)
        rows = repo.conn.execute(
            "SELECT dataset_version, markets, min_trade_date, max_trade_date, is_stable "
            "FROM dataset_versions ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        return {"ok": True, "versions": [list(r) for r in rows]}

    # 同步执行进程内刷新
    task_id = f"sync_{req.market}_{int(time.time())}"
    _run_refresh_in_process(task_id, req.market.strip().lower())

    with _refresh_tasks_lock:
        result = _refresh_tasks.get(task_id, {})
    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail={
            "message": "refresh failed",
            "error": result.get("error", "unknown"),
            "log": result.get("log", []),
        })
    return {"ok": True, "task_id": task_id, **result}


@router.post("/market/refresh/background")
def refresh_market_background(market: str = "cn"):
    """后台拉取 A 股或美股最新数据（异步任务，立刻返回）。"""
    normalized = market.strip().lower()
    if normalized not in {"cn", "us"}:
        raise HTTPException(status_code=400, detail="market 仅支持 cn 或 us")

    task_id = f"{normalized}_{int(time.time())}"
    thread = threading.Thread(target=_run_refresh_in_process, args=(task_id, normalized), daemon=True)
    thread.start()
    return {"ok": True, "task_id": task_id, "market": normalized, "message": "拉取任务已启动"}


@router.get("/market/refresh/task-status")
def refresh_task_status(task_id: str = ""):
    """查询后台数据刷新的任务状态。"""
    if not task_id:
        with _refresh_tasks_lock:
            tasks = dict(_refresh_tasks)
        return {"tasks": tasks}
    with _refresh_tasks_lock:
        task = _refresh_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return {"task_id": task_id, **task}


# ---------- 新闻 ----------
@router.get("/news/us")
def us_stock_news():
    """美股市场新闻：聚合 WSJ / CNBC 等主流媒体 RSS。"""
    try:
        items = get_us_stock_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取美股新闻失败: {e}")


@router.get("/news/cn")
def cn_a_share_news():
    """A 股相关新闻：来源于上交所英文官网指数新闻。"""
    try:
        items = get_cn_a_share_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取 A 股新闻失败: {e}")


@router.get("/news/futures")
def futures_news():
    """期货市场新闻：优先东方财富，备选新浪财经。"""
    try:
        items = get_futures_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取期货新闻失败: {e}")


@router.get("/news/hk")
def hk_stock_news():
    """港股市场新闻：优先东方财富，备选新浪财经。"""
    try:
        items = get_hk_stock_news(limit=10)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取港股新闻失败: {e}")


# ---------- 行业板块 ----------
@router.get("/market/industries/hot")
def hot_industry_boards(
    board_type: str = "industry",
    limit: int = 30,
    min_change_pct: float = 0.0,
    refresh: bool = False,
    enrich: bool = False,
):
    """A 股热门上涨行业/概念板块（东财实时快照，5 分钟缓存）。"""
    if limit < 1 or limit > 100:
        limit = 30
    try:
        return get_hot_boards(
            board_type=board_type,
            limit=limit,
            min_change_pct=min_change_pct,
            force_refresh=refresh,
            enrich_historical=enrich,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"获取热门板块失败: {exc}") from exc


@router.get("/market/boards/{board_type}/{board_code}/constituents")
def board_constituents(board_type: str, board_code: str):
    """查询板块成份股列表。"""
    from marketdata.sources.industry_board import get_board_constituents

    normalized = board_type.strip().lower()
    if normalized not in {"industry", "concept"}:
        raise HTTPException(status_code=400, detail="board_type 仅支持 industry 或 concept")
    try:
        items = get_board_constituents(symbol=board_code, board_type=normalized)
        return {"board_type": normalized, "board_code": board_code, "count": len(items), "items": items}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"获取成份股失败: {exc}") from exc


# ---------- 股票推荐 ----------
_COMPACT_FIELDS = (
    "stock_name", "stock_code", "industry", "listed", "source_count", "author_count",
    "strong_count", "likes", "comprehensive_score", "score_reason", "score_source",
    "expected_weekly", "expected_monthly", "reasons_summary", "performance",
    "first_seen", "last_seen",
)


def _compact_stock(stock: dict) -> dict:
    """列表接口只返回展示所需字段，避免把全部 sources/reasons 都传给前端。"""
    out = {k: stock.get(k) for k in _COMPACT_FIELDS}
    out["concepts"] = (stock.get("concepts") or [])[:5]
    out["sectors"] = (stock.get("sectors") or [])[:6]
    return out


@router.get("/recommendations/top")
def get_top_recommendations(
    limit: int = 20,
    sector: str = "",
    listed_only: bool = True,
    include_details: bool = False,
):
    """获取高分股票推荐列表（知识星球挖掘，读本地缓存，不访问网络）。"""
    if limit < 1 or limit > 500:
        limit = 20
    try:
        data = load_scored_data()
        stocks = list(data.get("stocks", []))
        if listed_only:
            stocks = [s for s in stocks if s.get("listed", True)]
        if sector:
            stocks = [
                s for s in stocks
                if sector == s.get("industry") or sector in (s.get("concepts") or [])
                or sector in (s.get("sectors") or [])
            ]
        total = len(stocks)
        page = stocks[:limit]
        if not include_details:
            page = [_compact_stock(s) for s in page]
        return {
            "total": total,
            "generated_at": data.get("generated_at", ""),
            "data_as_of": data.get("data_as_of", ""),
            "llm_used": data.get("llm_used", False),
            "llm_model": data.get("llm_model", ""),
            "total_topics": data.get("total_topics", 0),
            "stocks": page,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取推荐失败: {e}")


@router.get("/recommendations/by-sector")
def get_recommendations_by_sector(
    limit: int = 30,
    per_sector: int = 5,
    include_details: bool = False,
):
    """按板块（行业）汇总高分股票。"""
    if limit < 1 or limit > 200:
        limit = 30
    if per_sector < 1 or per_sector > 30:
        per_sector = 5
    try:
        result = get_top_stocks_by_sector(limit=limit, per_sector=per_sector)
        if not include_details:
            for sec in result.get("sectors", []):
                sec["top_stocks"] = [_compact_stock(s) for s in sec.get("top_stocks", [])]
            result["top_stocks"] = [_compact_stock(s) for s in result.get("top_stocks", [])]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取板块推荐失败: {e}")


@router.post("/recommendations/refresh")
def refresh_recommendations(use_llm: bool = True, llm_top_n: int = 120):
    """重新解析语料并评分（可选大模型精评），结果写入本地缓存。"""
    if llm_top_n < 0 or llm_top_n > 300:
        llm_top_n = 120
    try:
        result = score_all_recommendations(use_llm=use_llm, llm_top_n=llm_top_n)
        return {
            "status": "ok",
            "total_stocks": result["total_stocks"],
            "total_topics": result["total_topics"],
            "llm_used": result["llm_used"],
            "llm_model": result["llm_model"],
            "data_as_of": result.get("data_as_of", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新失败: {e}")
