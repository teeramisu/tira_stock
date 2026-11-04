"""Registry 驱动的数据刷新服务。

按 fallback 链依次尝试数据源，记录每个源的成功/失败，
写入更新日志，并更新 registry 中的源状态。
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

from marketdata.registry import DataSourceRegistry, SourceStatus, get_registry

logger = logging.getLogger(__name__)


# ── 每种数据类型对应的刷新函数注册表 ──────────────────────
# key = data_type, value = callable(source_id, **kwargs) -> dict | None
# 返回 {"records": int, "details": str} 或 None 表示该源不可用

_REFRESH_HANDLERS: dict[str, dict[str, Any]] = {}


def register_handler(data_type: str, handler):
    """注册某个 data_type 的刷新处理函数。"""
    _REFRESH_HANDLERS[data_type] = handler


def get_supported_data_types() -> list[dict[str, Any]]:
    """返回所有支持刷新的数据类型及其源信息。"""
    reg = get_registry()
    result = []
    for entry in reg.all_entries():
        info = entry.to_dict()
        info["refreshable"] = entry.data_type in _REFRESH_HANDLERS
        result.append(info)
    return result


def refresh_data(
    data_type: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    limit: int = 20,
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """按 fallback 链刷新指定数据类型。

    依次尝试每个源，直到某个源成功。记录调用链和结果到更新日志。

    Returns:
        {
            "ok": bool,
            "data_type": str,
            "source_used": str,        # 最终成功的源 ID
            "source_chain": [...],     # 每个源的尝试结果
            "records": int,            # 写入记录数
            "summary": str,            # 一句话摘要
            "details": str,            # 详细信息
        }
    """
    from services.update_log import log_update

    reg = get_registry()
    entry = reg.get_entry(data_type)
    if not entry:
        return {"ok": False, "error": f"未知的数据类型: {data_type}", "source_chain": []}

    handler = _REFRESH_HANDLERS.get(data_type)
    if not handler:
        return {"ok": False, "error": f"数据类型 {data_type} 暂不支持在线刷新", "source_chain": []}

    # 解析日期
    end_date = date.fromisoformat(end) if end else date.today()
    start_date = date.fromisoformat(start) if start else end_date - __import__("datetime").timedelta(days=30)

    # 按 fallback 链依次尝试
    chain_log: list[str] = []
    sources = entry.sources  # 按注册顺序（primary 在前）

    for src in sources:
        source_id = src.id
        source_name = src.name
        logger.info("尝试数据源 %s (%s) for %s", source_id, source_name, data_type)
        chain_log.append(f"{source_name}(trying)")

        try:
            kwargs = {
                "source_id": source_id,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "symbols": symbols or [],
                "limit": limit,
                **(extra or {}),
            }
            result = handler(**kwargs)

            if result is None:
                # 该源返回 None 表示不可用
                reg.update_source_status(data_type, source_id, SourceStatus.DOWN, "handler returned None")
                chain_log[-1] = f"{source_name}(failed:无数据)"
                logger.warning("数据源 %s 返回空结果", source_id)
                continue

            # 成功
            reg.update_source_status(data_type, source_id, SourceStatus.ACTIVE)
            chain_log[-1] = f"{source_name}(success)"

            records = result.get("records", 0)
            summary = result.get("summary", f"{data_type} 从 {source_name} 刷新成功，{records} 条记录")
            details = result.get("details", "")

            # 写更新日志
            market = entry.market
            log_update(
                market=market,
                data_type=data_type,
                source_used=source_id,
                summary=summary,
                symbols=symbols[:20] if symbols else None,
                records_added=records,
                date_range={"start": str(start_date), "end": str(end_date)},
                details=details,
                source_chain=chain_log,
            )

            return {
                "ok": True,
                "data_type": data_type,
                "source_used": source_id,
                "source_name": source_name,
                "source_chain": chain_log,
                "records": records,
                "summary": summary,
                "details": details,
            }

        except Exception as exc:
            error_msg = str(exc)[:200]
            reg.update_source_status(data_type, source_id, SourceStatus.DOWN, error_msg)
            chain_log[-1] = f"{source_name}(failed:{error_msg[:50]})"
            logger.warning("数据源 %s 失败: %s", source_id, error_msg)
            continue

    # 所有源都失败
    error_summary = f"{data_type} 所有 {len(sources)} 个数据源均失败"
    log_update(
        market=entry.market,
        data_type=data_type,
        source_used="none",
        summary=error_summary,
        symbols=symbols[:10] if symbols else None,
        date_range={"start": str(start_date), "end": str(end_date)},
        details=f"调用链: {' → '.join(chain_log)}",
        source_chain=chain_log,
    )

    return {
        "ok": False,
        "data_type": data_type,
        "source_used": None,
        "source_chain": chain_log,
        "records": 0,
        "summary": error_summary,
        "details": f"所有源失败。链路: {' → '.join(chain_log)}",
    }


# ── 注册各数据类型的刷新 handler ──────────────────────

def _register_cn_daily_handler():
    """A 股日线刷新 handler。"""
    def handler(source_id, start, end, symbols, limit, **kw):
        from datetime import date as _d
        start_d = _d.fromisoformat(start)
        end_d = _d.fromisoformat(end)

        from marketdata.db import connect
        from marketdata.repository import MarketDataRepository
        from marketdata.refresh import refresh_daily_bars_with_retries
        from marketdata.sources.akshare_source import (
            load_cn_daily_bars_for_symbol,
            load_cn_instruments,
        )

        repo = MarketDataRepository(connect())

        # 获取 symbol 列表
        if not symbols:
            instruments = load_cn_instruments(limit=limit)
            symbols = [item.symbol for item in instruments[:limit]]

        if not symbols:
            return None

        # 写入 instruments
        from marketdata.sources.akshare_source import load_cn_instruments_for_symbols
        instruments = load_cn_instruments_for_symbols(symbols)
        repo.upsert_instruments(instruments)

        # 根据 source_id 选择不同的加载方式
        if source_id == "akshare_em":
            loader = lambda sym: load_cn_daily_bars_for_symbol(sym, start_d, end_d)
            source_label = "akshare(东方财富)"
        elif source_id == "akshare_tx":
            # 腾讯接口通过 akshare 的 stock_zh_a_hist_tx
            def _load_tx(sym):
                from marketdata.sources.akshare_source import _fetch_cn_daily_with_fallbacks
                import akshare as ak
                from marketdata.sources.akshare_source import _akshare_daily_frame_to_bars
                start_s = start_d.strftime("%Y%m%d")
                end_s = end_d.strftime("%Y%m%d")
                normalized = str(sym).zfill(6)
                if hasattr(ak, "stock_zh_a_hist_tx"):
                    from marketdata.sources.akshare_source import _call_with_timeout, _normalize_daily_columns
                    df = _call_with_timeout(
                        lambda: ak.stock_zh_a_hist_tx(symbol=normalized, start_date=start_s, end_date=end_s)
                    )
                    if df is not None and not df.empty:
                        return _akshare_daily_frame_to_bars(normalized, _normalize_daily_columns(df), source="akshare_tx")
                return []
            loader = _load_tx
            source_label = "akshare(腾讯)"
        elif source_id == "tencent_direct":
            def _load_tencent(sym):
                from marketdata.sources.akshare_source import _fetch_cn_daily_tencent, _akshare_daily_frame_to_bars
                start_s = start_d.strftime("%Y%m%d")
                end_s = end_d.strftime("%Y%m%d")
                normalized = str(sym).zfill(6)
                df = _fetch_cn_daily_tencent(normalized, start_s, end_s)
                if df is not None and not df.empty:
                    return _akshare_daily_frame_to_bars(normalized, df, source="tencent")
                return []
            loader = _load_tencent
            source_label = "腾讯直连"
        else:
            return None

        summary = refresh_daily_bars_with_retries(
            repo, symbols=symbols, loader=loader, max_attempts=2,
        )

        return {
            "records": summary.bars_written,
            "summary": f"A股 {len(symbols)} 只标的日线更新至 {end_d}，{summary.bars_written} 条记录（源: {source_label}）",
            "details": f"成功={summary.symbols_success}, 失败={summary.symbols_failed}",
        }
    register_handler("cn_daily", handler)


def _register_us_daily_handler():
    """美股日线刷新 handler。"""
    def handler(source_id, start, end, symbols, limit, **kw):
        from datetime import date as _d
        start_d = _d.fromisoformat(start)
        end_d = _d.fromisoformat(end)

        from marketdata.db import connect
        from marketdata.repository import MarketDataRepository
        from marketdata.refresh import refresh_daily_bars_with_retries
        from marketdata.sources.yfinance_source import (
            load_us_daily_bars,
            load_us_instruments,
        )

        repo = MarketDataRepository(connect())

        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "SPY", "QQQ"][:limit]

        # 写入 instruments
        repo.upsert_instruments(load_us_instruments(symbols))

        # 映射 source_id 到 provider
        provider_map = {
            "yfinance": "yfinance",
            "stooq": "stooq",
            "akshare_us": "akshare",
        }
        provider = provider_map.get(source_id)
        if not provider:
            return None

        loader = lambda sym: load_us_daily_bars([sym], start_d, end_d, provider=provider)
        source_label = {"yfinance": "yfinance", "stooq": "Stooq", "akshare_us": "AkShare美股"}.get(source_id, source_id)

        summary = refresh_daily_bars_with_retries(
            repo, symbols=symbols, loader=loader, max_attempts=2,
        )

        return {
            "records": summary.bars_written,
            "summary": f"美股 {len(symbols)} 只标的日线更新至 {end_d}，{summary.bars_written} 条记录（源: {source_label}）",
            "details": f"成功={summary.symbols_success}, 失败={summary.symbols_failed}",
        }
    register_handler("us_daily", handler)


def _register_cn_instruments_handler():
    """A 股 Instrument 元数据刷新。"""
    def handler(source_id, start, end, symbols, limit, **kw):
        from marketdata.db import connect
        from marketdata.repository import MarketDataRepository
        repo = MarketDataRepository(connect())

        if source_id == "akshare_a_code_name":
            from marketdata.sources.akshare_source import load_cn_instruments
            instruments = load_cn_instruments(limit=limit or 20)
        elif source_id == "akshare_sh_sz":
            from marketdata.sources.akshare_source import _load_cn_instruments_without_bse
            import akshare as ak
            df = _load_cn_instruments_without_bse(ak)
            from marketdata.sources.akshare_source import _normalize_instrument_columns
            from marketdata.models import Instrument
            instruments = []
            for row in df.head(limit or 20).to_dict(orient="records"):
                code = str(row.get("code", "")).zfill(6)
                exchange = "SSE" if code.startswith(("6", "9")) else "SZSE"
                instruments.append(Instrument(
                    market="cn", symbol=code, name=str(row.get("name", "")),
                    exchange=exchange, asset_type="stock", currency="CNY",
                    is_st="ST" in str(row.get("name", "")).upper(), source="akshare",
                ))
        elif source_id == "local_db":
            # 本地 DB 不需要刷新，直接返回现有数量
            rows = repo.conn.execute("SELECT count(*) FROM instruments WHERE market='cn'").fetchone()
            count = rows[0] if rows else 0
            if count == 0:
                return None
            return {
                "records": 0,
                "summary": f"本地 DuckDB 已有 {count} 条 A 股 instrument 记录",
                "details": "local_db 无需刷新",
            }
        else:
            return None

        repo.upsert_instruments(instruments)
        return {
            "records": len(instruments),
            "summary": f"A股 instrument 元数据刷新 {len(instruments)} 条（源: {source_id}）",
            "details": "",
        }
    register_handler("cn_instruments", handler)


def _register_us_instruments_handler():
    """美股 Instrument 元数据刷新。"""
    def handler(source_id, start, end, symbols, limit, **kw):
        from marketdata.db import connect
        from marketdata.repository import MarketDataRepository
        repo = MarketDataRepository(connect())

        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"][:limit]

        if source_id == "akshare_us_spot":
            from marketdata.sources.yfinance_source import load_us_instruments
            instruments = load_us_instruments(symbols)
        elif source_id == "yfinance_info":
            from marketdata.sources.yfinance_source import load_us_instruments
            instruments = load_us_instruments(symbols)
        elif source_id == "local_db":
            rows = repo.conn.execute("SELECT count(*) FROM instruments WHERE market='us'").fetchone()
            count = rows[0] if rows else 0
            if count == 0:
                return None
            return {
                "records": 0,
                "summary": f"本地 DuckDB 已有 {count} 条美股 instrument 记录",
                "details": "local_db 无需刷新",
            }
        else:
            return None

        repo.upsert_instruments(instruments)
        return {
            "records": len(instruments),
            "summary": f"美股 instrument 元数据刷新 {len(instruments)} 条（源: {source_id}）",
            "details": "",
        }
    register_handler("us_instruments", handler)


# ── 初始化：注册所有 handler ──
def _init_handlers():
    try:
        _register_cn_daily_handler()
    except Exception as e:
        logger.warning("Failed to register cn_daily handler: %s", e)
    try:
        _register_us_daily_handler()
    except Exception as e:
        logger.warning("Failed to register us_daily handler: %s", e)
    try:
        _register_cn_instruments_handler()
    except Exception as e:
        logger.warning("Failed to register cn_instruments handler: %s", e)
    try:
        _register_us_instruments_handler()
    except Exception as e:
        logger.warning("Failed to register us_instruments handler: %s", e)

_init_handlers()
