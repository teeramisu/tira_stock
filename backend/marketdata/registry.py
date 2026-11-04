"""数据源注册中心：集中管理所有数据源及其 fallback 链。

每种数据类型注册至少 3 个数据源（primary + 2 fallback），
启动时自动检测可用性，提供统一的状态查询接口。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SourceStatus(str, Enum):
    ACTIVE = "active"       # 正常可用
    DEGRADED = "degraded"   # 可用但有限制（限速、部分数据缺失）
    DOWN = "down"           # 不可用
    UNKNOWN = "unknown"     # 尚未检测


@dataclass
class DataSourceInfo:
    """单个数据源的元信息。"""
    id: str
    name: str
    description: str = ""
    status: SourceStatus = SourceStatus.UNKNOWN
    last_check: Optional[str] = None
    last_error: Optional[str] = None
    health_check_fn: Optional[Callable[[], bool]] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "last_check": self.last_check,
            "last_error": self.last_error,
        }


@dataclass
class DataTypeEntry:
    """一种数据类型的注册信息，包含 fallback 链。"""
    data_type: str
    name: str
    market: str  # us / cn / hk / global
    sources: list[DataSourceInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_type": self.data_type,
            "name": self.name,
            "market": self.market,
            "sources": [s.to_dict() for s in self.sources],
            "source_count": len(self.sources),
            "primary": self.sources[0].to_dict() if self.sources else None,
            "available": any(s.status == SourceStatus.ACTIVE for s in self.sources),
        }

    def get_fallback_chain(self) -> list[DataSourceInfo]:
        """按优先级返回源列表（active 优先，然后 degraded，最后 unknown）。"""
        order = {SourceStatus.ACTIVE: 0, SourceStatus.DEGRADED: 1, SourceStatus.UNKNOWN: 2, SourceStatus.DOWN: 3}
        return sorted(self.sources, key=lambda s: order.get(s.status, 99))


class DataSourceRegistry:
    """集中注册所有数据源及其 fallback 链。"""

    def __init__(self) -> None:
        self._entries: dict[str, DataTypeEntry] = {}

    # ── 注册 ──────────────────────────────────────────────

    def register(
        self,
        data_type: str,
        name: str,
        market: str,
        sources: list[dict[str, Any]],
    ) -> DataTypeEntry:
        """注册一种数据类型及其数据源列表。

        sources 格式: [{"id": ..., "name": ..., "description": ..., "health_check_fn": ...}, ...]
        """
        entry = DataTypeEntry(data_type=data_type, name=name, market=market)
        for s in sources:
            info = DataSourceInfo(
                id=s["id"],
                name=s.get("name", s["id"]),
                description=s.get("description", ""),
                health_check_fn=s.get("health_check_fn"),
            )
            entry.sources.append(info)
        self._entries[data_type] = entry
        return entry

    def update_source_status(
        self, data_type: str, source_id: str, status: SourceStatus, error: Optional[str] = None
    ) -> None:
        """更新某个数据源的状态（在 fallback 调用后更新）。"""
        entry = self._entries.get(data_type)
        if not entry:
            return
        for s in entry.sources:
            if s.id == source_id:
                s.status = status
                s.last_check = datetime.now().isoformat(timespec="seconds")
                s.last_error = error
                break

    # ── 查询 ──────────────────────────────────────────────

    def get_entry(self, data_type: str) -> Optional[DataTypeEntry]:
        return self._entries.get(data_type)

    def get_fallback_chain(self, data_type: str) -> list[DataSourceInfo]:
        entry = self._entries.get(data_type)
        if not entry:
            return []
        return entry.get_fallback_chain()

    def all_entries(self) -> list[DataTypeEntry]:
        return list(self._entries.values())

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries.values()]

    # ── 健康检测 ──────────────────────────────────────────

    def check_all(self, timeout_per_source: float = 5.0, max_workers: int = 8) -> dict[str, Any]:
        """对所有注册的源执行健康检测（并发），每个源最多等 timeout_per_source 秒。"""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

        def _run_check(data_type: str, src: DataSourceInfo) -> dict[str, Any]:
            src_result = {"id": src.id, "name": src.name, "data_type": data_type}
            if not src.health_check_fn:
                src_result["status"] = src.status.value
                return src_result
            t0 = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(src.health_check_fn)
                    ok = fut.result(timeout=timeout_per_source)
                elapsed = round(time.time() - t0, 2)
                if ok:
                    src.status = SourceStatus.ACTIVE
                    src.last_error = None
                else:
                    src.status = SourceStatus.DOWN
                    src.last_error = "health_check returned False"
                src.last_check = datetime.now().isoformat(timespec="seconds")
                src_result.update({"status": src.status.value, "elapsed_s": elapsed})
            except FutTimeout:
                src.status = SourceStatus.DOWN
                src.last_error = f"timeout ({timeout_per_source}s)"
                src.last_check = datetime.now().isoformat(timespec="seconds")
                src_result.update({"status": "down", "error": src.last_error, "elapsed_s": round(time.time() - t0, 2)})
            except Exception as exc:
                src.status = SourceStatus.DOWN
                src.last_error = str(exc)[:200]
                src.last_check = datetime.now().isoformat(timespec="seconds")
                src_result.update({"status": "down", "error": src.last_error, "elapsed_s": round(time.time() - t0, 2)})
            return src_result

        results: dict[str, Any] = {"checked_at": datetime.now().isoformat(timespec="seconds"), "entries": []}
        # 收集所有需要检测的 (entry, source) 对
        tasks = []
        for entry in self._entries.values():
            for src in entry.sources:
                tasks.append((entry, src))
        # 并发执行所有检测
        check_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for entry, src in tasks:
                fut = pool.submit(_run_check, entry.data_type, src)
                future_map[fut] = (entry.data_type, src.id)
            for fut in future_map:
                dt, sid = future_map[fut]
                key = f"{dt}:{sid}"
                try:
                    check_results[key] = fut.result(timeout=timeout_per_source + 2)
                except Exception:
                    check_results[key] = {"id": sid, "name": sid, "data_type": dt, "status": "down", "error": "check failed"}
        # 组装结果
        for entry in self._entries.values():
            entry_result: dict[str, Any] = {
                "data_type": entry.data_type,
                "name": entry.name,
                "market": entry.market,
                "sources": [],
            }
            for src in entry.sources:
                key = f"{entry.data_type}:{src.id}"
                entry_result["sources"].append(check_results.get(key, {"id": src.id, "name": src.name, "status": src.status.value}))
            results["entries"].append(entry_result)
        return results


# ── 全局单例 ──────────────────────────────────────────────

_registry: Optional[DataSourceRegistry] = None


def get_registry() -> DataSourceRegistry:
    """获取全局 DataSourceRegistry 单例（懒初始化 + 自动注册）。"""
    global _registry
    if _registry is None:
        _registry = DataSourceRegistry()
        _register_all_sources(_registry)
    return _registry


def _register_all_sources(reg: DataSourceRegistry) -> None:
    """注册所有已知数据类型及其 fallback 链（每类 5 个源）。"""
    import requests as _req

    _UA = {"User-Agent": "Mozilla/5.0"}

    # ── 通用检测工具 ──
    def _check_local_db(table: str = "daily_bars"):
        def _fn() -> bool:
            try:
                from marketdata.db import connect
                conn = connect()
                rows = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
                return rows is not None and rows[0] > 0
            except Exception:
                return False
        return _fn

    def _check_akshare_func(func_name: str, **kwargs):
        def _fn() -> bool:
            try:
                import akshare as ak
                fn = getattr(ak, func_name, None)
                if fn is None:
                    return False
                result = fn(**kwargs)
                return result is not None and not (hasattr(result, 'empty') and result.empty)
            except Exception:
                return False
        return _fn

    def _check_akshare_light(func_name: str, **kwargs):
        def _fn() -> bool:
            try:
                import akshare as ak
                fn = getattr(ak, func_name, None)
                if fn is None or not callable(fn):
                    return False
                result = fn(**kwargs) if kwargs else fn()
                return result is not None and not (hasattr(result, 'empty') and result.empty)
            except Exception:
                return False
        return _fn

    def _check_url(url: str, expect: str = "", timeout: float = 5.0):
        def _fn() -> bool:
            try:
                r = _req.get(url, headers=_UA, timeout=timeout)
                if expect:
                    return r.status_code == 200 and expect in r.text
                return r.status_code == 200 and len(r.content) > 100
            except Exception:
                return False
        return _fn

    def _check_yfinance_symbol(symbol: str):
        def _fn() -> bool:
            try:
                import yfinance as yf
                df = yf.download(symbol, period="1d", progress=False, timeout=5)
                return df is not None and not df.empty
            except Exception:
                return False
        return _fn

    # ══════════════════════════════════════════════════════════
    # 1. A 股日线 OHLCV（5 源）
    #    实测可用: 腾讯直连, AkShare东方财富(间歇), 东方财富spot
    # ══════════════════════════════════════════════════════════
    reg.register("cn_daily", "A股日线OHLCV", "cn", [
        {"id": "tencent_direct", "name": "腾讯财经直连", "description": "web.ifzq.gtimg.cn K线API",
         "health_check_fn": _check_url("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,2026-07-01,2026-07-02,1,qfq", "data")},
        {"id": "akshare_em", "name": "AkShare东方财富", "description": "akshare stock_zh_a_hist",
         "health_check_fn": _check_akshare_func("stock_zh_a_hist", symbol="000001", period="daily", start_date="20260701", end_date="20260702", adjust="")},
        {"id": "eastmoney_spot", "name": "东方财富行情", "description": "push2.eastmoney.com 实时行情",
         "health_check_fn": _check_url("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=m:0+t:6")},
        {"id": "sina_cn", "name": "新浪财经", "description": "hq.sinajs.cn 实时行情",
         "health_check_fn": _check_url("https://hq.sinajs.cn/list=sh600519")},
        {"id": "netease_cn", "name": "网易财经", "description": "quotes.money.163.com 历史数据",
         "health_check_fn": _check_url("https://quotes.money.163.com/service/chddata.html?code=0600519&start=20260701&end=20260702")},
    ])

    # ══════════════════════════════════════════════════════════
    # 2. A 股 ETF 日线（5 源）
    #    实测可用: 腾讯ETF, AkShare ETF东方财富(间歇)
    # ══════════════════════════════════════════════════════════
    reg.register("cn_etf_daily", "A股ETF日线", "cn", [
        {"id": "tencent_etf", "name": "腾讯财经ETF", "description": "腾讯 ETF K线接口",
         "health_check_fn": _check_url("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh510300,day,2026-07-01,2026-07-02,1,qfq")},
        {"id": "akshare_etf_em", "name": "AkShare ETF东方财富", "description": "akshare fund_etf_hist_em",
         "health_check_fn": _check_akshare_light("fund_etf_hist_em", symbol="510300", period="daily", start_date="20260701", end_date="20260702", adjust="")},
        {"id": "akshare_etf_sina", "name": "AkShare ETF新浪", "description": "akshare fund_etf_hist_sina",
         "health_check_fn": _check_akshare_light("fund_etf_hist_sina", symbol="sh510300")},
        {"id": "eastmoney_etf", "name": "东方财富ETF", "description": "东方财富 ETF 行情列表",
         "health_check_fn": _check_url("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=b:MK0021")},
        {"id": "sina_etf", "name": "新浪财经ETF", "description": "新浪 ETF 实时行情",
         "health_check_fn": _check_url("https://hq.sinajs.cn/list=sh510300")},
    ])

    # ══════════════════════════════════════════════════════════
    # 3. A 股 Instrument 元数据（5 源）
    #    实测可用: AkShare全量代码, 东方财富spot
    # ══════════════════════════════════════════════════════════
    reg.register("cn_instruments", "A股Instrument元数据", "cn", [
        {"id": "akshare_a_code_name", "name": "AkShare全量代码", "description": "akshare stock_info_a_code_name",
         "health_check_fn": _check_akshare_func("stock_info_a_code_name")},
        {"id": "eastmoney_spot", "name": "东方财富全量", "description": "东方财富 A 股列表接口",
         "health_check_fn": _check_url("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=m:0+t:6")},
        {"id": "akshare_sh_sz", "name": "AkShare分交易所", "description": "akshare stock_info_sh/sz_name_code",
         "health_check_fn": _check_akshare_func("stock_info_sh_name_code", symbol="上海证券交易所")},
        {"id": "tencent_cn", "name": "腾讯财经", "description": "腾讯股票行情接口",
         "health_check_fn": _check_url("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,2026-07-01,2026-07-02,1,qfq", "data")},
        {"id": "local_db", "name": "本地DuckDB", "description": "从本地 DuckDB instruments 表读取",
         "health_check_fn": _check_local_db("instruments")},
    ])

    # ══════════════════════════════════════════════════════════
    # 4. 美股日线 OHLCV（5 源）
    #    实测可用: Stooq, AkShare美股, AlphaVantage
    # ══════════════════════════════════════════════════════════
    reg.register("us_daily", "美股日线OHLCV", "us", [
        {"id": "yfinance", "name": "yfinance", "description": "Yahoo Finance yf.download",
         "health_check_fn": _check_yfinance_symbol("AAPL")},
        {"id": "stooq", "name": "Stooq", "description": "Stooq CSV 下载",
         "health_check_fn": _check_url("https://stooq.com/q/d/l/?s=aapl.us&i=d")},
        {"id": "akshare_us", "name": "AkShare美股", "description": "akshare stock_us_daily",
         "health_check_fn": _check_akshare_func("stock_us_daily", symbol="AAPL", adjust="")},
        {"id": "alphavantage", "name": "AlphaVantage", "description": "alphavantage.co TIME_SERIES_DAILY",
         "health_check_fn": _check_url("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=demo")},
        {"id": "polygon", "name": "Polygon.io", "description": "polygon.io 美股数据",
         "health_check_fn": _check_url("https://api.polygon.io/v2/aggs/ticker/AAPL/range?from=2026-07-01&to=2026-07-02")},
    ])

    # ══════════════════════════════════════════════════════════
    # 5. 美股 Instrument 元数据（5 源）
    #    实测可用: AkShare美股Spot, AlphaVantage
    # ══════════════════════════════════════════════════════════
    reg.register("us_instruments", "美股Instrument元数据", "us", [
        {"id": "akshare_us_spot", "name": "AkShare美股Spot", "description": "akshare stock_us_spot",
         "health_check_fn": _check_akshare_light("stock_us_spot")},
        {"id": "yfinance_info", "name": "yfinance Ticker.info", "description": "yfinance 批量 Ticker.info",
         "health_check_fn": _check_yfinance_symbol("AAPL")},
        {"id": "alphavantage", "name": "AlphaVantage", "description": "AlphaVantage 股票概览",
         "health_check_fn": _check_url("https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey=demo")},
        {"id": "stockanalysis", "name": "StockAnalysis", "description": "stockanalysis.com 股票列表",
         "health_check_fn": _check_url("https://stockanalysis.com/stocks/", "AAPL")},
        {"id": "local_db", "name": "本地DuckDB", "description": "从本地 DuckDB instruments 表读取",
         "health_check_fn": _check_local_db("instruments")},
    ])

    # ══════════════════════════════════════════════════════════
    # 6. 美股期权链（5 源）
    #    实测可用: CBOE, StockAnalysis
    # ══════════════════════════════════════════════════════════
    def _check_yfinance_options() -> bool:
        try:
            import yfinance as yf
            t = yf.Ticker("SPY")
            dates = t.options
            return dates is not None and len(dates) > 0
        except Exception:
            return False

    reg.register("us_options", "美股期权链", "us", [
        {"id": "yfinance_options", "name": "yfinance期权链", "description": "yf.Ticker.option_chain",
         "health_check_fn": _check_yfinance_options},
        {"id": "cboe", "name": "CBOE数据", "description": "CBOE 延迟期权报价",
         "health_check_fn": _check_url("https://www.cboe.com/delayed_quotes/spy/quote_table")},
        {"id": "stockanalysis", "name": "StockAnalysis", "description": "stockanalysis.com 期权数据",
         "health_check_fn": _check_url("https://stockanalysis.com/etf/spy/options/", "SPY")},
        {"id": "barchart", "name": "Barchart", "description": "barchart.com 期权链",
         "health_check_fn": _check_url("https://www.barchart.com/stocks/quotes/spy/options", "SPY")},
        {"id": "local_db", "name": "本地DuckDB", "description": "从本地期权数据库读取",
         "health_check_fn": _check_local_db("daily_bars")},
    ])

    # ══════════════════════════════════════════════════════════
    # 7. 港股日线 OHLCV（5 源）
    #    实测可用: AAStocks, AkShare港股(间歇)
    # ══════════════════════════════════════════════════════════
    reg.register("hk_daily", "港股日线OHLCV", "hk", [
        {"id": "yfinance_hk", "name": "yfinance港股", "description": "yfinance .HK 后缀",
         "health_check_fn": _check_yfinance_symbol("0700.HK")},
        {"id": "akshare_hk", "name": "AkShare港股", "description": "akshare stock_hk_hist",
         "health_check_fn": _check_akshare_light("stock_hk_hist", symbol="00700", period="daily", start_date="20260701", end_date="20260702", adjust="")},
        {"id": "aastocks", "name": "AAStocks", "description": "aastocks.com 港股行情",
         "health_check_fn": _check_url("http://www.aastocks.com/en/stocks/quote/quick-quote.aspx?symbol=00700")},
        {"id": "sina_hk", "name": "新浪财经港股", "description": "新浪财经港股 API",
         "health_check_fn": _check_url("https://stock.finance.sina.com.cn/hkstock/api/openapi.php/HK_StockService.getHKStockData?symbol=00700")},
        {"id": "eastmoney_hk", "name": "东方财富港股", "description": "东方财富港股行情列表",
         "health_check_fn": _check_url("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=m:128+t:3")},
    ])

    # ══════════════════════════════════════════════════════════
    # 8. ETF 估值数据（5 源）
    #    实测可用: StockAnalysis, CBOE
    # ══════════════════════════════════════════════════════════
    reg.register("etf_valuation", "ETF估值数据", "us", [
        {"id": "stockanalysis", "name": "StockAnalysis", "description": "stockanalysis.com ETF 数据",
         "health_check_fn": _check_url("https://stockanalysis.com/etf/spy/", "SPY")},
        {"id": "yfinance_info", "name": "yfinance Ticker.info", "description": "P/E, P/B, 股息率等",
         "health_check_fn": _check_yfinance_symbol("SPY")},
        {"id": "etfcom", "name": "ETF.com", "description": "etf.com ETF 分析",
         "health_check_fn": _check_url("https://www.etf.com/SPY")},
        {"id": "alphavantage", "name": "AlphaVantage", "description": "AlphaVantage 基本面数据",
         "health_check_fn": _check_url("https://www.alphavantage.co/query?function=OVERVIEW&symbol=SPY&apikey=demo")},
        {"id": "etfdb", "name": "ETFdb", "description": "etfdb.com ETF 数据库",
         "health_check_fn": _check_url("https://etfdb.com/etf/SPY/#holdings", "SPY")},
    ])

    # ══════════════════════════════════════════════════════════
    # 9. ETF 持仓数据（5 源）
    #    实测可用: StockAnalysis
    # ══════════════════════════════════════════════════════════
    reg.register("etf_holdings", "ETF持仓数据", "us", [
        {"id": "stockanalysis", "name": "StockAnalysis", "description": "stockanalysis.com ETF holdings",
         "health_check_fn": _check_url("https://stockanalysis.com/etf/spy/holdings/", "AAPL")},
        {"id": "yfinance_funds", "name": "yfinance Funds Data", "description": "yf.Ticker.funds_data.top_holdings",
         "health_check_fn": _check_yfinance_symbol("SPY")},
        {"id": "etfdb", "name": "ETFdb", "description": "etfdb.com 持仓数据",
         "health_check_fn": _check_url("https://etfdb.com/etf/SPY/#holdings", "SPY")},
        {"id": "etfcom", "name": "ETF.com", "description": "etf.com 持仓信息",
         "health_check_fn": _check_url("https://www.etf.com/SPY")},
        {"id": "wisesheets", "name": "Wisesheets", "description": "wisesheets.io ETF 持仓",
         "health_check_fn": _check_url("https://wisesheets.io/spy-holdings", "SPY")},
    ])

    # ══════════════════════════════════════════════════════════
    # 10. 新闻资讯（5 源）
    #     实测可用: 新浪财经
    # ══════════════════════════════════════════════════════════
    reg.register("news", "新闻资讯", "global", [
        {"id": "sina", "name": "新浪财经", "description": "新浪财经滚动新闻 API",
         "health_check_fn": _check_url("https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&num=1")},
        {"id": "eastmoney", "name": "东方财富", "description": "东方财富行情+资讯",
         "health_check_fn": _check_url("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=m:0+t:6")},
        {"id": "cls", "name": "财联社", "description": "财联社电报接口",
         "health_check_fn": _check_url("https://www.cls.cn/nodeapi/updateTelegraph")},
        {"id": "wallstreetcn", "name": "华尔街见闻", "description": "华尔街见闻新闻 API",
         "health_check_fn": _check_url("https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=1")},
        {"id": "akshare_news", "name": "AkShare新闻", "description": "akshare 新闻接口",
         "health_check_fn": _check_akshare_light("stock_news_em")},
    ])

    # ══════════════════════════════════════════════════════════
    # 11. A 股行业板块（5 源）
    #     实测可用: 东方财富spot(板块列表), 腾讯
    # ══════════════════════════════════════════════════════════
    reg.register("cn_industry_board", "A股行业板块", "cn", [
        {"id": "akshare_em_board", "name": "AkShare东方财富板块", "description": "akshare stock_board_industry_name_em",
         "health_check_fn": _check_akshare_light("stock_board_industry_name_em")},
        {"id": "eastmoney_board", "name": "东方财富板块", "description": "东方财富行业板块接口",
         "health_check_fn": _check_url("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&fs=m:90+t:2")},
        {"id": "akshare_ths_board", "name": "AkShare同花顺板块", "description": "akshare stock_board_industry_summary_ths",
         "health_check_fn": _check_akshare_light("stock_board_industry_summary_ths")},
        {"id": "tencent_board", "name": "腾讯财经板块", "description": "腾讯行业行情",
         "health_check_fn": _check_url("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,2026-07-01,2026-07-02,1,qfq", "data")},
        {"id": "local_db", "name": "本地DuckDB", "description": "本地缓存的板块数据",
         "health_check_fn": _check_local_db("daily_bars")},
    ])

    # ══════════════════════════════════════════════════════════
    # 12. 股票搜索（5 源）
    #     实测可用: 本地DuckDB, AkShare全量代码, 东方财富
    # ══════════════════════════════════════════════════════════
    reg.register("stock_search", "股票搜索", "global", [
        {"id": "local_db", "name": "本地DuckDB", "description": "instruments 表全量搜索",
         "health_check_fn": _check_local_db("instruments")},
        {"id": "akshare_search", "name": "AkShare搜索", "description": "akshare stock_info_a_code_name",
         "health_check_fn": _check_akshare_light("stock_info_a_code_name")},
        {"id": "eastmoney_search", "name": "东方财富搜索", "description": "东方财富 suggest 接口",
         "health_check_fn": _check_url("https://searchapi.eastmoney.com/api/suggest/get?input=000001&type=14&count=5")},
        {"id": "yfinance_search", "name": "yfinance搜索", "description": "yf.Ticker search",
         "health_check_fn": _check_yfinance_symbol("AAPL")},
        {"id": "sina_search", "name": "新浪财经搜索", "description": "新浪股票搜索接口",
         "health_check_fn": _check_url("https://suggest3.sinajs.cn/suggest/type=11,12,13,14,15&key=000001&name=suggestdata")},
    ])