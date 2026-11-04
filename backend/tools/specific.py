"""Agent tools — thin wrappers over existing services."""

from __future__ import annotations

from typing import Any

from .base import BaseTool


class NewsTool(BaseTool):
    """Read market news from scraping service."""

    @property
    def name(self) -> str:
        return "news"

    @property
    def description(self) -> str:
        return "Read latest CN A-share or US stock market news"

    def execute(self, market: str = "cn", limit: int = 10, **_: Any) -> list[dict[str, Any]]:
        try:
            from services.news import get_cn_a_share_news, get_us_stock_news
        except ImportError:
            return []

        fetcher = get_cn_a_share_news if market == "cn" else get_us_stock_news
        result = fetcher()
        items = result.get("items", []) if isinstance(result, dict) else []
        return items[:limit]


class IndustryTool(BaseTool):
    """Read industry/concept board data."""

    @property
    def name(self) -> str:
        return "industry"

    @property
    def description(self) -> str:
        return "Read A-share industry/concept board heat data"

    def execute(self, board_type: str = "industry", limit: int = 10, **_: Any) -> list[dict[str, Any]]:
        try:
            from services.industry_heat import get_hot_boards
        except ImportError:
            return []

        result = get_hot_boards(board_type=board_type, limit=limit, force_refresh=False)
        return result.get("items", [])[:limit]


class MarketDataTool(BaseTool):
    """Read OHLCV and universe from DuckDB or fetch_ohlcv."""

    @property
    def name(self) -> str:
        return "market_data"

    @property
    def description(self) -> str:
        return "Read OHLCV data or tradable universe"

    def execute(self, ticker: str | None = None, market: str = "cn", **_: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            from marketdata.repository import MarketDataRepository
            from marketdata.db import connect as connect_market_db
        except ImportError:
            return {"error": "repository not available"}

        conn = connect_market_db()
        repo = MarketDataRepository(conn)

        if ticker:
            bars = repo.read_daily_bars(market=market, symbol=ticker)
            result["ticker"] = ticker
            result["bars_count"] = len(bars)
            if bars:
                latest = bars[-1]
                result["latest_close"] = latest.close
                result["latest_date"] = str(latest.trade_date)

        return result


class TechnicalTool(BaseTool):
    """Compute technical indicators."""

    @property
    def name(self) -> str:
        return "technical"

    @property
    def description(self) -> str:
        return "Compute technical indicators (MA, RSI, MACD, etc.)"

    def execute(self, **_: Any) -> str:
        return "TechnicalTool: use factors/primitives/ for indicator computation"
