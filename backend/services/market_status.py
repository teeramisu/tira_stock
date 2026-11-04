"""Build local full-market data status payloads for API and UI."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from marketdata.errors import MarketDataNotFound
from marketdata.repository import MarketDataRepository


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    text = str(value).strip()
    return text[:10] if text else None


def latest_stable_dataset_row(repo: MarketDataRepository, market: str) -> dict[str, Any] | None:
    """Return metadata for the best stable full-market dataset version."""
    row = repo.conn.execute(
        """
        SELECT
            dataset_version,
            source,
            min_trade_date,
            max_trade_date,
            instrument_count,
            daily_bar_count,
            quality_status,
            created_at
        FROM dataset_versions
        WHERE is_stable = true
          AND contains(markets, ?)
          AND COALESCE(daily_bar_count, 0) > 0
        ORDER BY
            max_trade_date DESC NULLS LAST,
            created_at DESC,
            instrument_count DESC,
            daily_bar_count DESC
        LIMIT 1
        """,
        [market],
    ).fetchone()
    if row is None:
        return None
    return {
        "dataset_version": str(row[0]),
        "source": str(row[1] or ""),
        "min_trade_date": _iso(row[2]),
        "max_trade_date": _iso(row[3]),
        "instrument_count": int(row[4] or 0),
        "daily_bar_count": int(row[5] or 0),
        "quality_status": str(row[6] or ""),
        "created_at": str(row[7]) if row[7] is not None else None,
    }


def latest_stable_dataset_version(repo: MarketDataRepository, market: str) -> str:
    """Return dataset_version id for the best stable full-market snapshot."""
    row = latest_stable_dataset_row(repo, market)
    if row is None:
        raise MarketDataNotFound(f"没有可用的稳定全市场数据版本: {market}")
    return str(row["dataset_version"])


def build_market_data_status(repo: MarketDataRepository) -> dict[str, Any]:
    """Aggregate daily_bars coverage and latest stable dataset versions per market."""
    markets_df = repo.conn.execute(
        """
        SELECT
            market,
            COUNT(DISTINCT symbol) AS symbols,
            COUNT(*) AS daily_bars,
            MIN(trade_date) AS min_trade_date,
            MAX(trade_date) AS max_trade_date
        FROM daily_bars
        GROUP BY market
        ORDER BY market
        """
    ).fetchdf()

    markets: list[dict[str, Any]] = []
    recommended_versions: dict[str, str] = {}
    as_of: date | None = None

    for _, row in markets_df.iterrows():
        market = str(row["market"])
        max_trade = row["max_trade_date"]
        if isinstance(max_trade, pd.Timestamp):
            max_trade = max_trade.date()
        if max_trade is not None and (as_of is None or max_trade > as_of):
            as_of = max_trade

        entry: dict[str, Any] = {
            "market": market,
            "symbols": int(row["symbols"]),
            "daily_bars": int(row["daily_bars"]),
            "min_trade_date": _iso(row["min_trade_date"]),
            "max_trade_date": _iso(max_trade),
        }
        stable = latest_stable_dataset_row(repo, market)
        if stable is not None:
            entry["latest_stable"] = stable
            recommended_versions[market] = stable["dataset_version"]
        markets.append(entry)

    return {
        "exists": True,
        "markets": markets,
        "as_of": _iso(as_of),
        "recommended_versions": recommended_versions,
    }
