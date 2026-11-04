"""Repository helpers for DuckDB-backed market data."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import json
from typing import Iterable, Sequence

import duckdb
import pandas as pd

from .errors import InvalidMarketData, MarketDataNotFound
from .models import (
    DailyBar,
    DatasetVersion,
    Instrument,
    IntradayBar,
    OptionChainSnapshot,
    RefreshRun,
    TradingCalendarDay,
    TradingSession,
)


def _now_if_none(value: datetime | None) -> datetime:
    """Return a timestamp suitable for persistence."""
    return value or datetime.utcnow()


def _validate_ohlc(open_: float, high: float, low: float, close: float) -> None:
    """Validate basic OHLC invariants.

    Args:
        open_: Open price.
        high: High price.
        low: Low price.
        close: Close price.

    Raises:
        InvalidMarketData: If prices are negative or OHLC ordering is invalid.
    """
    values = [open_, high, low, close]
    if any(v < 0 for v in values):
        raise InvalidMarketData("OHLC prices must be non-negative")
    if high < low or high < open_ or high < close or low > open_ or low > close:
        raise InvalidMarketData("OHLC prices violate high/low bounds")


class MarketDataRepository:
    """Persistence API for market data tables."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Initialize repository.

        Args:
            conn: Initialized DuckDB connection.
        """
        self.conn = conn

    def upsert_instruments(self, instruments: Iterable[Instrument]) -> int:
        """Insert or replace instrument master records.

        Args:
            instruments: Instrument records.

        Returns:
            Number of records written.
        """
        rows = []
        for item in instruments:
            rows.append(
                {
                    **asdict(item),
                    "instrument_id": item.instrument_id,
                    "updated_at": _now_if_none(item.updated_at),
                }
            )
        if not rows:
            return 0
        self.conn.register("_instrument_rows", pd.DataFrame(rows))
        self.conn.execute(
            """
            DELETE FROM instruments
            WHERE instrument_id IN (SELECT instrument_id FROM _instrument_rows)
            """
        )
        self.conn.execute(
            """
            INSERT INTO instruments
            SELECT
                instrument_id, market, symbol, name, exchange, asset_type, currency,
                list_date, delist_date, is_active, is_st, source, updated_at
            FROM _instrument_rows
            """
        )
        self.conn.unregister("_instrument_rows")
        return len(rows)

    def upsert_option_chain_snapshots(self, rows: Iterable[OptionChainSnapshot]) -> int:
        """Insert or replace US option chain snapshot rows.

        Args:
            rows: Option chain rows keyed by (market, contract_symbol, snapshot_date).

        Returns:
            Number of rows written.
        """
        row_list = list(rows)
        if not row_list:
            return 0
        dicts = []
        for item in row_list:
            row = asdict(item)
            row["updated_at"] = _now_if_none(item.updated_at)
            dicts.append(row)
        self.conn.register("_option_chain_rows", pd.DataFrame(dicts))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO option_chain_snapshots
            SELECT
                market, snapshot_date, underlying, expiry, strike, contract_right,
                contract_symbol, last_price, bid, ask, volume, open_interest,
                implied_volatility, in_the_money, currency, source, updated_at
            FROM _option_chain_rows
            """
        )
        self.conn.unregister("_option_chain_rows")
        return len(dicts)

    def upsert_daily_bars(self, bars: Iterable[DailyBar]) -> int:
        """Insert or replace daily OHLCV bars.

        Args:
            bars: Daily bar records.

        Returns:
            Number of records written.

        Raises:
            InvalidMarketData: If any bar violates basic OHLC constraints.
        """
        rows = []
        for item in bars:
            _validate_ohlc(item.open, item.high, item.low, item.close)
            row = asdict(item)
            row["updated_at"] = _now_if_none(item.updated_at)
            rows.append(row)
        if not rows:
            return 0
        self.conn.register("_daily_bar_rows", pd.DataFrame(rows))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO daily_bars
            SELECT
                market, symbol, trade_date, open, high, low, close, pre_close,
                adj_open, adj_high, adj_low, adj_close, adjustment, volume, amount,
                adj_factor, paused, limit_up, limit_down, source, updated_at
            FROM _daily_bar_rows
            """
        )
        self.conn.unregister("_daily_bar_rows")
        return len(rows)

    def upsert_intraday_bars(self, bars: Iterable[IntradayBar]) -> int:
        """Insert or replace intraday OHLCV bars.

        Args:
            bars: Intraday bar records.

        Returns:
            Number of records written.
        """
        rows = []
        for item in bars:
            _validate_ohlc(item.open, item.high, item.low, item.close)
            row = asdict(item)
            row["updated_at"] = _now_if_none(item.updated_at)
            rows.append(row)
        if not rows:
            return 0
        self.conn.register("_intraday_bar_rows", pd.DataFrame(rows))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO intraday_bars
            SELECT
                market, symbol, bar_time, trade_date, frequency, open, high, low,
                close, volume, amount, source, updated_at
            FROM _intraday_bar_rows
            """
        )
        self.conn.unregister("_intraday_bar_rows")
        return len(rows)

    def upsert_trading_sessions(self, sessions: Iterable[TradingSession]) -> int:
        """Insert or replace trading session definitions."""
        rows = [asdict(item) for item in sessions]
        if not rows:
            return 0
        self.conn.register("_session_rows", pd.DataFrame(rows))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trading_sessions
            SELECT market, exchange, session_type, start_time, end_time, timezone, is_auction
            FROM _session_rows
            """
        )
        self.conn.unregister("_session_rows")
        return len(rows)

    def upsert_trading_calendar(self, days: Iterable[TradingCalendarDay]) -> int:
        """Insert or replace trading calendar rows keyed by (market, trade_date)."""
        day_list = list(days)
        if not day_list:
            return 0
        for d in day_list:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO trading_calendar (market, trade_date, is_open)
                VALUES (?, ?, ?)
                """,
                [d.market, d.trade_date, d.is_open],
            )
        return len(day_list)

    def upsert_dataset_version(self, version: DatasetVersion) -> None:
        """Insert or replace a dataset version record."""
        created_at = _now_if_none(version.created_at)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO dataset_versions (
                dataset_version, schema_version, markets, source, created_at,
                min_trade_date, max_trade_date, instrument_count, daily_bar_count,
                intraday_bar_count, quality_status, quality_summary, is_stable
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                version.dataset_version,
                version.schema_version,
                ",".join(version.markets),
                version.source,
                created_at,
                version.min_trade_date,
                version.max_trade_date,
                version.instrument_count,
                version.daily_bar_count,
                version.intraday_bar_count,
                version.quality_status,
                json.dumps(version.quality_summary, ensure_ascii=False, sort_keys=True),
                version.is_stable,
            ],
        )

    def get_dataset_version(self, dataset_version: str) -> DatasetVersion:
        """Load a dataset version by id.

        Raises:
            MarketDataNotFound: If the version is missing.
        """
        row = self.conn.execute(
            """
            SELECT
                dataset_version, schema_version, markets, source, created_at,
                min_trade_date, max_trade_date, instrument_count, daily_bar_count,
                intraday_bar_count, quality_status, quality_summary, is_stable
            FROM dataset_versions
            WHERE dataset_version = ?
            """,
            [dataset_version],
        ).fetchone()
        if row is None:
            raise MarketDataNotFound(f"Dataset version not found: {dataset_version}")
        return DatasetVersion(
            dataset_version=row[0],
            schema_version=row[1],
            markets=tuple(x for x in str(row[2]).split(",") if x),
            source=row[3],
            created_at=row[4],
            min_trade_date=row[5],
            max_trade_date=row[6],
            instrument_count=int(row[7]),
            daily_bar_count=int(row[8]),
            intraday_bar_count=int(row[9]),
            quality_status=row[10],
            quality_summary=json.loads(row[11] or "{}"),
            is_stable=bool(row[12]),
        )

    def upsert_refresh_run(self, run: RefreshRun) -> None:
        """Insert or replace a refresh run record."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO refresh_runs (
                id, dataset_version, job_type, started_at, finished_at, status,
                symbols_total, symbols_success, symbols_failed, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run.id,
                run.dataset_version,
                run.job_type,
                run.started_at,
                run.finished_at,
                run.status,
                run.symbols_total,
                run.symbols_success,
                run.symbols_failed,
                run.error,
            ],
        )

    def read_daily_bars(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Read daily bars for symbols in a date range.

        Args:
            market: Market code.
            symbols: Symbols to read.
            start: Start trade date, inclusive.
            end: End trade date, inclusive.
            adjusted: Whether to expose adjusted OHLC as Open/High/Low/Close.

        Returns:
            DataFrame sorted by symbol and trade date.

        Raises:
            MarketDataNotFound: If no rows match.
        """
        if not symbols:
            raise MarketDataNotFound("No symbols requested")
        fields = (
            "COALESCE(adj_open, open) AS Open, COALESCE(adj_high, high) AS High, "
            "COALESCE(adj_low, low) AS Low, COALESCE(adj_close, close) AS Close"
            if adjusted
            else "open AS Open, high AS High, low AS Low, close AS Close"
        )
        query = f"""
            SELECT
                market, symbol, trade_date, {fields}, volume AS Volume, amount,
                paused, limit_up, limit_down
            FROM daily_bars
            WHERE market = ?
              AND symbol IN ({",".join(["?"] * len(symbols))})
              AND trade_date BETWEEN ? AND ?
            ORDER BY symbol, trade_date
        """
        df = self.conn.execute(query, [market, *symbols, start, end]).fetchdf()
        if df.empty:
            raise MarketDataNotFound("No daily bars found for requested range")
        return df

    def read_daily_bars_tail(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        end: date,
        bars: int,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Read last N daily bars per symbol up to end date."""
        if not symbols:
            raise MarketDataNotFound("No symbols requested")
        if bars <= 0:
            raise ValueError("bars must be positive")
        fields = (
            "COALESCE(adj_open, open) AS Open, COALESCE(adj_high, high) AS High, "
            "COALESCE(adj_low, low) AS Low, COALESCE(adj_close, close) AS Close"
            if adjusted
            else "open AS Open, high AS High, low AS Low, close AS Close"
        )
        query = f"""
            WITH ranked AS (
                SELECT
                    market, symbol, trade_date, {fields}, volume AS Volume, amount,
                    paused, limit_up, limit_down,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
                FROM daily_bars
                WHERE market = ?
                  AND symbol IN ({",".join(["?"] * len(symbols))})
                  AND trade_date <= ?
            )
            SELECT
                market, symbol, trade_date, Open, High, Low, Close, Volume, amount,
                paused, limit_up, limit_down
            FROM ranked
            WHERE rn <= ?
            ORDER BY symbol, trade_date
        """
        df = self.conn.execute(query, [market, *symbols, end, bars]).fetchdf()
        if df.empty:
            raise MarketDataNotFound("No daily bars found for requested range")
        return df

    def read_intraday_bars(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        start_time: datetime,
        end_time: datetime,
        frequency: str,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read intraday bars for symbols in a time range."""
        if not symbols:
            raise MarketDataNotFound("No symbols requested")
        available_fields = {
            "Open": "open AS Open",
            "High": "high AS High",
            "Low": "low AS Low",
            "Close": "close AS Close",
            "Volume": "volume AS Volume",
            "amount": "amount",
        }
        selected_fields = fields or ("Open", "High", "Low", "Close", "Volume", "amount")
        projected = [available_fields[field] for field in selected_fields if field in available_fields]
        if not projected:
            projected = [available_fields["Close"]]
        query = f"""
            SELECT
                market, symbol, bar_time, trade_date, frequency,
                {",".join(projected)}
            FROM intraday_bars
            WHERE market = ?
              AND symbol IN ({",".join(["?"] * len(symbols))})
              AND bar_time BETWEEN ? AND ?
              AND frequency = ?
            ORDER BY symbol, bar_time
        """
        df = self.conn.execute(query, [market, *symbols, start_time, end_time, frequency]).fetchdf()
        if df.empty:
            raise MarketDataNotFound("No intraday bars found for requested range")
        return df

