"""DuckDB schema management for the local market data warehouse."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .models import SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_DB_PATH = PROJECT_ROOT / "data" / "market" / "market_data.duckdb"


def connect(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and initialize schema.

    Args:
        db_path: Optional DuckDB file path. ``None`` uses the project default.

    Returns:
        Initialized DuckDB connection.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_MARKET_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    initialize_schema(conn)
    return conn


def initialize_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create market data tables if they do not exist.

    Args:
        conn: Open DuckDB connection.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO schema_metadata (key, value, updated_at)
        VALUES ('schema_version', ?, now())
        """,
        [SCHEMA_VERSION],
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instruments (
            instrument_id TEXT PRIMARY KEY,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            exchange TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            currency TEXT NOT NULL,
            list_date DATE,
            delist_date DATE,
            is_active BOOLEAN NOT NULL,
            is_st BOOLEAN NOT NULL,
            source TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            UNIQUE (market, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date DATE NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            pre_close DOUBLE,
            adj_open DOUBLE,
            adj_high DOUBLE,
            adj_low DOUBLE,
            adj_close DOUBLE,
            adjustment TEXT NOT NULL,
            volume DOUBLE,
            amount DOUBLE,
            adj_factor DOUBLE,
            paused BOOLEAN NOT NULL,
            limit_up DOUBLE,
            limit_down DOUBLE,
            source TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (market, symbol, trade_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS intraday_bars (
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            bar_time TIMESTAMP NOT NULL,
            trade_date DATE NOT NULL,
            frequency TEXT NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume DOUBLE,
            amount DOUBLE,
            source TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (market, symbol, bar_time, frequency)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trading_calendar (
            trade_date DATE NOT NULL,
            market TEXT NOT NULL,
            is_open BOOLEAN NOT NULL,
            PRIMARY KEY (market, trade_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trading_sessions (
            market TEXT NOT NULL,
            exchange TEXT NOT NULL,
            session_type TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            timezone TEXT NOT NULL,
            is_auction BOOLEAN NOT NULL,
            PRIMARY KEY (market, exchange, session_type, start_time, end_time)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            dataset_version TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            markets TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            min_trade_date DATE,
            max_trade_date DATE,
            instrument_count BIGINT NOT NULL,
            daily_bar_count BIGINT NOT NULL,
            intraday_bar_count BIGINT NOT NULL,
            quality_status TEXT NOT NULL,
            quality_summary TEXT NOT NULL,
            is_stable BOOLEAN NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_runs (
            id TEXT PRIMARY KEY,
            dataset_version TEXT NOT NULL,
            job_type TEXT NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            status TEXT NOT NULL,
            symbols_total BIGINT NOT NULL,
            symbols_success BIGINT NOT NULL,
            symbols_failed BIGINT NOT NULL,
            error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS option_chain_snapshots (
            market TEXT NOT NULL,
            snapshot_date DATE NOT NULL,
            underlying TEXT NOT NULL,
            expiry DATE NOT NULL,
            strike DOUBLE NOT NULL,
            contract_right TEXT NOT NULL,
            contract_symbol TEXT NOT NULL,
            last_price DOUBLE,
            bid DOUBLE,
            ask DOUBLE,
            volume DOUBLE,
            open_interest DOUBLE,
            implied_volatility DOUBLE,
            in_the_money BOOLEAN,
            currency TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (market, contract_symbol, snapshot_date)
        )
        """
    )

