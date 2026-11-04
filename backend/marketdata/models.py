"""Typed models for the local market data warehouse."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


SCHEMA_VERSION = "marketdata_v1"


@dataclass(frozen=True)
class Instrument:
    """Security master record.

    Args:
        market: Market code, for example ``cn`` or ``us``.
        symbol: Market-local symbol.
        name: Display name.
        exchange: Exchange code such as ``SSE`` or ``NASDAQ``.
        asset_type: Asset class, for example ``stock`` or ``etf``.
        currency: Trading currency.
        list_date: First listing date if known.
        delist_date: Delisting date if known.
        is_active: Whether the instrument is currently active.
        is_st: Whether the instrument is ST/*ST at the current source snapshot.
        source: Source adapter name.
    """

    market: str
    symbol: str
    name: str
    exchange: str
    asset_type: str = "stock"
    currency: str = "CNY"
    list_date: date | None = None
    delist_date: date | None = None
    is_active: bool = True
    is_st: bool = False
    source: str = "fixture"
    updated_at: datetime | None = None

    @property
    def instrument_id(self) -> str:
        """Return stable global instrument identifier."""
        return f"{self.market}:{self.symbol}"


@dataclass(frozen=True)
class DailyBar:
    """Daily OHLCV bar with raw and adjusted prices."""

    market: str
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    pre_close: float | None = None
    adj_open: float | None = None
    adj_high: float | None = None
    adj_low: float | None = None
    adj_close: float | None = None
    adjustment: str = "qfq"
    volume: float | None = None
    amount: float | None = None
    adj_factor: float | None = None
    paused: bool = False
    limit_up: float | None = None
    limit_down: float | None = None
    source: str = "fixture"
    updated_at: datetime | None = None


@dataclass(frozen=True)
class OptionChainSnapshot:
    """US equity option chain row at a snapshot date (typically end-of-day pull).

    Note:
        Free yfinance exposes the *current* chain for listed expiries; historical
        chain-as-of past dates is not reliably available without a paid vendor.
        ``snapshot_date`` records when the row was captured for reproducibility.
    """

    market: str
    snapshot_date: date
    underlying: str
    expiry: date
    strike: float
    contract_right: str
    contract_symbol: str
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    implied_volatility: float | None = None
    in_the_money: bool | None = None
    currency: str = "USD"
    source: str = "yfinance"
    updated_at: datetime | None = None


@dataclass(frozen=True)
class IntradayBar:
    """Intraday OHLCV bar keyed by market, symbol, time, and frequency."""

    market: str
    symbol: str
    bar_time: datetime
    trade_date: date
    frequency: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    amount: float | None = None
    source: str = "fixture"
    updated_at: datetime | None = None


@dataclass(frozen=True)
class TradingCalendarDay:
    """Single calendar row for a market (typically exchange-wide holidays embedded as is_open)."""

    market: str
    trade_date: date
    is_open: bool


@dataclass(frozen=True)
class TradingSession:
    """Market trading session definition."""

    market: str
    exchange: str
    session_type: str
    start_time: str
    end_time: str
    timezone: str
    is_auction: bool = False


@dataclass(frozen=True)
class DatasetVersion:
    """Dataset snapshot metadata for reproducible backtests."""

    dataset_version: str
    schema_version: str = SCHEMA_VERSION
    markets: tuple[str, ...] = ("cn",)
    source: str = "fixture"
    created_at: datetime | None = None
    min_trade_date: date | None = None
    max_trade_date: date | None = None
    instrument_count: int = 0
    daily_bar_count: int = 0
    intraday_bar_count: int = 0
    quality_status: str = "unknown"
    quality_summary: dict[str, Any] = field(default_factory=dict)
    is_stable: bool = False


@dataclass(frozen=True)
class RefreshRun:
    """Refresh job record for observability and recovery."""

    id: str
    dataset_version: str
    job_type: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    symbols_total: int = 0
    symbols_success: int = 0
    symbols_failed: int = 0
    error: str | None = None

