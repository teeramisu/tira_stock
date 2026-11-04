"""Universe selection and diagnostics for portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import pandas as pd

from .repository import MarketDataRepository

US_DEFAULT_EXCHANGES = ("NYSE", "NASDAQ", "AMEX")


def effective_include_exchanges(market: str, exchanges: tuple[str, ...]) -> tuple[str, ...]:
    """Expand default US exchange filter to include legacy UNKNOWN metadata."""
    if market != "us" or not exchanges:
        return exchanges
    if set(exchanges) == set(US_DEFAULT_EXCHANGES):
        return (*exchanges, "UNKNOWN")
    return exchanges


@dataclass(frozen=True)
class UniverseConfig:
    """Configuration for point-in-time universe selection.

    Args:
        market: Market code.
        include_exchanges: Exchanges allowed in the universe.
        symbols: Optional explicit symbol list.
        exclude_neeq: Whether to exclude NEEQ instruments.
        exclude_st: Whether to exclude ST/*ST instruments.
        exclude_suspended: Whether to exclude paused instruments on trade date.
        min_list_days: Minimum calendar days since listing.
        exclude_limit_up_buy: Whether to exclude limit-up buy candidates.
        exclude_limit_down_sell: Whether to exclude limit-down sell candidates.
    """

    market: str = "cn"
    include_exchanges: tuple[str, ...] = ("SSE", "SZSE", "BSE")
    symbols: tuple[str, ...] = ()
    exclude_neeq: bool = True
    exclude_st: bool = True
    exclude_suspended: bool = True
    min_list_days: int = 60
    exclude_limit_up_buy: bool = True
    exclude_limit_down_sell: bool = True


@dataclass(frozen=True)
class UniverseResult:
    """Universe symbols and filter diagnostics."""

    symbols: list[str]
    diagnostics: dict[str, int] = field(default_factory=dict)


def get_universe(
    repository: MarketDataRepository,
    trade_date: date,
    config: UniverseConfig | None = None,
) -> UniverseResult:
    """Return filtered symbols for a trade date.

    Args:
        repository: Market data repository.
        trade_date: Point-in-time trade date.
        config: Universe selection config.

    Returns:
        Universe result with diagnostics.
    """
    cfg = config or UniverseConfig()
    rows = repository.conn.execute(
        """
        SELECT
            i.market, i.symbol, i.exchange, i.list_date, i.is_active, i.is_st,
            b.close, b.limit_up, b.limit_down, b.paused
        FROM instruments i
        LEFT JOIN daily_bars b
          ON i.market = b.market
         AND i.symbol = b.symbol
         AND b.trade_date = ?
        WHERE i.market = ?
        """,
        [trade_date, cfg.market],
    ).fetchdf()
    diagnostics: dict[str, int] = {"initial": int(len(rows))}
    if rows.empty:
        return UniverseResult(symbols=[], diagnostics=diagnostics)

    diagnostics["missing_list_date"] = int(rows["list_date"].isna().sum())
    diagnostics["missing_trade_bar"] = int(rows["paused"].isna().sum())
    rows["paused"] = rows["paused"].fillna(False)

    if cfg.symbols:
        allowed_symbols = {symbol.upper() for symbol in cfg.symbols}
        rows = rows[rows["symbol"].str.upper().isin(allowed_symbols)]
        diagnostics["after_custom_symbols"] = int(len(rows))

    exchanges = effective_include_exchanges(cfg.market, cfg.include_exchanges)
    if exchanges:
        rows = rows[rows["exchange"].isin(exchanges)]
        diagnostics["after_exchange"] = int(len(rows))

    if cfg.exclude_neeq:
        rows = rows[rows["exchange"] != "NEEQ"]
        diagnostics["after_neeq"] = int(len(rows))

    rows = rows[rows["is_active"] == True]  # noqa: E712
    diagnostics["after_active"] = int(len(rows))

    if cfg.exclude_st:
        rows = rows[rows["is_st"] == False]  # noqa: E712
        diagnostics["after_st"] = int(len(rows))

    if cfg.min_list_days > 0:
        trade_timestamp = pd.Timestamp(trade_date)
        rows = rows[
            rows["list_date"].isna()
            | ((trade_timestamp - rows["list_date"]).dt.days >= cfg.min_list_days)
        ]
        diagnostics["after_min_list_days"] = int(len(rows))

    if cfg.exclude_suspended:
        rows = rows[rows["paused"] == False]  # noqa: E712
        diagnostics["after_suspended"] = int(len(rows))

    if cfg.exclude_limit_up_buy:
        is_limit_up = rows["limit_up"].notna() & rows["close"].notna() & (rows["close"] >= rows["limit_up"])
        rows = rows[~is_limit_up]
        diagnostics["after_limit_up_buy"] = int(len(rows))

    if cfg.exclude_limit_down_sell:
        is_limit_down = (
            rows["limit_down"].notna() & rows["close"].notna() & (rows["close"] <= rows["limit_down"])
        )
        rows = rows[~is_limit_down]
        diagnostics["after_limit_down_sell"] = int(len(rows))

    symbols = sorted(rows["symbol"].astype(str).tolist())
    diagnostics["final"] = len(symbols)
    return UniverseResult(symbols=symbols, diagnostics=diagnostics)


def us_universe_config(symbols: Sequence[str] = ()) -> UniverseConfig:
    """Return default US universe config."""
    return UniverseConfig(
        market="us",
        include_exchanges=effective_include_exchanges("us", US_DEFAULT_EXCHANGES),
        symbols=tuple(symbol.upper() for symbol in symbols),
        exclude_neeq=True,
        exclude_st=False,
        exclude_suspended=True,
        min_list_days=0,
    )

