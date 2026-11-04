"""Refresh orchestration helpers for the local market data warehouse."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
from typing import Callable

from .errors import InvalidMarketData
from .models import DailyBar, DatasetVersion, IntradayBar
from .repository import MarketDataRepository
from .versioning import build_dataset_version_id


DailyBarLoader = Callable[[str], list[DailyBar]]
IntradayBarLoader = Callable[[str], list[IntradayBar]]


@dataclass(frozen=True)
class DailyRefreshSummary:
    """Summary for a daily bar refresh run."""

    symbols_requested: int
    symbols_success: int
    symbols_failed: int
    bars_written: int
    failed_symbols: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IntradayRefreshSummary:
    """Summary for an intraday bar refresh run."""

    symbols_requested: int
    symbols_success: int
    symbols_failed: int
    bars_written: int
    failed_symbols: dict[str, str] = field(default_factory=dict)


def run_daily_quality_checks(
    repository: MarketDataRepository,
    *,
    market: str,
    start: date,
    end: date,
    expected_symbols: int | None = None,
    coverage_threshold: float = 0.8,
) -> dict[str, object]:
    """Run basic quality checks for refreshed daily bars.

    Args:
        repository: Market data repository.
        market: Target market code.
        start: Refresh start date.
        end: Refresh end date.
        expected_symbols: Optional expected symbol count for coverage checks.
        coverage_threshold: Coverage alert threshold on latest trade date.

    Returns:
        Quality check summary.

    Raises:
        InvalidMarketData: If fatal OHLC or duplicate-key violations are found.
    """
    row = repository.conn.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            SUM(CASE WHEN open < 0 OR high < 0 OR low < 0 OR close < 0 THEN 1 ELSE 0 END) AS negative_ohlc,
            SUM(CASE WHEN high < low OR high < open OR high < close OR low > open OR low > close THEN 1 ELSE 0 END) AS invalid_ohlc,
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS missing_volume
        FROM daily_bars
        WHERE market = ? AND trade_date BETWEEN ? AND ?
        """,
        [market, start, end],
    ).fetchone()
    duplicate_rows = repository.conn.execute(
        """
        SELECT COALESCE(SUM(cnt - 1), 0)
        FROM (
            SELECT COUNT(*) AS cnt
            FROM daily_bars
            WHERE market = ? AND trade_date BETWEEN ? AND ?
            GROUP BY symbol, trade_date
            HAVING COUNT(*) > 1
        ) t
        """,
        [market, start, end],
    ).fetchone()[0]
    latest_trade_date = repository.conn.execute(
        """
        SELECT MAX(trade_date)
        FROM daily_bars
        WHERE market = ?
        """,
        [market],
    ).fetchone()[0]
    if expected_symbols is None:
        expected_symbols = int(
            repository.conn.execute(
                "SELECT COUNT(*) FROM instruments WHERE market = ?",
                [market],
            ).fetchone()[0]
        )
    covered_symbols = 0
    if latest_trade_date is not None:
        covered_symbols = int(
            repository.conn.execute(
                """
                SELECT COUNT(DISTINCT symbol)
                FROM daily_bars
                WHERE market = ? AND trade_date = ?
                """,
                [market, latest_trade_date],
            ).fetchone()[0]
        )
    coverage_ratio = float(covered_symbols / expected_symbols) if expected_symbols > 0 else 1.0
    missing_volume = int(row[3] or 0)
    summary = {
        "row_count": int(row[0] or 0),
        "negative_ohlc": int(row[1] or 0),
        "invalid_ohlc": int(row[2] or 0),
        "missing_volume": missing_volume,
        "missing_volume_allowed": True,
        "missing_volume_recorded": True,
        "missing_volume_warning": missing_volume > 0,
        "duplicate_rows": int(duplicate_rows or 0),
        "latest_trade_date": latest_trade_date.isoformat() if latest_trade_date else None,
        "expected_symbols": int(expected_symbols),
        "covered_symbols": int(covered_symbols),
        "coverage_ratio": coverage_ratio,
        "coverage_threshold": float(coverage_threshold),
        "coverage_warning": coverage_ratio < coverage_threshold,
    }
    if summary["negative_ohlc"] > 0 or summary["invalid_ohlc"] > 0 or summary["duplicate_rows"] > 0:
        raise InvalidMarketData(f"daily quality checks failed: {summary}")
    return summary


def run_intraday_quality_checks(
    repository: MarketDataRepository,
    *,
    market: str,
    start: datetime,
    end: datetime,
    frequency: str,
    expected_symbols: int | None = None,
    coverage_threshold: float = 0.8,
) -> dict[str, object]:
    """Run basic quality checks for refreshed intraday bars."""
    row = repository.conn.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            SUM(CASE WHEN open < 0 OR high < 0 OR low < 0 OR close < 0 THEN 1 ELSE 0 END) AS negative_ohlc,
            SUM(CASE WHEN high < low OR high < open OR high < close OR low > open OR low > close THEN 1 ELSE 0 END) AS invalid_ohlc,
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS missing_volume
        FROM intraday_bars
        WHERE market = ? AND frequency = ? AND bar_time BETWEEN ? AND ?
        """,
        [market, frequency, start, end],
    ).fetchone()
    duplicate_rows = repository.conn.execute(
        """
        SELECT COALESCE(SUM(cnt - 1), 0)
        FROM (
            SELECT COUNT(*) AS cnt
            FROM intraday_bars
            WHERE market = ? AND frequency = ? AND bar_time BETWEEN ? AND ?
            GROUP BY symbol, bar_time, frequency
            HAVING COUNT(*) > 1
        ) t
        """,
        [market, frequency, start, end],
    ).fetchone()[0]
    session_count = int(
        repository.conn.execute(
            "SELECT COUNT(*) FROM trading_sessions WHERE market = ?",
            [market],
        ).fetchone()[0]
    )
    outside_session_rows = 0
    if session_count > 0:
        outside_session_rows = int(
            repository.conn.execute(
                """
                SELECT COUNT(*)
                FROM intraday_bars b
                LEFT JOIN instruments i
                  ON b.market = i.market AND b.symbol = i.symbol
                WHERE b.market = ? AND b.frequency = ? AND b.bar_time BETWEEN ? AND ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM trading_sessions s
                      WHERE s.market = b.market
                        AND s.exchange = COALESCE(i.exchange, '')
                        AND CAST(b.bar_time AS TIME) BETWEEN CAST(s.start_time AS TIME) AND CAST(s.end_time AS TIME)
                  )
                """,
                [market, frequency, start, end],
            ).fetchone()[0]
        )
    latest_trade_date = repository.conn.execute(
        """
        SELECT MAX(trade_date)
        FROM intraday_bars
        WHERE market = ? AND frequency = ?
        """,
        [market, frequency],
    ).fetchone()[0]
    if expected_symbols is None:
        expected_symbols = int(
            repository.conn.execute(
                "SELECT COUNT(*) FROM instruments WHERE market = ?",
                [market],
            ).fetchone()[0]
        )
    covered_symbols = 0
    if latest_trade_date is not None:
        covered_symbols = int(
            repository.conn.execute(
                """
                SELECT COUNT(DISTINCT symbol)
                FROM intraday_bars
                WHERE market = ? AND frequency = ? AND trade_date = ?
                """,
                [market, frequency, latest_trade_date],
            ).fetchone()[0]
        )
    coverage_ratio = float(covered_symbols / expected_symbols) if expected_symbols > 0 else 1.0
    missing_volume = int(row[3] or 0)
    summary = {
        "row_count": int(row[0] or 0),
        "negative_ohlc": int(row[1] or 0),
        "invalid_ohlc": int(row[2] or 0),
        "missing_volume": missing_volume,
        "missing_volume_allowed": True,
        "missing_volume_recorded": True,
        "missing_volume_warning": missing_volume > 0,
        "duplicate_rows": int(duplicate_rows or 0),
        "outside_session_rows": int(outside_session_rows),
        "latest_trade_date": latest_trade_date.isoformat() if latest_trade_date else None,
        "expected_symbols": int(expected_symbols),
        "covered_symbols": int(covered_symbols),
        "coverage_ratio": coverage_ratio,
        "coverage_threshold": float(coverage_threshold),
        "coverage_warning": coverage_ratio < coverage_threshold,
    }
    if (
        summary["negative_ohlc"] > 0
        or summary["invalid_ohlc"] > 0
        or summary["duplicate_rows"] > 0
        or summary["outside_session_rows"] > 0
    ):
        raise InvalidMarketData(f"intraday quality checks failed: {summary}")
    return summary


def create_stable_version_from_current(
    repository: MarketDataRepository,
    *,
    source: str,
    markets: tuple[str, ...],
    quality_summary: dict | None = None,
) -> DatasetVersion:
    """Create a stable dataset version summarizing current warehouse contents.

    Args:
        repository: Market data repository.
        source: Source label for this refresh.
        markets: Markets covered by this version.
        quality_summary: Optional quality summary.

    Returns:
        Persisted stable dataset version.
    """
    created_at = datetime.utcnow()
    row = repository.conn.execute(
        """
        SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
        FROM daily_bars
        WHERE market IN ({})
        """.format(",".join(["?"] * len(markets))),
        list(markets),
    ).fetchone()
    instrument_count = repository.conn.execute(
        """
        SELECT COUNT(*)
        FROM instruments
        WHERE market IN ({})
        """.format(",".join(["?"] * len(markets))),
        list(markets),
    ).fetchone()[0]
    intraday_count = repository.conn.execute(
        """
        SELECT COUNT(*)
        FROM intraday_bars
        WHERE market IN ({})
        """.format(",".join(["?"] * len(markets))),
        list(markets),
    ).fetchone()[0]
    version = DatasetVersion(
        dataset_version=build_dataset_version_id(source, markets, created_at),
        markets=markets,
        source=source,
        created_at=created_at,
        min_trade_date=row[0],
        max_trade_date=row[1],
        instrument_count=int(instrument_count),
        daily_bar_count=int(row[2]),
        intraday_bar_count=int(intraday_count),
        quality_status="ok",
        quality_summary=quality_summary or {},
        is_stable=True,
    )
    repository.upsert_dataset_version(version)
    return version


def refresh_daily_bars_with_retries(
    repository: MarketDataRepository,
    *,
    symbols: list[str],
    loader: DailyBarLoader,
    max_attempts: int = 3,
    checkpoint_path: str | Path | None = None,
) -> DailyRefreshSummary:
    """Refresh daily bars symbol-by-symbol with retry and checkpoint support.

    Args:
        repository: Market data repository.
        symbols: Symbols to refresh.
        loader: Callable that loads bars for one symbol.
        max_attempts: Attempts per symbol before marking failed.
        checkpoint_path: Optional JSON checkpoint path for resuming.

    Returns:
        Refresh summary.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    checkpoint = load_refresh_checkpoint(checkpoint_path)
    succeeded = set(checkpoint.get("succeeded", []))
    failed_symbols: dict[str, str] = dict(checkpoint.get("failed_symbols", {}))
    bars_written = 0
    requested = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    for symbol in requested:
        if symbol in succeeded:
            continue
        last_error = ""
        for _ in range(max_attempts):
            try:
                bars = loader(symbol)
                if not bars:
                    raise RuntimeError("no bars returned")
                bars_written += repository.upsert_daily_bars(bars)
                succeeded.add(symbol)
                failed_symbols.pop(symbol, None)
                _save_checkpoint(checkpoint_path, sorted(succeeded), failed_symbols)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - record provider failures and continue.
                last_error = str(exc)
        if last_error:
            failed_symbols[symbol] = last_error
            _save_checkpoint(checkpoint_path, sorted(succeeded), failed_symbols)

    return DailyRefreshSummary(
        symbols_requested=len(requested),
        symbols_success=len(succeeded.intersection(requested)),
        symbols_failed=len(failed_symbols),
        bars_written=bars_written,
        failed_symbols=failed_symbols,
    )


def refresh_intraday_bars_with_retries(
    repository: MarketDataRepository,
    *,
    symbols: list[str],
    loader: IntradayBarLoader,
    max_attempts: int = 3,
    checkpoint_path: str | Path | None = None,
) -> IntradayRefreshSummary:
    """Refresh intraday bars symbol-by-symbol with retry and checkpoint support."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    checkpoint = load_refresh_checkpoint(checkpoint_path)
    succeeded = set(checkpoint.get("succeeded", []))
    failed_symbols: dict[str, str] = dict(checkpoint.get("failed_symbols", {}))
    bars_written = 0
    requested = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    for symbol in requested:
        if symbol in succeeded:
            continue
        last_error = ""
        for _ in range(max_attempts):
            try:
                bars = loader(symbol)
                if not bars:
                    raise RuntimeError("no bars returned")
                bars_written += repository.upsert_intraday_bars(bars)
                succeeded.add(symbol)
                failed_symbols.pop(symbol, None)
                _save_checkpoint(checkpoint_path, sorted(succeeded), failed_symbols)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        if last_error:
            failed_symbols[symbol] = last_error
            _save_checkpoint(checkpoint_path, sorted(succeeded), failed_symbols)

    return IntradayRefreshSummary(
        symbols_requested=len(requested),
        symbols_success=len(succeeded.intersection(requested)),
        symbols_failed=len(failed_symbols),
        bars_written=bars_written,
        failed_symbols=failed_symbols,
    )


def load_refresh_checkpoint(checkpoint_path: str | Path | None) -> dict:
    """Load refresh checkpoint state from JSON file."""
    if checkpoint_path is None:
        return {}
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(
    checkpoint_path: str | Path | None,
    succeeded: list[str],
    failed_symbols: dict[str, str],
) -> None:
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"succeeded": succeeded, "failed_symbols": failed_symbols},
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

