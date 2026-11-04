"""Point-in-time data access facade for strategies and engines."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Sequence

import pandas as pd

from .repository import MarketDataRepository
from .versioning import require_stable_dataset_version


class DataPortal:
    """Single read interface for market data.

    Strategies should use this class instead of reading DuckDB tables directly.
    The portal is bound to a stable dataset version and enforces a simple
    end-time boundary for history reads.
    """

    def __init__(
        self,
        repository: MarketDataRepository,
        dataset_version: str,
        as_of: date | datetime,
    ) -> None:
        """Initialize the data portal.

        Args:
            repository: Market data repository.
            dataset_version: Stable dataset version id.
            as_of: Current simulation date or timestamp.

        Raises:
            UnstableDatasetVersion: If ``dataset_version`` is not stable.
        """
        self.repository = repository
        self.dataset_version = require_stable_dataset_version(repository, dataset_version)
        self.as_of = as_of

    def history(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        fields: Sequence[str],
        end_time: date | datetime,
        bars: int,
        frequency: str = "1d",
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Read historical bars with a point-in-time end boundary.

        Args:
            market: Market code.
            symbols: Symbols to read.
            fields: Requested output fields.
            end_time: Last visible date or timestamp.
            bars: Number of bars to return per symbol.
            frequency: Bar frequency, such as ``1d`` or ``5m``.
            adjusted: Whether daily OHLC should use adjusted prices.

        Returns:
            DataFrame containing requested fields and key columns.

        Raises:
            ValueError: If ``end_time`` is after the portal ``as_of`` or if
                ``bars`` is invalid.
        """
        if bars <= 0:
            raise ValueError("bars must be positive")
        self._validate_fields_for_point_in_time(fields)
        if self._after_as_of(end_time):
            raise ValueError("DataPortal cannot read data after as_of")
        if frequency != "1d" and isinstance(self.as_of, datetime):
            end_dt = self._as_datetime(end_time)
            if end_dt >= self.as_of:
                raise ValueError("DataPortal cannot read an unclosed intraday bar")
        if frequency == "1d":
            end_date = end_time if isinstance(end_time, date) and not isinstance(end_time, datetime) else end_time.date()
            df = self.repository.read_daily_bars_tail(
                market=market,
                symbols=symbols,
                end=end_date,
                bars=bars,
                adjusted=adjusted,
            )
            return self._select_fields(df, fields, time_col="trade_date")

        end_dt = self._as_datetime(end_time)
        start_dt = datetime(1900, 1, 1)
        df = self.repository.read_intraday_bars(
            market=market,
            symbols=symbols,
            start_time=start_dt,
            end_time=end_dt,
            frequency=frequency,
            fields=fields,
        )
        df = df.groupby("symbol", group_keys=False).tail(bars)
        return self._select_fields(df, fields, time_col="bar_time")

    def current(
        self,
        *,
        market: str,
        symbols: Sequence[str],
        fields: Sequence[str],
        frequency: str = "1d",
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Read the latest visible bar for each symbol."""
        return self.history(
            market=market,
            symbols=symbols,
            fields=fields,
            end_time=self.as_of,
            bars=1,
            frequency=frequency,
            adjusted=adjusted,
        )

    def _after_as_of(self, value: date | datetime) -> bool:
        if isinstance(self.as_of, datetime):
            if isinstance(value, date) and not isinstance(value, datetime):
                return value > self.as_of.date()
            return self._as_datetime(value) > self.as_of
        value_date = value.date() if isinstance(value, datetime) else value
        return value_date > self.as_of

    @staticmethod
    def _as_datetime(value: date | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.combine(value, time.max)

    @staticmethod
    def _select_fields(df: pd.DataFrame, fields: Sequence[str], *, time_col: str) -> pd.DataFrame:
        base_cols = ["market", "symbol", time_col]
        requested = [field for field in fields if field in df.columns]
        return df[base_cols + requested].copy()

    @staticmethod
    def _validate_fields_for_point_in_time(fields: Sequence[str]) -> None:
        """Reject unsupported non-price fields until known/effective-date modeling is ready."""
        allowed = {"Open", "High", "Low", "Close", "Volume", "amount"}
        unsupported = [field for field in fields if field not in allowed]
        if unsupported:
            raise ValueError(
                "Fields require known_date/effective_date modeling before strategy use: "
                + ",".join(unsupported)
            )

