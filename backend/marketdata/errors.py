"""Domain exceptions for the local market data warehouse."""

from __future__ import annotations


class MarketDataError(Exception):
    """Base class for market data domain errors."""


class MarketDataNotFound(MarketDataError):
    """Raised when requested market data is missing."""


class UnstableDatasetVersion(MarketDataError):
    """Raised when a backtest tries to read an unstable dataset version."""


class InvalidMarketData(MarketDataError):
    """Raised when market data rows violate schema or quality constraints."""

