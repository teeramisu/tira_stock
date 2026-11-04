"""Tests for portfolio strategy registry."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.imported_strategies import ADAPTED_PORTFOLIO_STRATEGIES
from portfolio.strategy_registry import (
    default_symbols_for_strategy,
    is_etf_pool_strategy,
    missing_adapted_portfolio_builders,
    resolve_portfolio_symbols,
    supported_portfolio_strategies,
)


def test_all_adapted_portfolio_strategies_have_builders():
    assert missing_adapted_portfolio_builders() == []


def test_jq_etf_polyfit_is_supported():
    assert "jq_etf_polyfit_momentum_top1" in supported_portfolio_strategies()
    assert "jq_etf_polyfit_momentum_top1" in ADAPTED_PORTFOLIO_STRATEGIES


def test_etf_strategy_resolves_default_symbol_pool():
    assert is_etf_pool_strategy("jq_etf_polyfit_momentum_top1")
    assert not is_etf_pool_strategy("momentum_top_k")
    defaults = default_symbols_for_strategy("jq_etf_polyfit_momentum_top1")
    assert "510300" in defaults
    resolved = resolve_portfolio_symbols("jq_etf_polyfit_momentum_top1", [])
    assert resolved == defaults
    explicit = resolve_portfolio_symbols("jq_etf_polyfit_momentum_top1", ["159915", "510300"])
    assert explicit == ("159915", "510300")


def test_stock_strategy_empty_symbols_stays_empty():
    assert resolve_portfolio_symbols("momentum_top_k", []) == ()
