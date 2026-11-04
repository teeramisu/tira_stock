"""Market breadth primitives."""

from __future__ import annotations

import numpy as np


def market_breadth_bullish(
    closes_by_symbol: dict[str, np.ndarray],
    *,
    prev_return_weight: float = 0.05,
) -> tuple[bool, float]:
    """
    Pick strongest 1-day return; require today>0 and today > weight×yesterday.

    Returns:
        (allows_risk, best_today_return)
    """
    best_today = float("-inf")
    best_prev = 0.0
    for closes in closes_by_symbol.values():
        if closes.size < 3:
            continue
        c = closes.astype(float)
        prev_ret = float(c[1] / c[0] - 1.0) if c[0] > 0 else 0.0
        today_ret = float(c[2] / c[1] - 1.0) if c[1] > 0 else 0.0
        if today_ret > best_today:
            best_today = today_ret
            best_prev = prev_ret
    if not np.isfinite(best_today):
        return False, 0.0
    allows = best_today > 0.0 and best_today > prev_return_weight * best_prev
    return allows, best_today


def breadth_best_return(closes_by_symbol: dict[str, np.ndarray]) -> float:
    """Return the strongest 1-day return across symbols (needs >=2 bars)."""
    best = float("-inf")
    for closes in closes_by_symbol.values():
        if closes.size < 2:
            continue
        c = closes.astype(float)
        if c[-2] <= 0:
            continue
        ret = float(c[-1] / c[-2] - 1.0)
        if ret > best:
            best = ret
    return best if np.isfinite(best) else 0.0
