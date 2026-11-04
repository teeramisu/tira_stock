"""ETF cross-section ranking helpers (jq-0020 family)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EtfMomentumRow:
    symbol: str
    period_return_pct: float
    ma_premium_pct: float
    two_day_return_pct: float
    today_return_pct: float


def build_etf_momentum_row(
    symbol: str,
    closes: np.ndarray,
    *,
    period_bars: int = 13,
) -> EtfMomentumRow | None:
    """Build jq-0020 style momentum metrics from daily closes."""
    if closes.size < period_bars:
        return None
    c = closes.astype(float)
    if np.any(c <= 0):
        return None
    period_ret = (c[-1] / c[0] - 1.0) * 100.0
    ma_n = float(np.mean(c))
    ma_premium = (c[-1] / ma_n - 1.0) * 100.0 if ma_n > 0 else 0.0
    two_day = (c[-1] / c[-2] - 1.0) * 100.0 if c.size >= 2 and c[-2] > 0 else 0.0
    today = (c[-1] / c[-3] - 1.0) * 100.0 if c.size >= 3 and c[-3] > 0 else two_day
    return EtfMomentumRow(
        symbol=symbol,
        period_return_pct=float(period_ret),
        ma_premium_pct=float(ma_premium),
        two_day_return_pct=float(two_day),
        today_return_pct=float(today),
    )


def etf_passes_jq0020_filters(
    row: EtfMomentumRow,
    *,
    max_three_day_pct: float = 10.6,
    max_period_pct: float = 31.0,
) -> bool:
    """Return True when row meets jq-0020 buy candidate filters."""
    return bool(
        row.period_return_pct >= 1.0
        and row.ma_premium_pct >= 0.1
        and -max_three_day_pct * 0.5 < row.two_day_return_pct < max_three_day_pct
        and -max_three_day_pct * 0.2 < row.today_return_pct < max_three_day_pct * 0.6
        and max_period_pct * 0.1 <= row.period_return_pct < max_period_pct
    )


def select_best_etf_momentum(
    rows: list[EtfMomentumRow],
    *,
    max_three_day_pct: float = 10.6,
    max_period_pct: float = 31.0,
) -> EtfMomentumRow | None:
    """Pick highest period-return ETF that passes jq-0020 filters."""
    candidates = [
        row
        for row in rows
        if etf_passes_jq0020_filters(
            row,
            max_three_day_pct=max_three_day_pct,
            max_period_pct=max_period_pct,
        )
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.period_return_pct, reverse=True)
    return candidates[0]


def market_breadth_clear_signal(rows: list[EtfMomentumRow], *, green_red_ratio: float = 1.34) -> bool:
    """True when jq-0020 would clear to cash (weak cross-ETF breadth)."""
    if not rows:
        return True
    period_sum = sum(row.period_return_pct for row in rows)
    red = sum(1 for row in rows if row.two_day_return_pct > 0)
    green = len(rows) - red
    return bool(period_sum < 0.1 and green > red * green_red_ratio)
