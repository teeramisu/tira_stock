"""Factor Catalog — comprehensive factor library with classifications.

Factor Hierarchy:
  By Time: short_term (1-5d) / medium_term (5-30d) / macro (30d+)
  By Market: cn / us / universal
  By Type: momentum / reversal / volatility / volume / quality / value / breadth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd


class FactorTime(Enum):
    SHORT = "short_term"    # 1-5 trading days
    MEDIUM = "medium_term"  # 5-30 trading days
    MACRO = "macro"         # 30+ trading days


class FactorMarket(Enum):
    CN = "cn"
    US = "us"
    UNIVERSAL = "universal"


class FactorType(Enum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    QUALITY = "quality"
    VALUE = "value"
    BREADTH = "breadth"
    TECHNICAL = "technical"


@dataclass
class FactorSpec:
    name: str
    description: str
    time: FactorTime
    market: FactorMarket
    ftype: FactorType
    param_lookback: int = 20
    param_extra: dict = field(default_factory=dict)
    compute: Callable | None = None


# ============================================================
# Factor computation helpers
# ============================================================

def _roc(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _std(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).std()

def _max(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).max()

def _min(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).min()

def _corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n).corr(b)

def _rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)

def _ts_sum(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).sum()


# ============================================================
# Factor definitions
# ============================================================

FACTOR_CATALOG: list[FactorSpec] = []


def register_factor(spec: FactorSpec) -> FactorSpec:
    FACTOR_CATALOG.append(spec)
    return spec


# ---- Short-term momentum ----

register_factor(FactorSpec(
    "ret_1d", "1-day return",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 1,
))
register_factor(FactorSpec(
    "ret_3d", "3-day return",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 3,
))
register_factor(FactorSpec(
    "ret_5d", "5-day return",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 5,
))

# ---- Short-term reversal ----

register_factor(FactorSpec(
    "rev_1d", "1-day reversal (negative yesterday return predicts positive today)",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.REVERSAL, 1,
))
register_factor(FactorSpec(
    "rev_intraday", "Intraday reversal: (close-open)/open",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.REVERSAL, 1,
))

# ---- Gap factors ----

register_factor(FactorSpec(
    "gap", "Gap: (open - prev_close) / prev_close",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 1,
))

# ---- Short-term volatility ----

register_factor(FactorSpec(
    "vol_5d", "5-day volatility (std of returns)",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.VOLATILITY, 5,
))
register_factor(FactorSpec(
    "vol_10d", "10-day volatility",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.VOLATILITY, 10,
))

# ---- Volume factors ----

register_factor(FactorSpec(
    "vol_ratio_5d", "5-day avg volume / 20-day avg volume",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.VOLUME, 5,
    {"denom_lookback": 20},
))
register_factor(FactorSpec(
    "turnover_5d", "5-day turnover (volume / volume_ma20)",
    FactorTime.SHORT, FactorMarket.UNIVERSAL, FactorType.VOLUME, 5,
))

# ---- Medium-term momentum ----

register_factor(FactorSpec(
    "ret_10d", "10-day return",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 10,
))
register_factor(FactorSpec(
    "ret_20d", "20-day return",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 20,
))
register_factor(FactorSpec(
    "ret_30d", "30-day return",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 30,
))

# ---- Medium-term reversal (mean reversion) ----

register_factor(FactorSpec(
    "mr_20d", "20-day mean reversion: distance from 20-day MA",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.REVERSAL, 20,
))

# ---- Trend following ----

register_factor(FactorSpec(
    "ma_cross_5_20", "5-day / 20-day MA crossover ratio",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 5,
    {"slow": 20},
))
register_factor(FactorSpec(
    "ma_cross_10_30", "10-day / 30-day MA crossover ratio",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 10,
    {"slow": 30},
))
register_factor(FactorSpec(
    "ma_cross_20_60", "20-day / 60-day MA crossover ratio",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 20,
    {"slow": 60},
))

# ---- MACD ----

register_factor(FactorSpec(
    "macd", "MACD (12,26,9) indicator",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 12,
    {"slow": 26, "signal": 9},
))
register_factor(FactorSpec(
    "macd_div", "MACD divergence: MACD - signal line",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 12,
    {"slow": 26, "signal": 9},
))

# ---- RSI ----

register_factor(FactorSpec(
    "rsi_14", "14-day RSI",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 14,
))

# ---- Bollinger Bands ----

register_factor(FactorSpec(
    "bb_position", "Bollinger Band position: (close - lower)/(upper - lower)",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 20,
    {"num_std": 2},
))

# ---- Volatility (medium) ----

register_factor(FactorSpec(
    "vol_20d", "20-day volatility",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.VOLATILITY, 20,
))
register_factor(FactorSpec(
    "vol_ratio_20_60", "20-day vol / 60-day vol",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.VOLATILITY, 20,
    {"denom_lookback": 60},
))

# ---- Volume (medium) ----

register_factor(FactorSpec(
    "vol_ratio_20d", "20-day volume ratio: vol_ma5 / vol_ma20",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.VOLUME, 5,
    {"denom_lookback": 20},
))

# ---- Macro: Long-term momentum ----

register_factor(FactorSpec(
    "ret_60d", "60-day return (~1 quarter)",
    FactorTime.MACRO, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 60,
))
register_factor(FactorSpec(
    "ret_120d", "120-day return (~half year)",
    FactorTime.MACRO, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 120,
))
register_factor(FactorSpec(
    "ret_250d", "250-day return (~1 year)",
    FactorTime.MACRO, FactorMarket.UNIVERSAL, FactorType.MOMENTUM, 250,
))

# ---- Macro: Moving Average ----

register_factor(FactorSpec(
    "ma_cross_50_200", "50-day / 200-day MA golden cross ratio",
    FactorTime.MACRO, FactorMarket.UNIVERSAL, FactorType.TECHNICAL, 50,
    {"slow": 200},
))

# ---- Macro: Volatility regimes ----

register_factor(FactorSpec(
    "vol_60d", "60-day volatility",
    FactorTime.MACRO, FactorMarket.UNIVERSAL, FactorType.VOLATILITY, 60,
))

# ---- Quality / Momentum quality ----

register_factor(FactorSpec(
    "ret_stability", "Return stability: ret_20d / vol_20d (sharpe-like)",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.QUALITY, 20,
))
register_factor(FactorSpec(
    "ret_consistency", "Percentage of positive days in last 20 days",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.QUALITY, 20,
))

# ---- Max drawdown ----

register_factor(FactorSpec(
    "max_dd_20d", "20-day max drawdown from peak",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.REVERSAL, 20,
))
register_factor(FactorSpec(
    "max_dd_60d", "60-day max drawdown",
    FactorTime.MACRO, FactorMarket.UNIVERSAL, FactorType.REVERSAL, 60,
))

# ---- Volume-price correlation ----

register_factor(FactorSpec(
    "vp_corr_20d", "Volume-price correlation over 20 days",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.VOLUME, 20,
))

# ---- Skewness / tail risk ----

register_factor(FactorSpec(
    "skew_20d", "20-day return skewness (tail risk)",
    FactorTime.MEDIUM, FactorMarket.UNIVERSAL, FactorType.VOLATILITY, 20,
))

# ---- CN-specific: limit-up probability ----

register_factor(FactorSpec(
    "cn_limit_up", "CN: distance to limit-up price",
    FactorTime.SHORT, FactorMarket.CN, FactorType.TECHNICAL, 1,
))

# ---- CN-specific: ST avoidance ----

register_factor(FactorSpec(
    "cn_st_filter", "CN: ST flag (0=normal, 1=ST)",
    FactorTime.MACRO, FactorMarket.CN, FactorType.QUALITY, 1,
))


def list_factors(
    market: str | None = None,
    time: str | None = None,
    ftype: str | None = None,
) -> list[FactorSpec]:
    """Filter factor catalog by market/time/type."""
    result = FACTOR_CATALOG
    if market:
        result = [f for f in result if f.market.value in (market, "universal")]
    if time:
        result = [f for f in result if f.time.value == time]
    if ftype:
        result = [f for f in result if f.ftype.value == ftype]
    return result


def factor_summary() -> dict:
    """Summary of factor catalog."""
    from collections import Counter
    times = Counter(f.time.value for f in FACTOR_CATALOG)
    markets = Counter(f.market.value for f in FACTOR_CATALOG)
    types = Counter(f.ftype.value for f in FACTOR_CATALOG)
    return {
        "total": len(FACTOR_CATALOG),
        "by_time": dict(times),
        "by_market": dict(markets),
        "by_type": dict(types),
    }
