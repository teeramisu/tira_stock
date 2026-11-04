"""Discovery Loop 的第一批可进化信号原语（纯因果、向量化）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

SignalFactory = Callable[..., pd.Series]


def _as_price(prices: pd.Series) -> pd.Series:
    return pd.Series(prices).astype(float).sort_index()


def sma_signal(prices: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    p = _as_price(prices)
    pos = (p.rolling(int(fast)).mean() > p.rolling(int(slow)).mean()).astype(float)
    return pos.fillna(0.0)


def ema_signal(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    p = _as_price(prices)
    pos = (
        p.ewm(span=int(fast), adjust=False).mean()
        > p.ewm(span=int(slow), adjust=False).mean()
    ).astype(float)
    return pos.fillna(0.0)


def rsi_signal(
    prices: pd.Series,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    p = _as_price(prices)
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / int(period), adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / int(period), adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    pos = pd.Series(np.nan, index=p.index)
    pos[rsi < float(oversold)] = 1.0
    pos[rsi > float(overbought)] = 0.0
    return pos.ffill().fillna(0.0)


def macd_signal(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    p = _as_price(prices)
    dif = p.ewm(span=int(fast), adjust=False).mean() - p.ewm(span=int(slow), adjust=False).mean()
    dea = dif.ewm(span=int(signal), adjust=False).mean()
    return (dif > dea).astype(float).fillna(0.0)


def bollinger_signal(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    p = _as_price(prices)
    mid = p.rolling(int(period)).mean()
    sd = p.rolling(int(period)).std(ddof=0)
    lower = mid - float(num_std) * sd
    pos = pd.Series(np.nan, index=p.index)
    pos[p < lower] = 1.0   # 触及下轨买入
    pos[p > mid] = 0.0     # 回到中轨卖出
    return pos.ffill().fillna(0.0)


SIGNAL_FACTORIES: Dict[str, SignalFactory] = {
    "sma": sma_signal,
    "ema": ema_signal,
    "rsi": rsi_signal,
    "macd": macd_signal,
    "bollinger": bollinger_signal,
}


DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "sma": {"fast": [5, 10, 20], "slow": [30, 60, 120]},
    "ema": {"fast": [5, 12, 20], "slow": [26, 60, 120]},
    "rsi": {
        "period": [7, 14, 21],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
    },
    "macd": {"fast": [8, 12], "slow": [21, 26], "signal": [9]},
    "bollinger": {"period": [20, 40], "num_std": [1.5, 2.0, 2.5]},
}
