"""ETF 评估：多标的买入持有对比，统一区间与指标。"""
from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd

from .data import fetch_ohlcv
from backtest.engine import run_backtest, STRATEGY_BUY_HOLD

# 多标的时每只之间间隔（秒），降低 Yahoo 限流概率
_ETF_FETCH_DELAY = 3


def evaluate_etfs(
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "5y",
    market: Optional[str] = "us",
    initial_cash: float = 100_000.0,
) -> list[dict[str, Any]]:
    """
    对多个标的（ETF/股票）做买入持有评估，返回统一指标与归一化净值曲线（起点=100）。
    symbols: 至少 2 个、最多 20 个代码。
    """
    if not symbols or len(symbols) < 2:
        raise ValueError("请至少提供 2 个标的进行对比")
    if len(symbols) > 20:
        raise ValueError("最多支持 20 个标的对比")

    period = (period or "5y").strip().lower()
    if not (start and end):
        allowed = ("6mo", "1y", "2y", "5y", "10y")
        period = period if period in allowed else "5y"

    results = []
    for i, sym in enumerate(symbols):
        sym = (sym or "").strip().upper()
        if not sym:
            continue
        if i > 0:
            time.sleep(_ETF_FETCH_DELAY)
        try:
            df = fetch_ohlcv(sym, start=start, end=end, period=period, market=market or "us")
        except Exception as e:
            raise ValueError(f"获取 {sym} 数据失败: {e}") from e

        res = run_backtest(df, STRATEGY_BUY_HOLD, initial_cash=initial_cash)
        # 归一化净值：起点 100
        eq = res.equity_curve
        if eq is None or len(eq) == 0:
            raise ValueError(f"{sym} 无有效净值数据")
        base = float(eq.iloc[0])
        if base <= 0:
            base = initial_cash
        normalized = (eq.astype(float) / base * 100.0).round(4)
        equity_curve = [{"date": str(d)[:10], "value": float(normalized.iloc[i])} for i, d in enumerate(eq.index)]

        results.append({
            "symbol": sym,
            "start_date": str(df.index.min())[:10],
            "end_date": str(df.index.max())[:10],
            "total_return": round(res.total_return, 4),
            "annual_return": round(res.annual_return, 4),
            "volatility": round(res.volatility, 4),
            "sharpe_ratio": round(res.sharpe_ratio, 4),
            "max_drawdown": round(res.max_drawdown, 4),
            "final_value": round(res.final_value, 2),
            "equity_curve": equity_curve,
        })
    return results
