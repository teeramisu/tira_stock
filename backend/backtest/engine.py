"""
美股回测引擎：仅做多、收盘价成交。支持多策略，结果可单测验证。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# 策略标识，与 API 一致
STRATEGY_SMA = "sma_crossover"
STRATEGY_EMA = "ema_crossover"
STRATEGY_RSI = "rsi"
STRATEGY_MACD = "macd"
STRATEGY_BOLLINGER = "bollinger"
STRATEGY_BUY_HOLD = "buy_hold"


@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series
    trades: List[Trade] = field(default_factory=list)
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    n_trades: int = 0
    win_rate: float = 0.0
    initial_value: float = 0.0
    final_value: float = 0.0


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：Open/High/Low/Close/Volume"""
    df = df.copy()
    if df.index.name is None and "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    col_map = {c.lower(): c for c in df.columns}
    for std in ["open", "high", "low", "close", "volume"]:
        if std in col_map and std != col_map[std].lower():
            df.rename(columns={col_map[std]: std.capitalize()}, inplace=True)
    if "Close" not in df.columns and "close" in df.columns:
        df["Close"] = df["close"]
    if "Close" not in df.columns:
        raise ValueError("DataFrame 需要包含 Close 列")
    return df


def _run_from_position(
    df: pd.DataFrame,
    position: pd.Series,
    initial_cash: float,
    commission_rate: float,
) -> BacktestResult:
    """
    根据 0/1 持仓序列执行交易并计算指标。position 与 df 同长度，1=持多 0=空仓。
    使用前一日信号、当日收盘价成交。
    """
    df = _ensure_ohlcv(df)
    close = df["Close"].astype(float)
    n = len(close)
    # 对齐到 df.index，缺省填 0
    if not position.index.equals(df.index):
        position = position.reindex(df.index, method="ffill").fillna(0)
    pos_arr = np.nan_to_num(position.values.astype(float), nan=0.0)
    if len(pos_arr) != n:
        pos_arr = np.zeros(n)

    cash = initial_cash
    shares = 0.0
    equity_curve = []
    dates = []
    trades: List[dict] = []
    prev_pos = 0.0

    for i in range(n):
        d = df.index[i]
        c = float(close.iloc[i])
        pos = 1 if pos_arr[i] > 0.5 else 0
        dates.append(str(d.date()))

        if prev_pos == 0 and pos == 1:
            if cash > 0 and c > 0:
                commission = cash * commission_rate
                buy_value = cash - commission
                new_shares = buy_value / c
                shares = new_shares
                cash = 0.0
                trades.append({
                    "entry_date": str(d.date()),
                    "entry_price": c,
                    "exit_date": "",
                    "exit_price": np.nan,
                    "shares": new_shares,
                    "pnl": np.nan,
                    "pnl_pct": np.nan,
                })
        elif prev_pos == 1 and pos == 0:
            if shares > 0 and c > 0:
                sell_value = shares * c
                commission = sell_value * commission_rate
                cash = sell_value - commission
                entry = trades[-1]
                entry["exit_date"] = str(d.date())
                entry["exit_price"] = c
                entry_pv = entry["shares"] * entry["entry_price"]
                pnl = sell_value - commission - entry_pv
                entry["pnl"] = pnl
                entry["pnl_pct"] = (pnl / entry_pv * 100) if entry_pv else 0
                trades[-1] = entry
                shares = 0.0

        equity = cash + shares * c
        equity_curve.append(equity)
        prev_pos = pos

    equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
    trade_list = [
        Trade(
            entry_date=t["entry_date"],
            entry_price=t["entry_price"],
            exit_date=t["exit_date"],
            exit_price=t["exit_price"],
            shares=t["shares"],
            pnl=t["pnl"],
            pnl_pct=t["pnl_pct"],
        )
        for t in trades if not (isinstance(t.get("pnl"), float) and np.isnan(t["pnl"]))
    ]

    final_value = equity_curve[-1] if equity_curve else initial_cash
    total_return = (final_value - initial_cash) / initial_cash if initial_cash else 0.0
    n_days = len(equity_curve)
    if n_days >= 2:
        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        annual_return = (final_value / initial_cash) ** (252.0 / n_days) - 1.0 if initial_cash else 0.0
        volatility = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 0 else 0.0
        sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0.0
        peak = pd.Series(equity_curve).cummax()
        drawdown = (peak - pd.Series(equity_curve)) / peak.replace(0, np.nan)
        max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0
        in_dd = (pd.Series(equity_curve) < peak).astype(int)
        dd_duration = in_dd.groupby((in_dd != in_dd.shift()).cumsum()).sum()
        max_drawdown_duration = int(dd_duration.max()) if len(dd_duration) else 0
    else:
        annual_return = total_return
        volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        max_drawdown_duration = 0

    n_trades = len(trade_list)
    wins = sum(1 for t in trade_list if t.pnl > 0)
    win_rate = (wins / n_trades * 100) if n_trades else 0.0

    return BacktestResult(
        equity_curve=equity_series,
        trades=trade_list,
        total_return=round(total_return, 6),
        annual_return=round(annual_return, 6),
        volatility=round(volatility, 6),
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 6),
        max_drawdown_duration=max_drawdown_duration,
        n_trades=n_trades,
        win_rate=round(win_rate, 2),
        initial_value=initial_cash,
        final_value=round(final_value, 2),
    )


def _sma_crossover_signal(close: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
    fast_ma = close.rolling(window=fast_period, min_periods=fast_period).mean()
    slow_ma = close.rolling(window=slow_period, min_periods=slow_period).mean()
    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)
    return (prev_fast > prev_slow).astype(int)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _ema_crossover_signal(close: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
    fast_ma = _ema(close, fast_period)
    slow_ma = _ema(close, slow_period)
    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)
    return (prev_fast > prev_slow).astype(int)


def _rsi_signal(close: pd.Series, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    prev_rsi = rsi.shift(1)
    # 状态：进入超卖区买入，进入超买区卖出，否则保持
    position = pd.Series(0.0, index=close.index)
    pos = 0
    for i in range(len(close)):
        if i == 0:
            position.iloc[i] = 0
            continue
        r = prev_rsi.iloc[i]
        if np.isnan(r):
            position.iloc[i] = pos
            continue
        if r < oversold:
            pos = 1
        elif r > overbought:
            pos = 0
        position.iloc[i] = pos
    return position


def _macd_signal(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal_period)
    prev_macd = macd_line.shift(1)
    prev_signal = signal_line.shift(1)
    return (prev_macd > prev_signal).astype(int)


def _bollinger_signal(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    prev_close = close.shift(1)
    prev_lower = lower.shift(1)
    prev_upper = upper.shift(1)
    position = pd.Series(0.0, index=close.index)
    pos = 0
    for i in range(len(close)):
        if i == 0:
            continue
        c, lo, hi = prev_close.iloc[i], prev_lower.iloc[i], prev_upper.iloc[i]
        if np.isnan(lo) or np.isnan(hi):
            position.iloc[i] = pos
            continue
        if c <= lo:
            pos = 1
        elif c >= hi:
            pos = 0
        position.iloc[i] = pos
    return position


def run_sma_crossover(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    fast_period: int = 10,
    slow_period: int = 30,
    commission_rate: float = 0.0,
) -> BacktestResult:
    """双均线（SMA）金叉/死叉。"""
    df = _ensure_ohlcv(df)
    close = df["Close"].astype(float)
    if len(close) < slow_period:
        raise ValueError(f"数据长度 {len(close)} 小于慢线周期 {slow_period}")
    position = _sma_crossover_signal(close, fast_period, slow_period)
    return _run_from_position(df, position, initial_cash, commission_rate)


def run_ema_crossover(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    fast_period: int = 10,
    slow_period: int = 30,
    commission_rate: float = 0.0,
) -> BacktestResult:
    """双均线（EMA）金叉/死叉。"""
    df = _ensure_ohlcv(df)
    close = df["Close"].astype(float)
    if len(close) < slow_period:
        raise ValueError(f"数据长度 {len(close)} 小于慢线周期 {slow_period}")
    position = _ema_crossover_signal(close, fast_period, slow_period)
    return _run_from_position(df, position, initial_cash, commission_rate)


def run_rsi(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    commission_rate: float = 0.0,
) -> BacktestResult:
    """RSI 超卖买入、超买卖出。"""
    df = _ensure_ohlcv(df)
    close = df["Close"].astype(float)
    if len(close) < period:
        raise ValueError(f"数据长度 {len(close)} 小于 RSI 周期 {period}")
    position = _rsi_signal(close, period=period, oversold=oversold, overbought=overbought)
    return _run_from_position(df, position, initial_cash, commission_rate)


def run_macd(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    commission_rate: float = 0.0,
) -> BacktestResult:
    """MACD 金叉/死叉。"""
    df = _ensure_ohlcv(df)
    close = df["Close"].astype(float)
    if len(close) < slow + signal_period:
        raise ValueError(f"数据长度不足，需至少 {slow + signal_period} 日")
    position = _macd_signal(close, fast=fast, slow=slow, signal_period=signal_period)
    return _run_from_position(df, position, initial_cash, commission_rate)


def run_bollinger(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    period: int = 20,
    num_std: float = 2.0,
    commission_rate: float = 0.0,
) -> BacktestResult:
    """布林带：触及下轨买入、上轨卖出。"""
    df = _ensure_ohlcv(df)
    close = df["Close"].astype(float)
    if len(close) < period:
        raise ValueError(f"数据长度 {len(close)} 小于周期 {period}")
    position = _bollinger_signal(close, period=period, num_std=num_std)
    return _run_from_position(df, position, initial_cash, commission_rate)


def run_buy_hold(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0,
) -> BacktestResult:
    """买入持有：首日全仓买入，持有到结束。"""
    df = _ensure_ohlcv(df)
    position = pd.Series(1, index=df.index)
    return _run_from_position(df, position, initial_cash, commission_rate)


def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0,
    **params: Any,
) -> BacktestResult:
    """
    统一入口：按 strategy 调用对应策略。
    支持: sma_crossover, ema_crossover, rsi, macd, bollinger, buy_hold
    """
    strategy = (strategy or STRATEGY_SMA).strip().lower()
    if strategy == STRATEGY_SMA:
        return run_sma_crossover(
            df,
            initial_cash=initial_cash,
            fast_period=params.get("fast_period", 10),
            slow_period=params.get("slow_period", 30),
            commission_rate=commission_rate,
        )
    if strategy == STRATEGY_EMA:
        return run_ema_crossover(
            df,
            initial_cash=initial_cash,
            fast_period=params.get("fast_period", 10),
            slow_period=params.get("slow_period", 30),
            commission_rate=commission_rate,
        )
    if strategy == STRATEGY_RSI:
        return run_rsi(
            df,
            initial_cash=initial_cash,
            period=params.get("period", 14),
            oversold=params.get("oversold", 30),
            overbought=params.get("overbought", 70),
            commission_rate=commission_rate,
        )
    if strategy == STRATEGY_MACD:
        return run_macd(
            df,
            initial_cash=initial_cash,
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal_period=params.get("signal_period", 9),
            commission_rate=commission_rate,
        )
    if strategy == STRATEGY_BOLLINGER:
        return run_bollinger(
            df,
            initial_cash=initial_cash,
            period=params.get("period", 20),
            num_std=params.get("num_std", 2.0),
            commission_rate=commission_rate,
        )
    if strategy == STRATEGY_BUY_HOLD:
        return run_buy_hold(df, initial_cash=initial_cash, commission_rate=commission_rate)
    raise ValueError(f"不支持的策略: {strategy}")


# 供 API 使用的策略列表及默认参数
STRATEGIES = [
    {"id": STRATEGY_SMA, "name": "双均线(SMA)金叉死叉", "params": ["fast_period", "slow_period"]},
    {"id": STRATEGY_EMA, "name": "双均线(EMA)金叉死叉", "params": ["fast_period", "slow_period"]},
    {"id": STRATEGY_RSI, "name": "RSI 超卖超买", "params": ["period", "oversold", "overbought"]},
    {"id": STRATEGY_MACD, "name": "MACD 金叉死叉", "params": ["fast", "slow", "signal_period"]},
    {"id": STRATEGY_BOLLINGER, "name": "布林带", "params": ["period", "num_std"]},
    {"id": STRATEGY_BUY_HOLD, "name": "买入持有", "params": []},
]
