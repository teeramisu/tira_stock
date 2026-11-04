"""
投资大师策略回测研究工具 (v2 - 高效向量化版本)
================================================
直接用 DuckDB + pandas 向量化计算，跳过 PortfolioEngine 开销。
实现巴菲特/索罗斯/利弗莫尔/达里奥/林奇 5 位大师策略。

Usage:
    python scripts/master_strategies_v2.py                    # 全策略对比
    python scripts/master_strategies_v2.py --strategy soros   # 单策略
    python scripts/master_strategies_v2.py --grid-search      # 参数遍历
    python scripts/master_strategies_v2.py --period 2020-2024 # 指定区间
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "market" / "market_data.duckdb"

# Global market setting (set by run_backtest)
_current_market: str = "cn"

# ─── 数据加载 ─────────────────────────────────────────────────────────

def load_price_data(
    symbols: list[str] | None = None,
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    market: str = "cn",
) -> pd.DataFrame:
    """Load daily bars from DuckDB. Returns DataFrame with columns:
    symbol, trade_date, open, high, low, close, volume, amount.
    """
    import duckdb
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    warmup_start = _shift_date(start, -400)  # 200-day warmup

    if symbols:
        sym_clause = "AND symbol IN (" + ",".join(f"'{s}'" for s in symbols) + ")"
    else:
        sym_clause = ""

    query = f"""
        SELECT symbol, trade_date,
               COALESCE(adj_open, open) as open,
               COALESCE(adj_high, high) as high,
               COALESCE(adj_low, low) as low,
               COALESCE(adj_close, close) as close,
               volume, amount
        FROM daily_bars
        WHERE market = '{market}'
          AND trade_date BETWEEN '{warmup_start}' AND '{end}'
          AND volume > 0
          {sym_clause}
        ORDER BY symbol, trade_date
    """
    df = conn.execute(query).fetchdf()
    conn.close()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def get_top_liquid_stocks(n: int = 300, market: str = "cn") -> list[str]:
    """Get top N stocks by liquidity. CN: by trading days; US: by avg volume."""
    import duckdb
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    if market == "us":
        # US: sort by avg volume to capture mega-caps
        result = conn.execute(f"""
            SELECT symbol, avg(volume) as avg_vol
            FROM daily_bars
            WHERE market = 'us' AND trade_date >= '2020-01-01' AND volume > 0
            GROUP BY symbol
            HAVING count(*) >= 500
            ORDER BY avg_vol DESC
            LIMIT {n}
        """).fetchall()
    else:
        # CN: sort by trading days (most active), exclude ETFs (1xx, 5xx)
        result = conn.execute(f"""
            SELECT symbol, count(*) as cnt
            FROM daily_bars
            WHERE market = '{market}' AND trade_date >= '2020-01-01' AND volume > 0
              AND symbol NOT LIKE '1%' AND symbol NOT LIKE '5%'
            GROUP BY symbol
            ORDER BY cnt DESC
            LIMIT {n}
        """).fetchall()
    conn.close()
    return [r[0] for r in result]


def get_etf_universe(market: str = "cn") -> list[str]:
    """Multi-asset ETF pool for Dalio strategy."""
    if market == "us":
        # Available: QQQ(nasdaq), IWM(small), TLT(long-bond), DBC(commodity),
        #            VNQ(real-estate), LQD(corp-bond), SHY(short-bond), IEF(mid-bond)
        return ["QQQ", "IWM", "TLT", "DBC", "VNQ", "LQD", "SHY", "IEF"]
    return ["510300", "510500", "159915", "513100", "518880", "510180"]


# ─── 技术指标 ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_val = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def _shift_date(date_str: str, days: int) -> str:
    from datetime import datetime, timedelta
    d = datetime.fromisoformat(date_str) + timedelta(days=days)
    return d.strftime("%Y-%m-%d")


# ─── 策略信号生成（向量化） ────────────────────────────────────────────

def compute_signals(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Pre-compute all strategy signals for all symbols on each rebalance date.

    Returns a DataFrame: trade_date x symbol x signal_type
    """
    # Filter to simulation period
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    # Pivot to wide format for vectorized computation
    pivot_close = df.pivot_table(index="trade_date", columns="symbol", values="close")
    pivot_high = df.pivot_table(index="trade_date", columns="symbol", values="high")
    pivot_low = df.pivot_table(index="trade_date", columns="symbol", values="low")
    pivot_volume = df.pivot_table(index="trade_date", columns="symbol", values="volume")

    # Only keep dates in range (with warmup)
    all_dates = pivot_close.index
    sim_dates = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]

    # Pre-compute indicators (vectorized across all symbols)
    ma20 = pivot_close.rolling(20).mean()
    ma60 = pivot_close.rolling(60).mean()
    ma120 = pivot_close.rolling(120).mean()
    ma200 = pivot_close.rolling(200).mean()

    # Daily returns
    daily_ret = pivot_close.pct_change()

    # 20-day momentum
    mom20 = pivot_close / pivot_close.shift(20) - 1
    # 60-day momentum
    mom60 = pivot_close / pivot_close.shift(60) - 1

    # 60-day volatility (annualized)
    vol60 = daily_ret.rolling(60).std() * np.sqrt(252)

    # 20-day high
    high20 = pivot_high.rolling(20).max()
    # 10-day low
    low10 = pivot_low.rolling(10).min()

    # 20-day average volume
    avg_vol20 = pivot_volume.rolling(20).mean()

    # RSI 14
    rsi14 = pivot_close.apply(lambda col: _rsi(col, 14))

    # ADX 14 (per-symbol, can't easily vectorize)
    adx14 = pd.DataFrame(index=pivot_close.index, columns=pivot_close.columns, dtype=float)
    for sym in pivot_close.columns:
        try:
            adx14[sym] = _adx(pivot_high[sym], pivot_low[sym], pivot_close[sym], 14)
        except Exception:
            pass

    # 120-day high for Lynch
    high120 = pivot_close.rolling(120).max()

    return {
        "close": pivot_close,
        "high": pivot_high,
        "low": pivot_low,
        "volume": pivot_volume,
        "ma20": ma20, "ma60": ma60, "ma120": ma120, "ma200": ma200,
        "daily_ret": daily_ret,
        "mom20": mom20, "mom60": mom60,
        "vol60": vol60,
        "high20": high20, "low10": low10,
        "avg_vol20": avg_vol20,
        "rsi14": rsi14, "adx14": adx14,
        "high120": high120,
        "sim_dates": sim_dates,
    }


# ─── 策略选股函数 ────────────────────────────────────────────────────

def buffett_select(signals: dict, date_idx: int, top_k: int = 10) -> list[tuple[str, float]]:
    """巴菲特: 上升趋势 + 安全边际回调 + 低波动护城河"""
    d = signals["sim_dates"][date_idx]
    close = signals["close"].loc[d]
    ma200 = signals["ma200"].loc[d]
    ma20 = signals["ma20"].loc[d]
    vol60 = signals["vol60"].loc[d]
    ma120 = signals["ma120"].loc[d]

    # Valid data
    valid = close.dropna().index.intersection(ma200.dropna().index).intersection(
        ma20.dropna().index).intersection(vol60.dropna().index)
    if len(valid) == 0:
        return []

    scores = {}
    for sym in valid:
        c = close[sym]
        m200 = ma200[sym]
        m20 = ma20[sym]
        v = vol60[sym]
        if c <= 0 or m200 <= 0 or m20 <= 0 or v <= 0:
            continue

        # Trend: price above MA200 and MA200 rising
        if c < m200:
            continue

        # Pullback: price near MA20 (within -10% to +2%)
        pullback = c / m20 - 1
        if pullback < -0.10 or pullback > 0.02:
            continue

        # Moat: low volatility (< median)
        if v > vol60[valid].median():
            continue

        # Score: low vol + close to MA20
        score = (1 - v) * (1 - abs(pullback))
        scores[sym] = score

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = ranked[:top_k]
    w = 1.0 / len(selected)
    return [(s, w) for s, _ in selected]


def soros_select(signals: dict, date_idx: int, top_k: int = 8, lookback: int = 20, min_adx: float = 20) -> list[tuple[str, float]]:
    """索罗斯: 强趋势 + 动量加速 + ADX确认 + 趋势过滤"""
    d = signals["sim_dates"][date_idx]
    close = signals["close"].loc[d]
    mom20 = signals["mom20"].loc[d]
    adx14 = signals["adx14"].loc[d]
    vol60 = signals["vol60"].loc[d]
    ma60 = signals["ma60"].loc[d]

    # Also need lookback-ago close for acceleration
    if date_idx < lookback:
        return []
    d_prev = signals["sim_dates"][date_idx - lookback]
    close_prev = signals["close"].loc[d_prev]
    d_prev2 = signals["sim_dates"][max(0, date_idx - lookback * 2)]
    close_prev2 = signals["close"].loc[d_prev2]

    valid = close.dropna().index.intersection(mom20.dropna().index).intersection(
        adx14.dropna().index).intersection(vol60.dropna().index).intersection(ma60.dropna().index)
    if len(valid) == 0:
        return []

    scores = {}
    for sym in valid:
        c = close[sym]
        mom = mom20[sym]
        adx_val = adx14[sym]
        v = vol60[sym]
        m60 = ma60[sym]

        if not all(np.isfinite(x) for x in [c, mom, adx_val, v, m60]):
            continue
        if c <= 0 or v <= 0:
            continue

        # Only buy above MA60 (uptrend filter for A-shares)
        if c < m60:
            continue

        # ADX threshold
        if adx_val < min_adx:
            continue

        # Momentum must be positive
        if mom <= 0:
            continue

        # Momentum acceleration
        cp = close_prev.get(sym, np.nan)
        cp2 = close_prev2.get(sym, np.nan)
        if np.isfinite(cp) and np.isfinite(cp2) and cp > 0 and cp2 > 0:
            mom_prev = cp / cp2 - 1
            accel = mom - mom_prev
        else:
            accel = 0

        # Risk-adjusted momentum
        risk_adj = mom / v if v > 0 else 0

        score = accel * 30 + (adx_val / 100) * 20 + risk_adj * 0.1 + mom * 5
        scores[sym] = max(score, 0)

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = ranked[:top_k]

    # Soros: conviction-based sizing
    total_score = sum(s for _, s in selected) or 1.0
    positions = [(s, min(0.25, max(0.05, sc / total_score))) for s, sc in selected]
    total_w = sum(w for _, w in positions)
    if total_w > 0:
        positions = [(s, w / total_w) for s, w in positions]
    return positions


def livermore_select(signals: dict, date_idx: int, top_k: int = 5) -> list[tuple[str, float]]:
    """利弗莫尔: 突破新高 + 放量 + MA趋势确认 + 止损检查"""
    d = signals["sim_dates"][date_idx]
    close = signals["close"].loc[d]
    high = signals["high"].loc[d]
    low = signals["low"].loc[d]
    volume = signals["volume"].loc[d]
    high20 = signals["high20"].loc[d]
    low10 = signals["low10"].loc[d]
    avg_vol20 = signals["avg_vol20"].loc[d]
    ma60 = signals["ma60"].loc[d]

    valid = close.dropna().index.intersection(high20.dropna().index).intersection(
        low10.dropna().index).intersection(avg_vol20.dropna().index).intersection(ma60.dropna().index)
    if len(valid) == 0:
        return []

    scores = {}
    for sym in valid:
        c = close[sym]
        h = high[sym]
        l = low[sym]
        h20 = high20[sym]
        l10 = low10[sym]
        v = volume[sym]
        av = avg_vol20[sym]
        m60 = ma60[sym]

        if not all(np.isfinite(x) for x in [c, h, h20, l10, v, av, m60]):
            continue
        if c <= 0 or av <= 0:
            continue

        # Must be above MA60 (uptrend)
        if c < m60:
            continue

        # Breakout: today's close > previous 20-day high
        if date_idx < 2:
            continue
        d_prev = signals["sim_dates"][date_idx - 1]
        prev_high20 = signals["high20"].loc[d_prev].get(sym, np.nan)
        if not np.isfinite(prev_high20) or prev_high20 <= 0:
            continue
        if c <= prev_high20:
            continue

        # Volume confirmation: > 1.3x average (relaxed from 1.5)
        if v < av * 1.3:
            continue

        # Stop check: price well above 10-day low
        if c < l10 * 1.03:
            continue

        breakout_pct = (c / prev_high20 - 1) * 100
        vol_ratio = v / av
        score = breakout_pct * vol_ratio
        scores[sym] = score

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = ranked[:top_k]
    w = 1.0 / len(selected)
    return [(s, w) for s, _ in selected]


def dalio_select(signals: dict, date_idx: int, symbols_subset: list[str] | None = None) -> list[tuple[str, float]]:
    """达里奥: 风险平价 + 动量叠加"""
    d = signals["sim_dates"][date_idx]
    close = signals["close"].loc[d]
    vol60 = signals["vol60"].loc[d]
    mom20 = signals["mom20"].loc[d]

    if symbols_subset:
        valid = pd.Index([s for s in symbols_subset if s in close.index and s in vol60.index])
    else:
        valid = close.dropna().index.intersection(vol60.dropna().index)

    valid = valid.intersection(mom20.dropna().index)
    if len(valid) == 0:
        return []

    assets = []
    for sym in valid:
        v = vol60[sym]
        mom = mom20[sym]
        c = close[sym]
        if not np.isfinite(v) or v <= 0 or not np.isfinite(c) or c <= 0:
            continue
        inv_vol = 1.0 / v
        # Momentum tilt
        mom_adj = 1 + (mom if np.isfinite(mom) else 0) * 0.3
        adjusted_weight = inv_vol * max(0.3, mom_adj)
        assets.append((sym, adjusted_weight, mom))

    if not assets:
        return []

    total_w = sum(a[1] for a in assets)
    if total_w <= 0:
        return []
    return [(s, w / total_w) for s, w, _ in assets]


def lynch_select(signals: dict, date_idx: int, top_k: int = 15) -> list[tuple[str, float]]:
    """林奇: GARP — 增长 + 合理价格 + 波动调整"""
    d = signals["sim_dates"][date_idx]
    close = signals["close"].loc[d]
    mom60 = signals["mom60"].loc[d]
    vol60 = signals["vol60"].loc[d]
    high120 = signals["high120"].loc[d]

    valid = close.dropna().index.intersection(mom60.dropna().index).intersection(
        vol60.dropna().index).intersection(high120.dropna().index)
    if len(valid) == 0:
        return []

    scores = {}
    for sym in valid:
        c = close[sym]
        growth = mom60[sym]
        v = vol60[sym]
        h120 = high120[sym]

        if not all(np.isfinite(x) for x in [c, growth, v, h120]):
            continue
        if c <= 0 or v <= 0 or h120 <= 0:
            continue

        # Must have positive growth
        if growth <= 0:
            continue

        # Reasonable price: within 10% of 120-day high
        dist_from_high = c / h120 - 1
        if dist_from_high < -0.10:
            continue

        # GARP score: growth / volatility
        garp = growth / v
        scores[sym] = garp

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = ranked[:top_k]
    w = 1.0 / len(selected)
    return [(s, w) for s, _ in selected]


def ensemble_select(signals: dict, date_idx: int, top_k: int = 10) -> list[tuple[str, float]]:
    """大师混合: 多策略投票 + 风控门控"""
    # Collect votes from all sub-strategies
    votes: dict[str, dict] = {}
    for selector_fn in [buffett_select, soros_select, livermore_select, lynch_select]:
        try:
            positions = selector_fn(signals, date_idx, top_k=top_k * 2)
            for sym, w in positions:
                if sym not in votes:
                    votes[sym] = {"count": 0, "weight_sum": 0}
                votes[sym]["count"] += 1
                votes[sym]["weight_sum"] += w
        except Exception:
            continue

    if not votes:
        return []

    # Require at least 2 strategies agreeing
    consensus = {s: v for s, v in votes.items() if v["count"] >= 2}
    if not consensus:
        consensus = votes  # fallback to all

    ranked = sorted(consensus.items(), key=lambda x: x[1]["count"] * x[1]["weight_sum"], reverse=True)
    selected = ranked[:top_k]
    w = 1.0 / len(selected)

    positions = [(s, w) for s, _ in selected]

    # Risk gate: check recent portfolio volatility
    if date_idx >= 20:
        d = signals["sim_dates"][date_idx]
        daily_ret = signals["daily_ret"]
        syms = [s for s, _ in positions]
        valid_syms = [s for s in syms if s in daily_ret.columns]
        if valid_syms:
            recent_ret = daily_ret[valid_syms].loc[:d].tail(20).mean(axis=1)
            port_vol = recent_ret.std() * np.sqrt(252)
            max_vol = 0.20
            if np.isfinite(port_vol) and port_vol > max_vol:
                scale = max_vol / port_vol
                positions = [(s, w * scale) for s, w in positions]

    return positions


def adaptive_select(signals: dict, date_idx: int, top_k: int = 10, etf_symbols: list[str] | None = None) -> list[tuple[str, float]]:
    """自适应策略: 牛市选股(巴菲特+利弗莫尔), 熊市切ETF(达里奥风险平价).

    Market regime detection (multi-timeframe):
    - Breadth: % stocks above MA60 > 50% → bullish
    - Trend: 20-day median return > -2% → not crashing
    - Crash guard: 5-day median return < -5% → emergency exit
    """
    d = signals["sim_dates"][date_idx]
    close = signals["close"]

    if date_idx < 60:
        return buffett_select(signals, date_idx, top_k=top_k)

    ma60 = signals["ma60"]
    if d not in close.index or d not in ma60.index:
        return []

    valid_syms = close.loc[d].dropna().index.intersection(ma60.loc[d].dropna().index)
    if len(valid_syms) < 10:
        return []

    above_ma60 = sum(1 for s in valid_syms if close.loc[d, s] > ma60.loc[d, s])
    breadth = above_ma60 / len(valid_syms)

    # 20-day trend
    if date_idx >= 20:
        d_20ago = signals["sim_dates"][date_idx - 20]
        avg_ret_20d = (close.loc[d, valid_syms] / close.loc[d_20ago, valid_syms] - 1).median()
    else:
        avg_ret_20d = 0

    # 5-day crash guard
    if date_idx >= 5:
        d_5ago = signals["sim_dates"][date_idx - 5]
        avg_ret_5d = (close.loc[d, valid_syms] / close.loc[d_5ago, valid_syms] - 1).median()
    else:
        avg_ret_5d = 0

    # Regime decision (multi-timeframe)
    crash_detected = np.isfinite(avg_ret_5d) and avg_ret_5d < -0.04

    # 10-day momentum (faster than 20-day)
    if date_idx >= 10:
        d_10ago = signals["sim_dates"][date_idx - 10]
        avg_ret_10d = (close.loc[d, valid_syms] / close.loc[d_10ago, valid_syms] - 1).median()
    else:
        avg_ret_10d = avg_ret_20d

    # Combine signals
    # Market-specific regime detection
    if _current_market == 'us':
        # US: conservative thresholds (higher volatility)
        is_bull = (breadth > 0.55
                   and (np.isfinite(avg_ret_10d) and avg_ret_10d > 0.0)
                   and (np.isfinite(avg_ret_20d) and avg_ret_20d > -0.01)
                   and not crash_detected)
    else:
        # CN: original thresholds (breadth > 50%, 20d > -2%)
        is_bull = (breadth > 0.50
                   and (np.isfinite(avg_ret_20d) and avg_ret_20d > -0.02)
                   and not crash_detected)

    if is_bull:
        # Bull market: Buffett + Livermore
        buffett_pos = buffett_select(signals, date_idx, top_k=top_k)
        livermore_pos = livermore_select(signals, date_idx, top_k=min(5, top_k // 2))

        # Merge: 70% Buffett, 30% Livermore
        merged: dict[str, float] = {}
        for sym, w in buffett_pos:
            merged[sym] = merged.get(sym, 0) + w * 0.7
        for sym, w in livermore_pos:
            merged[sym] = merged.get(sym, 0) + w * 0.3

        total = sum(merged.values())
        if total > 0:
            positions = [(s, w / total) for s, w in merged.items()]
        else:
            positions = buffett_pos
        return positions[:top_k]
    elif crash_detected:
        # Emergency: all bonds/cash
        if etf_symbols:
            if _current_market == 'us':
                bond_etfs = [s for s in etf_symbols if s in ('SHY', 'IEF', 'LQD')
                             and d in close.index and s in close.loc[d].index
                             and np.isfinite(close.loc[d, s]) and close.loc[d, s] > 0]
            else:
                # CN: use available ETFs
                bond_etfs = [s for s in etf_symbols
                             if d in close.index and s in close.loc[d].index
                             and np.isfinite(close.loc[d, s]) and close.loc[d, s] > 0]
            if bond_etfs:
                w = 1.0 / len(bond_etfs)
                return [(s, w) for s in bond_etfs]
        return []  # cash
    else:
        # Bear (not crash): use Dalio-style ETF risk parity
        if etf_symbols:
            return dalio_select(signals, date_idx, symbols_subset=etf_symbols)
        # No ETFs: hold reduced positions
        positions = buffett_select(signals, date_idx, top_k=max(3, top_k // 2))
        return [(s, w * 0.5) for s, w in positions]


# ─── 回测模拟 ────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    "buffett": (buffett_select, "巴菲特·价值护城河"),
    "soros": (soros_select, "索罗斯·反身性趋势"),
    "livermore": (livermore_select, "利弗莫尔·突破金字塔"),
    "dalio": (dalio_select, "达里奥·全天候风险平价"),
    "lynch": (lynch_select, "林奇·GARP成长合理价"),
    "ensemble": (ensemble_select, "大师混合·集成风控"),
    "adaptive": (adaptive_select, "自适应·牛熊切换"),
}


def run_backtest(
    strategy_name: str,
    df: pd.DataFrame,
    start: str,
    end: str,
    rebalance_freq: str = "weekly",
    initial_cash: float = 1_000_000,
    market: str = "cn",
) -> dict:
    """Run portfolio backtest simulation. Returns metrics dict."""
    global _current_market
    _current_market = market
    t0 = time.time()
    _, display_name = ALL_STRATEGIES.get(strategy_name, (None, strategy_name))
    selector_fn = ALL_STRATEGIES[strategy_name][0]

    print(f"\n{'='*60}")
    print(f"回测: {display_name} ({strategy_name})")
    print(f"区间: {start} ~ {end} | 再平衡: {rebalance_freq}")
    print(f"{'='*60}")

    # Compute signals
    signals = compute_signals(df, start, end)
    sim_dates = signals["sim_dates"]
    close_matrix = signals["close"]

    if len(sim_dates) == 0:
        print("No simulation dates found!")
        return {}

    # Determine rebalance dates
    if rebalance_freq == "weekly":
        rebal_indices = list(range(0, len(sim_dates), 5))  # every 5 trading days
    elif rebalance_freq == "monthly":
        rebal_indices = list(range(0, len(sim_dates), 20))
    else:  # daily
        rebal_indices = list(range(len(sim_dates)))

    # Simulate
    cash = initial_cash
    holdings: dict[str, float] = {}  # symbol -> shares
    equity_curve = []
    trade_count = 0
    lot_size = 1 if _current_market == "us" else 100  # US: 1 share, CN: 100 shares

    # Track purchase prices for stop-loss
    purchase_prices: dict[str, float] = {}  # symbol -> avg purchase price
    stop_loss_pct = -0.15  # 15% individual stop-loss (loose for A-shares)
    market_stop_pct = -0.18  # 18% portfolio-level stop (from peak)
    peak_equity = initial_cash

    for idx in range(len(sim_dates)):
        d = sim_dates[idx]

        # Mark-to-market
        portfolio_value = cash
        for sym in list(holdings.keys()):
            if sym.startswith("_"):
                continue
            price = close_matrix.loc[d, sym] if sym in close_matrix.columns else np.nan
            if np.isfinite(price):
                portfolio_value += holdings[sym] * price
                holdings[f"_last_price_{sym}"] = price
            else:
                portfolio_value += holdings[sym] * holdings.get(f"_last_price_{sym}", 0)

        # Track peak for portfolio-level stop
        if portfolio_value > peak_equity:
            peak_equity = portfolio_value

        equity_curve.append({"date": d, "equity": portfolio_value})

        # --- Daily stop-loss checks ---
        # Individual stock stop-loss
        for sym in list(holdings.keys()):
            if sym.startswith("_"):
                continue
            pp = purchase_prices.get(sym, 0)
            if pp <= 0:
                continue
            current_price = holdings.get(f"_last_price_{sym}", 0)
            if current_price > 0 and (current_price / pp - 1) < stop_loss_pct:
                # Trigger stop-loss: sell this stock
                price = close_matrix.loc[d, sym] if sym in close_matrix.columns else 0
                if np.isfinite(price) and price > 0:
                    cash += holdings[sym] * price * 0.999
                    trade_count += 1
                del holdings[sym]
                purchase_prices.pop(sym, None)
                # Clean metadata
                holdings.pop(f"_last_price_{sym}", None)

        # Portfolio-level stop: go to cash if drawdown too deep
        port_dd = portfolio_value / peak_equity - 1 if peak_equity > 0 else 0
        if port_dd < market_stop_pct and holdings:
            # Sell everything
            for sym in list(holdings.keys()):
                if sym.startswith("_"):
                    continue
                price = close_matrix.loc[d, sym] if sym in close_matrix.columns else 0
                if np.isfinite(price) and price > 0:
                    cash += holdings[sym] * price * 0.999
                    trade_count += 1
                holdings.pop(sym, None)
            holdings = {}
            purchase_prices = {}

        # Rebalance?
        if idx in rebal_indices and idx > 0:
            try:
                if strategy_name == "dalio":
                    positions = selector_fn(signals, idx)
                else:
                    positions = selector_fn(signals, idx)
            except Exception:
                positions = []

            if positions:
                # Sell all current holdings
                for sym in list(holdings.keys()):
                    if sym.startswith("_"):
                        continue
                    price = close_matrix.loc[d, sym] if sym in close_matrix.columns else 0
                    if np.isfinite(price) and price > 0:
                        cash += holdings[sym] * price
                        trade_count += 1
                    holdings.pop(sym, None)

                # Clean metadata keys
                holdings = {k: v for k, v in holdings.items() if not k.startswith("_")}

                # Buy new positions
                for sym, weight in positions:
                    price = close_matrix.loc[d, sym] if sym in close_matrix.columns else 0
                    if not np.isfinite(price) or price <= 0:
                        continue
                    alloc = portfolio_value * weight * 0.98  # 2% slippage reserve
                    shares = int(alloc / price / lot_size) * lot_size  # lot size (US=1, CN=100)
                    if shares > 0:
                        cost = shares * price * 1.001  # 0.1% commission
                        if cost <= cash:
                            holdings[sym] = shares
                            holdings[f"_last_price_{sym}"] = price
                            purchase_prices[sym] = price
                            cash -= cost
                            trade_count += 1

    # Compute metrics
    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) < 2:
        print("Insufficient data for metrics")
        return {}

    eq_df["return"] = eq_df["equity"].pct_change()
    total_return = eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1
    n_years = len(eq_df) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Max drawdown
    cummax = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Sharpe
    daily_std = eq_df["return"].std()
    sharpe = (ann_return - 0.03) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0

    # Win rate
    win_rate = (eq_df["return"] > 0).sum() / (eq_df["return"] != 0).sum() if (eq_df["return"] != 0).sum() > 0 else 0

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    elapsed = time.time() - t0
    metrics = {
        "strategy": strategy_name,
        "name": display_name,
        "total_return": total_return,
        "annualized_return": ann_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "final_equity": eq_df["equity"].iloc[-1],
        "elapsed_seconds": round(elapsed, 1),
        "equity_curve": eq_df,
    }

    print(f"\n结果 ({elapsed:.1f}s):")
    print(f"  总收益率:   {total_return:.2%}")
    print(f"  年化收益:   {ann_return:.2%}")
    print(f"  最大回撤:   {max_dd:.2%}")
    print(f"  夏普比率:   {sharpe:.3f}")
    print(f"  卡玛比率:   {calmar:.3f}")
    print(f"  胜率:       {win_rate:.2%}")
    print(f"  交易次数:   {trade_count}")
    print(f"  终值:       {eq_df['equity'].iloc[-1]:,.0f}")

    return metrics


# ─── 参数遍历 ────────────────────────────────────────────────────────

def grid_search(
    df: pd.DataFrame,
    start: str,
    end: str,
    max_dd_limit: float = -0.20,
    market: str = "cn",
) -> list[dict]:
    """Grid search across all strategies and parameters."""
    global _current_market
    _current_market = market
    print("\n" + "=" * 80)
    print(f"参数遍历 | 最大回撤限制: {max_dd_limit:.0%}")
    print("=" * 80)

    all_results = []

    # --- Adaptive strategy params ---
    etf_syms = get_etf_universe(_current_market)
    etf_data = load_price_data(symbols=etf_syms, start=start, end=end)
    adaptive_df = pd.concat([df, etf_data], ignore_index=True)

    for top_k in [8, 10, 15, 20]:
        for rebal in ["weekly", "monthly"]:
            try:
                def make_adaptive(tk):
                    def fn(signals, date_idx, top_k=tk, etf_symbols=etf_syms):
                        return adaptive_select(signals, date_idx, top_k=top_k, etf_symbols=etf_symbols)
                    return fn

                ALL_STRATEGIES["adaptive"] = (make_adaptive(top_k), f"自适应 k={top_k}")
                m = run_backtest("adaptive", adaptive_df, start, end, rebalance_freq=rebal, market=_current_market)
                if m:
                    m["params"] = f"k={top_k},rebal={rebal}"
                    m["dd_ok"] = m["max_drawdown"] >= max_dd_limit
                    tag = "✓" if m["dd_ok"] else "✗"
                    print(f"  Adaptive k={top_k} {rebal}: Sharpe={m['sharpe_ratio']:.2f} "
                          f"DD={m['max_drawdown']:.1%} AnnRet={m['annualized_return']:.1%} {tag}")
                    all_results.append(m)
            except Exception as e:
                print(f"  Adaptive k={top_k} {rebal}: ERROR {e}")

    # --- Buffett params ---
    for top_k in [8, 10, 15, 20]:
        try:
            def make_buffett(tk):
                def fn(signals, date_idx, top_k=tk):
                    return buffett_select(signals, date_idx, top_k=top_k)
                return fn
            ALL_STRATEGIES["buffett"] = (make_buffett(top_k), f"巴菲特 k={top_k}")
            m = run_backtest("buffett", df, start, end, rebalance_freq="weekly", market=_current_market)
            if m:
                m["params"] = f"k={top_k}"
                m["dd_ok"] = m["max_drawdown"] >= max_dd_limit
                tag = "✓" if m["dd_ok"] else "✗"
                print(f"  Buffett k={top_k}: Sharpe={m['sharpe_ratio']:.2f} "
                      f"DD={m['max_drawdown']:.1%} AnnRet={m['annualized_return']:.1%} {tag}")
                all_results.append(m)
        except Exception as e:
            print(f"  Buffett k={top_k}: ERROR {e}")

    # Reset strategies to defaults
    ALL_STRATEGIES["buffett"] = (buffett_select, "巴菲特·价值护城河")
    ALL_STRATEGIES["adaptive"] = (adaptive_select, "自适应·牛熊切换")

    # Summary
    valid = [r for r in all_results if r.get("dd_ok")]
    if valid:
        # Deduplicate by params
        seen = set()
        deduped = []
        for r in valid:
            key = r.get("params", r["name"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        deduped.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        print(f"\n{'='*80}")
        print(f"满足回撤约束的最优方案 (DD >= {max_dd_limit:.0%})")
        print(f"{'='*80}")
        print(f"{'排名':>4} {'策略':<30} {'年化':>8} {'最大回撤':>8} {'夏普':>8} {'卡玛':>8} {'参数'}")
        print("-" * 80)
        for i, r in enumerate(deduped[:10]):
            print(f"{i+1:>4} {r.get('name',''):<30} {r['annualized_return']:>7.1%} "
                  f"{r['max_drawdown']:>7.1%} {r['sharpe_ratio']:>7.2f} {r['calmar_ratio']:>7.2f} "
                  f"{r.get('params','')}")
    else:
        print(f"\n没有找到满足 DD >= {max_dd_limit:.0%} 的方案")

    return all_results


# ─── 全策略对比 ──────────────────────────────────────────────────────

def run_all(df: pd.DataFrame, start: str, end: str, market: str = "cn") -> list[dict]:
    """Run all strategies and compare."""
    global _current_market
    _current_market = market
    results = []

    # Use ETF pool for Dalio
    etf_syms = get_etf_universe(_current_market)

    # For adaptive strategy, load both stock and ETF data
    adaptive_df = df.copy()
    if not adaptive_df["symbol"].isin(etf_syms).any():
        # Load ETF data separately
        etf_data = load_price_data(symbols=etf_syms, start=start, end=end)
        adaptive_df = pd.concat([adaptive_df, etf_data], ignore_index=True)

    for name in ALL_STRATEGIES:
        try:
            if name == "dalio":
                etf_df = df[df["symbol"].isin(etf_syms)]
                if etf_df.empty:
                    etf_df = load_price_data(symbols=etf_syms, start=start, end=end)
                m = run_backtest(name, etf_df, start, end, rebalance_freq="monthly", market=market)
            elif name == "adaptive":
                m = run_backtest(name, adaptive_df, start, end, rebalance_freq="weekly", market=market)
            else:
                m = run_backtest(name, df, start, end, rebalance_freq="weekly", market=market)
            if m:
                results.append(m)
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print(f"\n{'='*90}")
        print("策略对比汇总")
        print(f"{'='*90}")
        print(f"{'策略':<25} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'卡玛':>8} {'胜率':>8} {'交易':>6}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x.get("sharpe_ratio", 0), reverse=True):
            print(f"{r['name']:<25} {r['total_return']:>9.1%} {r['annualized_return']:>9.1%} "
                  f"{r['max_drawdown']:>9.1%} {r['sharpe_ratio']:>7.2f} {r['calmar_ratio']:>7.2f} "
                  f"{r['win_rate']:>7.1%} {r['trade_count']:>5}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="投资大师策略回测研究 v2")
    parser.add_argument("--market", default="cn")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--strategy", choices=list(ALL_STRATEGIES.keys()) + ["all"], default="all")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--top-stocks", type=int, default=200, help="Number of top liquid stocks")
    parser.add_argument("--rebalance", default="weekly", choices=["daily", "weekly", "monthly"])
    args = parser.parse_args()

    print(f"加载数据 (market={args.market}, top {args.top_stocks} 流动性股票)...")
    t0 = time.time()
    _current_market = args.market  # set global before any data loads

    if args.strategy == "dalio":
        symbols = get_etf_universe(args.market)
    elif args.strategy == "all" or args.grid_search:
        symbols = get_top_liquid_stocks(args.top_stocks, market=args.market)
    else:
        symbols = get_top_liquid_stocks(args.top_stocks, market=args.market)

    df = load_price_data(symbols=symbols, start=args.start, end=args.end, market=args.market)
    print(f"数据加载完成: {len(df):,} 条记录, {df['symbol'].nunique()} 只股票 ({time.time()-t0:.1f}s)")

    if args.grid_search:
        grid_search(df, args.start, args.end, market=args.market)
    elif args.strategy == "all":
        run_all(df, args.start, args.end, market=args.market)
    else:
        run_backtest(args.strategy, df, args.start, args.end, rebalance_freq=args.rebalance, market=args.market)
