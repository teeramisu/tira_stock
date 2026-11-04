"""
JoinQuant 策略适配器。

原始策略多为多标的轮动或依赖 jqdata，这里只提取可用 OHLCV 复现的择时核心，
统一输出本项目回测引擎需要的 0/1 单标的持仓序列。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest.jq_risk_control import compute_value_timing_gate_series, rsi as _rsi
from factors.primitives.timing import index_pullback_entry_rate


STRATEGY_JQ_BBI_MOMENTUM = "jq_bbi_momentum_timing"
STRATEGY_JQ_RSRS_TIMING = "jq_rsrs_timing"
STRATEGY_JQ_RSI_GAP_REVERSAL = "jq_rsi_gap_reversal"
STRATEGY_JQ_VALUE_MA_RSI_TIMING = "jq_value_ma_rsi_timing"
STRATEGY_JQ_VALUE_TIMING_TOP_K = "jq_value_timing_top_k"
STRATEGY_JQ_ETF_POLYFIT_TOP1 = "jq_etf_polyfit_momentum_top1"
STRATEGY_JQ_ETF_MOMENTUM_RSRS_TOP1 = "jq_etf_momentum_rsrs_top1"
STRATEGY_JQ_SMALL_CAP_TOP_K = "jq_small_cap_top_k"
STRATEGY_JQ_ETF_MA_CROSS = "jq_etf_ma_cross_timing"
STRATEGY_JQ_INDUSTRY_ETF_BBI = "jq_industry_etf_bbi_rotation"
STRATEGY_JQ_ETF_MOMENTUM_VOLUME_GATE = "jq_etf_momentum_volume_gate_top1"
STRATEGY_JQ_MA_CONSOLIDATION_BREAKOUT = "jq_ma_consolidation_breakout"
STRATEGY_JQ_BIAS_MOMENTUM_TIMING = "jq_bias_momentum_timing"
STRATEGY_JQ_RSRS_WILLIAMS_TIMING = "jq_rsrs_williams_timing"
STRATEGY_JQ_WIDE_BASE_BBI = "jq_wide_base_etf_bbi_rotation"
STRATEGY_JQ_ETF_VALUE_AVERAGING = "jq_etf_value_averaging"
STRATEGY_JQ_ETF_VALUE_AVERAGING_TIMING = "jq_etf_value_averaging_timing"
STRATEGY_JQ_WPETF_MOMENTUM = "jq_wpetf_momentum_top1"


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col not in out.columns and col.lower() in out.columns:
            out[col] = out[col.lower()]
    if "Close" not in out.columns:
        raise ValueError("DataFrame 需要包含 Close 列")
    for col in ("Open", "High", "Low"):
        if col not in out.columns:
            out[col] = out["Close"]
    return out


def jq_bbi_momentum_signal(
    df: pd.DataFrame,
    ma1: int = 3,
    ma2: int = 6,
    ma3: int = 12,
    ma4: int = 24,
    min_daily_return: float = 0.0,
) -> pd.Series:
    """
    适配 jq-0001：宽基 ETF 轮动中的 BBI 动量择时。

    原策略在多指数中挑选动量，并用 BBI/价格关系决定是否开仓。单标的版本只保留
    “收盘价强于 BBI 且日动量不弱”的择时条件。
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)
    periods = [int(ma1), int(ma2), int(ma3), int(ma4)]
    if min(periods) < 1:
        raise ValueError("BBI 均线周期必须为正数")
    bbi = sum(close.rolling(p, min_periods=p).mean() for p in periods) / len(periods)
    prev_close = close.shift(1)
    prev_bbi = bbi.shift(1)
    prev_ret = close.pct_change().shift(1).fillna(0.0)
    position = ((prev_close > prev_bbi) & (prev_ret >= float(min_daily_return))).astype(int)
    return position.fillna(0)


def _rolling_rsrs(high: pd.Series, low: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """向量化 RSRS：用 sliding_window_view 替代 for-loop。"""
    h = high.astype(float).to_numpy()
    l = low.astype(float).to_numpy()
    length = len(h)
    slopes = np.full(length, np.nan)
    r2_values = np.full(length, np.nan)
    if length < n:
        return pd.Series(slopes, index=high.index), pd.Series(r2_values, index=high.index)
    # 滑动窗口矩阵：shape = (length - n + 1, n)
    from numpy.lib.stride_tricks import sliding_window_view
    y_wins = sliding_window_view(h, n)  # high 窗口
    x_wins = sliding_window_view(l, n)  # low 窗口
    # 向量化 polyfit: slope = cov(x,y) / var(x)
    x_mean = x_wins.mean(axis=1)
    y_mean = y_wins.mean(axis=1)
    x_dev = x_wins - x_mean[:, None]
    y_dev = y_wins - y_mean[:, None]
    cov_xy = (x_dev * y_dev).sum(axis=1)
    var_x = (x_dev ** 2).sum(axis=1)
    var_y = (y_dev ** 2).sum(axis=1)
    # 避免除零
    valid = (var_x > 0) & (var_y > 0)
    calc_slopes = np.where(valid, cov_xy / var_x, np.nan)
    intercept = y_mean - np.where(valid, calc_slopes * x_mean, 0)
    fitted = np.where(valid[:, None], calc_slopes[:, None] * x_wins + intercept[:, None], 0)
    ss_res = np.where(valid, ((y_wins - fitted) ** 2).sum(axis=1), 0)
    ss_tot = np.where(valid, (y_dev ** 2).sum(axis=1), 1)
    r2 = np.where(valid & (ss_tot > 0), np.clip(1 - ss_res / ss_tot, 0, 1), 0)
    # 填充结果
    offset = n - 1
    slopes[offset:] = calc_slopes
    r2_values[offset:] = r2
    return pd.Series(slopes, index=high.index), pd.Series(r2_values, index=high.index)


def jq_rsrs_timing_signal(
    df: pd.DataFrame,
    n: int = 18,
    m: int = 120,
    threshold: float = 0.7,
    ma_period: int = 30,
) -> pd.Series:
    """
    适配 jq-0004 / jq-0008：ETF 动量轮动 + RSRS 择时。

    单标的版本使用 High/Low 回归斜率的滚动 zscore * R2 作为择时分数，
    高于阈值开仓，低于负阈值清仓，中间保持原仓位。
    """
    df = _ensure_price_columns(df)
    n = int(n)
    m = int(m)
    ma_period = int(ma_period)
    if n < 2 or m < 10 or ma_period < 2:
        raise ValueError("RSRS 参数需满足 n>=2, m>=10, ma_period>=2")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    slopes, r2 = _rolling_rsrs(high, low, n)
    slope_mean = slopes.rolling(m, min_periods=m).mean()
    slope_std = slopes.rolling(m, min_periods=m).std().replace(0, np.nan)
    rsrs_score = ((slopes - slope_mean) / slope_std) * r2
    ma = close.rolling(ma_period, min_periods=ma_period).mean()

    position = pd.Series(0.0, index=df.index)
    pos = 0
    for i in range(len(df)):
        score = rsrs_score.shift(1).iloc[i]
        prev_close = close.shift(1).iloc[i]
        prev_ma = ma.shift(1).iloc[i]
        if np.isnan(score) or np.isnan(prev_ma):
            position.iloc[i] = pos
            continue
        if score > threshold and prev_close >= prev_ma:
            pos = 1
        elif score < -threshold or prev_close < prev_ma:
            pos = 0
        position.iloc[i] = pos
    return position


def jq_rsi_gap_reversal_signal(
    df: pd.DataFrame,
    rsi_period: int = 6,
    buy_threshold: float = 40.0,
    sell_threshold: float = 55.0,
    ma_period: int = 12,
    weak_body_threshold: float = 0.8,
    strong_body_threshold: float = 0.6,
    stop_loss: float = 0.10,
    take_profit: float = 0.10,
) -> pd.Series:
    """
    适配 2022 年精选 RSI 衍生 ETF 策略。

    原策略对 log(Close / High) 求 RSI，并结合趋势均线、十字星形态、止盈止损。
    这里复现为单标的择时信号。
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)
    high = df["High"].astype(float).replace(0, np.nan)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    gap = np.log((close / high).clip(lower=1e-12))
    gap_rsi = _rsi(gap, int(rsi_period))
    ma = close.ewm(span=int(ma_period), adjust=False).mean()
    body_ratio = ((open_ - close).abs() / (high - low).replace(0, np.nan)).fillna(1.0)

    position = pd.Series(0.0, index=df.index)
    pos = 0
    entry_price = 0.0
    for i in range(len(df)):
        if i == 0:
            continue
        r = gap_rsi.shift(1).iloc[i]
        prev_close = close.shift(1).iloc[i]
        prev_ma = ma.shift(1).iloc[i]
        prev_body = body_ratio.shift(1).iloc[i]
        if np.isnan(r) or np.isnan(prev_ma) or prev_close <= 0:
            position.iloc[i] = pos
            continue
        if pos == 1 and entry_price > 0:
            ret = prev_close / entry_price - 1
            if ret <= -abs(stop_loss) or ret >= abs(take_profit):
                pos = 0
                entry_price = 0.0
        if pos == 0 and r <= buy_threshold and (prev_close > prev_ma or prev_body <= weak_body_threshold):
            pos = 1
            entry_price = float(prev_close)
        elif pos == 1 and r >= sell_threshold and (prev_close <= prev_ma or prev_body <= strong_body_threshold):
            pos = 0
            entry_price = 0.0
        position.iloc[i] = pos
    return position


def jq_value_ma_rsi_timing_signal(
    df: pd.DataFrame,
    ma_period: int = 250,
    ma_high: float = 2.5,
    ma_low: float = 0.30,
    ma_recover_low: float = 0.35,
    ma_recover_high: float = 0.70,
) -> pd.Series:
    """
    适配 jq-0002「54.价值投资策略-大盘择时」中的 RiskControl 大盘择时层。

    原策略还包含 get_fundamentals 基本面选股与多标的调仓，此处仅复现
    check_for_benchmark()：MA 乖离率状态机 + 多周期 RSI 过滤。
    建议在沪深300 ETF（如 510300）上单标的回测择时效果。
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)
    return compute_value_timing_gate_series(
        close,
        ma_period=ma_period,
        ma_high=ma_high,
        ma_low=ma_low,
        ma_recover_low=ma_recover_low,
        ma_recover_high=ma_recover_high,
    )


def jq_etf_ma_cross_timing_signal(
    df: pd.DataFrame,
    ma_long: int = 60,
    ma_short: int = 5,
    ma_sell_fast: int = 20,
    ma_sell_slow: int = 10,
) -> pd.Series:
    """
    适配 jq-0026「简单ETF策略」单标的版。

    买入：MA60 向上 + 收盘价低于 MA60 + MA5 向上。
    卖出：MA20 下穿 MA10。
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)
    ma60 = close.rolling(ma_long, min_periods=ma_long).mean()
    ma5 = close.rolling(ma_short, min_periods=ma_short).mean()
    ma20 = close.rolling(ma_sell_fast, min_periods=ma_sell_fast).mean()
    ma10 = close.rolling(ma_sell_slow, min_periods=ma_sell_slow).mean()

    position = pd.Series(0.0, index=df.index)
    pos = 0
    for i in range(len(close)):
        if i == 0:
            continue
        ma60_up = (
            i >= 2
            and not np.isnan(ma60.iloc[i - 2])
            and not np.isnan(ma60.iloc[i - 1])
            and not np.isnan(ma60.iloc[i])
            and ma60.iloc[i - 2] <= ma60.iloc[i - 1] <= ma60.iloc[i]
        )
        ma5_up = (
            not np.isnan(ma5.iloc[i - 1])
            and not np.isnan(ma5.iloc[i])
            and ma5.iloc[i - 1] < ma5.iloc[i]
        )
        prev_close = close.iloc[i - 1]
        prev_ma60 = ma60.iloc[i - 1]
        cross_down = (
            not np.isnan(ma20.iloc[i - 1])
            and not np.isnan(ma10.iloc[i - 1])
            and not np.isnan(ma20.iloc[i])
            and not np.isnan(ma10.iloc[i])
            and ma20.iloc[i - 1] >= ma10.iloc[i - 1]
            and ma20.iloc[i] < ma10.iloc[i]
        )
        if pos == 1 and cross_down:
            pos = 0
        elif (
            pos == 0
            and ma60_up
            and prev_close < prev_ma60
            and ma5_up
            and not np.isnan(prev_ma60)
        ):
            pos = 1
        position.iloc[i] = pos
    return position.fillna(0).astype(int)


def jq_etf_value_averaging_timing_signal(
    df: pd.DataFrame,
    *,
    entry_pullback_rate: float = 0.318,
    take_profit_return_pct: float = 40.0,
    take_profit_drawdown_pct: float = 9.0,
    full_take_profit_return_pct: float = 100.0,
) -> pd.Series:
    """
    jq-0003 单标的 proxy：回撤入场 + 移动止盈；不含月度申购金额与 IRR 计算。
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)
    position = pd.Series(0.0, index=df.index)
    pos = 0
    entry = 0.0
    peak = 0.0
    for i in range(len(close)):
        if i < 60:
            position.iloc[i] = pos
            continue
        price = float(close.iloc[i - 1])
        if price <= 0:
            position.iloc[i] = pos
            continue
        hist = close.iloc[:i]
        rate1 = index_pullback_entry_rate(hist.to_numpy())
        if pos == 1:
            peak = max(peak, price)
            ret_pct = (price / entry - 1.0) * 100.0 if entry > 0 else 0.0
            dd_pct = (1.0 - price / peak) * 100.0 if peak > 0 else 0.0
            if ret_pct > full_take_profit_return_pct:
                pos = 0
                entry = 0.0
                peak = 0.0
            elif ret_pct > take_profit_return_pct and dd_pct > take_profit_drawdown_pct:
                pos = 0
                entry = 0.0
                peak = 0.0
        if pos == 0 and rate1 < entry_pullback_rate:
            pos = 1
            entry = price
            peak = price
        position.iloc[i] = pos
    return position.fillna(0).astype(int)


ADAPTED_PORTFOLIO_STRATEGIES: dict[str, dict[str, Any]] = {
    "jq_ma_consolidation_breakout": {
        "id": "jq_ma_consolidation_breakout",
        "name": "JoinQuant 均线黏合突破组合",
        "source_strategy_id": "jq-0033",
        "source_strategy_ids": ["jq-0033"],
        "description": "jq-0033 proxy：均线黏合 + 突破前高选股，等权持有最多6只；不含市值/PE过滤。",
        "reproducible_scope": "portfolio_ohlcv_ma_consolidation",
        "non_reproducible_diff": "market_cap/pe/fundamentals filters omitted",
        "score": 58.7,
        "params": ["max_positions"],
        "default_params": {"max_positions": 6},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": "jq_ma_consolidation_breakout",
            "max_positions": 6,
            "rebalance_frequency": "daily",
        },
    },
    "jq_simple_momentum": {
        "id": "jq_simple_momentum",
        "name": "JoinQuant 简单动量选股",
        "source_strategy_id": "jq-0059",
        "source_strategy_ids": ["jq-0059"],
        "description": "jq-0059 proxy：N日收益率排序选股，等权前十；含价格/ST/停牌/次新过滤。",
        "reproducible_scope": "portfolio_ohlcv_momentum",
        "non_reproducible_diff": "get_fundamentals/current_data filters replaced by OHLCV proxies; intraday stop-loss omitted",
        "score": 62.0,
        "params": ["lookback_days", "max_positions"],
        "default_params": {"lookback_days": 91, "max_positions": 10},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": "jq_simple_momentum",
            "max_positions": 10,
            "lookback_days": 91,
            "rebalance_frequency": "weekly",
        },
    },
    STRATEGY_JQ_VALUE_TIMING_TOP_K: {
        "id": STRATEGY_JQ_VALUE_TIMING_TOP_K,
        "name": "JoinQuant 价值投资大盘择时组合（无财报）",
        "source_strategy_id": "jq-0002",
        "description": "jq-0002 无财报 proxy：成交额排序 + MA/RSI 大盘 gate + 5 只等权；不含 get_fundamentals 与止盈止损。",
        "reproducible_scope": "portfolio_ohlcv_proxy_no_fundamentals",
        "non_reproducible_diff": "fundamentals replaced by amount/return proxies; stop_loss/take_profit omitted",
        "score": 72.6,
        "params": ["lookback_days", "max_positions", "ma_period"],
        "default_params": {"lookback_days": 20, "max_positions": 5, "ma_period": 250},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_VALUE_TIMING_TOP_K,
            "max_positions": 5,
            "lookback_days": 20,
            "rebalance_frequency": "weekly",
        },
    },
    STRATEGY_JQ_ETF_POLYFIT_TOP1: {
        "id": STRATEGY_JQ_ETF_POLYFIT_TOP1,
        "name": "JoinQuant ETF polyfit 动量 Top1",
        "source_strategy_id": "jq-0019",
        "source_strategy_ids": ["jq-0019"],
        "description": "jq-0019：log(close) 线性回归年化收益×R²，持动量最强 1 只 ETF。",
        "reproducible_scope": "portfolio_etf_pool_polyfit_top1",
        "non_reproducible_diff": "etf_pool must be supplied via portfolio symbols; industry ETF list not auto-loaded",
        "score": 65.0,
        "params": ["portfolio_lookback_days"],
        "default_params": {"lookback_days": 25},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_ETF_POLYFIT_TOP1,
            "lookback_days": 25,
            "symbols": "510300,510500,159915,513100,518880,510180",
        },
    },
    STRATEGY_JQ_ETF_MOMENTUM_RSRS_TOP1: {
        "id": STRATEGY_JQ_ETF_MOMENTUM_RSRS_TOP1,
        "name": "JoinQuant ETF 动量+RSRS Top1",
        "source_strategy_id": "jq-0010",
        "source_strategy_ids": ["jq-0010", "jq-0008"],
        "description": "jq-0010/0008：polyfit 动量选 1 只 ETF，510300 RSRS×R² + MA 端点比较 gate。",
        "reproducible_scope": "portfolio_etf_pool_momentum_rsrs_top1",
        "non_reproducible_diff": "northbound flow and multi-etf pool defaults simplified",
        "score": 68.0,
        "params": ["portfolio_lookback_days"],
        "default_params": {"lookback_days": 29},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_ETF_MOMENTUM_RSRS_TOP1,
            "lookback_days": 29,
            "symbols": "510180,159915,513100,510500",
        },
    },
    STRATEGY_JQ_SMALL_CAP_TOP_K: {
        "id": STRATEGY_JQ_SMALL_CAP_TOP_K,
        "name": "JoinQuant 小市值 TopK（无财报）",
        "source_strategy_id": "jq-0006",
        "source_strategy_ids": ["jq-0006", "jq-0012", "jq-0022", "jq-0023"],
        "description": "jq-0006 等：成交额 proxy 升序取小盘 + MA5 过滤；不含 valuation.market_cap 与国九条 filter。",
        "reproducible_scope": "portfolio_small_cap_liquidity_proxy",
        "non_reproducible_diff": "market_cap/guojiu/dividend filters omitted; uses avg amount ascending",
        "score": 64.0,
        "params": ["portfolio_lookback_days", "max_positions"],
        "default_params": {"lookback_days": 5, "max_positions": 3},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_SMALL_CAP_TOP_K,
            "max_positions": 3,
            "lookback_days": 5,
            "rebalance_frequency": "weekly",
        },
    },
    STRATEGY_JQ_INDUSTRY_ETF_BBI: {
        "id": STRATEGY_JQ_INDUSTRY_ETF_BBI,
        "name": "JoinQuant 行业ETF BBI轮动+择时",
        "source_strategy_id": "jq-0007",
        "source_strategy_ids": ["jq-0007"],
        "description": "jq-0007：大盘广度 + 510300 EMA5 gate；行业 ETF 取 BBI/价最低（最偏多），弱市持 511880。",
        "reproducible_scope": "portfolio_industry_etf_bbi_rotation_daily_proxy",
        "non_reproducible_diff": "30m BBI/industry index mapping simplified to daily ETF BBI; Wednesday 11:15 schedule → weekly rebalance",
        "score": 66.0,
        "params": ["portfolio_symbols"],
        "default_params": {},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_INDUSTRY_ETF_BBI,
            "rebalance_frequency": "weekly",
            "symbols": (
                "515220,516070,512660,159611,159944,516120,159928,512690,512170,"
                "512980,512800,159841,512480,515080,515210,159825,516970,511880"
            ),
        },
    },
    STRATEGY_JQ_ETF_MOMENTUM_VOLUME_GATE: {
        "id": STRATEGY_JQ_ETF_MOMENTUM_VOLUME_GATE,
        "name": "JoinQuant ETF动量+成交量gate Top1",
        "source_strategy_id": "jq-0020",
        "source_strategy_ids": ["jq-0020"],
        "description": "jq-0020：13日涨幅排序 + 成交量/价格情绪 gate；弱市持 511880。",
        "reproducible_scope": "portfolio_etf_momentum_volume_gate",
        "non_reproducible_diff": "stop_loss/take_profit per-ETF and dynamic target_market omitted",
        "score": 63.0,
        "params": ["portfolio_lookback_days"],
        "default_params": {"lookback_days": 13},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_ETF_MOMENTUM_VOLUME_GATE,
            "symbols": (
                "510050,510180,510880,159905,159949,510300,510500,515800,"
                "159966,159967,513500,513100,510900,512160,511380,511880"
            ),
        },
    },
    STRATEGY_JQ_WIDE_BASE_BBI: {
        "id": STRATEGY_JQ_WIDE_BASE_BBI,
        "name": "JoinQuant 宽基ETF BBI轮动",
        "source_strategy_id": "jq-0001",
        "source_strategy_ids": ["jq-0001"],
        "description": "jq-0001 完整池：大盘广度 gate + 最低 BBI/价 ETF 等权持有。",
        "reproducible_scope": "portfolio_wide_base_etf_bbi_rotation",
        "non_reproducible_diff": "30m BBI on indices mapped to daily ETF BBI proxy",
        "score": 75.2,
        "params": ["portfolio_symbols"],
        "default_params": {},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_WIDE_BASE_BBI,
            "symbols": (
                "510050,510180,510880,159905,159949,510300,510500,159915,"
                "588000,512100,159966,159967"
            ),
        },
    },
    STRATEGY_JQ_ETF_VALUE_AVERAGING: {
        "id": STRATEGY_JQ_ETF_VALUE_AVERAGING,
        "name": "JoinQuant ETF价值平均定投",
        "source_strategy_id": "jq-0003",
        "source_strategy_ids": ["jq-0003"],
        "description": "jq-0003 proxy：指数回撤入场 + 月度加仓权重 + 移动止盈；非真实申购金额 DCA。",
        "reproducible_scope": "portfolio_value_averaging_proxy",
        "non_reproducible_diff": "monthly cash DCA amounts and IRR tracking omitted",
        "score": 60.0,
        "params": ["portfolio_symbols"],
        "default_params": {},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_ETF_VALUE_AVERAGING,
            "rebalance_frequency": "monthly",
            "symbols": "159905",
        },
    },
    STRATEGY_JQ_WPETF_MOMENTUM: {
        "id": STRATEGY_JQ_WPETF_MOMENTUM,
        "name": "JoinQuant WPETF动量 Top1",
        "source_strategy_id": "jq-0009",
        "source_strategy_ids": ["jq-0009"],
        "description": "jq-0009 仅 ETF 腿：多项式动量 + RSI 修正，持 Top1。",
        "reproducible_scope": "portfolio_wpetf_momentum_top1",
        "non_reproducible_diff": "multi-strategy capital split and small-cap leg omitted",
        "score": 65.2,
        "params": ["portfolio_lookback_days"],
        "default_params": {"lookback_days": 25},
        "run_preset": {
            "mode": "portfolio",
            "market": "cn",
            "strategy": STRATEGY_JQ_WPETF_MOMENTUM,
            "lookback_days": 25,
            "symbols": "513100,159915,513520,511010,518880",
        },
    },
}

def jq_ma_consolidation_breakout_signal(
    df: pd.DataFrame,
    *,
    fast_period: int = 13,
    mid_period: int = 34,
    slow_period: int = 55,
    consolidation_threshold: float = 0.03,
    breakout_lookback: int = 30,
    sell_ma_period: int = 13,
    sell_consecutive_days: int = 3,
) -> pd.Series:
    """
    均线黏合突破: Buy on MA consolidation + price breakout above N-day high.

    Buy conditions (all must hold):
      1. MA13 > MA34 > MA55 (aligned bull)
      2. MA pairwise max diff <= consolidation_threshold * close (黏合)
         checked at T-10 and T-19
      3. close breaks above breakout_lookback-day high
      4. close >= open (收阳)

    Sell: close < MA13 for sell_consecutive_days consecutive days.
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)
    open_px = df["Open"].astype(float) if "Open" in df.columns else close

    ma_fast = close.rolling(int(fast_period), min_periods=int(fast_period)).mean()
    ma_mid = close.rolling(int(mid_period), min_periods=int(mid_period)).mean()
    ma_slow = close.rolling(int(slow_period), min_periods=int(slow_period)).mean()

    position = pd.Series(0.0, index=df.index)
    pos = 0
    for i in range(len(df)):
        if i < slow_period + breakout_lookback:
            position.iloc[i] = pos
            continue
        c = float(close.iloc[i - 1])
        o = float(open_px.iloc[i - 1])
        mf = float(ma_fast.iloc[i - 1])
        mm = float(ma_mid.iloc[i - 1])
        ms = float(ma_slow.iloc[i - 1])
        if any(np.isnan(x) for x in [c, mf, mm, ms]):
            position.iloc[i] = pos
            continue

        if pos == 1:
            below_count = 0
            for j in range(int(sell_consecutive_days)):
                idx = i - 1 - j
                if idx < 0:
                    break
                if float(close.iloc[idx]) < float(ma_fast.iloc[idx]):
                    below_count += 1
            if below_count >= int(sell_consecutive_days):
                pos = 0
                position.iloc[i] = pos
                continue

        if pos == 0 and mf > mm > ms:
            t10 = i - 10
            t19 = i - 19
            if t10 >= 0 and t19 >= 0:
                mf10 = float(ma_fast.iloc[t10])
                mm10 = float(ma_mid.iloc[t10])
                ms10 = float(ma_slow.iloc[t10])
                mf19 = float(ma_fast.iloc[t19])
                mm19 = float(ma_mid.iloc[t19])
                ms19 = float(ma_slow.iloc[t19])
                pairs10 = [abs(mf10 - mm10), abs(mm10 - ms10), abs(mf10 - ms10)]
                pairs19 = [abs(mf19 - mm19), abs(mm19 - ms19), abs(mf19 - ms19)]
                all_close = (
                    max(pairs10) <= consolidation_threshold * c
                    and max(pairs19) <= consolidation_threshold * c
                )
                if all_close:
                    lookback_start = max(0, i - 1 - int(breakout_lookback))
                    prev_high = float(np.max(close.iloc[lookback_start:i - 1].to_numpy()))
                    if c > prev_high and c >= o:
                        pos = 1
        position.iloc[i] = pos
    return position.fillna(0).astype(int)


def jq_bias_momentum_timing_signal(
    df: pd.DataFrame,
    *,
    bias_period: int = 90,
    momentum_days: int = 20,
) -> pd.Series:
    """
    乖离动量择时: Score based on (close/MA_bias) trend via polyfit.

    From jq-0010 V2.1 momentum ranking core: bias = close / MA90,
    score = polyfit slope of bias/bias[0] over momentum_days.
    """
    df = _ensure_price_columns(df)
    close = df["Close"].astype(float)

    from factors.primitives.timing import bias_momentum_score

    scores = np.full(len(close), float("-inf"))
    for i in range(len(close)):
        if i < bias_period + momentum_days:
            continue
        segment = close.iloc[: i + 1].to_numpy()
        scores[i] = bias_momentum_score(
            segment, bias_period=bias_period, momentum_days=momentum_days
        )
    valid = np.isfinite(scores)
    if not np.any(valid):
        return pd.Series(0.0, index=df.index)
    median = float(np.median(scores[valid]))
    position = pd.Series(np.where(scores > median, 1.0, 0.0), index=df.index)
    return position


def jq_rsrs_williams_timing_signal(
    df: pd.DataFrame,
    *,
    rsrs_n: int = 18,
    rsrs_m: int = 600,
    score_threshold: float = -0.68,
    williams_period: int = 21,
    williams_oversold: float = 97.0,
) -> pd.Series:
    """
    RSRS + Williams %R composite timing.

    From jq-0010 V2.1 multi-factor timing:
    - Williams %R oversold (>= 97) = BUY
    - RSRS slope < 0 and RSRS score > 0 = SELL
    - RSRS score > threshold = BUY
    - Else SELL
    """
    df = _ensure_price_columns(df)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    from factors.primitives.timing import williams_r, rsrs_zscore_slope

    position = pd.Series(0.0, index=df.index)
    all_slopes: list[float] = []
    all_zscores: list[float] = []
    pos = 0

    for i in range(len(df)):
        if i < rsrs_n + rsrs_m:
            position.iloc[i] = pos
            continue
        seg_h = high.iloc[: i + 1].to_numpy()
        seg_l = low.iloc[: i + 1].to_numpy()
        seg_c = close.iloc[: i + 1].to_numpy()

        hi = seg_h[-rsrs_n:]
        lo = seg_l[-rsrs_n:]
        if np.std(lo) == 0 or np.std(hi) == 0:
            position.iloc[i] = pos
            continue
        slope, intercept = np.polyfit(lo, hi, 1)
        fitted = slope * lo + intercept
        ss_res = float(np.sum((hi - fitted) ** 2))
        ss_tot = float(np.sum((hi - np.mean(hi)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        all_slopes.append(float(slope))
        if len(all_slopes) >= rsrs_m:
            recent = np.array(all_slopes[-rsrs_m:], dtype=float)
            mean_s = float(np.mean(recent))
            std_s = float(np.std(recent))
            zscore = (float(slope) - mean_s) / std_s if std_s > 0 else 0.0
            rsrs_score = zscore * max(0.0, r2)
            all_zscores.append(rsrs_score)
        else:
            position.iloc[i] = pos
            continue

        wr = williams_r(seg_h, seg_l, seg_c, period=williams_period)
        z_slope = rsrs_zscore_slope(np.array(all_zscores, dtype=float), k=8)

        if wr >= williams_oversold:
            pos = 1
        elif z_slope < 0 and len(all_zscores) >= 3 and all_zscores[-2] > 0:
            pos = 0
        elif rsrs_score > score_threshold:
            pos = 1
        else:
            pos = 0
        position.iloc[i] = pos
    return position


ADAPTED_STRATEGIES: dict[str, dict[str, Any]] = {
    STRATEGY_JQ_BBI_MOMENTUM: {
        "id": STRATEGY_JQ_BBI_MOMENTUM,
        "name": "JoinQuant 宽基ETF BBI动量择时",
        "source_strategy_id": "jq-0001",
        "description": "从宽基 ETF 轮动策略中抽取 BBI 动量择时，适配为单标的 0/1 持仓信号。",
        "reproducible_scope": "single_symbol_ohlcv_timing_only",
        "non_reproducible_diff": "original multi-index rotation and jqdata context are not reproduced",
        "score": 75.2,
        "params": ["ma1", "ma2", "ma3", "ma4", "min_daily_return"],
        "default_params": {"ma1": 3, "ma2": 6, "ma3": 12, "ma4": 24, "min_daily_return": 0.0},
        "signal": jq_bbi_momentum_signal,
    },
    STRATEGY_JQ_RSRS_TIMING: {
        "id": STRATEGY_JQ_RSRS_TIMING,
        "name": "JoinQuant ETF RSRS择时",
        "source_strategy_id": "jq-0004",
        "source_strategy_ids": ["jq-0004", "jq-0008"],
        "description": "复现 RSRS 斜率标准分择时，并增加均线过滤，适合指数/ETF 单标的回测。",
        "reproducible_scope": "single_symbol_ohlcv_rsrs_timing_only",
        "non_reproducible_diff": "original cross-asset rotation and live scheduling details are simplified",
        "score": 68.4,
        "params": ["n", "m", "threshold", "ma_period"],
        "default_params": {"n": 18, "m": 120, "threshold": 0.7, "ma_period": 30},
        "signal": jq_rsrs_timing_signal,
    },
    STRATEGY_JQ_RSI_GAP_REVERSAL: {
        "id": STRATEGY_JQ_RSI_GAP_REVERSAL,
        "name": "JoinQuant RSI衍生ETF择时",
        "source_strategy_id": "jq-0036",
        "description": "对 Close/High 缺口序列计算 RSI，结合 EMA 趋势、十字星形态和止盈止损。",
        "reproducible_scope": "single_symbol_ohlcv_gap_rsi_timing_only",
        "non_reproducible_diff": "jq platform data dependencies and execution assumptions are approximated",
        "score": 58.3,
        "params": [
            "rsi_period",
            "buy_threshold",
            "sell_threshold",
            "ma_period",
            "weak_body_threshold",
            "strong_body_threshold",
            "stop_loss",
            "take_profit",
        ],
        "default_params": {
            "rsi_period": 6,
            "buy_threshold": 40.0,
            "sell_threshold": 55.0,
            "ma_period": 12,
            "weak_body_threshold": 0.8,
            "strong_body_threshold": 0.6,
            "stop_loss": 0.10,
            "take_profit": 0.10,
        },
        "signal": jq_rsi_gap_reversal_signal,
    },
    STRATEGY_JQ_VALUE_MA_RSI_TIMING: {
        "id": STRATEGY_JQ_VALUE_MA_RSI_TIMING,
        "name": "JoinQuant 价值投资大盘择时（MA+RSI）",
        "source_strategy_id": "jq-0002",
        "description": "从 jq-0002 抽取 RiskControl 大盘择时；不含基本面选股与多标的调仓。",
        "reproducible_scope": "single_symbol_hs300_etf_timing_only",
        "non_reproducible_diff": "get_fundamentals stock screen, monthly rebalance, stop_loss rules omitted",
        "score": 72.6,
        "params": ["ma_period"],
        "default_params": {"ma_period": 250},
        "run_preset": {"market": "cn", "symbol": "510300", "mode": "single"},
        "signal": jq_value_ma_rsi_timing_signal,
    },
    STRATEGY_JQ_ETF_MA_CROSS: {
        "id": STRATEGY_JQ_ETF_MA_CROSS,
        "name": "JoinQuant 简单ETF MA交叉择时",
        "source_strategy_id": "jq-0026",
        "source_strategy_ids": ["jq-0026"],
        "description": "jq-0026 单标的版：MA60 向上 + 价低于 MA60 + MA5 向上买入；MA20 下穿 MA10 卖出。",
        "reproducible_scope": "single_symbol_etf_ma_cross_timing",
        "non_reproducible_diff": "multi-etf pool scan omitted; apply rules to one symbol",
        "score": 62.0,
        "params": ["slow_period", "fast_period"],
        "default_params": {"ma_long": 60, "ma_short": 5, "ma_sell_fast": 20, "ma_sell_slow": 10},
        "run_preset": {"market": "cn", "symbol": "510300", "mode": "single"},
        "signal": jq_etf_ma_cross_timing_signal,
    },
    STRATEGY_JQ_ETF_VALUE_AVERAGING_TIMING: {
        "id": STRATEGY_JQ_ETF_VALUE_AVERAGING_TIMING,
        "name": "JoinQuant ETF价值平均择时（proxy）",
        "source_strategy_id": "jq-0003",
        "description": "jq-0003 单标的 proxy：回撤入场 + 移动止盈；不含月度申购。",
        "reproducible_scope": "single_symbol_value_avg_timing_proxy",
        "non_reproducible_diff": "value-averaging monthly buy amounts omitted",
        "score": 60.0,
        "params": ["entry_pullback_rate"],
        "default_params": {"entry_pullback_rate": 0.318},
        "run_preset": {"market": "cn", "symbol": "159905", "mode": "single"},
        "signal": jq_etf_value_averaging_timing_signal,
    },
    "jq_ma_consolidation_breakout": {
        "id": "jq_ma_consolidation_breakout",
        "name": "JoinQuant 均线黏合突破择时",
        "source_strategy_id": "jq-0033",
        "source_strategy_ids": ["jq-0033"],
        "description": "jq-0033 均线黏合突破法：MA13/34/55 多头排列+黏合+突破30日高点买入；连3日跌破MA13卖出。",
        "reproducible_scope": "single_symbol_ohlcv_ma_consolidation_breakout",
        "non_reproducible_diff": "market_cap/pe screening replaced by pure OHLCV entry conditions",
        "score": 58.7,
        "params": ["fast_period", "mid_period", "slow_period", "consolidation_threshold", "breakout_lookback"],
        "default_params": {"fast_period": 13, "mid_period": 34, "slow_period": 55, "consolidation_threshold": 0.03, "breakout_lookback": 30},
        "run_preset": {"market": "cn", "symbol": "000300.XSHG", "mode": "single"},
        "signal": jq_ma_consolidation_breakout_signal,
    },
    "jq_bias_momentum_timing": {
        "id": "jq_bias_momentum_timing",
        "name": "JoinQuant 乖离动量择时",
        "source_strategy_id": "jq-0010",
        "source_strategy_ids": ["jq-0010"],
        "description": "jq-0010 V2.1 偏离动量：close/MA90 的 polyfit 斜率；高于中位数持仓。",
        "reproducible_scope": "single_symbol_ohlcv_bias_momentum",
        "non_reproducible_diff": "multi-etf pool ranking simplified to single-symbol bias score threshold",
        "score": 64.1,
        "params": ["bias_period", "momentum_days"],
        "default_params": {"bias_period": 90, "momentum_days": 20},
        "run_preset": {"market": "cn", "symbol": "510300", "mode": "single"},
        "signal": jq_bias_momentum_timing_signal,
    },
    "jq_rsrs_williams_timing": {
        "id": "jq_rsrs_williams_timing",
        "name": "JoinQuant RSRS+威廉指标择时",
        "source_strategy_id": "jq-0010",
        "source_strategy_ids": ["jq-0010"],
        "description": "jq-0010 多因子择时：RSRS zscore×R² + RSRS斜率 + Williams %R + 指数趋势复合信号。",
        "reproducible_scope": "single_symbol_ohlcv_rsrs_williams",
        "non_reproducible_diff": "intraday 60m stop-loss and message alerts omitted",
        "score": 64.1,
        "params": ["rsrs_n", "rsrs_m", "score_threshold", "williams_period"],
        "default_params": {"rsrs_n": 18, "rsrs_m": 600, "score_threshold": -0.68, "williams_period": 21, "williams_oversold": 97.0},
        "run_preset": {"market": "cn", "symbol": "510300", "mode": "single"},
        "signal": jq_rsrs_williams_timing_signal,
    },
}


def get_adapted_strategy(strategy_id: str) -> dict[str, Any] | None:
    key = (strategy_id or "").strip().lower()
    item = ADAPTED_STRATEGIES.get(key)
    if item:
        return item
    return ADAPTED_PORTFOLIO_STRATEGIES.get(key)


def get_adapted_strategy_signal(strategy_id: str) -> Callable[..., pd.Series] | None:
    item = ADAPTED_STRATEGIES.get((strategy_id or "").strip().lower())
    if not item:
        return None
    return item["signal"]


def _adapted_source_ids(item: dict[str, Any]) -> list[str]:
    ids = item.get("source_strategy_ids")
    if isinstance(ids, list) and ids:
        return [str(x) for x in ids if x]
    sid = item.get("source_strategy_id")
    return [str(sid)] if sid else []


def list_adapted_strategies() -> list[dict[str, Any]]:
    out = []
    for item in ADAPTED_STRATEGIES.values():
        public = {k: v for k, v in item.items() if k != "signal"}
        public["engine_status"] = "adapted_backtest_ready"
        if "run_preset" in item:
            public["run_preset"] = item["run_preset"]
        out.append(public)
    for item in ADAPTED_PORTFOLIO_STRATEGIES.values():
        public = dict(item)
        public["engine_status"] = "adapted_backtest_ready"
        if "run_preset" in item:
            public["run_preset"] = item["run_preset"]
        out.append(public)
    return out
