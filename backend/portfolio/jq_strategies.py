"""JoinQuant-adapted portfolio strategies (OHLCV / no-fundamentals proxies)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from factors.primitives.breadth import breadth_best_return, market_breadth_bullish
from factors.primitives.etf_selection import (
    build_etf_momentum_row,
    market_breadth_clear_signal,
    select_best_etf_momentum,
)
from factors.primitives.momentum import polyfit_log_momentum_score, wpetf_momentum_score
from factors.primitives.timing import (
    index_pullback_entry_rate,
    price_emotion_monitor,
    rsrs_ma_timing_allows,
    volume_emotion_monitor,
)
from factors.primitives.trend import bbi_price_ratio, ema5_bearish_gate
from backtest.jq_risk_control import compute_value_timing_gate_series
from marketdata.data_portal import DataPortal

from .models import StrategyContext, TargetPosition


def _history_matrix(
    data: DataPortal,
    *,
    market: str,
    symbols: tuple[str, ...],
    fields: list[str],
    end_time: date,
    bars: int,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return data.history(
        market=market,
        symbols=symbols,
        fields=fields,
        end_time=end_time,
        bars=bars,
        frequency="1d",
    )


class EtfPolyfitMomentumTop1Strategy:
    """jq-0019: log-close polyfit momentum × R², hold top-1 ETF."""

    id = "jq_etf_polyfit_momentum_top1"
    name = "JoinQuant ETF polyfit 动量 Top1"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(self, lookback_days: int = 25) -> None:
        if lookback_days <= 2:
            raise ValueError("lookback_days must be > 2")
        self.lookback_days = lookback_days
        self.warmup_days = lookback_days + 1

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        history = _history_matrix(
            data,
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days,
        )
        if history.empty:
            return []
        best_symbol = ""
        best_score = float("-inf")
        for symbol, group in history.groupby("symbol", sort=False):
            closes = group.sort_values("trade_date")["Close"].astype(float).to_numpy()
            if closes.size < self.lookback_days:
                continue
            score = polyfit_log_momentum_score(closes)
            if score > best_score:
                best_score = score
                best_symbol = str(symbol)
        if not best_symbol or not np.isfinite(best_score):
            return []
        return [
            TargetPosition(
                symbol=best_symbol,
                weight=1.0,
                score=float(best_score),
                reason="polyfit_log_momentum_top1",
            )
        ]


class EtfMomentumRsrsTop1Strategy:
    """jq-0010 / jq-0008: polyfit momentum top-1 gated by HS300 RSRS+MA timing."""

    id = "jq_etf_momentum_rsrs_top1"
    name = "JoinQuant ETF 动量+RSRS Top1"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(
        self,
        lookback_days: int = 29,
        benchmark_symbol: str = "510300",
        rsrs_n: int = 18,
        rsrs_m: int = 120,
        rsrs_threshold: float = 0.7,
        mean_day: int = 20,
        mean_diff_day: int = 3,
    ) -> None:
        self.lookback_days = lookback_days
        self.benchmark_symbol = benchmark_symbol
        self.rsrs_n = rsrs_n
        self.rsrs_m = rsrs_m
        self.rsrs_threshold = rsrs_threshold
        self.mean_day = mean_day
        self.mean_diff_day = mean_diff_day
        self.warmup_days = max(lookback_days, rsrs_n + rsrs_m, mean_day + mean_diff_day) + 5

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        if not self._timing_allows(trade_date, data, context.market):
            return []
        history = _history_matrix(
            data,
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days,
        )
        if history.empty:
            return []
        best_symbol = ""
        best_score = float("-inf")
        for symbol, group in history.groupby("symbol", sort=False):
            closes = group.sort_values("trade_date")["Close"].astype(float).to_numpy()
            if closes.size < self.lookback_days:
                continue
            score = polyfit_log_momentum_score(closes)
            if score > best_score:
                best_score = score
                best_symbol = str(symbol)
        if not best_symbol:
            return []
        return [
            TargetPosition(
                symbol=best_symbol,
                weight=1.0,
                score=float(best_score),
                reason="momentum_rsrs_top1",
            )
        ]

    def _timing_allows(self, trade_date: date, data: DataPortal, market: str) -> bool:
        bars = self.rsrs_n + self.rsrs_m + self.mean_day + self.mean_diff_day + 5
        bench = _history_matrix(
            data,
            market=market,
            symbols=(self.benchmark_symbol,),
            fields=["Open", "High", "Low", "Close"],
            end_time=trade_date,
            bars=bars,
        )
        if bench.empty:
            return False
        group = bench.sort_values("trade_date")
        return rsrs_ma_timing_allows(
            group["High"].astype(float).to_numpy(),
            group["Low"].astype(float).to_numpy(),
            group["Close"].astype(float).to_numpy(),
            n=self.rsrs_n,
            m=self.rsrs_m,
            threshold=self.rsrs_threshold,
            mean_day=self.mean_day,
            mean_diff_day=self.mean_diff_day,
        )


class JqSmallCapTopKStrategy:
    """jq-0006 / jq-0012 / jq-0022 proxy: smallest liquidity (market-cap proxy) + optional MA5."""

    id = "jq_small_cap_top_k"
    name = "JoinQuant 小市值 TopK（无财报）"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(
        self,
        top_k: int = 3,
        lookback_days: int = 5,
        require_ma5: bool = True,
        benchmark_symbol: str = "510300",
        use_timing_gate: bool = False,
        ma_period: int = 250,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.lookback_days = lookback_days
        self.require_ma5 = require_ma5
        self.benchmark_symbol = benchmark_symbol
        self.use_timing_gate = use_timing_gate
        self.ma_period = ma_period
        self.warmup_days = max(lookback_days, 5, ma_period if use_timing_gate else 0) + 2

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        if self.use_timing_gate and not self._benchmark_allows(trade_date, data, context.market):
            return []
        bars = max(self.lookback_days, 5) + 1
        history = _history_matrix(
            data,
            market=context.market,
            symbols=context.universe,
            fields=["Close", "Volume", "amount"],
            end_time=trade_date,
            bars=bars,
        )
        if history.empty:
            return []
        rows: list[tuple[str, float, pd.Series]] = []
        for symbol, group in history.groupby("symbol", sort=False):
            group = group.sort_values("trade_date")
            if len(group) < bars:
                continue
            close = group["Close"].astype(float)
            if close.iloc[-1] <= 0:
                continue
            amount = group["amount"].astype(float) if "amount" in group.columns else pd.Series(np.nan, index=group.index)
            volume = group["Volume"].astype(float) if "Volume" in group.columns else pd.Series(np.nan, index=group.index)
            liquidity = amount.fillna(volume * close)
            avg_liq = float(liquidity.tail(self.lookback_days).mean())
            if not np.isfinite(avg_liq) or avg_liq <= 0:
                continue
            if self.require_ma5 and len(close) >= 5:
                if float(close.iloc[-1]) < float(close.tail(5).mean()):
                    continue
            rows.append((str(symbol), avg_liq, close))
        if not rows:
            return []
        rows.sort(key=lambda item: item[1])
        selected = rows[: self.top_k]
        weight = 1.0 / len(selected)
        return [
            TargetPosition(
                symbol=symbol,
                weight=weight,
                score=float(1.0 / max(avg_liq, 1.0)),
                reason="small_cap_liquidity_proxy",
            )
            for symbol, avg_liq, _ in selected
        ]

    def _benchmark_allows(self, trade_date: date, data: DataPortal, market: str) -> bool:
        bench = _history_matrix(
            data,
            market=market,
            symbols=(self.benchmark_symbol,),
            fields=["Close"],
            end_time=trade_date,
            bars=self.ma_period + 95,
        )
        if bench.empty:
            return False
        closes = bench.sort_values("trade_date")["Close"].astype(float)
        if len(closes) < self.ma_period + 2:
            return False
        gate = compute_value_timing_gate_series(closes.reset_index(drop=True), ma_period=self.ma_period)
        return bool(gate.iloc[-1])


DEFAULT_INDUSTRY_ETF_POOL: tuple[str, ...] = (
    "515220",
    "516070",
    "512660",
    "159611",
    "159944",
    "516120",
    "159928",
    "512690",
    "512170",
    "512980",
    "512800",
    "159841",
    "512480",
    "515080",
    "515210",
    "159825",
    "516970",
)

DEFAULT_MARKET_BREADTH_SYMBOLS: tuple[str, ...] = (
    "510300",
    "510500",
    "159915",
    "510180",
    "512100",
    "588000",
)


class JqIndustryEtfBbiRotationStrategy:
    """
    jq-0007: industry ETF rotation with Fibonacci BBI + market breadth timing.

    Weak market or 510300 EMA5 bearish → hold ``bond_symbol`` (511880 proxy).
    Strong market → hold industry ETF with lowest BBI/price (most bullish).
    """

    id = "jq_industry_etf_bbi_rotation"
    name = "JoinQuant 行业ETF BBI轮动+择时"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        *,
        bond_symbol: str = "511880",
        timing_symbol: str = "510300",
        bbi_periods: tuple[int, ...] = (21, 34, 55, 89),
        market_symbols: tuple[str, ...] = DEFAULT_MARKET_BREADTH_SYMBOLS,
        etf_pool: tuple[str, ...] | None = None,
        prev_return_weight: float = 0.05,
    ) -> None:
        self.bond_symbol = bond_symbol
        self.timing_symbol = timing_symbol
        self.bbi_periods = bbi_periods
        self.market_symbols = market_symbols
        self.etf_pool = etf_pool or DEFAULT_INDUSTRY_ETF_POOL
        self.prev_return_weight = prev_return_weight
        self.warmup_days = max(bbi_periods) + 5

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        pool = self._etf_pool(context)
        if not pool:
            return []

        timing_closes = self._load_closes(
            data, context.market, (self.timing_symbol,), trade_date, bars=10
        )
        timing_arr = timing_closes.get(self.timing_symbol)
        if timing_arr is None or ema5_bearish_gate(timing_arr):
            return self._bond_target("ema5_bearish")

        breadth_closes = self._load_closes(
            data, context.market, self.market_symbols, trade_date, bars=3
        )
        market_ok, _ = market_breadth_bullish(
            breadth_closes,
            prev_return_weight=self.prev_return_weight,
        )
        if not market_ok:
            return self._bond_target("market_breadth_weak")

        bars = max(self.bbi_periods) + 1
        etf_history = self._load_closes(data, context.market, pool, trade_date, bars=bars)
        scored: list[tuple[str, float]] = []
        for symbol in pool:
            closes = etf_history.get(symbol)
            if closes is None:
                continue
            ratio = bbi_price_ratio(closes, self.bbi_periods)
            if np.isfinite(ratio):
                scored.append((symbol, ratio))
        if not scored:
            return self._bond_target("no_etf_scores")
        scored.sort(key=lambda item: item[1])
        target_symbol, target_ratio = scored[0]
        if target_ratio >= 1.0:
            return self._bond_target("target_bbi_not_bullish")
        return [
            TargetPosition(
                symbol=target_symbol,
                weight=1.0,
                score=float(1.0 - target_ratio),
                reason="industry_etf_bbi_min_ratio",
            )
        ]

    def _etf_pool(self, context: StrategyContext) -> tuple[str, ...]:
        if context.universe:
            pool = tuple(s for s in context.universe if s != self.bond_symbol)
            return pool if pool else self.etf_pool
        return self.etf_pool

    def _bond_target(self, reason: str) -> list[TargetPosition]:
        return [
            TargetPosition(
                symbol=self.bond_symbol,
                weight=1.0,
                score=0.0,
                reason=f"bond_{reason}",
            )
        ]

    @staticmethod
    def _load_closes(
        data: DataPortal,
        market: str,
        symbols: tuple[str, ...],
        trade_date: date,
        *,
        bars: int,
    ) -> dict[str, np.ndarray]:
        if not symbols:
            return {}
        history = _history_matrix(
            data,
            market=market,
            symbols=symbols,
            fields=["Close"],
            end_time=trade_date,
            bars=bars,
        )
        if history.empty:
            return {}
        out: dict[str, np.ndarray] = {}
        for symbol, group in history.groupby("symbol", sort=False):
            closes = group.sort_values("trade_date")["Close"].astype(float).to_numpy()
            if closes.size >= bars or (bars <= 3 and closes.size >= 3):
                out[str(symbol)] = closes
        return out


DEFAULT_JQ0020_ETF_POOL: tuple[str, ...] = (
    "510050",
    "510180",
    "510880",
    "159905",
    "159949",
    "510300",
    "510500",
    "515800",
    "159966",
    "159967",
    "513500",
    "513100",
    "510900",
    "512160",
    "511380",
)


DEFAULT_WIDE_BASE_ETF_POOL: tuple[str, ...] = (
    "510050",
    "510180",
    "510880",
    "159905",
    "159949",
    "510300",
    "510500",
    "159915",
    "588000",
    "512100",
    "159966",
    "159967",
)

DEFAULT_BREADTH_ETF_PROXIES: tuple[str, ...] = (
    "510300",
    "510500",
    "159915",
    "512100",
    "510880",
)

DEFAULT_WPETF_POOL: tuple[str, ...] = (
    "513100",
    "159915",
    "513520",
    "511010",
    "518880",
)


class JqEtfMomentumVolumeGateTop1Strategy:
    """jq-0020: ETF period momentum Top1 gated by volume/price emotion monitors."""

    id = "jq_etf_momentum_volume_gate_top1"
    name = "JoinQuant ETF动量+成交量gate Top1"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(
        self,
        *,
        bond_symbol: str = "511880",
        volume_benchmark: str = "510300",
        price_benchmark: str = "510500",
        period_bars: int = 13,
        etf_pool: tuple[str, ...] | None = None,
        volume_ma_window: int = 7,
        volume_below_days: int = 6,
    ) -> None:
        self.bond_symbol = bond_symbol
        self.volume_benchmark = volume_benchmark
        self.price_benchmark = price_benchmark
        self.period_bars = period_bars
        self.etf_pool = etf_pool or DEFAULT_JQ0020_ETF_POOL
        self.volume_ma_window = volume_ma_window
        self.volume_below_days = volume_below_days
        self.warmup_days = period_bars + volume_ma_window + volume_below_days + 5

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        pool = self._resolve_pool(context)
        if not pool:
            return []

        vol_hist = self._load_field(
            data, context.market, (self.volume_benchmark,), trade_date, "Volume", self.warmup_days
        )
        vol_arr = vol_hist.get(self.volume_benchmark)
        if vol_arr is not None and volume_emotion_monitor(
            vol_arr,
            ma_window=self.volume_ma_window,
            below_ma_days=self.volume_below_days,
        ) == -1:
            return self._bond_target("volume_emotion_bearish")

        px_hist = self._load_field(
            data, context.market, (self.price_benchmark,), trade_date, "Close", self.warmup_days
        )
        px_arr = px_hist.get(self.price_benchmark)
        if px_arr is not None and price_emotion_monitor(px_arr) in {-1, -2}:
            return self._bond_target("price_emotion_bearish")

        bars = self.period_bars + 1
        close_hist = self._load_field(data, context.market, pool, trade_date, "Close", bars)
        rows = []
        for symbol in pool:
            closes = close_hist.get(symbol)
            if closes is None:
                continue
            row = build_etf_momentum_row(symbol, closes, period_bars=self.period_bars)
            if row is not None:
                rows.append(row)
        if market_breadth_clear_signal(rows):
            return self._bond_target("etf_breadth_weak")

        best = select_best_etf_momentum(rows)
        if best is None or best.period_return_pct < 0.1 or best.ma_premium_pct < 0:
            return self._bond_target("no_qualified_etf")
        return [
            TargetPosition(
                symbol=best.symbol,
                weight=1.0,
                score=best.period_return_pct,
                reason="jq0020_momentum_top1",
            )
        ]

    def _resolve_pool(self, context: StrategyContext) -> tuple[str, ...]:
        if context.universe:
            pool = tuple(s for s in context.universe if s not in {self.bond_symbol})
            return pool if pool else self.etf_pool
        return self.etf_pool

    def _bond_target(self, reason: str) -> list[TargetPosition]:
        return [
            TargetPosition(
                symbol=self.bond_symbol,
                weight=1.0,
                score=0.0,
                reason=f"bond_{reason}",
            )
        ]

    @staticmethod
    def _load_field(
        data: DataPortal,
        market: str,
        symbols: tuple[str, ...],
        trade_date: date,
        field: str,
        bars: int,
    ) -> dict[str, np.ndarray]:
        if not symbols:
            return {}
        history = _history_matrix(
            data,
            market=market,
            symbols=symbols,
            fields=[field],
            end_time=trade_date,
            bars=bars,
        )
        if history.empty:
            return {}
        out: dict[str, np.ndarray] = {}
        col = field if field in history.columns else field.lower()
        for symbol, group in history.groupby("symbol", sort=False):
            values = group.sort_values("trade_date")[col].astype(float).to_numpy()
            if values.size >= min(bars, 3):
                out[str(symbol)] = values
        return out


class JqWideBaseEtfBbiRotationStrategy:
    """jq-0001 full pool: pick lowest BBI/price ETF when breadth gate allows."""

    id = "jq_wide_base_etf_bbi_rotation"
    name = "JoinQuant 宽基ETF BBI轮动"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(
        self,
        *,
        etf_pool: tuple[str, ...] | None = None,
        breadth_symbols: tuple[str, ...] = DEFAULT_BREADTH_ETF_PROXIES,
        bbi_periods: tuple[int, ...] = (3, 6, 12, 24),
        dapan_threshold: float = 0.0,
    ) -> None:
        self.etf_pool = etf_pool or DEFAULT_WIDE_BASE_ETF_POOL
        self.breadth_symbols = breadth_symbols
        self.bbi_periods = bbi_periods
        self.dapan_threshold = dapan_threshold
        self.warmup_days = max(bbi_periods) + 5

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        pool = self._resolve_pool(context)
        if not pool:
            return []

        breadth = JqIndustryEtfBbiRotationStrategy._load_closes(
            data, context.market, self.breadth_symbols, trade_date, bars=2
        )
        today_increase = breadth_best_return(breadth)
        if today_increase <= self.dapan_threshold:
            return []

        bars = max(self.bbi_periods) + 1
        etf_closes = JqIndustryEtfBbiRotationStrategy._load_closes(
            data, context.market, pool, trade_date, bars=bars
        )
        scored: list[tuple[str, float]] = []
        for symbol, closes in etf_closes.items():
            ratio = bbi_price_ratio(closes, self.bbi_periods)
            if np.isfinite(ratio):
                scored.append((symbol, ratio))
        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        target_symbol, target_ratio = scored[-1]
        if target_ratio >= 1.0:
            return []
        return [
            TargetPosition(
                symbol=target_symbol,
                weight=1.0,
                score=float(1.0 - target_ratio),
                reason="wide_base_bbi_min_ratio",
            )
        ]

    def _resolve_pool(self, context: StrategyContext) -> tuple[str, ...]:
        return tuple(context.universe) if context.universe else self.etf_pool


class WpetfMomentumTop1Strategy:
    """jq-0009 ETF leg: WPETF polynomial momentum, hold top-1."""

    id = "jq_wpetf_momentum_top1"
    name = "JoinQuant WPETF动量 Top1"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(
        self,
        *,
        lookback_days: int = 25,
        etf_pool: tuple[str, ...] | None = None,
        min_score: float = 0.01,
        blocked_symbols: tuple[str, ...] = (),
    ) -> None:
        self.lookback_days = lookback_days
        self.etf_pool = etf_pool or DEFAULT_WPETF_POOL
        self.min_score = min_score
        self.blocked_symbols = set(blocked_symbols)
        self.warmup_days = lookback_days + 5

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        pool = tuple(context.universe) if context.universe else self.etf_pool
        if not pool:
            return []
        history = _history_matrix(
            data,
            market=context.market,
            symbols=pool,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days,
        )
        if history.empty:
            return []
        best_symbol = ""
        best_score = float("-inf")
        for symbol, group in history.groupby("symbol", sort=False):
            sym = str(symbol)
            if sym in self.blocked_symbols:
                continue
            closes = group.sort_values("trade_date")["Close"].astype(float).to_numpy()
            if closes.size < self.lookback_days:
                continue
            score = wpetf_momentum_score(closes, m_days=self.lookback_days)
            if score > best_score:
                best_score = score
                best_symbol = sym
        if not best_symbol or best_score < self.min_score:
            return []
        return [
            TargetPosition(
                symbol=best_symbol,
                weight=1.0,
                score=float(best_score),
                reason="wpetf_momentum_top1",
            )
        ]


class JqEtfValueAveragingStrategy:
    """
    jq-0003 proxy: monthly value-averaging on one ETF with take-profit state.

    Uses internal state to approximate target-value DCA without broker cash API.
    """

    id = "jq_etf_value_averaging"
    name = "JoinQuant ETF价值平均定投"
    data_frequency = "1d"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        *,
        target_symbol: str = "159905",
        benchmark_symbol: str = "159905",
        entry_pullback_rate: float = 0.318,
        initial_weight: float = 0.7,
        month_growth_rate: float = 1.005,
        take_profit_return_pct: float = 40.0,
        take_profit_drawdown_pct: float = 9.0,
        full_take_profit_return_pct: float = 100.0,
    ) -> None:
        self.target_symbol = target_symbol
        self.benchmark_symbol = benchmark_symbol
        self.entry_pullback_rate = entry_pullback_rate
        self.initial_weight = initial_weight
        self.month_growth_rate = month_growth_rate
        self.take_profit_return_pct = take_profit_return_pct
        self.take_profit_drawdown_pct = take_profit_drawdown_pct
        self.full_take_profit_return_pct = full_take_profit_return_pct
        self.warmup_days = 400
        self._active = False
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._target_weight = 0.0
        self._last_month: tuple[int, int] | None = None

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        symbol = self._resolve_symbol(context)
        bench = _history_matrix(
            data,
            market=context.market,
            symbols=(self.benchmark_symbol, symbol),
            fields=["Close"],
            end_time=trade_date,
            bars=self.warmup_days,
        )
        if bench.empty:
            return []

        bench_close = self._symbol_closes(bench, self.benchmark_symbol)
        etf_close = self._symbol_closes(bench, symbol)
        if etf_close.size == 0:
            return []

        price = float(etf_close[-1])
        if price <= 0:
            return []

        ret_pct = (price / self._entry_price - 1.0) * 100.0 if self._active and self._entry_price > 0 else 0.0

        if self._active:
            self._peak_price = max(self._peak_price, price)
            drawdown_pct = (1.0 - price / self._peak_price) * 100.0 if self._peak_price > 0 else 0.0
            if ret_pct > self.full_take_profit_return_pct:
                self._reset()
                return []
            if ret_pct > self.take_profit_return_pct and drawdown_pct > self.take_profit_drawdown_pct:
                self._reset()
                return []

        month_key = (trade_date.year, trade_date.month)
        if self._last_month == month_key:
            if self._active and self._target_weight > 0:
                return [
                    TargetPosition(
                        symbol=symbol,
                        weight=self._target_weight,
                        score=ret_pct,
                        reason="value_avg_hold",
                    )
                ]
            return []

        self._last_month = month_key
        if not self._active:
            if bench_close.size == 0:
                return []
            rate1 = index_pullback_entry_rate(bench_close)
            if rate1 >= self.entry_pullback_rate:
                return []
            self._active = True
            self._entry_price = price
            self._peak_price = price
            self._target_weight = self.initial_weight
            return [
                TargetPosition(
                    symbol=symbol,
                    weight=self._target_weight,
                    score=float(rate1),
                    reason="value_avg_entry",
                )
            ]

        self._target_weight = min(1.0, self._target_weight * self.month_growth_rate)
        ret_pct = (price / self._entry_price - 1.0) * 100.0 if self._entry_price > 0 else 0.0
        return [
            TargetPosition(
                symbol=symbol,
                weight=self._target_weight,
                score=float(ret_pct),
                reason="value_avg_monthly",
            )
        ]

    def _reset(self) -> None:
        self._active = False
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._target_weight = 0.0

    def _resolve_symbol(self, context: StrategyContext) -> str:
        if context.universe:
            for sym in context.universe:
                if sym != self.benchmark_symbol:
                    return str(sym)
        return self.target_symbol

    @staticmethod
    def _symbol_closes(history: pd.DataFrame, symbol: str) -> np.ndarray:
        group = history[history["symbol"] == symbol]
        if group.empty:
            return np.array([])
        return group.sort_values("trade_date")["Close"].astype(float).to_numpy()


class JqMaConsolidationBreakoutStrategy:
    """
    jq-0033 proxy: MA consolidation breakout stock selection.

    Select stocks where MA13/34/55 are aligned bull + consolidated within 3%,
    and price breaks above 30-day high. Hold max 6 equal-weight positions.
    Sell on 3 consecutive closes below MA13.
    """

    id = "jq_ma_consolidation_breakout"
    name = "JoinQuant 均线黏合突破组合"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(
        self,
        *,
        fast_period: int = 13,
        mid_period: int = 34,
        slow_period: int = 55,
        consolidation_threshold: float = 0.03,
        breakout_lookback: int = 30,
        sell_ma_period: int = 13,
        sell_consecutive_days: int = 3,
        max_positions: int = 6,
    ) -> None:
        self.fast_period = fast_period
        self.mid_period = mid_period
        self.slow_period = slow_period
        self.consolidation_threshold = consolidation_threshold
        self.breakout_lookback = breakout_lookback
        self.sell_ma_period = sell_ma_period
        self.sell_consecutive_days = sell_consecutive_days
        self.max_positions = max_positions
        self.warmup_days = slow_period + breakout_lookback + sell_consecutive_days

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        bars = self.warmup_days + 5
        history = _history_matrix(
            data,
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=bars,
        )
        if history.empty:
            return []

        candidates: list[tuple[str, float]] = []
        for symbol, group in history.groupby("symbol", sort=False):
            group = group.sort_values("trade_date")
            closes = group["Close"].astype(float).to_numpy()
            if closes.size < bars:
                continue
            c = float(closes[-2]) if closes.size >= 2 else float(closes[-1])
            ma_fast = pd.Series(closes).rolling(self.fast_period, min_periods=self.fast_period).mean().to_numpy()
            ma_mid = pd.Series(closes).rolling(self.mid_period, min_periods=self.mid_period).mean().to_numpy()
            ma_slow = pd.Series(closes).rolling(self.slow_period, min_periods=self.slow_period).mean().to_numpy()
            if closes.size < 2:
                continue
            mf = float(ma_fast[-2])
            mm = float(ma_mid[-2])
            ms = float(ma_slow[-2])
            if any(np.isnan(x) for x in [mf, mm, ms]):
                continue
            if not (mf > mm > ms):
                continue
            if closes.size < 20:
                continue
            t10 = -11
            t19 = -20
            for offset in [-11, -20]:
                idx = offset
                if abs(idx) > len(ma_fast):
                    continue
                mf_i = float(ma_fast[idx])
                mm_i = float(ma_mid[idx])
                ms_i = float(ma_slow[idx])
                if any(np.isnan(x) for x in [mf_i, mm_i, ms_i]):
                    continue
                p1 = abs(mf_i - mm_i)
                p2 = abs(mm_i - ms_i)
                p3 = abs(mf_i - ms_i)
                if max(p1, p2, p3) > self.consolidation_threshold * c:
                    break
            else:
                lookback_start = max(0, closes.size - 2 - self.breakout_lookback)
                prev_max = float(np.max(closes[lookback_start:-1]))
                if c > prev_max:
                    candidates.append((str(symbol), c))
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[: self.max_positions]
        if not selected:
            return []
        weight = 1.0 / len(selected)
        return [
            TargetPosition(
                symbol=symbol,
                weight=weight,
                score=float(price),
                reason="ma_consolidation_breakout",
            )
            for symbol, price in selected
        ]


class JqSimpleMomentumStrategy:
    """
    jq-0059 风格: Simple N-day momentum stock selection.

    Rank universe by (close[-1] - close[-N]) / close[-N].
    Filter: price > min_price, non-ST, not suspended, listed > min_list_days.
    Top-K equal weight, weekly rebalance.
    """

    id = "jq_simple_momentum"
    name = "JoinQuant 简单动量选股"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        *,
        lookback_days: int = 91,
        top_k: int = 10,
        min_price: float = 5.0,
    ) -> None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.min_price = min_price
        self.warmup_days = lookback_days + 2

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        history = _history_matrix(
            data,
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days + 1,
        )
        if history.empty:
            return []
        scored: list[tuple[str, float]] = []
        for symbol, group in history.groupby("symbol", sort=False):
            group = group.sort_values("trade_date")
            closes = group["Close"].astype(float).to_numpy()
            if closes.size < self.lookback_days + 1:
                continue
            first = float(closes[0])
            last = float(closes[-1])
            if first <= 0 or last < self.min_price:
                continue
            ret = (last - first) / first
            if np.isfinite(ret):
                scored.append((str(symbol), ret))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[: self.top_k]
        if not selected:
            return []
        weight = 1.0 / len(selected)
        return [
            TargetPosition(
                symbol=symbol,
                weight=weight,
                score=score,
                reason=f"{self.lookback_days}d_momentum",
            )
            for symbol, score in selected
        ]
