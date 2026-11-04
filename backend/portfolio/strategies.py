"""Portfolio strategy interfaces and built-in strategies."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from backtest.jq_risk_control import compute_value_timing_gate_series
from marketdata.data_portal import DataPortal

from .models import StrategyContext, TargetPosition


def _score_weighted_targets(
    selected: list[tuple[str, float]],
    reason: str,
    *,
    min_score: float = -1e9,
    cash_fallback: bool = True,
) -> list[TargetPosition]:
    """Build target positions with score-weighted (not equal) sizing.

    When cash_fallback is True and all scores are below min_score, returns
    empty list (fully in cash).
    """
    if not selected:
        return []
    # Filter by min_score
    qualified = [(s, sc) for s, sc in selected if sc >= min_score]
    if cash_fallback and not qualified:
        return []
    if not qualified:
        qualified = selected
    # Score-weighted (shifted to ensure positivity)
    scores = np.array([sc for _, sc in qualified])
    score_min = scores.min()
    shifted = scores - score_min + abs(score_min) * 0.01 + 0.001
    weights = shifted / shifted.sum()
    return [
        TargetPosition(symbol=s, weight=float(w), score=float(sc), reason=reason)
        for (s, sc), w in zip(qualified, weights)
    ]


class PortfolioStrategy(Protocol):
    """Protocol implemented by portfolio strategies."""

    id: str
    name: str
    warmup_days: int
    data_frequency: str
    rebalance_frequency: str

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        """Generate target weights for a signal date."""


class MomentumTopKStrategy:
    """Select top K symbols by lookback return and equal weight them."""

    id = "momentum_top_k"
    name = "Momentum Top K"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(self, lookback_days: int = 20, top_k: int = 10) -> None:
        """Initialize strategy.

        Args:
            lookback_days: Number of daily bars used for return ranking.
            top_k: Maximum number of selected symbols.
        """
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.warmup_days = lookback_days

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        """Rank universe by lookback return and return equal weights."""
        if not context.universe:
            return []
        history = data.history(
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days + 1,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []
        valid = history.dropna(subset=["Close"])
        if valid.empty:
            return []
        grouped = valid.groupby("symbol", sort=False)["Close"].agg(first="first", last="last", count="count")
        grouped = grouped[
            (grouped["count"] >= self.lookback_days + 1)
            & (grouped["first"] > 0)
            & (grouped["last"] > 0)
        ]
        if grouped.empty:
            return []
        grouped["score"] = grouped["last"] / grouped["first"] - 1.0
        selected_scores = grouped["score"].nlargest(self.top_k)
        selected = list(selected_scores.items())

        return _score_weighted_targets(
            selected,
            f"{self.lookback_days}d_momentum",
            cash_fallback=True,
        )


class EtfRotationTop1Strategy:
    """Rotate into the strongest symbol by lookback return."""

    id = "etf_rotation_top1"
    name = "ETF Rotation Top1"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(self, lookback_days: int = 20) -> None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        self.lookback_days = lookback_days
        self.warmup_days = lookback_days

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        """Select one symbol with max lookback return."""
        if not context.universe:
            return []
        history = data.history(
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days + 1,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []
        valid = history.dropna(subset=["Close"])
        if valid.empty:
            return []
        grouped = valid.groupby("symbol", sort=False)["Close"].agg(first="first", last="last", count="count")
        grouped = grouped[
            (grouped["count"] >= self.lookback_days + 1)
            & (grouped["first"] > 0)
            & (grouped["last"] > 0)
        ]
        if grouped.empty:
            return []
        grouped["score"] = grouped["last"] / grouped["first"] - 1.0
        # Cash fallback: go to cash when no symbol has positive momentum
        best_score = float(grouped["score"].max())
        if best_score <= 0:
            return []
        top_symbol = str(grouped["score"].idxmax())
        return [TargetPosition(symbol=top_symbol, weight=1.0, score=best_score, reason="etf_rotation_top1")]


def _judge_more_average(closes: pd.Series) -> bool:
    """True when short MAs are below longer MAs (original skip-buy condition)."""
    if len(closes) < 30:
        return False
    ma5 = closes.tail(5).mean()
    ma10 = closes.tail(10).mean()
    ma20 = closes.tail(20).mean()
    ma30 = closes.tail(30).mean()
    return bool(ma5 < ma20 and ma10 < ma30)


class MeanReversionTopKStrategy:
    """均值回归策略：买入近期跌幅最大的 K 只股票（超跌反弹）。

    逻辑：
    - 质量过滤：排除收盘价 < min_price 的股票（退市股/仙股代理）
    - 双重择时门控：60日均线方向 + 20日收益率，仅趋势向上时开仓
    - 计算 lookback_days 收益率
    - 选出跌幅最深（收益率最低）的 K 只
    - 等权配置，周频调仓
    - 适合震荡市/上升趋势中的短期回调，不适合单边熊市
    """

    id = "mean_reversion_top_k"
    name = "均值回归 Top K（超跌反弹+质量过滤+双择时门控）"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        lookback_days: int = 20,
        top_k: int = 10,
        min_price: float = 2.0,
        ma_period: int = 60,
        gate_lookback: int = 20,
        gate_threshold: float = -0.05,
        benchmark_symbol: str = "510300",
    ) -> None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.min_price = min_price  # 最低价过滤（退市股代理）
        self.ma_period = ma_period  # 均线周期
        self.gate_lookback = gate_lookback
        self.gate_threshold = gate_threshold
        self.benchmark_symbol = benchmark_symbol
        self.warmup_days = max(lookback_days, gate_lookback, ma_period) + 10

    def _benchmark_allows_trade(self, trade_date: date, data: DataPortal, market: str) -> bool:
        """双重择时门控：60日均线方向 + 20日收益率，至少一个满足才开仓。"""
        bars_needed = max(self.ma_period, self.gate_lookback) + 10
        history = data.history(
            market=market,
            symbols=[self.benchmark_symbol],
            fields=["Close"],
            end_time=trade_date,
            bars=bars_needed,
            frequency=self.data_frequency,
        )
        if history.empty:
            return False
        closes = history.sort_values("trade_date")["Close"].astype(float)
        if len(closes) < max(self.ma_period, self.gate_lookback) + 1:
            return False
        current = float(closes.iloc[-1])

        # 条件1：价格 > MA60（中期趋势向上）
        ma_condition = False
        if len(closes) >= self.ma_period:
            ma = float(closes.tail(self.ma_period).mean())
            ma_condition = current > ma

        # 条件2：20日收益率 > -5%（短期未大跌）
        ret_condition = False
        if len(closes) >= self.gate_lookback + 1:
            past = float(closes.iloc[-self.gate_lookback - 1])
            if past > 0:
                benchmark_return = current / past - 1.0
                ret_condition = benchmark_return > self.gate_threshold

        # 至少一个条件满足才允许开仓
        return ma_condition or ret_condition

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        # 双重择时门控：趋势向下时空仓
        if not self._benchmark_allows_trade(trade_date, data, context.market):
            return []
        history = data.history(
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=self.lookback_days + 1,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []
        valid = history.dropna(subset=["Close"])
        if valid.empty:
            return []
        grouped = valid.groupby("symbol", sort=False)["Close"].agg(
            first="first", last="last", count="count"
        )
        # 质量过滤：排除低价股（退市股/仙股代理）+ 数据充分
        grouped = grouped[
            (grouped["count"] >= self.lookback_days + 1)
            & (grouped["first"] > 0)
            & (grouped["last"] >= self.min_price)  # 关键过滤：最新价 >= min_price
        ]
        if grouped.empty:
            return []
        grouped["score"] = grouped["last"] / grouped["first"] - 1.0
        # 额外过滤：排除跌幅 > 50% 的异常股票（退市/重大利空）
        grouped = grouped[grouped["score"] >= -0.50]
        if grouped.empty:
            return []
        # 选出跌幅最大的 K 只（收益率最低的）
        selected_scores = grouped["score"].nsmallest(self.top_k)
        selected = list(selected_scores.items())

        return _score_weighted_targets(
            selected,
            f"{self.lookback_days}d_reversion+MA{self.ma_period}+price>{self.min_price}",
            cash_fallback=False,
        )


class MultiFactorTopKStrategy:
    """多因子选股策略：综合动量、波动率、成交量稳定性打分。

    因子权重：
    - 动量因子(20d收益率): 40% — 中等周期趋势
    - 波动率因子(20d收益标准差): 30% — 低波动优选
    - 成交量稳定性(成交量变异系数): 30% — 成交稳定优于暴涨暴跌

    每个因子在横截面上 z-score 标准化后加权合成。
    """

    id = "multi_factor_top_k"
    name = "多因子选股 Top K"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        lookback_days: int = 20,
        top_k: int = 10,
        momentum_weight: float = 0.4,
        volatility_weight: float = 0.3,
        volume_weight: float = 0.3,
    ) -> None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.warmup_days = lookback_days + 10
        self.momentum_weight = momentum_weight
        self.volatility_weight = volatility_weight
        self.volume_weight = volume_weight

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        bars = self.lookback_days + 5
        history = data.history(
            market=context.market,
            symbols=context.universe,
            fields=["Close", "Volume"],
            end_time=trade_date,
            bars=bars,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []

        factors: dict[str, dict[str, float]] = {}
        for symbol, group in history.groupby("symbol", sort=False):
            group = group.sort_values("trade_date")
            if len(group) < self.lookback_days:
                continue
            close = group["Close"].astype(float).tail(self.lookback_days + 1)
            vol = group["Volume"].astype(float).tail(self.lookback_days)
            if len(close) < self.lookback_days + 1 or close.iloc[0] <= 0:
                continue
            ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
            daily_ret = close.pct_change().dropna()
            vol_std = float(daily_ret.std()) if len(daily_ret) > 1 else 1.0
            if vol_std <= 0 or not np.isfinite(vol_std):
                vol_std = 1.0
            avg_vol = float(vol.mean())
            std_vol = float(vol.std()) if len(vol) > 1 else 1.0
            vol_cv = std_vol / avg_vol if avg_vol > 0 else 1.0
            if not np.isfinite(vol_cv):
                vol_cv = 1.0
            factors[str(symbol)] = {
                "momentum": ret,
                "volatility": vol_std,
                "volume_cv": vol_cv,
            }

        if len(factors) < 3:
            return []

        # 横截面 z-score 标准化
        sym_list = list(factors.keys())
        mom_vals = np.array([factors[s]["momentum"] for s in sym_list])
        vol_vals = np.array([factors[s]["volatility"] for s in sym_list])
        cv_vals = np.array([factors[s]["volume_cv"] for s in sym_list])

        def _zscore(arr: np.ndarray) -> np.ndarray:
            mu, sigma = arr.mean(), arr.std()
            if sigma <= 0 or not np.isfinite(sigma):
                return np.zeros_like(arr)
            return (arr - mu) / sigma

        mom_z = _zscore(mom_vals)
        vol_z = _zscore(vol_vals)
        cv_z = _zscore(cv_vals)

        # 合成综合分: 动量正向, 波动率反向(低波动好), 变异系数反向(稳定好)
        composite = (
            self.momentum_weight * mom_z
            - self.volatility_weight * vol_z
            - self.volume_weight * cv_z
        )

        top_idx = np.argsort(-composite)[: self.top_k]
        selected = [(sym_list[i], float(composite[i])) for i in top_idx]

        return _score_weighted_targets(
            selected,
            "multi_factor(mom+vol+vol_cv)",
            cash_fallback=True,
        )


class RiskAdjustedMomentumStrategy:
    """风险调整动量策略：收益/波动率排名。

    与纯动量不同，用 收益率/波动率 作为夏普近似，
    避免选中暴涨暴跌的投机股。
    """

    id = "risk_adj_momentum_top_k"
    name = "风险调整动量 Top K"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(self, lookback_days: int = 20, top_k: int = 10) -> None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.warmup_days = lookback_days + 5

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        bars = self.lookback_days + 5
        history = data.history(
            market=context.market,
            symbols=context.universe,
            fields=["Close"],
            end_time=trade_date,
            bars=bars,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []

        scored: list[tuple[str, float]] = []
        for symbol, group in history.groupby("symbol", sort=False):
            group = group.sort_values("trade_date")
            close = group["Close"].astype(float).tail(self.lookback_days + 1)
            if len(close) < self.lookback_days + 1 or close.iloc[0] <= 0:
                continue
            ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
            daily_ret = close.pct_change().dropna()
            vol = float(daily_ret.std()) if len(daily_ret) > 1 else 1.0
            if vol <= 0 or not np.isfinite(vol):
                vol = 1.0
            risk_adj_score = ret / vol  # 夏普近似
            if np.isfinite(risk_adj_score):
                scored.append((str(symbol), risk_adj_score))

        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[: self.top_k]

        return _score_weighted_targets(
            selected,
            f"{self.lookback_days}d_return/volatility",
            cash_fallback=True,
        )


class JqValueTimingTopKStrategy:
    """
    jq-0002 proxy without fundamentals: liquidity/size ranking + HS300 timing gate.

    Replaces ``get_fundamentals`` filters with OHLCV proxies:
    - circulating cap sort -> average ``amount`` (fallback Volume*Close)
    - size filter -> avg amount above universe mean * ratio
    - revenue/EPS growth -> 60d return band
    - buy filters -> 5d drawdown cap + ``judge_More_average``
    """

    id = "jq_value_timing_top_k"
    name = "JoinQuant 价值投资大盘择时组合（无财报）"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        *,
        top_k: int = 5,
        lookback_days: int = 20,
        ma_period: int = 250,
        benchmark_symbol: str = "510300",
        min_amount_ratio: float = 1.2,
        ret_5d_min: float = -0.10,
        ret_60d_min: float = 0.06,
        ret_60d_max: float = 0.50,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if ma_period <= 0:
            raise ValueError("ma_period must be positive")
        self.top_k = top_k
        self.lookback_days = lookback_days
        self.ma_period = ma_period
        self.benchmark_symbol = benchmark_symbol
        self.min_amount_ratio = min_amount_ratio
        self.ret_5d_min = ret_5d_min
        self.ret_60d_max = ret_60d_max
        self.ret_60d_min = ret_60d_min
        self.warmup_days = max(lookback_days, ma_period, 60) + 5
        self._last_selection_month: tuple[int, int] | None = None
        self._monthly_candidates: list[tuple[str, float]] = []

    def generate_targets(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[TargetPosition]:
        if not context.universe:
            return []
        if not self._benchmark_allows_trade(trade_date, data, context.market):
            return []

        month_key = (trade_date.year, trade_date.month)
        if month_key != self._last_selection_month:
            self._monthly_candidates = self._select_monthly_candidates(trade_date, data, context)
            self._last_selection_month = month_key

        selected = self._apply_buy_filters(trade_date, data, context, self._monthly_candidates)

        return _score_weighted_targets(
            selected,
            "jq_value_timing_top_k",
            cash_fallback=True,
        )

    def _benchmark_allows_trade(self, trade_date: date, data: DataPortal, market: str) -> bool:
        bars = self.ma_period + 95
        history = data.history(
            market=market,
            symbols=[self.benchmark_symbol],
            fields=["Close"],
            end_time=trade_date,
            bars=bars,
            frequency=self.data_frequency,
        )
        if history.empty:
            return False
        closes = history.sort_values("trade_date")["Close"].astype(float)
        if len(closes) < self.ma_period + 2:
            return False
        gate = compute_value_timing_gate_series(closes.reset_index(drop=True), ma_period=self.ma_period)
        return bool(gate.iloc[-1])

    def _select_monthly_candidates(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
    ) -> list[tuple[str, float]]:
        bars = max(self.lookback_days, 60) + 1
        history = data.history(
            market=context.market,
            symbols=context.universe,
            fields=["Close", "Volume", "amount"],
            end_time=trade_date,
            bars=bars,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []
        liquidity_rows: list[tuple[str, float, pd.Series]] = []
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
            liquidity_rows.append((str(symbol), avg_liq, close))
        if not liquidity_rows:
            return []
        universe_mean_liq = float(np.mean([row[1] for row in liquidity_rows]))
        threshold = universe_mean_liq * self.min_amount_ratio
        filtered: list[tuple[str, float]] = []
        for symbol, avg_liq, close in liquidity_rows:
            if avg_liq < threshold:
                continue
            ret_60 = float(close.iloc[-1] / close.iloc[0] - 1.0)
            if not np.isfinite(ret_60) or ret_60 < self.ret_60d_min or ret_60 > self.ret_60d_max:
                continue
            filtered.append((symbol, avg_liq))
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered

    def _apply_buy_filters(
        self,
        trade_date: date,
        data: DataPortal,
        context: StrategyContext,
        candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        symbols = [symbol for symbol, _ in candidates[: self.top_k * 3]]
        history = data.history(
            market=context.market,
            symbols=symbols,
            fields=["Close"],
            end_time=trade_date,
            bars=6,
            frequency=self.data_frequency,
        )
        if history.empty:
            return []
        score_map = {symbol: score for symbol, score in candidates}
        selected: list[tuple[str, float]] = []
        for symbol in symbols:
            if len(selected) >= self.top_k:
                break
            group = history[history["symbol"] == symbol].sort_values("trade_date")
            closes = group["Close"].astype(float)
            if len(closes) < 6:
                continue
            ret_5 = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
            if ret_5 < self.ret_5d_min:
                continue
            long_hist = data.history(
                market=context.market,
                symbols=[symbol],
                fields=["Close"],
                end_time=trade_date,
                bars=25,
                frequency=self.data_frequency,
            )
            if long_hist.empty:
                continue
            long_closes = long_hist.sort_values("trade_date")["Close"].astype(float)
            if _judge_more_average(long_closes):
                continue
            selected.append((symbol, score_map.get(symbol, 0.0)))
        return selected

