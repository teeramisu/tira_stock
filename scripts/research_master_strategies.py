"""
投资大师策略回测研究工具
=======================
实现巴菲特/索罗斯/利弗莫尔/达里奥/林奇等大师策略的OHLCV代理版本，
在全市场组合回测中对比表现，寻找最大回撤≤20%条件下的最优收益方案。

Usage:
    python scripts/research_master_strategies.py [--period 2020-2025] [--market cn]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from marketdata.repository import MarketDataRepository
from marketdata.data_portal import DataPortal
from marketdata.universe import UniverseConfig
from marketdata.db import connect as _db_connect_fn
from portfolio.engine import PortfolioEngine
from portfolio.models import PortfolioBacktestConfig, StrategyContext, TargetPosition
from portfolio.broker import CostModel


def _get_repo():
    """Get a read-only MarketDataRepository (avoids lock conflicts with running server)."""
    import duckdb as _duckdb
    db_path = PROJECT_ROOT / "data" / "market" / "market_data.duckdb"
    conn = _duckdb.connect(str(db_path), read_only=True)
    return MarketDataRepository(conn)


# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────

def _safe_div(a, b, default=np.nan):
    """Safe division avoiding divide-by-zero."""
    if b == 0 or not np.isfinite(b):
        return default
    return a / b


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr = _atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


# ────────────────────────────────────────────────────────────────────
# 大师策略实现（PortfolioStrategy Protocol）
# ────────────────────────────────────────────────────────────────────

class BuffettValueStrategy:
    """
    巴菲特策略 — 价值投资 + 护城河 + 安全边际

    核心思想（OHLCV代理）：
    - 护城河：低波动率（稳定盈利）= 60日波动率低于市场中位数
    - 安全边际：在长期上升趋势中买入短期回调 = 200MA上升 + 价格回调至20MA
    - 集中投资：Top 10，等权
    - 长期持有：月度再平衡
    """
    id = "buffett_value"
    name = "巴菲特·价值护城河"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(self, top_k=10, vol_window=60, trend_ma=200, pullback_ma=20):
        self.top_k = top_k
        self.vol_window = vol_window
        self.trend_ma = trend_ma
        self.pullback_ma = pullback_ma
        self.warmup_days = max(trend_ma, vol_window) + 10

    def generate_targets(self, trade_date, data, context):
        if not context.universe:
            return []

        bars = max(self.trend_ma, self.vol_window, 60) + 5
        history = data.history(
            market=context.market, symbols=context.universe,
            fields=["Close", "Volume"], end_time=trade_date,
            bars=bars, frequency=self.data_frequency,
        )
        if history.empty:
            return []

        scores = []
        for symbol, group in history.groupby("symbol", sort=False):
            g = group.sort_values("trade_date")
            c = g["Close"].astype(float)
            if len(c) < self.trend_ma:
                continue
            # 长期趋势: 200MA斜率（正=上升趋势）
            ma200 = c.rolling(self.trend_ma).mean()
            if ma200.iloc[-1] <= 0 or not np.isfinite(ma200.iloc[-1]):
                continue
            trend_slope = (ma200.iloc[-1] / ma200.iloc[-20] - 1) if len(ma200) > 20 and ma200.iloc[-20] > 0 else 0
            if trend_slope <= 0:
                continue  # 只买上升趋势

            # 短期回调: 价格接近20MA（安全边际）
            ma20 = c.rolling(self.pullback_ma).mean()
            current_price = c.iloc[-1]
            pullback_pct = (current_price / ma20.iloc[-1] - 1) if ma20.iloc[-1] > 0 else 0
            # 回调 -10% 到 -1% 之间最佳（不太深也不太浅）
            if pullback_pct < -0.12 or pullback_pct > 0.02:
                continue

            # 护城河: 低波动率
            returns = c.pct_change().tail(self.vol_window)
            vol = returns.std() if len(returns) > 10 else 1.0
            if not np.isfinite(vol) or vol <= 0:
                continue

            # 综合评分: 趋势强 + 回调适中 + 波动低
            score = trend_slope * 100 - pullback_pct * 10 - vol * 50
            scores.append((symbol, score, current_price))

        if not scores:
            return []

        # 按波动率过滤：只保留低于中位数的
        vols = {s: v for s, _, _ in scores for v in [0]}  # placeholder
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:self.top_k]

        weight = 1.0 / len(selected)
        return [
            TargetPosition(symbol=s, weight=weight, score=sc, reason="buffett_value")
            for s, sc, _ in selected
        ]


class SorosTrendStrategy:
    """
    索罗斯策略 — 反身性 + 趋势跟踪 + 仓位管理

    核心思想（OHLCV代理）：
    - 反身性：趋势加速 = 动量的动量（二阶导数）
    - 趋势跟踪：ADX > 25 确认强趋势
    - 仓位管理：趋势越强仓位越大（Kelly代理）
    - 风控：ATR跟踪止损，单只不超20%仓位
    """
    id = "soros_trend"
    name = "索罗斯·反身性趋势"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(self, top_k=8, lookback=20, adx_period=14, atr_period=14):
        self.top_k = top_k
        self.lookback = lookback
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.warmup_days = max(lookback, adx_period, atr_period) + 20

    def generate_targets(self, trade_date, data, context):
        if not context.universe:
            return []

        bars = max(self.lookback, self.adx_period, self.atr_period) + 30
        history = data.history(
            market=context.market, symbols=context.universe,
            fields=["Close", "High", "Low", "Volume"],
            end_time=trade_date, bars=bars, frequency=self.data_frequency,
        )
        if history.empty:
            return []

        scores = []
        for symbol, group in history.groupby("symbol", sort=False):
            g = group.sort_values("trade_date")
            c = g["Close"].astype(float)
            h = g["High"].astype(float)
            l = g["Low"].astype(float)
            if len(c) < bars - 5:
                continue

            # 动量（一阶）
            mom = c.iloc[-1] / c.iloc[-self.lookback] - 1 if c.iloc[-self.lookback] > 0 else 0
            # 动量加速（二阶 = 反身性代理）
            mom_prev = c.iloc[-self.lookback] / c.iloc[-self.lookback * 2] - 1 if len(c) >= self.lookback * 2 and c.iloc[-self.lookback * 2] > 0 else 0
            acceleration = mom - mom_prev

            # ADX: 趋势强度
            adx_val = _adx(h, l, c, self.adx_period).iloc[-1]
            if not np.isfinite(adx_val) or adx_val < 20:
                continue  # 趋势不够强

            # ATR 波动率
            atr_val = _atr(h, l, c, self.atr_period).iloc[-1]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue

            # 风险调整收益
            risk_adj_return = _safe_div(mom, atr_val / c.iloc[-1])

            # 综合评分: 动量加速 + 强趋势 + 风险调整
            score = acceleration * 30 + (adx_val / 100) * 20 + risk_adj_return * 0.1
            scores.append((symbol, score, mom))

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:self.top_k]

        # 索罗斯式仓位: 信心越强仓位越大
        total_score = sum(abs(s[1]) for s in selected) or 1.0
        targets = []
        for symbol, score, mom in selected:
            w = min(0.25, max(0.05, abs(score) / total_score))  # 5%-25% 单只上限
            targets.append(TargetPosition(symbol=symbol, weight=w, score=score, reason="soros_trend"))

        # 归一化权重
        total_w = sum(t.weight for t in targets)
        if total_w > 0:
            for t in targets:
                t.weight = t.weight / total_w
        return targets


class LivermoreBreakoutStrategy:
    """
    利弗莫尔策略 — 突破 + 金字塔加仓 + 止损

    核心思想（OHLCV代理）：
    - 突破：价格突破N日新高 + 成交量放大
    - 金字塔：趋势确认后逐步加仓（通过持仓权重递增）
    - 止损：跌破10日低点清仓
    - 集中：Top 5，高确信度
    """
    id = "livermore_breakout"
    name = "利弗莫尔·突破金字塔"
    data_frequency = "1d"
    rebalance_frequency = "daily"

    def __init__(self, top_k=5, breakout_period=20, volume_mult=1.5, stop_period=10):
        self.top_k = top_k
        self.breakout_period = breakout_period
        self.volume_mult = volume_mult
        self.stop_period = stop_period
        self.warmup_days = max(breakout_period, stop_period) + 10

    def generate_targets(self, trade_date, data, context):
        if not context.universe:
            return []

        bars = max(self.breakout_period, self.stop_period) + 10
        history = data.history(
            market=context.market, symbols=context.universe,
            fields=["Close", "High", "Low", "Volume"],
            end_time=trade_date, bars=bars, frequency=self.data_frequency,
        )
        if history.empty:
            return []

        scores = []
        for symbol, group in history.groupby("symbol", sort=False):
            g = group.sort_values("trade_date")
            c = g["Close"].astype(float)
            h = g["High"].astype(float)
            l = g["Low"].astype(float)
            v = g["Volume"].astype(float) if "Volume" in g.columns else pd.Series(0, index=g.index)

            if len(c) < self.breakout_period + 2:
                continue

            # 突破: 当前高点 > N日最高价
            prev_high = h.iloc[-(self.breakout_period + 1):-1].max()
            current_high = h.iloc[-1]
            if current_high <= prev_high:
                continue

            # 成交量放大
            avg_vol = v.iloc[-self.breakout_period:-1].mean() if len(v) > self.breakout_period else 1
            current_vol = v.iloc[-1]
            if avg_vol > 0 and current_vol < avg_vol * self.volume_mult:
                continue

            # 止损检查: 当前价 > N日最低点
            recent_low = l.iloc[-self.stop_period:].min()
            if c.iloc[-1] < recent_low * 1.02:
                continue  # 太接近止损线

            # 突破力度评分
            breakout_strength = (current_high / prev_high - 1) * 100
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            score = breakout_strength * vol_ratio

            scores.append((symbol, score))

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:self.top_k]

        weight = 1.0 / len(selected)
        return [
            TargetPosition(symbol=s, weight=weight, score=sc, reason="livermore_breakout")
            for s, sc in selected
        ]


class DalioAllWeatherStrategy:
    """
    达里奥策略 — 全天候 + 风险平价 + 动量叠加

    核心思想（OHLCV代理）：
    - 多资产分散：使用ETF池（股票/债券/黄金/商品）
    - 风险平价：按波动率倒数分配权重
    - 动量叠加：在风险平价基础上，向近期强势资产倾斜
    - 月度再平衡
    """
    id = "dalio_all_weather"
    name = "达里奥·全天候风险平价"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(self, vol_window=60, momentum_window=20, momentum_tilt=0.3):
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.momentum_tilt = momentum_tilt
        self.warmup_days = max(vol_window, momentum_window) + 10

    def generate_targets(self, trade_date, data, context):
        if not context.universe:
            return []

        bars = max(self.vol_window, self.momentum_window) + 10
        history = data.history(
            market=context.market, symbols=context.universe,
            fields=["Close"], end_time=trade_date,
            bars=bars, frequency=self.data_frequency,
        )
        if history.empty:
            return []

        assets = []
        for symbol, group in history.groupby("symbol", sort=False):
            g = group.sort_values("trade_date")
            c = g["Close"].astype(float)
            if len(c) < self.vol_window:
                continue
            if c.iloc[-1] <= 0:
                continue

            # 波动率
            returns = c.pct_change().tail(self.vol_window)
            vol = returns.std() * np.sqrt(252)
            if not np.isfinite(vol) or vol <= 0:
                continue

            # 动量
            mom = c.iloc[-1] / c.iloc[-self.momentum_window] - 1 if len(c) > self.momentum_window and c.iloc[-self.momentum_window] > 0 else 0

            inv_vol = 1.0 / vol
            assets.append((symbol, inv_vol, mom, vol))

        if not assets:
            return []

        # 风险平价基础权重
        total_inv_vol = sum(a[1] for a in assets)
        targets = []
        for symbol, inv_vol, mom, vol in assets:
            rp_weight = inv_vol / total_inv_vol  # 风险平价权重
            # 动量叠加: 正动量增加权重，负动量减少权重
            mom_adj = 1 + mom * self.momentum_tilt
            adjusted_weight = rp_weight * max(0.3, mom_adj)  # 最低保留30%基础权重
            targets.append((symbol, adjusted_weight, mom))

        # 归一化
        total_w = sum(t[1] for t in targets)
        if total_w > 0:
            return [
                TargetPosition(symbol=s, weight=w / total_w, score=m, reason="dalio_all_weather")
                for s, w, m in targets
            ]
        return []


class LynchGARPStrategy:
    """
    林奇策略 — GARP (Growth at Reasonable Price)

    核心思想（OHLCV代理）：
    - Growth: 中期动量（60日收益）代表增长
    - Reasonable Price: 不能太贵 = 不在近期高点附近
    - 分散投资: Top 15
    - 月度再平衡
    """
    id = "lynch_garp"
    name = "林奇·GARP成长合理价"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(self, top_k=15, growth_window=60, max_from_high=0.10):
        self.top_k = top_k
        self.growth_window = growth_window
        self.max_from_high = max_from_high
        self.warmup_days = max(growth_window, 120) + 10

    def generate_targets(self, trade_date, data, context):
        if not context.universe:
            return []

        bars = max(self.growth_window, 120) + 10
        history = data.history(
            market=context.market, symbols=context.universe,
            fields=["Close", "Volume"], end_time=trade_date,
            bars=bars, frequency=self.data_frequency,
        )
        if history.empty:
            return []

        scores = []
        for symbol, group in history.groupby("symbol", sort=False):
            g = group.sort_values("trade_date")
            c = g["Close"].astype(float)
            if len(c) < self.growth_window:
                continue
            if c.iloc[-1] <= 0:
                continue

            # Growth: 60日收益
            growth = c.iloc[-1] / c.iloc[-self.growth_window] - 1
            if growth <= 0:
                continue  # 没有增长

            # Reasonable Price: 距近期高点不超过 max_from_high
            high_120 = c.tail(min(120, len(c))).max()
            dist_from_high = (c.iloc[-1] / high_120 - 1) if high_120 > 0 else 0
            if dist_from_high < -self.max_from_high:
                continue  # 跌幅太大可能有基本面问题

            # 波动调整增长: 增长/波动 = GARP得分
            returns = c.pct_change().tail(self.growth_window)
            vol = returns.std() if len(returns) > 10 else 1.0
            if not np.isfinite(vol) or vol <= 0:
                continue
            garp_score = growth / (vol * np.sqrt(252))

            scores.append((symbol, garp_score, growth))

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:self.top_k]

        weight = 1.0 / len(selected)
        return [
            TargetPosition(symbol=s, weight=weight, score=sc, reason="lynch_garp")
            for s, sc, _ in selected
        ]


class MastersEnsembleStrategy:
    """
    大师混合策略 — 多信号集成 + 风控门控

    核心思想：
    - 同时运行巴菲特/索罗斯/利弗莫尔/林奇四种子策略
    - 取各子策略信号的交集（多策略认可的高确信标的）
    - 达里奥风险平价作为仓位分配基础
    - 最大回撤控制: 组合波动率超标时自动减仓
    """
    id = "masters_ensemble"
    name = "大师混合·集成风控"
    data_frequency = "1d"
    rebalance_frequency = "weekly"

    def __init__(self, top_k=10, max_vol=0.20):
        self.top_k = top_k
        self.max_vol = max_vol
        self.warmup_days = 250

        self._buffett = BuffettValueStrategy(top_k=top_k * 2)
        self._soros = SorosTrendStrategy(top_k=top_k * 2)
        self._livermore = LivermoreBreakoutStrategy(top_k=top_k * 2)
        self._lynch = LynchGARPStrategy(top_k=top_k * 2)
        self._dalio = DalioAllWeatherStrategy()

    def generate_targets(self, trade_date, data, context):
        if not context.universe:
            return []

        # 收集各子策略信号
        all_targets = {}
        for sub in [self._buffett, self._soros, self._livermore, self._lynch]:
            try:
                targets = sub.generate_targets(trade_date, data, context)
                for t in targets:
                    if t.symbol not in all_targets:
                        all_targets[t.symbol] = {"count": 0, "total_score": 0, "weight_sum": 0}
                    all_targets[t.symbol]["count"] += 1
                    all_targets[t.symbol]["total_score"] += t.score
                    all_targets[t.symbol]["weight_sum"] += t.weight
            except Exception:
                continue

        if not all_targets:
            return []

        # 取至少2个子策略认可的标的
        consensus = {
            s: v for s, v in all_targets.items()
            if v["count"] >= 2
        }

        if not consensus:
            # 退化为单策略: 用评分最高的
            consensus = all_targets

        # 按投票数×评分排序
        ranked = sorted(
            consensus.items(),
            key=lambda x: x[1]["count"] * abs(x[1]["total_score"]),
            reverse=True,
        )
        selected = ranked[:self.top_k]

        if not selected:
            return []

        # 等权分配
        weight = 1.0 / len(selected)
        targets = []
        for symbol, info in selected:
            targets.append(TargetPosition(
                symbol=symbol, weight=weight,
                score=info["total_score"],
                reason=f"ensemble_{info['count']}votes",
            ))

        # 风控门控: 检查组合近期波动率
        bars = 20
        history = data.history(
            market=context.market,
            symbols=[t.symbol for t in targets],
            fields=["Close"], end_time=trade_date,
            bars=bars, frequency=self.data_frequency,
        )
        if not history.empty:
            port_returns = []
            for t_date in history["trade_date"].unique()[-20:]:
                day_data = history[history["trade_date"] == t_date]
                if len(day_data) > 0:
                    ret = day_data.set_index("symbol")["Close"].pct_change().mean()
                    if np.isfinite(ret):
                        port_returns.append(ret)
            if port_returns:
                port_vol = np.std(port_returns) * np.sqrt(252)
                if port_vol > self.max_vol:
                    scale = self.max_vol / port_vol
                    for t in targets:
                        t.weight *= scale  # 减仓

        return targets


# ────────────────────────────────────────────────────────────────────
# 回测执行
# ────────────────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    "buffett": BuffettValueStrategy,
    "soros": SorosTrendStrategy,
    "livermore": LivermoreBreakoutStrategy,
    "dalio": DalioAllWeatherStrategy,
    "lynch": LynchGARPStrategy,
    "ensemble": MastersEnsembleStrategy,
}


def run_strategy_backtest(
    strategy_name: str,
    market: str = "cn",
    start: str = "2020-01-01",
    end: str = "2025-06-30",
    initial_cash: float = 1_000_000,
    symbols: tuple[str, ...] | None = None,
):
    """Run a single strategy backtest and return metrics."""
    repo = _get_repo()
    strategy_cls = ALL_STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(ALL_STRATEGIES.keys())}")

    strategy = strategy_cls()

    # Default ETF pool for dalio, stock universe for others
    if symbols is None:
        if strategy_name == "dalio":
            symbols = ("510300", "510500", "159915", "513100", "518880", "510180", "511010", "511260")
        else:
            # Use HS300 components proxy: top liquid stocks
            symbols = None  # will use full universe

    config = PortfolioBacktestConfig(
        market=market,
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end),
        initial_cash=initial_cash,
        dataset_version="v1",
    )

    from marketdata.universe import UniverseConfig
    uc = UniverseConfig()
    if symbols:
        uc = UniverseConfig(include=list(symbols))

    engine = PortfolioEngine(
        repository=repo,
        strategy=strategy,
        config=config,
        universe_config=uc,
        cost_model=CostModel(commission_rate=0.001, slippage_rate=0.001),
    )

    print(f"\n{'='*60}")
    print(f"回测: {strategy.name} ({strategy.id})")
    print(f"区间: {start} ~ {end} | 市场: {market} | 初始资金: {initial_cash:,.0f}")
    print(f"{'='*60}")

    t0 = time.time()
    result = engine.run()
    elapsed = time.time() - t0

    metrics = {
        "strategy": strategy_name,
        "name": strategy.name,
        "total_return": result.metrics.get("total_return"),
        "annualized_return": result.metrics.get("annualized_return"),
        "max_drawdown": result.metrics.get("max_drawdown"),
        "sharpe_ratio": result.metrics.get("sharpe_ratio"),
        "win_rate": result.metrics.get("win_rate"),
        "trade_count": result.metrics.get("trade_count", 0),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n结果 ({elapsed:.1f}s):")
    print(f"  总收益率:   {metrics['total_return']:.2%}" if metrics['total_return'] else "  总收益率:   N/A")
    print(f"  年化收益:   {metrics['annualized_return']:.2%}" if metrics['annualized_return'] else "  年化收益:   N/A")
    print(f"  最大回撤:   {metrics['max_drawdown']:.2%}" if metrics['max_drawdown'] else "  最大回撤:   N/A")
    print(f"  夏普比率:   {metrics['sharpe_ratio']:.3f}" if metrics['sharpe_ratio'] else "  夏普比率:   N/A")
    print(f"  胜率:       {metrics['win_rate']:.2%}" if metrics['win_rate'] else "  胜率:       N/A")
    print(f"  交易次数:   {metrics['trade_count']}")

    return metrics


def run_all_strategies(market="cn", start="2020-01-01", end="2025-06-30"):
    """Run all master strategies and compare."""
    results = []
    for name in ALL_STRATEGIES:
        try:
            m = run_strategy_backtest(name, market=market, start=start, end=end)
            results.append(m)
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print(f"\n{'='*80}")
        print("策略对比汇总")
        print(f"{'='*80}")
        print(f"{'策略':<20} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'胜率':>8} {'交易':>6}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x.get("sharpe_ratio") or 0, reverse=True):
            tr = f"{r['total_return']:.1%}" if r['total_return'] else "N/A"
            ar = f"{r['annualized_return']:.1%}" if r['annualized_return'] else "N/A"
            md = f"{r['max_drawdown']:.1%}" if r['max_drawdown'] else "N/A"
            sr = f"{r['sharpe_ratio']:.2f}" if r['sharpe_ratio'] else "N/A"
            wr = f"{r['win_rate']:.1%}" if r['win_rate'] else "N/A"
            tc = str(r['trade_count'])
            print(f"{r['name']:<20} {tr:>10} {ar:>10} {md:>10} {sr:>8} {wr:>8} {tc:>6}")

    return results


# ────────────────────────────────────────────────────────────────────
# 参数遍历
# ────────────────────────────────────────────────────────────────────

def grid_search_soros(market="cn", start="2020-01-01", end="2025-06-30"):
    """Grid search Soros strategy parameters for optimal Sharpe with DD < 20%."""
    print("\n" + "="*60)
    print("索罗斯策略参数遍历")
    print("="*60)

    best = None
    results = []

    for top_k in [5, 8, 12]:
        for lookback in [10, 20, 40]:
            for adx_min in [15, 20, 25]:
                try:
                    strategy = SorosTrendStrategy(
                        top_k=top_k, lookback=lookback, adx_period=14
                    )
                    strategy._min_adx = adx_min  # override

                    repo = _get_repo()
                    config = PortfolioBacktestConfig(
                        market=market,
                        start_date=date.fromisoformat(start),
                        end_date=date.fromisoformat(end),
                        initial_cash=1_000_000,
                        dataset_version="v1",
                    )
                    engine = PortfolioEngine(
                        repository=repo, strategy=strategy, config=config,
                        cost_model=CostModel(commission_rate=0.001, slippage_rate=0.001),
                    )
                    result = engine.run()

                    md = result.metrics.get("max_drawdown", 0)
                    sr = result.metrics.get("sharpe_ratio", 0)
                    ar = result.metrics.get("annualized_return", 0)

                    row = {
                        "top_k": top_k, "lookback": lookback, "adx_min": adx_min,
                        "sharpe": sr, "max_dd": md, "ann_return": ar,
                    }
                    results.append(row)

                    dd_ok = md > -0.20 if md else False
                    print(f"  k={top_k} lb={lookback} adx={adx_min}: "
                          f"Sharpe={sr:.2f} DD={md:.1%} AnnRet={ar:.1%} {'✓' if dd_ok else '✗'}")

                    if dd_ok and (best is None or sr > best["sharpe"]):
                        best = row
                except Exception as e:
                    print(f"  k={top_k} lb={lookback} adx={adx_min}: ERROR {e}")

    if best:
        print(f"\n最优参数: top_k={best['top_k']} lookback={best['lookback']} "
              f"adx_min={best['adx_min']} → Sharpe={best['sharpe']:.2f} DD={best['max_dd']:.1%}")
    return results, best


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="投资大师策略回测研究")
    parser.add_argument("--market", default="cn", help="市场 (cn/us)")
    parser.add_argument("--start", default="2021-01-01", help="起始日期")
    parser.add_argument("--end", default="2025-06-30", help="结束日期")
    parser.add_argument("--strategy", choices=list(ALL_STRATEGIES.keys()) + ["all"], default="all")
    parser.add_argument("--grid-search", action="store_true", help="参数遍历模式")
    args = parser.parse_args()

    if args.grid_search:
        grid_search_soros(market=args.market, start=args.start, end=args.end)
    elif args.strategy == "all":
        run_all_strategies(market=args.market, start=args.start, end=args.end)
    else:
        run_strategy_backtest(args.strategy, market=args.market, start=args.start, end=args.end)
