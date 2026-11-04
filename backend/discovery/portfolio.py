"""组合级 Discovery 评估：选股 → 权重 → 调仓 → 成本 → 组合净值。

与单票信号回测的区别
- 评估对象是**组合净值曲线**，不是某只股票的择时信号；
- 每个调仓日做横截面选股（因子打分 → TopK），再按权重持有到下一个调仓日；
- 成本按组合换手扣除，收益按持仓权重加权；
- 与等权基准比较，输出超额收益 / 信息比率；
- 单票评估只作为“信号诊断”，不能作为晋升依据。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluator import PurgedSplit
from .stats import (
    cagr,
    deflated_sharpe_ratio,
    max_drawdown,
    performance_summary,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


# ------------------------------------------------------------------ Genome
@dataclass(frozen=True)
class PortfolioGenome:
    """组合策略基因：选股因子 + 持仓 + 调仓 + 风控。"""

    name: str
    factor: str = "momentum"          # momentum / reversal / low_vol / multi
    lookback: int = 60
    top_k: int = 20
    weighting: str = "equal"          # equal / score / inv_vol
    rebalance_days: int = 20
    min_adv: float = 0.0
    max_weight: float = 0.10
    cash_buffer: float = 0.0
    regime_gate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ Panel
@dataclass
class Panel:
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        self.close = self.close.sort_index().astype(float)
        self.volume = self.volume.reindex_like(self.close).fillna(0.0).astype(float)
        # 前向填充停牌价格；保留 NaN 标记“从未上市/退市”
        self.close = self.close.ffill()

    @property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change().fillna(0.0)

    def adv(self, window: int = 20) -> pd.DataFrame:
        return (self.close * self.volume).rolling(window, min_periods=max(5, window // 4)).mean()

    def slice(self, start: Any, end: Any) -> "Panel":
        return Panel(self.close.loc[start:end], self.volume.loc[start:end])


def load_panel(
    *,
    market: str = "cn",
    symbols: Optional[Sequence[str]] = None,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    limit: int = 500,
    min_coverage: float = 0.6,
    warmup_days: int = 400,
) -> Panel:
    """从本地 DuckDB 加载价格面板（自动带 warmup）。"""
    import duckdb

    from marketdata.db import DEFAULT_MARKET_DB_PATH

    db_path = DEFAULT_MARKET_DB_PATH
    start_d = date.fromisoformat(start) - timedelta(days=warmup_days)
    end_d = date.fromisoformat(end) if end else date.today()

    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        con = duckdb.connect(str(db_path))  # 同进程已有读写连接时共享实例
    try:
        if not symbols:
            symbols = [
                row[0]
                for row in con.execute(
                    """
                    WITH latest AS (SELECT MAX(trade_date) AS d FROM daily_bars WHERE market = ?),
                    recent AS (
                        SELECT symbol, AVG(close * volume) AS adv
                        FROM daily_bars, latest
                        WHERE market = ? AND trade_date >= latest.d - INTERVAL 90 DAY AND volume > 0
                        GROUP BY symbol
                    )
                    SELECT symbol FROM recent ORDER BY adv DESC LIMIT ?
                    """,
                    [market, market, int(limit)],
                ).fetchall()
            ]
        symbols = [str(s) for s in symbols]
        placeholders = ",".join(["?"] * len(symbols))
        df = con.execute(
            f"""
            SELECT symbol, trade_date,
                   COALESCE(adj_close, close) AS close,
                   COALESCE(volume, 0) AS volume
            FROM daily_bars
            WHERE market = ? AND symbol IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date, symbol
            """,
            [market, *symbols, start_d, end_d],
        ).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"{market} 没有可用行情（{start_d} ~ {end_d}）")
    close = df.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="trade_date", columns="symbol", values="volume").sort_index()
    coverage = close.notna().mean()
    keep = coverage[coverage >= min_coverage].index
    if len(keep) == 0:
        raise RuntimeError("行情覆盖率过低，无法构建面板")
    panel = Panel(close[keep], volume.reindex(columns=keep))
    return panel


def panel_quality(panel: Panel, *, threshold: float = 0.11) -> Dict[str, Any]:
    """数据质量检查：只标记极端负向断层或异常正向暴涨。

    A 股真实普涨日（如 2024-09-30 +11.2%）不应误报；复权断层的典型特征是
    全市场同步大幅下跌（本次为 2026-07-20 -12.9%），因此正向阈值放宽。
    """
    mean_ret = panel.returns.mean(axis=1)
    negative = mean_ret[mean_ret < -threshold]
    positive = mean_ret[mean_ret > threshold * 1.45]
    bad = pd.concat([negative, positive]).sort_values()
    return {
        "passed": bool(bad.empty),
        "negative_threshold": threshold,
        "positive_threshold": round(threshold * 1.45, 4),
        "bad_dates": [
            {"date": str(idx.date()), "mean_return": round(float(val), 6)}
            for idx, val in bad.items()
        ],
        "max_abs_mean_return": round(float(mean_ret.abs().max()), 6) if len(mean_ret) else 0.0,
    }


# ------------------------------------------------------------------ 因子
def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def compute_factor(panel: Panel, genome: PortfolioGenome) -> pd.DataFrame:
    rets = panel.returns
    if genome.factor == "momentum":
        return panel.close / panel.close.shift(genome.lookback) - 1.0
    if genome.factor == "reversal":
        return -(panel.close / panel.close.shift(genome.lookback) - 1.0)
    if genome.factor == "low_vol":
        return -rets.rolling(genome.lookback).std()
    if genome.factor == "multi":
        mom20 = panel.close / panel.close.shift(20) - 1.0
        mom60 = panel.close / panel.close.shift(60) - 1.0
        vol20 = rets.rolling(20).std()
        return _zscore(mom20) + _zscore(mom60) - _zscore(vol20)
    raise ValueError(f"unknown factor: {genome.factor}")


# ------------------------------------------------------------------ 权重
def _normalize_weights(
    scores: pd.Series,
    *,
    weighting: str,
    top_k: int,
    max_weight: float,
    gross_exposure: float,
    volatility: Optional[pd.Series] = None,
) -> pd.Series:
    selected = scores.dropna().sort_values(ascending=False).head(top_k)
    if selected.empty:
        return pd.Series(dtype=float)
    n = len(selected)

    if weighting == "score":
        raw = selected.clip(lower=0.0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=selected.index)
    elif weighting == "inv_vol" and volatility is not None:
        vol = volatility.reindex(selected.index).replace(0, np.nan)
        raw = 1.0 / vol
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(raw.mean() if raw.notna().any() else 1.0)
    else:
        raw = pd.Series(1.0, index=selected.index)

    weights = raw / raw.sum() * gross_exposure
    cap = min(max_weight, gross_exposure)
    for _ in range(6):
        over = weights > cap + 1e-12
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        under = ~over & (weights > 0)
        if not under.any():
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    return weights[weights > 1e-12]


def build_target_weights(
    panel: Panel,
    genome: PortfolioGenome,
    *,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """构建每日目标权重矩阵（仅使用当日及之前数据）。"""
    dates = panel.close.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) == 0:
        return pd.DataFrame(index=panel.close.index)

    factor = compute_factor(panel, genome)
    adv = panel.adv(20)
    vol20 = panel.returns.rolling(20).std()

    # 市场状态门控：等权指数在 200 日线上方才满仓，否则降至 30%
    exposure = pd.Series(1.0, index=panel.close.index)
    if genome.regime_gate:
        index_level = panel.close.mean(axis=1)
        ma200 = index_level.rolling(200, min_periods=60).mean()
        exposure = pd.Series(
            np.where(index_level >= ma200, 1.0, 0.3), index=panel.close.index
        ).fillna(0.3)

    rebalance_dates = dates[:: max(1, int(genome.rebalance_days))]
    rows = {}
    for dt in rebalance_dates:
        scores = factor.loc[dt]
        eligible = scores.notna() & panel.close.loc[dt].notna()
        if genome.min_adv > 0:
            eligible &= adv.loc[dt].fillna(0.0) >= genome.min_adv
        scores = scores[eligible]
        weight = _normalize_weights(
            scores,
            weighting=genome.weighting,
            top_k=int(genome.top_k),
            max_weight=float(genome.max_weight),
            gross_exposure=float(1.0 - genome.cash_buffer) * float(exposure.loc[dt]),
            volatility=vol20.loc[dt],
        )
        # 显式补 0：未入选的旧持仓必须清掉，否则 ffill 会把旧仓和新仓叠加
        rows[dt] = weight.reindex(panel.close.columns).fillna(0.0)

    if not rows:
        return pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    weights = pd.DataFrame(rows).T
    # 调仓日之间保持目标权重（ffill）；首个调仓日前为空仓
    weights = (
        weights.reindex(index=panel.close.index, columns=panel.close.columns)
        .ffill()
        .fillna(0.0)
    )
    weights = weights.clip(lower=0.0)
    return weights


# ------------------------------------------------------------------ 组合回测
def run_portfolio_backtest(
    panel: Panel,
    genome: PortfolioGenome,
    *,
    start: Any = None,
    end: Any = None,
    cost_bps: float = 10.0,
    lag: int = 1,
) -> Dict[str, Any]:
    """在给定面板上运行组合回测，返回净值、收益、换手与指标。"""
    weights = build_target_weights(panel, genome, start=start, end=end)
    if weights.empty:
        return {"error": "empty weights", "metrics": performance_summary([])}
    rets = panel.returns
    weights = weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

    executed = weights.shift(lag).fillna(0.0)
    gross = (executed * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    net = gross - turnover * (cost_bps / 10_000.0)

    if start is not None:
        net = net.loc[net.index >= pd.Timestamp(start)]
        turnover = turnover.loc[turnover.index >= pd.Timestamp(start)]
        weights = weights.loc[weights.index >= pd.Timestamp(start)]
    if end is not None:
        net = net.loc[net.index <= pd.Timestamp(end)]
        turnover = turnover.loc[turnover.index <= pd.Timestamp(end)]
        weights = weights.loc[weights.index <= pd.Timestamp(end)]

    equity = (1.0 + net).cumprod()
    bench_ret = rets.mean(axis=1).loc[net.index]
    bench_equity = (1.0 + bench_ret).cumprod()

    summary = performance_summary(net.to_numpy())
    bench_summary = performance_summary(bench_ret.to_numpy())
    years = max(len(net) / 252.0, 1e-9)
    active_weight = weights.sum(axis=1)
    return {
        "genome": genome.to_dict(),
        "returns": net,
        "equity": equity,
        "benchmark_returns": bench_ret,
        "benchmark_equity": bench_equity,
        "weights": weights,
        "turnover": turnover,
        "metrics": {
            **summary,
            "benchmark_cagr": bench_summary["cagr"],
            "benchmark_sharpe": bench_summary["sharpe"],
            "excess_cagr": round(summary["cagr"] - bench_summary["cagr"], 6),
            "excess_total_return": round(summary["total_return"] - bench_summary["total_return"], 6),
            "annual_turnover": round(float(turnover.sum()) / years, 4),
            "avg_positions": round(float((weights > 0).sum(axis=1).mean()), 2),
            "avg_exposure": round(float(active_weight.mean()), 4),
            "max_single_weight": round(float(weights.max().max()), 4),
        },
    }


# ------------------------------------------------------------------ 组合级 WF + 闸门
def evaluate_portfolio_grid(
    panel: Panel,
    genomes: Sequence[PortfolioGenome],
    splits: Sequence[PurgedSplit],
    *,
    cost_bps: float = 10.0,
    select_metric: str = "sharpe",
) -> Dict[str, Any]:
    """每个 fold 在 train 上选 Genome，在 test 上评估；输出组合级 OOS 指标。"""
    if not genomes or not splits:
        return {"error": "genomes or splits empty"}

    precomputed = []
    for genome in genomes:
        try:
            precomputed.append(run_portfolio_backtest(panel, genome, cost_bps=cost_bps))
        except Exception:
            precomputed.append(None)
    valid = [(g, r) for g, r in zip(genomes, precomputed) if r is not None and "error" not in r]
    if not valid:
        return {"error": "all genomes failed"}

    folds: List[Dict[str, Any]] = []
    best_oos_parts: List[pd.Series] = []
    per_genome_oos: List[List[pd.Series]] = [[] for _ in valid]
    dates = panel.close.index

    for split in splits:
        train_dates = dates[split.train]
        test_dates = dates[split.test]
        train_scores = []
        for _, result in valid:
            train_ret = result["returns"].reindex(train_dates).dropna()
            train_scores.append(performance_summary(train_ret.to_numpy()).get(select_metric, 0.0))
        best_i = int(np.argmax(train_scores))
        best_genome, best_result = valid[best_i]
        oos = best_result["returns"].reindex(test_dates).fillna(0.0)
        best_oos_parts.append(oos)
        for j, (_, result) in enumerate(valid):
            per_genome_oos[j].append(result["returns"].reindex(test_dates).fillna(0.0))

        folds.append({
            "fold": split.fold,
            "genome": best_genome.to_dict(),
            "train_score": round(float(train_scores[best_i]), 6),
            "test": performance_summary(oos.to_numpy()),
            "test_excess_cagr": round(
                performance_summary(oos.to_numpy())["cagr"]
                - performance_summary(
                    panel.returns.mean(axis=1).reindex(test_dates).fillna(0.0).to_numpy()
                )["cagr"],
                6,
            ),
        })

    agg = pd.concat(best_oos_parts) if best_oos_parts else pd.Series(dtype=float)
    summary = performance_summary(agg.to_numpy())
    bench_ret = panel.returns.mean(axis=1).reindex(agg.index).fillna(0.0)
    bench_summary = performance_summary(bench_ret.to_numpy())

    oos_series = [
        pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=float)
        for parts in per_genome_oos
    ]
    oos_matrix = pd.DataFrame({f"g{j}": s.to_numpy() for j, s in enumerate(oos_series)})
    oos_sharpes = [sharpe_ratio(oos_matrix[c].dropna().to_numpy()) for c in oos_matrix.columns]
    dsr = deflated_sharpe_ratio(agg.to_numpy(), trial_sharpes=oos_sharpes)
    pbo = probability_of_backtest_overfitting(oos_matrix.to_numpy()) if not oos_matrix.empty else None

    fold_sharpes = [f["test"]["sharpe"] for f in folds]
    years = max(len(agg) / 252.0, 1e-9)
    turnover = 0.0
    for _, res in valid:
        turnover += float(res["turnover"].reindex(agg.index).fillna(0.0).sum())
    turnover /= max(len(valid), 1)

    result = {
        "aggregate": summary,
        "benchmark": bench_summary,
        "excess_cagr": round(summary["cagr"] - bench_summary["cagr"], 6),
        "excess_total_return": round(summary["total_return"] - bench_summary["total_return"], 6),
        "folds": folds,
        "fold_sharpe_mean": round(float(np.mean(fold_sharpes)), 6) if fold_sharpes else 0.0,
        "fold_sharpe_std": round(float(np.std(fold_sharpes, ddof=1)), 6) if len(fold_sharpes) > 1 else 0.0,
        "fold_positive_ratio": round(float(np.mean([s > 0 for s in fold_sharpes])), 4) if fold_sharpes else 0.0,
        "n_trials": len(valid),
        "dsr": round(dsr, 6) if dsr is not None else None,
        "pbo": pbo,
        "annual_turnover": round(turnover / years, 4),
    }
    result["gate"] = portfolio_promotion_gate(result)
    return result


def portfolio_promotion_gate(
    result: Dict[str, Any],
    *,
    min_sharpe: float = 1.0,
    min_excess_cagr: float = 0.0,
    max_drawdown: float = 0.20,
    min_dsr: float = 0.95,
    max_pbo: float = 0.20,
    min_positive_folds: float = 0.60,
    max_annual_turnover: float = 30.0,
) -> Dict[str, Any]:
    """组合级晋升闸门：收益、超额、回撤、显著性、稳定性、换手。"""
    agg = result.get("aggregate", {})
    checks = {
        "sharpe": float(agg.get("sharpe", 0.0)) >= min_sharpe,
        "excess_cagr": float(result.get("excess_cagr", -1.0)) > min_excess_cagr,
        "max_drawdown": float(agg.get("max_drawdown", 1.0)) <= max_drawdown,
        "dsr": float(result.get("dsr") or 0.0) >= min_dsr,
        "pbo": float((result.get("pbo") or {}).get("pbo", 1.0)) <= max_pbo,
        "positive_folds": float(result.get("fold_positive_ratio", 0.0)) >= min_positive_folds,
        "turnover": float(result.get("annual_turnover", 1e9)) <= max_annual_turnover,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [k for k, v in checks.items() if not v],
    }
