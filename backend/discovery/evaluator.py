"""Discovery Loop 评估器：Purged Walk-Forward / CPCV / 因果性与稳健性检查。

核心约束
- 所有切分都显式区分 train / test，并在两者之间留 ``purge`` 间隔；
- 信号必须因果：``causality_test`` 会用截断数据校验；
- 交易成本按换手扣除，默认 T+1 执行（lag=1）；
- 选择只在 train 内做，test 只用于评估，禁止回看。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    performance_summary,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

SignalFactory = Callable[..., Sequence[float]]


# ------------------------------------------------------------------ 切分
@dataclass(frozen=True)
class PurgedSplit:
    fold: int
    train: np.ndarray
    test: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "train_start": int(self.train[0]) if self.train.size else None,
            "train_end": int(self.train[-1]) if self.train.size else None,
            "test_start": int(self.test[0]) if self.test.size else None,
            "test_end": int(self.test[-1]) if self.test.size else None,
            "train_size": int(self.train.size),
            "test_size": int(self.test.size),
        }


def make_walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    purge: int = 0,
    embargo: int = 0,
) -> List[PurgedSplit]:
    """滚动 purged walk-forward 切分（基于样本位置）。"""
    if n_samples <= 0 or train_size <= 0 or test_size <= 0:
        return []
    if purge < 0 or embargo < 0:
        raise ValueError("purge/embargo must be >= 0")
    step = int(step_size or test_size)
    if step <= 0:
        raise ValueError("step_size must be positive")

    splits: List[PurgedSplit] = []
    start = 0
    fold = 0
    while start + train_size + purge + test_size <= n_samples:
        train = np.arange(start, start + train_size)
        test_start = start + train_size + purge
        test = np.arange(test_start, test_start + test_size)
        splits.append(PurgedSplit(fold=fold, train=train, test=test))
        fold += 1
        # 至少 step_size 的滚动步长，但测试窗口不重叠，避免 OOS 收益被重复计数
        start = max(start + step, int(test[-1]) + 1 + embargo)
    return splits


def make_cpcv_splits(
    n_samples: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge: int = 0,
    embargo: int = 0,
) -> List[PurgedSplit]:
    """Combinatorial Purged Cross-Validation（López de Prado）。"""
    if n_samples < n_groups * 5 or n_groups < 2:
        return []
    if not (1 <= n_test_groups < n_groups):
        raise ValueError("n_test_groups must be in [1, n_groups)")
    boundaries = np.linspace(0, n_samples, n_groups + 1, dtype=int)
    groups = [np.arange(boundaries[i], boundaries[i + 1]) for i in range(n_groups)]

    splits: List[PurgedSplit] = []
    for fold, combo in enumerate(itertools.combinations(range(n_groups), n_test_groups)):
        test = np.concatenate([groups[i] for i in combo])
        mask = np.ones(n_samples, dtype=bool)
        mask[test] = False
        for i in combo:
            start = int(boundaries[i])
            end = int(boundaries[i + 1]) - 1
            if purge:
                mask[max(0, start - purge):start] = False
            if embargo:
                mask[end + 1:min(n_samples, end + 1 + embargo)] = False
        train = np.where(mask)[0]
        if train.size < 5 or test.size < 2:
            continue
        splits.append(PurgedSplit(fold=fold, train=train, test=np.sort(test)))
    return splits


def expand_grid(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """把 {name: [v1, v2]} 展开为参数组合列表。"""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ------------------------------------------------------------------ 收益与仓位
def positions_to_returns(
    prices: pd.Series,
    positions: Sequence[float],
    *,
    cost_bps: float = 0.0,
    lag: int = 1,
    long_only: bool = True,
) -> pd.Series:
    """仓位 → 策略净收益。lag=1 表示 T 日收盘信号、T+1 收益生效。"""
    price = pd.Series(prices).astype(float).sort_index()
    pos = pd.Series(np.asarray(positions, dtype=float), index=price.index)
    if long_only:
        pos = pos.clip(lower=0.0, upper=1.0)
    else:
        pos = pos.clip(-1.0, 1.0)
    pos_exec = pos.shift(lag).fillna(0.0)
    ret = price.pct_change().fillna(0.0)
    turnover = pos_exec.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(float(pos_exec.iloc[0]))
    return (pos_exec * ret - turnover * (cost_bps / 10_000.0)).rename("strategy_return")


def _slice_returns(returns: pd.Series, idx: np.ndarray) -> pd.Series:
    return returns.iloc[idx]


# ------------------------------------------------------------------ 单策略评估
def evaluate_positions(
    prices: pd.Series,
    positions: Sequence[float],
    splits: Sequence[PurgedSplit],
    *,
    cost_bps: float = 0.0,
    lag: int = 1,
    long_only: bool = True,
) -> Dict[str, Any]:
    """在给定切分上评估一条固定仓位序列。"""
    rets = positions_to_returns(prices, positions, cost_bps=cost_bps, lag=lag, long_only=long_only)
    folds = []
    oos_parts = []
    for split in splits:
        oos = _slice_returns(rets, split.test)
        oos_parts.append(oos)
        m = performance_summary(oos)
        m["fold"] = split.fold
        folds.append(m)
    agg = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    summary = performance_summary(agg)
    fold_sharpes = [f["sharpe"] for f in folds]
    return {
        "folds": folds,
        "aggregate": summary,
        "fold_sharpe_mean": round(float(np.mean(fold_sharpes)), 6) if fold_sharpes else 0.0,
        "fold_sharpe_std": round(float(np.std(fold_sharpes, ddof=1)), 6) if len(fold_sharpes) > 1 else 0.0,
        "fold_positive_ratio": round(float(np.mean([s > 0 for s in fold_sharpes])), 4) if fold_sharpes else 0.0,
    }


# ------------------------------------------------------------------ 参数搜索 + OOS
def evaluate_param_grid(
    prices: pd.Series,
    signal_factory: SignalFactory,
    param_grid: Sequence[Dict[str, Any]],
    splits: Sequence[PurgedSplit],
    *,
    cost_bps: float = 0.0,
    lag: int = 1,
    long_only: bool = True,
    select_metric: str = "sharpe",
    include_series: bool = False,
) -> Dict[str, Any]:
    """每个 fold 在 train 上选参，在 test 上评估；再做 DSR / PBO / FDR。

    注意：``signal_factory(prices, **params)`` 必须是因果函数；
    评估器会用全历史价格生成仓位，再按切分窗口截取收益。
    """
    price = pd.Series(prices).astype(float).sort_index()
    params_list = [dict(p) for p in param_grid] or [{}]
    if not splits:
        return {"error": "no valid splits", "n_trials": len(params_list)}

    # 预计算每个参数组合的收益序列
    returns_by_param: List[pd.Series] = []
    valid_params: List[Dict[str, Any]] = []
    for params in params_list:
        try:
            pos = signal_factory(price, **params)
            rets = positions_to_returns(price, pos, cost_bps=cost_bps, lag=lag, long_only=long_only)
            returns_by_param.append(rets)
            valid_params.append(params)
        except Exception:
            continue
    if not returns_by_param:
        return {"error": "all parameter sets failed", "n_trials": len(params_list)}

    # 每个 fold：train 选优 → test 记录
    fold_records: List[Dict[str, Any]] = []
    best_oos_parts: List[pd.Series] = []
    per_param_oos: List[List[pd.Series]] = [[] for _ in valid_params]
    for split in splits:
        train_scores = []
        for params, rets in zip(valid_params, returns_by_param):
            train_m = performance_summary(_slice_returns(rets, split.train))
            train_scores.append(train_m.get(select_metric, 0.0))
        best_i = int(np.argmax(train_scores)) if train_scores else 0
        best_params = valid_params[best_i]
        oos_rets = _slice_returns(returns_by_param[best_i], split.test)
        best_oos_parts.append(oos_rets)

        for j, rets in enumerate(returns_by_param):
            per_param_oos[j].append(_slice_returns(rets, split.test))

        fold_records.append({
            "fold": split.fold,
            "best_params": best_params,
            "train_score": round(float(train_scores[best_i]), 6) if train_scores else 0.0,
            "test": performance_summary(oos_rets),
        })

    agg = pd.concat(best_oos_parts) if best_oos_parts else pd.Series(dtype=float)
    summary = performance_summary(agg)
    fold_sharpes = [f["test"]["sharpe"] for f in fold_records]

    # 多重检验：每个参数组合自己的 OOS 收益序列（按 fold 拼接，长度一致）
    oos_series_by_param = [
        pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=float)
        for parts in per_param_oos
    ]
    oos_matrix = pd.DataFrame({f"p{j}": s.to_numpy() for j, s in enumerate(oos_series_by_param)})
    oos_sharpes = []
    for col in oos_matrix.columns:
        oos_sharpes.append(sharpe_ratio(oos_matrix[col].dropna().to_numpy()))
    dsr = deflated_sharpe_ratio(agg.to_numpy(), trial_sharpes=oos_sharpes)
    pbo = probability_of_backtest_overfitting(oos_matrix.to_numpy()) if not oos_matrix.empty else None

    result: Dict[str, Any] = {
        "aggregate": summary,
        "folds": fold_records,
        "fold_sharpe_mean": round(float(np.mean(fold_sharpes)), 6) if fold_sharpes else 0.0,
        "fold_sharpe_std": round(float(np.std(fold_sharpes, ddof=1)), 6) if len(fold_sharpes) > 1 else 0.0,
        "fold_positive_ratio": round(float(np.mean([s > 0 for s in fold_sharpes])), 4) if fold_sharpes else 0.0,
        "n_trials": len(valid_params),
        "dsr": round(dsr, 6) if dsr is not None else None,
        "pbo": pbo,
        "oos_sharpes": [round(float(x), 6) for x in oos_sharpes],
    }
    if include_series:
        result["oos_returns"] = {str(k): round(float(v), 8) for k, v in agg.items()}
    return result


# ------------------------------------------------------------------ 稳健性 / 因果性
def causality_test(
    prices: pd.Series,
    signal_factory: SignalFactory,
    params: Dict[str, Any],
    *,
    n_probes: int = 5,
) -> Dict[str, Any]:
    """截断测试：用 t 之前的数据重算信号，历史仓位不应改变。

    若信号在 t 之前的取值发生变化，说明函数使用了未来信息。
    """
    price = pd.Series(prices).astype(float).sort_index()
    if len(price) < 30 or n_probes <= 0:
        return {"passed": None, "reason": "sample too short"}
    full = np.asarray(signal_factory(price, **params), dtype=float)
    probes = np.linspace(20, len(price) - 2, num=min(n_probes, max(1, len(price) - 22)), dtype=int)
    violations = 0
    checked = 0
    for p in probes:
        trunc = price.iloc[: p + 1]
        try:
            part = np.asarray(signal_factory(trunc, **params), dtype=float)
        except Exception:
            continue
        # 比较 t=p 时刻（含）之前的信号：因果函数在截断前后应完全一致
        n = min(len(part), p + 1)
        if n < 5:
            continue
        checked += 1
        if not np.allclose(full[:n], part[:n], equal_nan=True, rtol=1e-8, atol=1e-8):
            violations += 1
    if checked == 0:
        return {"passed": None, "reason": "no valid probes"}
    return {
        "passed": violations == 0,
        "violations": violations,
        "checked": checked,
        "probes": [int(x) for x in probes],
    }


def delay_stress_test(
    prices: pd.Series,
    signal_factory: SignalFactory,
    params: Dict[str, Any],
    splits: Sequence[PurgedSplit],
    *,
    cost_bps: float = 0.0,
    long_only: bool = True,
) -> Dict[str, Any]:
    """执行延迟压力：lag=1（正常） vs lag=2（多延迟一天）。"""
    price = pd.Series(prices).astype(float).sort_index()
    pos = signal_factory(price, **params)
    base = evaluate_positions(price, pos, splits, cost_bps=cost_bps, lag=1, long_only=long_only)
    delayed = evaluate_positions(price, pos, splits, cost_bps=cost_bps, lag=2, long_only=long_only)
    base_sr = base["aggregate"]["sharpe"]
    delay_sr = delayed["aggregate"]["sharpe"]
    if base_sr is None or abs(base_sr) < 0.2:
        retention = None
        passed = bool(delay_sr >= 0)
    else:
        retention = round(float(delay_sr / base_sr), 4)
        passed = bool(delay_sr >= 0 and retention >= 0.5)
    return {
        "base_sharpe": base_sr,
        "delayed_sharpe": delay_sr,
        "retention": retention,
        "passed": passed,
    }


def cost_stress_test(
    prices: pd.Series,
    signal_factory: SignalFactory,
    params: Dict[str, Any],
    splits: Sequence[PurgedSplit],
    *,
    cost_bps: float = 10.0,
    multipliers: Sequence[float] = (1.0, 2.0, 3.0),
    long_only: bool = True,
) -> Dict[str, Any]:
    """交易成本压力测试。"""
    price = pd.Series(prices).astype(float).sort_index()
    pos = signal_factory(price, **params)
    results = []
    for m in multipliers:
        r = evaluate_positions(price, pos, splits, cost_bps=cost_bps * m, lag=1, long_only=long_only)
        results.append({
            "cost_multiplier": m,
            "cost_bps": round(cost_bps * m, 2),
            "sharpe": r["aggregate"]["sharpe"],
            "cagr": r["aggregate"]["cagr"],
        })
    return {"results": results, "passed": all(x["sharpe"] >= 0 for x in results)}


def shuffled_returns_significance(
    prices: pd.Series,
    signal_factory: SignalFactory,
    params: Dict[str, Any],
    splits: Sequence[PurgedSplit],
    *,
    n_permutations: int = 50,
    cost_bps: float = 0.0,
    seed: int = 42,
    long_only: bool = True,
) -> Dict[str, Any]:
    """打乱收益序列后重跑信号：真实 Sharpe 应显著高于随机分布。"""
    price = pd.Series(prices).astype(float).sort_index()
    rets = price.pct_change().fillna(0.0).to_numpy()
    if rets.size < 50:
        return {"passed": None, "reason": "sample too short"}
    real_pos = signal_factory(price, **params)
    real = evaluate_positions(price, real_pos, splits, cost_bps=cost_bps, lag=1, long_only=long_only)
    real_sr = real["aggregate"]["sharpe"]

    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(rets)
        sim_price = pd.Series(
            float(price.iloc[0]) * np.cumprod(1.0 + shuffled),
            index=price.index,
        )
        try:
            pos = signal_factory(sim_price, **params)
            r = evaluate_positions(sim_price, pos, splits, cost_bps=cost_bps, lag=1, long_only=long_only)
            sims.append(r["aggregate"]["sharpe"])
        except Exception:
            continue
    sims_arr = np.asarray([s for s in sims if math.isfinite(s)], dtype=float)
    if sims_arr.size == 0:
        return {"passed": None, "reason": "no valid permutations"}
    p_value = (1.0 + float(np.sum(sims_arr >= real_sr))) / (1.0 + sims_arr.size)
    return {
        "real_sharpe": real_sr,
        "shuffle_mean": round(float(sims_arr.mean()), 6),
        "shuffle_p95": round(float(np.quantile(sims_arr, 0.95)), 6),
        "p_value": round(p_value, 6),
        "passed": bool(p_value <= 0.1 and real_sr > 0),
        "n_permutations": int(sims_arr.size),
    }


def capacity_report(
    positions: Sequence[float],
    prices: pd.Series,
    volume: pd.Series,
    *,
    capital: float = 1_000_000.0,
    max_participation: float = 0.05,
) -> Dict[str, Any]:
    """单标的容量估算：调仓金额 / ADV。"""
    price = pd.Series(prices).astype(float).sort_index()
    vol = pd.Series(volume).astype(float).reindex(price.index).fillna(0.0)
    pos = pd.Series(np.asarray(positions, dtype=float), index=price.index).clip(0, 1)
    turnover = pos.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(float(pos.iloc[0]))
    trade_notional = turnover * capital
    adv = (vol * price).replace(0, np.nan)
    participation = (trade_notional / adv).replace([np.inf, -np.inf], np.nan).dropna()
    if participation.empty:
        return {"passed": None, "reason": "no ADV data"}
    p95 = float(np.quantile(participation, 0.95))
    return {
        "p50_participation": round(float(participation.median()), 6),
        "p95_participation": round(p95, 6),
        "max_participation": round(float(participation.max()), 6),
        "max_allowed": max_participation,
        "passed": bool(p95 <= max_participation),
    }


# ------------------------------------------------------------------ 一键闸门
def promotion_gate(
    result: Dict[str, Any],
    *,
    min_sharpe: float = 0.8,
    min_dsr: float = 0.95,
    max_pbo: float = 0.2,
    min_positive_folds: float = 0.6,
    min_fold_sharpe: float = 0.0,
) -> Dict[str, Any]:
    """根据评估结果给出晋升/淘汰判断（不含模拟盘与人工审批）。"""
    agg = result.get("aggregate", {})
    checks = {
        "sharpe": agg.get("sharpe", 0.0) >= min_sharpe,
        "dsr": (result.get("dsr") or 0.0) >= min_dsr,
        "pbo": (result.get("pbo") or {}).get("pbo", 1.0) <= max_pbo,
        "positive_folds": result.get("fold_positive_ratio", 0.0) >= min_positive_folds,
        "fold_sharpe": result.get("fold_sharpe_mean", 0.0) >= min_fold_sharpe,
        "max_drawdown": agg.get("max_drawdown", 1.0) <= 0.35,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [k for k, v in checks.items() if not v],
    }
