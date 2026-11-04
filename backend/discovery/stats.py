"""Discovery Loop 统计闸门：绩效指标、Deflated Sharpe、PBO、FDR。

设计原则
- 所有函数接受"单期收益率"序列（未年化），内部统一使用 numpy。
- 空样本 / 零波动 / 样本过短一律返回 ``None`` 或 0，不抛异常，方便批量搜索。
- 随机过程（CSCV 组合抽样、bootstrap）全部支持 seed，保证实验可复现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from statistics import NormalDist

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni
_NORMAL = NormalDist()


def _norm_cdf(z: float) -> float:
    return float(_NORMAL.cdf(z))


def _norm_ppf(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(_NORMAL.inv_cdf(p))


def _skew(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 3.0
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= 0:
        return 3.0
    return float(np.mean(((x - mean) / std) ** 4))


# ------------------------------------------------------------------ 基础指标
def _clean(returns: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if returns is None:
        return np.array([], dtype=float)
    arr = np.asarray(list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float)
    return arr[np.isfinite(arr)]


def total_return(returns: Sequence[float]) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def cagr(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    years = r.size / periods_per_year
    return float(growth ** (1.0 / years) - 1.0) if years > 0 else 0.0


def volatility(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Sequence[float],
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """年化夏普比率。risk_free 为年化无风险利率。"""
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std <= 1e-12 or not math.isfinite(std):
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float],
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / dd * math.sqrt(periods_per_year))


def max_drawdown(returns: Sequence[float]) -> float:
    """最大回撤，返回正数幅度（0.2 = 回撤 20%），与现有回测引擎一致。"""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = np.divide(peak - equity, peak, out=np.zeros_like(equity), where=peak > 0)
    return float(np.max(dd))


def max_drawdown_duration(returns: Sequence[float]) -> int:
    r = _clean(returns)
    if r.size == 0:
        return 0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    in_dd = equity < peak
    best = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return int(best)


def calmar_ratio(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    dd = max_drawdown(returns)
    if dd <= 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / dd)


def win_rate(returns: Sequence[float]) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    return float(np.mean(r > 0))


def performance_summary(
    returns: Sequence[float],
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> Dict[str, float]:
    """单条收益序列的标准指标集合。"""
    r = _clean(returns)
    return {
        "n_periods": int(r.size),
        "total_return": round(total_return(r), 6),
        "cagr": round(cagr(r, periods_per_year), 6),
        "volatility": round(volatility(r, periods_per_year), 6),
        "sharpe": round(sharpe_ratio(r, risk_free, periods_per_year), 6),
        "sortino": round(sortino_ratio(r, risk_free, periods_per_year), 6),
        "max_drawdown": round(max_drawdown(r), 6),
        "max_drawdown_duration": max_drawdown_duration(r),
        "calmar": round(calmar_ratio(r, periods_per_year), 6),
        "win_rate": round(win_rate(r), 6),
        "skew": round(_skew(r), 6),
        "kurtosis": round(_kurtosis(r), 6),
    }


# ------------------------------------------------------------------ 概率夏普 / DSR
def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    benchmark_sr: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> Optional[float]:
    """PSR：观测 Sharpe 大于 benchmark 的概率（0~1）。benchmark_sr 用年化值。"""
    r = _clean(returns)
    n = r.size
    if n < 3:
        return None
    sr = sharpe_ratio(r, periods_per_year=periods_per_year) / math.sqrt(periods_per_year)
    if not math.isfinite(sr):
        return None
    sr_bench = benchmark_sr / math.sqrt(periods_per_year)
    skew = _skew(r)
    kurt = _kurtosis(r)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or not math.isfinite(denom):
        return None
    z = (sr - sr_bench) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    trial_sharpes: Optional[Sequence[float]] = None,
    n_trials: Optional[int] = None,
    periods_per_year: int = TRADING_DAYS,
) -> Optional[float]:
    """Deflated Sharpe Ratio（Bailey & López de Prado）。

    参数
    - returns: 最优候选的收益序列。
    - trial_sharpes: 同一轮搜索中所有候选的**年化** Sharpe，用于估计 SR 的横截面方差。
    - n_trials: 若没有 trial_sharpes，至少给出试验次数 K。

    返回 0~1；K<=1 时无多重检验惩罚，返回 PSR（等价于 1.0 上限）。
    """
    r = _clean(returns)
    n = r.size
    if n < 3:
        return None

    sr = sharpe_ratio(r, periods_per_year=periods_per_year) / math.sqrt(periods_per_year)
    if not math.isfinite(sr):
        return None
    skew = _skew(r)
    kurt = _kurtosis(r)

    trial = _clean(trial_sharpes) if trial_sharpes is not None else np.array([])
    if trial.size >= 2:
        trial = trial / math.sqrt(periods_per_year)  # 年化转单期
        k = int(trial.size)
        sr_std = float(np.std(trial, ddof=1))
    else:
        k = int(n_trials or 1)
        denom0 = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
        sr_std = math.sqrt(max(denom0, 0.0) / (n - 1)) if denom0 > 0 else 0.0

    if k <= 1 or sr_std <= 0:
        # 单次试验：退化为 PSR vs 0
        return probabilistic_sharpe_ratio(r, 0.0, periods_per_year)

    sr0 = sr_std * (
        (1.0 - EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / k)
        + EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (k * math.e))
    )
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or not math.isfinite(denom):
        return None
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


# ------------------------------------------------------------------ PBO (CSCV)
def _sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    """按列计算 Sharpe（单期，不年化）。matrix: T x N。"""
    if matrix.size == 0:
        return np.array([])
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    out = np.divide(mean, std, out=np.zeros_like(mean), where=std > 0)
    return out


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    n_splits: int = 10,
    max_combinations: int = 300,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """CSCV 估计 PBO。

    returns_matrix: T x N，T 个观测、N 个候选（或参数组合）。
    返回 {"pbo": 0~1, "n_combinations": int, "n_candidates": int}。
    """
    mat = np.asarray(returns_matrix, dtype=float)
    if mat.ndim != 2:
        return None
    mat = mat[np.isfinite(mat).all(axis=1)]
    t, n = mat.shape
    if t < 40 or n < 2:
        return None

    s = int(min(n_splits, max(4, t // 10)))
    if s % 2:
        s -= 1
    if s < 4:
        return None

    blocks = np.array_split(np.arange(t), s)
    all_combos = list(combinations(range(s), s // 2))
    if len(all_combos) > max_combinations:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combos = [all_combos[i] for i in sorted(chosen)]
    else:
        combos = all_combos

    overfit = 0
    valid = 0
    ranks: List[float] = []
    for combo in combos:
        test_groups = set(combo)
        is_idx = np.concatenate([blocks[i] for i in range(s) if i not in test_groups])
        oos_idx = np.concatenate([blocks[i] for i in range(s) if i in test_groups])
        if is_idx.size < 2 or oos_idx.size < 2:
            continue
        is_sr = _sharpe_columns(mat[is_idx])
        oos_sr = _sharpe_columns(mat[oos_idx])
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue
        best = int(np.nanargmax(is_sr))
        # 相对排名：1 = 最差，N = 最好
        order = np.argsort(np.argsort(oos_sr)) + 1
        omega = float(order[best]) / (n + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        rank_logit = math.log(omega / (1 - omega))
        ranks.append(omega)
        valid += 1
        if rank_logit <= 0:  # 样本内最优在样本外低于中位数
            overfit += 1

    if valid == 0:
        return None
    return {
        "pbo": round(overfit / valid, 6),
        "n_combinations": valid,
        "n_candidates": n,
        "n_splits": s,
        "median_oos_percentile": round(float(np.median(ranks)), 4),
    }


# ------------------------------------------------------------------ 多重检验
@dataclass
class FdrResult:
    alpha: float
    n_tests: int
    n_rejected: int
    threshold: float
    rejected: List[bool]
    adjusted_pvalues: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.1,
) -> FdrResult:
    """Benjamini-Hochberg FDR 控制。"""
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return FdrResult(alpha, 0, 0, 0.0, [], [])
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    cutoff = float(thresholds[k - 1]) if k else 0.0

    rejected_sorted = np.zeros(n, dtype=bool)
    if k:
        rejected_sorted[:k] = True
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = rejected_sorted

    # adjusted p-values（BH step-up）
    adjusted_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.zeros(n)
    adjusted[order] = adjusted_sorted

    return FdrResult(
        alpha=alpha,
        n_tests=n,
        n_rejected=int(rejected.sum()),
        threshold=cutoff,
        rejected=rejected.tolist(),
        adjusted_pvalues=[round(float(x), 6) for x in adjusted],
    )


# ------------------------------------------------------------------ Bootstrap
def block_bootstrap_ci(
    returns: Sequence[float],
    *,
    statistic: str = "sharpe",
    n_boot: int = 1000,
    block: int = 20,
    alpha: float = 0.05,
    seed: int = 42,
    periods_per_year: int = TRADING_DAYS,
) -> Optional[Dict[str, float]]:
    """块 bootstrap 置信区间（保留序列自相关）。"""
    r = _clean(returns)
    if r.size < block * 2 or n_boot < 10:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(r.size / block))
    funcs = {
        "sharpe": lambda x: sharpe_ratio(x, periods_per_year=periods_per_year),
        "cagr": lambda x: cagr(x, periods_per_year),
        "max_drawdown": max_drawdown,
        "total_return": total_return,
    }
    fn = funcs.get(statistic)
    if fn is None:
        raise ValueError(f"unknown statistic: {statistic}")

    stats_list = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, r.size - block + 1), size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[: r.size]
        stats_list.append(fn(sample))
    arr = np.asarray(stats_list, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "mean": round(float(arr.mean()), 6),
        "lower": round(float(np.quantile(arr, alpha / 2)), 6),
        "upper": round(float(np.quantile(arr, 1 - alpha / 2)), 6),
        "n_boot": int(arr.size),
    }
