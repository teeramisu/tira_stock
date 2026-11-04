"""Factor evaluation interfaces and basic fixture implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """Pearson IC."""
    aligned = pd.concat([factor.rename("f"), forward_return.rename("r")], axis=1).dropna()
    if aligned.empty:
        return 0.0
    return float(aligned["f"].corr(aligned["r"]))


def compute_rank_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """Spearman rank IC."""
    aligned = pd.concat([factor.rename("f"), forward_return.rename("r")], axis=1).dropna()
    if aligned.empty:
        return 0.0
    return float(aligned["f"].rank().corr(aligned["r"].rank()))


def compute_ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """IC 均值、标准差、IR、正IC比例。"""
    if ic_series.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_positive_ratio": 0.0}
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_positive_ratio = float((ic_series > 0).mean())
    return {
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "ic_ir": round(ic_ir, 2),
        "ic_positive_ratio": round(ic_positive_ratio, 4),
    }


def compute_ic_decay(
    ic_series: pd.Series,
    windows: list[int] | None = None,
) -> dict[int, float]:
    """滚动IC均值随窗口衰减，用于评估因子时效性。"""
    if windows is None:
        windows = [5, 10, 20, 60]
    if ic_series.empty:
        return {w: 0.0 for w in windows}
    return {
        w: round(float(ic_series.rolling(w, min_periods=1).mean().mean()), 4)
        for w in windows
    }


def compute_long_short_return(
    frame: pd.DataFrame,
    factor_col: str,
    return_col: str,
    quantiles: int = 5,
) -> dict[str, float]:
    """多空收益: Top组收益 - Bottom组收益。"""
    out = frame[[factor_col, return_col]].dropna().copy()
    if out.empty or out[factor_col].nunique() < quantiles:
        return {"top_return": 0.0, "bottom_return": 0.0, "long_short": 0.0}
    out["q"] = pd.qcut(out[factor_col].rank(method="first"), quantiles, labels=False, duplicates="drop")
    top_ret = float(out[out["q"] == out["q"].max()][return_col].mean())
    bottom_ret = float(out[out["q"] == out["q"].min()][return_col].mean())
    return {
        "top_return": round(top_ret, 6),
        "bottom_return": round(bottom_ret, 6),
        "long_short": round(top_ret - bottom_ret, 6),
    }


def compute_quantile_returns(frame: pd.DataFrame, factor_col: str, return_col: str, quantiles: int = 5) -> pd.DataFrame:
    """Layered return interface placeholder."""
    out = frame[[factor_col, return_col]].dropna().copy()
    if out.empty:
        return pd.DataFrame(columns=["quantile", "mean_return"])
    out["quantile"] = pd.qcut(out[factor_col], q=quantiles, labels=False, duplicates="drop")
    return out.groupby("quantile", as_index=False)[return_col].mean().rename(columns={return_col: "mean_return"})


def factor_coverage(frame: pd.DataFrame, factor_col: str) -> float:
    if frame.empty:
        return 0.0
    return float(frame[factor_col].notna().mean())


def factor_autocorr(current: pd.Series, previous: pd.Series) -> float:
    aligned = pd.concat([current.rename("c"), previous.rename("p")], axis=1).dropna()
    if aligned.empty:
        return 0.0
    return float(aligned["c"].corr(aligned["p"]))


def build_factor_report(
    ic_series: pd.Series,
    rank_ic_series: pd.Series,
    frame: pd.DataFrame,
    factor_col: str,
    return_col: str,
) -> dict[str, Any]:
    """一站式因子评估报告: IC摘要 + RankIC摘要 + 衰减 + 多空收益 + 覆盖率。"""
    return {
        "ic": compute_ic_summary(ic_series),
        "rank_ic": compute_ic_summary(rank_ic_series),
        "ic_decay": compute_ic_decay(ic_series),
        "long_short": compute_long_short_return(frame, factor_col, return_col),
        "coverage": factor_coverage(frame, factor_col),
    }
