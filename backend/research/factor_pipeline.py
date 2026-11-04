"""Factor pipeline primitives for cross-sectional research."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

import pandas as pd

from marketdata.data_portal import DataPortal

MissingValuePolicy = Literal["drop", "zero", "median"]


@dataclass(frozen=True)
class FactorDefinition:
    """Definition of one factor column in the research pipeline."""

    name: str
    version: str
    dependencies: tuple[str, ...]
    window: int
    frequency: str
    point_in_time_rule: str
    compute_fn: Callable[[date, tuple[str, ...], DataPortal], pd.Series]


@dataclass(frozen=True)
class FactorPipelineConfig:
    """Cross-sectional transform options."""

    winsorize: bool = False
    standardize: bool = False
    missing_value_policy: MissingValuePolicy = "drop"


@dataclass(frozen=True)
class FactorComputeResult:
    """Output cross-section and diagnostics for one date."""

    frame: pd.DataFrame
    diagnostics: dict[str, int]


@dataclass
class FactorPipeline:
    """Compute point-in-time cross-sectional factors."""

    definitions: tuple[FactorDefinition, ...]
    config: FactorPipelineConfig = field(default_factory=FactorPipelineConfig)

    def compute(
        self,
        as_of: date,
        universe: tuple[str, ...],
        data: DataPortal,
    ) -> FactorComputeResult:
        out = pd.DataFrame({"symbol": list(universe)})
        for definition in self.definitions:
            values = definition.compute_fn(as_of, universe, data)
            values = values.reindex(universe)
            out[definition.name] = values.values
        out = self._apply_missing_policy(out)
        factor_cols = [d.name for d in self.definitions]
        if self.config.winsorize and factor_cols:
            out[factor_cols] = out[factor_cols].clip(
                lower=out[factor_cols].quantile(0.01),
                upper=out[factor_cols].quantile(0.99),
                axis=1,
            )
        if self.config.standardize and factor_cols:
            std = out[factor_cols].std(ddof=0).replace(0, 1.0)
            out[factor_cols] = (out[factor_cols] - out[factor_cols].mean()) / std
        diagnostics = {
            "universe_size": len(universe),
            "tradable_size": len(out),
            "factor_valid_count": int(out[factor_cols].notna().all(axis=1).sum()) if factor_cols else 0,
        }
        return FactorComputeResult(frame=out, diagnostics=diagnostics)

    def _apply_missing_policy(self, frame: pd.DataFrame) -> pd.DataFrame:
        factor_cols = [d.name for d in self.definitions]
        if not factor_cols:
            return frame
        if self.config.missing_value_policy == "drop":
            return frame.dropna(subset=factor_cols).reset_index(drop=True)
        if self.config.missing_value_policy == "zero":
            out = frame.copy()
            out[factor_cols] = out[factor_cols].fillna(0.0)
            return out
        out = frame.copy()
        out[factor_cols] = out[factor_cols].fillna(out[factor_cols].median())
        return out


def neutralize_by_industry(
    frame: pd.DataFrame,
    factor_col: str,
    industry_col: str,
) -> pd.Series:
    """对因子值做行业中性化（回归残差法）。

    对每个截面，用 OLS: factor ~ industry_dummies，取残差作为中性化后的因子值。
    """
    if frame.empty or factor_col not in frame.columns or industry_col not in frame.columns:
        return pd.Series(dtype=float)
    sub = frame[[factor_col, industry_col]].dropna()
    if sub.empty or sub[industry_col].nunique() < 2:
        return sub[factor_col]
    dummies = pd.get_dummies(sub[industry_col], prefix="ind", drop_first=True, dtype=float)
    X = dummies.values
    y = sub[factor_col].values
    # OLS residual: y - X @ (X'X)^-1 X'y
    XtX = X.T @ X
    try:
        import numpy as np
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X.T @ y)
        residual = y - X @ beta
    except Exception:
        residual = y - y.mean()
    return pd.Series(residual, index=sub.index, name=f"{factor_col}_neutralized")


def cross_sectional_pipeline(
    frame: pd.DataFrame,
    factor_cols: list[str],
    *,
    winsorize: bool = True,
    standardize: bool = True,
    missing_policy: MissingValuePolicy = "median",
    industry_col: str | None = None,
) -> pd.DataFrame:
    """一站式截面管线: 缺失值 → 去极值 → (行业中性化) → 标准化。"""
    out = frame.copy()
    # 1. 缺失值处理
    if missing_policy == "drop":
        out = out.dropna(subset=factor_cols).reset_index(drop=True)
    elif missing_policy == "zero":
        out[factor_cols] = out[factor_cols].fillna(0.0)
    elif missing_policy == "median":
        out[factor_cols] = out[factor_cols].fillna(out[factor_cols].median())
    # 2. 去极值 (winsorize)
    if winsorize and factor_cols:
        for col in factor_cols:
            lo = out[col].quantile(0.01)
            hi = out[col].quantile(0.99)
            out[col] = out[col].clip(lower=lo, upper=hi)
    # 3. 行业中性化
    if industry_col and industry_col in out.columns:
        for col in factor_cols:
            neutralized = neutralize_by_industry(out, col, industry_col)
            if not neutralized.empty:
                out.loc[neutralized.index, col] = neutralized.values
    # 4. 标准化
    if standardize and factor_cols:
        std = out[factor_cols].std(ddof=0).replace(0, 1.0)
        out[factor_cols] = (out[factor_cols] - out[factor_cols].mean()) / std
    return out
