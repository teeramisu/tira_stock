"""akshare 宏观适配器：把 NBS/PBOC 指标转换为统一长表。"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, Iterable, List, Optional, Sequence

import pandas as pd

from ..models import MacroObservation
from ..parsers import parse_period, to_float

FETCHED_AT = lambda: datetime.now().isoformat(timespec="seconds")  # noqa: E731


def _obs(series_id: str, period: Optional[date], value: Optional[float], *, unit: str = "",
         source: str, provider: str = "eastmoney", url: str = "", label: str = "") -> Optional[MacroObservation]:
    if period is None or value is None:
        return None
    return MacroObservation(
        series_id=series_id, period=period, value=float(value), unit=unit,
        source=source, provider=provider, source_url=url, raw_label=label,
        fetched_at=FETCHED_AT(),
    )


def _iter_rows(df: pd.DataFrame, period_col: Any, mappings: Sequence[tuple[str, str, str]],
               *, source: str, url: str = "") -> List[MacroObservation]:
    out: List[MacroObservation] = []
    if df is None or df.empty or period_col not in df.columns:
        return out
    for _, row in df.iterrows():
        period = parse_period(row.get(period_col))
        if period is None:
            continue
        for series_id, value_col, unit in mappings:
            if value_col not in df.columns:
                continue
            value = to_float(row.get(value_col))
            obs = _obs(series_id, period, value, unit=unit, source=source, url=url,
                       label=str(row.get(period_col)))
            if obs:
                out.append(obs)
    return out


def _safe(name: str, fn: Callable, errors: List[str]) -> Optional[pd.DataFrame]:
    try:
        df = fn()
        if df is None or df.empty:
            errors.append(f"{name}: empty")
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{name}: {exc}")
        return None


def fetch_akshare_macro() -> tuple[List[MacroObservation], List[str]]:
    import akshare as ak

    obs: List[MacroObservation] = []
    errors: List[str] = []

    # ---------------- NBS ----------------
    cpi = _safe("macro_china_cpi", ak.macro_china_cpi, errors)
    if cpi is not None:
        obs += _iter_rows(cpi, "月份", [
            ("nbs_cpi_yoy", "全国-同比增长", "%"),
            ("nbs_cpi_mom", "全国-环比增长", "%"),
        ], source="NBS", url="https://data.stats.gov.cn/")

    ppi = _safe("macro_china_ppi", ak.macro_china_ppi, errors)
    if ppi is not None:
        obs += _iter_rows(ppi, "月份", [("nbs_ppi_yoy", "当月同比增长", "%")], source="NBS")

    pmi = _safe("macro_china_pmi", ak.macro_china_pmi, errors)
    if pmi is not None:
        obs += _iter_rows(pmi, "月份", [
            ("nbs_pmi_manufacturing", "制造业-指数", "指数"),
            ("nbs_pmi_non_manufacturing", "非制造业-指数", "指数"),
        ], source="NBS")

    gdp = _safe("macro_china_gdp", ak.macro_china_gdp, errors)
    if gdp is not None:
        obs += _iter_rows(gdp, "季度", [
            ("nbs_gdp_yoy", "国内生产总值-同比增长", "%"),
            ("nbs_gdp_value", "国内生产总值-绝对值", "亿元"),
        ], source="NBS")

    retail = _safe("macro_china_consumer_goods_retail", ak.macro_china_consumer_goods_retail, errors)
    if retail is not None:
        obs += _iter_rows(retail, "月份", [
            ("nbs_retail_sales_yoy", "同比增长", "%"),
            ("nbs_retail_sales_value", "当月", "亿元"),
        ], source="NBS")

    industrial = _safe("macro_china_industrial_production_yoy", ak.macro_china_industrial_production_yoy, errors)
    if industrial is not None:
        obs += _iter_rows(industrial, "日期", [("nbs_industrial_production_yoy", "今值", "%")], source="NBS")

    # ---------------- PBOC ----------------
    money = _safe("macro_china_money_supply", ak.macro_china_money_supply, errors)
    if money is not None:
        obs += _iter_rows(money, "月份", [
            ("pbc_m2_yoy", "货币和准货币(M2)-同比增长", "%"),
            ("pbc_m2_value", "货币和准货币(M2)-数量(亿元)", "亿元"),
            ("pbc_m1_yoy", "货币(M1)-同比增长", "%"),
            ("pbc_m1_value", "货币(M1)-数量(亿元)", "亿元"),
            ("pbc_m0_yoy", "流通中的现金(M0)-同比增长", "%"),
            ("pbc_m0_value", "流通中的现金(M0)-数量(亿元)", "亿元"),
        ], source="PBOC")

    loans = _safe("macro_china_new_financial_credit", ak.macro_china_new_financial_credit, errors)
    if loans is not None:
        obs += _iter_rows(loans, "月份", [("pbc_new_rmb_loans", "当月", "亿元")], source="PBOC")

    # 社融：优先官方 Excel；akshare 仅作为兜底
    shrzgm = _safe("macro_china_shrzgm", ak.macro_china_shrzgm, errors)
    if shrzgm is not None:
        obs += _iter_rows(shrzgm, "月份", [
            ("pbc_shrzgm_increment", "社会融资规模增量", "亿元"),
            ("pbc_shrzgm_rmb_loan", "其中-人民币贷款", "亿元"),
            ("pbc_shrzgm_corporate_bond", "其中-企业债券", "亿元"),
        ], source="PBOC")

    lpr = _safe("macro_china_lpr", ak.macro_china_lpr, errors)
    if lpr is not None:
        obs += _iter_rows(lpr, "TRADE_DATE", [
            ("pbc_lpr_1y", "LPR1Y", "%"),
            ("pbc_lpr_5y", "LPR5Y", "%"),
        ], source="PBOC")

    rrr = _safe("macro_china_reserve_requirement_ratio", ak.macro_china_reserve_requirement_ratio, errors)
    if rrr is not None:
        for _, row in rrr.iterrows():
            period = parse_period(row.get("生效时间"))
            if period is None:
                continue
            for series_id, col in [
                ("pbc_rrr_large", "大型金融机构-调整后"),
                ("pbc_rrr_small", "中小金融机构-调整后"),
            ]:
                value = to_float(row.get(col))
                o = _obs(series_id, period, value, unit="%", source="PBOC")
                if o:
                    obs.append(o)

    return obs, errors
