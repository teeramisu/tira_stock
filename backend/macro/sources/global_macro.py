"""美国/日本宏观适配器：akshare 全球宏观 + 世界银行年度补充 + 中美利率曲线。"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, List, Optional

import pandas as pd
import requests

from ..models import MacroObservation
from ..parsers import parse_period, to_float


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _obs(series_id: str, period, value, *, unit: str, source: str, country: str,
         provider: str, url: str = "", label: str = "") -> Optional[MacroObservation]:
    if period is None or value is None:
        return None
    return MacroObservation(
        series_id=series_id, period=period, value=float(value), unit=unit,
        source=source, provider=provider, source_url=url, raw_label=label, fetched_at=_now(),
    )


def _from_frame(df: pd.DataFrame | None, series_id: str, date_col: str, value_col: str,
                *, unit: str, source: str, country: str, provider: str = "eastmoney",
                url: str = "") -> List[MacroObservation]:
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return []
    out: List[MacroObservation] = []
    for _, row in df.iterrows():
        period = parse_period(row.get(date_col))
        value = to_float(row.get(value_col))
        o = _obs(series_id, period, value, unit=unit, source=source, country=country,
                 provider=provider, url=url, label=str(row.get(date_col)))
        if o:
            out.append(o)
    return out


def _safe(name: str, fn: Callable, errors: List[str]) -> pd.DataFrame | None:
    try:
        df = fn()
        if df is None or df.empty:
            errors.append(f"{name}: empty")
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{name}: {exc}")
        return None


def fetch_us_japan_akshare() -> tuple[List[MacroObservation], List[str]]:
    import akshare as ak

    obs: List[MacroObservation] = []
    errors: List[str] = []

    # ---------------- 美国 ----------------
    us_specs = [
        ("macro_usa_cpi_yoy", "us_cpi_yoy", "时间", "现值", "%"),
        ("macro_usa_core_cpi_monthly", "us_core_cpi_mom", "日期", "今值", "%"),
        ("macro_usa_core_pce_price", "us_core_pce_yoy", "日期", "今值", "%"),
        ("macro_usa_unemployment_rate", "us_unemployment", "日期", "今值", "%"),
        ("macro_usa_non_farm", "us_nonfarm_payrolls", "日期", "今值", "万人"),
        ("macro_usa_gdp_monthly", "us_gdp_growth", "日期", "今值", "%"),
        ("macro_usa_ism_pmi", "us_ism_pmi", "日期", "今值", "指数"),
        ("macro_usa_retail_sales", "us_retail_sales_mom", "日期", "今值", "%"),
        ("macro_usa_industrial_production", "us_industrial_production_mom", "日期", "今值", "%"),
        ("macro_usa_michigan_consumer_sentiment", "us_consumer_sentiment", "日期", "今值", "指数"),
        ("macro_usa_trade_balance", "us_trade_balance", "日期", "今值", "亿美元"),
        ("macro_bank_usa_interest_rate", "us_fed_funds", "日期", "今值", "%"),
    ]
    for fn_name, series_id, date_col, value_col, unit in us_specs:
        df = _safe(fn_name, getattr(ak, fn_name), errors)
        obs += _from_frame(df, series_id, date_col, value_col, unit=unit,
                           source="US", country="US")

    # ---------------- 日本 ----------------
    jp_specs = [
        ("macro_japan_cpi_yearly", "jp_cpi_yoy", "时间", "现值", "%"),
        ("macro_japan_core_cpi_yearly", "jp_core_cpi_yoy", "时间", "现值", "%"),
        ("macro_japan_unemployment_rate", "jp_unemployment", "时间", "现值", "%"),
        ("macro_japan_bank_rate", "jp_boj_rate", "时间", "现值", "%"),
        ("macro_japan_head_indicator", "jp_leading_indicator", "时间", "现值", "指数"),
    ]
    for fn_name, series_id, date_col, value_col, unit in jp_specs:
        df = _safe(fn_name, getattr(ak, fn_name), errors)
        obs += _from_frame(df, series_id, date_col, value_col, unit=unit,
                           source="JP", country="JP")

    # ---------------- 中美国债收益率 ----------------
    rates = _safe("bond_zh_us_rate", ak.bond_zh_us_rate, errors)
    if rates is not None:
        mapping = {
            "中国国债收益率2年": ("cn_gov_2y", "CN"),
            "中国国债收益率5年": ("cn_gov_5y", "CN"),
            "中国国债收益率10年": ("cn_gov_10y", "CN"),
            "中国国债收益率30年": ("cn_gov_30y", "CN"),
            "中国国债收益率10年-2年": ("cn_gov_10y_2y", "CN"),
            "美国国债收益率2年": ("us_gov_2y", "US"),
            "美国国债收益率10年": ("us_gov_10y", "US"),
            "美国国债收益率30年": ("us_gov_30y", "US"),
            "美国国债收益率10年-2年": ("us_gov_10y_2y", "US"),
        }
        for col, (series_id, country) in mapping.items():
            if col not in rates.columns:
                continue
            for _, row in rates.iterrows():
                period = parse_period(row.get("日期"))
                value = to_float(row.get(col))
                o = _obs(series_id, period, value, unit="%", source=country,
                         country=country, provider="eastmoney", url="https://data.eastmoney.com/",
                         label=str(row.get("日期")))
                if o:
                    obs.append(o)

    return obs, errors


# ---------------------------------------------------------------- 世界银行
_WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": ("gdp_annual", "实际 GDP 增速"),
    "FP.CPI.TOTL.ZG": ("inflation_annual", "CPI 通胀"),
    "SL.UEM.TOTL.ZS": ("unemployment_annual", "ILO 失业率"),
    "NV.IND.MANF.KD.ZG": ("manufacturing_growth", "制造业增加值增速"),
    "BN.CAB.XOKA.GD.ZS": ("current_account_gdp", "经常账户/GDP"),
}
_WB_COUNTRIES = {"USA": "US", "JPN": "JP", "CHN": "CN"}


def _wb_get(country: str, indicator: str) -> list:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 200, "date": "2015:2026"}
    last_err: Exception | None = None
    for trust_env in (False, True):
        try:
            session = requests.Session()
            session.trust_env = trust_env
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data[1] if isinstance(data, list) and len(data) > 1 and data[1] else []
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"worldbank {country}/{indicator}: {last_err}")


def fetch_worldbank() -> tuple[List[MacroObservation], List[str]]:
    obs: List[MacroObservation] = []
    errors: List[str] = []
    for code, (suffix, label) in _WB_INDICATORS.items():
        for wb_country, country in _WB_COUNTRIES.items():
            series_id = f"{country.lower()}_{suffix}"
            try:
                rows = _wb_get(wb_country, code)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                continue
            for row in rows:
                value = to_float(row.get("value"))
                if value is None:
                    continue
                year = row.get("date")
                try:
                    period = date(int(year), 12, 31)
                except Exception:
                    continue
                obs.append(MacroObservation(
                    series_id=series_id, period=period, value=value, unit="%",
                    source="WorldBank", provider="worldbank",
                    source_url=f"https://data.worldbank.org/indicator/{code}",
                    raw_label=f"{label} {year}", fetched_at=_now(),
                ))
    return obs, errors
