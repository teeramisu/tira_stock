"""中国扩展指标：投资、用电、房价、央行资产负债表、SHIBOR、两融、估值、北向。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, List

import pandas as pd

from ..models import MacroObservation
from ..parsers import parse_period, to_float


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _obs(series_id: str, period, value, *, unit: str, category: str = "",
         provider: str = "eastmoney", url: str = "", label: str = "") -> MacroObservation | None:
    if period is None or value is None:
        return None
    return MacroObservation(
        series_id=series_id, period=period, value=float(value), unit=unit,
        source="CN", provider=provider, source_url=url, raw_label=label, fetched_at=_now(),
    )


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


def fetch_cn_extra() -> tuple[List[MacroObservation], List[str]]:
    import akshare as ak

    obs: List[MacroObservation] = []
    errors: List[str] = []

    # 工业增加值（用当前口径覆盖旧的日历序列）
    df = _safe("macro_china_gyzjz", ak.macro_china_gyzjz, errors)
    if df is not None:
        for _, row in df.iterrows():
            o = _obs("nbs_industrial_production_yoy", parse_period(row.get("月份")),
                     to_float(row.get("同比增长")), unit="%", provider="eastmoney_current",
                     label=str(row.get("月份")))
            if o:
                obs.append(o)

    # 固定资产投资
    df = _safe("macro_china_gdzctz", ak.macro_china_gdzctz, errors)
    if df is not None:
        for _, row in df.iterrows():
            period = parse_period(row.get("月份"))
            for sid, col, unit in [
                ("cn_fai_month_yoy", "同比增长", "%"),
                ("cn_fai_cum_value", "自年初累计", "亿元"),
            ]:
                o = _obs(sid, period, to_float(row.get(col)), unit=unit, label=str(row.get("月份")))
                if o:
                    obs.append(o)

    # 全社会用电量
    df = _safe("macro_china_society_electricity", ak.macro_china_society_electricity, errors)
    if df is not None:
        for _, row in df.iterrows():
            o = _obs("cn_electricity_yoy", parse_period(row.get("统计时间")),
                     to_float(row.get("全社会用电量同比")), unit="%",
                     label=str(row.get("统计时间")))
            if o:
                obs.append(o)

    # 70 城房价：取城市中位数，并转换为同比百分比
    df = _safe("macro_china_new_house_price", ak.macro_china_new_house_price, errors)
    if df is not None:
        for col, sid in [
            ("新建商品住宅价格指数-同比", "cn_new_house_price_yoy"),
            ("二手住宅价格指数-同比", "cn_second_hand_price_yoy"),
        ]:
            if col not in df.columns:
                continue
            grouped = df.assign(_p=pd.to_datetime(df["日期"], errors="coerce")).dropna(subset=["_p"])
            grouped["_v"] = pd.to_numeric(grouped[col], errors="coerce")
            for period, value in grouped.groupby(grouped["_p"].dt.date)["_v"].median().items():
                o = _obs(sid, period, value - 100.0 if pd.notna(value) else None,
                         unit="%", label=f"70城中位数 {col}")
                if o:
                    obs.append(o)

    # 央行资产负债表
    df = _safe("macro_china_central_bank_balance", ak.macro_china_central_bank_balance, errors)
    if df is not None:
        for _, row in df.iterrows():
            period = parse_period(row.get("统计时间"))
            for sid, col in [
                ("cn_cb_total_assets", "总资产"),
                ("cn_cb_reserve_money", "储备货币"),
            ]:
                o = _obs(sid, period, to_float(row.get(col)), unit="亿元",
                         label=str(row.get("统计时间")))
                if o:
                    obs.append(o)

    # SHIBOR
    df = _safe("macro_china_shibor_all", ak.macro_china_shibor_all, errors)
    if df is not None:
        for _, row in df.iterrows():
            period = parse_period(row.get("日期"))
            for sid, col in [("cn_shibor_3m", "3M-定价"), ("cn_shibor_1y", "1Y-定价")]:
                o = _obs(sid, period, to_float(row.get(col)), unit="%",
                         label=str(row.get("日期")))
                if o:
                    obs.append(o)

    # 两融余额（沪深合计，亿元）
    sh = _safe("macro_china_market_margin_sh", ak.macro_china_market_margin_sh, errors)
    sz = _safe("macro_china_market_margin_sz", ak.macro_china_market_margin_sz, errors)
    if sh is not None and sz is not None:
        a = sh[["日期", "融资融券余额"]].copy()
        b = sz[["日期", "融资融券余额"]].copy()
        a["日期"] = pd.to_datetime(a["日期"], errors="coerce")
        b["日期"] = pd.to_datetime(b["日期"], errors="coerce")
        merged = a.merge(b, on="日期", suffixes=("_sh", "_sz"))
        for _, row in merged.iterrows():
            v_sh = to_float(row["融资融券余额_sh"])
            v_sz = to_float(row["融资融券余额_sz"])
            if v_sh is None or v_sz is None:
                continue
            o = _obs("cn_margin_balance", row["日期"].date(), (v_sh + v_sz) / 1e8,
                     unit="亿元", label=str(row["日期"].date()))
            if o:
                obs.append(o)

    # 全 A 估值分位（自行计算近10年分位，保证历史序列完整）
    def _append_valuation(df: pd.DataFrame, date_col: str, value_col: str, median_id: str, pct_id: str, unit: str) -> None:
        if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
            return
        frame = df[[date_col, value_col]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        frame = frame.dropna().sort_values(date_col)
        if frame.empty:
            return
        for _, row in frame.iterrows():
            d = row[date_col].date()
            v = float(row[value_col])
            window = frame[(frame[date_col] >= row[date_col] - pd.DateOffset(years=10)) & (frame[date_col] <= row[date_col])]
            pct = float((window[value_col] < v).mean() * 100) if len(window) else 50.0
            for sid, value, u in [(median_id, v, unit), (pct_id, pct, "%")]:
                o = _obs(sid, d, value, unit=u, label=str(d))
                if o:
                    obs.append(o)

    pe = _safe("stock_a_ttm_lyr", ak.stock_a_ttm_lyr, errors)
    _append_valuation(pe, "date", "middlePETTM", "cn_a_pe_median", "cn_a_pe_10y_pct", "倍")

    pb = _safe("stock_a_all_pb", ak.stock_a_all_pb, errors)
    _append_valuation(pb, "date", "middlePB", "cn_a_pb_median", "cn_a_pb_10y_pct", "倍")

    # 北向资金（披露口径变化，仅保留有值日期）
    hsgt = _safe("stock_hsgt_hist_em", ak.stock_hsgt_hist_em, errors)
    if hsgt is not None:
        for _, row in hsgt.iterrows():
            o = _obs("cn_hsgt_net_buy", parse_period(row.get("日期")),
                     to_float(row.get("当日成交净买额")), unit="亿元",
                     label=str(row.get("日期")))
            if o:
                obs.append(o)

    return obs, errors
