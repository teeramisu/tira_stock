"""宏观数据解析：期间、数值清洗、NBS 网页表格、PBOC Excel 表格。"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .models import MacroObservation, RawTable


# ---------------------------------------------------------------- 基础解析
def to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text in {"--", "-", "—", "…", "nan", "None"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "").replace("，", "")
    text = re.sub(r"(亿元|万亿元|%|％|点|个|美元|人|万户|家|倍|元)", "", text).strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - __import__("datetime").timedelta(days=1)


def parse_period(value) -> Optional[date]:
    """把常见中文/数字期间解析为 date（月度取月初，季度取季末）。"""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    text = str(value).strip()
    if not text:
        return None

    # 2026年第1-3季度 / 2026年第3季度
    m = re.search(r"(\d{4})\s*年\s*第\s*(\d)\s*(?:[-—~至]\s*(\d))?\s*季度", text)
    if m:
        year = int(m.group(1))
        q = int(m.group(3) or m.group(2))
        month = q * 3
        return _month_end(year, month)

    # 2026年09月份 / 2026年9月
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)

    # 2026.09 / 2026.9
    m = re.match(r"^(\d{4})\.(\d{1,2})$", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)

    # 202609
    m = re.match(r"^(\d{4})(\d{2})$", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)

    # 2026-09-01 / 2026/09/01
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # 2026-09
    m = re.match(r"^(\d{4})[-/](\d{1,2})$", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    return None


def parse_ytd_period(text: str) -> Optional[date]:
    """解析 '2026年1—9月份' 这类累计期间，返回最后月份月末。"""
    m = re.search(r"(\d{4})\s*年\s*1\s*[—\-~至]\s*(\d{1,2})\s*月", text)
    if m:
        return _month_end(int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        return _month_end(int(m.group(1)), int(m.group(2)))
    return None


# ---------------------------------------------------------------- 原始表
def dataframe_to_raw(df: pd.DataFrame, *, source: str, title: str, url: str, table_index: int) -> RawTable:
    columns = [str(c) for c in df.columns]
    rows: List[list] = []
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append(None)
            elif isinstance(v, (datetime, date)):
                vals.append(v.isoformat())
            else:
                vals.append(str(v))
        rows.append(vals)
    return RawTable(
        source=source,
        title=title,
        url=url,
        table_index=table_index,
        columns=columns,
        rows=rows,
        fetched_at=datetime.now().isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------- NBS PMI
def normalize_nbs_pmi(tables: Sequence[pd.DataFrame], *, url: str = "") -> tuple[List[MacroObservation], List[RawTable]]:
    obs: List[MacroObservation] = []
    raws: List[RawTable] = []
    for idx, table in enumerate(tables):
        if table.empty or table.shape[1] < 2:
            continue
        header_text = " ".join(str(x) for x in table.iloc[:4].fillna("").to_numpy().ravel())
        if "PMI" not in header_text and "商务活动" not in header_text and "采购经理" not in header_text:
            continue
        target = None
        if "PMI" in header_text and "商务活动" not in header_text:
            target = "nbs_pmi_manufacturing"
        elif "商务活动" in header_text:
            target = "nbs_pmi_non_manufacturing"
        if target is None:
            continue
        raws.append(dataframe_to_raw(table, source="NBS", title="采购经理指数", url=url, table_index=idx))
        for _, row in table.iterrows():
            cells = [str(x).strip() for x in row.tolist() if str(x).strip() not in ("", "nan")]
            if not cells:
                continue
            period = parse_period(cells[0])
            if period is None:
                continue
            value = to_float(row.iloc[1])
            if value is None:
                continue
            obs.append(MacroObservation(
                series_id=target, period=period, value=value, unit="指数",
                source="NBS", provider="nbs_html", source_url=url, raw_label=str(cells[0]),
            ))
    return obs, raws


# ---------------------------------------------------------------- NBS FAI
def normalize_nbs_fai(
    tables: Sequence[pd.DataFrame],
    *,
    url: str = "",
    title: str = "",
) -> tuple[List[MacroObservation], List[RawTable]]:
    obs: List[MacroObservation] = []
    raws: List[RawTable] = []
    period = parse_ytd_period(title)
    for idx, table in enumerate(tables):
        if table.empty or table.shape[1] < 2:
            continue
        found = None
        for _, row in table.iterrows():
            label = str(row.iloc[0]).replace("　", "").strip()
            if "固定资产投资（不含农户）" in label or "固定资产投资(不含农户)" in label:
                found = to_float(row.iloc[1])
                break
        if found is None:
            continue
        raws.append(dataframe_to_raw(table, source="NBS", title=title or "固定资产投资", url=url, table_index=idx))
        if period is None:
            continue
        obs.append(MacroObservation(
            series_id="nbs_fai_yoy", period=period, value=found, unit="%",
            source="NBS", provider="nbs_html", source_url=url, raw_label="固定资产投资（不含农户）",
        ))
        break
    return obs, raws


# ---------------------------------------------------------------- PBOC 社融
def normalize_pboc_shrzgm(df: pd.DataFrame, *, url: str = "") -> tuple[List[MacroObservation], Optional[RawTable]]:
    """解析 PBOC 社会融资规模增量表（xlsx/htm 均可，header 行自动识别）。"""
    if df is None or df.empty:
        return [], None

    header_row = None
    col_map: dict[str, int] = {}
    for i in range(min(len(df), 15)):
        row = df.iloc[i].tolist()
        texts = [str(x).replace(" ", "") for x in row]
        if any("月份" in x or "Month" in x for x in texts) and any("社会融资规模增量" in x for x in texts):
            header_row = i
            for j, text in enumerate(texts):
                if "社会融资规模增量" in text:
                    col_map["increment"] = j
                elif "人民币贷款" in text:
                    col_map["rmb_loan"] = j
                elif "政府债券" in text:
                    col_map["government_bond"] = j
                elif "企业债券" in text:
                    col_map["corporate_bond"] = j
            break
    if header_row is None or "increment" not in col_map:
        return [], None

    obs: List[MacroObservation] = []
    mapping = {
        "increment": "pbc_shrzgm_increment",
        "rmb_loan": "pbc_shrzgm_rmb_loan",
        "government_bond": "pbc_shrzgm_government_bond",
        "corporate_bond": "pbc_shrzgm_corporate_bond",
    }
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i].tolist()
        period = parse_period(row[0])
        if period is None:
            continue
        for key, series_id in mapping.items():
            j = col_map.get(key)
            if j is None or j >= len(row):
                continue
            value = to_float(row[j])
            if value is None:
                continue
            obs.append(MacroObservation(
                series_id=series_id, period=period, value=value, unit="亿元",
                source="PBOC", provider="pboc_excel", source_url=url, raw_label=str(row[0]),
            ))
    raw = dataframe_to_raw(df, source="PBOC", title="社会融资规模增量统计表", url=url, table_index=0)
    return obs, raw
