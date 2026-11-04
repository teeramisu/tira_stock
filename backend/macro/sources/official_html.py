"""官方网页/Excel 适配器：NBS 最新发布、PBOC 社融统计表。"""
from __future__ import annotations

import io
from datetime import datetime
from typing import List
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..models import MacroObservation, RawTable
from ..parsers import (
    dataframe_to_raw,
    normalize_nbs_fai,
    normalize_nbs_pmi,
    normalize_pboc_shrzgm,
)

NBS_INDEX = "https://www.stats.gov.cn/sj/zxfb/"
PBOC_SHRZGM_INDEX = "http://www.pbc.gov.cn/diaochatongjisi/116219/116319/2026ntjsj/shrzgm/index.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def _get(url: str, timeout: int = 30) -> requests.Response:
    last_err: Exception | None = None
    for trust_env in (True, False):
        try:
            session = requests.Session()
            session.trust_env = trust_env
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or resp.encoding
            return resp
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"fetch failed: {url}: {last_err}")


def _tables(html: str) -> List[pd.DataFrame]:
    return list(pd.read_html(io.StringIO(html)))


def _find_link(index_html: str, base_url: str, keyword: str) -> tuple[str, str] | None:
    soup = BeautifulSoup(index_html, "lxml")
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if keyword in text:
            return urljoin(base_url, a["href"]), text
    return None


def fetch_nbs_official() -> tuple[List[MacroObservation], List[RawTable], List[str]]:
    obs: List[MacroObservation] = []
    raws: List[RawTable] = []
    errors: List[str] = []
    try:
        index_html = _get(NBS_INDEX).text
    except Exception as exc:  # noqa: BLE001
        return [], [], [f"NBS index: {exc}"]

    for keyword, kind in [("采购经理指数", "pmi"), ("固定资产投资", "fai")]:
        try:
            found = _find_link(index_html, NBS_INDEX, keyword)
            if not found:
                errors.append(f"NBS {kind}: link not found")
                continue
            url, link_text = found
            resp = _get(url)
            tables = _tables(resp.text)
            if kind == "pmi":
                o, r = normalize_nbs_pmi(tables, url=url)
            else:
                title = link_text or ""
                if not title:
                    soup = BeautifulSoup(resp.text, "lxml")
                    h1 = soup.find("h1")
                    title = h1.get_text(" ", strip=True) if h1 else soup.title.get_text(strip=True) if soup.title else ""
                o, r = normalize_nbs_fai(tables, url=url, title=title)
            obs += o
            raws += r
        except Exception as exc:  # noqa: BLE001
            errors.append(f"NBS {kind}: {exc}")
    return obs, raws, errors


def _find_pboc_attachments(index_html: str, base_url: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(index_html, "lxml")
    xlsx: list[str] = []
    htm: list[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        low = href.lower()
        if low.endswith(".xlsx"):
            xlsx.append(urljoin(base_url, href))
        elif low.endswith(".htm"):
            htm.append(urljoin(base_url, href))
    return (xlsx[0] if xlsx else None), (htm[0] if htm else None)


def fetch_pboc_official() -> tuple[List[MacroObservation], List[RawTable], List[str]]:
    obs: List[MacroObservation] = []
    raws: List[RawTable] = []
    errors: List[str] = []
    try:
        index_html = _get(PBOC_SHRZGM_INDEX).text
    except Exception as exc:  # noqa: BLE001
        return [], [], [f"PBOC index: {exc}"]

    xlsx_url, htm_url = _find_pboc_attachments(index_html, PBOC_SHRZGM_INDEX)
    if xlsx_url:
        try:
            resp = _get(xlsx_url)
            df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=None)
            o, raw = normalize_pboc_shrzgm(df, url=xlsx_url)
            obs += o
            if raw:
                raws.append(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"PBOC xlsx: {exc}")
    if not obs and htm_url:
        try:
            resp = _get(htm_url)
            tables = _tables(resp.text)
            for table in tables:
                o, raw = normalize_pboc_shrzgm(table, url=htm_url)
                if o:
                    obs += o
                    if raw:
                        raws.append(raw)
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"PBOC htm: {exc}")
    return obs, raws, errors
