"""中国分省年度数据：NBS 新 API（data.stats.gov.cn/dg/...）。

输出长表：指标 × 省份 × 年份。支持省级总量、同比变化与排名变化。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
import time
from typing import Dict, List, Tuple

import requests

BASE = "https://data.stats.gov.cn/dg/website/publicrelease/web/external"
ROOT_ID = "c4d82af16c3d4f0cb4f09d4af7d5888e"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Referer": "https://data.stats.gov.cn/dg/website/page.html",
    "Content-Type": "application/json;charset=UTF-8",
}

PROVINCES: List[Tuple[str, str]] = [
    ("北京市", "110000000000"), ("天津市", "120000000000"), ("河北省", "130000000000"),
    ("山西省", "140000000000"), ("内蒙古自治区", "150000000000"), ("辽宁省", "210000000000"),
    ("吉林省", "220000000000"), ("黑龙江省", "230000000000"), ("上海市", "310000000000"),
    ("江苏省", "320000000000"), ("浙江省", "330000000000"), ("安徽省", "340000000000"),
    ("福建省", "350000000000"), ("江西省", "360000000000"), ("山东省", "370000000000"),
    ("河南省", "410000000000"), ("湖北省", "420000000000"), ("湖南省", "430000000000"),
    ("广东省", "440000000000"), ("广西壮族自治区", "450000000000"), ("海南省", "460000000000"),
    ("重庆市", "500000000000"), ("四川省", "510000000000"), ("贵州省", "520000000000"),
    ("云南省", "530000000000"), ("西藏自治区", "540000000000"), ("陕西省", "610000000000"),
    ("甘肃省", "620000000000"), ("青海省", "630000000000"), ("宁夏回族自治区", "640000000000"),
    ("新疆维吾尔自治区", "650000000000"),
]

# indicator_id -> (name, unit, cid, nbs_indicator_id, transform)
INDICATORS: Dict[str, dict] = {
    "gdp_total": {"name": "GDP 总量", "unit": "亿元", "cid": "6f8fbd415cbc40ffa7ecb7fd917f2598",
                  "indicator": "aff57de5ee994283974705914fbed246", "transform": "level"},
    "gdp_real_growth": {"name": "GDP 实际增速", "unit": "%", "cid": "05cf8a895d3e41658f4dbc56fdfabf9b",
                        "indicator": "8fc8b9f0d07944688b199884db43cb68", "transform": "index_minus_100"},
    "population": {"name": "常住人口", "unit": "万人", "cid": "755d5f6efbcf41a6a411ad819aa93c17",
                   "indicator": "4810ad6e9ddc41fc804a66134afe587f", "transform": "level"},
    "retail_total": {"name": "社零总额", "unit": "亿元", "cid": "77cb81a53dd9438ab00f6b4f27196ad9",
                     "indicator": "3512aeee72d74be7b4008817a4dd2d03", "transform": "level"},
    "income_per_capita": {"name": "居民人均可支配收入", "unit": "元", "cid": "5c072e1073514e448de4132760308fcf",
                          "indicator": "bda568266e4c445b867d25626e1f6b8a", "transform": "level"},
    "fiscal_revenue": {"name": "地方一般公共预算收入", "unit": "亿元", "cid": "bcd3d7159957431b8f3f5b7d10bdeceb",
                       "indicator": "c8dafa31105e4b93850f3820cdd4eae8", "transform": "level"},
    "real_estate_investment": {"name": "房地产开发投资", "unit": "亿元", "cid": "d58795588d10411b883fa09af92f3cad",
                               "indicator": "6b3195bc0a94444f98418d320ee284e3", "transform": "level"},
    "cpi_yoy": {"name": "CPI 同比", "unit": "%", "cid": "d1345232d611428bb56474a4f9ce848d",
                "indicator": "dc68d688633d4806b3034821f0a3482d", "transform": "index_minus_100"},
    "industrial_va": {"name": "工业增加值", "unit": "亿元", "cid": "6f8fbd415cbc40ffa7ecb7fd917f2598",
                      "indicator": "a3ba91c8d6b84bbb8731128853eaa675", "transform": "level"},
    "manufacturing_fai_growth": {"name": "制造业固投增速", "unit": "%", "cid": "7eb1430a78174e569cea9bb6d1e1023b",
                                 "indicator": "66fbe4f88fbf4ee9a88310613fce469e", "transform": "level"},
    "real_estate_fai_growth": {"name": "房地产固投增速", "unit": "%", "cid": "7eb1430a78174e569cea9bb6d1e1023b",
                               "indicator": "5623bf058560439a92936b71cda858fa", "transform": "level"},
}


@dataclass(frozen=True)
class RegionObservation:
    indicator_id: str
    indicator_name: str
    region_code: str
    region_name: str
    period: date
    value: float
    unit: str
    provider: str = "nbs_new_api"
    source_url: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return {
            "indicator_id": self.indicator_id,
            "indicator_name": self.indicator_name,
            "region_code": self.region_code,
            "region_name": self.region_name,
            "period": self.period.isoformat(),
            "value": self.value,
            "unit": self.unit,
            "provider": self.provider,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
        }


def _fetch_one(indicator_id: str, meta: dict, province: str, code: str, dts: str,
               session: requests.Session | None = None, attempts: int = 3) -> tuple[str, list[RegionObservation], str | None]:
    body = {
        "cid": meta["cid"],
        "indicatorIds": [meta["indicator"]],
        "daCatalogId": "",
        "das": [{"text": province, "value": code}],
        "showType": "1",
        "dts": [dts],
        "rootId": ROOT_ID,
    }
    payload = None
    last_err = ""
    sess = session or requests.Session()
    sess.trust_env = False
    for attempt in range(attempts):
        try:
            resp = sess.post(BASE + "/stream/esData", json=body, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("success") is False or not payload.get("data"):
                raise RuntimeError(payload.get("message") or "empty data")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(0.5 * (attempt + 1))
    if payload is None:
        return indicator_id, [], f"{indicator_id}/{province}: {last_err}"

    return indicator_id, parse_province_payload(payload, indicator_id, meta, province, code), None


def parse_province_payload(
    payload: dict,
    indicator_id: str,
    meta: dict,
    province: str,
    code: str,
) -> list[RegionObservation]:
    """解析国家统计局分省接口返回的数据。

    Args:
        payload: ``stream/esData`` 返回的 JSON 对象。
        indicator_id: 指标标识，如 ``gdp_total``。
        meta: ``INDICATORS`` 中的指标元数据（含 name / unit / transform）。
        province: 省份名称。
        code: 省份行政区划代码。

    Returns:
        解析出的观测列表；年份或数值不可解析的条目会被跳过。
    """
    out: list[RegionObservation] = []
    for item in payload.get("data", []) or []:
        year_text = str(item.get("name") or "")
        try:
            year = int(year_text[:4])
        except ValueError:
            continue
        for val in item.get("values", []) or []:
            raw = val.get("value")
            try:
                value = float(str(raw).replace(",", "")) if raw not in (None, "", "-", "—") else None
            except ValueError:
                value = None
            if value is None:
                continue
            if meta["transform"] == "index_minus_100":
                value -= 100.0
            out.append(RegionObservation(
                indicator_id=indicator_id,
                indicator_name=meta["name"],
                region_code=code,
                region_name=province,
                period=date(year, 12, 31),
                value=round(value, 4),
                unit=meta["unit"],
                source_url=BASE + "/stream/esData",
                fetched_at=datetime.now().isoformat(timespec="seconds"),
            ))
    return out


def fetch_province_data(
    *,
    indicators: List[str] | None = None,
    start_year: int = 2019,
    end_year: int = 2025,
    workers: int = 4,
) -> tuple[list[RegionObservation], list[str]]:
    selected = indicators or list(INDICATORS)
    dts = f"{start_year}YY-{end_year}YY"
    tasks = [(iid, INDICATORS[iid], p, c) for iid in selected for p, c in PROVINCES]

    observations: list[RegionObservation] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_fetch_one, iid, meta, p, c, dts): (iid, p, c) for iid, meta, p, c in tasks}
        for fut in as_completed(futures):
            iid, rows, err = fut.result()
            observations.extend(rows)
            if err:
                errors.append(err)

    # 第二轮：对完全无数据的 (指标, 省份) 做串行重试（避免并发触发限流）
    have = {(o.indicator_id, o.region_code) for o in observations}
    missing = [(iid, meta, p, c) for iid, meta, p, c in tasks if (iid, c) not in have]
    if missing:
        retry_session = requests.Session()
        retry_session.trust_env = False
        for iid, meta, p, c in missing:
            _, rows, err = _fetch_one(iid, meta, p, c, dts, session=retry_session, attempts=5)
            observations.extend(rows)
            if err:
                errors.append(err)
            time.sleep(0.15)

    # 人均 GDP = GDP总量(亿元)*1e8 / 人口(万人)*1e4
    gdp = {(o.region_code, o.period): o.value for o in observations if o.indicator_id == "gdp_total"}
    pop = {(o.region_code, o.period): o.value for o in observations if o.indicator_id == "population"}
    for region_code, period in sorted(set(gdp) & set(pop)):
        if pop[(region_code, period)]:
            value = gdp[(region_code, period)] / pop[(region_code, period)] * 1e4
            region_name = next(o.region_name for o in observations if o.region_code == region_code)
            observations.append(RegionObservation(
                indicator_id="gdp_per_capita", indicator_name="人均 GDP",
                region_code=region_code, region_name=region_name,
                period=period, value=round(value, 2), unit="元",
                source_url=BASE + "/stream/esData",
                fetched_at=datetime.now().isoformat(timespec="seconds"),
            ))
    return observations, errors
