"""宏观解读与未来 1-2 季度判断（规则化，非投资建议）。"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


def _series_map(payload: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for country in payload.get("countries", []):
        for item in country.get("series", []):
            out[item["series_id"]] = item
    return out


def _latest(series: Dict[str, dict], sid: str) -> Optional[float]:
    item = series.get(sid)
    if not item:
        return None
    value = item.get("latest")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Optional[float], unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "—"
    if unit == "%":
        return f"{value:.{digits}f}%"
    if unit == "亿元":
        return f"{value:,.0f} 亿元"
    if unit == "倍":
        return f"{value:.2f} 倍"
    if unit == "pp":
        return f"{value:+.2f}pp"
    return f"{value:.{digits}f}"


def _tone(score: float) -> str:
    if score >= 60:
        return "positive"
    if score <= 40:
        return "negative"
    return "neutral"


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _pct_change(observations: List[list], lookback: int) -> Optional[float]:
    if len(observations) <= lookback:
        return None
    first = observations[-lookback - 1][1]
    last = observations[-1][1]
    if not first:
        return None
    return (last / first - 1) * 100


def _year_ago_value(observations: List[list]) -> Optional[float]:
    if not observations:
        return None
    latest = pd.Timestamp(observations[-1][0])
    target = latest - pd.Timedelta(days=365)
    candidates = [o for o in observations[:-1] if abs((pd.Timestamp(o[0]) - target).days) <= 60]
    return float(candidates[-1][1]) if candidates else None


def _china_analysis(series: Dict[str, dict]) -> dict:
    pmi = _latest(series, "nbs_pmi_manufacturing")
    ind = _latest(series, "nbs_industrial_production_yoy")
    elec = _latest(series, "cn_electricity_yoy")
    fai = _latest(series, "cn_fai_month_yoy")
    cpi = _latest(series, "nbs_cpi_yoy")
    ppi = _latest(series, "nbs_ppi_yoy")
    m1 = _latest(series, "pbc_m1_yoy")
    m2 = _latest(series, "pbc_m2_yoy")
    scissors = _latest(series, "cn_m1_m2_scissors")
    shibor = _latest(series, "cn_shibor_3m")
    lpr = _latest(series, "pbc_lpr_1y")
    real_rate = _latest(series, "cn_real_rate")
    cb_assets = _latest(series, "cn_cb_total_assets")
    cb_series = series.get("cn_cb_total_assets", {})
    cb_yoy = None
    if cb_series.get("observations"):
        base = _year_ago_value(cb_series["observations"])
        # latest 可能是字符串（接口原样返回），必须走 _latest 做类型转换，
        # 否则这里会 TypeError: str / float
        latest_cb = _latest(series, "cn_cb_total_assets")
        if base and latest_cb is not None:
            cb_yoy = (latest_cb / base - 1) * 100
    house = _latest(series, "cn_new_house_price_yoy")
    house2 = _latest(series, "cn_second_hand_price_yoy")
    pe_pct = _latest(series, "cn_a_pe_10y_pct")
    pb_pct = _latest(series, "cn_a_pb_10y_pct")
    pe = _latest(series, "cn_a_pe_median")
    pb = _latest(series, "cn_a_pb_median")
    margin_series = series.get("cn_margin_balance", {})
    margin = _latest(series, "cn_margin_balance")
    margin_chg = _pct_change(margin_series.get("observations", []), 20) if margin_series else None

    # ---- 增长
    growth = 50.0
    if pmi is not None:
        growth += (pmi - 50) * 4
    if ind is not None:
        growth += (ind - 5) * 2
    if elec is not None:
        growth += (elec - 3) * 2
    if fai is not None:
        growth += max(-12.0, min(12.0, fai * 0.8))
    growth = _clamp(growth)
    if growth >= 65:
        growth_label = "扩张"
    elif growth >= 50:
        growth_label = "温和修复"
    elif growth >= 40:
        growth_label = "弱修复"
    else:
        growth_label = "走弱"

    # ---- 通胀
    inflation = 55.0
    if cpi is None:
        inflation_label = "数据不足"
    elif cpi < 0:
        inflation -= 20
        inflation_label = "通缩压力"
    elif cpi < 1:
        inflation += 5
        inflation_label = "低通胀"
    elif cpi <= 3:
        inflation += 10
        inflation_label = "温和通胀"
    else:
        inflation -= 10
        inflation_label = "通胀偏高"
    if ppi is not None and cpi is not None:
        if ppi > cpi + 3:
            inflation -= 10
            inflation_label += "·上游涨价"
        elif ppi < 0:
            inflation -= 8
            inflation_label += "·工业品通缩"
    inflation = _clamp(inflation)

    # ---- 流动性
    liquidity = 50.0
    if scissors is not None:
        liquidity += 8 if scissors > 0 else -8
    if shibor is not None:
        if shibor < 2:
            liquidity += 10
        elif shibor > 3:
            liquidity -= 10
    if lpr is not None and lpr <= 3.5:
        liquidity += 5
    if m2 is not None:
        liquidity += 8 if m2 >= 8 else (-5 if m2 < 7 else 0)
    if cb_yoy is not None and cb_yoy > 0:
        liquidity += 5
    liquidity = _clamp(liquidity)
    if liquidity >= 65:
        liquidity_label = "宽松"
    elif liquidity >= 50:
        liquidity_label = "中性偏松"
    elif liquidity >= 40:
        liquidity_label = "中性"
    else:
        liquidity_label = "收紧"

    # ---- 地产
    property_score = 50.0
    if house is not None:
        property_score += max(-30.0, min(15.0, house * 4))
    if house2 is not None:
        property_score += max(-15.0, min(10.0, house2 * 2))
    property_score = _clamp(property_score)
    if house is None:
        property_label = "数据不足"
    elif house < -3:
        property_label = "深度磨底"
    elif house < 0:
        property_label = "磨底"
    elif house < 2:
        property_label = "企稳"
    else:
        property_label = "回升"

    # ---- 估值（分位越低越便宜，得分越高）
    if pe_pct is not None and pb_pct is not None:
        valuation = _clamp(100 - (pe_pct + pb_pct) / 2)
        avg_pct = (pe_pct + pb_pct) / 2
        valuation_label = "偏低" if avg_pct < 40 else ("偏高" if avg_pct > 70 else "中性")
    else:
        valuation = 50.0
        valuation_label = "数据不足"

    # ---- 情绪/杠杆
    sentiment = 50.0
    if margin_chg is not None:
        sentiment += max(-20.0, min(20.0, margin_chg * 1.5))
    sentiment = _clamp(sentiment)
    if margin_chg is None:
        sentiment_label = "数据不足"
    elif margin_chg > 2:
        sentiment_label = "杠杆回升"
    elif margin_chg < -2:
        sentiment_label = "杠杆回落"
    else:
        sentiment_label = "杠杆平稳"

    # ---- 未来 1-2 季度
    if growth >= 55 and liquidity >= 60 and valuation >= 55:
        stance = "震荡偏强"
        stance_tone = "positive"
    elif growth < 40 and property_score < 40:
        stance = "震荡偏弱"
        stance_tone = "negative"
    elif liquidity >= 60 and valuation >= 60:
        stance = "区间震荡，结构性机会"
        stance_tone = "neutral"
    else:
        stance = "区间震荡"
        stance_tone = "neutral"

    summary = (
        f"增长端 PMI {_fmt(pmi)}、工业增加值 {_fmt(ind, '%')}、用电量 {_fmt(elec, '%')}，"
        f"但固投当月 {_fmt(fai, '%')}，呈“生产有韧性、投资偏弱”；"
        f"通胀端 CPI {_fmt(cpi, '%')}、PPI {_fmt(ppi, '%')}；"
        f"流动性 M1-M2 剪刀差 {_fmt(scissors, 'pp')}、SHIBOR 3M {_fmt(shibor, '%')}，"
        f"资金价格低但活化不足；"
        f"地产 70 城新房/二手房价同比 {_fmt(house, '%')} / {_fmt(house2, '%')}；"
        f"估值 PE/PB 分位 {_fmt(pe_pct, '%')} / {_fmt(pb_pct, '%')}，整体{valuation_label}。"
    )

    catalysts = []
    risks = []
    watch = []
    if liquidity >= 55:
        catalysts.append("货币政策维持宽松，降准/降息仍有空间（关注汇率与银行息差约束）")
    if property_score < 45:
        risks.append("地产销售/投资仍在磨底，拖累居民资产负债表与地方财政")
        watch.append("70 城房价环比、地产销售面积、新开工")
    if fai is not None and fai < 0:
        risks.append("固定资产投资当月同比为负，基建/制造业投资需政策接力")
        watch.append("专项债发行、基建投资、制造业中长期贷款")
    if ppi is not None and cpi is not None and ppi > cpi:
        risks.append("PPI 高于 CPI，上游涨价对中下游利润形成挤压")
    if valuation_label == "偏低":
        catalysts.append("估值处于历史偏低分位，若盈利预期修复，指数弹性较大")
    elif valuation_label == "偏高":
        risks.append("估值分位偏高，对盈利兑现要求更高")
    if margin_chg is not None and margin_chg > 2:
        catalysts.append("两融余额回升，风险偏好改善")
    if growth < 50:
        watch.append("PMI 能否回到 50 以上、用电量与工业增加值能否延续")
    if not watch:
        watch = ["PMI、CPI/PPI、社融与 LPR"]

    return {
        "as_of": payload_as_of(series),
        "signals": [
            {"key": "growth", "label": growth_label, "score": round(growth, 1), "tone": _tone(growth),
             "detail": f"PMI {_fmt(pmi)} · 工业增加值 {_fmt(ind, '%')} · 用电量 {_fmt(elec, '%')} · 固投当月 {_fmt(fai, '%')}"},
            {"key": "inflation", "label": inflation_label, "score": round(inflation, 1), "tone": _tone(inflation),
             "detail": f"CPI {_fmt(cpi, '%')} · PPI {_fmt(ppi, '%')}"},
            {"key": "liquidity", "label": liquidity_label, "score": round(liquidity, 1), "tone": _tone(liquidity),
             "detail": f"M1-M2 {_fmt(scissors, 'pp')} · 实际利率 {_fmt(real_rate, '%')} · SHIBOR 3M {_fmt(shibor, '%')} · LPR {_fmt(lpr, '%')} · 央行总资产同比 {_fmt(cb_yoy, '%')}"},
            {"key": "property", "label": property_label, "score": round(property_score, 1), "tone": _tone(property_score),
             "detail": f"新房 {_fmt(house, '%')} · 二手 {_fmt(house2, '%')}"},
            {"key": "valuation", "label": valuation_label, "score": round(valuation, 1), "tone": _tone(valuation),
             "detail": f"PE {_fmt(pe, '倍')}（{_fmt(pe_pct, '%')} 分位）· PB {_fmt(pb, '倍')}（{_fmt(pb_pct, '%')} 分位）"},
            {"key": "sentiment", "label": sentiment_label, "score": round(sentiment, 1), "tone": _tone(sentiment),
             "detail": f"两融余额 {_fmt(margin, '亿元')} · 20日变化 {_fmt(margin_chg, '%')}"},
        ],
        "outlook": {
            "horizon": "未来 1–2 季度",
            "stance": stance,
            "tone": stance_tone,
            "summary": summary,
            "catalysts": catalysts[:4],
            "risks": risks[:4],
            "watch": watch[:4],
        },
        "commentary": summary,
        "disclaimer": "以上为基于公开数据的规则化判断，不构成投资建议；市场有风险，决策需结合自身约束。",
    }


def payload_as_of(series: Dict[str, dict]) -> str:
    dates = [s.get("latest_period") for s in series.values() if s.get("latest_period")]
    return max(dates) if dates else ""


def _simple_analysis(series: Dict[str, dict], country: str) -> dict:
    if country == "us":
        cpi = _latest(series, "us_cpi_yoy")
        unemp = _latest(series, "us_unemployment")
        ism = _latest(series, "us_ism_pmi")
        y2 = _latest(series, "us_gov_2y")
        y10 = _latest(series, "us_gov_10y")
        gdp = _latest(series, "us_gdp_annual")
        signals = [
            {"key": "growth", "label": "温和" if (ism or 50) >= 50 else "走弱",
             "score": round(_clamp(50 + ((ism or 50) - 50) * 2), 1),
             "tone": "positive" if (ism or 50) >= 50 else "negative",
             "detail": f"GDP(年) {_fmt(gdp, '%')} · ISM {_fmt(ism)} · 失业率 {_fmt(unemp, '%')}"},
            {"key": "inflation", "label": "偏高" if (cpi or 0) > 3 else "温和",
             "score": round(_clamp(100 - (cpi or 0) * 12), 1),
             "tone": "negative" if (cpi or 0) > 3 else "neutral",
             "detail": f"CPI {_fmt(cpi, '%')}"},
            {"key": "rates", "label": "曲线正常" if (y10 or 0) >= (y2 or 0) else "曲线倒挂",
             "score": 60 if (y10 or 0) >= (y2 or 0) else 35,
             "tone": "positive" if (y10 or 0) >= (y2 or 0) else "negative",
             "detail": f"2Y {_fmt(y2, '%')} · 10Y {_fmt(y10, '%')}"},
        ]
        summary = f"美国 CPI {_fmt(cpi, '%')}、ISM {_fmt(ism)}、2Y/10Y {_fmt(y2, '%')} / {_fmt(y10, '%')}。"
    else:
        cpi = _latest(series, "jp_cpi_yoy")
        core = _latest(series, "jp_core_cpi_yoy")
        unemp = _latest(series, "jp_unemployment")
        rate = _latest(series, "jp_boj_rate")
        lead = _latest(series, "jp_leading_indicator")
        signals = [
            {"key": "growth", "label": "扩张" if (lead or 100) >= 100 else "走弱",
             "score": round(_clamp(50 + ((lead or 100) - 100) * 1.5), 1),
             "tone": "positive" if (lead or 100) >= 100 else "negative",
             "detail": f"领先指标 {_fmt(lead)} · 失业率 {_fmt(unemp, '%')}"},
            {"key": "inflation", "label": "再通胀" if (cpi or 0) > 1 else "低通胀",
             "score": round(_clamp(50 + (cpi or 0) * 8), 1),
             "tone": "positive" if (cpi or 0) > 1 else "neutral",
             "detail": f"CPI {_fmt(cpi, '%')} · 核心 {_fmt(core, '%')}"},
            {"key": "rates", "label": "加息周期" if (rate or 0) > 0.5 else "低利率",
             "score": round(_clamp(40 + (rate or 0) * 8), 1),
             "tone": "neutral",
             "detail": f"BOJ 政策利率 {_fmt(rate, '%')}"},
        ]
        summary = f"日本 CPI {_fmt(cpi, '%')}、核心 CPI {_fmt(core, '%')}、BOJ 利率 {_fmt(rate, '%')}、失业率 {_fmt(unemp, '%')}。"

    return {
        "as_of": payload_as_of(series),
        "signals": signals,
        "outlook": {
            "horizon": "未来 1–2 季度",
            "stance": "区间震荡",
            "tone": "neutral",
            "summary": summary + " 关注通胀与政策利率路径对估值的影响。",
            "catalysts": ["通胀回落/政策转向", "盈利预期修复"],
            "risks": ["通胀超预期", "外部需求走弱"],
            "watch": ["CPI", "政策利率", "PMI/领先指标"],
        },
        "commentary": summary,
        "disclaimer": "以上为基于公开数据的规则化判断，不构成投资建议。",
    }


def build_analysis(payload: dict) -> Dict[str, dict]:
    series = _series_map(payload)
    out: Dict[str, dict] = {}
    if any(k.startswith(("nbs_", "pbc_", "cn_")) for k in series):
        out["cn"] = _china_analysis(series)
    if any(k.startswith("us_") for k in series):
        out["us"] = _simple_analysis(series, "us")
    if any(k.startswith("jp_") for k in series):
        out["jp"] = _simple_analysis(series, "jp")
    return out
