"""派生指标：M1-M2 剪刀差、实际利率。"""
from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Dict, List

from .models import MacroObservation


def _month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def derive_china_observations(observations: List[MacroObservation]) -> List[MacroObservation]:
    by_id: Dict[str, List[MacroObservation]] = {}
    for obs in observations:
        by_id.setdefault(obs.series_id, []).append(obs)

    out: List[MacroObservation] = []
    now = datetime.now().isoformat(timespec="seconds")

    # M1 - M2 剪刀差
    m1 = {o.period: o.value for o in by_id.get("pbc_m1_yoy", [])}
    m2 = {o.period: o.value for o in by_id.get("pbc_m2_yoy", [])}
    for period in sorted(set(m1) & set(m2)):
        out.append(MacroObservation(
            series_id="cn_m1_m2_scissors", period=period,
            value=round(m1[period] - m2[period], 4), unit="pp",
            source="CN", provider="derived", raw_label="M1同比-M2同比",
            fetched_at=now,
        ))

    # 实际利率 = LPR1Y - CPI同比
    lpr = sorted(by_id.get("pbc_lpr_1y", []), key=lambda o: o.period)
    cpi = sorted(by_id.get("nbs_cpi_yoy", []), key=lambda o: o.period)
    for c in cpi:
        end = _month_end(c.period)
        candidates = [o for o in lpr if o.period <= end]
        if not candidates:
            continue
        rate = candidates[-1].value
        out.append(MacroObservation(
            series_id="cn_real_rate", period=c.period,
            value=round(rate - c.value, 4), unit="%",
            source="CN", provider="derived", raw_label="LPR1Y-CPI同比",
            fetched_at=now,
        ))
    return out
