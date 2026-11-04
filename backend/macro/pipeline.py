"""宏观数据刷新管线：官方网页/Excel + akshare 兜底。"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Iterable, Sequence

from .models import SERIES, MacroObservation, RawTable
from .derive import derive_china_observations
from .sources.akshare_macro import fetch_akshare_macro
from .sources.cn_extra_macro import fetch_cn_extra
from .sources.global_macro import fetch_us_japan_akshare, fetch_worldbank
from .provinces import fetch_province_data
from .sources.official_html import fetch_nbs_official, fetch_pboc_official
from .storage import MacroStore


def refresh(
    sources: Sequence[str] = ("akshare", "cn_extra", "nbs", "pbc", "usjp", "worldbank"),
    *,
    store: MacroStore | None = None,
    export: bool = True,
    export_dir: str | None = None,
) -> dict:
    """刷新宏观数据。

    sources:
      - akshare: NBS/PBOC 指标的稳定序列（东方财富镜像）
      - nbs: 国家统计局最新发布网页表格（PMI / 固定资产投资）
      - pbc: 人民银行社融统计表（Excel/HTML）
    """
    store = store or MacroStore()
    run_id = f"macro_{uuid.uuid4().hex[:12]}"
    started = datetime.now().isoformat(timespec="seconds")
    observations: list[MacroObservation] = []
    region_observations: list = []
    raw_tables: list[RawTable] = []
    errors: list[str] = []

    selected = {s.strip().lower() for s in sources}
    if "akshare" in selected or "all" in selected:
        obs, errs = fetch_akshare_macro()
        observations += obs
        errors += errs
    if "cn_extra" in selected or "all" in selected:
        obs, errs = fetch_cn_extra()
        observations += obs
        errors += errs
    if "nbs" in selected or "all" in selected:
        obs, raws, errs = fetch_nbs_official()
        observations += obs
        raw_tables += raws
        errors += errs
    if "pbc" in selected or "all" in selected:
        obs, raws, errs = fetch_pboc_official()
        observations += obs
        raw_tables += raws
        errors += errs
    if "usjp" in selected or "global" in selected or "all" in selected:
        obs, errs = fetch_us_japan_akshare()
        observations += obs
        errors += errs
    if "worldbank" in selected or "global" in selected or "all" in selected:
        obs, errs = fetch_worldbank()
        observations += obs
        errors += errs
    if "provinces" in selected or "all" in selected:
        region_observations, errs = fetch_province_data()
        errors += errs

    # 派生指标（M1-M2 剪刀差、实际利率）
    observations += derive_china_observations(observations)

    store.upsert_series(SERIES.values())
    n_obs = store.upsert_observations(observations)
    n_regions = store.upsert_region_observations(region_observations)
    n_raw = store.record_raw_tables(raw_tables)

    status = "ok" if not errors else "partial"
    detail = json.dumps({
        "observations": n_obs,
        "raw_tables": n_raw,
        "region_observations": n_regions,
        "errors": errors[:30],
        "series": len({o.series_id for o in observations}),
    }, ensure_ascii=False)
    store.log_run(run_id, sources, status, detail, started)

    exports = store.export_all(export_dir) if export else {}
    return {
        "run_id": run_id,
        "status": status,
        "observations": n_obs,
        "raw_tables": n_raw,
        "region_observations": n_regions,
        "series": len({o.series_id for o in observations}),
        "errors": errors,
        "exports": exports,
        "db": str(store.path),
    }
