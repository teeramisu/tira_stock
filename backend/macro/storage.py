"""宏观数据存储：DuckDB + 长表/宽表 CSV/Excel 导出。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import duckdb
import pandas as pd

from .models import SERIES, MacroObservation, MacroSeries, RawTable
from .provinces import RegionObservation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "macro" / "macro.duckdb"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "macro" / "exports"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_series (
    series_id TEXT PRIMARY KEY,
    name_cn TEXT NOT NULL,
    source TEXT NOT NULL,
    frequency TEXT NOT NULL,
    unit TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    country TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS macro_observations (
    series_id TEXT NOT NULL,
    period DATE NOT NULL,
    value DOUBLE NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    raw_label TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (series_id, period)
);
CREATE TABLE IF NOT EXISTS macro_raw_tables (
    table_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    table_index INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    rows_json TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS macro_region_observations (
    indicator_id TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    region_code TEXT NOT NULL,
    region_name TEXT NOT NULL,
    period DATE NOT NULL,
    value DOUBLE NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (indicator_id, region_code, period)
);
CREATE TABLE IF NOT EXISTS macro_fetch_log (
    run_id TEXT PRIMARY KEY,
    sources TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL
);
"""


def _date_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


class MacroStore:
    def __init__(self, db_path: Optional[str | Path] = None):
        self.path = Path(db_path) if db_path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        try:
            return duckdb.connect(str(self.path), read_only=read_only)
        except Exception:
            return duckdb.connect(str(self.path))

    def _init_schema(self) -> None:
        con = self._connect()
        try:
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    con.execute(statement)
            # 兼容旧库：补充 country 列
            try:
                con.execute("ALTER TABLE macro_series ADD COLUMN IF NOT EXISTS country TEXT DEFAULT ''")
            except Exception:
                pass
        finally:
            con.close()

    # ------------------------------------------------------------ 写入
    def upsert_series(self, series: Iterable[MacroSeries] = SERIES.values()) -> int:
        rows = [
            {
                "series_id": s.series_id,
                "name_cn": s.name_cn,
                "source": s.source,
                "frequency": s.frequency,
                "unit": s.unit,
                "category": s.category,
                "description": s.description,
                "country": s.country,
                "updated_at": datetime.now(),
            }
            for s in series
        ]
        if not rows:
            return 0
        con = self._connect()
        try:
            con.register("_series_rows", pd.DataFrame(rows))
            con.execute(
                "INSERT OR REPLACE INTO macro_series "
                "(series_id, name_cn, source, frequency, unit, category, description, country, updated_at) "
                "SELECT series_id, name_cn, source, frequency, unit, category, description, country, updated_at "
                "FROM _series_rows"
            )
            con.unregister("_series_rows")
        finally:
            con.close()
        return len(rows)

    def upsert_observations(self, observations: Sequence[MacroObservation]) -> int:
        if not observations:
            return 0
        # 同一 (series, period) 只保留最后一条
        deduped: dict[tuple[str, date], MacroObservation] = {}
        for obs in observations:
            deduped[(obs.series_id, obs.period)] = obs
        rows = [
            {
                "series_id": obs.series_id,
                "period": obs.period,
                "value": float(obs.value),
                "unit": obs.unit,
                "source": obs.source,
                "provider": obs.provider,
                "source_url": obs.source_url,
                "raw_label": obs.raw_label,
                "fetched_at": obs.fetched_at or datetime.now().isoformat(timespec="seconds"),
            }
            for obs in deduped.values()
        ]
        con = self._connect()
        try:
            con.register("_obs_rows", pd.DataFrame(rows))
            con.execute("INSERT OR REPLACE INTO macro_observations SELECT * FROM _obs_rows")
            con.unregister("_obs_rows")
        finally:
            con.close()
        return len(rows)

    def record_raw_tables(self, tables: Sequence[RawTable]) -> int:
        if not tables:
            return 0
        rows = []
        for table in tables:
            key = f"{table.source}|{table.url}|{table.table_index}".encode("utf-8")
            table_id = hashlib.sha1(key).hexdigest()
            rows.append({
                "table_id": table_id,
                "source": table.source,
                "title": table.title,
                "url": table.url,
                "table_index": int(table.table_index),
                "columns_json": json.dumps(table.columns, ensure_ascii=False),
                "rows_json": json.dumps(table.rows, ensure_ascii=False),
                "row_count": len(table.rows),
                "fetched_at": table.fetched_at or datetime.now().isoformat(timespec="seconds"),
            })
        con = self._connect()
        try:
            con.register("_raw_rows", pd.DataFrame(rows))
            con.execute("INSERT OR REPLACE INTO macro_raw_tables SELECT * FROM _raw_rows")
            con.unregister("_raw_rows")
        finally:
            con.close()
        return len(rows)

    def log_run(self, run_id: str, sources: Sequence[str], status: str, detail: str, started_at: str) -> None:
        con = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO macro_fetch_log (run_id, sources, status, detail, started_at, finished_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [run_id, ",".join(sources), status, detail, started_at, datetime.now().isoformat(timespec="seconds")],
            )
        finally:
            con.close()

    # ------------------------------------------------------------ 查询
    def list_indicators(self) -> list[dict[str, Any]]:
        con = self._connect(read_only=True)
        try:
            rows = con.execute(
                """
                SELECT s.*, COUNT(o.period) AS observations,
                       MIN(o.period) AS first_period, MAX(o.period) AS last_period
                FROM macro_series s LEFT JOIN macro_observations o USING (series_id)
                GROUP BY ALL ORDER BY s.source, s.category, s.series_id
                """
            ).fetchdf()
        finally:
            con.close()
        return rows.replace({pd.NA: None}).to_dict(orient="records")

    def query_series(self, series_id: str, start: Optional[str] = None, end: Optional[str] = None) -> list[dict[str, Any]]:
        query = "SELECT period, value, unit, source, provider, source_url, raw_label, fetched_at FROM macro_observations WHERE series_id = ?"
        params: list[Any] = [series_id]
        if start:
            query += " AND period >= CAST(? AS DATE)"
            params.append(start)
        if end:
            query += " AND period <= CAST(? AS DATE)"
            params.append(end)
        query += " ORDER BY period"
        con = self._connect(read_only=True)
        try:
            df = con.execute(query, params).fetchdf()
        finally:
            con.close()
        return [
            {**row, "period": _date_str(row["period"])}
            for row in df.to_dict(orient="records")
        ]

    def latest(self) -> list[dict[str, Any]]:
        con = self._connect(read_only=True)
        try:
            df = con.execute(
                """
                WITH ranked AS (
                    SELECT o.*, s.name_cn, s.frequency, s.category,
                           ROW_NUMBER() OVER (PARTITION BY o.series_id ORDER BY o.period DESC) AS rn
                    FROM macro_observations o JOIN macro_series s USING (series_id)
                )
                SELECT series_id, name_cn, source, frequency, category, period, value, unit,
                       provider, source_url, fetched_at
                FROM ranked WHERE rn = 1 ORDER BY source, category, series_id
                """
            ).fetchdf()
        finally:
            con.close()
        return [
            {**row, "period": _date_str(row["period"])}
            for row in df.to_dict(orient="records")
        ]

    def wide(self, frequency: str = "monthly") -> pd.DataFrame:
        con = self._connect(read_only=True)
        try:
            df = con.execute(
                """
                SELECT o.period, o.series_id, o.value
                FROM macro_observations o JOIN macro_series s USING (series_id)
                WHERE s.frequency = ?
                ORDER BY o.period
                """,
                [frequency],
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            return df
        pivot = df.pivot_table(index="period", columns="series_id", values="value", aggfunc="last").sort_index()
        pivot.index = pd.to_datetime(pivot.index).strftime("%Y-%m-%d")
        return pivot

    def status(self) -> dict[str, Any]:
        con = self._connect(read_only=True)
        try:
            counts = con.execute(
                "SELECT (SELECT COUNT(*) FROM macro_series), (SELECT COUNT(*) FROM macro_observations), "
                "(SELECT COUNT(*) FROM macro_raw_tables), (SELECT COUNT(*) FROM macro_fetch_log), "
                "(SELECT COUNT(*) FROM macro_region_observations)"
            ).fetchone()
            by_source = con.execute(
                "SELECT source, COUNT(*) AS series, MIN(period), MAX(period) FROM macro_observations GROUP BY source"
            ).fetchall()
            last_run = con.execute(
                "SELECT run_id, sources, status, detail, finished_at FROM macro_fetch_log ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
        finally:
            con.close()
        return {
            "db_path": str(self.path),
            "series": counts[0],
            "observations": counts[1],
            "raw_tables": counts[2],
            "runs": counts[3],
            "region_observations": counts[4],
            "by_source": [
                {"source": r[0], "series": r[1], "first_period": str(r[2]), "last_period": str(r[3])}
                for r in by_source
            ],
            "last_run": (
                {"run_id": last_run[0], "sources": last_run[1], "status": last_run[2],
                 "detail": last_run[3], "finished_at": last_run[4]}
                if last_run else None
            ),
        }



    def dashboard(self, years: int = 3, countries: Sequence[str] | None = None) -> dict[str, Any]:
        """三国宏观看板数据：最新值、变化、同比与近 N 年序列。"""
        years = max(1, min(10, int(years)))
        today = date.today()
        chart_start = today - __import__("datetime").timedelta(days=366 * years)
        query_start = today - __import__("datetime").timedelta(days=366 * (years + 2))
        con = self._connect(read_only=True)
        try:
            df = con.execute(
                """
                SELECT o.series_id, o.period, o.value, s.name_cn, s.source, s.country,
                       s.frequency, s.category, s.unit, o.provider, o.source_url
                FROM macro_observations o JOIN macro_series s USING (series_id)
                WHERE o.period >= CAST(? AS DATE)
                ORDER BY s.country, s.category, o.series_id, o.period
                """,
                [query_start.isoformat()],
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            return {"as_of": None, "years": years, "countries": []}

        wanted = {c.strip().lower() for c in (countries or []) if c.strip()}
        series_map: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            sid = row["series_id"]
            item = series_map.setdefault(sid, {
                "series_id": sid,
                "name_cn": row["name_cn"],
                "country": str(row["country"] or "").lower(),
                "source": row["source"],
                "frequency": row["frequency"],
                "category": row["category"],
                "unit": row["unit"],
                "provider": row["provider"],
                "source_url": row["source_url"],
                "observations": [],
            })
            period = pd.Timestamp(row["period"]).date()
            item["observations"].append([period.isoformat(), float(row["value"])])

        result = []
        as_of = None
        for item in series_map.values():
            if wanted and item["country"] not in wanted:
                continue
            obs = item["observations"]
            if len(obs) < 2:
                continue
            obs.sort(key=lambda x: x[0])
            item["observations"] = [o for o in obs if o[0] >= chart_start.isoformat()]
            if len(item["observations"]) < 2:
                continue
            latest_period, latest_value = obs[-1]
            prev_value = obs[-2][1]
            item["latest"] = latest_value
            item["latest_period"] = latest_period
            item["previous"] = prev_value
            is_percent = item["unit"] == "%"
            item["change"] = round(latest_value - prev_value, 6)
            item["change_unit"] = "pp" if is_percent else item["unit"]
            # 百分比型指标（利率/通胀/增速）不做“涨跌幅%”，避免 0.9% vs 0.7% 被算成 +28%
            item["change_pct"] = (
                None if is_percent
                else (round((latest_value / prev_value - 1) * 100, 4) if prev_value else None)
            )
            # 同比：百分比型取 pp 差；水平/指数型取百分比变化
            item["yoy"] = None
            item["yoy_unit"] = "pp" if is_percent else "%"
            if item["frequency"] in {"monthly", "quarterly", "annual"}:
                target = pd.Timestamp(latest_period) - pd.Timedelta(days=365)
                candidates = [o for o in obs[:-1] if abs((pd.Timestamp(o[0]) - target).days) <= 75]
                if candidates:
                    base = candidates[-1][1]
                    if base is not None:
                        if is_percent:
                            item["yoy"] = round(latest_value - base, 4)
                        elif base:
                            item["yoy"] = round((latest_value / base - 1) * 100, 4)
            as_of = max(as_of, latest_period) if as_of else latest_period
            result.append(item)

        order = {"cn": 0, "us": 1, "jp": 2}
        category_order = {"增长": 0, "通胀": 1, "价格": 1, "就业": 2, "货币": 3,
                          "信贷": 4, "社融": 5, "利率": 6, "消费": 7, "景气": 8, "贸易": 9}
        result.sort(key=lambda x: (
            order.get(x["country"], 9),
            category_order.get(x["category"], 99),
            x["series_id"],
        ))
        grouped: dict[str, list] = {}
        for item in result:
            grouped.setdefault(item["country"], []).append(item)
        result = {
            "as_of": as_of,
            "years": years,
            "countries": [
                {"code": code, "name": {"cn": "中国", "us": "美国", "jp": "日本"}.get(code, code.upper()),
                 "series": items}
                for code, items in sorted(grouped.items(), key=lambda kv: order.get(kv[0], 9))
            ],
        }
        try:
            from .analysis import build_analysis
            result["analysis"] = build_analysis(result)
        except Exception:
            result["analysis"] = {}
        return result


    # ------------------------------------------------------------ 分省数据
    def upsert_region_observations(self, observations: Sequence[RegionObservation]) -> int:
        if not observations:
            return 0
        deduped: dict[tuple[str, str, date], RegionObservation] = {}
        for obs in observations:
            deduped[(obs.indicator_id, obs.region_code, obs.period)] = obs
        rows = [o.to_dict() for o in deduped.values()]
        con = self._connect()
        try:
            con.register("_region_rows", pd.DataFrame(rows))
            con.execute(
                "INSERT OR REPLACE INTO macro_region_observations "
                "(indicator_id, indicator_name, region_code, region_name, period, value, unit, "
                " provider, source_url, fetched_at) "
                "SELECT indicator_id, indicator_name, region_code, region_name, period, value, unit, "
                "provider, source_url, fetched_at FROM _region_rows"
            )
            con.unregister("_region_rows")
        finally:
            con.close()
        return len(rows)

    def region_indicators(self) -> list[dict[str, Any]]:
        con = self._connect(read_only=True)
        try:
            df = con.execute(
                """
                SELECT indicator_id, any_value(indicator_name) AS indicator_name,
                       any_value(unit) AS unit, COUNT(DISTINCT region_code) AS regions,
                       COUNT(*) AS observations, MIN(period) AS first_period, MAX(period) AS last_period
                FROM macro_region_observations
                GROUP BY indicator_id ORDER BY indicator_id
                """
            ).fetchdf()
        finally:
            con.close()
        return [
            {**row, "first_period": _date_str(row["first_period"]), "last_period": _date_str(row["last_period"])}
            for row in df.to_dict(orient="records")
        ]

    def region_table(self, indicator_id: str, history_years: int = 5) -> dict[str, Any]:
        con = self._connect(read_only=True)
        try:
            df = con.execute(
                "SELECT indicator_name, region_code, region_name, period, value, unit "
                "FROM macro_region_observations WHERE indicator_id = ? ORDER BY region_name, period",
                [indicator_id],
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            return {"indicator_id": indicator_id, "as_of": None, "rows": []}

        by_region: dict[str, dict] = {}
        for _, row in df.iterrows():
            key = row["region_code"]
            item = by_region.setdefault(key, {
                "region_code": key, "region_name": row["region_name"],
                "indicator_name": row["indicator_name"], "unit": row["unit"],
                "observations": [],
            })
            item["observations"].append([_date_str(row["period"]), float(row["value"])])

        rows = []
        for item in by_region.values():
            obs = sorted(item.pop("observations"), key=lambda x: x[0])
            if not obs:
                continue
            latest_period, latest = obs[-1]
            previous = obs[-2][1] if len(obs) >= 2 else None
            is_rate = item["unit"] == "%"
            item.update({
                "latest_period": latest_period,
                "latest": latest,
                "previous_period": obs[-2][0] if len(obs) >= 2 else None,
                "previous": previous,
                "change": round(latest - previous, 4) if previous is not None else None,
                "change_unit": "pp" if is_rate else item["unit"],
                "change_pct": (None if is_rate or not previous else round((latest / previous - 1) * 100, 2)),
                "history": obs[-history_years:],
            })
            rows.append(item)

        # 最新排名与上期排名变化
        ranked = sorted(rows, key=lambda x: x["latest"], reverse=True)
        for i, item in enumerate(ranked, 1):
            item["rank"] = i
        prev_rows = [r for r in rows if r.get("previous") is not None]
        prev_ranked = sorted(prev_rows, key=lambda x: x["previous"], reverse=True)
        prev_rank = {r["region_code"]: i for i, r in enumerate(prev_ranked, 1)}
        for item in rows:
            p = prev_rank.get(item["region_code"])
            item["previous_rank"] = p
            item["rank_change"] = (p - item["rank"]) if p else None
        rows.sort(key=lambda x: x["rank"])
        return {
            "indicator_id": indicator_id,
            "indicator_name": rows[0]["indicator_name"] if rows else indicator_id,
            "unit": rows[0]["unit"] if rows else "",
            "as_of": max(r["latest_period"] for r in rows),
            "count": len(rows),
            "rows": rows,
        }

    # ------------------------------------------------------------ 导出
    def export_all(self, out_dir: Optional[str | Path] = None) -> dict[str, str]:
        out = Path(out_dir) if out_dir else DEFAULT_EXPORT_DIR
        out.mkdir(parents=True, exist_ok=True)
        con = self._connect(read_only=True)
        try:
            long_df = con.execute(
                """
                SELECT o.series_id, s.name_cn, s.source, s.frequency, s.category, o.period,
                       o.value, o.unit, o.provider, o.source_url, o.raw_label, o.fetched_at
                FROM macro_observations o JOIN macro_series s USING (series_id)
                ORDER BY s.source, o.series_id, o.period
                """
            ).fetchdf()
        finally:
            con.close()

        paths: dict[str, str] = {}
        long_path = out / "macro_long.csv"
        long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
        paths["long"] = str(long_path)

        for freq in ("monthly", "quarterly"):
            wide = self.wide(freq)
            if not wide.empty:
                path = out / f"macro_wide_{freq}.csv"
                wide.to_csv(path, encoding="utf-8-sig")
                paths[f"wide_{freq}"] = str(path)

        con = self._connect(read_only=True)
        try:
            prov_long = con.execute(
                "SELECT * FROM macro_region_observations ORDER BY indicator_id, region_name, period"
            ).fetchdf()
        finally:
            con.close()
        if not prov_long.empty:
            prov_long_path = out / "macro_provinces_long.csv"
            prov_long.to_csv(prov_long_path, index=False, encoding="utf-8-sig")
            paths["provinces_long"] = str(prov_long_path)
            latest_rows = []
            for iid in prov_long["indicator_id"].unique():
                table = self.region_table(str(iid))
                for row in table.get("rows", []):
                    latest_rows.append({
                        "indicator_id": iid,
                        "indicator_name": table.get("indicator_name"),
                        "region_name": row["region_name"],
                        "latest_period": row["latest_period"],
                        "latest": row["latest"],
                        "previous": row["previous"],
                        "change": row["change"],
                        "change_pct": row["change_pct"],
                        "rank": row["rank"],
                        "rank_change": row["rank_change"],
                        "unit": table.get("unit"),
                    })
            prov_latest_path = out / "macro_provinces_latest.csv"
            pd.DataFrame(latest_rows).to_csv(prov_latest_path, index=False, encoding="utf-8-sig")
            paths["provinces_latest"] = str(prov_latest_path)

        latest_path = out / "macro_latest.csv"
        pd.DataFrame(self.latest()).to_csv(latest_path, index=False, encoding="utf-8-sig")
        paths["latest"] = str(latest_path)

        try:
            import openpyxl  # noqa: F401
            excel_path = out / "macro_data.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                long_df.to_excel(writer, sheet_name="long", index=False)
                for freq in ("monthly", "quarterly"):
                    wide = self.wide(freq)
                    if not wide.empty:
                        wide.to_excel(writer, sheet_name=f"wide_{freq}")
                pd.DataFrame(self.latest()).to_excel(writer, sheet_name="latest", index=False)
            paths["excel"] = str(excel_path)
        except ImportError:
            pass
        return paths
