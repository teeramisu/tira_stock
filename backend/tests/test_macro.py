"""宏观数据解析、存储与 API 测试。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macro.models import SERIES, MacroObservation, RawTable  # noqa: E402
from macro.provinces import RegionObservation  # noqa: E402
from macro.parsers import (  # noqa: E402
    normalize_nbs_fai,
    normalize_nbs_pmi,
    normalize_pboc_shrzgm,
    parse_period,
    parse_ytd_period,
    to_float,
)
from macro.analysis import build_analysis  # noqa: E402
from macro.derive import derive_china_observations  # noqa: E402
from macro.storage import MacroStore  # noqa: E402


class ParserTest(unittest.TestCase):
    def test_parse_period(self):
        self.assertEqual(parse_period("2026年09月份"), date(2026, 9, 1))
        self.assertEqual(parse_period("2026年9月"), date(2026, 9, 1))
        self.assertEqual(parse_period("2026年第1-3季度"), date(2026, 9, 30))
        self.assertEqual(parse_period("2026年第3季度"), date(2026, 9, 30))
        self.assertEqual(parse_period("202609"), date(2026, 9, 1))
        self.assertEqual(parse_period("2026.09"), date(2026, 9, 1))
        self.assertEqual(parse_period("2026-09-15"), date(2026, 9, 15))
        self.assertIsNone(parse_period("--"))

    def test_parse_ytd_period(self):
        self.assertEqual(parse_ytd_period("2026年1—9月份全国固定资产投资基本情况"), date(2026, 9, 30))
        self.assertEqual(parse_ytd_period("2026年10月份"), date(2026, 10, 31))

    def test_to_float(self):
        self.assertAlmostEqual(to_float("1,234.5"), 1234.5)
        self.assertAlmostEqual(to_float("7.2%"), 7.2)
        self.assertAlmostEqual(to_float("(1.2)"), -1.2)
        self.assertIsNone(to_float("--"))
        self.assertIsNone(to_float(None))

    def test_nbs_pmi(self):
        table = pd.DataFrame({
            0: ["单位：%", None, "2026年10月"],
            1: ["单位：%", "PMI", 49.3],
            2: ["单位：%", "生产", 49.8],
        })
        obs, raws = normalize_nbs_pmi([table], url="http://nbs")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].series_id, "nbs_pmi_manufacturing")
        self.assertEqual(obs[0].period, date(2026, 10, 1))
        self.assertAlmostEqual(obs[0].value, 49.3)
        self.assertEqual(len(raws), 1)

    def test_nbs_fai(self):
        table = pd.DataFrame({0: ["指标", "固定资产投资（不含农户）"], 1: ["同比增长（%）", -7.2]})
        obs, raws = normalize_nbs_fai(
            [table], url="http://nbs", title="2026年1—9月份全国固定资产投资基本情况"
        )
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].series_id, "nbs_fai_yoy")
        self.assertEqual(obs[0].period, date(2026, 9, 30))
        self.assertAlmostEqual(obs[0].value, -7.2)
        self.assertEqual(len(raws), 1)

    def test_pboc_shrzgm(self):
        df = pd.DataFrame([
            ["社会融资规模增量统计表"] + [None] * 11,
            ["项目"] + [None] * 11,
            ["月份", "社会融资规模增量", "人民币贷款", None, None, None, None, "企业债券", "政府债券", None, None, None],
            ["2026.01", 72185, 49016, 468, -192, -4, 6293, 5033, 9764, 291, -99, 355],
            ["2026.02", 23837, 8458, -36, -181, 310, -1755, 1522, 14014, 454, -56, 379],
        ])
        obs, raw = normalize_pboc_shrzgm(df, url="http://pboc")
        self.assertEqual(len(obs), 8)  # 2 months × 4 series
        self.assertEqual(obs[0].period, date(2026, 1, 1))
        self.assertEqual(obs[0].series_id, "pbc_shrzgm_increment")
        self.assertAlmostEqual(obs[0].value, 72185)
        self.assertIsNotNone(raw)
        self.assertEqual(raw.source, "PBOC")


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MacroStore(Path(self.tmp.name) / "macro.duckdb")
        self.store.upsert_series(SERIES.values())

    def tearDown(self):
        self.tmp.cleanup()

    def test_upsert_query_latest(self):
        self.store.upsert_observations([
            MacroObservation("nbs_cpi_yoy", date(2026, 9, 1), 0.9, "%", "NBS", "eastmoney"),
            MacroObservation("nbs_cpi_yoy", date(2026, 9, 1), 1.0, "%", "NBS", "eastmoney"),  # last wins
        ])
        items = self.store.query_series("nbs_cpi_yoy")
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]["value"], 1.0)
        latest = self.store.latest()
        self.assertTrue(any(x["series_id"] == "nbs_cpi_yoy" for x in latest))

    def test_raw_table_and_status(self):
        raw = RawTable("NBS", "PMI", "http://x", 0, ["a"], [["1"]])
        self.store.record_raw_tables([raw])
        status = self.store.status()
        self.assertEqual(status["raw_tables"], 1)
        self.assertEqual(status["series"], len(SERIES))

    def test_export(self):
        self.store.upsert_observations([
            MacroObservation("nbs_cpi_yoy", date(2026, 9, 1), 0.9, "%", "NBS", "eastmoney"),
            MacroObservation("pbc_m2_yoy", date(2026, 9, 1), 7.2, "%", "PBOC", "eastmoney"),
        ])
        paths = self.store.export_all(self.tmp.name)
        self.assertIn("long", paths)
        self.assertTrue(Path(paths["long"]).exists())
        self.assertIn("wide_monthly", paths)
        wide = pd.read_csv(paths["wide_monthly"])
        self.assertIn("nbs_cpi_yoy", wide.columns)


class MacroApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import app.routers.macro as macro_router

        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = MacroStore(Path(cls.tmp.name) / "api.duckdb")
        cls.store.upsert_series(SERIES.values())
        cls.store.upsert_observations([
            MacroObservation("nbs_cpi_yoy", date(2026, 9, 1), 0.9, "%", "NBS", "eastmoney"),
            MacroObservation("nbs_cpi_yoy", date(2026, 10, 1), 1.1, "%", "NBS", "eastmoney"),
            MacroObservation("pbc_m2_yoy", date(2026, 9, 1), 7.2, "%", "PBOC", "eastmoney"),
        ])
        cls.store.upsert_region_observations([
            RegionObservation("gdp_total", "GDP 总量", "110000000000", "北京市",
                             date(2024, 12, 31), 49670.2, "亿元"),
            RegionObservation("gdp_total", "GDP 总量", "110000000000", "北京市",
                             date(2025, 12, 31), 52073.4, "亿元"),
            RegionObservation("gdp_total", "GDP 总量", "440000000000", "广东省",
                             date(2024, 12, 31), 141488.9, "亿元"),
            RegionObservation("gdp_total", "GDP 总量", "440000000000", "广东省",
                             date(2025, 12, 31), 145846.8, "亿元"),
        ])
        macro_router._store = cls.store
        from app.main import app

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_status(self):
        res = self.client.get("/api/macro/status")
        self.assertEqual(res.status_code, 200)
        self.assertGreaterEqual(res.json()["observations"], 2)

    def test_indicators(self):
        res = self.client.get("/api/macro/indicators")
        self.assertEqual(res.status_code, 200)
        self.assertGreater(res.json()["items"].__len__(), 0)

    def test_series(self):
        res = self.client.get("/api/macro/series/nbs_cpi_yoy?start=2026-01-01")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)

    def test_region_table(self):
        table = self.store.region_table("gdp_total")
        self.assertEqual(table["count"], 2)
        self.assertEqual(table["rows"][0]["region_name"], "广东省")
        self.assertGreater(table["rows"][0]["change"], 0)
        self.assertEqual(table["rows"][0]["rank"], 1)

    def test_region_api(self):
        res = self.client.get("/api/macro/regions/gdp_total")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)
        res = self.client.get("/api/macro/regions/indicators")
        self.assertEqual(res.status_code, 200)
        self.assertGreaterEqual(len(res.json()["items"]), 1)

    def test_dashboard(self):
        res = self.client.get("/api/macro/dashboard?years=3")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("countries", body)
        cn = next(c for c in body["countries"] if c["code"] == "cn")
        self.assertGreaterEqual(len(cn["series"]), 1)
        self.assertIn("change_unit", cn["series"][0])
        self.assertIn("yoy_unit", cn["series"][0])

    def test_wide(self):
        res = self.client.get("/api/macro/wide?frequency=monthly")
        self.assertEqual(res.status_code, 200)
        self.assertIn("nbs_cpi_yoy", res.json()["columns"])
        # 2026-10 只有 CPI，M2 为 null，验证 NaN 已被序列化为 null
        oct_row = next(r for r in res.json()["rows"] if r["period"] == "2026-10-01")
        self.assertIsNone(oct_row["pbc_m2_yoy"])


class DeriveAnalysisTest(unittest.TestCase):
    def test_derive_m1_m2_and_real_rate(self):
        obs = [
            MacroObservation("pbc_m1_yoy", date(2026, 9, 1), 3.2, "%", "PBOC"),
            MacroObservation("pbc_m2_yoy", date(2026, 9, 1), 7.2, "%", "PBOC"),
            MacroObservation("nbs_cpi_yoy", date(2026, 9, 1), 0.9, "%", "NBS"),
            MacroObservation("pbc_lpr_1y", date(2026, 9, 20), 3.0, "%", "PBOC"),
        ]
        derived = derive_china_observations(obs)
        by_id = {x.series_id: x for x in derived}
        self.assertAlmostEqual(by_id["cn_m1_m2_scissors"].value, -4.0)
        self.assertAlmostEqual(by_id["cn_real_rate"].value, 2.1)

    def test_china_analysis(self):
        def mk(sid, value, unit="%", period="2026-09-01"):
            return {
                "series_id": sid, "name_cn": sid, "country": "cn", "source": "CN",
                "frequency": "monthly", "category": "增长", "unit": unit,
                "latest": value, "latest_period": period, "previous": value,
                "change": 0, "change_unit": "pp", "yoy": 0, "yoy_unit": "pp",
                "observations": [["2026-08-01", value], [period, value]],
            }
        series = [
            mk("nbs_pmi_manufacturing", 49.3, "指数"), mk("nbs_industrial_production_yoy", 5.5),
            mk("cn_electricity_yoy", 3.9), mk("cn_fai_month_yoy", -9.7),
            mk("nbs_cpi_yoy", 0.9), mk("nbs_ppi_yoy", 5.0),
            mk("pbc_m1_yoy", 3.2), mk("pbc_m2_yoy", 7.2), mk("cn_m1_m2_scissors", -4.0, "pp"),
            mk("cn_shibor_3m", 1.405), mk("pbc_lpr_1y", 3.0), mk("cn_real_rate", 2.1),
            mk("cn_cb_total_assets", 505003, "亿元"),
            mk("cn_new_house_price_yoy", 0.05), mk("cn_second_hand_price_yoy", -0.85),
            mk("cn_a_pe_median", 32.6, "倍"), mk("cn_a_pe_10y_pct", 36.4),
            mk("cn_a_pb_median", 2.39, "倍"), mk("cn_a_pb_10y_pct", 28.9),
            mk("cn_margin_balance", 23356, "亿元", "2026-10-30"),
        ]
        payload = {"as_of": "2026-10-30", "countries": [{"code": "cn", "name": "中国", "series": series}]}
        analysis = build_analysis(payload)
        self.assertIn("cn", analysis)
        self.assertEqual(len(analysis["cn"]["signals"]), 6)
        self.assertIn("growth", {x["key"] for x in analysis["cn"]["signals"]})
        self.assertTrue(analysis["cn"]["outlook"]["stance"])
        self.assertTrue(analysis["cn"]["commentary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
