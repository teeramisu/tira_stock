"""宏观六维评分的 spec 派生测试。

验收标准见 docs/spec-template.md（SC-1 ~ SC-8）。
断言的是「业务口径」，不是当前实现返回的具体数字 ——
系数调整时这里应该仍然成立，只有口径变化才需要同步修改。
"""
from __future__ import annotations

import unittest

from macro.analysis import build_analysis


def series(sid: str, latest, period: str = "2026-09-01", observations=None) -> dict:
    """构造一条宏观序列（analysis 只需要 series_id / latest / latest_period）。"""
    return {
        "series_id": sid,
        "latest": latest,
        "latest_period": period,
        "observations": observations or [],
    }


def payload(*items: dict) -> dict:
    return {"countries": [{"code": "cn", "name": "中国", "series": list(items)}]}


STRONG = (
    series("nbs_pmi_manufacturing", 52.0),
    series("nbs_industrial_production_yoy", 6.0),
    series("cn_electricity_yoy", 4.0),
    series("cn_fai_month_yoy", 3.0),
    series("nbs_cpi_yoy", 1.2),
    series("nbs_ppi_yoy", 0.5),
    series("pbc_m2_yoy", 8.5),
    series("cn_m1_m2_scissors", 1.0),
    series("cn_shibor_3m", 1.5),
    series("pbc_lpr_1y", 3.0),
    series("cn_new_house_price_yoy", 0.5),
    series("cn_second_hand_price_yoy", 0.2),
    series("cn_a_pe_10y_pct", 30.0),
    series("cn_a_pb_10y_pct", 30.0),
    series("cn_a_pe_median", 12.0),
    series("cn_a_pb_median", 1.2),
)

WEAK = (
    series("nbs_pmi_manufacturing", 47.0),
    series("nbs_industrial_production_yoy", 2.0),
    series("cn_electricity_yoy", 1.0),
    series("cn_fai_month_yoy", -8.0),
    series("nbs_cpi_yoy", 0.2),
    series("nbs_ppi_yoy", -2.0),
    series("cn_new_house_price_yoy", -5.0),
    series("cn_second_hand_price_yoy", -4.0),
    series("cn_a_pe_10y_pct", 80.0),
    series("cn_a_pb_10y_pct", 80.0),
)


class MacroSpecTest(unittest.TestCase):
    """六维评分的验收标准。"""

    def test_scores_are_clamped_to_0_100(self):
        """SC-1：极端输入下六维分数都必须落在 0–100。"""
        extremes = (
            series("nbs_pmi_manufacturing", 999.0),
            series("nbs_industrial_production_yoy", 999.0),
            series("cn_electricity_yoy", 999.0),
            series("cn_fai_month_yoy", 999.0),
            series("nbs_cpi_yoy", -999.0),
            series("nbs_ppi_yoy", 999.0),
            series("cn_m1_m2_scissors", -999.0),
            series("cn_shibor_3m", 999.0),
            series("cn_new_house_price_yoy", -999.0),
            series("cn_second_hand_price_yoy", -999.0),
            series("cn_a_pe_10y_pct", -999.0),
            series("cn_a_pb_10y_pct", -999.0),
        )
        signals = build_analysis(payload(*extremes))["cn"]["signals"]
        self.assertEqual(len(signals), 6)
        for signal in signals:
            self.assertGreaterEqual(signal["score"], 0.0, signal["key"])
            self.assertLessEqual(signal["score"], 100.0, signal["key"])

    def test_tone_thresholds(self):
        """SC-2：tone 阈值固定为 ≥60 偏正、≤40 偏负、其余中性。"""
        for name, items in (("strong", STRONG), ("weak", WEAK)):
            analysis = build_analysis(payload(*items))["cn"]
            self.assertEqual(len(analysis["signals"]), 6, name)
            for signal in analysis["signals"]:
                score = signal["score"]
                expected = "positive" if score >= 60 else ("negative" if score <= 40 else "neutral")
                self.assertEqual(signal["tone"], expected, f"{name}/{signal['key']} score={score}")

    def test_missing_data_degrades_gracefully(self):
        """SC-3：只有部分数据时，缺数据的维度标注「数据不足」而不是报错。"""
        analysis = build_analysis(payload(series("nbs_cpi_yoy", 1.0)))["cn"]
        self.assertEqual(len(analysis["signals"]), 6)
        labels = {s["key"]: s["label"] for s in analysis["signals"]}
        # 有数据的维度正常出结论，缺数据的维度明确标注
        self.assertEqual(labels["inflation"], "温和通胀")
        for key in ("property", "valuation", "sentiment"):
            self.assertEqual(labels[key], "数据不足", key)
        # 缺失值在文案里显示为占位符，而不是 None / 异常
        self.assertIn("—", analysis["outlook"]["summary"])
        self.assertTrue(analysis["disclaimer"])

    def test_strong_conditions_give_positive_stance(self):
        """SC-4：增长强 + 流动性宽 + 估值低 → 「震荡偏强」。"""
        analysis = build_analysis(payload(*STRONG))["cn"]
        scores = {s["key"]: s["score"] for s in analysis["signals"]}
        self.assertGreaterEqual(scores["growth"], 55, scores)
        self.assertGreaterEqual(scores["liquidity"], 60, scores)
        self.assertGreaterEqual(scores["valuation"], 55, scores)
        self.assertEqual(analysis["outlook"]["stance"], "震荡偏强")
        self.assertEqual(analysis["outlook"]["tone"], "positive")

    def test_weak_growth_and_property_give_negative_stance(self):
        """SC-5：增长 <40 且地产 <40 → 「震荡偏弱」。"""
        analysis = build_analysis(payload(*WEAK))["cn"]
        scores = {s["key"]: s["score"] for s in analysis["signals"]}
        self.assertLess(scores["growth"], 40, scores)
        self.assertLess(scores["property"], 40, scores)
        self.assertEqual(analysis["outlook"]["stance"], "震荡偏弱")
        self.assertEqual(analysis["outlook"]["tone"], "negative")

    def test_outlook_lists_are_capped(self):
        """SC-6：catalysts / risks / watch 各不超过 4 条（前端按 4 条排版）。"""
        analysis = build_analysis(payload(*STRONG))["cn"]
        for key in ("catalysts", "risks", "watch"):
            self.assertLessEqual(len(analysis["outlook"][key]), 4, key)
        # watch 一定有兜底项；catalysts / risks 允许为空（没有明确触发条件时不硬凑）
        self.assertTrue(analysis["outlook"]["watch"])

    def test_disclaimer_present(self):
        """SC-7：必须带免责声明。"""
        for items in (STRONG, WEAK, (series("nbs_cpi_yoy", 1.0),)):
            analysis = build_analysis(payload(*items))["cn"]
            self.assertIn("不构成投资建议", analysis["disclaimer"])

    def test_as_of_is_latest_period(self):
        """SC-8：as_of 取所有序列里最新的日期。"""
        items = (
            series("nbs_cpi_yoy", 1.0, period="2026-08-01"),
            series("nbs_ppi_yoy", 0.5, period="2026-09-30"),
            series("cn_new_house_price_yoy", -1.0, period="2026-07-01"),
        )
        self.assertEqual(build_analysis(payload(*items))["cn"]["as_of"], "2026-09-30")

    def test_us_and_jp_sections_use_prefix(self):
        """SC-9：按 series_id 前缀决定生成哪几个国家，互不干扰。"""
        data = {
            "countries": [
                {
                    "code": "us",
                    "name": "美国",
                    "series": [series("us_cpi_yoy", 3.2), series("us_ism_pmi", 51.0)],
                },
                {"code": "jp", "name": "日本", "series": [series("jp_cpi_yoy", 2.8)]},
            ]
        }
        analysis = build_analysis(data)
        self.assertEqual(set(analysis), {"us", "jp"})
        self.assertEqual(len(analysis["us"]["signals"]), 3)
        self.assertEqual(len(analysis["jp"]["signals"]), 3)
        self.assertTrue(analysis["us"]["outlook"]["stance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
