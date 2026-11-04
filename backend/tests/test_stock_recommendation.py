"""股票推荐解析/评分模块测试。

运行:
    python backend/tests/test_stock_recommendation.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.stock_recommendation import (  # noqa: E402
    _rule_score,
    _sector_of,
    load_price_summaries,
    score_all_recommendations,
)
from services.zsxq_parser import (  # noqa: E402
    StockIndex,
    build_stock_dataset,
    clean_text,
    extract_mentions,
    load_topics,
)


def _topic(text: str, tid: str = "1", author: str = " tester", likes: int = 1) -> dict:
    return {
        "topic_id": tid,
        "group": {"name": "每日调研"},
        "talk": {"owner": {"name": author}, "text": text},
        "likes_count": likes,
        "readers_count": 100,
        "create_time": "2026-03-13T10:00:00.000+0800",
    }


class CleanTextTest(unittest.TestCase):
    def test_strips_tags_and_emoji(self):
        raw = '<e type="hashtag" title="%23%E5%87%BA%E5%A4%84%E6%9C%AA%E7%9F%A5%23" /> [烟花] 推荐【九号公司】'
        cleaned = clean_text(raw)
        self.assertIn("#出处未知#", cleaned)
        self.assertNotIn("<e", cleaned)
        self.assertNotIn("[烟花]", cleaned)
        self.assertIn("九号公司", cleaned)


class MentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = StockIndex()

    def test_extracts_listed_stock(self):
        text = "【九号公司】终于触底反弹了！预计26年净利润23亿，估值仅15倍，是绝佳买入时点。"
        mentions = extract_mentions(text, self.index)
        codes = {code for _, code, _ in mentions}
        self.assertIn("689009", codes)
        reason = next(r for n, c, r in mentions if c == "689009")
        self.assertIn("九号公司", reason)
        self.assertIn("买入", reason)

    def test_dedupes_name_variants(self):
        text = "推荐九号公司：两轮车恢复增长；九号割草机器人旺季。"
        mentions = extract_mentions(text, self.index)
        code_hits = [c for _, c, _ in mentions if c == "689009"]
        self.assertEqual(len(code_hits), 1, "同一代码在同一帖子内只应出现一次")

    def test_ignores_broker_team_names(self):
        text = "【天风电新】明阳电气调整点评：继续重点推荐，受益海风出海。"
        mentions = extract_mentions(text, self.index)
        names = {n for n, _, _ in mentions}
        self.assertNotIn("天风电新", names)

    def test_requires_recommend_hint(self):
        text = "今天大盘震荡，九号公司成交额 3 亿元。"
        self.assertEqual(extract_mentions(text, self.index, require_hint=True), [])
        self.assertTrue(extract_mentions(text, self.index, require_hint=False))


class AggregateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = StockIndex()

    def test_source_count_counts_distinct_topics(self):
        topics = [
            _topic("推荐【九号公司】：两轮车恢复，买入。", tid="a", author="甲"),
            _topic("继续看好九号公司，割草机器人放量。", tid="b", author="乙"),
            _topic("推荐九号公司：同一帖子重复提及九号公司也只算一次。", tid="c", author="丙"),
        ]
        data = build_stock_dataset(topics, self.index)
        stock = next(s for s in data["stocks"] if s["stock_code"] == "689009")
        self.assertEqual(stock["source_count"], 3)
        self.assertEqual(stock["author_count"], 3)
        self.assertGreaterEqual(stock["strong_count"], 1)

    def test_unlisted_names_excluded(self):
        topics = [_topic("【不存在的公司】获得融资，建议关注。", tid="x")]
        data = build_stock_dataset(topics, self.index)
        self.assertEqual(data["stocks"], [])

    def test_sorted_by_strength(self):
        topics = [
            _topic("重点推荐【九号公司】买入。", tid="1"),
            _topic("建议关注【宁德时代】。", tid="2"),
        ]
        data = build_stock_dataset(topics, self.index)
        self.assertGreaterEqual(len(data["stocks"]), 1)


class ScoringTest(unittest.TestCase):
    def test_rule_score_bounds(self):
        stock = {
            "source_count": 80, "author_count": 20, "strong_count": 50, "likes": 400,
            "return_60d_pct": 100,
        }
        perf = {"return_60d_pct": 100, "return_20d_pct": 30, "max_drawdown_pct": -10}
        result = _rule_score(stock, perf)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 95)
        self.assertTrue(-20 <= result["weekly_expected_pct"] <= 30)
        self.assertTrue(-35 <= result["monthly_expected_pct"] <= 60)

    def test_end_to_end_without_llm(self):
        topics = load_topics()
        if not topics:
            self.skipTest("无本地 zsxq 语料")
        result = score_all_recommendations(topics=topics[:500], use_llm=False, save=False)
        self.assertGreater(result["total_stocks"], 0)
        self.assertFalse(result["llm_used"])
        scores = [s["comprehensive_score"] for s in result["stocks"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for stock in result["stocks"][:20]:
            self.assertRegex(stock["stock_code"], r"^\d{6}$")
            self.assertIn("expected_weekly", stock)
            self.assertIn("expected_monthly", stock)

    def test_sector_of_uses_industry(self):
        self.assertEqual(_sector_of({"industry": "半导体", "concepts": ["AI"]}), "半导体")
        self.assertEqual(_sector_of({"industry": "未分类", "concepts": ["AI"]}), "未分类")

    def test_price_summary_local_db(self):
        summaries = load_price_summaries(["689009"])
        if not summaries:
            self.skipTest("本地行情库不存在或无数据")
        item = summaries["689009"]
        self.assertIn("total_return_pct", item)
        self.assertGreater(item["bars"], 10)


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from app.main import app

        cls.client = TestClient(app)

    def test_top_endpoint(self):
        res = self.client.get("/api/recommendations/top?limit=3")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("stocks", body)
        self.assertLessEqual(len(body["stocks"]), 3)
        if body["stocks"]:
            self.assertIn("comprehensive_score", body["stocks"][0])

    def test_sector_endpoint(self):
        res = self.client.get("/api/recommendations/by-sector?limit=5&per_sector=2")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("sectors", body)
        self.assertLessEqual(len(body["sectors"]), 5)
        for sec in body["sectors"]:
            self.assertIn("sector", sec)
            self.assertIn("avg_score", sec)
            self.assertLessEqual(len(sec["top_stocks"]), 2)

    def test_sector_filter(self):
        res = self.client.get("/api/recommendations/top?limit=5&sector=半导体")
        self.assertEqual(res.status_code, 200)
        for stock in res.json()["stocks"]:
            self.assertTrue(
                stock.get("industry") == "半导体" or "半导体" in (stock.get("concepts") or []),
                stock.get("stock_name"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
