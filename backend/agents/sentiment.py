"""Sentiment Agent — analyzes market news for emotional tone."""

from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent
from memory.models import SignalRecord
from memory.session import SessionMemory


class SentimentAgent(BaseAgent):
    """Analyze market news sentiment using keyword matching + optional LLM.

    Rule-based fallback:
      - Positive keywords (涨/增/利好/突破) → +score
      - Negative keywords (跌/减/利空/下挫) → -score
      - Score normalized to 0-100, with 50 as neutral.
    """

    POSITIVE = [r"涨", r"增", r"利好", r"突破", r"反弹", r"新高", r"涨停", r"回升", r"上涨",
                r"飙升", r"大涨", r"走强", r"回暖", r"买入", r"净利润增长", r"业绩超预期"]
    NEGATIVE = [r"跌", r"减", r"利空", r"下挫", r"大跌", r"暴跌", r"跌停", r"回落",
                r"下跌", r"走弱", r"卖出", r"亏损", r"业绩不及预期", r"退市"]

    def __init__(self, **kwargs: Any):
        super().__init__(name="sentiment", description="Market news sentiment analyzer", **kwargs)

    def analyze(self, ticker: str | None = None, session: SessionMemory | None = None, **_: Any) -> SignalRecord:
        self._trace("start", f"SentimentAgent: ticker={ticker}")

        news_items = self._fetch_news()
        score, pos_hits, neg_hits = self._keyword_score(news_items)
        confidence = self._calc_confidence(pos_hits, neg_hits)

        # 构建详细计算步骤
        calc_steps = []
        calc_steps.append(f"① 新闻数据: 获取 {len(news_items)} 条新闻")
        total = pos_hits + neg_hits
        if total == 0:
            calc_steps.append("② 关键词匹配: 无正面/负面词命中 → 默认 score=50")
            calc_steps.append(f"③ 置信度: total=0 → 默认 confidence=0.3")
        else:
            calc_steps.append(f"② 关键词匹配: 正面词 {pos_hits} 次, 负面词 {neg_hits} 次, 总计 {total} 次")
            raw_formula = f"50 + ({pos_hits}-{neg_hits})/{total}×40"
            calc_steps.append(f"③ 分数计算: {raw_formula} = {score}")
            ratio = abs(pos_hits - neg_hits) / max(total, 1)
            count_factor = min(total / 10, 1.0)
            calc_steps.append(f"④ 置信度: ratio={ratio:.2f}, count_factor={count_factor:.2f}, "
                              f"conf=0.4+0.6×{ratio:.2f}×{count_factor:.2f}={confidence:.2f}")

        reasoning = self._build_reasoning(ticker, score, pos_hits, neg_hits, news_items)

        if self.llm is not None and news_items:
            llm_reason = self._llm_reason(
                f"You are a sentiment analyst. Based on {len(news_items)} news headlines about China A-share market, "
                f"rate sentiment 0-100 (50=neutral, 75=bullish, 25=bearish). Headlines: "
                + " | ".join([n.get("title", "")[:60] for n in news_items[:10]])
                + f"\nCurrent keyword score: {score}/100.\nProvide a brief sentiment assessment in Chinese:",
            )
            if llm_reason:
                reasoning = llm_reason

        self._trace("complete", f"SentimentAgent: score={score}, confidence={confidence:.2f}")
        return SignalRecord(
            agent_name=self.name,
            signal_type="score",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            ticker=ticker,
            metadata={"pos_hits": pos_hits, "neg_hits": neg_hits, "news_count": len(news_items), "calc_steps": calc_steps},
        )

    def _fetch_news(self) -> list[dict[str, Any]]:
        news_tool = self.tools.get("news")
        if news_tool is None:
            return []
        try:
            return news_tool.execute(market="cn", limit=15)
        except Exception:
            return []

    def _keyword_score(self, items: list[dict[str, Any]]) -> tuple[float, int, int]:
        pos = 0
        neg = 0
        for item in items:
            text = (item.get("title", "") + item.get("summary", ""))
            pos += sum(1 for pat in self.POSITIVE if re.search(pat, text))
            neg += sum(1 for pat in self.NEGATIVE if re.search(pat, text))
        total = pos + neg
        if total == 0:
            return (50.0, 0, 0)
        raw = 50 + (pos - neg) / max(total, 1) * 40
        return (min(100, max(0, round(raw, 1))), pos, neg)

    @staticmethod
    def _calc_confidence(pos: int, neg: int) -> float:
        total = pos + neg
        if total == 0:
            return 0.3
        ratio = abs(pos - neg) / max(total, 1)
        count_factor = min(total / 10, 1.0)
        return min(1.0, 0.4 + 0.6 * ratio * count_factor)

    @staticmethod
    def _build_reasoning(ticker: str | None, score: float, pos: int, neg: int, items: list[dict[str, Any]]) -> str:
        tag = ticker or "全市场"
        if pos + neg == 0:
            return f"{tag} 无显著情绪信号，默认中性 50 分"
        tone = "偏多" if score >= 55 else ("偏空" if score <= 45 else "中性")
        return f"{tag} 情绪{tone}（{score}分），匹配正面关键词 {pos} 次，负面 {neg} 次，共分析 {len(items)} 条新闻"
