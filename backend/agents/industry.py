"""Industry Agent — analyzes sector heat and rotation trends."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from memory.models import SignalRecord
from memory.session import SessionMemory


class IndustryAgent(BaseAgent):
    """Analyze industry/concept board heat for sector rotation signals.

    Rule-based scoring:
      - Hot boards (涨幅>2%) → bullish signal
      - Cold boards (跌幅>2%) → bearish signal
      - Breadth of hot sectors → trend strength
    """

    HOT_THRESHOLD = 2.0
    COLD_THRESHOLD = -2.0
    MIN_BOARDS_FOR_CONFIDENCE = 5

    def __init__(self, **kwargs: Any):
        super().__init__(name="industry", description="Industry sector heat and rotation", **kwargs)

    def analyze(self, ticker: str | None = None, session: SessionMemory | None = None, **_: Any) -> SignalRecord:
        session = session or self.session
        self._trace("start", f"IndustryAgent: ticker={ticker}")

        boards = self._fetch_boards()
        score, confidence, reasoning, calc_steps = self._score_boards(boards, ticker)

        if self.llm is not None and boards:
            top_names = ", ".join(b.get("name", "") for b in boards[:5])
            llm_reason = self._llm_reason(
                f"作为行业分析师，当前A股热门板块: {top_names}。"
                f"综合评分: {score}/100。请用简洁中文给出行业配置建议（50字以内）：",
            )
            if llm_reason:
                reasoning = llm_reason

        self._trace("complete", f"IndustryAgent: score={score}, confidence={confidence:.2f}")
        return SignalRecord(
            agent_name=self.name,
            signal_type="score",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            ticker=ticker,
            metadata={
                "hot_count": sum(1 for b in boards if b.get("change_pct", 0) > self.HOT_THRESHOLD),
                "cold_count": sum(1 for b in boards if b.get("change_pct", 0) < self.COLD_THRESHOLD),
                "board_count": len(boards),
                "calc_steps": calc_steps,
            },
        )

    def _fetch_boards(self) -> list[dict[str, Any]]:
        tool = self.tools.get("industry")
        if tool is None:
            return []
        try:
            return tool.execute(board_type="industry", limit=20)
        except Exception:
            return []

    def _score_boards(self, boards: list[dict[str, Any]], ticker: str | None) -> tuple[float, float, str, list[str]]:
        if not boards:
            return (50.0, 0.3, "无行业数据，默认中性", ["① 无行业板块数据 → 默认 score=50, confidence=0.3"])

        changes = [b.get("change_pct", 0) for b in boards if isinstance(b.get("change_pct"), (int, float))]
        if not changes:
            return (50.0, 0.3, "行业涨跌幅数据缺失", ["① 涨跌幅数据缺失 → 默认 score=50, confidence=0.3"])

        avg_change = sum(changes) / len(changes)
        hot_count = sum(1 for c in changes if c > self.HOT_THRESHOLD)
        cold_count = sum(1 for c in changes if c < self.COLD_THRESHOLD)
        breadth = hot_count / max(len(changes), 1)

        # Score: 50 neutral baseline, shift by avg_change and breadth
        score = 50 + avg_change * 5 + (breadth - 0.3) * 30
        score = max(0, min(100, round(score, 1)))

        # Confidence based on data volume and dispersion
        count_factor = min(len(changes) / self.MIN_BOARDS_FOR_CONFIDENCE, 1.0)
        dispersion = abs(avg_change) / max(max(abs(c) for c in changes), 0.01)
        confidence = min(1.0, 0.3 + 0.5 * count_factor + 0.2 * dispersion)

        tag = ticker or "全市场"
        tone = "偏多" if score >= 55 else ("偏空" if score <= 45 else "中性")
        reasoning = (
            f"{tag} 行业情绪{tone}（{score}分），"
            f"热门板块 {hot_count} 个，冷门 {cold_count} 个，"
            f"平均涨跌幅 {avg_change:+.2f}%，广度 {breadth:.1%}"
        )

        # 构建详细计算步骤
        calc_steps = []
        calc_steps.append(f"① 板块数据: {len(changes)} 个板块")
        calc_steps.append(f"② 平均涨跌幅: {avg_change:+.2f}%")
        calc_steps.append(f"③ 热门板块(>{self.HOT_THRESHOLD}%): {hot_count} 个, 冷门板块(<{self.COLD_THRESHOLD}%): {cold_count} 个")
        calc_steps.append(f"④ 广度: {hot_count}/{len(changes)} = {breadth:.1%}")
        calc_steps.append(f"⑤ 分数: 50 + ({avg_change:+.2f}×5) + ({breadth:.2f}-0.3)×30 = {score}")
        calc_steps.append(f"⑥ 置信度: count_factor={count_factor:.2f}, dispersion={dispersion:.2f}, "
                          f"conf=0.3+0.5×{count_factor:.2f}+0.2×{dispersion:.2f}={confidence:.2f}")

        return (score, round(confidence, 2), reasoning, calc_steps)
