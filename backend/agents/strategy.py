"""Strategy Agent — combines all sub-agent signals into a final trading recommendation."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from memory.models import SignalRecord
from memory.session import SessionMemory
from signals.combiner import combine_signals


class StrategyAgent(BaseAgent):
    """Aggregate sentiment, industry, technique signals into one trading signal.

    Uses confidence-weighted combination of available sub-agent signals.
    Maps score ranges to buy/sell/hold decisions:
      - 70-100: strong buy
      - 55-70:  buy
      - 45-55:  hold
      - 30-45:  sell
      - 0-30:   strong sell
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="strategy", description="Trading strategy signal aggregator", **kwargs)

    def analyze(self, ticker: str | None = None, session: SessionMemory | None = None, **_: Any) -> SignalRecord:
        session = session or self.session
        self._trace("start", f"StrategyAgent: ticker={ticker}")

        sub_signals = self._gather_sub_signals(session)
        score, confidence, combo_reason = combine_signals(sub_signals, method="confidence_weighted")

        direction, suggestion = self._score_to_direction(score, confidence)
        position = self._score_to_position(score, confidence)
        stop_loss, take_profit = self._calc_risk_params(score, position)
        sell_index = self._calc_sell_index(score, confidence)

        # 构建详细计算步骤
        calc_steps = []
        if sub_signals:
            parts = []
            for s in sub_signals:
                parts.append(f"{s.agent_name}(分数{s.score}, 置信度{s.confidence:.2f})")
            calc_steps.append(f"① 子信号: {', '.join(parts)}")
            valid = [s for s in sub_signals if (s.confidence or 0) >= 0.3 and s.score is not None]
            if valid:
                total_conf = sum(s.confidence for s in valid)
                weighted_parts = [f"{s.score}×{s.confidence:.2f}" for s in valid]
                calc_steps.append(f"② 加权合并: ({' + '.join(weighted_parts)}) / {total_conf:.2f} = {score}")
            calc_steps.append(f"③ 置信度: {confidence:.2f}")
        else:
            calc_steps.append("① 无有效子信号，使用默认值 score=50, confidence=0")
        calc_steps.append(f"④ 方向判定: score={score} → {direction} ({suggestion})")
        calc_steps.append(f"⑤ 目标仓位: {position}%")
        raw_si = 100 - score
        calc_steps.append(f"⑥ 卖出指数: raw=100-{score}={raw_si}, "
                          f"adjusted=50+({raw_si}-50)×min({confidence:.2f}/0.5,1)={sell_index}")

        if self.llm is not None:
            llm_reason = self._llm_reason(
                "作为投资策略分析师，请基于以下Agent信号给出最终建议：\n"
                + combo_reason
                + f"\n综合评分: {score}/100, 置信度: {confidence:.2f}\n"
                + f"建议: {suggestion}\n"
                + "请用简洁中文给出理由（50字以内）：",
            )
            reasoning = llm_reason or suggestion
        else:
            reasoning = suggestion

        self._trace("complete", f"StrategyAgent: score={score}, direction={direction}")
        return SignalRecord(
            agent_name=self.name,
            signal_type="score",
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            ticker=ticker,
            metadata={
                "direction": direction,
                "suggestion": suggestion,
                "target_position": position,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "sell_index": sell_index,
                "sub_signals": combo_reason,
                "calc_steps": calc_steps,
            },
        )

    def _gather_sub_signals(self, session: SessionMemory) -> list[SignalRecord]:
        signals: list[SignalRecord] = []
        for name in ("sentiment", "industry", "technique"):
            val = session.get_intermediate(name)
            if isinstance(val, SignalRecord):
                signals.append(val)
        return signals

    @staticmethod
    def _score_to_direction(score: float, confidence: float) -> tuple[str, str]:
        if confidence < 0.3:
            return ("hold", "信号置信度不足，建议观望")
        if score >= 70:
            return ("buy", "强烈看多，建议积极布局")
        if score >= 55:
            return ("buy", "偏多信号，建议适当参与")
        if score >= 45:
            return ("hold", "中性偏震荡，建议持仓观望")
        if score >= 30:
            return ("sell", "偏空信号，建议减仓")
        return ("sell", "明显看空，建议清仓避险")

    @staticmethod
    def _score_to_position(score: float, confidence: float) -> float:
        if confidence < 0.3:
            return 0.0
        base = (score - 50) / 50 * 30
        return max(0.0, min(30.0, round(base * confidence, 1)))

    @staticmethod
    def _calc_risk_params(score: float, position: float) -> tuple[float | None, float | None]:
        if position <= 0:
            return None, None
        stop_loss = round(-min(5.0, position * 0.2), 1)
        take_profit = round(min(15.0, position * 0.5), 1)
        return stop_loss, take_profit

    @staticmethod
    def _calc_sell_index(score: float, confidence: float) -> float:
        """计算每日卖出指数: 0=不建议卖, 100=强烈建议卖。

        基于综合评分反转，置信度越低越趋近50（中性）。
        """
        raw = 100 - score
        # 置信度调节：低置信度时向50靠拢
        adjusted = 50 + (raw - 50) * min(confidence / 0.5, 1.0)
        return round(max(0, min(100, adjusted)), 1)
