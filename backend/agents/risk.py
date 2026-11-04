"""Risk Gate Agent — validates signal against risk constraints."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from memory.models import SignalRecord
from memory.session import SessionMemory


class RiskAgent(BaseAgent):
    """Risk gate that can block trade signals based on conservative rules.

    Rules:
      - If strategy score < 55 and confidence < 0.5: block as "low conviction"
      - If all prior signals are below 40: block as "bearish consensus"
      - If strategy recommends > 30% position: cap at 30%
    """

    MAX_POSITION = 30.0
    MIN_STRATEGY_SCORE = 55.0
    MIN_CONFIDENCE = 0.5
    BEARISH_THRESHOLD = 40.0

    def __init__(self, **kwargs: Any):
        super().__init__(name="risk", description="Risk gate validator", **kwargs)

    def analyze(self, ticker: str | None = None, session: SessionMemory | None = None, **_: Any) -> SignalRecord:
        session = session or self.session
        blocked = False
        reasons: list[str] = []
        adjustments: dict[str, Any] = {}
        calc_steps: list[str] = []

        strategy_signal = session.get_intermediate("strategy")
        all_agents = session.get_all_context().get("agent_results", {})

        if strategy_signal is None:
            blocked = True
            reasons.append("No strategy signal to validate")
            calc_steps.append("① 无策略信号 → 拦截")

        if isinstance(strategy_signal, SignalRecord):
            strat_score = strategy_signal.score or 50
            strat_conf = strategy_signal.confidence or 0.5
            calc_steps.append(f"① 策略信号: score={strat_score}, confidence={strat_conf:.2f}")

            if strat_score < self.MIN_STRATEGY_SCORE and strat_conf < self.MIN_CONFIDENCE:
                blocked = True
                reasons.append(f"Low conviction: score={strat_score}, confidence={strat_conf}")
                calc_steps.append(f"② 信念不足: score({strat_score})<{self.MIN_STRATEGY_SCORE} 且 conf({strat_conf:.2f})<{self.MIN_CONFIDENCE} → 拦截")
            else:
                calc_steps.append(f"② 信念检查: score={strat_score}, conf={strat_conf:.2f} → 通过")

            pos = strategy_signal.metadata.get("target_position", 0)
            if isinstance(pos, (int, float)) and pos > self.MAX_POSITION:
                adjustments["target_position"] = self.MAX_POSITION
                reasons.append(f"Position capped from {pos}% to {self.MAX_POSITION}%")
                calc_steps.append(f"③ 仓位封顶: {pos}% → {self.MAX_POSITION}%")

        other_scores = []
        for name, val in all_agents.items():
            if isinstance(val, SignalRecord) and val.score is not None and name != "risk":
                other_scores.append(val.score)

        if other_scores:
            max_score = max(other_scores)
            if max_score < self.BEARISH_THRESHOLD:
                blocked = True
                reasons.append("All non-risk agents bearish: " + ", ".join(f"{s:.0f}" for s in other_scores))
                calc_steps.append(f"④ 集体看空: 最高分={max_score}<{self.BEARISH_THRESHOLD} → 拦截")
            else:
                calc_steps.append(f"④ 看空检查: 最高分={max_score}≥{self.BEARISH_THRESHOLD} → 通过")

        status = "blocked" if blocked else "passed"
        reasoning = "; ".join(reasons) if reasons else "Risk gate passed, all checks OK"

        # 风控调整卖出指数：被拦截时提升卖出指数
        sell_index = None
        if isinstance(strategy_signal, SignalRecord):
            sell_index = strategy_signal.metadata.get("sell_index")
            if sell_index is not None and blocked:
                orig_si = sell_index
                sell_index = min(100, sell_index + 15)
                adjustments["sell_index"] = sell_index
                calc_steps.append(f"⑤ 卖出指数调整: {orig_si} + 15(风控拦截) = {sell_index}")
            elif sell_index is not None:
                calc_steps.append(f"⑤ 卖出指数: {sell_index}(未调整，风控通过)")

        if not blocked:
            calc_steps.append(f"✅ 风控通过")
        else:
            calc_steps.append(f"❌ 风控拦截")

        confidence = 1.0 if not blocked else 0.8
        self._trace("complete", f"RiskAgent: {status}")
        return SignalRecord(
            agent_name=self.name,
            signal_type="alert",
            score=0 if blocked else 80,
            confidence=confidence,
            reasoning=reasoning,
            ticker=ticker,
            metadata={"blocked": blocked, "status": status, "reasons": reasons, "adjustments": adjustments, "sell_index": sell_index, "calc_steps": calc_steps},
        )
