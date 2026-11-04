"""Orchestrator — coordinates multi-agent analysis pipelines.

Supports two execution modes:
1. DAG pipeline: agents run in a predefined order (sentiment → industry → strategy → risk)
2. Supervisor mode: a supervisor agent collects results and makes a final decision
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.models import DecisionRecord, MarketSnapshot, SignalRecord
from memory.session import SessionMemory
from memory.store import MemoryStore

from .registry import AgentRegistry


class Orchestrator:
    """Runs agents in DAG order and aggregates signals."""

    SUCCESSORS: dict[str, list[str]] = {
        "sentiment": [],
        "industry": [],
        "technique": [],
        "strategy": ["sentiment", "industry", "technique"],
        "risk": ["strategy"],
    }

    def __init__(
        self,
        registry: AgentRegistry,
        memory: MemoryStore,
        session: SessionMemory | None = None,
    ):
        self.registry = registry
        self.memory = memory
        self.session = session or SessionMemory()
        self.last_signals: dict[str, list[SignalRecord]] = {}
        self.last_decision: DecisionRecord | None = None

    def run(
        self,
        *,
        agents: list[str] | None = None,
        ticker: str | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> tuple[dict[str, list[SignalRecord]], DecisionRecord | None]:
        """Run analysis pipeline and return (signals_by_agent, final_decision).

        Args:
            agents: Agent names to run (default: all registered).
            ticker: Optional stock ticker to focus analysis on.
            force: Skip the latest-signal check (always re-analyze).
        """
        names = agents or self.registry.list_names()
        ordered = self._topological_sort(names)
        self.last_signals = {}
        self.last_decision = None

        for agent_name in ordered:
            if not force:
                last = self.memory.latest_signal(agent_name)
                if last is not None:
                    try:
                        ts = datetime.fromisoformat(last.created_at)
                        age_s = (datetime.now() - ts).total_seconds()
                        if age_s < 300:
                            self.last_signals[agent_name] = [last]
                            self.session.set_intermediate(agent_name, last)
                            continue
                    except Exception:
                        pass

            agent = self.registry.get(agent_name)
            if agent is None:
                continue

            self.session.set_intermediate(agent_name, {"status": "running"})
            try:
                signal = agent.analyze(
                    ticker=ticker,
                    session=self.session,
                    **kwargs,
                )
                self.last_signals[agent_name] = [signal]
                self.session.set_intermediate(agent_name, signal)
                self.memory.save_signal(signal)
            except Exception as exc:
                self.last_signals[agent_name] = []
                self.session.set_intermediate(agent_name, {"status": "error", "error": str(exc)})

        decision = self._build_decision(ticker)
        if decision is not None:
            self.last_decision = decision
            self.memory.save_decision(decision)

        return self.last_signals, self.last_decision

    def run_single(self, agent_name: str, **kwargs: Any) -> SignalRecord:
        agent = self.registry.get(agent_name)
        if agent is None:
            raise ValueError(f"Unknown agent: {agent_name}")
        signal = agent.analyze(session=self.session, **kwargs)
        self.memory.save_signal(signal)
        self.last_signals[agent_name] = [signal]
        return signal

    def take_snapshot(self, market: str = "cn", regime: str = "unknown") -> MarketSnapshot:
        today = datetime.now().strftime("%Y-%m-%d")
        snap = MarketSnapshot(
            trade_date=today,
            market=market,
            regime=regime,
            snapshot={"signals": {k: [s.model_dump() for s in v] for k, v in self.last_signals.items()}},
        )
        self.memory.save_snapshot(snap)
        return snap

    # ---- internal ----

    def _topological_sort(self, names: list[str]) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def visit(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            for pred in self.SUCCESSORS.get(n, []):
                if pred in names and pred not in visited:
                    visit(pred)
            if n in names:
                order.append(n)

        for name in sorted(names):
            visit(name)
        return order

    def _build_decision(self, ticker: str | None) -> DecisionRecord | None:
        strategy_sigs = self.last_signals.get("strategy", [])
        risk_sigs = self.last_signals.get("risk", [])

        if not strategy_sigs:
            return None

        strat = strategy_sigs[0]
        risk_blocked = any(s.metadata.get("blocked") for s in risk_sigs)

        if risk_blocked:
            return DecisionRecord(
                run_id=f"run_{int(datetime.now().timestamp())}",
                agent_name="orchestrator",
                signal_ids=[s.id for v in self.last_signals.values() for s in v],
                decision="pass",
                score=strat.score,
                confidence=strat.confidence,
                ticker=ticker,
                reasoning="Risk gate blocked execution",
                metadata={"blocked_by_risk": True},
            )

        return DecisionRecord(
            run_id=f"run_{int(datetime.now().timestamp())}",
            agent_name="orchestrator",
            signal_ids=[s.id for v in self.last_signals.values() for s in v],
            decision=strat.metadata.get("direction", "pass"),
            score=strat.score,
            confidence=strat.confidence,
            ticker=ticker or strat.ticker,
            target_position=strat.metadata.get("target_position"),
            stop_loss=strat.metadata.get("stop_loss"),
            take_profit=strat.metadata.get("take_profit"),
            reasoning=strat.reasoning,
        )
