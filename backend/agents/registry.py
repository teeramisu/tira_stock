"""AgentRegistry — central lookup for all agent instances."""

from __future__ import annotations


from .base import BaseAgent


class AgentRegistry:
    """Thread-safe registry of BaseAgent instances."""

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._order: list[str] = []

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        if agent.name not in self._order:
            self._order.append(agent.name)

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def list_all(self) -> list[BaseAgent]:
        return [self._agents[n] for n in self._order if n in self._agents]

    def list_names(self) -> list[str]:
        return [n for n in self._order if n in self._agents]

    def remove(self, name: str) -> None:
        self._agents.pop(name, None)
        if name in self._order:
            self._order.remove(name)

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"<AgentRegistry agents={self.list_names()}>"
