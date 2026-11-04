"""BaseAgent — abstract class for all analysis agents.

Each agent receives tools, memory, and an optional LLM interface,
and produces a SignalRecord as output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memory.models import SignalRecord
from memory.session import SessionMemory
from memory.store import MemoryStore


class BaseAgent(ABC):
    """Agent base class.

    Subclasses must implement ``analyze()`` and may optionally use
    ``self.llm`` to generate reasoning via a local or remote LLM.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        tools: dict[str, Any] | None = None,
        memory: MemoryStore | None = None,
        session: SessionMemory | None = None,
        llm: Any = None,
    ):
        self.name = name
        self.description = description
        self.tools = tools or {}
        self.memory = memory
        self.session = session or SessionMemory()
        self.llm = llm

    @abstractmethod
    def analyze(self, **kwargs: Any) -> SignalRecord:
        """Run this agent's analysis and return a signal."""

    def _trace(self, step: str, content: str, **meta: Any) -> None:
        self.session.add_trace(self.name, step, content, meta)

    def _tool(self, name: str) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not available to agent '{self.name}'")
        return tool

    def _recall(self, query: str, limit: int = 5) -> list[SignalRecord]:
        if self.memory:
            return self.memory.recall_signals(query, agent_name=self.name, limit=limit)
        return []

    def _llm_reason(self, prompt: str, max_tokens: int = 300) -> str:
        if self.llm is None:
            return ""
        try:
            result = self.llm(prompt, max_new_tokens=max_tokens)
            if isinstance(result, list) and result:
                result = result[0]
            if isinstance(result, dict):
                result = result.get("generated_text", "")
            return str(result).strip()
        except Exception:
            return ""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
