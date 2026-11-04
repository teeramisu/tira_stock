"""Short-term in-memory store for Agent session state."""

from __future__ import annotations

from threading import Lock
from typing import Any


class SessionMemory:
    """Thread-safe in-memory short-term memory.

    Holds the current session's agent reasoning traces, intermediate results,
    and hot context (positions, market state, risk parameters).
    """

    def __init__(self):
        self._lock = Lock()
        self._traces: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}
        self._intermediate: dict[str, Any] = {}

    def add_trace(self, agent_name: str, step: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._traces.append({
                "agent": agent_name,
                "step": step,
                "content": content,
                "metadata": metadata or {},
            })

    def get_traces(self, agent_name: str | None = None, last_n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            traces = self._traces
            if agent_name:
                traces = [t for t in traces if t["agent"] == agent_name]
            return traces[-last_n:]

    def set_context(self, key: str, value: Any) -> None:
        with self._lock:
            self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._context.get(key, default)

    def get_all_context(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._context)

    def set_intermediate(self, agent_name: str, result: Any) -> None:
        with self._lock:
            self._intermediate[agent_name] = result

    def get_intermediate(self, agent_name: str) -> Any:
        with self._lock:
            return self._intermediate.get(agent_name)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self._context.clear()
            self._intermediate.clear()

    def summary(self) -> str:
        with self._lock:
            parts = []
            parts.append(f"Traces: {len(self._traces)} entries")
            if self._context:
                parts.append(f"Context keys: {list(self._context.keys())}")
            if self._intermediate:
                parts.append(f"Intermediate agents: {list(self._intermediate.keys())}")
            return " | ".join(parts)
