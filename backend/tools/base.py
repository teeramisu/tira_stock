"""BaseTool — abstract class for agent-callable tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """A tool that an agent can invoke."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for agent prompts."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Run the tool and return results."""
