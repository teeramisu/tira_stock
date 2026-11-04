"""RL-MACS Information Agents.

Each agent produces a structured opinion vector (normalized to [-1, 1])
that captures its domain-specific market view. The Decision Agent learns
how to fuse these opinions via reinforcement learning.
"""

from __future__ import annotations

from .base_rl_agent import BaseRLAgent
from .factor_agent import FactorAgent
from .macro_agent import MacroAgent
from .sentiment_agent import SentimentRLAgent
from .technical_agent import TechnicalAgent

__all__ = [
    "BaseRLAgent",
    "FactorAgent",
    "MacroAgent",
    "SentimentRLAgent",
    "TechnicalAgent",
]
