"""RL-MACS: Multi-Agent Reinforcement Learning Collaborative Trading System.

Architecture:
- Information Agents (Factor, Macro, Sentiment, Technical) produce opinion vectors
- Decision Agent (PPO + Safety-Critic) learns to fuse opinions into trading actions
- Risk Overlay enforces hard constraints before execution

Inspired by MARS (AAAI 2026) dual-layer architecture.
"""

from __future__ import annotations

__version__ = "0.1.0"
