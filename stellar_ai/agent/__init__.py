"""Agent layer: a ReAct-style orchestrator with deterministic tool routing."""

from .agent import Agent, safe_eval

__all__ = ["Agent", "safe_eval"]
