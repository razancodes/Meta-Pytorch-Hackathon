"""
Memex OS-Agent Benchmark — Curriculum Engine.

Provides Prioritized Level Replay (PLR) for adaptive scenario sampling
and proxy-based regret computation for training curriculum optimization.
"""

from .plr_engine import PLREngine, ScenarioRecord
from .oracle import proxy_regret

__all__ = ["PLREngine", "ScenarioRecord", "proxy_regret"]
