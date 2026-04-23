"""
Memex OS-Agent Benchmark — Proxy Regret Oracle.

Computes regret for the PLR curriculum engine using a proxy score approach.
Instead of running a second agent (antagonist), we define the theoretical
maximum score as 1.0 and compute regret as the gap between that and the
protagonist's actual performance.

This is mathematically equivalent to PLR with a perfect oracle for the
purpose of weighted scenario sampling: scenarios where the agent scores
lowest get the highest replay priority.

Design decision:
  - Zero VRAM cost (no second model needed)
  - Zero compute overhead (no second episode rollout)
  - Deterministic and stable (no stochastic oracle variance)
  - The grader's terminal composite score is already normalized to [-1.0, +1.0],
    so 1.0 represents the theoretical maximum achievable performance.
"""


ORACLE_UPPER_BOUND = 1.01

def proxy_regret(protagonist_score: float, max_score: float = ORACLE_UPPER_BOUND) -> float:
    """Compute proxy regret as the gap between theoretical max and actual score.

    Args:
        protagonist_score: The agent's terminal composite score for this episode.
        max_score: Theoretical maximum score (default 1.01 from grader normalization).

    Returns:
        Non-negative regret value. Higher = agent performed worse = replay more.
    """
    return max(0.0, max_score - protagonist_score)
