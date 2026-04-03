"""
AML Investigation Environment — Pydantic models for Action, Observation, and State.
Compatible with OpenEnv base types but also works standalone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AMLAction(BaseModel):
    """Agent action — a single tool call with parameters.

    Compatible with the OpenEnv Action base class contract:
    - ``metadata`` dict field
    - ``model_config`` with ``extra="allow"``
    """

    model_config = {"extra": "allow"}

    tool: str = Field(..., description="Tool name to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters as key-value pairs",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata attached to this action",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class AMLObservation(BaseModel):
    """Observation returned to the agent after each step.

    Compatible with the OpenEnv Observation base class contract:
    - ``done`` (bool)
    - ``reward`` (float | None)
    - ``metadata`` (dict)
    """

    tool_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data returned by the executed tool",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names the agent can call next",
    )
    message: str = Field(
        default="",
        description="Human-readable message describing what happened",
    )
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: Optional[float] = Field(
        default=None,
        description="Step reward (None until a value is assigned)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (e.g., step count, task id)",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AMLState(BaseModel):
    """Internal environment state tracked across steps.

    Compatible with the OpenEnv State base class contract:
    - ``episode_id`` (str | None)
    - ``step_count`` (int)
    """

    episode_id: Optional[str] = None
    step_count: int = 0

    # Task identity
    task_id: str = ""

    # Evidence tracking flags
    alert_reviewed: bool = False
    customer_profiled: bool = False
    transactions_queried: bool = False
    watchlist_checked: List[str] = Field(default_factory=list)
    network_traced: bool = False
    source_checked: List[str] = Field(default_factory=list)
    risk_assessed: bool = False
    decision_made: bool = False

    # Collected evidence
    findings: List[str] = Field(
        default_factory=list,
        description="Key findings recorded during investigation",
    )

    # Reward accumulation
    accumulated_reward: float = 0.0

    # Previously called tool+param hashes for redundancy detection
    tool_call_hashes: List[str] = Field(default_factory=list)
