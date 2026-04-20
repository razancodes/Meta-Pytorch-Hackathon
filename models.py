"""
Memex OS-Agent Benchmark — Pydantic models for Action, Observation, State, and OS subsystems.

Defines the core data contracts for the environment:
- AMLAction: agent tool calls
- AMLObservation: step results returned to the agent
- AsyncJobInfo: metadata for background (interrupt) tasks
- AGUIState: frontend visualization payload
- AMLState: full internal environment state including OS mechanics
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AsyncJobStatus(str, Enum):
    """Status of an asynchronous background task."""
    PENDING = "pending"
    READY = "ready"
    RETRIEVED = "retrieved"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AMLAction(BaseModel):
    """Agent action — a single tool call with parameters."""

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
    """Observation returned to the agent after each step."""

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
        description="Additional context (e.g., step count, task id, agui_state)",
    )


# ---------------------------------------------------------------------------
# Async Job (Interrupt subsystem)
# ---------------------------------------------------------------------------

class AsyncJobInfo(BaseModel):
    """Metadata for a background task in the Interrupt subsystem."""

    job_id: str = Field(..., description="Unique job identifier (e.g., REQ-001)")
    tool: str = Field(..., description="The tool that enqueued this job")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original parameters of the async request",
    )
    eta_remaining: int = Field(
        ..., description="Steps until result is ready (0 = ready)"
    )
    status: AsyncJobStatus = Field(default=AsyncJobStatus.PENDING)
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The deferred result payload, populated when ETA reaches 0",
    )


# ---------------------------------------------------------------------------
# AGUI State (frontend visualization payload)
# ---------------------------------------------------------------------------

class RAMUsage(BaseModel):
    """Current context window state for the AGUI."""
    capacity: str = Field(..., description="e.g., '2/2 observations'")
    active_context: List[str] = Field(default_factory=list)


class AGUIState(BaseModel):
    """AGUI visualization payload emitted after every step."""

    ram_usage: RAMUsage = Field(default_factory=lambda: RAMUsage(capacity="0/2 observations"))
    disk_storage: List[str] = Field(default_factory=list)
    async_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    kernel_directives: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AMLState(BaseModel):
    """Internal environment state tracked across steps.

    Extends the base OpenEnv State contract with OS mechanic subsystems:
    - Virtual Memory (RAM + Disk)
    - Interrupts (Async Queue)
    - Kernel Updates (Directives)
    """

    episode_id: Optional[str] = None
    step_count: int = 0

    # Task identity
    task_id: str = ""

    # Evidence tracking flags (legacy, preserved for grader compatibility)
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

    # Redundancy detection
    tool_call_hashes: List[str] = Field(default_factory=list)

    # --- OS Mechanic: Virtual Memory ---
    ram_observations: List[str] = Field(
        default_factory=list,
        description="Current context window contents (capped at RAM_CAPACITY)",
    )
    disk_case_file: List[str] = Field(
        default_factory=list,
        description="Persistent scratchpad entries written by the agent",
    )
    evicted_entity_ids: List[str] = Field(
        default_factory=list,
        description="Entity IDs that were in evicted observations but NOT paged to disk",
    )

    # --- OS Mechanic: Interrupts ---
    async_jobs: Dict[str, AsyncJobInfo] = Field(
        default_factory=dict,
        description="Active background jobs keyed by job_id",
    )

    # --- OS Mechanic: Kernel Updates ---
    kernel_directives: List[str] = Field(
        default_factory=list,
        description="Mutable system prompt fragments (base + agent-injected rules)",
    )

    # --- OS Mechanic Counters (for reward computation) ---
    page_fault_count: int = 0
    async_timeout_count: int = 0
    successful_pages: int = 0
    meta_injections: int = 0
