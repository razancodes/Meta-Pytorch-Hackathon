"""
Memex OS-Agent Benchmark — Pydantic models for Action, Observation, State, and OS subsystems.

Defines the core data contracts for the environment:
- AMLAction: agent tool calls
- AMLObservation: step results returned to the agent
- AsyncJobInfo: metadata for background (interrupt) tasks
- AGUIState: frontend visualization payload
- AMLState: full internal environment state including OS mechanics
- DeviceFingerprint: device/IP/geo data for mule ring detection
- CustomsInvoice: trade invoice data for TBML phantom shipment detection
- BeneficialOwnerNode: ownership graph node for UBO tracing
- SARPayload: FinCEN-compliant SAR filing data contract
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


class TypologyEnum(str, Enum):
    """Shared typology enum used across procedural gen, launderer, and grader."""
    STRUCTURING = "structuring"
    LAYERING = "layering"
    TRADE_BASED_ML = "trade_based_ml"
    PASS_THROUGH = "pass_through"
    MULE_RING = "mule_ring"
    SANCTIONS_EVASION = "sanctions_evasion"

    @classmethod
    def values(cls) -> list:
        return [e.value for e in cls]


class SARTypology(str, Enum):
    """FinCEN-recognized AML typology classifications."""
    STRUCTURING = "structuring"
    LAYERING = "layering"
    TRADE_BASED_ML = "trade_based_ml"
    PASS_THROUGH = "pass_through"
    MULE_RING = "mule_ring"
    SANCTIONS_EVASION = "sanctions_evasion"
    FALSE_POSITIVE = "false_positive"


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


class CurriculumState(BaseModel):
    """PLR Curriculum Engine state for the 5th AGUI panel."""

    enabled: bool = Field(default=False, description="Whether PLR curriculum is active")
    buffer_size: int = Field(default=0, description="Number of scenarios in the PLR buffer")
    mean_regret: float = Field(default=0.0, description="Mean regret across buffer scenarios")
    max_regret: float = Field(default=0.0, description="Max regret in buffer")
    mean_difficulty: float = Field(default=1.0, description="Mean difficulty score (1=easy, 3=hard)")
    buffer_diversity: float = Field(default=0.0, description="Fraction of 9 scenario combos covered")
    current_scenario_regret: float = Field(default=0.0, description="Regret of current/latest scenario")
    difficulty_label: str = Field(default="easy", description="Current scenario difficulty label")


class AGUIState(BaseModel):
    """AGUI visualization payload emitted after every step."""

    ram_usage: RAMUsage = Field(default_factory=lambda: RAMUsage(capacity="0/2 observations"))
    disk_storage: List[str] = Field(default_factory=list)
    async_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    kernel_directives: List[str] = Field(default_factory=list)
    curriculum: CurriculumState = Field(default_factory=CurriculumState)


# ---------------------------------------------------------------------------
# Device Fingerprint (Pillar 1: Mule-Ring Detection)
# ---------------------------------------------------------------------------

class DeviceFingerprint(BaseModel):
    """Device and geolocation data attached to an entity or transaction."""

    device_id: str = Field(..., description="Unique hardware/session fingerprint")
    ip_address: str = Field(..., description="IPv4 address observed during session")
    mac_address: Optional[str] = Field(
        default=None,
        description="MAC address (populated for branch terminal devices)",
    )
    latitude: float = Field(..., description="Geolocation latitude (WGS-84)")
    longitude: float = Field(..., description="Geolocation longitude (WGS-84)")
    jurisdiction: str = Field(..., description="Resolved jurisdiction from IP/geo")
    session_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of the session",
    )


# ---------------------------------------------------------------------------
# Customs Invoice (Pillar 2: Trade-Based ML / Phantom Shipments)
# ---------------------------------------------------------------------------

class CustomsInvoice(BaseModel):
    """Trade invoice record for customs/TBML verification."""

    invoice_id: str = Field(..., description="Unique customs invoice ID")
    transaction_id: str = Field(..., description="Linked transaction ID")
    hs_code: str = Field(..., description="Harmonized System tariff code (6-digit)")
    commodity_description: str = Field(..., description="Human-readable goods description")
    declared_value_usd: float = Field(..., description="Declared customs value in USD")
    shipping_weight_kg: float = Field(..., description="Declared shipping weight in kg")
    origin_country: str = Field(..., description="Country of export")
    destination_country: str = Field(..., description="Country of import")
    shipper_name: Optional[str] = Field(default=None, description="Exporter/shipper name")
    consignee_name: Optional[str] = Field(default=None, description="Importer/consignee name")
    bill_of_lading: Optional[str] = Field(default=None, description="B/L reference number")
    is_phantom: bool = Field(
        default=False,
        description="Ground truth: True if this is a phantom/fraudulent shipment",
    )


# ---------------------------------------------------------------------------
# Beneficial Ownership Node (Pillar 4: Deep Graph / UBO Tracing)
# ---------------------------------------------------------------------------

class BeneficialOwnerNode(BaseModel):
    """Node in the beneficial ownership graph for UBO tracing."""

    entity_id: str = Field(..., description="Entity or person ID")
    entity_name: str = Field(..., description="Human-readable name")
    entity_type: str = Field(
        ..., description="Type: 'individual' | 'company' | 'trust' | 'fund'"
    )
    ownership_pct: Optional[float] = Field(
        default=None,
        description="Ownership percentage (0-100) in the parent entity",
    )
    jurisdiction: str = Field(default="", description="Jurisdiction of incorporation")
    hop_count: int = Field(
        default=0,
        description="Distance from the queried entity (0 = direct, 1 = one hop, etc.)",
    )
    is_ubo: bool = Field(
        default=False,
        description="True if this entity is the Ultimate Beneficial Owner",
    )
    parent_entity_id: Optional[str] = Field(
        default=None,
        description="ID of the entity this node owns a stake in (for graph traversal)",
    )
    relationship: str = Field(
        default="",
        description="Relationship label (e.g., 'director', 'shareholder', 'trustee')",
    )


# ---------------------------------------------------------------------------
# FinCEN SAR Payload (Step 3: Data Contract)
# ---------------------------------------------------------------------------

class SARPayload(BaseModel):
    """FinCEN-compliant Suspicious Activity Report data contract.

    This is the structured payload that file_sar MUST receive.
    The grader mathematically scores each field against ground truth.
    """

    primary_subjects: List[str] = Field(
        ...,
        min_length=1,
        description="Array of entity/customer IDs identified as primary subjects of the SAR",
    )
    detected_typology: str = Field(
        ...,
        description="Detected AML typology (structuring | layering | trade_based_ml | pass_through | mule_ring)",
    )
    red_flags_identified: List[str] = Field(
        ...,
        min_length=1,
        description="Array of specific red flag strings identified during investigation",
    )
    evidence_chain: str = Field(
        ...,
        min_length=10,
        description="Narrative summary of evidence chain linking subjects to suspicious activity",
    )
    # Optional enrichment fields (bonus scoring)
    ubo_identified: Optional[str] = Field(
        default=None,
        description="Ultimate Beneficial Owner ID if traced",
    )
    transaction_velocity_summary: Optional[str] = Field(
        default=None,
        description="Summary of transaction velocity analysis (e.g., '$500k in 24h')",
    )
    geographic_risk_summary: Optional[str] = Field(
        default=None,
        description="Summary of geographic/jurisdiction risk analysis",
    )


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

    # --- Phase 3: FinCEN tracking flags ---
    device_overlap_checked: bool = False
    customs_invoice_verified: bool = False
    beneficial_ownership_queried: bool = False

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

    # --- Reward Farming Hard Caps ---
    # Maximum rewarded calls per episode (prevents PPO farming).
    # write_to_case_file: max 3 rewarded calls (+0.10 each)
    # update_system_prompt: max 2 rewarded calls (+0.15 each)
    disk_write_reward_count: int = 0
    kernel_inject_reward_count: int = 0
