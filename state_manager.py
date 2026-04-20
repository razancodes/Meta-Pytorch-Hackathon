"""
Memex OS-Agent Benchmark — State Manager.

Central orchestrator for the three OS mechanics:
  I.   Virtual Memory (RAM eviction + Disk paging)
  II.  Interrupts (Async background tasks with ETA countdown)
  III. Kernel Updates (Mutable system prompt injection)

Also generates the AGUI visualization payload after every step.
"""

from __future__ import annotations

import re
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from models import (
    AGUIState,
    AMLState,
    AsyncJobInfo,
    AsyncJobStatus,
    RAMUsage,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAM_CAPACITY: int = 2          # Max observation summaries held in context
BASE_DIRECTIVE: str = "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."

# Entity-ID regex: matches patterns like CUST001, ENT_A, TXN-001-A, REQ-099, ACC-7701
_ENTITY_RE = re.compile(
    r"\b(?:CUST\d+|ENT_[A-Z]|TXN-\d{3}-[A-Z]|REQ-\d{3}|ACC-\d+|ALERT-\d{4}-\d{4})\b"
)


def _extract_entity_ids(text: str) -> Set[str]:
    """Extract all entity/transaction/job IDs from a text string."""
    return set(_ENTITY_RE.findall(text))


class StateManager:
    """Manages the OS-level subsystems layered on top of the AML environment.

    This class is instantiated once per episode (at reset) and mutated on
    every step.  It does NOT own the scenario data or tool routing — those
    remain in AMLEnvironment.  StateManager is the *middleware* between the
    raw tool results and the observation returned to the agent.

    Subsystems:
        ram:    deque of observation summary strings (capped at RAM_CAPACITY).
        disk:   list of strings written by the agent via write_to_case_file.
        async_q: dict of AsyncJobInfo, keyed by job_id.
        kernel: list of directive strings (base + agent-injected).

    Reward signals returned by tick/tool methods are *micro-rewards* that
    the grader aggregates into the step reward.
    """

    def __init__(self) -> None:
        self._ram: deque[str] = deque(maxlen=RAM_CAPACITY)
        self._disk: List[str] = []
        self._async_jobs: Dict[str, AsyncJobInfo] = {}
        self._kernel: List[str] = [BASE_DIRECTIVE]
        self._job_counter: int = 0

        # Track entity IDs that have been evicted from RAM but NOT paged to disk
        self._evicted_ids: Set[str] = set()
        # Track entity IDs currently on disk (from case file content)
        self._disk_ids: Set[str] = set()
        # Track entity IDs currently in RAM
        self._ram_ids: Set[str] = set()

    # ------------------------------------------------------------------ #
    # RAM — Virtual Memory                                                 #
    # ------------------------------------------------------------------ #

    def push_observation(self, summary: str) -> List[str]:
        """Push an observation summary into RAM, evicting old entries if full.

        Returns:
            List of evicted observation strings (empty if no eviction).
        """
        evicted: List[str] = []

        if len(self._ram) >= RAM_CAPACITY:
            oldest = self._ram[0]  # will be auto-evicted by deque
            evicted.append(oldest)

            # Track entity IDs from evicted observation
            evicted_ids = _extract_entity_ids(oldest)
            # Only add to evicted set if they are NOT on disk
            for eid in evicted_ids:
                if eid not in self._disk_ids:
                    self._evicted_ids.add(eid)

        self._ram.append(summary)

        # Update RAM entity ID set
        self._ram_ids = set()
        for obs in self._ram:
            self._ram_ids.update(_extract_entity_ids(obs))

        # Entity IDs that are now back in RAM are no longer "evicted"
        self._evicted_ids -= self._ram_ids

        return evicted

    def check_page_fault(self, tool: str, params: Dict[str, Any]) -> bool:
        """Check if the agent is referencing an entity ID that was evicted
        and never paged to disk.

        Returns True if a page fault should be triggered.
        """
        # Extract entity IDs from the tool call parameters
        param_str = " ".join(str(v) for v in params.values())
        referenced_ids = _extract_entity_ids(param_str)

        # A page fault occurs when a referenced ID exists in evicted set
        for eid in referenced_ids:
            if eid in self._evicted_ids:
                return True
        return False

    @property
    def ram_contents(self) -> List[str]:
        """Current RAM observation summaries."""
        return list(self._ram)

    # ------------------------------------------------------------------ #
    # Disk — Persistent Case File                                          #
    # ------------------------------------------------------------------ #

    def write_to_disk(self, content: str) -> None:
        """Page data to the persistent case file (Disk)."""
        self._disk.append(content)
        # Track entity IDs on disk
        new_ids = _extract_entity_ids(content)
        self._disk_ids.update(new_ids)
        # Remove from evicted set — they're now safe
        self._evicted_ids -= new_ids

    @property
    def disk_contents(self) -> List[str]:
        """Current disk (case file) entries."""
        return list(self._disk)

    # ------------------------------------------------------------------ #
    # Async Queue — Interrupts                                             #
    # ------------------------------------------------------------------ #

    def enqueue_async(
        self,
        tool: str,
        params: Dict[str, Any],
        eta_steps: int,
        result_payload: Dict[str, Any],
    ) -> str:
        """Enqueue a background task with a deferred result.

        Args:
            tool: The tool name that triggered the async request.
            params: Original parameters.
            eta_steps: Number of steps until the result is ready.
            result_payload: The pre-computed result to deliver when ETA=0.

        Returns:
            The generated Job ID (e.g., "REQ-001").
        """
        self._job_counter += 1
        job_id = f"REQ-{self._job_counter:03d}"
        self._async_jobs[job_id] = AsyncJobInfo(
            job_id=job_id,
            tool=tool,
            parameters=params,
            eta_remaining=eta_steps,
            status=AsyncJobStatus.PENDING,
            result=result_payload,
        )
        return job_id

    def tick_async_jobs(self) -> None:
        """Decrement ETA on all pending async jobs. Called at the start of each step."""
        for job in self._async_jobs.values():
            if job.status == AsyncJobStatus.PENDING and job.eta_remaining > 0:
                job.eta_remaining -= 1
                if job.eta_remaining <= 0:
                    job.status = AsyncJobStatus.READY

    def retrieve_async(self, job_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Attempt to retrieve an async job result.

        Returns:
            (result_dict, is_timeout_penalty)
            - If job is READY: returns (result, False)
            - If job is PENDING (ETA>0): returns (None, True) — timeout penalty
            - If job not found or already retrieved: returns (None, False)
        """
        job = self._async_jobs.get(job_id)
        if job is None:
            return None, False

        if job.status == AsyncJobStatus.RETRIEVED:
            return None, False

        if job.status == AsyncJobStatus.PENDING:
            # Premature retrieval — timeout penalty
            return None, True

        # READY
        job.status = AsyncJobStatus.RETRIEVED
        return job.result, False

    @property
    def active_jobs(self) -> List[AsyncJobInfo]:
        """All non-retrieved async jobs."""
        return [
            j for j in self._async_jobs.values()
            if j.status != AsyncJobStatus.RETRIEVED
        ]

    # ------------------------------------------------------------------ #
    # Kernel — Mutable System Prompt                                       #
    # ------------------------------------------------------------------ #

    def inject_directive(self, rule: str, step: int) -> None:
        """Append a compliance rule to the kernel directives."""
        self._kernel.append(f"Added (Step {step}): {rule}")

    @property
    def kernel_directives(self) -> List[str]:
        """Current system prompt fragments."""
        return list(self._kernel)

    # ------------------------------------------------------------------ #
    # AGUI Payload                                                         #
    # ------------------------------------------------------------------ #

    def build_agui_state(self) -> Dict[str, Any]:
        """Generate the AGUI visualization payload."""
        agui = AGUIState(
            ram_usage=RAMUsage(
                capacity=f"{len(self._ram)}/{RAM_CAPACITY} observations",
                active_context=list(self._ram),
            ),
            disk_storage=list(self._disk),
            async_jobs=[
                {
                    "id": j.job_id,
                    "tool": j.tool,
                    "eta_steps": j.eta_remaining,
                    "status": j.status.value,
                }
                for j in self._async_jobs.values()
                if j.status != AsyncJobStatus.RETRIEVED
            ],
            kernel_directives=list(self._kernel),
        )
        return agui.model_dump()

    # ------------------------------------------------------------------ #
    # Snapshot — sync state to AMLState                                    #
    # ------------------------------------------------------------------ #

    def sync_to_state(self, state: AMLState) -> None:
        """Write all OS mechanic data into the AMLState for serialization."""
        state.ram_observations = list(self._ram)
        state.disk_case_file = list(self._disk)
        state.async_jobs = dict(self._async_jobs)
        state.kernel_directives = list(self._kernel)
        state.evicted_entity_ids = list(self._evicted_ids)
