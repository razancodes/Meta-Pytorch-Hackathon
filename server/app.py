"""
AML Investigation Environment — FastAPI server.

Exposes OpenEnv-compatible HTTP endpoints:
  POST /reset   → initial observation
  POST /step    → step observation
  GET  /state   → current state snapshot
  GET  /health  → {"status": "ok"}

Dual import: if openenv-core is installed, use create_app();
otherwise fall back to manual FastAPI construction (standalone / HF Spaces mode).
"""

from __future__ import annotations

import sys
import os

# Make the parent package importable when running from the server/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Dual import: environment ----------------------------------------- #
try:
    from .aml_environment import AMLEnvironment
except ImportError:
    try:
        from aml_environment import AMLEnvironment
    except ImportError:
        from server.aml_environment import AMLEnvironment

# ---- Dual import: models ---------------------------------------------- #
try:
    from models import AMLAction, AMLObservation, AMLState
except ImportError:
    from server.models import AMLAction, AMLObservation, AMLState

# ---- Try openenv create_app first ------------------------------------- #
_openenv_available = False
try:
    from openenv.core.env_server.http_server import create_app  # type: ignore

<<<<<<< HEAD
    _global_env_for_openenv = AMLEnvironment()
    app = create_app(lambda: _global_env_for_openenv, AMLAction, AMLObservation, env_name="aml_investigation_env")
=======
    app = create_app(AMLEnvironment, AMLAction, AMLObservation, env_name="aml_investigation_env")
>>>>>>> 10edb24 (chore: First iteration of OpenEnv AML Environment ready for submission~)
    _openenv_available = True
except ImportError:
    pass

# ---- Standalone FastAPI app (used when openenv-core is not installed) -- #
if not _openenv_available:

    # ---- Request / response schemas ----------------------------------- #

    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None
        task_id: str = "easy"

    class StepRequest(BaseModel):
        action: Dict[str, Any] = {}
        timeout_s: Optional[float] = None
        request_id: Optional[str] = None

    class ObservationResponse(BaseModel):
        tool_result: Dict[str, Any] = {}
        available_tools: list = []
        message: str = ""
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = {}

    class StateResponse(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0
        task_id: str = ""
        alert_reviewed: bool = False
        customer_profiled: bool = False
        transactions_queried: bool = False
        watchlist_checked: list = []
        network_traced: bool = False
        source_checked: list = []
        risk_assessed: bool = False
        decision_made: bool = False
        findings: list = []
        accumulated_reward: float = 0.0

    # ---- App setup ---------------------------------------------------- #

    app = FastAPI(
        title="AML Investigation Environment",
        description=(
            "OpenEnv-compatible environment for Anti-Money Laundering investigation. "
            "An agent investigates alerts and decides whether to file a SAR."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # One environment instance per server process (stateful)
    _env = AMLEnvironment()

    # ------------------------------------------------------------------ #
    # Endpoints                                                            #
    # ------------------------------------------------------------------ #

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "aml_investigation_env"}

    @app.post("/reset")
    async def reset(request: Optional[ResetRequest] = None):
        """
        Reset the environment for a new episode.

        Body parameters:
        - task_id : "easy" | "medium" | "hard"  (default "easy")
        - seed    : optional random seed
        - episode_id : optional custom episode ID
        """
        if request is None:
            request = ResetRequest()
        obs = _env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )
        return {
            "observation": {
                "tool_result": obs.tool_result,
                "available_tools": obs.available_tools,
                "message": obs.message,
                "metadata": obs.metadata,
            },
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.post("/step")
    async def step(request: StepRequest):
        """
        Execute a tool action.

        Body: {"action": {"tool": "...", "parameters": {...}}, "timeout_s": null}
        Also accepts flat format: {"tool": "...", "parameters": {...}}
        """
        act_data = request.action
        # Support flat format too
        if not act_data:
            raise HTTPException(400, "Missing 'action' field")
        tool = act_data.get("tool", "")
        parameters = act_data.get("parameters", {})
        metadata = act_data.get("metadata", {})
        if not tool:
            raise HTTPException(400, "Missing 'tool' in action")
        action = AMLAction(tool=tool, parameters=parameters, metadata=metadata)
        obs = _env.step(action, timeout_s=request.timeout_s)
        return {
            "observation": {
                "tool_result": obs.tool_result,
                "available_tools": obs.available_tools,
                "message": obs.message,
                "metadata": obs.metadata,
            },
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.get("/state")
    async def get_state():
        """Return a snapshot of the current environment state."""
        s = _env.state
        return s.model_dump()

    @app.get("/")
    async def root():
        return {
            "name": "aml_investigation_env",
            "version": "0.1.0",
            "endpoints": ["/health", "/reset", "/step", "/state"],
            "tasks": ["easy", "medium", "hard"],
        }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
