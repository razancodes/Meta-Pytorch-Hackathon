"""
Memex OS-Agent Benchmark — FastAPI server.

Exposes OpenEnv-compatible HTTP endpoints:
  POST /reset   → initial observation (with AGUI state)
  POST /step    → step observation (with AGUI state)
  GET  /state   → current state snapshot
  GET  /agui    → latest AGUI visualization payload
  GET  /health  → {"status": "ok"}
"""

from __future__ import annotations

import sys
import os

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

# Environment import
try:
    from .aml_environment import AMLEnvironment
except ImportError:
    try:
        from aml_environment import AMLEnvironment
    except ImportError:
        from server.aml_environment import AMLEnvironment

# Models import
try:
    from models import AMLAction
except ImportError:
    from server.models import AMLAction

# Try openenv create_app first
_openenv_available = False
try:
    from openenv.core.env_server.http_server import create_app  # type: ignore
    from models import AMLObservation

    _global_env_for_openenv = AMLEnvironment()
    app = create_app(
        lambda: _global_env_for_openenv,
        AMLAction,
        AMLObservation,
        env_name="aml_investigation_env",
    )
    _openenv_available = True
except ImportError:
    pass

# Standalone FastAPI app (fallback when openenv-core is not installed)
if not _openenv_available:

    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None
        task_id: str = "easy"

    class StepRequest(BaseModel):
        action: Dict[str, Any] = {}
        timeout_s: Optional[float] = None
        request_id: Optional[str] = None

    app = FastAPI(
        title="Memex: AML OS-Agent Benchmark",
        description=(
            "OpenEnv-compatible environment testing LLMs as OS-agents "
            "over AML investigations. Implements Virtual Memory, Interrupts, "
            "and Kernel Updates."
        ),
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _env = AMLEnvironment()
    _last_agui_state: Dict[str, Any] = {}

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "aml_investigation_env", "version": "0.2.0"}

    @app.post("/reset")
    async def reset(request: ResetRequest):
        global _last_agui_state
        obs = _env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )
        _last_agui_state = obs.metadata.get("agui_state", {})
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
        global _last_agui_state
        act_data = request.action
        if not act_data:
            raise HTTPException(400, "Missing 'action' field")
        tool = act_data.get("tool", "")
        parameters = act_data.get("parameters", {})
        metadata = act_data.get("metadata", {})
        if not tool:
            raise HTTPException(400, "Missing 'tool' in action")
        action = AMLAction(tool=tool, parameters=parameters, metadata=metadata)
        obs = _env.step(action, timeout_s=request.timeout_s)
        _last_agui_state = obs.metadata.get("agui_state", {})
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
        s = _env.state
        return s.model_dump()

    @app.get("/agui")
    async def get_agui():
        """Return the latest AGUI visualization payload for the frontend."""
        return {
            "step": _env.state.step_count,
            "max_steps": 25,
            "environment_status": "done" if _env.state.decision_made else "in_progress",
            "agui_state": _last_agui_state,
        }

    @app.get("/")
    async def root():
        return {
            "name": "memex_aml_investigation_env",
            "version": "0.2.0",
            "endpoints": ["/health", "/reset", "/step", "/state", "/agui"],
            "tasks": ["easy", "medium", "hard"],
            "os_mechanics": ["virtual_memory", "interrupts", "kernel_updates"],
        }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
