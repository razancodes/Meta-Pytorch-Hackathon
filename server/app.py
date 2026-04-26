"""
Memex OS-Agent Benchmark — FastAPI server.

Exposes OpenEnv-compatible HTTP endpoints:
  POST /reset   → initial observation
  POST /step    → step observation
  GET  /state   → current state snapshot
  GET  /health  → {"status": "ok"}

Dual-mode: uses openenv-core create_app() when available,
falls back to standalone FastAPI for local dev / HF Spaces.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Path setup — ensure parent package is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# Internal imports
from models import AMLAction, AMLObservation, AMLState  # noqa: E402
from server.aml_environment import AMLEnvironment  # noqa: E402

# Try OpenEnv create_app first
_openenv_available = False
try:
    from openenv.core.env_server.http_server import create_app  # type: ignore

    _global_env = AMLEnvironment()
    app = create_app(
        lambda: _global_env, AMLAction, AMLObservation,
        env_name="aml_investigation_env",
    )
    _openenv_available = True
except ImportError:
    pass

# Standalone FastAPI (when openenv-core is not installed)
if not _openenv_available:

    # Request schemas (server-only; response types come from models.py)
    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None
        task_id: str = "easy"

    class StepRequest(BaseModel):
        action: Dict[str, Any] = {}
        timeout_s: Optional[float] = None

    # App
    app = FastAPI(
        title="Memex AML Investigation Environment",
        description="OpenEnv-compatible RL environment for Anti-Money Laundering investigation.",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    _env = AMLEnvironment()

    # -- Endpoints --------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "aml_investigation_env"}

    @app.post("/reset")
    async def reset(request: ResetRequest):
        """Reset the environment for a new episode."""
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
        """Execute a tool action."""
        act_data = request.action
        if not act_data:
            raise HTTPException(400, "Missing 'action' field")
        tool = act_data.get("tool", "")
        if not tool:
            raise HTTPException(400, "Missing 'tool' in action")

        action = AMLAction(
            tool=tool,
            parameters=act_data.get("parameters", {}),
            metadata=act_data.get("metadata", {}),
        )
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
        return _env.state.model_dump()

    @app.get("/")
    async def root():
        return {
            "name": "aml_investigation_env",
            "version": "0.2.0",
            "endpoints": ["/health", "/reset", "/step", "/state"],
            "tasks": ["easy", "medium", "hard"],
        }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
