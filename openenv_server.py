"""
Memex OS-Agent Benchmark — OpenEnv-Compatible FastAPI Server.

Production-grade entrypoint for deploying the AML Investigation Environment
to Hugging Face Spaces via the OpenEnv standard interface.

Endpoints (when openenv-core is available):
    POST /reset        → ResetResponse   (initial observation)
    POST /step         → StepResponse    (step observation)
    GET  /state        → State snapshot
    GET  /health       → HealthResponse
    GET  /schema       → Action/Observation/State JSON schemas
    GET  /metadata     → EnvironmentMetadata
    WS   /ws           → Full WebSocket session (reset/step/state/close)
    POST /mcp          → MCP JSON-RPC gateway

Fallback (standalone FastAPI when openenv-core is not installed):
    POST /reset, POST /step, GET /state, GET /health, GET /

Usage:
    # Local development
    uvicorn openenv_server:app --host 0.0.0.0 --port 8000

    # HF Spaces (Dockerfile CMD)
    uvicorn openenv_server:app --host 0.0.0.0 --port 7860

    # OpenEnv CLI
    openenv serve
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — guarantee the project root is importable regardless of
# how uvicorn discovers this module (cwd, PYTHONPATH, or package install).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("openenv_server")

# ---------------------------------------------------------------------------
# Internal imports (environment + models)
# ---------------------------------------------------------------------------
from models import AMLAction, AMLObservation, AMLState  # noqa: E402
from server.aml_environment import AMLEnvironment  # noqa: E402

# ---------------------------------------------------------------------------
# Boot-time validation — fail fast if the environment is broken.
# ---------------------------------------------------------------------------
_BOOT_START = time.monotonic()


def _validate_environment() -> None:
    """Run a minimal reset+step cycle at import time to catch config errors."""
    try:
        env = AMLEnvironment()
        obs = env.reset(seed=0, task_id="easy")
        assert obs.done is False, "reset() returned done=True"
        step_obs = env.step(AMLAction(tool="review_alert", parameters={}))
        assert isinstance(step_obs, AMLObservation), "step() did not return AMLObservation"
        logger.info(
            "Boot validation passed in %.1fms (reset → step OK)",
            (time.monotonic() - _BOOT_START) * 1000,
        )
    except Exception as exc:
        logger.critical("Boot validation FAILED: %s", exc, exc_info=True)
        raise SystemExit(f"Environment boot validation failed: {exc}") from exc


_validate_environment()


# =========================================================================== #
# App factory — prefer openenv-core, fall back to standalone FastAPI           #
# =========================================================================== #

_openenv_available: bool = False

try:
    from openenv.core.env_server.http_server import create_app  # type: ignore

    app = create_app(
        env=AMLEnvironment,           # factory (class), NOT an instance
        action_cls=AMLAction,
        observation_cls=AMLObservation,
        env_name="aml_investigation_env",
    )
    _openenv_available = True
    logger.info("OpenEnv create_app() — full SDK server initialized")

except ImportError:
    logger.warning(
        "openenv-core not installed — falling back to standalone FastAPI server"
    )

# ---------------------------------------------------------------------------
# Standalone fallback (mirrors the OpenEnv HTTP contract exactly)
# ---------------------------------------------------------------------------
if not _openenv_available:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse
    from pydantic import BaseModel, Field

    # -- Request / Response schemas matching OpenEnv types -------------------

    class ResetRequest(BaseModel):
        """Matches openenv.core.env_server.types.ResetRequest."""
        seed: Optional[int] = Field(default=None, ge=0, description="Random seed")
        episode_id: Optional[str] = Field(default=None, description="Episode ID")
        task_id: str = Field(default="easy", description="Scenario task ID")

    class StepRequest(BaseModel):
        """Matches openenv.core.env_server.types.StepRequest."""
        action: Dict[str, Any] = Field(..., description="Action payload")
        timeout_s: Optional[float] = Field(default=None, gt=0, description="Timeout")

    class ObservationPayload(BaseModel):
        """Flattened observation without reward/done (they go top-level)."""
        tool_result: Dict[str, Any] = Field(default_factory=dict)
        available_tools: List[str] = Field(default_factory=list)
        message: str = ""

    class ResetResponse(BaseModel):
        """Matches openenv.core.env_server.types.ResetResponse."""
        observation: Dict[str, Any]
        reward: Optional[float] = None
        done: bool = False

    class StepResponse(BaseModel):
        """Matches openenv.core.env_server.types.StepResponse."""
        observation: Dict[str, Any]
        reward: Optional[float] = None
        done: bool = False

    # -- App ----------------------------------------------------------------

    app = FastAPI(
        title="Memex AML Investigation Environment",
        description=(
            "OpenEnv-compatible RL environment for Anti-Money Laundering investigation.\n\n"
            "**Workflow:** `POST /reset` → `POST /step` (repeat) → episode ends when "
            "`done=true` → `GET /state` anytime for introspection."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single environment instance for the standalone fallback.
    # This mirrors the OpenEnv SDK's stateless HTTP mode where each /reset
    # creates a fresh episode on the same process.
    _env = AMLEnvironment()

    def _serialize_observation(obs: AMLObservation) -> Dict[str, Any]:
        """Convert AMLObservation → OpenEnv ResetResponse/StepResponse shape.

        OpenEnv's serialize_observation() strips `reward`, `done`, and `metadata`
        from the observation dict and promotes them to top-level keys.  We match
        that contract exactly so clients (EnvClient, SyncEnvClient) parse correctly.
        """
        obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
        return {
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
        }

    # -- Endpoints ----------------------------------------------------------

    @app.get("/health", tags=["Health"])
    async def health():
        """Service health probe (used by Docker HEALTHCHECK and HF Spaces)."""
        return {"status": "healthy"}

    @app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
    async def reset(request: ResetRequest):
        """Reset the environment and start a new episode.

        Returns the initial observation. The episode is parameterized by
        `task_id` which selects the AML scenario difficulty/typology.
        """
        try:
            obs = _env.reset(
                seed=request.seed,
                episode_id=request.episode_id,
                task_id=request.task_id,
            )
            return _serialize_observation(obs)
        except Exception as exc:
            logger.exception("reset() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/step", response_model=StepResponse, tags=["Environment Control"])
    async def step(request: StepRequest):
        """Execute a single investigation action.

        The `action` dict must contain at minimum `{"tool": "<tool_name>"}`.
        Optional `parameters` and `metadata` keys are forwarded to the environment.
        """
        act_data = request.action
        tool = act_data.get("tool", "")
        if not tool:
            raise HTTPException(
                status_code=422,
                detail="Missing 'tool' key in action payload",
            )

        action = AMLAction(
            tool=tool,
            parameters=act_data.get("parameters", {}),
            metadata=act_data.get("metadata", {}),
        )

        try:
            obs = _env.step(action, timeout_s=request.timeout_s)
            return _serialize_observation(obs)
        except Exception as exc:
            logger.exception("step() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/state", tags=["State Management"])
    async def get_state():
        """Return the full internal environment state snapshot."""
        return _env.state.model_dump()

    @app.get("/schema", tags=["Schema"])
    async def get_schema():
        """Return JSON schemas for Action, Observation, and State."""
        return {
            "action": AMLAction.model_json_schema(),
            "observation": AMLObservation.model_json_schema(),
            "state": AMLState.model_json_schema(),
        }

    @app.get("/metadata", tags=["Environment Info"])
    async def get_metadata():
        """Return environment metadata for documentation and UI."""
        return {
            "name": "aml_investigation_env",
            "description": (
                "Memex AML Investigation Environment with OS mechanics "
                "(Virtual Memory, Interrupts, Kernel Updates). "
                "15 tools across domain investigation and OS-mechanic actions."
            ),
            "version": "0.2.0",
        }

    @app.get("/", tags=["Environment Info"])
    async def root():
        """Redirect to the visual demo if available, else show API info."""
        _static_dir = os.path.join(_PROJECT_ROOT, "static_frontend")
        if os.path.isdir(_static_dir):
            return RedirectResponse(url="/web/")
        return {
            "name": "aml_investigation_env",
            "version": "0.2.0",
            "openenv_sdk": False,
            "endpoints": [
                "/health", "/reset", "/step", "/state",
                "/schema", "/metadata", "/docs", "/web",
            ],
            "tasks": ["easy", "medium", "hard"],
        }

    # -- Static Frontend (Next.js export) ------------------------------------
    # NOTE: For standalone mode only. The top-level mount below handles both modes.


# =========================================================================== #
# Static Frontend — mount AFTER app creation in either mode                    #
# =========================================================================== #
_static_frontend_dir = os.path.join(_PROJECT_ROOT, "static_frontend")
if os.path.isdir(_static_frontend_dir):
    from starlette.staticfiles import StaticFiles as _StaticFiles
    from starlette.responses import RedirectResponse as _RedirectResponse

    app.mount(
        "/web",
        _StaticFiles(directory=_static_frontend_dir, html=True),
        name="frontend",
    )
    logger.info("Mounted static frontend at /web from %s", _static_frontend_dir)

    # Root redirect: HF Spaces always hits "/" first.
    # OpenEnv SDK's create_app() has no root route, so "/" returns 404.
    # This adds a redirect from "/" → "/web/" in BOTH SDK and standalone modes.
    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return _RedirectResponse(url="/web/")

    logger.info("Added root / → /web/ redirect")
else:
    logger.info(
        "No static_frontend/ directory found at %s — /web route disabled",
        _static_frontend_dir,
    )


# =========================================================================== #
# CLI entrypoint                                                               #
# =========================================================================== #

def main() -> None:
    """Run the server via uvicorn with production-safe defaults."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info("Starting Memex OpenEnv server on %s:%d (workers=%d)", host, port, workers)

    uvicorn.run(
        "openenv_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        timeout_keep_alive=65,
    )


if __name__ == "__main__":
    main()

