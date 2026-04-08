"""Scenario registry for the AML investigation environment."""

from .easy import EasyScenario
from .medium import MediumScenario
from .hard import HardScenario

SCENARIO_REGISTRY = {
    "easy": EasyScenario,
    "medium": MediumScenario,
    "hard": HardScenario,
}


def get_scenario(task_id: str):
    """Return an instantiated scenario for the given task_id."""
    if task_id not in SCENARIO_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {list(SCENARIO_REGISTRY.keys())}"
        )
    return SCENARIO_REGISTRY[task_id]()


__all__ = ["get_scenario", "SCENARIO_REGISTRY", "EasyScenario", "MediumScenario", "HardScenario"]
