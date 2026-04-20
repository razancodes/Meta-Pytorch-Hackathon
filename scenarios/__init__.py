"""
Scenario registry for the Memex OS-Agent Benchmark.

Uses the ScenarioGenerator for procedural scenario creation.
Maintains backward compatibility with get_scenario(task_id) interface.
"""

from .base import BaseScenario
from .procedural_generator import ScenarioGenerator, GeneratedScenario, generate_scenario

# Singleton generator — re-seeded per episode in the environment
_generator = ScenarioGenerator()


def get_scenario(task_id: str, seed: int | None = None) -> BaseScenario:
    """Generate a procedural scenario for the given task_id.

    Args:
        task_id: Difficulty level ("easy", "medium", "hard") OR a typology
                 name ("structuring", "layering", "trade_based_ml").
                 Also accepts compound forms: "medium_layering", "hard_structuring", etc.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A GeneratedScenario conforming to BaseScenario.
    """
    global _generator

    if seed is not None:
        _generator = ScenarioGenerator(seed=seed)

    # Parse compound task_id like "medium_layering" or "hard_structuring"
    difficulty = None
    typology = None

    difficulties = {"easy", "medium", "hard"}
    typologies = {"structuring", "layering", "trade_based_ml"}

    parts = task_id.lower().replace("-", "_").split("_", 1)

    if task_id.lower() in difficulties:
        difficulty = task_id.lower()
    elif task_id.lower() in typologies:
        typology = task_id.lower()
    elif len(parts) >= 2:
        # Try "medium_layering" or "layering_hard" etc.
        if parts[0] in difficulties:
            difficulty = parts[0]
            rest = "_".join(parts[1:])
            if rest in typologies:
                typology = rest
        elif parts[0] in typologies:
            typology = parts[0]
        # Also try the full string as typology
        full = task_id.lower().replace("-", "_")
        if full in typologies:
            typology = full

    return _generator.generate(difficulty=difficulty, typology=typology)


__all__ = [
    "get_scenario",
    "BaseScenario",
    "ScenarioGenerator",
    "GeneratedScenario",
    "generate_scenario",
]
