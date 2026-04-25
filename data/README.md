# Data Directory

This directory is intentionally empty. All AML investigation scenarios are **procedurally generated** at runtime by the `scenarios/procedural_generator.py` engine.

Each `reset()` call creates a unique scenario graph with:
- Random entity networks (3-8 entities)
- Typed transactions (wires, ACH, trade invoices)
- Ground truth labels (typology, key entities, red flags)
- 3 typologies × 3 difficulty levels = 9 scenario classes

No static data files are needed.
