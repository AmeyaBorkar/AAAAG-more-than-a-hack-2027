"""Degradation simulator: turn clean labelled tracking (IDSSE) into ATD-like tracking.

The point is selection that transfers. A pipeline chosen on clean tracking is chosen for a problem
BAR-ATH does not have, so every selection suite is degraded with a versioned model stored in
configs/degrade/. v1-v4 were measured on BAR-ATH aggregates and are retired from selection (ADR
2026-11-05); v5 is fitted on the three Metrica sample games only (deadball.py, #50).

    from aaaag.degrade import degrade
    noisy = degrade(bundle, version="1", seed=0)

v1 drops the ball only (ballgap.py); track slots stay clean. See docs/VALIDATION.md.
"""

from .ballgap import degrade

__all__ = ["degrade"]
