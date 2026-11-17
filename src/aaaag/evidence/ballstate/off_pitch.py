"""Ball seen off the pitch: the simplest, highest-precision dead-ball producer."""

from __future__ import annotations

import numpy as np

from ...contracts import Evidence, MatchBundle
from ...registry import EVIDENCE
from .cues import off_pitch_mask


@EVIDENCE.register("ballstate_off_pitch", version="1", example={"margin_m": 0.5})
def ballstate_off_pitch(
    bundle: MatchBundle, *, margin_m: float = 0.5, p_dead: float = 0.95
) -> Evidence:
    """D with probability `p_dead` where the ball is seen beyond a line by `margin_m`, NaN
    elsewhere: an on-pitch ball says nothing, and an unseen ball is no evidence at all.

    Uses: ball position. Fails with homography error near the far touchline, and catches only the
    part of a stoppage while the ball is still out (restarts are taken on the pitch).
    """
    off = off_pitch_mask(bundle.tracking, margin_m)
    dead = np.where(off, p_dead, np.nan).astype(np.float32)
    return Evidence.from_factors("ballstate_off_pitch", dead, np.full(len(dead), 0.5))
