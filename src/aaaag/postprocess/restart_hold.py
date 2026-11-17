"""Restart hold (E131, #86): the first seconds of play after a stoppage belong to one team.

After every predicted stoppage (a D run of at least `min_dead_s` inside a period, followed by H or
A), the restart team is the majority team over the first `hold_s` seconds of play (or the first
label after the stoppage with `team="first"`). Inside that window, frames labelled with the other
team become the restart team's; D frames stay D. E128's oracle put the ceiling of a perfect
restart team in the first 3 s at +0.007 to +0.008 spellF1 on IDSSE dev.

Signals: the labels only. Fails where the restart team loses the ball within `hold_s` (a contested
header after a throw-in, an intercepted short free kick): that change is then pushed out of the
window. Off unless listed in a config's `postprocess`.
"""

from __future__ import annotations

import numpy as np

from aaaag.contracts import MatchBundle
from aaaag.registry import POSTPROCESS

H, A, D = 0, 1, 2


@POSTPROCESS.register("restart_hold", version="1", example={"hold_s": 2.0})
def restart_hold(
    labels: np.ndarray,
    bundle: MatchBundle,
    *,
    hold_s: float = 2.0,
    min_dead_s: float = 2.0,
    team: str = "majority",
) -> np.ndarray:
    """Other-team frames in the first `hold_s` after each predicted stoppage become the restart team's."""
    if team not in ("majority", "first"):
        raise ValueError(f"team must be 'majority' or 'first', not {team!r}")
    out = np.array(labels, copy=True)
    fps = bundle.meta.fps
    hold = int(round(hold_s * fps))
    min_dead = int(round(min_dead_s * fps))
    if hold <= 0:
        return out
    for period in bundle.meta.periods:
        x = out[period.start : period.stop]
        b = np.flatnonzero(x[1:] != x[:-1]) + 1
        bounds = np.concatenate(([0], b, [len(x)]))
        for i in range(len(bounds) - 2):
            a, z = bounds[i], bounds[i + 1]
            if x[a] != D or z - a < min_dead or x[z] not in (H, A):
                continue
            w = x[z : min(len(x), z + hold)]
            if team == "first":
                restart = int(x[z])
            else:
                counts = np.bincount(w[(w == H) | (w == A)], minlength=2)
                restart = int(x[z]) if counts[0] == counts[1] else int(counts.argmax())
            w[w == 1 - restart] = restart
    return out
