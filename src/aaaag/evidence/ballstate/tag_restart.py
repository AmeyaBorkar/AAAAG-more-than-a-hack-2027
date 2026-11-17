"""Set-piece tags as dead-ball evidence: every set piece follows a stoppage (Laws 13-17).

Smart Tagging marks set pieces as a window around the restart (`DATA.md`, "Why it is worth
using"). Inside a `SET PIECES` window of one of `types`, this producer says P(dead) =
sigmoid(`llr`) and nothing about the team; everywhere else it has no opinion. Under `factorised`
fusion with this producer in `dead_from`, that adds `llr` nats to the dead log-odds of the other
dead-channel producers inside the windows and leaves every other frame untouched: a naive-Bayes
update by the likelihood ratio of "a set-piece tag covers this frame".

`llr` is one number set from external data, not a score: logit(P(D | inside a window)) -
logit(P(D)) on IDSSE truth with simulated windows (#86, P2; `analysis/scripts/tag_prior_grid.py`).
No frame, moment or offset of any match is written here: the windows are the tags' own.

Signals: `bundle.tags` only (code, `Type`, start, stop). No tracking.

Where it fails: a false tag pushes live play towards dead for the length of its window; a missed
tag adds nothing; a window that runs on after the restart pushes the first seconds of play
towards dead, which the other producers must outweigh. Kick-offs are left to `pre_kickoff_dead`
and the network by default. A bundle without tags (every external source) gets an all-NaN
Evidence, so the pipeline output is the same as without this step.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Evidence, MatchBundle
from ...registry import EVIDENCE

SET_PIECES = "SET PIECES"
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick")


def tag_windows(bundle: MatchBundle, types: tuple[str, ...] | list[str]) -> np.ndarray:
    """(T,) bool: inside the half-open window of at least one `SET PIECES` tag of `types`.

    `types` are compared lower case with the tag's `Type` value; a tag without a `Type` is skipped.
    """
    n = bundle.meta.n_frames
    mask = np.zeros(n, dtype=bool)
    wanted = {str(t).strip().lower() for t in types}
    for tag in bundle.tags or []:
        if tag.code != SET_PIECES:
            continue
        if str(tag.tags.get("Type", "")).strip().lower() not in wanted:
            continue
        start, stop = max(0, int(tag.start)), min(n, int(tag.stop))
        if stop > start:
            mask[start:stop] = True
    return mask


@EVIDENCE.register("tag_restart", version="1", example={"llr": 1.0})
def tag_restart(
    bundle: MatchBundle,
    *,
    llr: float = 1.0,
    types: list[str] | tuple[str, ...] = RESTART_TYPES,
) -> Evidence:
    """P(dead) = sigmoid(`llr`) inside set-piece tag windows of `types`, a 0.5 team tie there,
    and NaN rows elsewhere. See the module docstring for the signals and the failure modes."""
    inside = tag_windows(bundle, tuple(types))
    p = 1.0 / (1.0 + np.exp(-float(llr)))
    dead = np.where(inside, p, np.nan).astype(np.float32)
    return Evidence.from_factors("tag_restart", dead, np.full(len(dead), 0.5))
