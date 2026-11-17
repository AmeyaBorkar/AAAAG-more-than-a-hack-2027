"""Label-free diagnostics: what a set of labels claims about a match, with no ground truth.

These are the only quality signals on a match without labels, so they are worth reading on every
run: a pipeline that claims 900 possessions of 5 s, or a 70% dead-ball share, is wrong whatever any
validation suite says. Compare them with the same statistics of real labels on external matches
(the priors), never with anything derived from the BAR-ATH public window. Aggregates only, so they
are safe to print in an agent session.
"""

from __future__ import annotations

import numpy as np

from ..contracts import A, D, H, MatchMeta, X
from ..official import pm


def describe(labels: np.ndarray, meta: MatchMeta) -> dict[str, float]:
    runs = meta.runs()
    filled = pm.fill_abstentions(labels.astype(np.int8), runs)
    spells = pm.possession_spells(filled, runs)
    seconds = np.array([stop - start for start, stop, _ in spells], dtype=np.float64) / meta.fps
    played = meta.played_mask()
    minutes = played.sum() / meta.fps / 60.0
    per_10 = 10.0 / minutes if minutes else 0.0

    kinds: dict[tuple[int, int], int] = {}
    for _, origin, target in pm.transitions(filled, runs):
        kinds[(origin, target)] = kinds.get((origin, target), 0) + 1
    alive = (filled[played] == H) | (filled[played] == A)

    def q(p: float) -> float:
        return float(np.percentile(seconds, p)) if len(seconds) else 0.0

    return {
        "n_spells": float(len(spells)),
        "spells_per_10min": len(spells) * per_10,
        "spell_s_mean": float(seconds.mean()) if len(seconds) else 0.0,
        "spell_s_p10": q(10),
        "spell_s_p50": q(50),
        "spell_s_p90": q(90),
        "short_spell_share_lt2s": float((seconds < 2.0).mean()) if len(seconds) else 0.0,
        "dead_share": float((filled[played] == D).mean()),
        "home_share_of_alive": float((filled[played][alive] == H).mean()) if alive.any() else 0.0,
        "turnovers_per_10min": (kinds.get((H, A), 0) + kinds.get((A, H), 0)) * per_10,
        "stoppages_per_10min": (kinds.get((H, D), 0) + kinds.get((A, D), 0)) * per_10,
        "abstain_share": float((labels[played] == X).mean()),
    }
