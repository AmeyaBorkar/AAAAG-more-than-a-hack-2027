"""Minimum possession length: absorb every possession shorter than `min_s` into the previous one
(#86 E7, the label-granularity decision).

The organisers' definition (vendor/mtah2027/CHALLENGE.md) changes possession only when the other
team gains *controlled* possession: a touch without control, a deflection that does not intercept,
or an aerial duel won without control are not changes, and while the ball is contested or loose
possession stays with the last team. A decoder trained on the provider's per-touch flag emits some
of exactly those short spells. This step removes them by rule, the k-second rule of the WF1
control-semantics study: inside each alive stretch (the H/A runs between two stoppages or a period
edge) the shortest run below `min_s` is relabelled as the run before it, which then merges with the
run after it (the same team, by alternation); a short first run of a stretch joins the next run;
a stretch of one run is left alone; repeat until no run is short. Stoppages (D) and X never change.

Why 1.5 s (E7, external data only, the shipped control-memory stack, leave-one-match-out fold
models, IDSSE dev seeds 5-7): among the truths a human reading of "controlled possession" makes
plausible (the provider's own labels and the 0.5, 1 and 2 s variants of the rule), 1.5 s has the
smallest worst-case regret (0.043; 0 s has 0.098 and 1 s 0.074). It costs 0.041 spellF1 on the
provider's labels and gains 0.07 on the 2 s truth. The public five minutes of competition truth were not
used. Where it fails: wherever the real truth counts sub-second possessions (then every absorbed
spell is a lost hit), and on a decoder that already emits few short spells (it then changes little).

This is an exact port of `absorb` in analysis/scripts/wf1_control_semantics_lib.py (no pass rule);
tests/test_min_spell.py checks it against the analysis copy on synthetic labels.
"""

from __future__ import annotations

import numpy as np

from ..contracts import A, H, MatchBundle, MatchMeta
from ..registry import POSTPROCESS


def _stretches(labels: np.ndarray, meta: MatchMeta) -> list[list[list[int]]]:
    """Every alive stretch of every period as a list of [start, stop, code] runs."""
    out: list[list[list[int]]] = []
    for period in meta.periods:
        seg = labels[period.start : period.stop]
        if len(seg) == 0:
            continue
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [len(seg)]))
        codes = seg[starts]
        i = 0
        while i < len(starts):
            if codes[i] not in (H, A):
                i += 1
                continue
            j = i
            while j < len(starts) and codes[j] in (H, A):
                j += 1
            out.append(
                [
                    [int(starts[t]) + period.start, int(stops[t]) + period.start, int(codes[t])]
                    for t in range(i, j)
                ]
            )
            i = j
    return out


def _merge_run(runs: list[list[int]], i: int) -> None:
    """Absorb run i: into the previous run (which then also takes the next run, whose team is the
    previous run's by alternation); a first run goes into the next one."""
    run = runs[i]
    if i > 0:
        prev = runs[i - 1]
        prev[1] = run[1]
        del runs[i]
        if i < len(runs) and runs[i][2] == prev[2]:
            prev[1] = runs[i][1]
            del runs[i]
    else:
        runs[1][0] = run[0]
        del runs[0]


def absorb_short(labels: np.ndarray, meta: MatchMeta, min_s: float) -> np.ndarray:
    """`labels` with every possession shorter than `min_s` seconds absorbed (see the module)."""
    min_frames = min_s * meta.fps
    out = labels.copy()
    for runs in _stretches(labels, meta):
        while len(runs) > 1:
            weak = [(r[1] - r[0], i) for i, r in enumerate(runs) if r[1] - r[0] < min_frames]
            if not weak:
                break
            _merge_run(runs, min(weak)[1])
        for a, b, c in runs:
            out[a:b] = c
    return out


@POSTPROCESS.register("min_spell", version="1", example={"min_s": 1.5})
def min_spell(labels: np.ndarray, bundle: MatchBundle, *, min_s: float = 1.5) -> np.ndarray:
    """Absorb possessions shorter than `min_s` seconds into the previous possession."""
    if min_s <= 0:
        return labels.copy()
    return absorb_short(labels, bundle.meta, float(min_s)).astype(np.int8)
