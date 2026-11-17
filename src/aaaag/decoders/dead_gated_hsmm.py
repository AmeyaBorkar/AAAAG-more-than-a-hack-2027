"""Dead-gated semi-Markov decoder: stoppages from P(D) as `dead_gated_vote` fixes them, then the
explicit-duration segmentation of H and A inside each alive run (#86, E1c).

The 3-state `hsmm` decides alive and dead from the same tempered evidence as the team, and with a
confident network posterior that is the failure of the declared E1 and E1b grids: noisy frame votes
move the stoppage edges and the segmentation follows them. Here the dead mask is decided first, from
P(D) >= `dead_threshold` with the vote's cleaning (dead runs shorter than `min_dead_s` dropped, alive
gaps shorter than `min_alive_s` between two stoppages filled), the posterior is forced to that mask
(dead frames (0, 0, 1); alive frames keep their H/A odds and get P(D) = 1e-6) and `hsmm` runs with a
binding `lam_dead`, so only the team is left to decide inside each alive run: spells of H or A with
a lognormal duration prior, a cost `beta` per emitted spell (the F1/2 rule), no state carried across
a period.

Signals: the fused (H, A, D) posterior, and, when `switch_weight` > 0, the switch logit forwarded
to `hsmm` unchanged (the dead mask does not use it). On the network's posterior (IDSSE dev, leave-one-match-
out, degrade-v5): +0.018 spellF1 over the 0.5 s vote on fresh seeds, 5 of 5 matches, and positive on
all three Metrica games (lam_team 0.6, beta 0, no boundary refinement, the defaults below).

Where it fails: it can only re-segment what the posterior says; `lam_team` is a scale that depends
on how confident the posterior is (0.6 suits the network's, 0.14 the champion's one-hot team
evidence); it predicts about 0.77 of the true spell count, which is the F1/2 rule at work and also
what a coarser truth would reward; a wrong dead mask is final.

Short-spell relief (E74, off by default: `relief_margin` None). The loss anatomy of 7 Nov puts
48% of the wrong-team gap on a spell of one team held through a short true spell of the other.
With `relief_margin` set, after the segmentation each decoded H or A spell is searched for an
interior run of frames whose team log-odds (the same clipped logit P(H) / (P(H) + P(A)) the
decoder reads, before `lam_team`) favour the other team. A run of at least `relief_min_s` whose
mean margin is at least `relief_margin` is relabelled to that team. It never touches a stoppage,
a frame outside the periods, or a spell edge (moving an edge is E42's boundary refinement, which
failed), and it is symmetric in H and A. It acts after the HSMM and before the post-process, so
`controlled_spell` still absorbs a recovered spell shorter than its `hold_s` unless that spell
shows a pass. This is not one of E60's global knobs: it is gated on local evidence only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import A, H, MatchMeta
from ..registry import DECODERS
from . import whistle_fusion
from .dead_gated_vote import _clean
from .hsmm import _segments, hsmm, posterior_logits

GATED_LAM_DEAD = 50.0  # with the dead mask forced in the posterior this makes it binding
ALIVE_P_DEAD = 1e-6


def gate_posterior(
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    dead_threshold: float = 0.5,
    min_dead_s: float = 2.0,
    min_alive_s: float = 1.0,
    exit_threshold: float | None = None,
    whistle: np.ndarray | None = None,
    whistle_bonus: float = 0.0,
    whistle_switch_mult: float = 1.0,
    whistle_pre_s: float = 0.4,
    whistle_post_s: float = 2.0,
    whistle_switch_s: float = 0.5,
) -> np.ndarray:
    """A copy of `proba` with the dead mask fixed as `dead_gated_vote` fixes it (see the module
    docstring). NaN H/A rows on alive frames stay NaN (no opinion on the team).

    Whistles (E156, off by default): with a `whistle` channel, `whistle_bonus` > 0 adds that many
    nats to logit P(D) in [t - whistle_pre_s, t + whistle_post_s] around each whistle before the
    threshold (F1), and `whistle_switch_mult` < 1 scales the cleaning's minimum lengths for runs
    that start or end within +-whistle_switch_s of a whistle (F2); see `whistle_fusion`.

    `exit_threshold` (E56, off by default) trims the tail of each cleaned stoppage back to its
    last frame with P(D) >= `exit_threshold`, so the restart edge comes where the network stops
    being fairly sure the ball is dead rather than where it stops being slightly sure. The onset
    edge is untouched. A stoppage with no such frame keeps its length."""
    out = proba.copy()
    fps = meta.fps
    onsets = whistle_fusion.onset_frames(whistle)
    use_bonus = len(onsets) > 0 and whistle_bonus != 0.0
    use_switch = len(onsets) > 0 and whistle_switch_mult != 1.0
    source = proba
    if use_bonus:
        source = whistle_fusion.dead_bonus(
            proba, onsets, fps, bonus=whistle_bonus, pre_s=whistle_pre_s, post_s=whistle_post_s
        )  # only the dead mask reads it: alive frames keep the H/A odds of `proba`
    p_dead = np.nan_to_num(source[:, 2], nan=0.0)
    with np.errstate(invalid="ignore"):
        dead_all = p_dead >= dead_threshold
    min_dead, min_alive = int(round(min_dead_s * fps)), int(round(min_alive_s * fps))
    half = int(round(whistle_switch_s * fps))
    for start, stop in meta.runs():
        if use_switch:
            dead = whistle_fusion.clean_near(
                dead_all[start:stop],
                min_dead,
                min_alive,
                onsets=onsets - start,
                half=half,
                mult=whistle_switch_mult,
            )
        else:
            dead = _clean(dead_all[start:stop], min_dead, min_alive)
        if exit_threshold is not None:
            dead = _trim_tails(dead, p_dead[start:stop] >= exit_threshold)
        seg = out[start:stop]
        alive = ~dead
        seg[dead] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ha = seg[alive, 0] + seg[alive, 1]
        scale = np.where(ha > 0, (1.0 - ALIVE_P_DEAD) / np.where(ha > 0, ha, 1.0), 1.0)
        seg[alive, 0] *= scale
        seg[alive, 1] *= scale
        seg[alive, 2] = ALIVE_P_DEAD
    return out


def _trim_tails(dead: np.ndarray, sure: np.ndarray) -> np.ndarray:
    """`dead` with each run's frames after its last `sure` frame set alive."""
    out = dead.copy()
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        hits = np.flatnonzero(sure[a:b])
        if len(hits):
            out[a + hits[-1] + 1 : b] = False
    return out


def relieve_short_spells(
    labels: np.ndarray,
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    margin: float,
    min_s: float = 0.5,
) -> np.ndarray:
    """A copy of `labels` with strongly evidenced interior runs of the other team let through (E74).

    For each H or A spell, per period: the maximal runs of frames strictly inside the spell whose
    team log-odds favour the other team (logit P(H) / (P(H) + P(A)) from `posterior_logits`, so
    clipped to +/- 6.9 and 0 where there is no team opinion; a 0 frame ends a run). A run of at
    least `min_s` (rounded to frames, at least 1) whose mean log-odds margin for the other team is
    at least `margin` becomes that team. D and X frames, and the first and last frame of every
    spell, never change."""
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError(f"relief_margin must be a finite number > 0, got {margin!r}")
    if not np.isfinite(min_s) or min_s <= 0.0:
        raise ValueError(f"relief_min_s must be a finite number > 0, got {min_s!r}")
    out = labels.copy()
    min_frames = max(1, int(round(min_s * meta.fps)))
    _, lt = posterior_logits(proba)
    for start, stop in meta.runs():
        for lo, hi, code in _segments(labels[start:stop]):
            if code not in (H, A) or hi - lo < min_frames + 2:
                continue
            a, b = start + lo + 1, start + hi - 1  # interior frames only
            other = A if code == H else H
            favour = -lt[a:b] if code == H else lt[a:b]  # > 0: the frame prefers `other`
            edges = np.diff(np.concatenate(([0], (favour > 0).astype(np.int8), [0])))
            for r0, r1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
                if r1 - r0 >= min_frames and float(favour[r0:r1].mean()) >= margin:
                    out[a + r0 : a + r1] = other
    return out


@DECODERS.register(
    "dead_gated_hsmm",
    version="1",
    example={"lam_team": 0.6, "beta": 0.0, "refine_s": 0.0},
)
def dead_gated_hsmm(
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    priors: dict | str | Path | None = None,
    lam_team: float = 0.6,
    beta: float = 0.0,
    dead_threshold: float = 0.5,
    min_dead_s: float = 2.0,
    min_alive_s: float = 1.0,
    block: int = 5,
    duration_prior: str = "lognormal",
    max_alive_s: float = 40.0,
    max_dead_s: float = 100.0,
    self_prob: float = 0.02,
    refine_s: float = 0.0,
    refine_mode: str = "factorised",
    min_spell_s: float = 0.0,
    outside: str = "X",
    switch_weight: float = 0.0,
    switch: np.ndarray | None = None,
    exit_threshold: float | None = None,
    relief_margin: float | None = None,
    relief_min_s: float = 0.5,
    whistle: np.ndarray | None = None,
    whistle_bonus: float = 0.0,
    whistle_switch_mult: float = 1.0,
    whistle_pre_s: float = 0.4,
    whistle_post_s: float = 2.0,
    whistle_switch_s: float = 0.5,
) -> np.ndarray:
    """(T, 3) posterior over (H, A, D) to (T,) int8 labels: the dead mask of `dead_gated_vote`, then
    `hsmm` on the forced posterior. Defaults are the E1c selection (lam_team 0.6, beta 0, lognormal
    duration prior, no boundary refinement, no minimum spell). X outside the periods.

    `relief_margin` (E74, default None: off, the labels are the call without it) runs
    `relieve_short_spells` on the result with `relief_min_s`.

    `whistle` (E156, default None) is the whistle channel (1.0 on onset frames, from the fusion's
    `whistle` channel); `whistle_bonus` (F1, default 0) and `whistle_switch_mult` (F2, default 1)
    act on the dead gate only (`gate_posterior`). Off, the labels are the call without them."""
    gated = gate_posterior(
        proba,
        meta,
        dead_threshold=dead_threshold,
        min_dead_s=min_dead_s,
        min_alive_s=min_alive_s,
        exit_threshold=exit_threshold,
        whistle=whistle,
        whistle_bonus=whistle_bonus,
        whistle_switch_mult=whistle_switch_mult,
        whistle_pre_s=whistle_pre_s,
        whistle_post_s=whistle_post_s,
        whistle_switch_s=whistle_switch_s,
    )
    labels = hsmm(
        gated,
        meta,
        priors=priors,
        lam_team=lam_team,
        lam_dead=GATED_LAM_DEAD,
        beta=beta,
        block=block,
        duration_prior=duration_prior,
        max_alive_s=max_alive_s,
        max_dead_s=max_dead_s,
        self_prob=self_prob,
        refine_s=refine_s,
        refine_mode=refine_mode,
        min_spell_s=min_spell_s,
        outside=outside,
        switch_weight=switch_weight,
        switch=switch,
    )
    if relief_margin is None:
        return labels
    return relieve_short_spells(
        labels, gated, meta, margin=float(relief_margin), min_s=float(relief_min_s)
    )
