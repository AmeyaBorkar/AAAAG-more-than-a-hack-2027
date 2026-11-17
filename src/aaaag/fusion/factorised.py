"""Factorised fusion: P(dead) from the ball-state producers, P(home | alive) from the others.

A possession producer such as `nearest_player` says "home" as a one-hot, which in a joint (H, A, D)
row also claims "certainly alive"; pooling it with a dead-ball producer by `log_pool` or `mean`
would let the possession side veto every stoppage the ball-state side sees. Here each producer
speaks only to its own factor, and the joint is rebuilt with Evidence.from_factors.

v2 (#31) adds `team_first_from`: producers whose team opinion, where they have one, replaces the
pooled team factor instead of being summed into it. The ball-state producer's restarting side is
the case in point. By the Laws (8, 13-17) a restart is taken by one side, so play starts in that
side's possession, while in the second after a restart the player nearest the ball can be the
receiver's marker. Summed as log-odds, a 0.75 lean would never outweigh a nearest-player one-hot.
"""

from __future__ import annotations

import numpy as np

from ..contracts import Evidence, MatchMeta
from ..registry import FUSION

TIE_TOL = 1e-6  # |P(home | alive) - 0.5| at or below this is a tie, i.e. no team opinion


class _WithChannels(np.ndarray):
    """A (T, 3) posterior that also carries named (T,) channels.

    Used only when at least one channel is present. With none, fusion returns the plain ndarray.
    """

    def __new__(cls, data: np.ndarray, channels: dict[str, np.ndarray]) -> _WithChannels:
        obj = np.asarray(data).view(cls)
        obj.channels = channels
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        self.channels = getattr(obj, "channels", {})


def _carry(proba: np.ndarray, channels: dict[str, np.ndarray]) -> np.ndarray:
    """`proba` itself when `channels` is empty, otherwise a view that carries them."""
    if not channels:
        return proba
    return _WithChannels(proba, channels)


@FUSION.register("factorised", version="2", example={"dead_from": [1], "team_first_from": [1]})
def factorised(
    evidence: list[Evidence],
    meta: MatchMeta,
    *,
    dead_from: list[int | str],
    team_from: list[int | str] | None = None,
    team_first_from: list[int | str] | None = None,
    floor: float = 1e-3,
    p_dead_unknown: float = 0.0,
    team_weights: dict[int | str, float] | None = None,
) -> np.ndarray:
    """P(dead): the log-odds sum over the `dead_from` producers that have an opinion.
    P(home | alive): the log-odds sum of P(H) / (P(H) + P(A)) over `team_from` (default: all the
    others). Producers are named by position in the pipeline or by Evidence name. Both factors are
    clipped to [floor, 1 - floor], so no single producer is certain. A frame with no dead opinion
    gets `p_dead_unknown`; a frame with no team opinion gets 0.5 (a tie, which decoders treat as
    no vote); a frame with neither is a NaN row.

    `team_first_from` (default: none): where one of these producers has a team opinion (not NaN,
    not a 0.5 tie), its own P(H) / (P(H) + P(A)) replaces the pooled team factor; when several
    have one, the later in the list wins. It may name a `dead_from` producer: `ballstate` leans
    towards the restarting side for a second after each attributed restart and is 0.5 elsewhere.
    Fails where that producer names the wrong side (on IDSSE dev, about 15% of attributed
    restarts, and corners at chance; #31).

    `team_weights` (default: every producer 1.0): a weight per `team_from` producer on its log-odds
    in the sum, named like the producers. #86 (E13): the learned network at weight 1 plus the
    last-controller rule at 0.1 (a 0.7-nat nudge from a predictor with an unbounded memory) beats
    the network alone on every row; at 0.35 and above it hurts. Unlisted producers keep weight 1.

    `channels["switch"]` on the `team_from` producers, when any of them has it, is averaged (NaN
    skipped per frame) and attached to the returned array. `channels["whistle_onset"]` on any
    producer (E156, `ballstate_whistle`) is attached as `whistle`, its frame-wise maximum. With
    neither, the return is a plain ndarray, the same object the probabilities were built as.
    """
    dead_idx = _indices(evidence, dead_from)
    weight = {i: 1.0 for i in range(len(evidence))}
    for ref, value in (team_weights or {}).items():
        weight[_indices(evidence, [ref])[0]] = float(value)
    team_idx = (
        [i for i in range(len(evidence)) if i not in dead_idx]
        if team_from is None
        else _indices(evidence, team_from)
    )
    first_idx = _indices(evidence, team_first_from or [])
    n = evidence[0].proba.shape[0]
    dead_logit, dead_has = np.zeros(n), np.zeros(n, dtype=bool)
    for i in dead_idx:
        p = evidence[i].proba[:, 2].astype(np.float64)
        has = ~np.isnan(p)
        dead_logit[has] += _logit(p[has], floor)
        dead_has |= has
    team_logit, team_has = np.zeros(n), np.zeros(n, dtype=bool)
    for i in team_idx:
        proba = evidence[i].proba.astype(np.float64)
        alive = proba[:, 0] + proba[:, 1]
        has = ~np.isnan(alive) & (alive > floor)
        team_logit[has] += weight[i] * _logit(proba[has, 0] / alive[has], floor)
        team_has |= has
    p_dead = np.where(dead_has, _sigmoid(dead_logit), p_dead_unknown)
    p_home = np.where(team_has, _sigmoid(team_logit), 0.5)
    for i in first_idx:
        proba = evidence[i].proba.astype(np.float64)
        alive = proba[:, 0] + proba[:, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = proba[:, 0] / alive
        has = ~np.isnan(alive) & (alive > floor) & (np.abs(share - 0.5) > TIE_TOL)
        p_home[has] = np.clip(share[has], floor, 1.0 - floor)
        team_has |= has
    fused = Evidence.from_factors("fused", p_dead, p_home).proba
    fused[~dead_has & ~team_has] = np.nan
    switch = _mean_switch(evidence, team_idx)
    channels = {} if switch is None else {"switch": switch}
    whistle = _any_whistle(evidence)
    if whistle is not None:
        channels["whistle"] = whistle
    return _carry(fused, channels)


def _any_whistle(evidence: list[Evidence]) -> np.ndarray | None:
    """E156: the frame-wise maximum of `channels["whistle_onset"]` over every producer that carries
    it (NaN skipped: 1.0 a whistle onset, 0.0 covered and quiet, NaN no audio). None when nobody
    does, so a pipeline without whistle evidence gets exactly the channels it had."""
    parts = [
        np.asarray(e.channels["whistle_onset"], dtype=np.float32)
        for e in evidence
        if "whistle_onset" in e.channels
    ]
    if not parts:
        return None
    return np.fmax.reduce(np.stack(parts, axis=0), axis=0).astype(np.float32)


def _mean_switch(evidence: list[Evidence], team_idx: list[int]) -> np.ndarray | None:
    """Mean of `channels["switch"]` over the team producers that carry it. None when nobody does."""
    parts = [
        np.asarray(evidence[i].channels["switch"], dtype=np.float64)
        for i in team_idx
        if "switch" in evidence[i].channels
    ]
    if not parts:
        return None
    stacked = np.stack(parts, axis=0)
    count = np.sum(~np.isnan(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    mean = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean.astype(np.float32)


def _indices(evidence: list[Evidence], refs: list[int | str]) -> list[int]:
    names = [e.name for e in evidence]
    out = []
    for ref in refs:
        if isinstance(ref, int) and 0 <= ref < len(evidence):
            out.append(ref)
        elif isinstance(ref, str) and ref in names:
            out.append(names.index(ref))
        else:
            raise ValueError(f"no producer {ref!r} among {names}")
    return out


def _logit(p: np.ndarray, floor: float) -> np.ndarray:
    p = np.clip(p, floor, 1.0 - floor)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
