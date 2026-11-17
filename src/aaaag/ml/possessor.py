"""Algorithmic possessor targets for the optional TeamNet head.

For each frame the target is one token slot in that variant's own order: the valid player of the
labelled team nearest the clean ball, or the nobody slot when the ball is dead. A frame whose
nearest teammate is farther than `reach_m`, whose clean ball is missing, whose label is X, or
which is not played, is -1 and the loss ignores it.

Signals: token X/Y/VALID (attack-normalised, as `build_match_tokens` writes them), the clean ball,
the attack-direction sign, the label and the played mask. No player identity, so a slot permutation
from the degrader needs no mapping: coupled and decoupled tokens are scored on their own.
Where it fails: the nearest player inside the reach is not always the possessor (a presser, a ball
in flight), and a distance above the reach teaches the head nothing. Ties take the lowest slot.
"""

from __future__ import annotations

import numpy as np

from ..contracts import A, D, H, MatchMeta
from .spec import Col, TokenSpec


def possessor_targets(
    tokens: np.ndarray,
    labels: np.ndarray,
    ball_xy_clean: np.ndarray,
    sign: np.ndarray,
    meta: MatchMeta,
    spec: TokenSpec,
    reach_m: float,
    team_agnostic: bool = False,
) -> np.ndarray:
    """(T,) int16. Home slots are 0..K-1, away slots K..2K-1, dead is 2K, and -1 is ignored.

    `team_agnostic` (label-free extra matches, E50): the target is the nearest observed slot of
    either team within `reach_m` of the clean ball, ignoring the labels; there is no dead target."""
    tokens = np.asarray(tokens)
    labels = np.asarray(labels)
    k = int(spec.tokens_per_team)
    n = int(tokens.shape[0])
    half_l = float(meta.pitch_length) / 2.0
    half_w = float(meta.pitch_width) / 2.0
    block = tokens[:, : 2 * k]
    xy = np.empty((n, 2 * k, 2), dtype=np.float64)
    xy[..., 0] = np.asarray(block[..., Col.X], dtype=np.float64) * half_l
    xy[..., 1] = np.asarray(block[..., Col.Y], dtype=np.float64) * half_w
    valid = np.asarray(block[..., Col.VALID], dtype=np.float64) > 0.5
    sign_1d = np.asarray(sign, dtype=np.float64).reshape(-1)
    ball = np.asarray(ball_xy_clean, dtype=np.float64).reshape(-1, 2) * sign_1d[:, None]
    ball_ok = np.isfinite(ball).all(axis=-1)
    dist = np.hypot(xy[..., 0] - ball[:, None, 0], xy[..., 1] - ball[:, None, 1])
    dist = np.where(valid & np.isfinite(dist), dist, np.inf)

    home_dist = dist[:, :k]
    away_dist = dist[:, k:]
    home_slot = np.argmin(home_dist, axis=1)
    away_slot = np.argmin(away_dist, axis=1)
    rows = np.arange(n)
    home_best = home_dist[rows, home_slot]
    away_best = away_dist[rows, away_slot]

    played = np.asarray(meta.played_mask(), dtype=bool)
    out = np.full(n, -1, dtype=np.int16)
    if team_agnostic:
        pick_h = played & ball_ok & (home_best <= reach_m) & (home_best <= away_best)
        pick_a = played & ball_ok & (away_best <= reach_m) & (away_best < home_best)
        out[pick_h] = home_slot[pick_h]
        out[pick_a] = away_slot[pick_a] + k
        return out
    pick_h = (labels == H) & played & ball_ok & (home_best <= reach_m)
    pick_a = (labels == A) & played & ball_ok & (away_best <= reach_m)
    out[pick_h] = home_slot[pick_h]
    out[pick_a] = away_slot[pick_a] + k
    out[(labels == D) & played] = np.int16(2 * k)
    return out
