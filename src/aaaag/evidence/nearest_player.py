"""Nearest tracked player to the ball -- the official baseline's per-frame step, vectorised."""

from __future__ import annotations

import numpy as np

from ..contracts import D, Evidence, MatchBundle, Tracking
from ..registry import EVIDENCE

CHUNK = 16384  # frames per block: bounds the (B, S) distance matrix to a few MB


@EVIDENCE.register("nearest_player", version="1")
def nearest_player(
    bundle: MatchBundle,
    *,
    when_unseen: str = "dead",
    max_distance_m: float | None = None,
) -> Evidence:
    """One-hot team of the slot nearest the ball.

    Uses: ball position, every observed slot position. Fails when the ball is not seen (41% of
    graded frames), in flight between players, contested, or when the tracker lost the carrier.

    when_unseen: "dead" reproduces the official baseline (no ball -> D); "abstain" returns no
    opinion instead, which is what a temporal decoder should usually be given.
    max_distance_m: frames whose nearest slot is farther than this count as unseen.
    """
    codes = nearest_team_codes(bundle.tracking, max_distance_m)
    if when_unseen == "dead":
        codes = np.where(codes < 0, D, codes).astype(np.int8)
    elif when_unseen != "abstain":
        raise ValueError(f"when_unseen must be 'dead' or 'abstain', got {when_unseen!r}")
    return Evidence.from_codes("nearest_player", codes)


def nearest_team_codes(tracking: Tracking, max_distance_m: float | None = None) -> np.ndarray:
    """(T,) int8: team of the nearest slot where the ball and at least one slot are seen, else -1.

    Ties go to the lowest slot index, like the official baseline. Distances are computed in
    float64 metres.
    """
    out = np.full(tracking.n_frames, -1, dtype=np.int8)
    seen = np.flatnonzero(tracking.ball_seen())
    for begin in range(0, len(seen), CHUNK):
        rows = seen[begin : begin + CHUNK]
        ball = tracking.ball_xy[rows].astype(np.float64)
        slots = tracking.player_xy[rows].astype(np.float64)
        d2 = ((slots - ball[:, None, :]) ** 2).sum(axis=2)
        d2[np.isnan(d2)] = np.inf
        best = d2.argmin(axis=1)
        best_d2 = d2[np.arange(len(rows)), best]
        ok = np.isfinite(best_d2)
        if max_distance_m is not None:
            ok &= best_d2 <= max_distance_m**2
        out[rows[ok]] = tracking.slot_team[best[ok]]
    return out
