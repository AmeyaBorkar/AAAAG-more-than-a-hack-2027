"""Snap the edges of each predicted stoppage to the missing-ball run that overlaps it (#66).

On this provider's tracking (Metrica, which made the BAR-ATH ATD) the ball is blanked through dead
time: it disappears one frame after a stoppage starts and returns exactly at the restart (#50, fitted
on the three Metrica games only). So the edges of an unseen-ball run are the stoppage edges, to
about a frame, while the HMM's tempering and the decoder's minimum durations blur them by about half a
second (#62).

For each run of P(dead) >= 0.5 inside a period, the unseen-ball runs that share a frame with it are
taken together. The run's start moves to (their earliest start - `onset_lag` frames) and its end to
their latest end, where the ball is back. Each edge moves only if the move is at most `max_s`.
Frames a run gains get the run's median P(dead); frames it gives up get the P(dead) of the alive
frame just outside its old edge. Edges at a period boundary never move. Runs are processed in order,
so where two runs' moves touch the same frames, the later run wins.

Uses: P(dead) and where the ball is seen. Fails where the provider keeps the ball in a stoppage, where
a live dropout touches a stoppage (the edge moves to the dropout, bounded by `max_s`), and on
tracking that does not blank the ball at all (nothing moves: an unseen run never overlaps).
"""

from __future__ import annotations

import numpy as np

from ...contracts import MatchMeta
from .cues import runs

DEAD = 0.5  # the decoders' dead threshold on P(dead)


def snap_to_ball_gaps(
    p_dead: np.ndarray,
    seen: np.ndarray,
    meta: MatchMeta,
    *,
    max_s: float,
    onset_lag: int = 1,
) -> np.ndarray:
    """(T,) P(dead) with each stoppage's edges moved onto the overlapping unseen-ball runs."""
    out = np.array(p_dead, dtype=np.float64, copy=True)
    bound = int(round(max_s * meta.fps))
    for period in meta.periods:
        lo, hi = period.start, period.stop
        p = np.asarray(p_dead[lo:hi], dtype=np.float64)
        new = out[lo:hi]
        n = hi - lo
        d_starts, d_stops = runs(np.nan_to_num(p, nan=0.0) >= DEAD)
        g_starts, g_stops = runs(~np.asarray(seen[lo:hi], dtype=bool))
        if not len(d_starts) or not len(g_starts):
            continue
        for start, stop in zip(d_starts, d_stops, strict=True):
            overlap = (g_starts < stop) & (g_stops > start)
            if not overlap.any():
                continue
            gap_start, gap_stop = int(g_starts[overlap].min()), int(g_stops[overlap].max())
            value = float(np.median(p[start:stop]))
            new_start = max(gap_start - onset_lag, 0)
            if 0 < start and new_start != start and abs(new_start - start) <= bound:
                if new_start < start:
                    new[new_start:start] = value
                else:
                    new[start:new_start] = p[start - 1]
            if stop < n and gap_stop != stop and abs(gap_stop - stop) <= bound:
                if gap_stop > stop:
                    new[stop:gap_stop] = value
                else:
                    new[gap_stop:stop] = p[stop]
    return out
