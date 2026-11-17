"""Dead-gated majority vote: stoppages from P(D), then the baseline's team vote inside alive play.

`majority_vote` treats a dead frame as an abstention and labels D only where a whole window has
no team votes, so every stoppage shrinks by `half_window` at each end and short ones vanish. Here
the dead mask is decided first, from P(D), and the team vote runs only inside each alive stretch,
so votes never leak across a stoppage and its edges stay where the evidence put them.
"""

from __future__ import annotations

import numpy as np

from ..contracts import A, D, H, MatchMeta, X
from ..registry import DECODERS


@DECODERS.register("dead_gated_vote", version="1", example={"half_window": 25})
def dead_gated_vote(
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    half_window: int,
    dead_threshold: float = 0.5,
    min_dead_s: float = 1.0,
    min_alive_s: float = 1.0,
    outside: str = "X",
) -> np.ndarray:
    """Per period:
    1. dead where P(D) >= `dead_threshold` (NaN rows count as alive); dead runs shorter than
       `min_dead_s` are dropped, then alive gaps shorter than `min_alive_s` between two stoppages
       are filled;
    2. inside each alive run, every frame takes the majority team among the strict votes
       (P(H) > P(A) or the reverse) within +/- half_window, clipped to the run; ties go to H;
    3. frames without a vote in reach copy the nearest voted frame of the same run (earlier first);
       a run with no vote at all takes the sign of its summed P(H) - P(A), else the previous run's
       team, else H.
    O(T) via cumulative sums, plus a loop over runs (hundreds), not frames.
    """
    fps = meta.fps
    with np.errstate(invalid="ignore"):
        dead = np.nan_to_num(proba[:, 2], nan=0.0) >= dead_threshold
        home_vote = np.nan_to_num(proba[:, 0], nan=0.0) > np.nan_to_num(proba[:, 1], nan=0.0)
        away_vote = np.nan_to_num(proba[:, 1], nan=0.0) > np.nan_to_num(proba[:, 0], nan=0.0)
    margin = np.nan_to_num(proba[:, 0] - proba[:, 1], nan=0.0)
    labels = np.full(len(proba), D, dtype=np.int8)
    for start, stop in meta.runs():
        mask = _clean(dead[start:stop], int(round(min_dead_s * fps)), int(round(min_alive_s * fps)))
        labels[start:stop] = _decode_period(
            mask, home_vote[start:stop], away_vote[start:stop], margin[start:stop], half_window
        )
    if outside == "X":
        labels[~meta.played_mask()] = X
    elif outside != "keep":
        raise ValueError(f"outside must be 'X' or 'keep', got {outside!r}")
    return labels


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _clean(dead: np.ndarray, min_dead: int, min_alive: int) -> np.ndarray:
    dead = dead.copy()
    for start, stop in zip(*_runs(dead), strict=True):
        if stop - start < min_dead:
            dead[start:stop] = False
    for start, stop in zip(*_runs(~dead), strict=True):
        if 0 < start and stop < len(dead) and stop - start < min_alive:
            dead[start:stop] = True
    return dead


def _decode_period(dead, home, away, margin, half_window) -> np.ndarray:
    n = len(dead)
    out = np.full(n, D, dtype=np.int8)
    starts, stops = _runs(~dead)
    if not len(starts):
        return out
    alive = ~dead
    home, away = home & alive, away & alive
    # each alive frame's run bounds, so the vote window never crosses a stoppage
    run_start = np.zeros(n, dtype=np.int64)
    run_stop = np.zeros(n, dtype=np.int64)
    for start, stop in zip(starts, stops, strict=True):
        run_start[start:stop], run_stop[start:stop] = start, stop
    index = np.arange(n)
    w = max(0, int(half_window))
    low = np.maximum(run_start, index - w)
    high = np.minimum(run_stop, index + w + 1)
    home_cum = np.concatenate(([0], np.cumsum(home)))
    away_cum = np.concatenate(([0], np.cumsum(away)))
    h = home_cum[high] - home_cum[low]
    a = away_cum[high] - away_cum[low]
    voted = alive & ((h + a) > 0)
    team = np.where(h >= a, H, A).astype(np.int8)

    big = n + 1
    prev = np.maximum.accumulate(np.where(voted, index, -1))
    nxt = np.minimum.accumulate(np.where(voted, index, big)[::-1])[::-1]
    use_prev = alive & (prev >= run_start) & (prev >= 0)
    use_next = alive & ~use_prev & (nxt < run_stop)
    out[use_prev] = team[prev[use_prev]]
    out[use_next] = team[nxt[use_next]]

    previous = None
    for start, stop in zip(starts, stops, strict=True):
        if not voted[start:stop].any():
            total = margin[start:stop].sum()
            fill = H if total > 0 else A if total < 0 else (previous if previous is not None else H)
            out[start:stop] = fill
        previous = int(out[stop - 1])
    return out
