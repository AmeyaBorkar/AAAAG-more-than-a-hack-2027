"""Referee whistles fused into the dead gate of `dead_gated_hsmm` (E156, #86). Off by default.

A whistle is a ball-state boundary cue: referees whistle most stoppages and some restarts. The
detector cannot tell the two apart, so both fusions below are symmetric in that sense. They act
only on the dead mask the gated decoder fixes before its team segmentation; the H/A odds of every
frame are left as they are.

- F1, dead log-odds bonus: `bonus` nats are added to logit P(D) on every frame in
  [t - pre_s, t + post_s] around each whistle t (overlapping windows get it once). P(H) and P(A)
  are rescaled so the row still sums to 1, which keeps their ratio.
- F2, ball-state switch-cost multiplier: in the gated decoder the cost of a ball-state change is
  the minimum length the gate's cleaning asks of the new state (`min_dead_s` for a stoppage,
  `min_alive_s` for an alive gap between two stoppages). A run that starts or ends within
  +-`switch_s` of a whistle needs only `mult` times that length.

Input is the whistle channel of `ballstate_whistle`: 1.0 on a whistle's onset frame, 0.0 on other
covered frames, NaN where the audio says nothing. No whistle, `bonus` 0 and `mult` 1 leave the
decoder's output byte-identical (the caller skips these functions).

Where it fails: whistles not tied to a boundary (advantage, fouls waved on, whistling fans), a
restart whistle under F1 (the bonus extends the stoppage past the restart), and an audio/video
offset larger than the windows.
"""

from __future__ import annotations

import numpy as np


def onset_frames(whistle: np.ndarray | None) -> np.ndarray:
    """(N,) int64 sorted frames where the whistle channel is 1.0 (NaN and 0.0 are not whistles)."""
    if whistle is None:
        return np.zeros(0, dtype=np.int64)
    w = np.asarray(whistle, dtype=np.float64)
    return np.flatnonzero(np.nan_to_num(w, nan=0.0) >= 0.5).astype(np.int64)


def window_mask(n: int, onsets: np.ndarray, before: int, after: int) -> np.ndarray:
    """(n,) bool: frames in [t - before, t + after] (inclusive) of any onset t, clipped to [0, n)."""
    diff = np.zeros(n + 1, dtype=np.int64)
    if len(onsets):
        lo = np.clip(onsets - before, 0, n)
        hi = np.clip(onsets + after + 1, 0, n)
        keep = hi > lo
        np.add.at(diff, lo[keep], 1)
        np.add.at(diff, hi[keep], -1)
    return np.cumsum(diff[:n]) > 0


def dead_bonus(
    proba: np.ndarray, onsets: np.ndarray, fps: float, *, bonus: float, pre_s: float, post_s: float
) -> np.ndarray:
    """A copy of the (T, 3) posterior with `bonus` added to logit P(D) in each whistle window (F1).

    P(D) = 0 or 1 stays where it is; NaN rows stay NaN."""
    out = np.array(proba, dtype=np.float32, copy=True)
    mask = window_mask(len(out), onsets, int(round(pre_s * fps)), int(round(post_s * fps)))
    if not mask.any() or bonus == 0.0:
        return out
    p = out[mask].astype(np.float64)
    pd_ = p[:, 2]
    up = pd_ * np.exp(bonus)
    new = up / (up + (1.0 - pd_))
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(pd_ < 1.0, (1.0 - new) / (1.0 - pd_), 1.0)
    p[:, 0] *= scale
    p[:, 1] *= scale
    p[:, 2] = new
    out[mask] = p.astype(np.float32)
    return out


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _cover(n: int, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """(n,) bool: the union of the half-open runs [starts, stops)."""
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, stops, -1)
    return np.cumsum(diff[:n]) > 0


def clean_near(
    dead: np.ndarray,
    min_dead: int,
    min_alive: int,
    *,
    onsets: np.ndarray,
    half: int,
    mult: float,
) -> np.ndarray:
    """The gate's cleaning (`dead_gated_vote._clean`) with F2: dead runs shorter than their need
    are dropped, then interior alive gaps shorter than theirs are filled. The need is `min_dead`
    (`min_alive`) frames, times `mult` (rounded) for a run whose first frame or whose end (the
    first frame after it) lies within +-`half` frames of a whistle onset. `onsets` are in the
    coordinates of `dead`."""
    n = len(dead)
    near = window_mask(n + 1, onsets, half, half)  # boundary positions 0..n
    starts, stops = _runs(dead)
    need = np.where(near[starts] | near[stops], int(round(min_dead * mult)), min_dead)
    short = stops - starts < need
    dead = dead & ~_cover(n, starts[short], stops[short])
    starts, stops = _runs(~dead)
    need = np.where(near[starts] | near[stops], int(round(min_alive * mult)), min_alive)
    fill = (starts > 0) & (stops < n) & (stops - starts < need)
    return dead | _cover(n, starts[fill], stops[fill])
