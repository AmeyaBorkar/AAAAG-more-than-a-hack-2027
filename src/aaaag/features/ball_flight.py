"""Ball flight gate (E127, #86): hide the tracked ball where it moves faster than a ball can.

A single tactical camera places an airborne ball where the camera ray through it meets the grass, so
the tracked ground point of a lofted ball runs ahead of the real ball and jumps when it lands.
A *jump* is a step between two seen ball frames (at most `gap_s` apart, inside one period) faster
than `fast_ms` (40 m/s, E115's constant). The gate hides every seen frame within `pad_s` of a jump's
two ends. With `merge_s`, consecutive hidden runs less than `merge_s` apart are joined while the
joined run stays within `max_s` (4 s: the flight time of a 20 m apex), so a kick and its landing
hide the flight between them.

Signals: `ball_xy` only (the same frames of `ball_extra` channels are hidden too). Physics constants,
never fitted to the competition match. Fails where a genuinely fast shot (over 40 m/s) is hidden, and where a
lofted ball moves slowly in projection and never jumps. Off unless asked for: an empty gate returns
the bundle unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from aaaag.contracts import Tracking

FAST_MS = 40.0
GAP_S = 1.0


def flight_mask(
    tracking: Tracking,
    fast_ms: float = FAST_MS,
    gap_s: float = GAP_S,
    pad_s: float = 0.2,
    merge_s: float = 0.0,
    max_s: float = 4.0,
) -> np.ndarray:
    """(T,) bool: the seen ball frames the gate hides."""
    fps = tracking.meta.fps
    T = tracking.meta.n_frames
    ball = tracking.ball_xy.astype(np.float64)
    seen_mask = np.isfinite(ball[:, 0])
    pad = int(round(pad_s * fps))
    diff = np.zeros(T + 1, dtype=np.int64)
    for period in tracking.meta.periods:
        seen = np.flatnonzero(seen_mask[period.start : period.stop]) + period.start
        if len(seen) < 2:
            continue
        gap = np.diff(seen)
        step = np.linalg.norm(ball[seen[1:]] - ball[seen[:-1]], axis=1)
        idx = np.flatnonzero((gap <= gap_s * fps) & (step / (gap / fps) > fast_ms))
        if not len(idx):
            continue
        lo = np.maximum(seen[idx] - pad, period.start)
        hi = np.minimum(seen[idx + 1] + pad + 1, period.stop)
        np.add.at(diff, lo, 1)
        np.add.at(diff, hi, -1)
    mask = np.cumsum(diff[:T]) > 0
    if merge_s > 0 and mask.any():
        edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        period_of = np.zeros(T, dtype=np.int64)
        for k, period in enumerate(tracking.meta.periods):
            period_of[period.start : period.stop] = k
        merge_f, max_f = merge_s * fps, max_s * fps
        g0, g1 = starts[0], stops[0]
        for s, e in zip(starts[1:], stops[1:], strict=True):
            joinable = s - g1 < merge_f and e - g0 <= max_f and period_of[s] == period_of[g0]
            if joinable:
                mask[g1:s] = True
                g1 = e
            else:
                g0, g1 = s, e
    return mask & seen_mask


def gated(tracking: Tracking, **params) -> Tracking:
    """`tracking` with the ball hidden on `flight_mask(tracking, **params)`; unchanged if no params."""
    if not params:
        return tracking
    mask = flight_mask(tracking, **params)
    if not mask.any():
        return tracking
    ball = tracking.ball_xy.copy()
    ball[mask] = np.nan
    extra = {}
    for name, channel in tracking.ball_extra.items():
        channel = channel.copy()
        channel[mask] = np.nan
        extra[name] = channel
    return dataclasses.replace(
        tracking, ball_xy=ball, ball_extra=extra, _fingerprint=None
    ).validate()


def gated_bundle(bundle, **params):
    """A bundle whose tracking is `gated(bundle.tracking, **params)`."""
    if not params:
        return bundle
    tracking = gated(bundle.tracking, **params)
    return bundle if tracking is bundle.tracking else dataclasses.replace(bundle, tracking=tracking)
