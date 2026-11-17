"""Restart edges traced back from the reappearing ball (E68, #86; research queue rank 9).

When the tracker loses the ball through a stoppage and finds it again only after the restart, the
decoded restart edge comes late, which truncates the possession that follows. The spellF1 credit
punishes truncation far more than an equal extension. This step looks at each decoded restart: a
stoppage of at least `min_dead_s` followed by H or A. It takes the first sighting of the ball after
a gap of at least `min_gap_s`, within `before_s` before and `after_s` after the edge. If the ball
leaves that sighting at `speed_mps` or more, its path is traced back at that constant velocity, up
to `max_flight_s`, until it passes within `reach_m` of a player of the restarting team at the same
moment. That moment is the kick. If the kick comes before the decoded edge, the edge moves back to
it, never by more than the trace and never leaving less than `keep_dead_s` of the stoppage.

Constant velocity overstates the flight of a decelerating ball, and a smoothed ball leaves a
sighting slower than it really is. Both errors put the kick early, the cheap direction. The
constants are physics and round values, and nothing is fitted.

Uses: the decoded labels, the ball and the player tracks with their teams. Fails when the first
sighting is a pass received after the restart (the ball is slow, so nothing moves), when the kicker
is unseen, or when a teammate stands on the traced path before the real kicker (the kick comes out
late, and nothing moves).
"""

from __future__ import annotations

import numpy as np

from ..contracts import A, D, H, MatchBundle
from ..registry import POSTPROCESS

VELOCITY_FRAMES = 4  # the leaving velocity is measured over this many frames after the sighting


def _restarts(labels: np.ndarray, start: int, stop: int, min_dead: int) -> list[tuple[int, int]]:
    """(stoppage start, restart frame) of D runs of at least `min_dead` frames followed by H/A."""
    x = labels[start:stop]
    b = np.flatnonzero(x[1:] != x[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(x)]))
    out = []
    for i in range(len(bounds) - 1):
        a, z = int(bounds[i]), int(bounds[i + 1])
        if x[a] == D and z - a >= min_dead and z < len(x) and x[z] in (H, A):
            out.append((start + a, start + z))
    return out


def trace_kick(
    ball: np.ndarray,
    players: np.ndarray,
    t_seen: int,
    fps: int,
    *,
    speed_mps: float,
    max_flight_s: float,
    reach_m: float,
) -> int | None:
    """The frame of the kick traced back from the sighting at `t_seen`, or None.

    `ball` is (T, 2) and `players` (T, k, 2) for the restarting team only."""
    end = t_seen + VELOCITY_FRAMES
    if end >= len(ball) or not np.isfinite(ball[t_seen : end + 1]).all():
        return None
    v = (ball[end] - ball[t_seen]) * fps / VELOCITY_FRAMES
    if np.hypot(*v) < speed_mps:
        return None
    lags = np.arange(1, min(int(round(max_flight_s * fps)), t_seen) + 1)
    if not len(lags):
        return None
    path = ball[t_seen] - v[None, :] * (lags[:, None] / fps)  # (L, 2)
    near = players[t_seen - lags]  # (L, k, 2)
    with np.errstate(invalid="ignore"):
        dist = np.hypot(near[..., 0] - path[:, None, 0], near[..., 1] - path[:, None, 1])
        hit = np.flatnonzero((np.nan_to_num(dist, nan=np.inf) <= reach_m).any(axis=1))
    return int(t_seen - lags[hit[0]]) if len(hit) else None


@POSTPROCESS.register("restart_backtrack", version="1", example={})
def restart_backtrack(
    labels: np.ndarray,
    bundle: MatchBundle,
    *,
    min_dead_s: float = 2.0,
    before_s: float = 1.0,
    after_s: float = 2.0,
    min_gap_s: float = 0.4,
    speed_mps: float = 3.0,
    max_flight_s: float = 2.5,
    reach_m: float = 1.5,
    keep_dead_s: float = 1.0,
) -> np.ndarray:
    """Late restart edges moved back to the kick traced from the ball (see the module docstring)."""
    tracking = bundle.tracking
    fps = tracking.meta.fps
    out = np.array(labels, copy=True)
    ball = tracking.ball_xy.astype(np.float64)
    seen = np.isfinite(ball).all(axis=1)
    team_xy = {
        team: tracking.player_xy[:, tracking.slot_team == team].astype(np.float64)
        for team in (H, A)
    }
    gap = max(1, int(round(min_gap_s * fps)))
    for period in tracking.meta.periods:
        for a, r in _restarts(out, period.start, period.stop, int(round(min_dead_s * fps))):
            lo = max(a + gap, r - int(round(before_s * fps)))
            hi = min(period.stop, r + int(round(after_s * fps)) + 1)
            first = next(
                (t for t in range(lo, hi) if seen[t] and not seen[t - gap : t].any()), None
            )
            if first is None:
                continue
            team = int(out[r])
            kick = trace_kick(
                ball,
                team_xy[team],
                first,
                fps,
                speed_mps=speed_mps,
                max_flight_s=max_flight_s,
                reach_m=reach_m,
            )
            if kick is None or kick >= r:
                continue
            kick = max(kick, a + int(round(keep_dead_s * fps)))
            if kick < r:
                out[kick:r] = team
    return out
