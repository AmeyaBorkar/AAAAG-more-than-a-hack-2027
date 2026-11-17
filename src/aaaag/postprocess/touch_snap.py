"""Gaining-team first-touch snap: move each H<->A boundary onto the nearest ball touch of the team
that gains the ball (#86, from the WF1 event-timing study).

The provider flips possession on the frame of the new team's first kinematic touch (flip minus
touch: median 0.00 s, IQR [0.00, +0.04 s], 76% of turnovers; a placebo gets 22%). A decoder works on
the fused posterior and puts a turnover where the evidence turns, which is typically a few frames
off that touch. This step re-times only the turnovers: every boundary between two alive spells of
different teams moves to the nearest touch of the team that takes the ball within +-`window_s`
(order kept, one frame per segment at least); stoppage edges and everything else stay.

Touch detector (`detector="w2"`, the default): at frame t the ball's velocity change
dv(t) = |v_out - v_in| with v_in = (p_t - p_{t-w}) fps / w and v_out = (p_{t+w} - p_t) fps / w
(an acceleration spike, including a pure change of direction). A touch is a local maximum of dv
within +-w frames with dv >= `theta` m/s while a tracked slot is within `reach_m` of the ball; the
touching team is the nearest slot's team.

`detector="g2l"` uses `detect_touches_g2l` instead: a straight line on each side of a guard, and
the impulse is the change of slope. The default is "w2", and then this step's labels are the ones
it has always produced.

Signals: ball and slot positions and the slot teams. On the external matches (IDSSE dev, degrade-v5,
leave-one-match-out) it adds +0.0125 spellF1 on top of the learned candidate, 5 of 5 matches, and
snapping to the nearest touch of ANY team is a tie (-0.0007), which is the negative control.
Where it fails: where the ball is unseen (no touch can be detected inside a gap, and the first w
frames after one have no incoming velocity), on a ball carried at the foot (many small touches), on
tracking noise that reads as acceleration, and wherever the provider's convention is not
"possession flips on the first touch" (BAR-ATH's truth is derived from corrected tracking, which is
the same mechanism, but that is a belief, not a measurement).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d, maximum_filter1d, uniform_filter1d

from ..contracts import A, D, H, MatchBundle, MatchMeta, Tracking
from ..registry import POSTPROCESS

CHUNK = 16384
TURNOVER = "turnover"
# G2L event grouping. These are part of the detector, not knobs: a run splits when τ* jumps by
# more than 3 frames, a run shorter than 3 frames is noise, and a later event within 4 frames of
# one already kept is the same touch.
_G2L_TAU_JUMP = 3.0
_G2L_MIN_RUN = 3
_G2L_SEPARATION = 4
_DETECTORS = ("w2", "g2l")


def nearest_player(tracking: Tracking) -> tuple[np.ndarray, np.ndarray]:
    """(distance (T,) m, team (T,) int8) of the slot nearest the ball; NaN and -1 where the ball is
    unseen or no slot is observed."""
    n = tracking.n_frames
    dist = np.full(n, np.nan)
    team = np.full(n, -1, dtype=np.int8)
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
        dist[rows[ok]] = np.sqrt(best_d2[ok])
        team[rows[ok]] = tracking.slot_team[best[ok]]
    return dist, team


def ball_velocity_change(ball_xy: np.ndarray, fps: float, w: int) -> np.ndarray:
    """(T,) |v_out - v_in| in m/s.

    v_in = (p_t - p_{t-w}) * fps / w and v_out = (p_{t+w} - p_t) * fps / w. NaN where the ball is
    unseen at t-w, t or t+w, including the w frames at each end of the array (the difference does
    not wrap around the match). Shared with the token builder so the two cannot drift.
    """
    p = np.asarray(ball_xy).astype(np.float64)
    before, after = np.roll(p, w, axis=0), np.roll(p, -w, axis=0)
    before[:w], after[-w:] = np.nan, np.nan
    return np.linalg.norm((after - p) * (fps / w) - (p - before) * (fps / w), axis=1)


def detect_touches(
    tracking: Tracking, *, theta: float = 4.0, reach_m: float = 2.5, w: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """(frames, teams) of the ball touches (see the module docstring)."""
    fps = tracking.meta.fps
    dv = ball_velocity_change(tracking.ball_xy, fps, w)
    dist, team = nearest_player(tracking)
    dv0 = np.nan_to_num(dv, nan=0.0)
    candidate = (dv0 >= theta) & (np.nan_to_num(dist, nan=np.inf) <= reach_m)
    score = np.where(candidate, dv0, 0.0)
    peak = maximum_filter1d(score, size=2 * w + 1, mode="constant")
    frames = np.flatnonzero(candidate & (score >= peak) & (score > 0))
    if len(frames):
        frames = frames[np.concatenate(([True], np.diff(frames) > 1))]
    return frames, team[frames]


def _ols_kernel(length: int) -> np.ndarray:
    """OLS slope weights for a window of `length` frames, q = 0 .. length-1.

    k_q = (q - (L-1)/2) / Σ (q - (L-1)/2)². The weighted sum of positions is the slope in
    metres per frame. L = 1 has no slope; the caller rejects it.
    """
    q = np.arange(length, dtype=np.float64)
    dev = q - (length - 1) / 2.0
    return dev / np.sum(dev * dev)


def _reindex_window(values: np.ndarray, length: int) -> np.ndarray:
    """Move a centred filter output so index s is the window that starts at frame s.

    `correlate1d` / `uniform_filter1d` with origin 0 write the window [s, s+L) at index
    s + L//2. Windows that hang off the array are left NaN.
    """
    shift = length // 2
    count = values.shape[0] - length + 1
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if count > 0:
        out[:count] = values[shift : shift + count]
    return out


def _line_stats(position: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """(slope, mean) of every length-L window of `position`, indexed by the window start.

    Slope is one `correlate1d` with the OLS kernel (metres per frame, per axis). The mean is a
    moving average of the same window. A NaN-propagating window (a correlate of a 0/NaN probe,
    constant NaN outside the array) forces both to NaN unless every frame of the window is seen.
    Unseen samples are zeros in the sums, and the probe throws those windows away, so a gap
    cannot bias a slope.
    """
    if position.shape[0] < length:
        empty = np.full(position.shape, np.nan, dtype=np.float64)
        return empty, empty.copy()
    seen = np.isfinite(position).all(axis=1)
    filled = np.where(seen[:, None], position, 0.0)
    slope = correlate1d(filled, _ols_kernel(length), axis=0, mode="constant", cval=0.0, origin=0)
    mean = uniform_filter1d(filled, size=length, axis=0, mode="constant", cval=0.0, origin=0)
    probe = np.where(seen, 0.0, np.nan)
    visible = np.isfinite(
        correlate1d(probe, np.ones(length), mode="constant", cval=np.nan, origin=0)
    )
    slope = np.where(visible[:, None], slope, np.nan)
    mean = np.where(visible[:, None], mean, np.nan)
    return _reindex_window(slope, length), _reindex_window(mean, length)


def _team_at_point(tracking: Tracking, frame: int, point: np.ndarray, reach_m: float) -> int:
    """Team of the observed slot nearest `point` at `frame`, or -1 when none is within reach."""
    slots = np.asarray(tracking.player_xy[frame], dtype=np.float64)
    distance = np.linalg.norm(slots - point, axis=1)
    distance[~np.isfinite(distance)] = np.inf
    if slots.shape[0] == 0 or not np.isfinite(distance).any():
        return -1
    nearest = int(np.argmin(distance))
    if distance[nearest] <= reach_m:
        return int(tracking.slot_team[nearest])
    return -1


def _g2l_period_events(
    position: np.ndarray,
    origin: int,
    fps: float,
    guard: int,
    fit: int,
    theta: float,
    max_miss_m: float,
) -> list[tuple[int, float, np.ndarray, float]]:
    """Events inside one period, before the toucher and the 4-frame suppression.

    Each item is (global frame round(τ̂), τ̂, vertex point P̂, D). D is the impulse at the
    candidate frame in the run closest to τ̂, which on an isolated kink is the vertex itself.
    A median of D over the run would mix in the shoulders and under-state the impulse.
    """
    n = position.shape[0]
    slope, mean = _line_stats(position, fit)
    beta_in = np.full((n, 2), np.nan)
    beta_out = np.full((n, 2), np.nan)
    mean_in = np.full((n, 2), np.nan)
    mean_out = np.full((n, 2), np.nan)
    # In = [t - g - L + 1, t - g], Out = [t + g, t + g + L - 1].
    offset_in = guard + fit - 1
    if n > offset_in:
        beta_in[offset_in:] = slope[: n - offset_in]
        mean_in[offset_in:] = mean[: n - offset_in]
    last_out = n - fit - guard + 1
    if last_out > 0:
        beta_out[:last_out] = slope[guard : guard + last_out]
        mean_out[:last_out] = mean[guard : guard + last_out]
    half = (fit - 1) / 2.0
    index = np.arange(n, dtype=np.float64)
    centre_in = index - guard - half
    centre_out = index + guard + half
    delta = beta_out - beta_in
    intercept = (mean_out - beta_out * centre_out[:, None]) - (
        mean_in - beta_in * centre_in[:, None]
    )
    denom = np.sum(delta * delta, axis=1)
    tau = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(-np.sum(intercept * delta, axis=1), denom, out=tau, where=denom > 1e-18)
        miss = np.linalg.norm(intercept + delta * tau[:, None], axis=1)
        impulse = np.linalg.norm(delta * fps, axis=1)
        incoming = mean_in + beta_in * (tau - centre_in)[:, None]
        outgoing = mean_out + beta_out * (tau - centre_out)[:, None]
    point = 0.5 * (incoming + outgoing)
    candidate = (
        (impulse >= theta)
        & (np.abs(tau - index) <= guard + 0.5)
        & (miss <= max_miss_m)
        & np.isfinite(impulse)
        & np.isfinite(tau)
        & np.isfinite(miss)
    )
    chosen = np.flatnonzero(candidate)
    if not len(chosen):
        return []
    # Consecutive candidate frames, split again where τ* jumps by more than 3.
    jump = (np.diff(chosen) > 1) | (np.abs(np.diff(tau[chosen])) > _G2L_TAU_JUMP)
    breaks = np.flatnonzero(jump) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(chosen)]))
    events: list[tuple[int, float, np.ndarray, float]] = []
    for start, stop in zip(starts, stops, strict=True):
        if stop - start < _G2L_MIN_RUN:
            continue
        run = chosen[start:stop]
        tau_hat = float(np.median(tau[run]))
        frame = origin + int(round(tau_hat))
        nearest = int(run[int(np.argmin(np.abs(run - tau_hat)))])
        vertex = np.median(point[run], axis=0)
        events.append((frame, tau_hat, vertex, float(impulse[nearest])))
    return events


def detect_touches_g2l(
    tracking: Tracking,
    *,
    guard: int = 8,
    fit: int = 6,
    theta: float = 4.0,
    max_miss_m: float = 1.0,
    reach_m: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frames, teams, strength) of guarded two-line impulses.

    For frame t, inside one period: the incoming window is [t-g-L+1, t-g] and the outgoing
    window is [t+g, t+g+L-1], with g = `guard` and L = `fit`. Both windows must be fully seen;
    the 2g-1 frames between them need not be. An OLS line on each window gives velocities
    v = fps · β. The impulse is D = |v_out - v_in|. The lines meet at τ*; the miss is how far
    apart they are there, and P* is the midpoint. A frame is a candidate when D >= `theta`,
    |τ* - t| <= g + 0.5 and the miss is <= `max_miss_m`.

    Candidates become events in runs of consecutive frames, split where τ* jumps by more than
    3 frames, kept when the run is at least 3 frames. τ̂ is the median τ*, P̂ the median P*,
    and the reported frame is round(τ̂). Strength is D at the candidate frame closest to τ̂,
    in m/s. An event within 4 frames of an earlier one is dropped. The toucher is the observed
    slot nearest P̂ (not the ball) at that frame, within `reach_m`; otherwise the team is -1.

    Signals: ball and slot positions, slot teams, the period bounds, fps. The only Python loop
    is over events (medians and the toucher). Windows never cross a period boundary.

    Where it fails: two touches closer than about half a second merge into one event; a single
    unseen frame in either fit window kills the candidates that use it; a gap inside the guard
    is bridged. Smooth wobble of the ball can still look like an impulse. Aerial balls bent by
    the homography are mostly rejected by the miss limit and the reach.
    """
    if fit < 2:
        raise ValueError(f"fit must be an integer >= 2, got {fit!r}")
    if guard < 0:
        raise ValueError(f"guard must be an integer >= 0, got {guard!r}")
    fps = float(tracking.meta.fps)
    found: list[tuple[int, float, np.ndarray, float]] = []
    for period in tracking.meta.periods:
        position = np.ascontiguousarray(
            tracking.ball_xy[period.start : period.stop], dtype=np.float64
        )
        found.extend(_g2l_period_events(position, period.start, fps, guard, fit, theta, max_miss_m))
    found.sort(key=lambda item: (item[0], item[1]))
    frames: list[int] = []
    teams: list[int] = []
    strength: list[float] = []
    last: int | None = None
    n = tracking.n_frames
    for frame, _tau, vertex, impulse in found:
        if frame < 0 or frame >= n:
            continue
        if last is not None and frame - last <= _G2L_SEPARATION:
            continue
        frames.append(frame)
        teams.append(_team_at_point(tracking, frame, vertex, reach_m))
        strength.append(impulse)
        last = frame
    if not frames:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(frames, dtype=np.int64),
        np.asarray(teams, dtype=np.int8),
        np.asarray(strength, dtype=np.float64),
    )


def _kind(before: int, after: int) -> str:
    if before in (H, A) and after in (H, A):
        return TURNOVER
    return "stop_start" if after == D else "stop_end"


def _nearest(candidates: np.ndarray, frame: int, window: int) -> int | None:
    if not len(candidates):
        return None
    i = int(np.searchsorted(candidates, frame))
    best, best_d = None, window + 1
    for j in (i - 1, i):
        if 0 <= j < len(candidates):
            d = abs(int(candidates[j]) - frame)
            if d < best_d:
                best, best_d = int(candidates[j]), d
    return best


def snap_turnovers(
    labels: np.ndarray,
    meta: MatchMeta,
    frames: np.ndarray,
    teams: np.ndarray,
    *,
    window: int = 50,
    any_team: bool = False,
) -> np.ndarray:
    """Move every H<->A boundary to the nearest touch of the team that gains the ball (of any team
    with `any_team`, the negative control) within +-`window` frames; order kept, one frame per
    segment at least. Loops over segments, never over frames."""
    frames = np.asarray(frames, dtype=np.int64)
    teams = np.asarray(teams)
    by_team = {t: frames[teams == t] for t in (H, A)}
    out = labels.copy()
    for period in meta.periods:
        lo, hi = period.start, period.stop
        seg = labels[lo:hi]
        breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        if not len(breaks):
            continue
        starts = np.concatenate(([0], breaks)) + lo
        codes = seg[starts - lo]
        new = starts.copy()
        for k in range(1, len(starts)):
            boundary = int(starts[k])
            if _kind(int(codes[k - 1]), int(codes[k])) != TURNOVER:
                continue
            candidates = frames if any_team else by_team.get(int(codes[k]), frames[:0])
            hit = _nearest(candidates, boundary, window)
            if hit is None:
                continue
            upper = int(starts[k + 1]) - 1 if k + 1 < len(starts) else hi - 1
            new[k] = min(max(hit, int(new[k - 1]) + 1), upper)
        stops = np.concatenate((new[1:], [hi]))
        for k in range(len(new)):
            out[new[k] : stops[k]] = codes[k]
    return out


@POSTPROCESS.register("touch_snap", version="1", example={})
def touch_snap(
    labels: np.ndarray,
    bundle: MatchBundle,
    *,
    window_s: float = 2.0,
    theta: float = 4.0,
    reach_m: float = 2.5,
    w: int = 2,
    any_team: bool = False,
    detector: str = "w2",
) -> np.ndarray:
    """`labels` with every H<->A boundary moved onto the nearest touch of the gaining team within
    +-`window_s` seconds.

    `detector` "w2" (default) is `detect_touches` (dv >= 4 m/s, 2.5 m, w = 2, +-2 s). "g2l" snaps
    to `detect_touches_g2l` event frames of the gaining team, at the same theta and reach; guard
    and fit stay 8 and 6. The version stays "1": the default arguments reproduce the previous
    labels, and a config that passes `detector` changes the cache key through its parameters.
    """
    if detector not in _DETECTORS:
        raise ValueError(f"touch_snap detector must be 'w2' or 'g2l', got {detector!r}")
    tracking = bundle.tracking
    if detector == "g2l":
        frames, teams, _strength = detect_touches_g2l(tracking, theta=theta, reach_m=reach_m)
    else:
        frames, teams = detect_touches(tracking, theta=theta, reach_m=reach_m, w=w)
    window = int(round(window_s * tracking.meta.fps))
    return snap_turnovers(
        labels, tracking.meta, frames, teams, window=window, any_team=any_team
    ).astype(np.int8)
