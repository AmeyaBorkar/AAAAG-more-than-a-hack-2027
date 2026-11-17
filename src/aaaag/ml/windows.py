"""Window builder: a Tracking becomes per-frame token tensors; windows and chunks slice them.

`build_match_tokens` is the one place that decides what the network sees. Per frame and per team
it keeps the `tokens_per_team` observed slots nearest to a reference point (the ball where it is
seen, else the team's centroid), sorted by distance, so a slot index never reaches the model and a
fragmenting tracker (46 slots for 22 players) costs nothing but padding. The ball is the last
token, with a visibility flag and two features for the length of the run of missing ball around
the frame (non-causal: the frame knows how long until the ball is back). When
`TokenSpec.control_memory` is on, the ball token also carries the last controlling team and the
age of that run (`control_possession`'s memory, which a +-5 s window cannot hold). When
`TokenSpec.contact_inputs` is on, four more columns follow that pair (or the base 14): the ball's
speed change, a decayed touch-team sign, the live control flag, and each slot's acceleration.
`touch_detector` "w2" (the default, also a missing key) builds that touch column with
`detect_touches`. "g2l" builds it from the guarded two-line events and appends TOUCH_DV after
the four contact columns, before the ball-path columns. The default detector leaves every tensor
identical to the layout without the key.
When `TokenSpec.ball_path_control` is on, four more follow those (or the control-memory pair, or
the base 14): signed pitch control along the ball's path, the probability the path leaves the
pitch or is still free at the horizon, the expected time of that control, and each slot's share.
The flags are off by default and then the tensor is the layout without them, unchanged.
When `TokenSpec.slot_index` is on, two identifier columns come last on the player tokens: the
slot the argsort picked (`SLOT_ID`, 1-based) and that slot's observed-run id (`SLOT_RUN`), so
`model.slot_temporal` can follow each tracker slot through time and stop at a fragment edge. The
network never embeds them, and they change no other column.

Signals: player and ball positions, slot teams, the period layout, and, with control memory or
contact inputs on, the nearest player within `control_reach_m`. Contact inputs also use the touch
detector: speed change over `contact_dv_w` frames, and, with `touch_detector` "w2", touches over
`contact_touch_w` frames of the nearest slot within 2.5 m. "g2l" replaces that touch column with
the guarded two-line events (impulse D over `velocity_scale` in TOUCH_DV, same decay) and does
not read `contact_touch_w`. Ball-path control uses the same gated velocities as the VX/VY
columns, every observed slot of both teams, and the pitch size.
Not used: slot identities, provider speeds, tags, labels. Everything is vectorised over frames;
the cost is a few argsorts of (T, slots per team) arrays and a handful of shifted differences
(about a second for a match). Control memory adds one nearest-player pass; contact inputs add the
touch detector and a second pass over the same control codes. Ball-path control adds one loop of
26 time steps over chunks of 8192 frames.

Where it fails: a team with fewer than `tokens_per_team` observed slots gets padding (fine); a
duplicate slot on one player takes a token that a teammate could have had; the orientation read
from geometry is wrong only if a period shows one team's centroid on the wrong side for most of
it, which no real match does. Control memory fails where `control_possession` fails: a carrier
beyond reach, a duel inside the reach, a change of possession inside a long unseen stretch.
Contact columns are zero where the ball is unseen in the speed-change window, where the touch
detector has no spike, and where a slot's velocity cannot be differenced. Path columns are zero
when the ball has not been seen for 1 s inside the period. A gated-out ball velocity is treated
as stationary, a slot with no position takes no part, and once the path is more than 0.5 m off
the pitch the remaining probability is PATH_OUT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import AWAY, HOME, Period, Tracking
from ..evidence.nearest_player import nearest_team_codes
from ..evidence.possession.control import last_controller
from ..postprocess.touch_snap import ball_velocity_change, detect_touches, detect_touches_g2l
from .geometry import attack_sign
from .spec import (
    BRACKET_FEATURES,
    MAX_SLOT_ID,
    SLOT_RUN_MOD,
    Col,
    TokenSpec,
    bracket_columns,
    contact_columns,
    path_columns,
    slot_index_columns,
    touch_dv_column,
)

FAR_M = 60.0  # distances to the reference are capped here before scaling
_CTRL_AGE_CAP_S = 60.0  # CTRL_AGE = log1p(seconds) / log1p(this), capped at 1
_BALL_DV_W = 5  # default TokenSpec.contact_dv_w; BALL_DV reads the spec field
_BALL_DV_CLIP = 4.0
_TOUCH_THETA = 4.0  # m/s, detect_touches default
_TOUCH_REACH_M = 2.5
_TOUCH_W = 2  # default TokenSpec.contact_touch_w; TOUCH_TEAM reads the spec field
# The columns read the spec. These constants are today's windows, so a missing key cannot drift.
assert (TokenSpec().contact_touch_w, TokenSpec().contact_dv_w) == (_TOUCH_W, _BALL_DV_W)
assert TokenSpec().touch_detector == "w2"
_TOUCH_LOOKBACK = 10  # most recent touch with frames_since <= this, inside the period
_TOUCH_DECAY = 5.0  # exp(-frames_since / this)
_ACCEL_SCALE = 8.0  # m/s^2; not velocity_scale
_ACCEL_CLIP = 4.0
_PATH_CHUNK = 8192  # frames per vectorised pitch-control block
_PATH_DT = 0.1  # seconds; τ step and the control-integration step
_PATH_STEPS = 26  # k = 0..25, horizon 2.5 s
_PATH_HOLD_S = 1.0  # coast the last seen ball for at most this long, inside the period
_PATH_REACTION_S = 0.7
_PATH_RUN_SPEED = 5.0  # m/s, after the reaction delay
_PATH_SIGMA = 0.45  # seconds; logistic scale of the arrival time
_PATH_LAMBDA = 4.3  # 1/s, Spearman's control rate
_PATH_MARGIN_M = 0.5  # still on the pitch until this far past the line
_PATH_T_SCALE = 2.5  # PATH_T divides the expected control time by the horizon
_PATH_COEF = float(np.pi / (np.sqrt(3.0) * _PATH_SIGMA))


def smoothed_velocity(xy: np.ndarray, half: int, fps: int) -> np.ndarray:
    """Velocity in m/s of a (n, ..., 2) track, NaN where it cannot be formed.

    A centred difference over +-`half` frames where both ends are seen, else a one-sided one over
    `half` frames from whichever side exists. It only looks inside the array, so call it per
    period. The centred form has no lag and averages out positional jitter.
    """
    n = len(xy)
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    if n <= half:
        return out
    dt = half / fps
    forward = (xy[half:] - xy[:-half]) / dt  # at t: (xy[t + half] - xy[t]) / dt, t < n - half
    centred = np.full(xy.shape, np.nan, dtype=np.float32)
    centred[half : n - half] = 0.5 * (forward[half:] + forward[: n - 2 * half])
    out[: n - half] = forward
    backward = np.full(xy.shape, np.nan, dtype=np.float32)
    backward[half:] = forward
    use = np.isnan(out[..., 0])
    out[use] = backward[use]
    use = ~np.isnan(centred[..., 0])
    out[use] = centred[use]
    return out


def absence_features(seen: np.ndarray, fps: int, cap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(before, after) float32 in [0, 1]: log-scaled seconds since the ball was last seen and until
    it is next seen, both 0 where it is seen and `cap_s` when it never is. `seen` is one period."""
    n = len(seen)
    index = np.arange(n)
    cap = max(1, int(round(cap_s * fps)))
    last = np.maximum.accumulate(np.where(seen, index, -(cap + 1)))
    following = np.minimum.accumulate(np.where(seen, index, n + cap + 1)[::-1])[::-1]
    before = np.minimum(index - last, cap)
    after = np.minimum(following - index, cap)
    scale = np.log1p(cap_s)
    return (
        (np.log1p(before / fps) / scale).astype(np.float32),
        (np.log1p(after / fps) / scale).astype(np.float32),
    )


def build_match_tokens(
    tracking: Tracking, spec: TokenSpec | None = None, *, sign: np.ndarray | None = None
) -> np.ndarray:
    """(T, spec.n_tokens, spec.n_features) float32; all zeros (VALID = 0) outside the periods.

    `sign` is the per-frame attack-direction multiplier (geometry.attack_sign); by default it is
    read from `tracking`, and training passes the clean match's, which degradation cannot change.
    With `spec.control_memory`, the ball token's `CTRL_TEAM` / `CTRL_AGE` are the last-controller
    rule on this same tracking (per period). They are zero on every other token. With
    `spec.contact_inputs`, BALL_DV / TOUCH_TEAM / CTRL_NOW are written on the ball token and ACCEL
    on the player tokens; the other tokens stay zero in those columns. With `touch_detector`
    "g2l" the touch column is the guarded two-line events and TOUCH_DV follows ACCEL on the ball
    token. With `spec.ball_path_control`, PATH_SIGN / PATH_OUT / PATH_T are written on the ball
    token and PPCF on the player tokens, after those columns. With `spec.gap_bracket`, the six
    bracket columns (`_gap_bracket_columns`) follow, on the ball token only. With
    `spec.slot_index`, SLOT_ID and SLOT_RUN (`_slot_run_ids`) come last, on valid player tokens
    only: the order the distance argsort chose, written as tracker slots."""
    spec = spec or TokenSpec()
    meta = tracking.meta
    T, S, K, fps = tracking.n_frames, tracking.n_slots, spec.tokens_per_team, meta.fps
    half_l, half_w = meta.pitch_length / 2, meta.pitch_width / 2
    half = max(1, int(round(spec.velocity_half_s * fps)))
    played = meta.played_mask()

    sign = attack_sign(tracking) if sign is None else sign
    players = tracking.player_xy * sign[:, None, None]
    ball = tracking.ball_xy * sign[:, None]
    seen = ~np.isnan(ball[:, 0]) & played

    player_v = np.full((T, S, 2), np.nan, dtype=np.float32)
    ball_v = np.full((T, 2), np.nan, dtype=np.float32)
    gap_before = np.zeros(T, dtype=np.float32)
    gap_after = np.zeros(T, dtype=np.float32)
    for period in meta.periods:
        window = slice(period.start, period.stop)
        player_v[window] = smoothed_velocity(players[window], half, fps)
        ball_v[window] = smoothed_velocity(ball[window], half, fps)
        gap_before[window], gap_after[window] = absence_features(seen[window], fps, spec.gap_cap_s)
    _drop_fast(player_v, spec.max_player_speed)
    _drop_fast(ball_v, spec.max_ball_speed)
    # Same per-slot velocity the VX/VY columns use, differenced before the distance sort so a token
    # keeps its own slot's acceleration when the nearest-slot order changes.
    slot_accel = (
        _slot_accel_magnitude(player_v, meta.periods, half, fps) if spec.contact_inputs else None
    )
    contact = contact_columns(spec) if spec.contact_inputs else None
    path_cols = path_columns(spec) if spec.ball_path_control else None
    if path_cols is not None:
        path_sign, path_out, path_time, slot_ppcf = _ball_path_columns(
            players,
            player_v,
            ball,
            ball_v,
            seen,
            tracking.slot_team,
            meta.periods,
            fps,
            meta.pitch_length,
            meta.pitch_width,
        )
    slot_cols = slot_index_columns(spec) if spec.slot_index else None
    if slot_cols is not None:
        if S > MAX_SLOT_ID:
            raise ValueError(f"slot_index holds at most {MAX_SLOT_ID} slots exactly, got {S}")
        slot_run = _slot_run_ids(players, meta.periods, played)

    out = np.zeros((T, spec.n_tokens, spec.n_features), dtype=np.float32)
    for team in (HOME, AWAY):
        slots = np.flatnonzero(tracking.slot_team == team)
        positions, velocities = _padded(players[:, slots], K), _padded(player_v[:, slots], K)
        observed = ~np.isnan(positions[..., 0])
        count = observed.sum(axis=1)
        centroid = np.where(
            count[:, None] > 0,
            np.nansum(positions, axis=1) / np.maximum(count, 1)[:, None],
            np.nan,
        )
        reference = np.where(seen[:, None], ball, centroid)  # NaN when the team is unseen too
        offset = positions - reference[:, None, :]
        distance2 = np.where(np.isnan(offset[..., 0]), np.inf, (offset**2).sum(axis=-1))
        order = np.argsort(distance2, axis=1, kind="stable")[:, :K]
        chosen = np.take_along_axis(distance2, order, axis=1)
        valid = np.isfinite(chosen) & played[:, None]

        def pick(array: np.ndarray, order: np.ndarray = order) -> np.ndarray:
            return np.take_along_axis(array, order[..., None], axis=1)

        pos, vel, off = pick(positions), pick(velocities), pick(offset)
        speed = np.hypot(vel[..., 0], vel[..., 1])
        block = out[:, team * K : (team + 1) * K]
        block[..., Col.X] = _on(valid, pos[..., 0] / half_l)
        block[..., Col.Y] = _on(valid, pos[..., 1] / half_w)
        block[..., Col.VX] = _on(valid, vel[..., 0] / spec.velocity_scale)
        block[..., Col.VY] = _on(valid, vel[..., 1] / spec.velocity_scale)
        block[..., Col.SPEED] = _on(valid, speed / spec.velocity_scale)
        block[..., Col.IS_AWAY] = float(team == AWAY)
        block[..., Col.VALID] = valid
        block[..., Col.DX] = _on(valid, off[..., 0] / spec.ref_scale_m)
        block[..., Col.DY] = _on(valid, off[..., 1] / spec.ref_scale_m)
        block[..., Col.DIST] = _on(valid, np.minimum(np.sqrt(chosen), FAR_M) / spec.ref_scale_m)
        block[..., Col.REF_BALL] = (valid & seen[:, None]).astype(np.float32)
        if slot_accel is not None:
            picked = np.take_along_axis(_padded(slot_accel[:, slots], K), order, axis=1)
            scaled = np.clip(picked / _ACCEL_SCALE, 0.0, _ACCEL_CLIP)
            block[..., contact[3]] = _on(valid, scaled)
        if path_cols is not None:
            picked = np.take_along_axis(_padded(slot_ppcf[:, slots], K), order, axis=1)
            block[..., path_cols[3]] = _on(valid, picked)
        if slot_cols is not None:
            ids = _padded((slots + 1).astype(np.float32)[None], K)[0]  # NaN past the team's slots
            block[..., slot_cols[0]] = _on(valid, ids[order])
            picked = np.take_along_axis(_padded(slot_run[:, slots], K), order, axis=1)
            block[..., slot_cols[1]] = _on(valid, picked)

    speed = np.hypot(ball_v[:, 0], ball_v[:, 1])
    tok = out[:, spec.ball_index]
    tok[:, Col.X] = _on(seen, ball[:, 0] / half_l)
    tok[:, Col.Y] = _on(seen, ball[:, 1] / half_w)
    tok[:, Col.VX] = _on(seen, ball_v[:, 0] / spec.velocity_scale)
    tok[:, Col.VY] = _on(seen, ball_v[:, 1] / spec.velocity_scale)
    tok[:, Col.SPEED] = _on(seen, speed / spec.velocity_scale)
    tok[:, Col.IS_BALL] = 1.0
    tok[:, Col.VALID] = seen
    tok[:, Col.REF_BALL] = seen
    tok[:, Col.GAP_BEFORE] = gap_before
    tok[:, Col.GAP_AFTER] = gap_after
    if spec.control_memory:
        ctrl_team, ctrl_age = _control_columns(tracking, spec)
        tok[:, Col.CTRL_TEAM] = ctrl_team
        tok[:, Col.CTRL_AGE] = ctrl_age
    if contact is not None:
        tok[:, contact[0]] = _ball_dv_column(tracking, spec)
        if spec.touch_detector == "g2l":
            touch_team, touch_dv = _g2l_touch_columns(tracking, spec)
            tok[:, contact[1]] = touch_team
            tok[:, touch_dv_column(spec)] = touch_dv
        else:
            tok[:, contact[1]] = _touch_team_column(tracking, spec)
        tok[:, contact[2]] = _ctrl_now_column(tracking, spec)
    if path_cols is not None:
        tok[:, path_cols[0]] = path_sign
        tok[:, path_cols[1]] = path_out
        tok[:, path_cols[2]] = path_time
    if spec.gap_bracket:
        bracket = _gap_bracket_columns(tracking, spec, ball, seen, half_l, half_w)
        for column, values in zip(bracket_columns(spec), bracket, strict=True):
            tok[:, column] = values
    out[~played] = 0.0
    return out


def _slot_run_ids(
    players: np.ndarray, periods: tuple[Period, ...], played: np.ndarray
) -> np.ndarray:
    """(T, S) float32 SLOT_RUN: 1 + (the slot's observed-run count so far - 1) % SLOT_RUN_MOD.

    A run is a stretch of frames where the slot has a position inside the periods. A new one
    starts after any frame where the slot is unobserved, and at every period start, so a reused
    slot (a new tracker fragment after a gap) gets a new id. The value is 0 where the slot is
    unobserved or the frame is not played. Two runs of one slot with the same id are at least
    2 * SLOT_RUN_MOD frames apart. A one-frame jump inside a run (an identity swap with no gap) is
    not a new run: the velocity gate in VX/VY is what sees that.
    """
    observed = ~np.isnan(players[..., 0]) & played[:, None]
    starts = observed.copy()
    starts[1:] &= ~observed[:-1]
    for period in periods:
        starts[period.start] = observed[period.start]
    count = np.cumsum(starts, axis=0, dtype=np.int32)
    ids = (count - 1) % SLOT_RUN_MOD + 1
    return np.where(observed, ids, 0).astype(np.float32)


def _signed_team(codes: np.ndarray) -> np.ndarray:
    """+1 home, -1 away, 0 for any other code, float32."""
    out = np.zeros(len(codes), dtype=np.float32)
    out[codes == HOME] = np.float32(1.0)
    out[codes == AWAY] = np.float32(-1.0)
    return out


def _gap_bracket_columns(
    tracking: Tracking,
    spec: TokenSpec,
    ball: np.ndarray,
    seen: np.ndarray,
    half_l: float,
    half_w: float,
) -> np.ndarray:
    """(6, T) float32: the two ends of the ball gap each frame sits in (#33).

    Rows: team at the last sighting, team from the next sighting on, then x, y at the last
    sighting and x, y at the next one. Nonzero only on unseen frames inside a period, and only on
    a side that has a sighting in the same period (a gap open to the period edge keeps that side 0).

    Signals: the ball-seen mask and attack-normalised ball position of this tracking (`ball` and
    `seen` as `build_match_tokens` computes them, so a hidden-ball augmentation hides the ends too),
    and the control rule `last_controller(nearest_team_codes(reach), hold)` with the
    control-memory reach and hold. The before team is the rule's held team at the last seen frame.
    The after team is the same rule run backwards in time, read at the next seen frame: the
    first qualifying run that ends at or after it. Both use only frames where the ball is seen, so
    a hidden stretch never leaks. The after side looks ahead, like `GAP_AFTER`: the labeller is
    offline and non-causal.

    Where it fails: a sighting where nobody is within reach gives 0 for that team; a loose ball
    that reappears mid-flight puts the after team on whoever is nearest the ball next, which can
    be the receiver, not the passer. Vectorised per period (a period is one loop step).
    """
    T = tracking.n_frames
    out = np.zeros((BRACKET_FEATURES, T), dtype=np.float32)
    codes = nearest_team_codes(tracking, max_distance_m=spec.control_reach_m)
    hold = max(1, int(spec.control_hold_frames))
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        n = hi - lo
        s = seen[lo:hi]
        if n <= 0 or not s.any():
            continue
        index = np.arange(n)
        last = np.maximum.accumulate(np.where(s, index, -1))
        following = np.minimum.accumulate(np.where(s, index, n)[::-1])[::-1]
        has_last = ~s & (last >= 0)
        has_next = ~s & (following < n)
        li = np.where(has_last, last, 0)
        ni = np.where(has_next, following, 0)
        segment = codes[lo:hi]
        before = _signed_team(last_controller(segment, hold))
        after = _signed_team(last_controller(segment[::-1], hold)[::-1])
        xy = np.nan_to_num(ball[lo:hi], nan=0.0)
        zero = np.float32(0.0)
        out[0, lo:hi] = np.where(has_last, before[li], zero)
        out[1, lo:hi] = np.where(has_next, after[ni], zero)
        out[2, lo:hi] = np.where(has_last, xy[li, 0] / half_l, zero)
        out[3, lo:hi] = np.where(has_last, xy[li, 1] / half_w, zero)
        out[4, lo:hi] = np.where(has_next, xy[ni, 0] / half_l, zero)
        out[5, lo:hi] = np.where(has_next, xy[ni, 1] / half_w, zero)
    return out


def _control_columns(tracking: Tracking, spec: TokenSpec) -> tuple[np.ndarray, np.ndarray]:
    """(T,) `CTRL_TEAM` and `CTRL_AGE`, both zero outside periods and before the first control.

    The team is `last_controller` on `nearest_team_codes` of this tracking, one period at a time,
    mapped to +1 home / -1 away. The age is log1p(seconds since that run's first frame) /
    log1p(60), capped at 1. The run scan matches `last_controller`: a stretch of one team lasting
    at least `control_hold_frames`, dated from its first frame and held through later gaps.
    """
    codes = nearest_team_codes(tracking, max_distance_m=spec.control_reach_m)
    team = np.zeros(tracking.n_frames, dtype=np.float32)
    age = np.zeros(tracking.n_frames, dtype=np.float32)
    hold = max(1, int(spec.control_hold_frames))
    fps = tracking.meta.fps
    scale = np.log1p(_CTRL_AGE_CAP_S)
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        segment = codes[lo:hi]
        controlled = last_controller(segment, hold)
        starts = _qualifying_run_starts(segment, hold)
        signed = np.zeros(hi - lo, dtype=np.float32)
        signed[controlled == HOME] = np.float32(1.0)
        signed[controlled == AWAY] = np.float32(-1.0)
        team[lo:hi] = signed
        known = starts >= 0
        if not known.any():
            continue
        seconds = (np.arange(hi - lo) - starts) / fps
        column = np.log1p(np.maximum(seconds, 0.0)) / scale
        age[lo:hi] = np.where(known, np.minimum(column, 1.0), 0.0).astype(np.float32)
    return team, age


def _run_bounds(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and stop (exclusive) of each constant run in `codes`."""
    n = len(codes)
    change = np.flatnonzero(np.diff(codes.astype(np.int16)) != 0) + 1
    return np.concatenate(([0], change)), np.concatenate((change, [n]))


def _qualifying_run_starts(codes: np.ndarray, hold_frames: int) -> np.ndarray:
    """(n,) int64: start index of the latest qualifying control run, else -1.

    The same runs as `last_controller` (which returns only the team). A second qualifying run by
    the same team restarts the clock; the output team does not change, so the start cannot be read
    off the team column.
    """
    n = len(codes)
    latest = np.full(n, -1, dtype=np.int64)
    if not n:
        return latest
    starts, stops = _run_bounds(codes)
    qualifying = (codes[starts] >= 0) & (stops - starts >= hold_frames)
    if not qualifying.any():
        return latest
    latest[starts[qualifying]] = starts[qualifying]
    return np.maximum.accumulate(latest)


def _ball_dv_column(tracking: Tracking, spec: TokenSpec) -> np.ndarray:
    """(T,) BALL_DV: |v_out - v_in| / velocity_scale, clipped to [0, 4], else 0.

    The difference is `ball_velocity_change` with w = `spec.contact_dv_w` (default `_BALL_DV_W`,
    5 frames), the same arithmetic as `detect_touches`, so a sample is 0 where the ball is unseen
    at t-w, t or t+w. Unplayed frames are cleared by the caller.
    """
    dv = ball_velocity_change(tracking.ball_xy, tracking.meta.fps, spec.contact_dv_w)
    scaled = np.clip(np.nan_to_num(dv, nan=0.0) / spec.velocity_scale, 0.0, _BALL_DV_CLIP)
    return scaled.astype(np.float32)


def _g2l_touch_columns(tracking: Tracking, spec: TokenSpec) -> tuple[np.ndarray, np.ndarray]:
    """(T,) TOUCH_TEAM and TOUCH_DV from one `detect_touches_g2l` pass.

    TOUCH_TEAM is +1 home / -1 away at round(τ̂), then exp(-age / 5) while that event is the latest
    one with a team and age <= 10, inside the period. An event with no team writes no sign.
    TOUCH_DV is that event's D / velocity_scale, clipped to [0, 4], with the same decay, including
    an event that has no team. A new period does not inherit the previous one. theta stays 4 m/s
    and the reach stays 2.5 m; `contact_touch_w` is not used.
    """
    frames, teams, strength = detect_touches_g2l(
        tracking, theta=_TOUCH_THETA, reach_m=_TOUCH_REACH_M
    )
    team_impulse = np.zeros(tracking.n_frames, dtype=np.float64)
    dv_impulse = np.zeros(tracking.n_frames, dtype=np.float64)
    if len(frames):
        team_impulse[frames[teams == HOME]] = 1.0
        team_impulse[frames[teams == AWAY]] = -1.0
        dv_impulse[frames] = np.clip(strength / spec.velocity_scale, 0.0, _BALL_DV_CLIP)
    team_out = np.zeros(tracking.n_frames, dtype=np.float32)
    dv_out = np.zeros(tracking.n_frames, dtype=np.float32)
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        team_out[lo:hi] = _decay_recent(team_impulse[lo:hi], _TOUCH_LOOKBACK, _TOUCH_DECAY)
        dv_out[lo:hi] = _decay_recent(dv_impulse[lo:hi], _TOUCH_LOOKBACK, _TOUCH_DECAY)
    return team_out, dv_out


def _touch_team_column(tracking: Tracking, spec: TokenSpec) -> np.ndarray:
    """(T,) TOUCH_TEAM: decayed sign of the latest `detect_touches` hit inside the period.

    +1 home, -1 away at the touch frame, then exp(-frames_since / 5) while that touch is the most
    recent one and frames_since <= 10. The detector window is `spec.contact_touch_w` (default
    `_TOUCH_W`, 2 frames); theta stays 4 m/s and the reach stays 2.5 m. A new period does not
    inherit the previous one. 0 if none.
    """
    frames, teams = detect_touches(
        tracking, theta=_TOUCH_THETA, reach_m=_TOUCH_REACH_M, w=spec.contact_touch_w
    )
    impulse = np.zeros(tracking.n_frames, dtype=np.float64)
    if len(frames):
        impulse[frames[teams == HOME]] = 1.0
        impulse[frames[teams == AWAY]] = -1.0
    out = np.zeros(tracking.n_frames, dtype=np.float32)
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        out[lo:hi] = _decay_recent(impulse[lo:hi], _TOUCH_LOOKBACK, _TOUCH_DECAY)
    return out


def _decay_recent(impulse: np.ndarray, lookback: int, decay: float) -> np.ndarray:
    """Sign of the latest nonzero impulse in [t-lookback, t], times exp(-age / decay)."""
    n = len(impulse)
    out = np.zeros(n, dtype=np.float32)
    events = np.flatnonzero(impulse != 0)
    if not len(events):
        return out
    index = np.arange(n)
    pos = np.searchsorted(events, index, side="right") - 1
    valid = pos >= 0
    age = np.zeros(n, dtype=np.int64)
    age[valid] = index[valid] - events[pos[valid]]
    recent = valid & (age <= lookback)
    if not recent.any():
        return out
    chosen = events[pos[recent]]
    out[recent] = (impulse[chosen] * np.exp(-age[recent] / decay)).astype(np.float32)
    return out


def _ctrl_now_column(tracking: Tracking, spec: TokenSpec) -> np.ndarray:
    """(T,) CTRL_NOW: +1/-1 while the in-reach team has held for `control_hold_frames` up to t.

    Same nearest-player codes and run split as `_control_columns`, but only on the frames of a
    qualifying run from its hold-th frame through the run's end. It does not backdate the run,
    does not survive a gap, and does not cross a period boundary. 0 otherwise.
    """
    codes = nearest_team_codes(tracking, max_distance_m=spec.control_reach_m)
    out = np.zeros(tracking.n_frames, dtype=np.float32)
    hold = max(1, int(spec.control_hold_frames))
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        out[lo:hi] = _current_control_sign(codes[lo:hi], hold)
    return out


def _current_control_sign(codes: np.ndarray, hold: int) -> np.ndarray:
    """(n,) float32. The team sign from the hold-th frame of each in-reach run, else 0."""
    n = len(codes)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    starts, stops = _run_bounds(codes)
    qualifying = (codes[starts] >= 0) & (stops - starts >= hold)
    if not qualifying.any():
        return out
    on_at = starts[qualifying] + (hold - 1)
    teams = codes[starts[qualifying]]
    sign = np.where(teams == HOME, np.float32(1.0), np.float32(-1.0)).astype(np.float32)
    delta = np.zeros(n, dtype=np.float32)
    np.add.at(delta, on_at, sign)
    stop = stops[qualifying]
    ending = stop < n
    np.add.at(delta, stop[ending], -sign[ending])
    return np.cumsum(delta, dtype=np.float32)


def _slot_accel_magnitude(
    velocity: np.ndarray, periods: tuple[Period, ...], half: int, fps: int
) -> np.ndarray:
    """(T, S) |centred difference of velocity| in m/s^2, NaN where either sample is missing.

    The window is the same ±`half` frames as `smoothed_velocity`, one period at a time, so a
    difference does not cross a break. `velocity` is the array already written into VX/VY.
    """
    out = np.full(velocity.shape[:2], np.nan, dtype=np.float32)
    for period in periods:
        lo, hi = period.start, period.stop
        diff = _centred_difference(velocity[lo:hi], half, fps)
        with np.errstate(invalid="ignore"):
            magnitude = np.hypot(diff[..., 0], diff[..., 1])
        out[lo:hi] = magnitude.astype(np.float32)
    return out


def _centred_difference(values: np.ndarray, half: int, fps: int) -> np.ndarray:
    """(v[t+half] - v[t-half]) * fps / (2*half). NaN at the edges and where either end is NaN."""
    n = values.shape[0]
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if half < 1 or n <= 2 * half:
        return out
    dt = (2 * half) / float(fps)
    out[half : n - half] = (values[2 * half :] - values[: n - 2 * half]) / dt
    return out


def _padded(array: np.ndarray, width: int) -> np.ndarray:
    """Pad axis 1 with NaN up to `width` slots, so a short team can still fill its block."""
    short = width - array.shape[1]
    if short <= 0:
        return array
    pad = [(0, 0), (0, short)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pad, constant_values=np.nan)


def _on(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`values` where `mask`, else 0; NaN (an undefined velocity) also becomes 0."""
    return np.where(mask & ~np.isnan(values), values, 0.0).astype(np.float32)


def _drop_fast(velocity: np.ndarray, limit: float) -> None:
    """NaN the velocities above `limit` m/s: a swap or a jump, not motion (in place)."""
    with np.errstate(invalid="ignore"):
        velocity[np.hypot(velocity[..., 0], velocity[..., 1]) > limit] = np.nan


def _expit(values: np.ndarray) -> np.ndarray:
    """Logistic sigmoid. ±60 already saturates far below the 1e-5 the columns are checked at."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _ball_path_columns(
    players: np.ndarray,
    player_v: np.ndarray,
    ball: np.ndarray,
    ball_v: np.ndarray,
    seen: np.ndarray,
    teams: np.ndarray,
    periods: tuple[Period, ...],
    fps: int,
    pitch_length: float,
    pitch_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(PATH_SIGN, PATH_OUT, PATH_T, PPCF)` float32. `PPCF` is `(T, S)`, the others `(T,)`.

    Spearman's pitch control along a straight coast of the ball, in the attack-normalised frame
    the caller already built. A seen frame uses that frame's gated velocity (NaN as 0, the same
    substitution as VX). A frame unseen for at most 1 s coasts from the last seen sample inside
    the period: `p0 = p_last + v_last * s`, `v0 = v_last`. Older than that, or with no earlier
    sample in the period, every column is 0.

    The path is `b_k = p0 + v0 * 0.1 * k` for `k = 0..25` (no drag). Every observed slot reacts
    for 0.7 s along its gated velocity, then runs at 5 m/s; an unseen slot takes no part. Arrival
    probability is a logistic with scale 0.45 s. Control integrates at 4.3/s by an exact
    exponential step of 0.1 s. A point more than 0.5 m off the pitch stops the integral, and
    whatever probability remains there or at the horizon is PATH_OUT. PATH_SIGN is home PPCF minus
    away PPCF. PATH_T is the expected control time divided by 2.5, and 0 when nobody takes any.
    """
    p0, v0, active = _ball_path_state(ball, ball_v, seen, periods, fps)
    n_frames, n_slots = players.shape[:2]
    path_sign = np.zeros(n_frames, dtype=np.float32)
    path_out = np.zeros(n_frames, dtype=np.float32)
    path_time = np.zeros(n_frames, dtype=np.float32)
    ppcf = np.zeros((n_frames, n_slots), dtype=np.float32)
    x_lim = pitch_length / 2.0 + _PATH_MARGIN_M
    y_lim = pitch_width / 2.0 + _PATH_MARGIN_M
    for start in range(0, n_frames, _PATH_CHUNK):
        stop = min(n_frames, start + _PATH_CHUNK)
        if not active[start:stop].any():
            continue
        sign, out, time, slot = _path_integrate(
            p0[start:stop],
            v0[start:stop],
            active[start:stop],
            players[start:stop],
            player_v[start:stop],
            teams,
            x_lim,
            y_lim,
        )
        path_sign[start:stop] = sign
        path_out[start:stop] = out
        path_time[start:stop] = time
        ppcf[start:stop] = slot
    return path_sign, path_out, path_time, ppcf


def _ball_path_state(
    ball: np.ndarray,
    ball_v: np.ndarray,
    seen: np.ndarray,
    periods: tuple[Period, ...],
    fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(p0, v0, active)` for a coast of at most 1 s from the last seen ball in the period."""
    n_frames = ball.shape[0]
    p0 = np.zeros((n_frames, 2), dtype=np.float64)
    v0 = np.zeros((n_frames, 2), dtype=np.float64)
    active = np.zeros(n_frames, dtype=bool)
    # age/fps <= 1 s, with the integer comparison so the 1.0 s frame is not lost to rounding.
    max_age = int(np.floor(_PATH_HOLD_S * float(fps) + 1e-9))
    for period in periods:
        lo, hi = period.start, period.stop
        length = hi - lo
        index = np.arange(length)
        last = np.maximum.accumulate(np.where(seen[lo:hi], index, -1))
        known = last >= 0
        age = np.where(known, index - last, max_age + 1)
        on = known & (age <= max_age)
        if not on.any():
            continue
        src = lo + np.where(known, last, 0)
        p_last = ball[src].astype(np.float64, copy=False)
        v_last = np.nan_to_num(ball_v[src].astype(np.float64, copy=False), nan=0.0)
        seconds = age.astype(np.float64) / float(fps)
        coast = p_last + v_last * seconds[:, None]
        p0[lo:hi] = np.where(on[:, None], coast, 0.0)
        v0[lo:hi] = np.where(on[:, None], v_last, 0.0)
        active[lo:hi] = on
    return p0, v0, active


def _path_integrate(
    p0: np.ndarray,
    v0: np.ndarray,
    active: np.ndarray,
    players: np.ndarray,
    player_v: np.ndarray,
    teams: np.ndarray,
    x_lim: float,
    y_lim: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One chunk. Arrival grids for every τ are one array; the loop only carries the free mass."""
    observed = np.isfinite(players[..., 0])
    velocity = np.nan_to_num(player_v.astype(np.float64), nan=0.0)
    position = np.where(observed[..., None], players.astype(np.float64), 0.0)
    reaction = position + _PATH_REACTION_S * velocity
    n_frames, n_slots = observed.shape
    # The arrival grid is float32 (the inputs already are). The mass below stays float64.
    tau = (_PATH_DT * np.arange(_PATH_STEPS)).astype(np.float32)
    bx = p0[None, :, 0].astype(np.float32) + v0[None, :, 0].astype(np.float32) * tau[:, None]
    by = p0[None, :, 1].astype(np.float32) + v0[None, :, 1].astype(np.float32) * tau[:, None]
    outside = (np.abs(bx) > x_lim) | (np.abs(by) > y_lim)
    dx = reaction[None, :, :, 0].astype(np.float32) - bx[:, :, None]
    dy = reaction[None, :, :, 1].astype(np.float32) - by[:, :, None]
    distance = np.hypot(dx, dy)
    del dx, dy
    gap = tau[:, None, None] - (
        np.float32(_PATH_REACTION_S) + distance * np.float32(1.0 / _PATH_RUN_SPEED)
    )
    del distance
    prob = _expit(np.float32(_PATH_COEF) * gap)
    del gap
    prob *= observed
    ppcf = np.zeros((n_frames, n_slots), dtype=np.float64)
    path_out = np.zeros(n_frames, dtype=np.float64)
    sum_q = np.zeros(n_frames, dtype=np.float64)
    sum_qt = np.zeros(n_frames, dtype=np.float64)
    remaining = np.where(active, 1.0, 0.0)
    alive = active.copy()
    alive_col = alive[:, None]
    for k in range(_PATH_STEPS):
        if not alive.any():
            break
        stop = alive & outside[k]
        if stop.any():
            path_out[stop] = remaining[stop]
            alive[stop] = False
            if not alive.any():
                break
        row = prob[k]
        row *= alive_col
        weight = np.sum(row, axis=1, dtype=np.float64)
        hazard = _PATH_LAMBDA * weight
        factor = np.where(alive, np.exp(-hazard * _PATH_DT), 1.0)
        gained = remaining * (1.0 - factor)
        share = np.divide(row, weight[:, None], out=np.zeros_like(row), where=weight[:, None] > 0.0)
        ppcf += gained[:, None] * share
        sum_q += gained
        sum_qt += gained * tau[k]
        remaining *= factor
    path_out[alive] += remaining[alive]
    path_t = np.zeros(n_frames, dtype=np.float64)
    controlled = sum_q > 0.0
    if controlled.any():
        scaled = sum_qt[controlled] / np.maximum(sum_q[controlled], 1e-6)
        path_t[controlled] = scaled / _PATH_T_SCALE
    sign = ppcf[:, teams == HOME].sum(axis=1) - ppcf[:, teams == AWAY].sum(axis=1)
    return (
        sign.astype(np.float32),
        path_out.astype(np.float32),
        path_t.astype(np.float32),
        ppcf.astype(np.float32),
    )


# -- windows for training, chunks for inference ---------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Compute frames [lo, hi) and keep [keep_lo, keep_hi): the rest is context for the receptive
    field of the temporal stack."""

    lo: int
    hi: int
    keep_lo: int
    keep_hi: int


def plan_chunks(periods: tuple[Period, ...], chunk: int, margin: int) -> list[Chunk]:
    """Tile every period with kept ranges of `chunk` frames, each computed with `margin` frames of
    context on both sides (clipped to the period). The kept ranges partition the played frames."""
    if chunk < 1 or margin < 0:
        raise ValueError(f"chunk must be positive and margin non-negative, got {chunk}, {margin}")
    plan = []
    for period in periods:
        for keep_lo in range(period.start, period.stop, chunk):
            keep_hi = min(period.stop, keep_lo + chunk)
            lo, hi = max(period.start, keep_lo - margin), min(period.stop, keep_hi + margin)
            plan.append(Chunk(lo, hi, keep_lo, keep_hi))
    return plan


def sample_starts(
    periods: tuple[Period, ...], window: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """(count,) start frames of windows that lie wholly inside one period, uniform over the frames
    that can start one. A period shorter than `window` is skipped."""
    room = np.array([max(0, p.length - window + 1) for p in periods], dtype=np.float64)
    if room.sum() == 0:
        raise ValueError(f"no period holds a window of {window} frames")
    which = rng.choice(len(periods), size=count, p=room / room.sum())
    starts = np.array([periods[i].start for i in which], dtype=np.int64)
    return starts + (rng.random(count) * room[which]).astype(np.int64)
