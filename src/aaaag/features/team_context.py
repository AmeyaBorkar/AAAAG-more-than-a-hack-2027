"""Team-in-possession frame features, version 1: 109 label-free columns per frame.

frame_features(tracking) returns FrameFeatures. X is float32 with shape (T, 109).
NaN means undefined, which LightGBM reads as missing. feature_names() and
feature_groups() are the column order and the G1..G6 tag of each column. Both are
fixed and need no tracking. Numbers match wf1_team_learnability_features VERSION 1.

Signals: ball and player positions, slot teams, fps, periods and pitch length.
A slot index is used only to follow that slot across the 0.16 s velocity stencil.
Delivered player_speed, ball_extra, labels and tags are ignored. Vectorised over
frames. Each chunk is 8192 frames, which bounds the (frames x slots) distance
matrix. Deterministic: no randomness.

- G1 ball proximity: per team, the 1st, 2nd and 3rd nearest slot distance to the
  ball, counts within 1, 2, 3 and 5 m, and that nearest slot's speed, relative
  speed and closing speed; plus home-minus-away contrasts.
- G2 control memory: last controller (nearest slot within 1.0 m for at least 3
  frames): last team, time since that run ended and started, its length, the
  previous team, time since the last change of team, the raw control state and
  its age.
- G3 ball kinematics: speed over 0.16 s and 0.48 s, acceleration, direction
  change, time since the ball was last seen, displacement over the previous
  0.5, 1 and 2 s. Shifts are int(round(seconds * fps)) frames (half-even).
- G4 team shape in the home-attacks-towards-+X frame, inferred per period without
  labels (home_defends_sign: each team's centroid sits towards its own goal).
  Ball position and distance to each goal; per team, centroid, median, defensive
  and attacking lines relative to the ball, spread, players ahead of the ball,
  lateral offset of the centroid. Relative to the last SEEN ball, so a gap stays
  defined.
- G5 density and pressure: counts within 10 and 15 m, opponents near each team's
  nearest player, the gap between the nearest home and nearest away slot, and the
  mean speed of slots within 10 m of the ball.
- G6 temporal context: centred windows and windows that reach only backward or
  only forward (the current frame is included), over 1, 3 and 10 s, of who is
  nearest, who is in control, ball territory and visibility; the next controlling
  run (non-causal, as the decoders are); displacement over the next 0.5, 1 and
  2 s; time until the ball is next seen; nearby changes of controlling team.

Fails when the ball is unseen for a long stretch (G1 and G3 are NaN; G2, G4 and
G6 still speak), when a team has no slots, and when a period never shows both
teams (the attacking direction is unknown and G4 is NaN). Ball speed, acceleration
and direction change use a stencil that is not clipped to the period, so a gap
shorter than 0.48 s mixes the two periods. Displacement, window means and control
runs are clipped to the period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aaaag.contracts import AWAY, HOME, Tracking
from aaaag.evidence.ballstate.cues import home_defends_sign
from aaaag.evidence.nearest_player import nearest_team_codes

__all__ = [
    "FEATURE_VERSION",
    "FrameFeatures",
    "feature_groups",
    "feature_names",
    "frame_features",
]

FEATURE_VERSION = "1"
CHUNK = 8192
RADII_G1 = (1.0, 2.0, 3.0, 5.0)
RADII_G5 = (10.0, 15.0)
MAX_BALL_SPEED = 45.0  # m/s, as the champion's cues: faster is a tracking jump
REACH_M, HOLD_FRAMES = 1.0, 3  # the champion's control_possession priors
GROUP_NAMES = ("G1", "G2", "G3", "G4", "G5", "G6")


@dataclass(frozen=True)
class FrameFeatures:
    """One match of team-context features. X is float32 (T, 109); NaN = undefined."""

    X: np.ndarray
    names: tuple[str, ...]
    groups: tuple[str, ...]

    def __post_init__(self) -> None:
        width = len(self.names)
        if self.X.ndim != 2 or self.X.shape[1] != width or self.X.dtype != np.float32:
            raise ValueError(f"X must be float32 of shape (T, {width})")
        if not isinstance(self.names, tuple) or not isinstance(self.groups, tuple):
            raise TypeError("names and groups must be tuples")
        if len(self.groups) != width:
            raise ValueError("groups must align with names")


class Table:
    """Ordered named columns with a group tag, assembled into a float32 matrix."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.cols: dict[str, np.ndarray] = {}
        self.group: dict[str, str] = {}

    def add(self, name: str, group: str, values: np.ndarray) -> None:
        if name in self.cols:
            raise ValueError(f"duplicate feature {name}")
        values = np.asarray(values)
        if values.shape != (self.n,):
            raise ValueError(f"{name}: shape {values.shape}")
        self.cols[name] = values.astype(np.float32)
        self.group[name] = group

    def matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        names = list(self.cols)
        return (
            np.stack([self.cols[k] for k in names], axis=1),
            names,
            [self.group[k] for k in names],
        )


# -- small vectorised helpers -------------------------------------------------------------------


def period_bounds(tracking: Tracking) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,) period start, period stop per frame (t, t for frames outside a period) and index."""
    n = tracking.n_frames
    start, stop = np.arange(n), np.arange(n)
    index = np.zeros(n, dtype=np.int64)
    for period in tracking.meta.periods:
        start[period.start : period.stop] = period.start
        stop[period.start : period.stop] = period.stop
        index[period.start : period.stop] = period.index
    return start, stop, index


def centred_diff(a: np.ndarray, half: int, fps: int) -> np.ndarray:
    """(a[t + half] - a[t - half]) / (2 half / fps) along axis 0; NaN at the edges / missing."""
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if a.shape[0] > 2 * half:
        out[half:-half] = (a[2 * half :] - a[: -2 * half]) / (2.0 * half / fps)
    return out


def shifted(a: np.ndarray, k: int) -> np.ndarray:
    """out[t] = a[t - k] (k > 0: the past; k < 0: the future), NaN where out of range."""
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def window_mean(
    x: np.ndarray, lo: np.ndarray, hi: np.ndarray, back: int, fwd: int
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and valid count of x over [t - back, t + fwd], clipped to the frame's period
    [lo, hi) (`period_bounds`). O(T) through cumulative sums."""
    n = len(x)
    t = np.arange(n)
    a = np.maximum(lo, t - back)
    b = np.minimum(hi, t + fwd + 1)
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0), dtype=np.float64)))
    ccnt = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    s, c = csum[b] - csum[a], ccnt[b] - ccnt[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return mean, c


def window_sum(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, back: int, fwd: int) -> np.ndarray:
    n = len(x)
    t = np.arange(n)
    a = np.maximum(lo, t - back)
    b = np.minimum(hi, t + fwd + 1)
    csum = np.concatenate(([0.0], np.cumsum(np.nan_to_num(x), dtype=np.float64)))
    return csum[b] - csum[a]


def ffill_ball(tracking: Tracking) -> np.ndarray:
    """(T, 2) float64 last SEEN ball position at or before t inside the period, back-filled from the
    first seen frame at a period start; NaN outside the periods or in a period with no ball."""
    ball = tracking.ball_xy.astype(np.float64)
    seen = ~np.isnan(ball[:, 0])
    out = np.full(ball.shape, np.nan)
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        s = seen[lo:hi]
        if not s.any():
            continue
        idx = np.where(s, np.arange(hi - lo), -1)
        last = np.maximum.accumulate(idx)
        first = int(np.flatnonzero(s)[0])
        last = np.where(last < 0, first, last)
        out[lo:hi] = ball[lo:hi][last]
    return out


def since_seen_until_seen(tracking: Tracking) -> tuple[np.ndarray, np.ndarray]:
    """(T,) seconds since the ball was last seen (0 if seen now) and until next seen.

    NaN where that event does not exist inside the period.
    """
    n, fps = tracking.n_frames, tracking.meta.fps
    seen = tracking.ball_seen()
    since = np.full(n, np.nan)
    until = np.full(n, np.nan)
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        s = seen[lo:hi]
        t = np.arange(hi - lo)
        last = np.maximum.accumulate(np.where(s, t, -(10**9)))
        nxt = np.minimum.accumulate(np.where(s, t, 10**9)[::-1])[::-1]
        a = (t - last) / fps
        b = (nxt - t) / fps
        since[lo:hi] = np.where(last < 0, np.nan, a)
        until[lo:hi] = np.where(nxt >= 10**9, np.nan, b)
    return since, until


def control_features(codes: np.ndarray, tracking: Tracking) -> dict[str, np.ndarray]:
    """G2 and the next-run part of G6 from the per-frame control codes (-1 none, 0 home, 1 away)."""
    n, fps = tracking.n_frames, tracking.meta.fps
    names = (
        "ctl_now",
        "ctl_age_s",
        "last_team",
        "since_last_end_s",
        "since_last_start_s",
        "last_run_len_s",
        "prev_team",
        "since_change_s",
        "next_team",
        "until_next_s",
        "next_run_len_s",
    )
    out = {k: np.full(n, np.nan) for k in names}
    out["change_event"] = np.zeros(n)  # 1 on the first frame of a run whose team differs
    sign = np.array([1.0, -1.0])
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        c = codes[lo:hi].astype(np.int16)
        m = len(c)
        t = np.arange(m)
        change = np.flatnonzero(np.diff(c) != 0) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [m]))
        length = stops - starts
        run_of = np.repeat(np.arange(len(starts)), length)
        out["ctl_age_s"][lo:hi] = (t - starts[run_of] + 1) / fps
        out["ctl_now"][lo:hi] = np.where(c == 0, 1.0, np.where(c == 1, -1.0, 0.0))
        keep = (c[starts] >= 0) & (length >= HOLD_FRAMES)
        qs, qe, qt = starts[keep], stops[keep], c[starts[keep]]
        if not len(qs):
            continue
        team = sign[qt]
        flag = np.ones(len(qs), dtype=bool)
        flag[1:] = qt[1:] != qt[:-1]
        block = np.maximum.accumulate(np.where(flag, np.arange(len(qs)), 0))
        ev = qs[flag][1:]  # a change of controlling team (the first run is not a change)
        out["change_event"][lo + ev] = 1.0
        last = np.searchsorted(qs, t, side="right") - 1
        has = last >= 0
        li = last[has]
        sel = lo + t[has]
        out["last_team"][sel] = team[li]
        out["since_last_end_s"][sel] = np.maximum(0, t[has] - (qe[li] - 1)) / fps
        out["since_last_start_s"][sel] = (t[has] - qs[li]) / fps
        out["last_run_len_s"][sel] = (qe[li] - qs[li]) / fps
        out["since_change_s"][sel] = (t[has] - qs[block[li]]) / fps
        prev_ok = li >= 1
        pt = np.full(len(li), np.nan)
        pt[prev_ok] = team[li[prev_ok] - 1]
        out["prev_team"][sel] = pt
        nxt = np.searchsorted(qe - 1, t, side="left")
        has_n = nxt < len(qs)
        ni = nxt[has_n]
        seln = lo + t[has_n]
        out["next_team"][seln] = team[ni]
        out["until_next_s"][seln] = np.maximum(0, qs[ni] - t[has_n]) / fps
        out["next_run_len_s"][seln] = (qe[ni] - qs[ni]) / fps
    return out


def _smallest3(d: np.ndarray) -> np.ndarray:
    """(B, 3) three smallest values per row (inf-padded)."""
    if d.shape[1] < 3:
        d = np.concatenate([d, np.full((len(d), 3 - d.shape[1]), np.inf)], axis=1)
    part = np.partition(d, 2, axis=1)[:, :3]
    part.sort(axis=1)
    return part


def _inf_nan(a: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(a), a, np.nan)


def _feature_spec() -> tuple[tuple[str, str], ...]:
    """(name, group) in the column order of frame_features. No tracking."""
    rows: list[tuple[str, str]] = []

    def add(name: str, group: str) -> None:
        rows.append((name, group))

    for tag in ("h", "a"):
        for j in (1, 2, 3):
            add(f"{tag}_d{j}", "G1")
        for radius in RADII_G1:
            add(f"{tag}_n{int(radius)}", "G1")
        for kind in ("speed", "relspeed", "closing"):
            add(f"{tag}_nn_{kind}", "G1")
    add("d1_diff", "G1")
    add("d1_logratio", "G1")
    add("n3_diff", "G1")
    for key in (
        "ctl_now",
        "ctl_age_s",
        "last_team",
        "since_last_end_s",
        "since_last_start_s",
        "last_run_len_s",
        "prev_team",
        "since_change_s",
    ):
        add(key, "G2")
    for key in (
        "ball_speed",
        "ball_speed_long",
        "ball_acc",
        "ball_dir_change",
        "ball_seen",
        "t_since_seen_s",
    ):
        add(key, "G3")
    for sec in (0.5, 1.0, 2.0):
        add(f"disp_past_{sec:g}s", "G3")
    for key in ("ball_X", "ball_absY", "ball_dist_goal_att_h", "ball_dist_goal_att_a"):
        add(key, "G4")
    for tag in ("h", "a"):
        for kind in (
            "cx_rel",
            "med_rel",
            "defline_rel",
            "attline_rel",
            "sd_x",
            "sd_y",
            "n_ahead",
            "lat_off",
            "cx_abs",
        ):
            add(f"{tag}_{kind}", "G4")
    for tag in ("h", "a"):
        for radius in RADII_G5:
            add(f"{tag}_n{int(radius)}", "G5")
        for kind in ("opp_min", "opp_n2", "opp_n4"):
            add(f"{tag}_{kind}", "G5")
    for key in ("n_all_5", "n_all_10", "duel_gap", "near10_mean_speed"):
        add(key, "G5")
    for width in (1, 3, 10):
        add(f"w{width}_near_sign", "G6")
        add(f"w{width}_d1_diff", "G6")
        add(f"w{width}_ctl_bal", "G6")
    add("past3_ctl_bal", "G6")
    add("fut3_ctl_bal", "G6")
    add("past3_near_sign", "G6")
    add("fut3_near_sign", "G6")
    add("past3_d1_diff", "G6")
    add("fut3_d1_diff", "G6")
    add("w3_ctl_cov", "G6")
    for width in (3, 10):
        add(f"w{width}_ball_X", "G6")
        add(f"w{width}_ball_seen", "G6")
        add(f"w{width}_geo", "G6")
    for width in (1, 3):
        add(f"w{width}_ball_speed", "G6")
    for sec in (0.5, 1.0, 2.0):
        add(f"disp_next_{sec:g}s", "G6")
    for key in ("next_team", "until_next_s", "next_run_len_s"):
        add(key, "G6")
    add("until_seen_s", "G6")
    add("n_changes_pm10s", "G6")
    add("n_changes_pm3s", "G6")
    spec = tuple(rows)
    if len(spec) != 109 or len({name for name, _group in spec}) != 109:
        raise RuntimeError(f"feature schema has {len(spec)} columns, expected 109 unique")
    return spec


_FEATURE_SPEC = _feature_spec()
_NAMES = tuple(name for name, _group in _FEATURE_SPEC)
_GROUPS = tuple(group for _name, group in _FEATURE_SPEC)


def feature_names() -> tuple[str, ...]:
    """Column names in order. Fixed; needs no tracking."""
    return _NAMES


def feature_groups() -> tuple[str, ...]:
    """Group tag per column (G1..G6), aligned with feature_names(). Needs no tracking."""
    return _GROUPS


def _check_schema(names: list[str], groups: list[str]) -> None:
    want_n, want_g = feature_names(), feature_groups()
    if len(names) != len(want_n) or len(groups) != len(want_g):
        raise RuntimeError(f"schema length {len(names)}/{len(groups)} vs {len(want_n)}")
    for i, (name, group, exp_n, exp_g) in enumerate(
        zip(names, groups, want_n, want_g, strict=True)
    ):
        if name != exp_n or group != exp_g:
            raise RuntimeError(f"schema drift at {i}: {name}/{group} vs {exp_n}/{exp_g}")


# -- the feature function ----------------------------------------------------------------------


def frame_features(tracking: Tracking) -> FrameFeatures:
    """(T, 109) team-context features. See the module docstring."""
    n, fps = tracking.n_frames, tracking.meta.fps
    P = tracking.player_xy
    ball = tracking.ball_xy
    seen = tracking.ball_seen()
    hs, as_ = tracking.team_slots(HOME), tracking.team_slots(AWAY)
    lo_p, hi_p, pidx = period_bounds(tracking)
    played = tracking.meta.played_mask()
    length = tracking.meta.pitch_length

    # orientation: X is positive towards the goal HOME attacks, inferred per period without labels
    defends = home_defends_sign(tracking)
    sgn = np.zeros(n)
    for period in tracking.meta.periods:
        sgn[period.start : period.stop] = -defends.get(period.index, 0.0)
    oriented = sgn != 0

    # velocities (0.16 s centred differences) of the slots and the ball
    vp = centred_diff(P.astype(np.float64), 2, fps)  # (T, S, 2)
    vb = centred_diff(ball.astype(np.float64), 2, fps)
    spb = np.linalg.norm(vb, axis=1)
    jump = spb > MAX_BALL_SPEED
    vb[jump] = np.nan
    spb[jump] = np.nan
    speed_p = np.linalg.norm(vp, axis=2)  # (T, S)
    bref = ffill_ball(tracking)

    T = Table(n)
    # ---- per-chunk geometry (G1, G4, G5) ------------------------------------------------------
    keys = {}

    def put(name: str, arr: np.ndarray, lo: int, hi: int) -> None:
        if name not in keys:
            keys[name] = np.full(n, np.nan)
        keys[name][lo:hi] = arr

    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        b = ball[lo:hi].astype(np.float64)
        bs = seen[lo:hi]
        Pc = P[lo:hi].astype(np.float64)
        d = np.sqrt(((Pc - b[:, None, :]) ** 2).sum(axis=2))
        d[np.isnan(d)] = np.inf
        d[~bs] = np.inf
        near = {}
        for tag, slots, opp in (("h", hs, as_), ("a", as_, hs)):
            dk = d[:, slots]
            three = _smallest3(dk)
            for j in range(3):
                put(f"{tag}_d{j + 1}", np.where(bs, _inf_nan(three[:, j]), np.nan), lo, hi)
            for r in RADII_G1 + RADII_G5:
                cnt = (dk <= r).sum(axis=1).astype(np.float64)
                put(f"{tag}_n{int(r)}", np.where(bs, cnt, np.nan), lo, hi)
            arg = dk.argmin(axis=1)
            gslot = slots[arg]
            ok = bs & np.isfinite(dk.min(axis=1))
            rows = np.arange(hi - lo)
            vs = vp[lo:hi][rows, gslot]  # (B, 2) nearest slot velocity
            pn = Pc[rows, gslot]  # (B, 2) nearest slot position
            vbc = vb[lo:hi]
            rel = np.linalg.norm(vs - vbc, axis=1)
            dist = np.linalg.norm(b - pn, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                u = (b - pn) / dist[:, None]
                closing = -((vbc - vs) * u).sum(axis=1)
            closing[dist < 0.05] = np.nan
            put(f"{tag}_nn_speed", np.where(ok, np.linalg.norm(vs, axis=1), np.nan), lo, hi)
            put(f"{tag}_nn_relspeed", np.where(ok, rel, np.nan), lo, hi)
            put(f"{tag}_nn_closing", np.where(ok, closing, np.nan), lo, hi)
            near[tag] = (ok, pn)
            # pressure on this team's nearest player: opponents around it (G5)
            do = np.sqrt(((Pc[:, opp] - pn[:, None, :]) ** 2).sum(axis=2))
            do[np.isnan(do)] = np.inf
            put(f"{tag}_opp_min", np.where(ok, _inf_nan(do.min(axis=1)), np.nan), lo, hi)
            put(f"{tag}_opp_n2", np.where(ok, (do <= 2.0).sum(axis=1), np.nan), lo, hi)
            put(f"{tag}_opp_n4", np.where(ok, (do <= 4.0).sum(axis=1), np.nan), lo, hi)
        ok2 = near["h"][0] & near["a"][0]
        gap = np.linalg.norm(near["h"][1] - near["a"][1], axis=1)
        put("duel_gap", np.where(ok2, gap, np.nan), lo, hi)
        within = d <= 10.0
        with np.errstate(invalid="ignore", divide="ignore"):
            msp = np.where(within, np.nan_to_num(speed_p[lo:hi]), 0.0).sum(axis=1) / np.maximum(
                (within & ~np.isnan(speed_p[lo:hi])).sum(axis=1), 1
            )
        put("near10_mean_speed", np.where(bs & within.any(axis=1), msp, np.nan), lo, hi)

        # team shape (G4), relative to the last seen ball
        br = bref[lo:hi]
        sg = sgn[lo:hi]
        for tag, slots, orient in (("h", hs, 1.0), ("a", as_, -1.0)):
            Xk = orient * sg[:, None] * Pc[:, slots, 0]
            yk = Pc[:, slots, 1]
            bx = orient * sg * br[:, 0]
            by = br[:, 1]
            m = ~np.isnan(Xk)
            cnt = m.sum(axis=1)
            safe = np.maximum(cnt, 1)
            cx = np.where(m, Xk, 0.0).sum(axis=1) / safe
            cy = np.where(m, yk, 0.0).sum(axis=1) / safe
            sdx = np.sqrt(np.where(m, (Xk - cx[:, None]) ** 2, 0.0).sum(axis=1) / safe)
            sdy = np.sqrt(np.where(m, (yk - cy[:, None]) ** 2, 0.0).sum(axis=1) / safe)
            Xs = np.sort(Xk, axis=1)  # NaN last
            rows = np.arange(hi - lo)
            med = Xs[rows, np.maximum(cnt - 1, 0) // 2]
            dl = Xs[:, 1:4].mean(axis=1) if Xs.shape[1] >= 4 else np.full(hi - lo, np.nan)
            top = np.stack([Xs[rows, np.clip(cnt - 1 - j, 0, None)] for j in range(3)], axis=1)
            al = top.mean(axis=1)
            good = (cnt >= 6) & oriented[lo:hi] & ~np.isnan(bx)
            ahead = np.where(m, Xk > (bx[:, None] + 0.5), False).sum(axis=1)
            put(f"{tag}_cx_rel", np.where(good, cx - bx, np.nan), lo, hi)
            put(f"{tag}_med_rel", np.where(good, med - bx, np.nan), lo, hi)
            put(f"{tag}_defline_rel", np.where(good & (cnt >= 8), dl - bx, np.nan), lo, hi)
            put(f"{tag}_attline_rel", np.where(good & (cnt >= 8), al - bx, np.nan), lo, hi)
            put(f"{tag}_sd_x", np.where(good, sdx, np.nan), lo, hi)
            put(f"{tag}_sd_y", np.where(good, sdy, np.nan), lo, hi)
            put(f"{tag}_n_ahead", np.where(good, ahead, np.nan), lo, hi)
            put(f"{tag}_lat_off", np.where(good, np.abs(cy - by), np.nan), lo, hi)
            put(f"{tag}_cx_abs", np.where(good, cx, np.nan), lo, hi)

    # ---- G1 -------------------------------------------------------------------------------
    for tag in ("h", "a"):
        for j in (1, 2, 3):
            T.add(f"{tag}_d{j}", "G1", keys[f"{tag}_d{j}"])
        for r in RADII_G1:
            T.add(f"{tag}_n{int(r)}", "G1", keys[f"{tag}_n{int(r)}"])
        for s in ("speed", "relspeed", "closing"):
            T.add(f"{tag}_nn_{s}", "G1", keys[f"{tag}_nn_{s}"])
    d1_diff = keys["a_d1"] - keys["h_d1"]
    T.add("d1_diff", "G1", d1_diff)
    T.add("d1_logratio", "G1", np.log((keys["a_d1"] + 0.1) / (keys["h_d1"] + 0.1)))
    T.add("n3_diff", "G1", keys["h_n3"] - keys["a_n3"])

    # ---- G2 control memory -----------------------------------------------------------------
    codes = nearest_team_codes(tracking, max_distance_m=REACH_M)
    ctl = control_features(codes, tracking)
    for k in (
        "ctl_now",
        "ctl_age_s",
        "last_team",
        "since_last_end_s",
        "since_last_start_s",
        "last_run_len_s",
        "prev_team",
        "since_change_s",
    ):
        T.add(k, "G2", ctl[k])

    # ---- G3 ball kinematics ----------------------------------------------------------------
    T.add("ball_speed", "G3", spb)
    vb6 = centred_diff(ball.astype(np.float64), 6, fps)
    sp6 = np.linalg.norm(vb6, axis=1)
    sp6[sp6 > MAX_BALL_SPEED] = np.nan
    T.add("ball_speed_long", "G3", sp6)
    acc = np.linalg.norm(centred_diff(vb, 2, fps), axis=1)
    T.add("ball_acc", "G3", acc)
    before, after = shifted(vb, 3), shifted(vb, -3)
    nb, na = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (before * after).sum(axis=1) / (nb * na)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    ang[~((nb > 1.0) & (na > 1.0))] = np.nan
    T.add("ball_dir_change", "G3", ang)
    since, until = since_seen_until_seen(tracking)
    T.add("ball_seen", "G3", seen.astype(np.float64))
    T.add("t_since_seen_s", "G3", since)
    ball64 = ball.astype(np.float64)
    for sec in (0.5, 1.0, 2.0):
        k = int(round(sec * fps))
        past = shifted(ball64, k)
        disp = np.linalg.norm(ball64 - past, axis=1)
        disp[(pidx != shifted(pidx.astype(np.float64), k)) | ~played] = np.nan
        T.add(f"disp_past_{sec:g}s", "G3", disp)

    # ---- G4 team shape ---------------------------------------------------------------------
    ball_X = np.where(oriented, sgn * bref[:, 0], np.nan)
    T.add("ball_X", "G4", ball_X)
    T.add("ball_absY", "G4", np.abs(bref[:, 1]))
    T.add("ball_dist_goal_att_h", "G4", np.hypot(length / 2 - ball_X, bref[:, 1]))
    T.add("ball_dist_goal_att_a", "G4", np.hypot(length / 2 + ball_X, bref[:, 1]))
    for tag in ("h", "a"):
        for s in (
            "cx_rel",
            "med_rel",
            "defline_rel",
            "attline_rel",
            "sd_x",
            "sd_y",
            "n_ahead",
            "lat_off",
            "cx_abs",
        ):
            T.add(f"{tag}_{s}", "G4", keys[f"{tag}_{s}"])

    # ---- G5 density and pressure ---------------------------------------------------------
    for tag in ("h", "a"):
        for r in RADII_G5:
            T.add(f"{tag}_n{int(r)}", "G5", keys[f"{tag}_n{int(r)}"])
        for s in ("opp_min", "opp_n2", "opp_n4"):
            T.add(f"{tag}_{s}", "G5", keys[f"{tag}_{s}"])
    T.add("n_all_5", "G5", keys["h_n5"] + keys["a_n5"])
    T.add("n_all_10", "G5", keys["h_n10"] + keys["a_n10"])
    T.add("duel_gap", "G5", keys["duel_gap"])
    T.add("near10_mean_speed", "G5", keys["near10_mean_speed"])

    # ---- G6 temporal context -----------------------------------------------------------------
    near_sign = np.where(np.isnan(d1_diff), np.nan, np.sign(d1_diff))
    d1_clip = np.clip(d1_diff, -10.0, 10.0)
    ctl_bal = np.where(seen, ctl["ctl_now"], np.nan)
    ctl_cov = np.where(seen, np.abs(ctl["ctl_now"]), np.nan)
    geo = keys["h_cx_rel"] - keys["a_cx_rel"]

    def _seconds(span: float) -> int:
        return int(round(span * fps))

    def win(name: str, x: np.ndarray, back_s: float, fwd_s: float) -> None:
        mean, _ = window_mean(x, lo_p, hi_p, _seconds(back_s), _seconds(fwd_s))
        mean[~played] = np.nan
        T.add(name, "G6", mean)

    for w in (1, 3, 10):
        win(f"w{w}_near_sign", near_sign, w, w)
        win(f"w{w}_d1_diff", d1_clip, w, w)
        win(f"w{w}_ctl_bal", ctl_bal, w, w)
    win("past3_ctl_bal", ctl_bal, 3, 0)
    win("fut3_ctl_bal", ctl_bal, 0, 3)
    win("past3_near_sign", near_sign, 3, 0)
    win("fut3_near_sign", near_sign, 0, 3)
    win("past3_d1_diff", d1_clip, 3, 0)
    win("fut3_d1_diff", d1_clip, 0, 3)
    win("w3_ctl_cov", ctl_cov, 3, 3)
    for w in (3, 10):
        win(f"w{w}_ball_X", ball_X, w, w)
        win(f"w{w}_ball_seen", seen.astype(np.float64), w, w)
        win(f"w{w}_geo", geo, w, w)
    for w in (1, 3):
        win(f"w{w}_ball_speed", spb, w, w)
    for k_s in (0.5, 1.0, 2.0):
        k = _seconds(k_s)
        fut = shifted(ball64, -k)
        disp = np.linalg.norm(fut - ball64, axis=1)
        disp[(pidx != shifted(pidx.astype(np.float64), -k)) | ~played] = np.nan
        T.add(f"disp_next_{k_s:g}s", "G6", disp)
    for k in ("next_team", "until_next_s", "next_run_len_s"):
        T.add(k, "G6", ctl[k])
    T.add("until_seen_s", "G6", until)
    chg = ctl["change_event"]
    T.add("n_changes_pm10s", "G6", window_sum(chg, lo_p, hi_p, _seconds(10), _seconds(10)))
    T.add("n_changes_pm3s", "G6", window_sum(chg, lo_p, hi_p, _seconds(3), _seconds(3)))

    F, names, groups = T.matrix()
    F[~played] = np.nan
    _check_schema(names, groups)
    return FrameFeatures(X=F, names=tuple(names), groups=tuple(groups))
