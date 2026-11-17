"""Ball-state cues: per-frame log-likelihood ratios log P(x | dead) / P(x | alive), NaN = no opinion.

Pure, vectorised functions of a Tracking; no registrations here. Every parameter is in metres,
seconds or m/s and has a physical reading. Two rules from the data lane hold throughout:

- **A missing ball is never evidence**, except in the opt-in `ball_missing_llr` (#62, weight 0
  by default). Ball cues are NaN wherever the ball is not seen, so a gap in the ball track can
  never by itself make a frame dead (BAR-ATH: missing-ball frames sit inside set-piece windows
  *less* often than seen ones, #7).
- **Raw or cleaned input.** Nothing is clipped to the pitch (off the pitch is the best dead signal)
  and implausible ball jumps are ignored, so the cues behave the same before and after the data
  lane's cleaning (jumps > 45 m/s to NaN, PR #16).

Sign convention: positive = evidence for dead, negative = evidence for alive.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...contracts import AWAY, HOME, MatchMeta, Tracking

MAX_PLAYER_SPEED = 12.0  # m/s: above any sprint, so a larger displacement is a slot swap or jump
MAX_BALL_SPEED = 45.0  # m/s: the data lane's cleaning bound for ball jumps
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16  # Laws of the Game, metres
CENTRE_CIRCLE = 9.15


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, stops) of the True runs of a 1-D bool array, half-open."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def centred_speed(xy: np.ndarray, half: int, fps: int) -> np.ndarray:
    """|xy(t + half) - xy(t - half)| / (2 half / fps) along axis 0; NaN at the edges or if unseen.

    Works for (T, 2) and (T, S, 2). A centred difference is non-causal, so it has no lag, and a
    window of a second or so averages out positional jitter.
    """
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if half <= 0 or len(xy) <= 2 * half:
        return out
    step = np.linalg.norm(xy[2 * half :] - xy[: -2 * half], axis=-1)
    out[half:-half] = step * (fps / (2.0 * half))
    return out


def collective_speed(
    tracking: Tracking, *, window_s: float = 1.0, min_players: int = 8
) -> np.ndarray:
    """(T,) median speed (m/s) over every observed slot, NaN with fewer than `min_players`.

    Team level and ball-free, so it survives anonymous, fragmented slots and the 41% of frames
    without a ball. Displacements above MAX_PLAYER_SPEED (slot swaps) are dropped.
    """
    half = max(1, int(round(window_s * tracking.meta.fps / 2)))
    speed = centred_speed(tracking.player_xy, half, tracking.meta.fps)
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    enough = (~np.isnan(speed)).sum(axis=1) >= min_players
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    if enough.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[enough] = np.nanmedian(speed[enough], axis=1)
    return out


def ball_speed(tracking: Tracking, *, window_s: float) -> np.ndarray:
    """(T,) centred ball speed in m/s, NaN where unseen or above MAX_BALL_SPEED (a jump)."""
    half = max(1, int(round(window_s * tracking.meta.fps / 2)))
    speed = centred_speed(tracking.ball_xy, half, tracking.meta.fps)
    speed[speed > MAX_BALL_SPEED] = np.nan
    speed[~tracking.ball_seen()] = np.nan
    return speed


def slowdown_llr(
    tracking: Tracking,
    *,
    window_s: float = 1.0,
    reaction_s: float = 0.75,
    ratio_at_even: float = 1.0,
    slope: float = 3.0,
    clip: float = 3.0,
) -> np.ndarray:
    """Collective slowdown: players walk while the ball is dead and run while it is alive.

    The median player speed is divided by its own median over the played frames, so the cue is
    free of the provider's speed scale (jitter inflates BAR-ATH speeds). LLR = slope * log(even /
    ratio): zero at `ratio_at_even`, positive (dead) below it.

    Players speed up only after a restart, a reaction time later, so each frame takes the larger
    of its own speed and the speed `reaction_s` ahead: a rise is pulled forward by `reaction_s`,
    while a slowdown (the ball going out) is not moved. Fails during slow build-up play.
    """
    speed = collective_speed(tracking, window_s=window_s)
    lead = int(round(reaction_s * tracking.meta.fps))
    if 0 < lead < len(speed):
        ahead = np.full_like(speed, np.nan)
        ahead[:-lead] = speed[lead:]
        speed = np.fmax(speed, ahead)
    played = tracking.meta.played_mask() & ~np.isnan(speed)
    if not played.any():
        return np.full(tracking.n_frames, np.nan, dtype=np.float32)
    scale = float(np.median(speed[played]))
    with np.errstate(divide="ignore", invalid="ignore"):
        llr = slope * np.log(ratio_at_even * scale / np.maximum(speed, 1e-3))
    return np.clip(llr, -clip, clip).astype(np.float32)


def ball_missing_llr(
    tracking: Tracking,
    *,
    bounds_s: tuple[float, ...] | list[float] = (1.0,),
    llr: tuple[float | None, ...] | list[float | None] = (None, 10.0),
    closed: str = "left",
) -> np.ndarray:
    """The missing ball as dead-ball evidence (#62): inside each run of unseen ball (per period),
    every frame gets the LLR of the run's total length L in seconds; NaN on seen frames.

    Bins: `bounds_s` splits L into len(bounds_s) + 1 bins, and `llr` gives one value per bin
    (None = no opinion). `closed` says which side of each bound is inclusive: "right" puts
    L == bound in the lower bin, "left" in the upper. The default is the plain rule "a ball
    missing for a second or more is dead" (+10, arm B of #62).

    Rests on the provider blanking the ball through stoppages, as Metrica (BAR-ATH's provider)
    does through 96% of dead frames. Fails where the ball is lost in live play for long (a high
    ball out of shot, a crowded box), and on clean tracking it never fires.
    """
    bounds = np.asarray(bounds_s, dtype=np.float64)
    values = np.array([np.nan if v is None else float(v) for v in llr], dtype=np.float32)
    if len(values) != len(bounds) + 1:
        raise ValueError(f"ball_missing_llr: {len(values)} llr values for {len(bounds)} bounds")
    if closed not in ("left", "right"):
        raise ValueError(f"closed must be 'left' or 'right', got {closed!r}")
    side = "left" if closed == "right" else "right"
    fps = tracking.meta.fps
    seen = tracking.ball_seen()
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    for period in tracking.meta.periods:
        lo = period.start
        starts, stops = runs(~seen[lo : period.stop])
        if not len(starts):
            continue
        bins = np.searchsorted(bounds, (stops - starts) / fps, side=side)
        for start, stop, b in zip(starts, stops, bins, strict=True):
            out[lo + start : lo + stop] = values[b]
    return out


def off_pitch_mask(tracking: Tracking, margin_m: float = 0.5) -> np.ndarray:
    """(T,) bool: the ball is seen beyond a touchline or goal line by more than `margin_m`."""
    meta = tracking.meta
    x, y = tracking.ball_xy[:, 0], tracking.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        out = (np.abs(x) > meta.pitch_length / 2 + margin_m) | (
            np.abs(y) > meta.pitch_width / 2 + margin_m
        )
    return out & tracking.ball_seen()


def off_pitch_llr(tracking: Tracking, *, margin_m: float = 0.5, llr: float = 3.0) -> np.ndarray:
    """Ball off the pitch: out of play by the Laws. +llr there, NaN elsewhere (no opinion).

    The ball being *on* the pitch says nothing (balls are placed on it for restarts). Fails with
    homography error at the far touchline.
    """
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    out[off_pitch_mask(tracking, margin_m)] = llr
    return out


def static_ball_llr(
    tracking: Tracking, *, window_s: float = 1.0, max_speed: float = 0.5, llr: float = 1.0
) -> np.ndarray:
    """A ball at rest for a second is usually placed for a restart: weak +llr, NaN elsewhere.

    Set from physics only. It is NOT calibrated on IDSSE, where the provider freezes the ball on
    every dead frame; on BAR-ATH a static ball is rare (#7), so it fires rarely.
    """
    speed = ball_speed(tracking, window_s=window_s)
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        out[speed < max_speed] = llr
    return out


def ball_in_flight_llr(
    tracking: Tracking,
    *,
    window_s: float = 0.2,
    min_speed: float = 8.0,
    margin_m: float = 0.5,
    llr: float = -1.0,
) -> np.ndarray:
    """A fast ball inside the pitch was just kicked: negative llr (alive), NaN elsewhere.

    Dead-ball passes back to a taker are usually slow or off the pitch. Physics default, not
    calibrated on IDSSE (its frozen dead ball would make this look perfect).
    """
    speed = ball_speed(tracking, window_s=window_s)
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        out[(speed > min_speed) & ~off_pitch_mask(tracking, margin_m)] = llr
    out[~tracking.ball_seen()] = np.nan
    return out


def restart_spot_mask(
    tracking: Tracking, *, corner_m: float = 2.5, line_m: float = 1.0, margin_m: float = 0.5
) -> np.ndarray:
    """(T,) bool: the ball is seen where a restart is taken -- a corner, a goal area, a touchline."""
    meta = tracking.meta
    half_l, half_w = meta.pitch_length / 2, meta.pitch_width / 2
    ax, ay = np.abs(tracking.ball_xy[:, 0]), np.abs(tracking.ball_xy[:, 1])
    with np.errstate(invalid="ignore"):
        corner = np.hypot(half_l - ax, half_w - ay) < corner_m
        goal_area = (ax > half_l - GOAL_AREA_DEPTH - margin_m) & (
            ay < GOAL_AREA_HALF_WIDTH + margin_m
        )
        touchline = ay > half_w - line_m
    return (corner | goal_area | touchline) & tracking.ball_seen()


def restart_spot_llr(
    tracking: Tracking, *, window_s: float = 1.0, max_speed: float = 1.5, llr: float = 1.0
) -> np.ndarray:
    """A slow ball at a corner, in a goal area or on a touchline is being set for a restart."""
    speed = ball_speed(tracking, window_s=window_s)
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        out[restart_spot_mask(tracking) & (speed < max_speed)] = llr
    return out


def home_defends_sign(tracking: Tracking) -> dict[int, float]:
    """Per period: +1 if the home team defends the +x goal, -1 if it defends -x.

    Each team's centroid sits on average towards its own goal (keeper and back line behind the
    ball), so the sign of the median of (home centroid x - away centroid x) gives the orientation.
    0.0 when a period has no frame with both teams seen.
    """
    out: dict[int, float] = {}
    x = tracking.player_xy[..., 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home = np.nanmean(x[:, tracking.team_slots(HOME)], axis=1)
        away = np.nanmean(x[:, tracking.team_slots(AWAY)], axis=1)
    for period in tracking.meta.periods:
        diff = (home - away)[period.start : period.stop]
        diff = diff[~np.isnan(diff)]
        out[period.index] = float(np.sign(np.median(diff))) if len(diff) else 0.0
    return out


def period_index(meta: MatchMeta) -> np.ndarray:
    """(T,) int: the period index of each frame, 0 outside the periods."""
    out = np.zeros(meta.n_frames, dtype=np.int64)
    for period in meta.periods:
        out[period.start : period.stop] = period.index
    return out
