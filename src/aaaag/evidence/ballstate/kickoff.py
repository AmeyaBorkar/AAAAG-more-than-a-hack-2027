"""Kick-off detection from the formation: both teams wholly in their own halves, standing still.

Ball-free, so it works through the long ball gaps. It finds the opening kick-off of each period
(everything before it is a stoppage, like BAR-ATH's graded pre-kick-off stretch, found here by
detection rather than by any frame number) and the kick-offs after goals. It also says who kicks
off: the team with the player nearest the centre spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts import HOME, Evidence, MatchBundle, Tracking
from ...registry import EVIDENCE
from .cues import CENTRE_CIRCLE, collective_speed, home_defends_sign, runs


@dataclass(frozen=True)
class Kickoff:
    period: int
    setup_start: int  # first frame of the formation
    kick: int  # first alive frame (estimated)
    team: int  # HOME or AWAY: the side kicking off
    opening: bool  # the first kick-off of its period


def own_half_share(tracking: Tracking, *, side_tol_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(T,) share of observed slots in their own half, and (T,) count of observed slots."""
    x = tracking.player_xy[..., 0]
    sign = np.zeros(tracking.n_frames, dtype=np.float32)
    orientation = home_defends_sign(tracking)
    for period in tracking.meta.periods:
        sign[period.start : period.stop] = orientation[period.index]
    toward_home_goal = x * sign[:, None]  # > 0 means in the home team's half
    home = tracking.slot_team == HOME
    with np.errstate(invalid="ignore"):
        ok = np.where(home[None, :], toward_home_goal > -side_tol_m, toward_home_goal < side_tol_m)
    seen = ~np.isnan(x)
    count = seen.sum(axis=1)
    share = np.where(count > 0, (ok & seen).sum(axis=1) / np.maximum(count, 1), 0.0)
    share[sign == 0] = 0.0
    return share.astype(np.float32), count


def detect_kickoffs(
    tracking: Tracking,
    *,
    side_tol_m: float = 1.0,
    min_share: float = 0.95,
    min_players: int = 14,
    min_hold_s: float = 2.0,
    bridge_s: float = 0.5,
    max_speed_ratio: float = 0.8,
    kick_lead_s: float = 0.5,
    kicker_m: float = 3.0,
    circle_m: float = CENTRE_CIRCLE - 1.0,
    max_opening_s: float = 300.0,
) -> list[Kickoff]:
    """Kick-offs: runs of at least `min_hold_s` with `min_share` of slots in their own half.

    Breaks shorter than `bridge_s` are bridged (one jittery slot on the line). A run counts only
    if the players are slow (median collective speed below `max_speed_ratio` of the match
    median): a team building up deep against a mid block can be in its own half, but it is not
    standing still. In the last second before the kick, one player must stand within `kicker_m` of
    the centre spot and every opponent of that player must be at least `circle_m` from it (the
    Laws keep them out of the centre circle). The kick is `kick_lead_s` before the formation
    breaks, because the forwards cross the line just after the kick. The opening kick-off must
    start within `max_opening_s` of the period start. Fails when slots duplicate across the halfway line or players are missing.
    """
    meta = tracking.meta
    fps = meta.fps
    share, count = own_half_share(tracking, side_tol_m=side_tol_m)
    formed = (share >= min_share) & (count >= min_players)
    gap_starts, gap_stops = runs(~formed)
    for start, stop in zip(gap_starts, gap_stops, strict=True):
        if 0 < start and stop < len(formed) and stop - start <= bridge_s * fps:
            formed[start:stop] = True

    speed = collective_speed(tracking)
    played = meta.played_mask() & ~np.isnan(speed)
    scale = float(np.median(speed[played])) if played.any() else np.nan
    centre_dist = np.hypot(tracking.player_xy[..., 0], tracking.player_xy[..., 1])

    out: list[Kickoff] = []
    for period in meta.periods:
        window = formed[period.start : period.stop]
        starts, stops = runs(window)
        opening_found = False
        for start, stop in zip(starts + period.start, stops + period.start, strict=True):
            if stop - start < min_hold_s * fps:
                continue
            run_speed = speed[start:stop]
            run_speed = run_speed[~np.isnan(run_speed)]
            if not len(run_speed) or not np.median(run_speed) <= max_speed_ratio * scale:
                continue
            kick = max(start + 1, stop - int(round(kick_lead_s * fps)))
            tail = centre_dist[max(start, kick - fps) : kick]
            nearest = np.where(np.isnan(tail), np.inf, tail).min(axis=0)
            kicker = int(nearest.argmin())
            if not nearest[kicker] <= kicker_m:
                continue
            team = int(tracking.slot_team[kicker])
            if nearest[tracking.slot_team != team].min(initial=np.inf) < circle_m:
                continue
            opening = not opening_found and start - period.start <= max_opening_s * fps
            opening_found = opening_found or opening
            out.append(Kickoff(period.index, int(start), int(kick), team, opening))
    return out


def kickoff_llr(
    tracking: Tracking,
    kickoffs: list[Kickoff],
    *,
    llr: float = 4.0,
    after_s: float = 1.0,
) -> np.ndarray:
    """(T,) +llr over every set-up (and before the opening kick), -llr just after each kick."""
    out = np.full(tracking.n_frames, np.nan, dtype=np.float32)
    starts = {p.index: p.start for p in tracking.meta.periods}
    for ko in kickoffs:
        first = starts[ko.period] if ko.opening else ko.setup_start
        out[first : ko.kick] = llr
        out[ko.kick : ko.kick + int(round(after_s * tracking.meta.fps))] = -llr
    return out


@EVIDENCE.register("ballstate_kickoff", version="1", example={})
def ballstate_kickoff(
    bundle: MatchBundle, *, p_dead: float = 0.98, team_p: float = 0.8, after_s: float = 1.0
) -> Evidence:
    """D from each kick-off set-up (and from the period start to the opening kick-off), then the
    kicking team for `after_s`. NaN everywhere else.

    Uses: slot positions and the per-period orientation only (no ball). Fails on duplicated or
    missing slots near the halfway line, and on very short post-goal set-ups.
    """
    tracking = bundle.tracking
    T = tracking.n_frames
    dead = np.full(T, np.nan, dtype=np.float32)
    home = np.full(T, np.nan, dtype=np.float32)
    starts = {p.index: p.start for p in tracking.meta.periods}
    after = int(round(after_s * tracking.meta.fps))
    for ko in detect_kickoffs(tracking):
        first = starts[ko.period] if ko.opening else ko.setup_start
        dead[first : ko.kick] = p_dead
        home[first : ko.kick] = 0.5
        dead[ko.kick : ko.kick + after] = 1.0 - p_dead
        home[ko.kick : ko.kick + after] = team_p if ko.team == HOME else 1.0 - team_p
    return Evidence.from_factors("ballstate_kickoff", dead, home)
