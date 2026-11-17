"""Restart events: where each stoppage ends, what kind of restart it is, and who takes it.

A restart is the first alive frame after a dead run of at least `min_dead_s` in P(dead). Its
kind comes from the kick-off detector, else from where the ball is seen around the restart:

| kind | where the ball is | who restarts |
|---|---|---|
| kickoff | (kick-off formation) | the side with the player nearest the centre spot |
| corner | within `corner_m` of a corner flag | the side attacking that goal |
| goal_kick | in a goal area | the side defending that goal |
| throw_in | within `line_m` of a touchline | the side of the nearest player (the thrower) |
| free_kick | anywhere else | the side of the nearest player (the taker) |
| unknown | not seen near the restart | nobody |

Uses: P(dead), ball position where seen, slot positions, per-period orientation. Fails when the
ball is unseen at the restart (then only kick-offs are typed), and when a defender stands nearer
the ball than the taker at a free kick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...contracts import AWAY, HOME, Tracking
from .cues import (
    GOAL_AREA_DEPTH,
    GOAL_AREA_HALF_WIDTH,
    ball_speed,
    home_defends_sign,
    period_index,
    runs,
)
from .kickoff import Kickoff

KINDS = ("kickoff", "corner", "goal_kick", "throw_in", "free_kick", "unknown")


@dataclass(frozen=True)
class Restart:
    frame: int  # first alive frame
    stoppage_start: int  # first dead frame of the stoppage it ends
    kind: str  # one of KINDS
    team: int | None  # HOME, AWAY, or None when unknown


def detect_restarts(
    tracking: Tracking,
    p_dead: np.ndarray,
    kickoffs: list[Kickoff],
    *,
    min_dead_s: float = 2.0,
    look_s: float = 1.0,
    kickoff_tol_s: float = 2.0,
    corner_m: float = 3.0,
    line_m: float = 1.5,
    margin_m: float = 0.5,
    placed_speed: float = 3.0,
) -> list[Restart]:
    """Every stoppage end in `p_dead` (>= 0.5 for at least `min_dead_s`), typed and attributed.

    The ball is read at its last sighting slower than `placed_speed` in the `2 * look_s` before the
    restart (where it was set down), else at the sighting nearest the restart within `look_s`.
    """
    meta = tracking.meta
    fps = meta.fps
    half_l, half_w = meta.pitch_length / 2, meta.pitch_width / 2
    orientation = home_defends_sign(tracking)
    period_of = period_index(meta)
    seen = tracking.ball_seen()
    speed = ball_speed(tracking, window_s=1.0)
    look = int(round(look_s * fps))
    with np.errstate(invalid="ignore"):
        dead = np.nan_to_num(p_dead, nan=0.0) >= 0.5

    out: list[Restart] = []
    for start, stop in zip(*runs(dead), strict=True):
        if stop - start < min_dead_s * fps or stop >= meta.n_frames or period_of[stop] == 0:
            continue
        near = [ko for ko in kickoffs if abs(ko.kick - stop) <= kickoff_tol_s * fps]
        if near:
            out.append(Restart(int(stop), int(start), "kickoff", near[0].team))
            continue
        first = max(start, stop - 2 * look)
        candidates = first + np.flatnonzero(seen[first : stop + look // 2 + 1])
        if not len(candidates):
            out.append(Restart(int(stop), int(start), "unknown", None))
            continue
        with np.errstate(invalid="ignore"):
            placed = candidates[speed[candidates] < placed_speed]
        # the ball where it was set for the restart: the last slow sighting, else the nearest one
        frame = (
            int(placed[-1]) if len(placed) else int(candidates[np.abs(candidates - stop).argmin()])
        )
        bx, by = (float(v) for v in tracking.ball_xy[frame])
        home_goal_sign = orientation.get(int(period_of[stop]), 0.0)  # home defends +x if +1
        end_sign = float(np.sign(bx)) or 1.0  # which goal the ball is nearest
        defending = HOME if end_sign == home_goal_sign else AWAY
        attacking = AWAY if defending == HOME else HOME
        if np.hypot(half_l - abs(bx), half_w - abs(by)) < corner_m:
            kind, team = "corner", attacking if home_goal_sign else None
        elif abs(bx) > half_l - GOAL_AREA_DEPTH - margin_m and abs(by) < (
            GOAL_AREA_HALF_WIDTH + margin_m
        ):
            kind, team = "goal_kick", defending if home_goal_sign else None
        else:
            kind = "throw_in" if abs(by) > half_w - line_m else "free_kick"
            d2 = ((tracking.player_xy[frame] - tracking.ball_xy[frame]) ** 2).sum(axis=1)
            team = None if np.isnan(d2).all() else int(tracking.slot_team[int(np.nanargmin(d2))])
        out.append(Restart(int(stop), int(start), kind, team))
    return out
