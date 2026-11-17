"""Ball context cells, shared by the BAR-ATH calibration and the simulator so both count alike.

A seen ball frame falls in one of four cells: near a line (within LINE_MARGIN_M of a touchline or
goal line, off the pitch included) x crowded (at least CROWD_MIN_PLAYERS tracked players within
CROWD_RADIUS_M of the ball). On the target match's automatic tracking a gap starts near a line and
in a crowd clearly more often than the base rate (measured label-free; the figures stay private).

A frozen ball -- static (< STATIC_SPEED_MPS between consecutive frames) for at least STATIC_MIN_S
-- counts as `open` when `frozen_as_open` (degrade-v2 on). The clean IDSSE ball freezes at the
out-of-play spot through every stoppage, so 91% of its near-line frames are dead and the near-line
hazard would fire through stoppages; in automatic tracking a static ball is rare, so the rule barely
touches the calibration (issue #21).

Where it fails: BAR-ATH counts crowding over fragmented, sometimes duplicated track slots, IDSSE
over true players, so the crowded cell is slightly wider on BAR-ATH. The near-line effect may
partly be the ball leaving the camera's view rather than occlusion; either way it is reproduced.
"""

from __future__ import annotations

import numpy as np

from ..contracts import Tracking

LINE_MARGIN_M = 2.0
CROWD_RADIUS_M = 3.0
CROWD_MIN_PLAYERS = 2
N_CELLS = 4
UNSEEN = -1
CELL_NAMES = ("open", "crowded", "near_line", "near_line_crowded")  # index = 2 * near + crowded
STATIC_SPEED_MPS = 0.5
STATIC_MIN_S = 1.0


def context_cells(tracking: Tracking, *, frozen_as_open: bool = False) -> np.ndarray:
    """(T,) int8: 2 * near_line + crowded where the ball is seen, UNSEEN where it is not."""
    ball = tracking.ball_xy
    line_dist = line_distance(tracking)
    gap = tracking.player_xy - ball[:, None, :]
    with np.errstate(invalid="ignore"):
        near_players = (np.einsum("tsk,tsk->ts", gap, gap) < CROWD_RADIUS_M**2).sum(axis=1)
        cells = 2 * (line_dist < LINE_MARGIN_M) + (near_players >= CROWD_MIN_PLAYERS)
    cells = cells.astype(np.int8)
    if frozen_as_open:
        cells[frozen_ball(tracking)] = 0
    cells[~tracking.ball_seen()] = UNSEEN
    return cells


def line_distance(tracking: Tracking) -> np.ndarray:
    """(T,) metres from the ball to the nearest touchline or goal line; negative off the pitch."""
    meta, ball = tracking.meta, tracking.ball_xy
    return np.minimum(
        meta.pitch_length / 2 - np.abs(ball[:, 0]), meta.pitch_width / 2 - np.abs(ball[:, 1])
    )


def off_pitch(tracking: Tracking) -> np.ndarray:
    """(T,) bool: the ball is seen beyond a touchline or goal line."""
    with np.errstate(invalid="ignore"):
        return line_distance(tracking) < 0


def frozen_ball(tracking: Tracking) -> np.ndarray:
    """(T,) bool: inside a run of >= STATIC_MIN_S frames whose ball moved < STATIC_SPEED_MPS."""
    fps = tracking.meta.fps
    step = np.full(tracking.n_frames, np.inf)
    with np.errstate(invalid="ignore"):
        step[1:] = np.linalg.norm(np.diff(tracking.ball_xy, axis=0), axis=1) * fps
    still = np.nan_to_num(step, nan=np.inf) < STATIC_SPEED_MPS
    edges = np.flatnonzero(np.diff(np.concatenate(([0], still.astype(np.int8), [0]))))
    out = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start >= STATIC_MIN_S * fps:
            out[max(start - 1, 0) : stop] = True  # the run's first frame is its anchor too
    return out
