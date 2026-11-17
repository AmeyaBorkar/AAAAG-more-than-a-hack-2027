"""Tracker errors that training never saw (E66, #86): wrong-team fragments, dropouts with
duplicates, and long ball gaps in open play.

`analysis/scripts/stress_battery.py` applies each of these to cached fold models at S8@8, and each
cost at least 0.005 spellF1 on IDSSE (E66 stage 1). Training uses the same functions with the same
round values as augmentations (`train.frag_swap_p`, `train.drop_dup_p`, `train.long_gap_p`).
Nothing is fitted, and nothing comes from BAR-ATH.

- **Wrong-team fragments.** `fragment_tracks` cuts every player track into anonymous fragments of
  2, 5, 10, 20 or 40 s (a uniform draw). Consecutive fragments of one player are separated by
  Exp(0.2 s) frames, at least 1. Each fragment goes to the first slot of its team that is free over
  its span and the frame before it, so slots are reused as on a video tracker. A fragment goes to
  the other team with probability `swap_p`; the stress and the augmentation use 3%. With
  `ghost_per_min`, extra objects copy a teammate's track shifted by N(0, 1 m) per axis.
- **Dropouts with duplicates.** `drop_and_duplicate` drops 10% of slot-frames in runs of mean 2 s
  (`degrade_ladder`, no noise). Then, in 2 s blocks, a slot reports a random teammate's position
  plus N(0, 0.3 m) with probability 0.05. These are `stress.py`'s s4 and s5, applied together.
- **Long ball gaps.** `long_ball_gaps` hides the ball for 6 s from starts at 1 per minute of alive
  play in which the ball is seen. A gap never runs past its period, and it ends without a restart.
  degrade-v5's open-play gaps follow Metrica's, which are shorter.

Where they fail:
- real swaps cluster in crowds and at the frame edge, while these are uniform;
- a duplicate's position is a teammate's, never a referee's;
- the long gaps ignore context (BAR-ATH probably loses the ball in the air and in crowds).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..contracts import AWAY, HOME, A, D, H, Period, Tracking
from ..degrade.ladder import LadderModel, degrade_ladder

FRAGMENT_S = (2.0, 5.0, 10.0, 20.0, 40.0)
FRAGMENT_GAP_MEAN_S = 0.2
SWAP_P = 0.03
GHOST_OFFSET_M = 1.0
DROP_SHARE = 0.10
DROP_MEAN_S = 2.0
DUP_P = 0.05
DUP_BLOCK_S = 2.0
DUP_JITTER_M = 0.3
BLOCK_SWAP_S = 10.0
BLOCK_SWAP_P = 0.5
LONG_GAP_S = 6.0
LONG_GAP_PER_MIN = 1.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def fragment_tracks(
    tracking: Tracking,
    rng: np.random.Generator,
    *,
    swap_p: float = 0.0,
    ghost_per_min: float = 0.0,
) -> Tracking:
    """Anonymous, reused slots with optional wrong-team fragments and ghosts; the ball is kept."""
    fps = tracking.meta.fps
    xy = tracking.player_xy.astype(np.float64)
    lengths = np.maximum(1, np.rint(np.asarray(FRAGMENT_S) * fps)).astype(np.int64)
    pieces: list[tuple[int, int, int, int, np.ndarray]] = []  # start, stop, slot, team, offset
    for slot, team in enumerate(tracking.slot_team):
        for start, stop in _runs(~np.isnan(xy[:, slot, 0])):
            t = start
            while t < stop:
                end = min(t + int(lengths[rng.integers(len(lengths))]), stop)
                out_team = int(team)
                if swap_p > 0 and rng.random() < swap_p:
                    out_team = AWAY if out_team == HOME else HOME
                pieces.append((t, end, slot, out_team, np.zeros(2)))
                t = end + max(1, round(rng.exponential(FRAGMENT_GAP_MEAN_S * fps)))
    if ghost_per_min > 0:
        played = np.flatnonzero(tracking.meta.played_mask())
        minutes = len(played) / fps / 60
        for team in (HOME, AWAY):
            slots = np.flatnonzero(tracking.slot_team == team)
            for _ in range(rng.poisson(ghost_per_min * minutes)):
                frame = int(rng.choice(played))
                mates = slots[~np.isnan(xy[frame, slots, 0])]
                offset = rng.normal(0.0, GHOST_OFFSET_M, size=2)
                length = int(lengths[rng.integers(len(lengths))])
                if len(mates) == 0:
                    continue
                mate = int(rng.choice(mates))
                seen = ~np.isnan(xy[frame:, mate, 0])
                stop = frame + (int(np.argmin(seen)) if not seen.all() else len(seen))
                pieces.append((frame, min(frame + length, stop), mate, team, offset))

    columns, teams = [], []
    for team in (HOME, AWAY):
        busy: list[int] = []  # per output slot: the end of its last piece
        first = len(columns)
        mine = sorted((p for p in pieces if p[3] == team), key=lambda p: (p[0], p[2]))
        for start, stop, slot, _, offset in mine:
            free = next((i for i, until in enumerate(busy) if until < start), None)
            if free is None:
                columns.append(np.full((tracking.n_frames, 2), np.nan, dtype=np.float64))
                teams.append(team)
                busy.append(0)
                free = len(busy) - 1
            columns[first + free][start:stop] = xy[start:stop, slot] + offset
            busy[free] = stop
    return Tracking(
        meta=tracking.meta,
        player_xy=np.stack(columns, axis=1).astype(np.float32),
        slot_team=np.array(teams, dtype=np.int8),
        ball_xy=tracking.ball_xy,
        player_speed=None,
        ball_extra=dict(tracking.ball_extra),
    ).validate()


def block_swap(
    tracking: Tracking,
    rng: np.random.Generator,
    *,
    block_s: float = BLOCK_SWAP_S,
    p: float = BLOCK_SWAP_P,
) -> tuple[Tracking, int]:
    """(tracking, blocks swapped): cross-team identity swaps that last a whole block (E101, #86).

    The frames are cut into blocks of `block_s` seconds. In each block, with probability `p`, a
    random home slot seen at the block's first frame and the away slot nearest to it there exchange
    their positions for the whole block, so each carries the other team's player. A block with no
    home or away slot seen at its first frame is left as it is. Slot teams and the ball are kept.
    Round values, nothing fitted: a heavier version of Anfield's 10 s guard swaps, far heavier than
    `fragment_tracks(swap_p=0.03)`.

    Where it fails: real swaps happen when two players cross, so the positions jump at the block
    edges here; and only one pair is swapped at a time.
    """
    block = max(1, int(round(block_s * tracking.meta.fps)))
    xy = tracking.player_xy.copy()
    home, away = tracking.team_slots(HOME), tracking.team_slots(AWAY)
    swapped = 0
    for start in range(0, tracking.n_frames, block):
        if rng.random() >= p:
            continue
        seen_h = home[np.isfinite(xy[start, home, 0])]
        seen_a = away[np.isfinite(xy[start, away, 0])]
        if not len(seen_h) or not len(seen_a):
            continue
        h = int(seen_h[rng.integers(len(seen_h))])
        a = int(seen_a[np.argmin(np.hypot(*(xy[start, seen_a] - xy[start, h]).T))])
        stop = min(start + block, tracking.n_frames)
        xy[start:stop, [h, a]] = xy[start:stop, [a, h]]
        swapped += 1
    out = dataclasses.replace(tracking, player_xy=xy, player_speed=None, _fingerprint=None)
    return out.validate(), swapped


def drop_and_duplicate(tracking: Tracking, rng: np.random.Generator) -> Tracking:
    """10% of slot-frames dropped (runs of mean 2 s), then 5% duplicates in 2 s blocks."""
    model = LadderModel("s4", DROP_SHARE, DROP_MEAN_S, noise_sigma_m=0.0, noise_smooth_s=0.5)
    tracking = degrade_ladder(tracking, model, rng)
    block = int(round(DUP_BLOCK_S * tracking.meta.fps))
    n_frames = tracking.n_frames
    n_blocks = -(-n_frames // block)
    source = tracking.player_xy
    out = source.copy()
    rows = np.arange(n_frames)[:, None]
    for team in (HOME, AWAY):
        columns = tracking.team_slots(team)
        donor = columns[rng.integers(0, len(columns), size=(n_blocks, len(columns)))]
        donor = np.repeat(donor, block, axis=0)[:n_frames]
        replaced = np.repeat(rng.random((n_blocks, len(columns))) < DUP_P, block, axis=0)[:n_frames]
        jitter = rng.normal(0.0, DUP_JITTER_M, size=(n_frames, len(columns), 2))
        copied = source[rows, donor, :] + jitter
        out[:, columns, :] = np.where(replaced[..., None], copied, source[:, columns, :])
    return dataclasses.replace(
        tracking, player_xy=out.astype(np.float32), player_speed=None, _fingerprint=None
    ).validate()


def long_ball_gaps(
    ball_xy: np.ndarray,
    labels: np.ndarray,
    periods: tuple[Period, ...],
    fps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(T,) bool: frames to hide, 6 s from starts at 1 per minute of alive play with the ball seen."""
    alive = np.isin(np.asarray(labels), (H, A)) & np.isfinite(ball_xy).all(axis=1)
    starts = np.flatnonzero(alive & (rng.random(len(ball_xy)) < LONG_GAP_PER_MIN / (60 * fps)))
    hidden = np.zeros(len(ball_xy), dtype=bool)
    span = int(round(LONG_GAP_S * fps))
    for period in periods:
        for s in starts[(starts >= period.start) & (starts < period.stop)]:
            hidden[s : min(s + span, period.stop)] = True
    return hidden


DEAD_MIN_S = 4.0
DEAD_VISIBLE_S = 3.0
DEAD_SPEED_MPS = 4.0
DEAD_REST_S = 1.0
_LOOK_S = 2.0  # how far to look for the clean ball's last alive and restart positions


def _stoppages(labels: np.ndarray, periods: tuple[Period, ...], min_dead: int):
    """(start, restart) of D runs of at least `min_dead` frames followed by H/A in one period (as
    `data.true_stoppages`, which this module cannot import without a cycle)."""
    out = []
    for period in periods:
        x = np.asarray(labels[period.start : period.stop])
        breaks = np.flatnonzero(x[1:] != x[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(x)]))
        for i in range(len(bounds) - 2):
            a, z = int(bounds[i]), int(bounds[i + 1])
            if x[a] == D and z - a >= min_dead and x[z] in (H, A):
                out.append((period.start + a, period.start + z))
    return out


def dead_ball_visible(
    ball_xy: np.ndarray,
    clean_ball: np.ndarray,
    labels: np.ndarray,
    periods: tuple[Period, ...],
    fps: int,
    rng: np.random.Generator | None = None,
    p: float = 1.0,
) -> tuple[np.ndarray, int]:
    """(ball, stoppages changed): the visible, uncarried dead ball (E71).

    In each true stoppage of at least 4 s (each with probability `p` when `rng` is given; every one
    when it is None) the ball is visible for the stoppage's last 3 s. It moves in a straight line
    from its last alive position (the clean ball just before the stoppage) to the restart spot (the
    clean ball at the restart) at 4 m/s, arriving 1 s before the restart and resting there. A
    farther start is cut to the last 4 m/s x 2 s of the path. Frames from the restart on are left
    as they are. A stoppage without both clean positions within 2 s is skipped.
    """
    out = np.array(ball_xy, dtype=np.float32, copy=True)
    seen = np.isfinite(clean_ball).all(axis=1)
    look = int(round(_LOOK_S * fps))
    span = int(round(DEAD_VISIBLE_S * fps))
    rest = int(round(DEAD_REST_S * fps))
    changed = 0
    for start, restart in _stoppages(labels, periods, int(round(DEAD_MIN_S * fps))):
        if rng is not None and rng.random() >= p:
            continue
        before = np.flatnonzero(seen[max(0, start - look) : start])
        after = np.flatnonzero(seen[restart : restart + look])
        if not len(before) or not len(after):
            continue
        p0 = clean_ball[max(0, start - look) + before[-1]].astype(np.float64)
        p1 = clean_ball[restart + after[0]].astype(np.float64)
        d = float(np.hypot(*(p0 - p1)))
        lo = restart - span
        t = np.arange(lo, restart)
        dist = np.clip(DEAD_SPEED_MPS * (restart - rest - t) / fps, 0.0, d)
        unit = (p0 - p1) / d if d > 0 else np.zeros(2)
        out[lo:restart] = (p1[None, :] + unit[None, :] * dist[:, None]).astype(np.float32)
        changed += 1
    return out, changed
