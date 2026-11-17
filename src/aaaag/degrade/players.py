"""degrade-p1: make clean IDSSE player tracks look like BAR-ATH's automatic tracking (issue #38).

The target match's automatic tracking against clean IDSSE (playerstats.py; IDSSE = design matches
J03WMX + J03WPY; measured label-free, the target-match figures stay private):
- identity: automatic tracks break into anonymous fragments of tens of seconds, with a varying
  number of objects per team, sometimes more than 11; IDSSE keeps true identities, always 11;
- smoothness: automatic positions are smooth; IDSSE positions are rounded to the centimetre, which
  reads as several m/s^2 of frame-to-frame noise.

Steps, in order, all seeded:
1. smooth: a centred Gaussian of `smooth_s` seconds along each seen run of a true track;
2. fragment: each run alternates a fragment (length drawn from the calibration's fragment-length
   quantiles) and an unseen dropout (exponential, mean `dropout_mean_s`);
3. ghosts: extra objects, `ghost_per_min` per team, each a teammate's trajectory shifted by a
   constant Gaussian offset (`ghost_offset_m` per axis) for a fragment-length draw;
4. re-slot: every fragment and ghost goes to the first output slot of its team that is free over
   its whole span and the frame before it, so slots are anonymous and reused as in automatic
   tracking, and
   two fragments never merge into one run with a jump.
`player_speed` is dropped: the delivered IDSSE channel would carry clean speeds.

Signals: player positions and teams of the clean match. Parameters:
configs/degrade/players-p<version>.yaml, written by analysis/scripts/data_degrade_players.py.

Where it fails: dropouts and ghosts ignore context (automatic tracking probably loses players in
crowds and at the frame edge, and its extra objects may be referees or substitutes rather than duplicates);
fragments of one player never overlap, so a duplicate is always a ghost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from ..contracts import AWAY, HOME, Tracking
from ..paths import CONFIGS_DIR


@dataclass(frozen=True)
class PlayerModel:
    version: str
    smooth_s: float
    fragment_s: tuple[float, ...]
    dropout_mean_s: float
    ghost_per_min: float
    ghost_offset_m: float

    @classmethod
    def load(cls, version: str = "1", path: str | Path | None = None) -> PlayerModel:
        path = Path(path) if path else CONFIGS_DIR / "degrade" / f"players-p{version}.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        model = cls(
            version=str(raw["version"]),
            smooth_s=float(raw["smooth_s"]),
            fragment_s=tuple(float(s) for s in raw["fragment_length_s"]),
            dropout_mean_s=float(raw["dropout_mean_s"]),
            ghost_per_min=float(raw["ghost_per_min"]),
            ghost_offset_m=float(raw["ghost_offset_m"]),
        )
        if model.version != str(version) or not model.fragment_s or min(model.fragment_s) <= 0:
            raise ValueError(f"{path}: not a degrade-p{version} model")
        if min(model.smooth_s, model.dropout_mean_s, model.ghost_per_min, model.ghost_offset_m) < 0:
            raise ValueError(f"{path}: parameters must be non-negative")
        return model


def degrade_players(tracking: Tracking, model: PlayerModel, rng) -> Tracking:
    """A copy of `tracking` with the player tracks degraded; the ball and meta are untouched."""
    fps = tracking.meta.fps
    xy = smooth(tracking.player_xy, model.smooth_s * fps)
    lengths = np.maximum(1, np.rint(np.asarray(model.fragment_s) * fps)).astype(np.int64)
    pieces: dict[int, list[tuple[int, int, int, np.ndarray]]] = {HOME: [], AWAY: []}
    for slot, team in enumerate(tracking.slot_team):
        for start, stop in _runs(~np.isnan(xy[:, slot, 0])):
            t = start
            while t < stop:
                end = min(t + int(lengths[rng.integers(len(lengths))]), stop)
                pieces[int(team)].append((t, end, slot, np.zeros(2)))
                t = end + max(1, round(rng.exponential(model.dropout_mean_s * fps)))
    played = tracking.meta.played_mask()
    minutes = played.sum() / fps / 60
    for team in (HOME, AWAY):
        slots = np.flatnonzero(tracking.slot_team == team)
        for _ in range(rng.poisson(model.ghost_per_min * minutes)):
            frame = int(rng.choice(np.flatnonzero(played)))
            mates = slots[~np.isnan(xy[frame, slots, 0])]
            offset = rng.normal(0.0, model.ghost_offset_m, size=2)
            length = int(lengths[rng.integers(len(lengths))])
            if len(mates) == 0:
                continue
            mate = int(rng.choice(mates))
            seen = ~np.isnan(xy[frame:, mate, 0])
            stop = frame + (int(np.argmin(seen)) if not seen.all() else len(seen))
            pieces[team].append((frame, min(frame + length, stop), mate, offset))

    columns, teams = [], []
    for team in (HOME, AWAY):
        busy: list[int] = []  # per output slot: end of its last piece; a gap of >= 1 frame follows
        first = len(columns)
        for start, stop, slot, offset in sorted(pieces[team], key=lambda p: (p[0], p[2])):
            free = next((i for i, until in enumerate(busy) if until < start), None)
            if free is None:
                columns.append(np.full((tracking.n_frames, 2), np.nan, dtype=np.float32))
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


def smooth(player_xy: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Centred Gaussian smoothing along each seen run of each slot; NaN stays NaN."""
    out = player_xy.astype(np.float64)
    if sigma_frames <= 0:
        return out
    for slot in range(out.shape[1]):
        for start, stop in _runs(~np.isnan(out[:, slot, 0])):
            out[start:stop, slot] = gaussian_filter1d(
                out[start:stop, slot], sigma_frames, axis=0, mode="nearest"
            )
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
