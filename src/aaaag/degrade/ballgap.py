"""Ball dropout in BAR-ATH-like blocks: a context-dependent start hazard and measured gap lengths.

Model (degrade-v1; v2 adds context.py's frozen-ball rule; v3 instead restages the dead ball first,
restage.py, and draws the top tenth of gap lengths from a fitted lognormal tail; v4 adds
outofplay.py's out-of-play mask and calibrates hazards and lengths on on-pitch gap starts only;
v5 fits on Metrica only, hides the ball through every stoppage with deadball.py, and applies the
hazard on alive frames only):
- each seen frame t is the last frame before a new gap with probability hazard[cell(t)]
  (context.py); the four hazards are the calibration match's gap starts per seen frame in each cell;
- a gap's length is drawn uniformly from the calibration's gap-length quantiles at the midpoints of
  40 equal-mass bins, each an actual observed length, so the shape survives (a hole in the observed
  lengths stays a hole, which interpolating between quantiles would fill);
- the ball comes back after the gap and the hazard applies again; gaps are cut at period ends.

Signals: ball position and player positions of the clean match (for the context only). Parameters:
configs/degrade/degrade-v<version>.yaml, written by analysis/scripts/data_degrade_calibration.py.

Where it fails: gap length does not depend on context; the clean IDSSE ball freezes during
stoppages while the ATD ball moves; track slots stay clean (true identities, no fragments or
duplicates), so identity-based features look better here than on BAR-ATH.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import lognorm

from ..contracts import A, H, MatchBundle, Period, Tracking
from ..paths import CONFIGS_DIR
from .context import N_CELLS, UNSEEN, context_cells, off_pitch
from .deadball import DeadBall, dead_ball_mask
from .ladder import LadderModel, degrade_ladder
from .outofplay import out_of_play_mask
from .players import PlayerModel, degrade_players
from .restage import restage_dead_ball

DEGRADE_DIR = CONFIGS_DIR / "degrade"


@dataclass(frozen=True)
class GapModel:
    version: str
    hazard: tuple[float, ...]  # per seen frame, by context cell
    length_s: tuple[float, ...]  # gap lengths, sampled uniformly
    frozen_as_open: bool = False  # v2: a frozen ball counts as open play (context.py)
    restage_dead_ball: bool = False  # v3: move the frozen dead ball to the restart (restage.py)
    tail: tuple[float, float, float, float] | None = None  # v3: (share, threshold_s, shape, scale)
    exit_flash_s: tuple[float, ...] | None = None  # v4: out-of-play mask (outofplay.py)
    dead_ball: DeadBall | None = None  # v5: hide the ball through stoppages (deadball.py)
    hazard_on_alive_only: bool = False  # v5: hazards are per seen alive frame

    @classmethod
    def load(cls, version: str = "1", path: str | Path | None = None) -> GapModel:
        path = Path(path) if path else DEGRADE_DIR / f"degrade-v{version}.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        model = cls(
            version=str(raw["version"]),
            hazard=tuple(float(h) for h in raw["hazard_per_seen_frame"]),
            length_s=tuple(float(s) for s in raw["gap_length_s"]),
            frozen_as_open=bool(raw.get("context", {}).get("frozen_as_open", False)),
            restage_dead_ball=bool(raw.get("restage_dead_ball", False)),
            tail=_tail(raw.get("gap_length_tail")),
            exit_flash_s=tuple(float(s) for s in raw["exit_flash_s"])
            if raw.get("exit_flash_s")
            else None,
            dead_ball=DeadBall.from_dict(raw["dead_ball"]) if raw.get("dead_ball") else None,
            hazard_on_alive_only=raw.get("hazard_on") == "alive",
        )
        if model.version != str(version) or len(model.hazard) != N_CELLS or not model.length_s:
            raise ValueError(f"{path}: not a degrade-v{version} model")
        if not all(0.0 <= h < 1.0 for h in model.hazard) or min(model.length_s) <= 0:
            raise ValueError(f"{path}: hazards must be in [0, 1), lengths positive")
        return model


def gap_mask(
    cells: np.ndarray, periods: tuple[Period, ...], model: GapModel, fps: int, rng
) -> np.ndarray:
    """(T,) bool, True on frames where the ball is dropped. O(T) draws plus one step per gap."""
    mask = np.zeros(len(cells), dtype=bool)
    hazard = np.append(np.asarray(model.hazard), 0.0)  # cell -1 (ball already unseen): no start
    lengths = np.maximum(1, np.rint(np.asarray(model.length_s) * fps)).astype(np.int64)
    if model.tail is not None:
        share, threshold, shape, scale = model.tail
        floor = lognorm.cdf(threshold, shape, scale=scale)

    def draw() -> int:
        if model.tail is not None and rng.random() < share:
            return max(1, round(lognorm.ppf(rng.uniform(floor, 1.0), shape, scale=scale) * fps))
        return int(lengths[rng.integers(len(lengths))])

    for period in periods:
        local = cells[period.start : period.stop]
        starts = np.flatnonzero(rng.random(period.length) < hazard[local])
        t = 0
        while (k := int(np.searchsorted(starts, t))) < len(starts):
            first = int(starts[k]) + 1  # the gap begins on the frame after the last seen one
            stop = min(first + draw(), period.length)
            mask[period.start + first : period.start + stop] = True
            t = stop
    return mask


def degrade(
    bundle: MatchBundle,
    *,
    version: str = "2",
    seed: int = 0,
    players: str | None = None,
    edge_jitter_s: float = 0.0,
    ladder: str | None = None,
) -> MatchBundle:
    """A copy of `bundle` with the ball dropped by degrade-v<version>; labels unchanged.

    `players` (e.g. "1") also degrades the player tracks with degrade-p<players> (players.py);
    the match id then becomes `<id>~d<version>p<players>s<seed>`. `edge_jitter_s` (v5 only, the
    robustness rungs of #50) shifts both ends of every hidden stoppage by U(-j, +j) seconds; the
    id then carries `j<j>`. `ladder` (e.g. "v51-medium", #73) also applies the v5.1 player ladder
    (ladder.py, physical priors); the id then carries the ladder name."""
    model = GapModel.load(str(version))
    if edge_jitter_s and model.dead_ball is None:
        raise ValueError(f"edge_jitter_s needs a dead-ball model; degrade-v{version} has none")
    tracking, meta = bundle.tracking, bundle.meta
    key = [int(seed), zlib.crc32(meta.match_id.encode())]
    if model.restage_dead_ball:
        restaged = restage_dead_ball(bundle, np.random.default_rng([*key, 1]))
        tracking = replace(tracking, ball_xy=restaged, _fingerprint=None)
    cells = context_cells(tracking, frozen_as_open=model.frozen_as_open)
    hidden = np.zeros(meta.n_frames, dtype=bool)
    if model.exit_flash_s is not None:
        hidden = out_of_play_mask(
            bundle, tracking.ball_xy, model.exit_flash_s, np.random.default_rng([*key, 2])
        )
        cells[hidden | off_pitch(tracking)] = UNSEEN  # v4 hazards are on-pitch only
    if model.dead_ball is not None:
        if bundle.labels is None:
            raise ValueError("degrade-v5 needs labels: the stoppages come from them")
        hidden |= dead_ball_mask(
            bundle.labels,
            meta.periods,
            model.dead_ball,
            meta.fps,
            np.random.default_rng([*key, 4]),
            edge_jitter_s=edge_jitter_s,
        )
        cells[hidden] = UNSEEN
    if model.hazard_on_alive_only:
        if bundle.labels is None:
            raise ValueError("alive-only hazards need labels")
        cells[~np.isin(bundle.labels, (H, A))] = UNSEEN
    mask = gap_mask(cells, meta.periods, model, meta.fps, np.random.default_rng(key)) | hidden
    ball_xy = tracking.ball_xy.copy()
    ball_xy[mask] = np.nan

    tag = f"d{model.version}" + (f"j{edge_jitter_s:g}" if edge_jitter_s else "")
    tag += f"p{players}" if players is not None else ""
    tag += ladder if ladder is not None else ""
    new_meta = replace(meta, match_id=f"{meta.match_id}~{tag}s{seed}")
    degraded = Tracking(
        meta=new_meta,
        player_xy=tracking.player_xy,
        slot_team=tracking.slot_team,
        ball_xy=ball_xy,
        player_speed=tracking.player_speed,
        ball_extra={},  # provider extras would leak clean-ball information
    )
    if players is not None:
        degraded = degrade_players(
            degraded, PlayerModel.load(str(players)), np.random.default_rng([*key, 3])
        )
    if ladder is not None:
        degraded = degrade_ladder(
            degraded, LadderModel.load(str(ladder)), np.random.default_rng([*key, 5])
        )
    played = meta.played_mask()
    info = {"version": model.version, "seed": int(seed), "ball_missing": float(mask[played].mean())}
    if edge_jitter_s:
        info["edge_jitter_s"] = float(edge_jitter_s)
    if players is not None:
        info["players"] = str(players)
    if ladder is not None:
        info["ladder"] = str(ladder)
    return MatchBundle(
        tracking=degraded.validate(),
        tags=bundle.tags,
        labels=bundle.labels,
        extras={**bundle.extras, "degrade": info},
    ).validate()


def _tail(raw: dict | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    tail = (
        float(raw["share"]),
        float(raw["threshold_s"]),
        float(raw["shape"]),
        float(raw["scale"]),
    )
    if not (0.0 < tail[0] < 1.0 and tail[1] > 0 and tail[2] > 0 and tail[3] > 0):
        raise ValueError(f"bad gap_length_tail {raw}")
    return tail
