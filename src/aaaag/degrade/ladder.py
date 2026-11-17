"""degrade-v5.1 player ladder (#73): physics-prior player degradation, not calibrated on BAR-ATH.

v5 (#50) degrades the ball only. The target match's player tracks come from video and fragment
(far more track slots than players), so pipelines that read player positions near the ball (`control_possession`)
must also be checked under degraded players. The degrade-p1 model (players.py) was calibrated on
BAR-ATH aggregates and is retired from selection (ADR 2026-11-05). This ladder is a set of
physical priors fixed before any run, at three severities (configs/degrade/degrade-v51-*.yaml):

| rung | slot-frames dropped | dropout runs | position noise sigma |
|---|---|---|---|
| mild | 5% | exponential, mean 2 s | 0.10 m |
| medium | 10% | exponential, mean 2 s | 0.25 m |
| severe | 20% | exponential, mean 2 s | 0.50 m |

Steps, in order, all seeded:
1. **noise:** per observed slot and axis, white Gaussian noise smoothed by a Gaussian kernel of
   `noise_smooth_s` (sigma, seconds) and rescaled to standard deviation `noise_sigma_m`, so the
   error drifts over about half a second instead of jittering frame to frame;
2. **dropout:** a dropout starts on an observed, not-yet-dropped frame with a per-frame hazard set
   so that `dropout_share` of slot-frames are dropped in expectation, with exponential lengths of
   mean `dropout_mean_s`. With `near_ball_factor` > 1 the hazard is that many times higher while
   the slot is within `near_ball_m` of the seen ball (the harsher medium variant), so the dropped
   share there comes out above `dropout_share`.

Slot identities, teams and the ball are untouched: no fragmentation or re-slotting, since no
pipeline reads slot identities. Where it fails: real tracker errors are correlated with crowds,
occlusion and the frame edge, and include identity swaps between teams, which this ladder does
not model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from ..contracts import Tracking
from ..paths import CONFIGS_DIR


@dataclass(frozen=True)
class LadderModel:
    name: str
    dropout_share: float
    dropout_mean_s: float
    noise_sigma_m: float
    noise_smooth_s: float
    near_ball_factor: float = 1.0
    near_ball_m: float = 5.0

    @classmethod
    def load(cls, name: str, path: str | Path | None = None) -> LadderModel:
        path = Path(path) if path else CONFIGS_DIR / "degrade" / f"degrade-{name}.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        model = cls(
            name=str(raw["name"]),
            dropout_share=float(raw["dropout_share"]),
            dropout_mean_s=float(raw["dropout_mean_s"]),
            noise_sigma_m=float(raw["noise_sigma_m"]),
            noise_smooth_s=float(raw["noise_smooth_s"]),
            near_ball_factor=float(raw.get("near_ball_factor", 1.0)),
            near_ball_m=float(raw.get("near_ball_m", 5.0)),
        )
        if model.name != name:
            raise ValueError(f"{path}: names {model.name!r}, expected {name!r}")
        if not 0.0 <= model.dropout_share < 1.0 or model.dropout_mean_s <= 0:
            raise ValueError(f"{path}: dropout_share must be in [0, 1), dropout_mean_s > 0")
        if min(model.noise_sigma_m, model.noise_smooth_s) < 0 or model.near_ball_factor < 1:
            raise ValueError(f"{path}: noise must be non-negative and near_ball_factor >= 1")
        return model


def degrade_ladder(tracking: Tracking, model: LadderModel, rng: np.random.Generator) -> Tracking:
    """A copy of `tracking` with player positions noised and dropped; ball, teams, meta untouched."""
    fps = tracking.meta.fps
    xy = tracking.player_xy.astype(np.float64)
    observed = ~np.isnan(xy[..., 0])
    xy = add_noise(xy, observed, model.noise_sigma_m, model.noise_smooth_s * fps, rng)
    dropped = dropout_mask(tracking, observed, model, rng)
    xy[dropped] = np.nan
    return replace(tracking, player_xy=xy.astype(np.float32), _fingerprint=None).validate()


def add_noise(
    xy: np.ndarray, observed: np.ndarray, sigma_m: float, smooth_frames: float, rng
) -> np.ndarray:
    """xy plus smoothed Gaussian noise of standard deviation `sigma_m` on observed entries."""
    if sigma_m <= 0:
        return xy
    white = rng.standard_normal(xy.shape)
    if smooth_frames > 0:
        white = gaussian_filter1d(white, smooth_frames, axis=0, mode="wrap")
        # the std of white noise after a unit-sum Gaussian kernel of sigma s is
        # 1 / sqrt(2 sqrt(pi) s) for s >> 1; use the exact kernel norm instead
        radius = int(4.0 * smooth_frames + 0.5)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / smooth_frames) ** 2)
        kernel /= kernel.sum()
        white /= np.sqrt((kernel**2).sum())
    out = xy.copy()
    out[observed] += sigma_m * white[observed]
    return out


def dropout_mask(tracking: Tracking, observed: np.ndarray, model: LadderModel, rng) -> np.ndarray:
    """(T, S) bool: dropped slot-frames. Starts are a per-frame Bernoulli on observed frames with
    hazard h = share / ((1 - share) * mean_len), so the expected dropped share is `share` when
    near_ball_factor is 1; lengths are exponential with mean `dropout_mean_s`."""
    T, S = observed.shape
    dropped = np.zeros((T, S), dtype=bool)
    if model.dropout_share <= 0:
        return dropped
    mean_len = model.dropout_mean_s * tracking.meta.fps
    hazard = np.full((T, S), model.dropout_share / ((1.0 - model.dropout_share) * mean_len))
    if model.near_ball_factor > 1.0:
        ball = tracking.ball_xy.astype(np.float64)
        d2 = ((tracking.player_xy.astype(np.float64) - ball[:, None, :]) ** 2).sum(axis=2)
        with np.errstate(invalid="ignore"):
            near = d2 <= model.near_ball_m**2  # NaN (ball or slot unseen) compares False
        hazard[near] *= model.near_ball_factor
    starts = observed & (rng.random((T, S)) < hazard)
    lengths = rng.exponential(mean_len, size=(T, S))
    for slot in range(S):
        until = -1
        for t in np.flatnonzero(starts[:, slot]):
            if t < until:
                continue  # inside the previous dropout: renewal, not overlap
            until = min(T, t + max(1, int(round(lengths[t, slot]))))
            dropped[t:until, slot] = True
    return dropped & observed
