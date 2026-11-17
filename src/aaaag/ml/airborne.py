"""E119 (#86): the airborne-ball projection as a training augmentation.

A tactical camera sees the ball in the image, and the homography maps that point to the pitch
plane, so a ball at height z is placed where the camera ray through it meets the ground:
G = C + (P - C) * cz / (cz - z). E118 showed that this reproduces the target's fast-ball profile
(measured label-free; the figures stay private) and costs v14 0.083 on IDSSE fold models.
TRACAB's ball, our training data, is the true ground position, so the network never saw it.

Each draw places one camera for the whole match: x uniform in [-20, 20] m along the halfway line,
20 m behind a touchline chosen at random, height uniform in [cz_min, cz_max] m. The height used is
capped at cz - 2 m. Only the coupled variant's ball is changed (as the other ball augmentations of
data.py do); missing ball frames stay missing. Signals: the ball's xy and its height
(`ball_height`): the IDSSE ball Z (`aaaag.sources.idsse_ballz`), and, with `train.airborne_extra`,
the PFF raw ball's z on extra matches (`aaaag.sources.pff_ballz`, train split only). Where it
fails: one fixed camera per match, no lens model, PFF's z is itself a broadcast estimate, and an
extra match from a source with no height (Metrica, SkillCorner, synthetic) is left as it is.
"""

from __future__ import annotations

import numpy as np

BEHIND_M = 20.0
CX_M = 20.0
Z_MARGIN = 2.0
HEIGHT_SOURCES = ("pff",)  # extra-match sources with a ball height


def ball_height(match_id: str, bundle, *, extra: bool) -> np.ndarray | None:
    """(T,) ball height in metres for the projection, or None when there is none.

    A non-extra match is an IDSSE dev match and reads `idsse_ballz.ball_z`, as before. An extra
    match reads its source's height: PFF's raw-ball z (`pff_ballz.ball_z`, which refuses the test
    split). Any other source returns None and is not projected.
    """
    if not extra:
        from ..sources.idsse_ballz import ball_z

        return ball_z(match_id, bundle)
    if bundle.meta.source == "pff":
        from ..sources.pff_ballz import ball_z as pff_ball_z

        return pff_ball_z(match_id, bundle)
    return None


def project(ball_xy: np.ndarray, z: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """(T, 2) ground points of the rays from `camera` (cx, cy, cz) through (x, y, z)."""
    cz = float(camera[2])
    zz = np.clip(np.nan_to_num(z, nan=0.0), 0.0, cz - Z_MARGIN)
    scale = cz / (cz - zz)
    c = np.asarray(camera[:2], dtype=np.float64)
    out = c + (ball_xy.astype(np.float64) - c) * scale[:, None]
    out[~np.isfinite(ball_xy[:, 0])] = np.nan
    return out.astype(np.float32)


def draw_camera(width: float, rng: np.random.Generator, cz_min: float, cz_max: float) -> np.ndarray:
    side = -1.0 if rng.random() < 0.5 else 1.0
    cx = rng.uniform(-CX_M, CX_M)
    cz = rng.uniform(cz_min, cz_max)
    return np.array([cx, side * (width / 2.0 + BEHIND_M), cz], dtype=np.float64)


def airborne_ball(
    ball_xy: np.ndarray,
    z: np.ndarray,
    width: float,
    rng: np.random.Generator,
    p: float,
    cz_min: float,
    cz_max: float,
) -> tuple[np.ndarray, bool]:
    """(ball, applied). One coin with probability `p`; when it lands, one camera and the projection."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"airborne p must be in [0, 1], got {p}")
    if not 0.0 < cz_min <= cz_max:
        raise ValueError(f"airborne heights must satisfy 0 < min <= max, got {cz_min}, {cz_max}")
    if rng.random() >= p:
        return ball_xy, False
    camera = draw_camera(width, rng, cz_min, cz_max)
    return project(ball_xy, z, camera), True
