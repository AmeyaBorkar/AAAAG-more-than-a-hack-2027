"""Restage the dead ball of a clean labelled match before dropping it (degrade-v3).

The IDSSE provider freezes the ball at the spot where play stopped, through the whole stoppage
(every one of 497 dev stoppages; issue #21). A real ball is fetched and carried to the restart,
and the ATD tracks it moving. Left frozen, the ball sits beyond a line for most of the dead time:
the ball-gap context then fires through stoppages, and any static-ball feature looks perfect.

Here every dead segment (label D inside a period) moves the ball in a straight line from its
freeze point to where play restarts, reaching it on the first alive frame; a segment that ends a
period keeps its freeze point. Gaussian jitter at the match's own alive-ball noise is added, so the
restaged path is neither straighter nor smoother than live play: sigma per axis is the median
second difference on alive frames over sqrt(2 ln 2) * sqrt(6) (the median of a 2-D Rayleigh, and
the second difference of white noise). Nothing is tuned.

Where it fails: the path is invented. A real ball goes out, waits, is thrown or carried, and is
placed; here it drifts steadily (median 0.36 m/s on the dev matches). The adversarial realism check
(arnav-10) is what can tell whether that drift is itself a tell.
"""

from __future__ import annotations

import numpy as np

from ..contracts import D, MatchBundle

_RAYLEIGH_MEDIAN = np.sqrt(2 * np.log(2))


def alive_ball_noise(bundle: MatchBundle) -> float:
    """Per-axis jitter sigma (m) implied by the median second difference on alive frames."""
    ball, labels = bundle.tracking.ball_xy.astype(np.float64), bundle.labels
    second = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=1)
    alive = (labels[2:] != D) & (labels[1:-1] != D) & (labels[:-2] != D) & (labels[1:-1] < D)
    values = second[alive & np.isfinite(second)]
    if not len(values):
        return 0.0
    return float(np.median(values) / (_RAYLEIGH_MEDIAN * np.sqrt(6)))


def restage_dead_ball(bundle: MatchBundle, rng: np.random.Generator) -> np.ndarray:
    """(T, 2) float32 ball positions with every dead segment restaged; alive frames untouched."""
    if bundle.labels is None:
        raise ValueError("restaging the dead ball needs labels (external matches only)")
    ball = bundle.tracking.ball_xy.copy()
    sigma = alive_ball_noise(bundle)
    for period in bundle.meta.periods:
        dead = (bundle.labels[period.start : period.stop] == D).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], dead, [0])))) + period.start
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            origin = ball[start].astype(np.float64)
            if not np.isfinite(origin).all():
                continue
            target = ball[stop].astype(np.float64) if stop < period.stop else origin
            if not np.isfinite(target).all():
                target = origin
            steps = np.arange(stop - start, dtype=np.float64)[:, None] / (stop - start)
            path = origin + (target - origin) * steps
            ball[start:stop] = path + rng.normal(0.0, sigma, path.shape)
    return ball
