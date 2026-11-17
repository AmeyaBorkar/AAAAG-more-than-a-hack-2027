"""degrade-v5 dead ball: blank the ball through every stoppage, with lags measured on Metrica.

On Metrica's tracking, made by the provider of the ATD, a missing ball is mostly a dead ball. In
dead play the ball is missing on 92-100% of frames. It goes about one frame after the stoppage
starts and comes back at the restart (#50). So v5 hides the ball through each stoppage of a
labelled external match, here IDSSE's provider `BallStatus`, instead of drawing dead-time gaps
from a hazard.

Every missing-ball run of a labelled match is assigned to one of two processes:
- a run that overlaps or touches a stoppage (a D run inside a period) belongs to the **dead
  process**;
- every other run is an **alive gap** (ballgap.py's hazard and lengths).

Per stoppage [s, e), with its attached runs spanning [a, b):
- the onset lag is a - s, negative when the ball vanished in play just before;
- the return lag is b - e, positive when the ball stays missing into play;
- a stoppage with no attached run keeps its ball.

`measure_stoppages` measures this, and the fit (analysis/scripts/data_degrade_v5_fit.py) runs it
on Metrica only. `dead_ball_mask` applies it: each stoppage keeps its ball with `keep_share`, else
it hides [s + onset, e + return), with the (onset, return) pair drawn jointly from the measured
stoppages. Robustness rungs (#50) shift both ends independently by U(-j, +j) s.

Signals: the labels' stoppages only, never the ball itself.

Where it fails:
- Re-sightings inside a stoppage are not modelled. The fit reports their share.
- Lags come from event-derived Metrica labels, whose stoppage starts are hand-tagged and off by a
  few frames.
- If BAR-ATH blanks less tightly than Metrica, the suites over-reward "missing ball means dead".
  The jitter rungs are there to catch that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import D, Period


@dataclass(frozen=True)
class DeadBall:
    keep_share: float  # share of stoppages whose ball is never hidden
    lags_s: tuple[tuple[float, float], ...]  # (onset, return) per measured stoppage, seconds

    @classmethod
    def from_dict(cls, raw: dict) -> DeadBall:
        model = cls(
            keep_share=float(raw["keep_share"]),
            lags_s=tuple((float(a), float(b)) for a, b in raw["lags_s"]),
        )
        if not 0.0 <= model.keep_share < 1.0 or not model.lags_s:
            raise ValueError(f"bad dead_ball block: keep_share {model.keep_share}, no lags")
        return model


@dataclass(frozen=True)
class Stoppage:
    start: int
    stop: int
    attached: tuple[int, int] | None  # [a, b) spanned by the attached missing runs, or None
    interior_seen: int  # seen frames inside [max(a, start), min(b, stop))
    at_period_edge: bool  # starts at a period start or ends at a period end

    @property
    def onset(self) -> int | None:
        return None if self.attached is None else self.attached[0] - self.start

    @property
    def ret(self) -> int | None:
        return None if self.attached is None else self.attached[1] - self.stop


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and stops of the True runs of a 1-D bool array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def measure_stoppages(
    labels: np.ndarray, seen: np.ndarray, periods: tuple[Period, ...]
) -> tuple[list[Stoppage], np.ndarray]:
    """Per stoppage, its attached missing runs; plus the (T,) mask of frames in attached runs.

    A missing run [a, b) is attached to stoppage [s, e) when a <= e and b >= s (it overlaps the
    stoppage or touches either end). Runs never cross a period boundary.
    """
    stoppages: list[Stoppage] = []
    attached = np.zeros(len(labels), dtype=bool)
    for period in periods:
        lo, hi = period.start, period.stop
        miss_a, miss_b = runs(~seen[lo:hi])
        dead_s, dead_e = runs(labels[lo:hi] == D)
        for s, e in zip(dead_s, dead_e, strict=True):
            hit = (miss_a <= e) & (miss_b >= s)
            span = None
            interior = 0
            if hit.any():
                a, b = int(miss_a[hit].min()), int(miss_b[hit].max())
                span = (lo + a, lo + b)
                inner = slice(lo + max(a, s), lo + min(b, e))
                interior = int(seen[inner].sum())
                for ra, rb in zip(miss_a[hit], miss_b[hit], strict=True):
                    attached[lo + ra : lo + rb] = True
            stoppages.append(
                Stoppage(lo + int(s), lo + int(e), span, interior, s == 0 or e == hi - lo)
            )
    return stoppages, attached


def dead_ball_mask(
    labels: np.ndarray,
    periods: tuple[Period, ...],
    model: DeadBall,
    fps: int,
    rng: np.random.Generator,
    *,
    edge_jitter_s: float = 0.0,
) -> np.ndarray:
    """(T,) bool, True where the ball is hidden by the dead process. One draw per stoppage."""
    mask = np.zeros(len(labels), dtype=bool)
    lags = np.rint(np.asarray(model.lags_s, dtype=np.float64) * fps).astype(np.int64)
    jitter = int(round(edge_jitter_s * fps))
    for period in periods:
        lo, hi = period.start, period.stop
        starts, stops = runs(labels[lo:hi] == D)
        for s, e in zip(starts, stops, strict=True):
            keep = rng.random() < model.keep_share
            onset, ret = lags[rng.integers(len(lags))]
            if jitter:
                onset += int(rng.integers(-jitter, jitter + 1))
                ret += int(rng.integers(-jitter, jitter + 1))
            if keep:
                continue
            first = max(0, int(s) + int(onset))
            last = min(hi - lo, int(e) + int(ret))
            if first < last:
                mask[lo + first : lo + last] = True
    return mask
