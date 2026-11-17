"""Boundary-weighted frame weights for the team cross entropy (E155, #86).

spellF1 matches possessions one to one by temporal IoU, so where a possession starts and stops
decides whether it counts. A frame cross entropy spends most of its weight deep inside spells,
where the network is already right. This weight raises the frames next to a change of the training
target label: a frame within ``window_s`` seconds of a change (H, A or D on both sides) weighs
``factor``, every other frame 1. It multiplies the class weights and, when they are on, the spell
weights (spellweight.py); the cross entropy stays the mean over the product of all the weights,
the convention of the spell weights and of ``cross_entropy(weight=)``.

The idea (3.0 within 1.0 s) is a public competitor's unverified description; the two numbers are
taken as they are, not swept. Signals: only the training target labels and the match periods. The
weights are built once per match on the full label sequence, after any label rewrite
(``target_rule``, ``target_min_s``) and before any window is cut, so a window edge never hides a
change, and a change never reaches across a period edge. X is an unknown label: a step into or out
of X is not a change. A team swap leaves the weights as they are, because a change is a change
whichever team holds the ball; a y flip does not read the labels.

Where it fails: the label changes of the training matches carry the annotators' timing noise, and
tripling the frames around them triples that noise too. The posterior near a change moves, so a
decoder tuned on the plain loss may no longer be exactly right; E155 judges the whole stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import X


@dataclass(frozen=True)
class BoundaryWeight:
    factor: float = 3.0  # the weight of a frame near a change; 1 elsewhere
    window_s: float = 1.0  # seconds on each side of the change

    def __post_init__(self) -> None:
        for name in ("factor", "window_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"boundary_weight.{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"boundary_weight.{name} must be finite, got {value!r}")
            if not isinstance(value, float):
                object.__setattr__(self, name, float(value))
        if not self.factor > 0.0:
            raise ValueError(f"boundary_weight.factor must be above 0, got {self.factor}")
        if not self.window_s >= 0.0:
            raise ValueError(f"boundary_weight.window_s must be >= 0, got {self.window_s}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BoundaryWeight:
        if not isinstance(raw, dict):
            raise ValueError(f"boundary_weight must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - {"factor", "window_s"}
        if unknown:
            raise ValueError(f"unknown boundary_weight keys: {sorted(unknown)}")
        return cls(**raw)

    def radius(self, fps: int) -> int:
        """Frames on each side of a change: ``window_s * fps``, rounded."""
        return int(round(self.window_s * fps))


def boundary_mask(labels: np.ndarray, runs: Iterable[tuple[int, int]], radius: int) -> np.ndarray:
    """(T,) bool, True on the frames within ``radius`` frames of a change of label.

    A change sits between frames t-1 and t of one period (``runs`` holds the (start, stop) of each
    period) whose labels differ and are both H, A or D. It marks the frames t-radius .. t+radius-1
    that lie inside the same period: ``radius`` frames of the old label and ``radius`` of the new.
    Frames outside every period are never marked. One numpy pass per period, no frame loop.
    """
    lab = np.asarray(labels)
    n = int(lab.shape[0])
    out = np.zeros(n, dtype=bool)
    if radius <= 0:
        return out
    known = lab != X
    for start, stop in runs:
        a, b = max(0, int(start)), min(n, int(stop))
        if b - a < 2:
            continue
        seg, ok = lab[a:b], known[a:b]
        change = np.flatnonzero((seg[1:] != seg[:-1]) & ok[1:] & ok[:-1]) + 1
        if change.size == 0:
            continue
        # +1 where a marked stretch opens, -1 where it closes; the running sum counts the
        # stretches covering each frame.
        steps = np.zeros(b - a + 1, dtype=np.int64)
        np.add.at(steps, np.clip(change - radius, 0, b - a), 1)
        np.add.at(steps, np.clip(change + radius, 0, b - a), -1)
        out[a:b] = np.cumsum(steps[:-1]) > 0
    return out


def boundary_weights(
    labels: np.ndarray, runs: Iterable[tuple[int, int]], fps: int, cfg: BoundaryWeight
) -> np.ndarray:
    """(T,) float32: ``cfg.factor`` within ``cfg.window_s`` of a change (`boundary_mask`), else 1."""
    near = boundary_mask(labels, runs, cfg.radius(fps))
    return np.where(near, np.float32(cfg.factor), np.float32(1.0)).astype(np.float32)
