"""Spell-balanced frame weights for the team cross entropy (plan E5, the cheap part).

spellF1 counts every possession once, however long it lasts, while a frame cross entropy lets a
20 s spell weigh 40 times a 0.5 s one. The loss anatomy of the candidate (#86) puts 48% of what is
left in short true spells that the prediction holds through. This module re-weights each alive
frame by the length of the spell it sits in, ``(ref_s / max(L, min_s)) ** gamma`` with L in seconds,
so short spells weigh more and very long ones less. gamma 0 is the plain loss.

Two views of L (`length_view`):

- ``provider`` (the default, the behaviour before E154): L is the run of identical labels inside the
  batch window. A run that touches a window edge keeps weight 1, because its true length is unknown
  there.
- ``aba1.0`` (E154): L is measured once per match on the A-B-A-merged view of its labels
  (`aba_merge` with N = 1.0 s, the "aba1.0" truth of the E106 analysis), so a sub-second flicker
  inside a possession takes the length of the merged spell around it. The class targets stay the
  labels as they are; only the weights read the merged view. No window edge rule is needed, since the
  length is the whole-match one.

Signals: only the labels. Dead frames (and the X label) keep weight 1. Where it fails: the weights
move the posterior toward "the possession changes", so the decoder's `lam_team`, tuned on the plain
loss, is no longer exactly right; E5 judges the whole stack, not the loss. The aba view assumes a
sub-second A-B-A interruption is labelling flicker, which is wrong for a genuine one-touch
interception that the ball returns from within a second.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

LIVE = (0, 1)  # aaaag.contracts H, A
LENGTH_VIEWS = {"provider": None, "aba1.0": 1.0}  # view -> A-B-A merge horizon N in seconds


@dataclass(frozen=True)
class SpellWeight:
    gamma: float = 0.5  # 0 switches the weights off
    ref_s: float = 6.0  # a spell this long keeps weight 1
    min_s: float = 0.5  # shorter spells are treated as this long, which caps the weight
    fps: int = 25
    length_view: str = "provider"  # "aba1.0": match-level lengths on the A-B-A-merged labels

    def __post_init__(self) -> None:
        for name in ("gamma", "ref_s", "min_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"spell_weight.{name} must be a finite number >= 0, got {value}")
        if float(self.ref_s) <= 0.0 or float(self.min_s) <= 0.0 or int(self.fps) <= 0:
            raise ValueError("spell_weight.ref_s, min_s and fps must be positive")
        if self.length_view not in LENGTH_VIEWS:
            raise ValueError(
                f"spell_weight.length_view must be one of {sorted(LENGTH_VIEWS)}, "
                f"got {self.length_view!r}"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SpellWeight:
        if not isinstance(raw, dict):
            raise ValueError(f"spell_weight must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - {"gamma", "ref_s", "min_s", "fps", "length_view"}
        if unknown:
            raise ValueError(f"unknown spell_weight keys: {sorted(unknown)}")
        return cls(
            gamma=float(raw.get("gamma", 0.5)),
            ref_s=float(raw.get("ref_s", 6.0)),
            min_s=float(raw.get("min_s", 0.5)),
            fps=int(raw.get("fps", 25)),
            length_view=str(raw.get("length_view", "provider")),
        )

    @property
    def max_weight(self) -> float:
        return float((self.ref_s / self.min_s) ** self.gamma) if self.ref_s > self.min_s else 1.0

    @property
    def match_level(self) -> bool:
        """True when the weights need the per-match lengths (`match_spell_seconds`) in the batch."""
        return float(self.gamma) != 0.0 and self.length_view != "provider"


def run_lengths(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per frame, the length (frames) of its run of identical labels within the row, and whether
    that run touches either end of the row. ``labels`` is (B, W) integer.

    No step reads a value back to the host (no ``bincount``, no ``int(tensor)``): a host sync per
    training step stalls a GPU that other runs share, and made this loss many times slower."""
    batch, width = labels.shape
    n = batch * width
    flat = labels.reshape(-1)
    start = torch.ones_like(flat, dtype=torch.bool)
    start[1:] = flat[1:] != flat[:-1]
    start[::width] = True  # a run never crosses from one window into the next
    run_id = torch.cumsum(start.long(), dim=0) - 1
    counts = torch.zeros(n, dtype=torch.long, device=labels.device)
    counts.scatter_add_(0, run_id, torch.ones_like(run_id))
    at_edge = torch.zeros(n, dtype=torch.bool, device=labels.device)
    at_edge[run_id[::width]] = True
    at_edge[run_id[width - 1 :: width]] = True
    return counts[run_id].reshape(batch, width), at_edge[run_id].reshape(batch, width)


def _merge_segment(seg: np.ndarray, limit: float) -> None:
    """`aba_merge` on one live segment, in place. Shortest interior run first, the earliest on a
    tie, the same order as the reference loop in analysis/scripts/e106_hold_regret.py; a heap with
    lazy deletion keeps it O(R log R) in the R runs of the segment."""
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    if change.size < 2:
        return  # fewer than three runs: no interior run
    starts = np.concatenate(([0], change)).tolist()
    ends = [*starts[1:], len(seg)]
    labels = seg[starts].tolist()
    n = len(starts)
    prev = [i - 1 for i in range(n)]  # -1: none
    nxt = [i + 1 if i + 1 < n else -1 for i in range(n)]
    alive = [True] * n
    heap = [(ends[i] - starts[i], starts[i], i) for i in range(1, n - 1)]
    heap = [item for item in heap if item[0] < limit]
    heapq.heapify(heap)
    while heap:
        length, _, i = heapq.heappop(heap)
        if not alive[i] or ends[i] - starts[i] != length:
            continue  # merged away, or re-pushed with a new length
        left, right = prev[i], nxt[i]
        if left < 0 or right < 0:
            continue  # no longer interior (cannot happen: a merge keeps the outer runs)
        # Live labels are H or A only, so an interior run always sits between two runs of the
        # other label: every interior run is an A-B-A. The left run absorbs the other two.
        ends[left] = ends[right]
        alive[i] = alive[right] = False
        nxt[left] = nxt[right]
        if nxt[right] >= 0:
            prev[nxt[right]] = left
        if prev[left] >= 0 and nxt[left] >= 0 and ends[left] - starts[left] < limit:
            heapq.heappush(heap, (ends[left] - starts[left], starts[left], left))
    i = 0
    while i >= 0:
        seg[starts[i] : ends[i]] = labels[i]
        i = nxt[i]


def aba_merge(labels: np.ndarray, fps: float, n_s: float) -> np.ndarray:
    """Inside each run of live (H/A) frames, an A-B-A whose interior B is shorter than `n_s`
    seconds becomes A, shortest interior first, until none is left. Non-live frames are kept and
    end a live run. Same output as `aba_merge` in analysis/scripts/e106_hold_regret.py."""
    out = np.array(labels, copy=True)
    limit = n_s * fps
    live = np.isin(out, LIVE)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], live.astype(np.int8), [0]))))
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        _merge_segment(out[lo:hi], limit)
    return out


def match_run_frames(labels: np.ndarray) -> np.ndarray:
    """(T,) int64: per frame, the length of its run of identical labels over the whole match."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.ones(labels.size, dtype=bool)
    start[1:] = labels[1:] != labels[:-1]
    run_id = np.cumsum(start) - 1
    return np.bincount(run_id)[run_id].astype(np.int64)


def match_spell_seconds(labels: np.ndarray, fps: float, view: str) -> np.ndarray:
    """(T,) float32: per frame, the length in seconds of its spell in `view` over the whole match.

    ``aba1.0`` measures the runs of `aba_merge(labels, fps, 1.0)`; ``provider`` the runs of the
    labels as they are. Dead and X frames get the length of their own run, which the weights
    ignore. `fps` is the match's frame rate."""
    if view not in LENGTH_VIEWS:
        raise ValueError(f"unknown spell length view {view!r}")
    horizon = LENGTH_VIEWS[view]
    merged = labels if horizon is None else aba_merge(labels, fps, horizon)
    return (match_run_frames(merged) / float(fps)).astype(np.float32)


def spell_weights(
    labels: torch.Tensor, cfg: SpellWeight, seconds: torch.Tensor | None = None
) -> torch.Tensor:
    """(B, W) float32 weights; 1 for dead and X frames and for gamma 0.

    ``provider``: the run lengths of the window, and 1 for runs at a window edge; `seconds` is not
    read. ``aba1.0``: `seconds` (B, W) are the match-level lengths of `match_spell_seconds`, sliced
    with the window, and are required."""
    if float(cfg.gamma) == 0.0:
        return torch.ones(labels.shape, dtype=torch.float32, device=labels.device)
    if cfg.length_view == "provider":
        length, at_edge = run_lengths(labels)
        length_s = (length.float() / float(cfg.fps)).clamp_min(float(cfg.min_s))
        weight = (float(cfg.ref_s) / length_s) ** float(cfg.gamma)
        keep = at_edge | (labels >= 2)
        return torch.where(keep, torch.ones_like(weight), weight)
    if seconds is None:
        raise ValueError(
            f"spell_weight.length_view {cfg.length_view!r} needs the match-level spell lengths "
            "in the batch ('spell_s')"
        )
    weight = (float(cfg.ref_s) / seconds.float().clamp_min(float(cfg.min_s))) ** float(cfg.gamma)
    return torch.where(labels >= 2, torch.ones_like(weight), weight)
