"""Ball-gap statistics, computed the same way on BAR-ATH (calibration targets) and on degraded
matches (the realism gate). Aggregates only: nothing here returns frame positions.

A gap is a maximal run of played frames without a ball position, inside one period, that does not
start on the period's first frame (its start context would be unknown). Its context is the cell of
the last seen frame before it (context.py), as in the ATD characterisation finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Tracking
from .context import N_CELLS, context_cells, off_pitch

GATE_QUANTILES = (0.25, 0.5, 0.75, 0.9)


@dataclass
class GapEvents:
    lengths_s: np.ndarray  # (G,) float64
    start_cells: np.ndarray  # (G,) int8, cell of the last seen frame before each gap
    seen_by_cell: np.ndarray  # (N_CELLS,) int64, seen played frames per cell
    played_frames: int
    missing_frames: int
    minutes: float

    @classmethod
    def pool(cls, parts: list[GapEvents]) -> GapEvents:
        return cls(
            np.concatenate([p.lengths_s for p in parts]),
            np.concatenate([p.start_cells for p in parts]),
            np.sum([p.seen_by_cell for p in parts], axis=0),
            sum(p.played_frames for p in parts),
            sum(p.missing_frames for p in parts),
            sum(p.minutes for p in parts),
        )

    def hazard(self) -> np.ndarray:
        """(N_CELLS,) gap starts per seen frame in each cell."""
        starts = np.bincount(self.start_cells, minlength=N_CELLS).astype(np.float64)
        return starts / np.maximum(self.seen_by_cell, 1)

    def summary(self) -> dict[str, float]:
        """The realism-gate statistics (docs: issue #21)."""
        out = {
            "missing_rate": self.missing_frames / max(self.played_frames, 1),
            "gaps_per_min": len(self.lengths_s) / max(self.minutes, 1e-9),
            "near_line_at_starts": float(np.mean(self.start_cells >= 2))
            if len(self.start_cells)
            else float("nan"),
            "n_gaps": float(len(self.lengths_s)),
        }
        for q in GATE_QUANTILES:
            out[f"gap_p{int(q * 100)}_s"] = (
                float(np.quantile(self.lengths_s, q)) if len(self.lengths_s) else float("nan")
            )
        return out


def gap_events(
    tracking: Tracking, *, frozen_as_open: bool = False, on_pitch_only: bool = False
) -> GapEvents:
    """Gap statistics. `on_pitch_only` (degrade-v4 calibration) leaves out seen frames beyond a
    line and the gaps that start from them, which v4's out-of-play mask produces instead."""
    meta, fps = tracking.meta, tracking.meta.fps
    cells = context_cells(tracking, frozen_as_open=frozen_as_open)
    excluded = off_pitch(tracking) if on_pitch_only else np.zeros(len(cells), dtype=bool)
    lengths, start_cells = [], []
    seen_by_cell = np.zeros(N_CELLS, dtype=np.int64)
    played = missing = 0
    for period in meta.periods:
        local = cells[period.start : period.stop]
        out = excluded[period.start : period.stop]
        seen = local >= 0
        seen_by_cell += np.bincount(local[seen & ~out], minlength=N_CELLS)
        played += period.length
        missing += int((~seen).sum())
        edges = np.flatnonzero(np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0]))))
        starts, stops = edges[::2], edges[1::2]
        interior = starts > 0
        interior[interior] = ~out[starts[interior] - 1]
        lengths.append((stops - starts)[interior] / fps)
        start_cells.append(local[starts[interior] - 1])
    return GapEvents(
        lengths_s=np.concatenate(lengths).astype(np.float64),
        start_cells=np.concatenate(start_cells).astype(np.int8),
        seen_by_cell=seen_by_cell,
        played_frames=played,
        missing_frames=missing,
        minutes=played / fps / 60,
    )


def exit_flash_lengths(tracking: Tracking) -> np.ndarray:
    """(F,) seconds of each run of seen off-pitch frames that ends in a gap, inside one period:
    how long the ATD keeps showing a ball that has gone out before losing it (degrade-v4)."""
    fps, off = tracking.meta.fps, off_pitch(tracking)
    seen = tracking.ball_seen()
    lengths = []
    for period in tracking.meta.periods:
        run = off[period.start : period.stop]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], run.astype(np.int8), [0]))))
        starts, stops = edges[::2], edges[1::2]
        ends_in_gap = stops < period.length
        ends_in_gap[ends_in_gap] = ~seen[period.start + stops[ends_in_gap]]
        lengths.append((stops - starts)[ends_in_gap] / fps)
    return np.concatenate(lengths).astype(np.float64)


GATED = (
    "missing_rate",
    "gaps_per_min",
    "gap_p25_s",
    "gap_p50_s",
    "gap_p75_s",
    "gap_p90_s",
    "near_line_at_starts",
)


def bootstrap_intervals(
    tracking: Tracking,
    *,
    block_s: float = 120.0,
    n_resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of the gated statistics under a block bootstrap of one match.

    Blocks are consecutive `block_s` stretches of each period (the last may be shorter); a gap
    belongs to the block of its last seen frame. Resampling whole blocks keeps gaps, their context
    and the time between them together, so the interval reflects how noisy one match's numbers are.
    """
    meta, fps = tracking.meta, tracking.meta.fps
    cells = context_cells(tracking)
    blocks = []  # (played, missing, lengths_s, start_cells)
    for period in meta.periods:
        local = cells[period.start : period.stop]
        seen = local >= 0
        edges = np.flatnonzero(np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0]))))
        starts, stops = edges[::2], edges[1::2]
        interior = starts > 0
        starts, stops = starts[interior], stops[interior]
        size = int(round(block_s * fps))
        for lo in range(0, period.length, size):
            hi = min(lo + size, period.length)
            own = (starts - 1 >= lo) & (starts - 1 < hi)
            blocks.append(
                (
                    hi - lo,
                    int((~seen[lo:hi]).sum()),
                    (stops - starts)[own] / fps,
                    local[starts[own] - 1],
                )
            )
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(n_resamples) for key in GATED}
    for r in range(n_resamples):
        pick = [blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]
        played = sum(b[0] for b in pick)
        summary = GapEvents(
            lengths_s=np.concatenate([b[2] for b in pick]),
            start_cells=np.concatenate([b[3] for b in pick]).astype(np.int8),
            seen_by_cell=np.zeros(N_CELLS, dtype=np.int64),
            played_frames=played,
            missing_frames=sum(b[1] for b in pick),
            minutes=played / fps / 60,
        ).summary()
        for key in GATED:
            samples[key][r] = summary[key]
    tail = (1 - level) / 2
    return {
        key: (float(np.nanquantile(v, tail)), float(np.nanquantile(v, 1 - tail)))
        for key, v in samples.items()
    }
