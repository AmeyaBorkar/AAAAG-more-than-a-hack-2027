"""Player-track statistics, computed the same way on BAR-ATH (calibration targets) and on degraded
IDSSE matches (the degrade-p1 gate, issue #38). Aggregates only: nothing here returns positions.

All statistics are over played frames:
- objects per team per frame: p10, the share below 10 and the share above 11 (both teams pooled);
- track fragments (maximal runs of a slot being seen): per minute, and length p25 / p50 / p75;
- frame-to-frame acceleration |x[t+1] - 2 x[t] + x[t-1]| * fps^2 of seen slots: p50 / p90;
- the share of frames with a same-team pair of objects closer than CLOSE_PAIR_M.

Accelerations are kept as a fixed-bin histogram so blocks can be pooled and bootstrapped cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import AWAY, HOME, Tracking

CLOSE_PAIR_M = 0.5
ACCEL_BIN = 0.01  # m/s^2
ACCEL_MAX = 60.0
MAX_OBJECTS = 40
PLAYER_GATED = (
    "objects_p10",
    "objects_below_10",
    "objects_above_11",
    "fragments_per_min",
    "fragment_p25_s",
    "fragment_p50_s",
    "fragment_p75_s",
    "accel_p50",
    "accel_p90",
    "close_pair_share",
)
_EDGES = np.arange(0.0, ACCEL_MAX + ACCEL_BIN, ACCEL_BIN)


@dataclass
class PlayerTally:
    """Sufficient statistics of one stretch of play; tallies add up."""

    played_frames: int
    object_counts: np.ndarray  # (MAX_OBJECTS + 1,) team-frames with n objects
    fragment_s: np.ndarray  # (F,) fragment lengths
    accel_hist: np.ndarray  # (len(_EDGES) - 1,) counts; values above ACCEL_MAX in the last bin
    close_pair_frames: int
    fps: int

    @classmethod
    def pool(cls, parts: list[PlayerTally]) -> PlayerTally:
        return cls(
            sum(p.played_frames for p in parts),
            np.sum([p.object_counts for p in parts], axis=0),
            np.concatenate([p.fragment_s for p in parts]),
            np.sum([p.accel_hist for p in parts], axis=0),
            sum(p.close_pair_frames for p in parts),
            parts[0].fps,
        )

    def summary(self) -> dict[str, float]:
        counts = self.object_counts / max(self.object_counts.sum(), 1)
        cdf = np.cumsum(counts)
        accel_cdf = np.cumsum(self.accel_hist) / max(self.accel_hist.sum(), 1)
        minutes = self.played_frames / self.fps / 60
        out = {
            "objects_p10": float(np.searchsorted(cdf, 0.1)),
            "objects_below_10": float(counts[:10].sum()),
            "objects_above_11": float(counts[12:].sum()),
            "fragments_per_min": len(self.fragment_s) / max(minutes, 1e-9),
            "accel_p50": float(_EDGES[np.searchsorted(accel_cdf, 0.5) + 1]),
            "accel_p90": float(_EDGES[np.searchsorted(accel_cdf, 0.9) + 1]),
            "close_pair_share": self.close_pair_frames / max(self.played_frames, 1),
        }
        for q in (25, 50, 75):
            out[f"fragment_p{q}_s"] = (
                float(np.percentile(self.fragment_s, q)) if len(self.fragment_s) else float("nan")
            )
        return out


def player_tally(
    tracking: Tracking, start: int, stop: int, fragment_s: np.ndarray | None = None
) -> PlayerTally:
    """Tally of frames [start, stop), which must lie inside one period. `fragment_s` overrides the
    fragments (the bootstrap passes whole fragments that start in the block)."""
    fps = tracking.meta.fps
    xy = tracking.player_xy[start:stop]
    seen = ~np.isnan(xy[..., 0])
    counts = np.zeros(MAX_OBJECTS + 1, dtype=np.int64)
    close = np.zeros(stop - start, dtype=bool)
    for team in (HOME, AWAY):
        slots = tracking.slot_team == team
        n = np.minimum(seen[:, slots].sum(axis=1), MAX_OBJECTS)
        counts += np.bincount(n, minlength=MAX_OBJECTS + 1)
        close |= _close_pair(xy[:, slots])
    if fragment_s is None:
        fragment_s = _fragments(seen)[1] / fps
    accel = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=2) * fps**2
    accel = np.minimum(accel[np.isfinite(accel)], ACCEL_MAX - ACCEL_BIN / 2)
    hist = np.histogram(accel, _EDGES)[0]
    return PlayerTally(stop - start, counts, fragment_s, hist, int(close.sum()), fps)


def player_events(tracking: Tracking) -> PlayerTally:
    return PlayerTally.pool(
        [player_tally(tracking, p.start, p.stop) for p in tracking.meta.periods]
    )


def player_bootstrap_intervals(
    tracking: Tracking,
    *,
    block_s: float = 120.0,
    n_resamples: int = 2_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of PLAYER_GATED under a block bootstrap of one match (as stats.py).

    A fragment belongs, whole, to the block it starts in, so blocks don't cut fragments."""
    fps = tracking.meta.fps
    size = int(round(block_s * fps))
    blocks = []
    for p in tracking.meta.periods:
        starts, lengths = _fragments(~np.isnan(tracking.player_xy[p.start : p.stop, :, 0]))
        for lo in range(0, p.length, size):
            own = (starts >= lo) & (starts < lo + size)
            blocks.append(
                player_tally(
                    tracking, p.start + lo, min(p.start + lo + size, p.stop), lengths[own] / fps
                )
            )
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(n_resamples) for key in PLAYER_GATED}
    for r in range(n_resamples):
        summary = PlayerTally.pool(
            [blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]
        ).summary()
        for key in PLAYER_GATED:
            samples[key][r] = summary[key]
    tail = (1 - level) / 2
    return {
        key: (float(np.nanquantile(v, tail)), float(np.nanquantile(v, 1 - tail)))
        for key, v in samples.items()
    }


def _fragments(seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start frames and lengths in frames of every run of a (T, S) seen mask, all slots."""
    pad = np.zeros((1, seen.shape[1]), np.int8)
    edges = np.diff(np.concatenate([pad, seen.astype(np.int8), pad]), axis=0).T.ravel()
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return starts % (seen.shape[0] + 1), stops - starts


def _close_pair(team_xy: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """(T,) bool: two objects of the team closer than CLOSE_PAIR_M."""
    out = np.zeros(len(team_xy), dtype=bool)
    rows, cols = np.triu_indices(team_xy.shape[1], 1)
    for lo in range(0, len(team_xy), chunk):
        part = team_xy[lo : lo + chunk]
        gap = part[:, rows] - part[:, cols]
        with np.errstate(invalid="ignore"):
            d2 = np.einsum("tpk,tpk->tp", gap, gap)
            out[lo : lo + chunk] = (d2 < CLOSE_PAIR_M**2).any(axis=1)
    return out
