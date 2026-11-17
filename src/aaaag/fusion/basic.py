"""Generic, parameter-light fusion rules. Learned fusion belongs in its own module."""

from __future__ import annotations

import numpy as np

from ..contracts import Evidence, MatchMeta
from ..registry import FUSION


@FUSION.register("first", version="1")
def first(evidence: list[Evidence], meta: MatchMeta) -> np.ndarray:
    """The first producer's opinion, unchanged. For single-evidence pipelines."""
    return evidence[0].proba.copy()


@FUSION.register("mean", version="1")
def mean(
    evidence: list[Evidence], meta: MatchMeta, *, weights: list[float] | None = None
) -> np.ndarray:
    """Weighted linear pool over the producers that have an opinion on each frame."""
    stack, w = _stack(evidence, weights)
    has = ~np.isnan(stack[:, :, 0])
    total = (np.nan_to_num(stack) * (w[:, None, None] * has[:, :, None])).sum(axis=0)
    norm = total.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / norm
    out[~has.any(axis=0)] = np.nan
    return out.astype(np.float32)


@FUSION.register("log_pool", version="1")
def log_pool(
    evidence: list[Evidence],
    meta: MatchMeta,
    *,
    weights: list[float] | None = None,
    floor: float = 1e-4,
) -> np.ndarray:
    """Weighted geometric pool (product of experts) with a probability floor, NaN-aware.

    Treats producers as conditionally independent experts: two weak agreeing signals compound,
    and one confident producer can override a vague one. The floor stops a single hard zero from
    vetoing a class outright.
    """
    stack, w = _stack(evidence, weights)
    has = ~np.isnan(stack[:, :, 0])
    logs = np.log(np.clip(np.nan_to_num(stack, nan=1.0), floor, 1.0))
    score = (logs * (w[:, None, None] * has[:, :, None])).sum(axis=0)
    score -= score.max(axis=1, keepdims=True)
    out = np.exp(score)
    out /= out.sum(axis=1, keepdims=True)
    out[~has.any(axis=0)] = np.nan
    return out.astype(np.float32)


def _stack(evidence: list[Evidence], weights: list[float] | None) -> tuple[np.ndarray, np.ndarray]:
    if not evidence:
        raise ValueError("fusion needs at least one evidence producer")
    w = np.ones(len(evidence)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(evidence),):
        raise ValueError(f"{len(evidence)} producers but {len(w)} weights")
    return np.stack([e.proba.astype(np.float64) for e in evidence]), w
