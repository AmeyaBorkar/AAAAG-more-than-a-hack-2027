"""Centred majority vote -- the official baseline's smoothing, plus per-period operation."""

from __future__ import annotations

import numpy as np

from ..contracts import A, D, H, MatchMeta, X
from ..registry import DECODERS


@DECODERS.register("majority_vote", version="1", example={"half_window": 25})
def majority_vote(
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    half_window: int,
    per_period: bool = True,
    outside: str = "X",
) -> np.ndarray:
    """Each frame takes the majority team among per-frame argmax votes within +/- half_window.

    Frames whose argmax is D, or that have no opinion, vote for nobody; a window with no votes is
    D; ties go to H. With per_period=False and outside="keep" this is byte-for-byte the official
    baseline (tests/test_official_parity.py). O(T) via cumulative sums.
    """
    top = argmax_or_none(proba)
    home, away = top == H, top == A
    labels = np.full(len(top), D, dtype=np.int8)
    spans = meta.runs() if per_period else [(0, len(top))]
    for start, stop in spans:
        labels[start:stop] = _vote(home[start:stop], away[start:stop], half_window)
    if outside == "X":
        labels[~meta.played_mask()] = X
    elif outside != "keep":
        raise ValueError(f"outside must be 'X' or 'keep', got {outside!r}")
    return labels


def argmax_or_none(proba: np.ndarray) -> np.ndarray:
    """(T,) int8 argmax class, -1 where the row has no opinion."""
    unknown = np.isnan(proba[:, 0])
    top = np.nan_to_num(proba, nan=-1.0).argmax(axis=1).astype(np.int8)
    top[unknown] = -1
    return top


def _vote(home: np.ndarray, away: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0:
        return np.where(home, H, np.where(away, A, D)).astype(np.int8)
    n = len(home)
    home_cum = np.concatenate(([0], np.cumsum(home)))
    away_cum = np.concatenate(([0], np.cumsum(away)))
    index = np.arange(n)
    low = np.maximum(0, index - half_window)
    high = np.minimum(n, index + half_window + 1)
    h = home_cum[high] - home_cum[low]
    a = away_cum[high] - away_cum[low]
    out = np.full(n, D, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], H, A)
    return out
