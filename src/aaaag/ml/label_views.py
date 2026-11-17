"""Label views for extra (non-IDSSE) training matches whose labels are timed by events (E80, #86).

Sources such as the Metrica sample games build possession from tagged events. The team that owns
the ball is mostly right in the middle of a possession, but the moment it changes hands is off by
up to seconds, and spellF1 punishes a shifted boundary. `interior_view` keeps only the frames that
are far from every label change, so such a match can teach who has the ball without teaching when
it changes.

Signals: the label array and the match's periods and frame rate. No tracking.

Where it fails: a turnover the events never recorded leaves wrong-team frames in the middle of a
kept stretch (measured on IDSSE by `analysis/scripts/event_label_timing.py`: the "flag says other
team" column); short possessions vanish entirely, so the kept frames over-represent long, settled
possessions; nothing about the dead ball or about boundary timing is taught. A stoppage the
events did not record stays labelled H or A unless the caller also passes `seen`.
"""

from __future__ import annotations

import numpy as np

from ..contracts import D, MatchMeta, X

# Seconds masked on each side of every label change in "interior" mode. Set from an external
# measurement, never from a score: on the Metrica sample games, 90% of H<->A label changes lie
# within 1.88 s of the nearest touch of the gaining team (`analysis/scripts/event_label_timing.py`,
# the touch yardstick; E80 stage one, #86). 2.0 s is that figure rounded up to half a second.
INTERIOR_S = 2.0


def change_mask(labels: np.ndarray, meta: MatchMeta, half_width: int) -> np.ndarray:
    """(T,) bool: within `half_width` frames of a label change inside a period.

    A change sits between frames c - 1 and c. The mask covers frames c - half_width to
    c + half_width - 1: `half_width` frames on each side. Changes to or from X (a period edge) do
    not count. One cumulative sum per period; no loop over frames.
    """
    labels = np.asarray(labels)
    mask = np.zeros(labels.shape[0], dtype=bool)
    if half_width <= 0:
        return mask
    for period in meta.periods:
        seg = labels[period.start : period.stop]
        n = len(seg)
        if n < 2:
            continue
        change = np.zeros(n, dtype=np.int64)  # change[c] = 1: frames c - 1 and c differ
        change[1:] = (seg[1:] != seg[:-1]) & (seg[1:] != X) & (seg[:-1] != X)
        total = np.concatenate(([0], np.cumsum(change)))
        index = np.arange(n)
        lo = np.clip(index - half_width + 1, 0, n)
        hi = np.clip(index + half_width + 1, 0, n)
        mask[period.start : period.stop] = total[hi] - total[lo] > 0
    return mask


def interior_view(
    labels: np.ndarray,
    meta: MatchMeta,
    window_s: float = INTERIOR_S,
    *,
    seen: np.ndarray | None = None,
) -> np.ndarray:
    """`labels` with D turned into X, and X on every frame within `window_s` seconds of a label
    change (H<->A, or to or from D). Only the interiors of H and A possessions stay.

    `seen`, when given, is a (T,) bool of frames where the source's own ball is observed; the other
    frames become X too. On Metrica's tracking the ball is missing on 92-100% of dead frames (#50),
    so an alive-labelled frame without a ball is likely a stoppage the events did not record.

    `window_s` 0 without `seen` is the "team" view (D becomes X, nothing else). The input is not
    modified, the result is int8, and X is ignored by the training loss.
    """
    if window_s < 0:
        raise ValueError(f"window_s must be >= 0, got {window_s}")
    labels = np.asarray(labels)
    out = labels.astype(np.int8, copy=True)
    near = change_mask(labels, meta, int(round(float(window_s) * meta.fps)))
    out[near | (labels == D)] = X
    if seen is not None:
        seen = np.asarray(seen, dtype=bool)
        if seen.shape != labels.shape:
            raise ValueError(f"seen has shape {seen.shape}, labels {labels.shape}")
        out[~seen] = X
    return out
