"""`refine_boundaries(mode="switch")` (E42): a H/A edge moves to the switch-logit peak. Synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchMeta, Period
from aaaag.decoders.hsmm import refine_boundaries


def _meta(n: int) -> MatchMeta:
    return MatchMeta(
        match_id="t", source="synthetic", n_frames=n, fps=25, pitch_length=105.0,
        pitch_width=68.0, periods=(Period(1, 0, n),), home="h", away="a",
    ).validate()  # fmt: skip


def test_switch_mode_moves_a_team_edge_to_the_switch_peak_and_leaves_dead_edges():
    n = 200
    meta = _meta(n)
    labels = np.array([H] * 100 + [A] * 60 + [D] * 40, dtype=np.int8)
    proba = np.tile(np.array([0.45, 0.45, 0.10], dtype=np.float32), (n, 1))
    proba[160:] = (0.05, 0.05, 0.90)  # a clear stoppage from frame 160: that edge stays
    switch = np.full(n, -3.0)
    switch[106] = 4.0  # the learned change is 6 frames later than the decoded edge
    out = refine_boundaries(labels, proba, meta, max_shift_s=0.4, mode="switch", switch=switch)
    assert np.all(out[:106] == H) and np.all(out[106:160] == A)
    assert np.all(out[160:] == D)
    same = refine_boundaries(
        labels, proba, meta, max_shift_s=0.4, mode="switch", switch=np.zeros(n)
    )
    assert np.array_equal(same, labels)  # a flat logit keeps the edge where it is


def test_switch_mode_needs_the_logit():
    meta = _meta(10)
    labels = np.array([H] * 5 + [A] * 5, dtype=np.int8)
    with pytest.raises(ValueError):
        refine_boundaries(labels, np.full((10, 3), 1 / 3), meta, max_shift_s=0.2, mode="switch")
