"""dead_gated_hsmm: the stoppages are exactly the vote's, the team is segmented inside alive runs."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchMeta, Period, X
from aaaag.decoders.dead_gated_hsmm import dead_gated_hsmm, gate_posterior
from aaaag.decoders.dead_gated_vote import dead_gated_vote
from aaaag.registry import DECODERS

FPS = 25


def make_meta(periods=((0, 3000), (3100, 6000))) -> MatchMeta:
    n = periods[-1][1] + 100
    return MatchMeta(
        match_id="synthetic",
        source="synthetic",
        n_frames=n,
        fps=FPS,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=tuple(Period(i + 1, a, b) for i, (a, b) in enumerate(periods)),
    )


def posterior(meta: MatchMeta, seed: int = 0) -> np.ndarray:
    """A confident but noisy posterior: stoppages every ~20 s, possession flips every ~4 s."""
    rng = np.random.default_rng(seed)
    proba = np.full((meta.n_frames, 3), np.nan, dtype=np.float32)
    for start, stop in meta.runs():
        n = stop - start
        team = np.zeros(n, dtype=np.int8)
        t = 0
        side = 0
        while t < n:
            length = int(rng.exponential(4 * FPS)) + FPS
            team[t : t + length] = side
            side ^= 1
            t += length
        dead = np.zeros(n, dtype=bool)
        t = int(rng.integers(300, 600))
        while t < n:
            length = int(rng.integers(3 * FPS, 10 * FPS))
            dead[t : t + length] = True
            t += length + int(rng.integers(10 * FPS, 30 * FPS))
        p_home = np.where(team == 0, 0.9, 0.1)
        flip = rng.random(n) < 0.1  # frame noise on the team call
        p_home = np.where(flip, 1.0 - p_home, p_home)
        p_dead = np.where(dead, 0.97, 0.02)
        seg = np.zeros((n, 3), dtype=np.float32)
        seg[:, 2] = p_dead
        seg[:, 0] = (1.0 - p_dead) * p_home
        seg[:, 1] = (1.0 - p_dead) * (1.0 - p_home)
        proba[start:stop] = seg
    return proba


def test_registered_with_an_example():
    component = DECODERS.get("dead_gated_hsmm")
    assert component.version == "1"
    assert component.example


def test_dead_mask_is_the_votes():
    meta = make_meta()
    proba = posterior(meta)
    labels = dead_gated_hsmm(proba, meta)
    vote = dead_gated_vote(proba, meta, half_window=12, min_dead_s=2.0, min_alive_s=1.0)
    played = meta.played_mask()
    assert np.isin(labels[played & (labels != D)], (H, A)).all()
    assert (labels[~played] == X).all()
    # the segmentation runs on 5-frame blocks, so an edge of the mask may move by up to 2 frames
    # (a boundary refinement would put it back); everywhere else the two masks are the same
    ours, theirs = (labels == D) & played, (vote == D) & played
    differing = np.flatnonzero(ours != theirs)
    edges = np.flatnonzero(np.diff(theirs.astype(int)) != 0)
    assert len(edges) > 0
    for index in differing:
        assert np.abs(edges - index).min() <= 2


def test_gate_posterior_forces_the_mask_and_keeps_the_odds():
    meta = make_meta()
    proba = posterior(meta)
    gated = gate_posterior(proba, meta)
    played = meta.played_mask()
    dead = gated[:, 2] == 1.0
    assert (gated[dead & played][:, :2] == 0.0).all()
    alive = played & ~dead
    assert np.allclose(gated[alive, 2], 1e-6)
    ratio_in = proba[alive, 0] / (proba[alive, 0] + proba[alive, 1])
    ratio_out = gated[alive, 0] / (gated[alive, 0] + gated[alive, 1])
    assert np.allclose(ratio_in, ratio_out, atol=1e-5)
    assert proba is not gated and not np.shares_memory(proba, gated)


def test_deterministic_and_nan_rows_are_harmless():
    meta = make_meta()
    proba = posterior(meta)
    proba[500:520] = np.nan  # no opinion
    a = dead_gated_hsmm(proba, meta)
    b = dead_gated_hsmm(proba, meta)
    assert np.array_equal(a, b)
    played = meta.played_mask()
    assert np.isin(a[played], (H, A, D)).all()


def test_fewer_spells_than_the_frame_votes_and_a_minimum_spell_removes_short_ones():
    meta = make_meta()
    proba = posterior(meta, seed=3)
    labels = dead_gated_hsmm(proba, meta)
    raw = np.where(proba[:, 0] > proba[:, 1], H, A).astype(np.int8)
    played = meta.played_mask() & (labels != D)
    changes_labels = int((np.diff(labels[played]) != 0).sum())
    changes_raw = int((np.diff(raw[played]) != 0).sum())
    assert changes_labels < changes_raw
    pruned = dead_gated_hsmm(proba, meta, min_spell_s=2.0)
    assert int((np.diff(pruned[meta.played_mask() & (pruned != D)]) != 0).sum()) <= changes_labels


@pytest.mark.slow
def test_a_match_decodes_in_about_a_second():
    import time

    meta = make_meta(((0, 80000), (80100, 160000)))
    proba = posterior(meta, seed=5)
    started = time.perf_counter()
    dead_gated_hsmm(proba, meta)
    assert time.perf_counter() - started < 3.0
