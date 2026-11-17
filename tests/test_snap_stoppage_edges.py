"""#66: snapping stoppage edges onto the overlapping unseen-ball run. Synthetic arrays and matches
only; nothing here selects."""

import numpy as np

from aaaag.contracts import MatchMeta, Period
from aaaag.evidence.ballstate import model
from aaaag.evidence.ballstate.snap import snap_to_ball_gaps

FPS = 25


def _meta(n: int = 1000) -> MatchMeta:
    return MatchMeta(
        match_id="snap-test",
        source="synthetic",
        n_frames=n,
        fps=FPS,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(Period(index=1, start=0, stop=n),),
    ).validate()


def _dead(n: int, runs: list[tuple[int, int]], inside: float = 0.9, outside: float = 0.1):
    p = np.full(n, outside)
    for start, stop in runs:
        p[start:stop] = inside
    return p


def _seen(n: int, gaps: list[tuple[int, int]]) -> np.ndarray:
    seen = np.ones(n, dtype=bool)
    for start, stop in gaps:
        seen[start:stop] = False
    return seen


def _dead_runs(p: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], (p >= 0.5).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def test_edges_move_onto_the_gap_inside_the_bound():
    # predicted stoppage 300..500; the ball vanishes at 313 and returns at 485
    p = _dead(1000, [(300, 500)])
    out = snap_to_ball_gaps(p, _seen(1000, [(313, 485)]), _meta(), max_s=2.0)
    assert _dead_runs(out) == [(312, 485)]  # onset one frame before the ball vanishes
    # an early stoppage start, late end: and the other way round, growing it
    p = _dead(1000, [(320, 470)])
    out = snap_to_ball_gaps(p, _seen(1000, [(313, 485)]), _meta(), max_s=2.0)
    assert _dead_runs(out) == [(312, 485)]
    assert np.allclose(out[312:320], 0.9) and np.allclose(out[470:485], 0.9)


def test_no_move_beyond_the_bound():
    p = _dead(1000, [(300, 500)])
    # start would move 60 frames (2.4 s): it stays; the end moves 10 frames: it moves
    out = snap_to_ball_gaps(p, _seen(1000, [(361, 490)]), _meta(), max_s=2.0)
    assert _dead_runs(out) == [(300, 490)]
    out = snap_to_ball_gaps(p, _seen(1000, [(361, 490)]), _meta(), max_s=0.2)
    assert _dead_runs(out) == [(300, 500)]  # 10 frames > 0.2 s: nothing moves


def test_several_gaps_use_the_earliest_start_and_the_latest_end():
    p = _dead(1000, [(300, 500)])
    out = snap_to_ball_gaps(
        p, _seen(1000, [(310, 330), (360, 400), (450, 480)]), _meta(), max_s=2.0
    )
    assert _dead_runs(out) == [(309, 480)]


def test_no_overlapping_gap_changes_nothing():
    p = _dead(1000, [(300, 500)])
    seen = _seen(1000, [(100, 150), (600, 650)])
    assert np.array_equal(snap_to_ball_gaps(p, seen, _meta(), max_s=4.0), p)
    assert np.array_equal(snap_to_ball_gaps(p, np.ones(1000, bool), _meta(), max_s=4.0), p)


def test_edges_at_a_period_boundary_never_move():
    p = _dead(1000, [(0, 200)])
    out = snap_to_ball_gaps(p, _seen(1000, [(10, 190)]), _meta(), max_s=2.0)
    assert _dead_runs(out) == [(0, 190)]


def test_none_reproduces_the_producer_exactly(small_bundle):
    params = {"weights": {"ball_missing": 1.0}}
    base = model.ballstate(small_bundle, **params).proba
    same = model.ballstate(small_bundle, **params, snap_to_ball_gaps_s=None).proba
    assert np.array_equal(base, same, equal_nan=True)


def test_snap_keeps_the_evidence_contract(small_bundle):
    ev = model.ballstate(small_bundle, weights={"ball_missing": 1.0}, snap_to_ball_gaps_s=2.0)
    assert ev.proba.shape == (small_bundle.meta.n_frames, 3)
    played = small_bundle.meta.played_mask()
    assert np.isfinite(ev.proba[played]).all()
