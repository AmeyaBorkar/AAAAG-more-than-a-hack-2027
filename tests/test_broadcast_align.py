"""E58 broadcast alignment and gap fill on synthetic tracking (no match data)."""

from __future__ import annotations

import numpy as np

from aaaag.contracts import A, H, MatchMeta, Period, Tracking
from aaaag.sources.broadcast import align, fill_gaps, mapped_ball, pair_costs

FPS = 25


def _tracking(xy: np.ndarray, ball: np.ndarray) -> Tracking:
    n = len(xy)
    meta = MatchMeta("syn", "synthetic", n, FPS, 105.0, 68.0, (Period(1, 0, n),)).validate()
    teams = np.array(
        [H] * (xy.shape[1] // 2) + [A] * (xy.shape[1] - xy.shape[1] // 2), dtype=np.int8
    )
    return Tracking(
        meta=meta, player_xy=xy.astype(np.float32), slot_team=teams, ball_xy=ball.astype(np.float32)
    )


def _walk(n: int, slots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = rng.uniform([-50, -30], [50, 30], size=(slots, 2))
    steps = rng.normal(0, 0.15, size=(n, slots, 2)).cumsum(axis=0)
    return start[None] + steps


def test_a_shifted_subset_is_placed_at_its_offset():
    tac_xy = _walk(3000, 20, 0)
    ball = tac_xy[:, 0] + 0.5
    tac = _tracking(tac_xy, ball)
    offset = 700
    bro_xy = tac_xy[offset : offset + 1500].copy()
    bro_xy[:, 12:] = np.nan  # the zoomed camera sees a subset
    bro = _tracking(bro_xy, ball[offset : offset + 1500])
    al = align(tac, bro, anchor_players=8, min_players=6)
    placed = al.t_of_b >= 0
    assert placed.mean() > 0.95
    assert np.all(al.t_of_b[placed] - np.flatnonzero(placed) == offset)
    assert np.all(pair_costs(bro, tac, np.array([10]), np.array([10 + offset])) < 0.05)
    cand = mapped_ball(tac, bro, al)
    assert np.allclose(cand[offset + 10], ball[offset + 10], atol=1e-4)


def test_fill_uses_the_candidate_only_when_the_edges_agree():
    n = 200
    truth = np.stack([np.linspace(0, 40, n), np.zeros(n)], axis=1)
    ball = truth.copy()
    ball[80:120] = np.nan
    periods = (Period(1, 0, n),)
    good = truth + np.array([0.3, 0.0])  # a constant camera bias, removed by the edge correction
    out, stats = fill_gaps(ball, good, periods)
    assert stats["gaps_filled"] == 1 and np.allclose(out[80:120], truth[80:120], atol=1e-6)
    bad = truth + np.array([5.0, 0.0])
    out, stats = fill_gaps(ball, bad, periods)
    assert stats["gaps_rejected"] == 1 and np.isnan(out[80:120]).all()
