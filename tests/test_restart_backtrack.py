"""E68 restart_backtrack: late restart edges traced back from the reappearing ball. Synthetic only."""

from __future__ import annotations

import numpy as np

from aaaag.contracts import A, D, H, MatchBundle, MatchMeta, Period, Tracking
from aaaag.postprocess.restart_backtrack import restart_backtrack, trace_kick
from aaaag.registry import POSTPROCESS

FPS = 25
N = 1000
KICK = 500  # the true restart: the home taker at (10, 30) kicks the ball at 10 m/s along +x


def _bundle(*, ball_from: int = 550, speed: float = 10.0, taker_team: int = H) -> MatchBundle:
    meta = MatchMeta(
        match_id="syn-backtrack",
        source="synthetic",
        n_frames=N,
        fps=FPS,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(Period(1, 0, N),),
    )
    players = np.full((N, 4, 2), np.nan, dtype=np.float32)
    players[:, 0] = [10.0, 30.0]  # the taker (slot 0), standing at the spot
    players[:, 1] = [-20.0, 0.0]
    players[:, 2] = [30.0, -20.0]
    players[:, 3] = [-30.0, 10.0]
    teams = np.array([taker_team, H if taker_team == A else A, A, A], dtype=np.int8)
    if taker_team == A:
        teams = np.array([A, H, H, H], dtype=np.int8)
    t = np.arange(N)
    ball = np.full((N, 2), np.nan, dtype=np.float32)
    ball[:400] = np.stack([np.linspace(-10, 10, 400), np.zeros(400)], axis=1)  # play before
    flight = (t[ball_from:] - KICK) / FPS
    ball[ball_from:] = np.stack([10.0 + speed * flight, np.full(len(flight), 30.0)], axis=1)
    tracking = Tracking(meta=meta, player_xy=players, slot_team=teams, ball_xy=ball)
    return MatchBundle(tracking=tracking.validate())


def _labels(restart: int = 560) -> np.ndarray:
    lab = np.full(N, A, dtype=np.int8)
    lab[400:restart] = D  # a stoppage decoded with a late restart edge
    lab[restart:] = H
    return lab


def test_a_late_edge_moves_back_to_the_traced_kick():
    out = restart_backtrack(_labels(), _bundle())
    edge = int(np.flatnonzero(out[400:] != D)[0]) + 400
    assert abs(edge - KICK) <= 10  # within 0.4 s of the kick (reach 1.5 m at 10 m/s)
    assert (out[edge:] == H).all() and (out[400:edge] == D).all()
    assert (out[:400] == A).all()


def test_an_early_or_on_time_edge_is_left_alone():
    lab = _labels(restart=490)
    assert np.array_equal(restart_backtrack(lab, _bundle()), lab)


def test_a_slow_ball_or_a_kicker_of_the_other_team_changes_nothing():
    lab = _labels()
    assert np.array_equal(restart_backtrack(lab, _bundle(speed=1.0)), lab)
    assert np.array_equal(restart_backtrack(lab, _bundle(taker_team=A)), lab)


def test_a_sighting_outside_the_search_window_changes_nothing():
    lab = _labels()
    assert np.array_equal(restart_backtrack(lab, _bundle(ball_from=700)), lab)


def test_the_stoppage_keeps_its_minimum_length():
    lab = np.full(N, A, dtype=np.int8)
    lab[495:560] = D  # the decoded stoppage starts just before the kick
    out = restart_backtrack(lab, _bundle())
    assert (out[495 : 495 + FPS] == D).all()  # keep_dead_s 1.0


def test_trace_kick_needs_a_seen_leaving_velocity():
    b = _bundle()
    ball = b.tracking.ball_xy.astype(np.float64)
    players = b.tracking.player_xy[:, [0]].astype(np.float64)
    assert (
        trace_kick(ball, players, 550, FPS, speed_mps=3.0, max_flight_s=2.5, reach_m=1.5)
        is not None
    )
    ball[552] = np.nan
    assert trace_kick(ball, players, 550, FPS, speed_mps=3.0, max_flight_s=2.5, reach_m=1.5) is None


def test_registered():
    assert POSTPROCESS.get("restart_backtrack").fn is restart_backtrack
