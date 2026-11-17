"""control_possession (#58) on hand-made tracking: passes hold, a held control switches, a touch
does not, no opinion before the first control, periods are independent."""

from __future__ import annotations

import numpy as np

from aaaag.contracts import AWAY, HOME, MatchBundle, MatchMeta, Period, Tracking
from aaaag.evidence.possession.control import control_possession, last_controller

FPS = 25
# two home players on the left, two away players on the right, all far apart
PLAYERS = np.array([[-30.0, 0.0], [-10.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
TEAMS = np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8)
MIDWAY = [-20.0, 10.0]  # far from everyone: a ball in flight


def _bundle(ball: np.ndarray, periods: tuple[Period, ...] | None = None) -> MatchBundle:
    T = len(ball)
    xy = np.repeat(PLAYERS[None], T, axis=0).astype(np.float32)
    meta = MatchMeta("t", "test", T, FPS, 105.0, 68.0, periods or (Period(1, 0, T),))
    tracking = Tracking(meta, xy, TEAMS, ball.astype(np.float32)).validate()
    return MatchBundle(tracking).validate()


def _team(ball: np.ndarray, **kw) -> np.ndarray:
    """-1 for no opinion, else the one-hot team."""
    ev = control_possession(_bundle(ball), **kw).validate(len(ball))
    p = ev.proba
    out = np.where(np.isnan(p[:, 0]), -1, np.where(p[:, 0] > 0.5, HOME, AWAY))
    known = ~np.isnan(p[:, 0])
    assert (p[known, 2] == 0).all()
    return out


def test_a_pass_between_teammates_holds_through_the_flight_and_a_gap():
    ball = np.full((100, 2), np.nan)
    ball[0:10] = PLAYERS[0] + [0.3, 0.0]  # home player 0 has it
    ball[10:30] = MIDWAY  # in flight
    ball[30:50] = np.nan  # unseen
    ball[50:60] = PLAYERS[1] + [0.0, 0.5]  # home player 1 receives
    ball[60:] = MIDWAY
    assert (_team(ball) == HOME).all()


def test_an_opponent_holding_three_frames_switches_from_the_first_frame():
    ball = np.full((60, 2), np.nan)
    ball[0:10] = PLAYERS[1] + [0.3, 0.0]  # home
    ball[10:20] = MIDWAY
    ball[20:23] = PLAYERS[2] + [-0.3, 0.0]  # away controls exactly 3 frames
    ball[23:] = MIDWAY
    team = _team(ball)
    assert (team[:20] == HOME).all()
    assert (team[20:] == AWAY).all()  # dated from the run's first frame


def test_a_one_or_two_frame_touch_by_the_opponent_changes_nothing():
    for touch in (1, 2):
        ball = np.full((60, 2), np.nan)
        ball[0:10] = PLAYERS[0] + [0.3, 0.0]
        ball[10:20] = MIDWAY
        ball[20 : 20 + touch] = PLAYERS[3] + [0.2, 0.0]  # away brushes it
        ball[20 + touch :] = MIDWAY
        assert (_team(ball) == HOME).all(), touch


def test_no_opinion_before_the_first_control_and_no_control_no_opinion():
    ball = np.full((50, 2), np.nan)
    ball[0:20] = MIDWAY
    ball[20] = PLAYERS[2]  # a single frame: not a control run
    ball[30:40] = PLAYERS[2] + [0.1, 0.1]
    team = _team(ball)
    assert (team[:30] == -1).all()
    assert (team[30:] == AWAY).all()
    assert (_team(np.full((20, 2), np.nan)) == -1).all()


def test_reach_bounds_control():
    ball = np.full((40, 2), np.nan)
    ball[0:10] = PLAYERS[0] + [0.5, 0.0]  # home within 1 m
    ball[10:40] = PLAYERS[2] + [1.5, 0.0]  # away 1.5 m away: no control at 1 m, control at 2 m
    assert (_team(ball) == HOME).all()
    team = _team(ball, reach_m=2.0)
    assert (team[:10] == HOME).all() and (team[10:] == AWAY).all()


def test_periods_start_without_an_opinion():
    ball = np.full((80, 2), np.nan)
    ball[0:10] = PLAYERS[0] + [0.2, 0.0]  # home in period 1
    ball[10:40] = MIDWAY
    ball[40:50] = MIDWAY  # period 2 starts with nobody in control
    ball[50:60] = PLAYERS[3] + [0.2, 0.0]
    ball[60:] = MIDWAY
    periods = (Period(1, 0, 40), Period(2, 40, 80))
    p = control_possession(_bundle(ball, periods)).validate(80).proba[:, 0]
    assert (p[:40] == 1.0).all()
    assert np.isnan(p[40:50]).all()
    assert (p[50:] == 0.0).all()


def test_last_controller_kernel():
    c = np.array([-1, 0, 0, 0, -1, 1, 1, -1, 1, 1, 1, 1, -1, 0], dtype=np.int8)
    np.testing.assert_array_equal(
        last_controller(c, 3), [-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        last_controller(c, 1), [-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    )
    assert (last_controller(np.array([], dtype=np.int8), 3) == -1).all()
