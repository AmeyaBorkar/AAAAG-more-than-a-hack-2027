"""The token builder (src/aaaag/ml/windows.py), the symmetries and the window plans.

Numpy only: these run in the repository's normal environment. Hand-made tracking where the right
answer is obvious, plus consistency checks that tie the builder to `symmetric` on a synthetic match.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.ml.geometry import attack_sign, swap_labels, symmetric
from aaaag.ml.spec import N_FEATURES, Col, TokenSpec
from aaaag.ml.windows import (
    absence_features,
    build_match_tokens,
    plan_chunks,
    sample_starts,
    smoothed_velocity,
)

FPS = 25
HALF_L, HALF_W = 52.5, 34.0


def _tracking(
    home_x, away_x, ball, *, n=60, periods=None, home_y=None, away_y=None, speed_x=0.0
) -> Tracking:
    """Static players (home on the left, away on the right by default), one ball track."""
    xs = np.array([*home_x, *away_x], dtype=np.float32)
    ys = np.array([*(home_y or [0.0] * len(home_x)), *(away_y or [0.0] * len(away_x))], np.float32)
    xy = np.stack([np.repeat(xs[None], n, 0), np.repeat(ys[None], n, 0)], axis=-1)
    xy = xy + np.array([speed_x, 0.0], dtype=np.float32) * (np.arange(n)[:, None, None] / FPS)
    teams = np.array([HOME] * len(home_x) + [AWAY] * len(away_x), dtype=np.int8)
    meta = MatchMeta("t", "test", n, FPS, 105.0, 68.0, periods or (Period(1, 0, n),))
    return Tracking(meta, xy.astype(np.float32), teams, np.asarray(ball, dtype=np.float32))


def _ball(n, x=0.0, y=0.0) -> np.ndarray:
    return np.tile(np.array([x, y], dtype=np.float32), (n, 1))


def test_tokens_are_the_nearest_players_per_team_sorted_by_distance():
    tracking = _tracking([-5, -10, -30], [3, 8, 20], _ball(60))
    spec = TokenSpec(tokens_per_team=2)
    tokens = build_match_tokens(tracking, spec)
    assert tokens.shape == (60, 5, N_FEATURES) and tokens.dtype == np.float32
    home = tokens[10, :2]
    assert np.allclose(home[:, Col.X] * HALF_L, [-5, -10], atol=1e-4)  # home defends -x: no flip
    assert np.allclose(home[:, Col.DIST] * spec.ref_scale_m, [5, 10], atol=1e-4)
    away = tokens[10, 2:4]
    assert np.allclose(away[:, Col.X] * HALF_L, [3, 8], atol=1e-4)
    assert (tokens[10, :4, Col.VALID] == 1).all() and (tokens[10, 2:4, Col.IS_AWAY] == 1).all()
    assert (tokens[10, :2, Col.IS_AWAY] == 0).all() and tokens[10, 4, Col.IS_BALL] == 1


def test_without_the_ball_the_team_centroid_is_the_reference():
    tracking = _tracking([-5, -10, -30], [3, 8, 20], np.full((60, 2), np.nan, np.float32))
    spec = TokenSpec(tokens_per_team=2)
    tokens = build_match_tokens(tracking, spec)
    home = tokens[10, :2]
    # the home centroid is -15: nearest -10 (5 m), then -5 (10 m)
    assert np.allclose(home[:, Col.X] * HALF_L, [-10, -5], atol=1e-4)
    assert np.allclose(home[:, Col.DIST] * spec.ref_scale_m, [5, 10], atol=1e-4)
    assert (home[:, Col.REF_BALL] == 0).all()
    ball = tokens[10, 4]
    assert ball[Col.VALID] == 0 and ball[Col.X] == 0 and ball[Col.GAP_BEFORE] == 1  # never seen


def test_short_teams_are_padded_with_invalid_zero_tokens():
    tracking = _tracking([-5, -10], [3], _ball(60))
    tokens = build_match_tokens(tracking, TokenSpec(tokens_per_team=4))
    assert tokens[5, :4, Col.VALID].tolist() == [1, 1, 0, 0]
    assert tokens[5, 4:8, Col.VALID].tolist() == [1, 0, 0, 0]
    pad = tokens[5, 2]
    assert not pad[[c for c in range(N_FEATURES) if c != Col.IS_AWAY]].any()
    assert not np.isnan(tokens).any()


def test_unobserved_slots_never_become_tokens():
    tracking = _tracking([-5, -10, -30], [3, 8, 20], _ball(60))
    xy = tracking.player_xy.copy()
    xy[20:30, 0] = np.nan  # the nearest home slot drops out for ten frames
    tracking = replace(tracking, player_xy=xy)
    tokens = build_match_tokens(tracking, TokenSpec(tokens_per_team=3))
    assert np.allclose(tokens[25, :3, Col.X] * HALF_L, [-10, -30, 0], atol=1e-4)
    assert tokens[25, :3, Col.VALID].tolist() == [1, 1, 0]
    assert tokens[40, 0, Col.X] * HALF_L == pytest.approx(-5, abs=1e-4)


def test_velocity_of_steady_motion_is_exact_everywhere():
    tracking = _tracking([-5, -10], [3, 8], _ball(60), speed_x=2.0)  # everyone runs at 2 m/s
    spec = TokenSpec(tokens_per_team=2)
    tokens = build_match_tokens(tracking, spec)
    vx = tokens[:, :4, Col.VX] * spec.velocity_scale
    assert np.allclose(vx, 2.0, atol=1e-4)  # centred, forward and backward ends all agree
    assert np.allclose(tokens[:, :4, Col.SPEED] * spec.velocity_scale, 2.0, atol=1e-4)
    assert np.allclose(tokens[:, :4, Col.VY], 0.0, atol=1e-6)


def test_a_slot_swap_jump_is_not_a_velocity():
    tracking = _tracking([-5, -10], [3, 8], _ball(60))
    xy = tracking.player_xy.copy()
    xy[30:, 0, 0] += 30.0  # a jump of 30 m: 150 m/s over 0.2 s
    tracking = replace(tracking, player_xy=xy)
    spec = TokenSpec(tokens_per_team=2)
    tokens = build_match_tokens(tracking, spec)
    speed = tokens[:, :2, Col.SPEED] * spec.velocity_scale
    assert speed.max() == 0.0  # frames whose window straddles the jump read as none, not 150 m/s


def test_smoothed_velocity_uses_one_side_when_the_other_is_missing():
    xy = np.zeros((20, 2), dtype=np.float32)
    xy[:, 0] = np.arange(20) * 0.2  # 5 m/s at 25 fps
    xy[12:, :] = np.nan  # the track ends at frame 11
    v = smoothed_velocity(xy, half=3, fps=FPS)
    assert np.allclose(v[:9, 0], 5.0) and np.allclose(v[9:12, 0], 5.0)  # backward at the end
    assert np.isnan(v[12:]).all()


def test_absence_features_count_the_run_both_ways():
    seen = np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool)
    before, after = absence_features(seen, FPS, cap_s=30.0)
    scale = np.log1p(30.0)
    frames_before = np.array([0, 0, 1, 2, 3, 0, 1])
    frames_after = np.array([0, 0, 3, 2, 1, 0, 30 * FPS])  # never seen again: the cap
    assert np.allclose(before, np.log1p(frames_before / FPS) / scale, atol=1e-6)
    assert np.allclose(after, np.log1p(frames_after / FPS) / scale, atol=1e-6)
    never_before, never_after = absence_features(np.zeros(10, dtype=bool), FPS, cap_s=30.0)
    assert (never_before == 1.0).all() and (never_after == 1.0).all()


def test_each_period_is_oriented_on_its_own_and_gaps_are_empty():
    n = 100
    first, second = Period(1, 10, 40), Period(2, 60, 90)
    tracking = _tracking([-20, -30], [20, 30], _ball(n), n=n, periods=(first, second))
    xy = tracking.player_xy.copy()
    xy[60:] *= np.array([-1.0, 1.0], dtype=np.float32)  # teams swap ends in the second period
    ball = tracking.ball_xy.copy()
    ball[60:] = (-8.0, 0.0)
    tracking = replace(tracking, player_xy=xy, ball_xy=ball)
    assert attack_sign(tracking)[[20, 70]].tolist() == [1.0, -1.0]
    spec = TokenSpec(tokens_per_team=2)
    tokens = build_match_tokens(tracking, spec)
    for t in (20, 70):  # home keeps defending -x in the normalised frame, in both periods
        assert (tokens[t, :2, Col.X] < 0).all() and (tokens[t, 2:4, Col.X] > 0).all()
    assert tokens[70, 4, Col.X] * HALF_L == pytest.approx(8.0, abs=1e-4)  # the ball is rotated too
    assert not tokens[:10].any() and not tokens[40:60].any() and not tokens[90:].any()


def test_a_match_builds_the_same_tokens_twice(small_bundle):
    a = build_match_tokens(small_bundle.tracking)
    b = build_match_tokens(small_bundle.tracking)
    assert a.shape == (small_bundle.meta.n_frames, 23, N_FEATURES) and not np.isnan(a).any()
    assert np.array_equal(a, b)
    valid = a[..., Col.VALID] > 0.5
    assert valid[small_bundle.meta.played_mask(), :22].sum(axis=1).min() >= 1
    assert not a[~small_bundle.meta.played_mask()].any()


def test_a_team_swap_of_the_tracking_is_the_team_swap_of_the_tokens(small_bundle):
    tracking = small_bundle.tracking
    spec = TokenSpec()
    swapped = replace(
        tracking, slot_team=(1 - tracking.slot_team).astype(np.int8), _fingerprint=None
    )
    expected = symmetric(build_match_tokens(tracking, spec), spec, swap=True)
    played = small_bundle.meta.played_mask()  # outside the periods the builder writes zero rows
    assert np.allclose(build_match_tokens(swapped, spec)[played], expected[played], atol=1e-6)


def test_a_y_mirror_of_the_tracking_is_the_y_flip_of_the_tokens(small_bundle):
    tracking = small_bundle.tracking
    spec = TokenSpec()
    mirror = np.array([1.0, -1.0], dtype=np.float32)
    flipped = replace(
        tracking,
        player_xy=tracking.player_xy * mirror,
        ball_xy=tracking.ball_xy * mirror,
        _fingerprint=None,
    )
    expected = symmetric(build_match_tokens(tracking, spec), spec, flip_y=True)
    assert np.allclose(build_match_tokens(flipped, spec), expected, atol=1e-6)


def test_symmetries_are_involutions_and_leave_the_input_alone(small_bundle):
    spec = TokenSpec()
    tokens = build_match_tokens(small_bundle.tracking, spec)[1000:1100]
    before = tokens.copy()
    for flip_y, swap in ((True, False), (False, True), (True, True)):
        once = symmetric(tokens, spec, flip_y=flip_y, swap=swap)
        assert np.array_equal(symmetric(once, spec, flip_y=flip_y, swap=swap), tokens)
    assert np.array_equal(tokens, before)
    labels = np.array([0, 1, 2, 3, 1, 0], dtype=np.int8)
    assert swap_labels(labels).tolist() == [1, 0, 2, 3, 0, 1]


def test_chunks_tile_the_played_frames_with_context_inside_the_periods():
    periods = (Period(1, 10, 100), Period(2, 130, 150))
    plan = plan_chunks(periods, chunk=40, margin=7)
    kept = np.concatenate([np.arange(c.keep_lo, c.keep_hi) for c in plan])
    played = np.concatenate([np.arange(p.start, p.stop) for p in periods])
    assert np.array_equal(kept, played)  # every played frame once, in order
    for c in plan:
        period = next(p for p in periods if p.start <= c.keep_lo < p.stop)
        assert period.start <= c.lo <= c.keep_lo < c.keep_hi <= c.hi <= period.stop
        assert c.keep_lo - c.lo == min(7, c.keep_lo - period.start)
        assert c.hi - c.keep_hi == min(7, period.stop - c.keep_hi)
    with pytest.raises(ValueError):
        plan_chunks(periods, chunk=0, margin=1)


def test_window_starts_stay_inside_one_period_and_are_seeded():
    periods = (Period(1, 10, 500), Period(2, 600, 700))
    starts = sample_starts(periods, 80, 500, np.random.default_rng(3))
    assert np.array_equal(starts, sample_starts(periods, 80, 500, np.random.default_rng(3)))
    inside = [any(p.start <= s and s + 80 <= p.stop for p in periods) for s in starts]
    assert all(inside) and (starts >= 600).any() and (starts < 500).any()
    with pytest.raises(ValueError):
        sample_starts((Period(1, 0, 50),), 80, 3, np.random.default_rng(0))
