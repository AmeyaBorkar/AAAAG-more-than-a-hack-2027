"""E66 tracker-error augmentations (track_stress.py): fragments, dropouts with duplicates, long
ball gaps. Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, A, D, H, MatchMeta, Period, Tracking
from aaaag.ml.data import DegradeMix, TrainMatch, build_entry
from aaaag.ml.spec import TokenSpec
from aaaag.ml.track_stress import (
    BLOCK_SWAP_S,
    LONG_GAP_S,
    block_swap,
    drop_and_duplicate,
    fragment_tracks,
    long_ball_gaps,
)

FPS = 25
N = 3000  # two minutes


def _tracking(n_per_team: int = 4) -> Tracking:
    rng = np.random.default_rng(0)
    slots = 2 * n_per_team
    walk = np.cumsum(rng.normal(0.0, 0.05, size=(N, slots, 2)), axis=0).astype(np.float32)
    walk[:, :, 0] += np.linspace(-30, 30, slots, dtype=np.float32)
    walk[1000:1100, 0] = np.nan  # one player unseen for 4 s
    meta = MatchMeta(
        match_id="synthetic-track-stress",
        source="synthetic",
        fps=FPS,
        n_frames=N,
        periods=(Period(1, 0, 1500), Period(2, 1500, N)),
        pitch_length=105.0,
        pitch_width=68.0,
    )
    return Tracking(
        meta=meta,
        player_xy=walk,
        slot_team=np.array([HOME] * n_per_team + [AWAY] * n_per_team, dtype=np.int8),
        ball_xy=np.zeros((N, 2), dtype=np.float32),
        player_speed=None,
    ).validate()


def _team_points(t: Tracking, team: int) -> set[tuple[int, float, float]]:
    xy = t.player_xy[:, t.slot_team == team]
    f, s = np.nonzero(np.isfinite(xy[..., 0]))
    return {(int(a), float(xy[a, b, 0]), float(xy[a, b, 1])) for a, b in zip(f, s, strict=True)}


def test_fragments_keep_each_team_and_only_drop_the_gaps():
    t = _tracking()
    out = fragment_tracks(t, np.random.default_rng(1))
    for team in (HOME, AWAY):
        before, after = _team_points(t, team), _team_points(out, team)
        assert after <= before  # nothing invented, nothing moved to the other team
        assert 0.95 < len(after) / len(before) < 1.0  # gaps of Exp(0.2 s) between fragments
    assert out.ball_xy is t.ball_xy and out.player_speed is None
    assert out.n_slots >= t.n_slots  # a fragment never shares a slot run with another


def test_fragments_never_overlap_inside_a_slot():
    out = fragment_tracks(_tracking(), np.random.default_rng(2), ghost_per_min=4.0)
    seen = np.isfinite(out.player_xy[..., 0])
    for slot in range(out.n_slots):
        runs = np.flatnonzero(np.diff(np.concatenate(([0], seen[:, slot].astype(np.int8), [0]))))
        starts, stops = runs[::2], runs[1::2]
        assert (starts[1:] > stops[:-1]).all()  # at least one unseen frame between pieces


def test_a_swap_probability_of_one_gives_every_fragment_to_the_other_team():
    t = _tracking()
    out = fragment_tracks(t, np.random.default_rng(3), swap_p=1.0)
    assert _team_points(out, HOME) <= _team_points(t, AWAY)
    assert _team_points(out, AWAY) <= _team_points(t, HOME)


def test_dropouts_with_duplicates_drop_and_copy_but_keep_the_shape():
    t = _tracking()
    out = drop_and_duplicate(t, np.random.default_rng(4))
    assert out.player_xy.shape == t.player_xy.shape
    assert np.array_equal(out.slot_team, t.slot_team)
    seen_before = np.isfinite(t.player_xy[..., 0]).mean()
    seen_after = np.isfinite(out.player_xy[..., 0]).mean()
    assert 0.80 < seen_after / seen_before < 0.97  # 10% dropped, some refilled by a copy
    again = drop_and_duplicate(t, np.random.default_rng(4))
    assert np.array_equal(out.player_xy, again.player_xy, equal_nan=True)


def test_long_gaps_start_in_alive_seen_play_and_stop_at_the_period_end():
    t = _tracking()
    labels = np.full(N, H, dtype=np.int8)
    labels[200:400] = D
    labels[1500:2000] = A
    ball = t.ball_xy.copy()
    ball[2500:2600] = np.nan
    rng = np.random.default_rng(5)
    hidden = long_ball_gaps(ball, labels, t.meta.periods, FPS, rng)
    span = int(round(LONG_GAP_S * FPS))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], hidden.astype(np.int8), [0]))))
    for a, z in zip(edges[::2], edges[1::2], strict=True):
        assert labels[a] in (H, A) and np.isfinite(ball[a]).all()
        assert not (a < 1500 < z)  # never across the period boundary
    assert hidden.sum() > 0 and all(
        z - a <= 4 * span for a, z in zip(edges[::2], edges[1::2], strict=True)
    )


def test_build_entry_with_zero_probabilities_is_the_previous_draw(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    mix = DegradeMix()
    plain = build_entry(match, np.random.default_rng(3), TokenSpec(), mix)
    zero = build_entry(
        match, np.random.default_rng(3), TokenSpec(), mix, track_stress=(0.0, 0.0, 0.0)
    )
    assert np.array_equal(plain.coupled, zero.coupled, equal_nan=True)
    assert np.array_equal(plain.decoupled, zero.decoupled, equal_nan=True)
    assert zero.info["track_stress"] == [] and "track_stress" not in plain.info
    on = build_entry(
        match, np.random.default_rng(3), TokenSpec(), mix, track_stress=(1.0, 1.0, 1.0)
    )
    assert on.info["track_stress"] == ["frag_swap", "drop_dup", "long_gaps"]
    assert on.coupled.shape == plain.coupled.shape  # tokens stay 11 per team


def test_the_config_round_trip_drops_the_defaults():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, replace_train

    run = RunConfig.from_dict({})
    train = run.to_dict()["train"]
    assert not {"frag_swap_p", "drop_dup_p", "long_gap_p"} & set(train)
    on = replace_train(run, frag_swap_p=0.5, long_gap_p=0.5)
    back = RunConfig.from_dict(on.to_dict()).train
    assert (back.frag_swap_p, back.drop_dup_p, back.long_gap_p) == (0.5, 0.0, 0.5)
    with pytest.raises(ValueError, match="drop_dup_p"):
        replace_train(run, drop_dup_p=-0.1)


def test_the_dead_ball_shows_for_the_last_3_s_and_rests_before_the_restart():
    from aaaag.ml.track_stress import dead_ball_visible

    fps = 25
    n = 1000
    periods = (Period(1, 0, n),)
    labels = np.full(n, H, dtype=np.int8)
    labels[300:500] = D  # an 8 s stoppage, restart at 500
    labels[500:] = A
    labels[700:750] = D  # 2 s: too short (< 4 s), left alone
    clean = np.zeros((n, 2), dtype=np.float32)
    clean[:300] = [10.0, 34.0]  # the ball went out here
    clean[500:] = [10.0, 30.0]  # the throw-in spot, 4 m away
    ball = clean.copy()
    ball[300:500] = np.nan  # degrade-v5 hides it through the stoppage
    out, changed = dead_ball_visible(ball, clean, labels, periods, fps)
    assert changed == 1
    assert np.isnan(out[300:425]).all() and np.isfinite(out[425:500]).all()  # last 3 s only
    assert np.allclose(out[475:500], [10.0, 30.0])  # rests 1 s on the spot
    step = np.hypot(*np.diff(out[425:500], axis=0).T) * fps
    assert step.max() <= 4.0 + 1e-4 and np.allclose(out[425], [10.0, 34.0])
    assert np.isnan(out[700:750]).sum() == 0 and np.array_equal(out[500:], ball[500:])
    rng = np.random.default_rng(0)
    none, zero = dead_ball_visible(ball, clean, labels, periods, fps, rng, p=0.0)
    assert zero == 0 and np.array_equal(none, ball, equal_nan=True)


def test_block_swap_exchanges_one_home_away_pair_for_whole_blocks():
    t = _tracking()
    out, n = block_swap(t, np.random.default_rng(3))
    block = int(BLOCK_SWAP_S * FPS)
    assert 0 < n <= -(-N // block)
    assert out.ball_xy is t.ball_xy and (out.slot_team == t.slot_team).all()
    changed = ~np.all((out.player_xy == t.player_xy) | np.isnan(t.player_xy), axis=2)  # (T, S)
    for start in range(0, N, block):
        cols = np.flatnonzero(changed[start : start + block].any(axis=0))
        if len(cols):
            assert len(cols) == 2 and t.slot_team[cols].tolist() == [HOME, AWAY]
            a, b = cols
            seg = slice(start, start + block)
            np.testing.assert_array_equal(out.player_xy[seg, a], t.player_xy[seg, b])
    # the same multiset of positions per frame: nothing invented or lost
    np.testing.assert_array_equal(
        np.sort(np.nan_to_num(out.player_xy[..., 0], nan=-999), axis=1),
        np.sort(np.nan_to_num(t.player_xy[..., 0], nan=-999), axis=1),
    )


def test_block_swap_is_seeded_and_p_zero_changes_nothing():
    t = _tracking()
    a, na = block_swap(t, np.random.default_rng(7))
    b, nb = block_swap(t, np.random.default_rng(7))
    assert na == nb and np.array_equal(a.player_xy, b.player_xy, equal_nan=True)
    c, nc = block_swap(t, np.random.default_rng(7), p=0.0)
    assert nc == 0 and np.array_equal(c.player_xy, t.player_xy, equal_nan=True)
