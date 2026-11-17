"""E59 stoppage_carry: the restart taker carries the ball through true stoppages. Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, Period, X
from aaaag.ml.data import (
    DegradeMix,
    TrainMatch,
    build_entry,
    restart_blank_mask,
    stoppage_carry_ball,
    true_stoppages,
)
from aaaag.ml.spec import TokenSpec

FPS = 10
PERIODS = (Period(1, 0, 200),)


def _labels() -> np.ndarray:
    lab = np.full(200, H, dtype=np.int8)
    lab[50:80] = D  # a 3 s stoppage, restart at 80
    lab[80:120] = A
    lab[120:125] = D  # 0.5 s: too short to count
    return lab


def _scene():
    t = np.arange(200, dtype=np.float32)
    players = np.full((200, 3, 2), np.nan, dtype=np.float32)
    players[:, 0] = np.stack([t * 0.1, np.full(200, 5.0)], axis=1)  # the taker, walking
    players[:, 1] = [30.0, 30.0]
    players[:, 2] = [-30.0, -30.0]
    clean = np.full((200, 2), [50.0, 0.0], dtype=np.float32)
    clean[80] = players[80, 0]  # at the restart the clean ball is at the taker's feet
    ball = np.full((200, 2), np.nan, dtype=np.float32)  # hidden, as degrade-v5 does
    return players, ball, clean


def test_true_stoppages_finds_the_restart_and_skips_short_and_final_runs():
    assert [(s, r) for _p, s, r in true_stoppages(_labels(), PERIODS, FPS)] == [(50, 80)]
    lab = np.full(200, H, dtype=np.int8)
    lab[170:200] = D
    assert true_stoppages(lab, PERIODS, FPS) == []
    lab[190:200] = X
    assert true_stoppages(lab, PERIODS, FPS) == []


def test_the_shared_helper_keeps_the_restart_blank_stream():
    mask = restart_blank_mask(_labels(), PERIODS, FPS, np.random.default_rng(0), 1.0, 1.0, 3.0)
    rng = np.random.default_rng(0)
    take, before, after = rng.random(3)
    lo, hi = 80 - int(round(before * 1.0 * FPS)), 80 + int(round(after * 3.0 * FPS))
    expected = np.zeros(200, dtype=bool)
    expected[lo:hi] = True
    assert take < 1.0 and np.array_equal(mask, expected)


def test_the_carried_ball_follows_the_taker_inside_the_stoppage_only():
    players, ball, clean = _scene()
    out, changed = stoppage_carry_ball(
        players, ball, clean, _labels(), PERIODS, FPS, np.random.default_rng(1), 1.0, 6.0
    )
    carried = np.flatnonzero(np.isfinite(out).all(axis=1))
    assert changed == len(carried) > 0
    assert carried.min() >= 50 and carried.max() == 79  # not before the stoppage, up to the restart
    gap = np.hypot(*(out[carried] - players[carried, 0]).T)
    assert np.allclose(gap, gap[0]) and 0.2 <= gap[0] <= 1.0  # one fixed offset of 0.2-1 m
    assert np.isnan(ball).all()  # the input is not modified


def test_p_zero_changes_nothing_and_draws_four_numbers_per_stoppage():
    players, ball, clean = _scene()
    rng = np.random.default_rng(5)
    out, changed = stoppage_carry_ball(players, ball, clean, _labels(), PERIODS, FPS, rng, 0.0, 6.0)
    assert changed == 0 and np.isnan(out).all()
    reference = np.random.default_rng(5)
    reference.random(4)
    assert rng.random() == reference.random()


def test_an_unseen_taker_leaves_the_ball_alone():
    players, ball, clean = _scene()
    players[:, 0] = np.nan
    players[80, 0] = clean[80]  # seen at the restart only
    out, changed = stoppage_carry_ball(
        players, ball, clean, _labels(), PERIODS, FPS, np.random.default_rng(1), 1.0, 6.0
    )
    assert changed == 0 and np.isnan(out).all()


def test_build_entry_carries_the_coupled_ball_only_when_asked(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    mix = DegradeMix()
    plain = build_entry(match, np.random.default_rng(3), TokenSpec(), mix)
    again = build_entry(match, np.random.default_rng(3), TokenSpec(), mix, stoppage_carry=None)
    assert np.array_equal(plain.coupled, again.coupled, equal_nan=True)
    assert "ball_carried_frames" not in plain.info
    carry = build_entry(
        match, np.random.default_rng(3), TokenSpec(), mix, stoppage_carry=(1.0, 6.0)
    )
    assert carry.info["ball_carried_frames"] >= 0


def test_the_config_round_trip_drops_the_default():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, replace_train

    run = RunConfig.from_dict({})
    assert "stoppage_carry_p" not in run.to_dict()["train"]
    on = replace_train(run, stoppage_carry_p=0.5)
    back = RunConfig.from_dict(on.to_dict()).train
    assert back.stoppage_carry_p == 0.5 and back.stoppage_carry_max_s == 6.0
    with pytest.raises(ValueError, match="stoppage_carry_p"):
        replace_train(run, stoppage_carry_p=1.5)
