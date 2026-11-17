"""E56 restart_blank: the ball hidden around true restarts in training. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, Period, X
from aaaag.ml.data import DegradeMix, TrainMatch, build_entry, restart_blank_mask
from aaaag.ml.spec import TokenSpec

FPS = 10
PERIODS = (Period(1, 0, 200),)


def _labels() -> np.ndarray:
    lab = np.full(200, H, dtype=np.int8)
    lab[50:80] = D  # a 3 s stoppage, restart at 80
    lab[80:120] = A
    lab[120:125] = D  # 0.5 s: too short to count
    return lab


def test_every_restart_is_blanked_at_p_one_and_none_at_p_zero():
    rng = np.random.default_rng(0)
    mask = restart_blank_mask(_labels(), PERIODS, FPS, rng, 1.0, 1.0, 3.0)
    hidden = np.flatnonzero(mask)
    assert len(hidden) and hidden.min() >= 70 and hidden.max() < 110
    assert not mask[120:130].any()  # the short dead run is not a restart
    none = restart_blank_mask(_labels(), PERIODS, FPS, np.random.default_rng(0), 0.0, 1.0, 3.0)
    assert not none.any()


def test_a_dead_run_at_the_period_end_is_not_a_restart():
    lab = np.full(200, H, dtype=np.int8)
    lab[170:200] = D
    assert not restart_blank_mask(lab, PERIODS, FPS, np.random.default_rng(0), 1.0, 1.0, 3.0).any()
    lab[190:200] = X
    assert not restart_blank_mask(lab, PERIODS, FPS, np.random.default_rng(0), 1.0, 1.0, 3.0).any()


def test_build_entry_hides_the_coupled_ball_only_when_asked(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    mix = DegradeMix()
    plain = build_entry(match, np.random.default_rng(3), TokenSpec(), mix)
    again = build_entry(match, np.random.default_rng(3), TokenSpec(), mix, restart_blank=None)
    assert np.array_equal(plain.coupled, again.coupled, equal_nan=True)
    blank = build_entry(
        match, np.random.default_rng(3), TokenSpec(), mix, restart_blank=(1.0, 1.0, 3.0)
    )
    assert blank.info["ball_hidden_coupled"] > plain.info["ball_hidden_coupled"]


def test_the_config_round_trip_drops_the_default():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, replace_train

    run = RunConfig.from_dict({})
    assert "restart_blank_p" not in run.to_dict()["train"]
    on = replace_train(run, restart_blank_p=0.5)
    assert RunConfig.from_dict(on.to_dict()).train.restart_blank_p == 0.5
