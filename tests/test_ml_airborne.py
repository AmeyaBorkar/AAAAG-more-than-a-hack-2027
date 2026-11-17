"""E119's airborne-ball projection and its training option (#86)."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.ml import airborne


def test_ground_ball_is_unchanged_and_airborne_ball_moves_away_from_camera():
    cam = np.array([0.0, -54.0, 18.0])
    ball = np.array([[10.0, 0.0], [10.0, 0.0], [np.nan, np.nan]], dtype=np.float32)
    z = np.array([0.0, 6.0, 3.0])
    out = airborne.project(ball, z, cam)
    assert np.allclose(out[0], [10.0, 0.0])
    assert np.allclose(out[1], [15.0, -54.0 + 54.0 * 1.5])
    assert np.isnan(out[2]).all()


def test_coin_off_returns_the_same_array_and_on_changes_it():
    ball = np.zeros((5, 2), dtype=np.float32)
    z = np.full(5, 5.0)
    same, applied = airborne.airborne_ball(ball, z, 68.0, np.random.default_rng(0), 0.0, 15, 25)
    assert same is ball and not applied
    moved, applied = airborne.airborne_ball(ball, z, 68.0, np.random.default_rng(0), 1.0, 15, 25)
    assert applied and not np.allclose(moved, ball)


def test_train_config_drops_the_option_when_off():
    pytest.importorskip("torch")  # train.py needs it; CI installs requirements-lock.txt only
    from aaaag.ml.train import TrainConfig

    assert TrainConfig().airborne_p == 0.0
    try:
        TrainConfig(airborne_p=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("airborne_p 1.5 must be refused")
