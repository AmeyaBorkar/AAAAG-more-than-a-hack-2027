"""Realism classifier pieces on synthetic tracking: window features and the cross-validated AUC."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.degrade.realism import GROUP_A, GROUP_B, WINDOW_S, cv_auc, matrix, window_features

FPS = 25


def _tracking(minutes: float = 12.0, seed: int = 0) -> Tracking:
    rng = np.random.default_rng(seed)
    T = int(minutes * 60 * FPS)
    walk = np.cumsum(rng.normal(0, 0.05, (T, 22, 2)), axis=0)
    xy = np.clip(rng.uniform(-30, 30, (1, 22, 2)) + walk, -50, 30).astype(np.float32)
    ball = np.clip(np.cumsum(rng.normal(0, 0.2, (T, 2)), axis=0), -50, 30).astype(np.float32)
    half = T // 2
    meta = MatchMeta("syn", "test", T, FPS, 105.0, 68.0, (Period(1, 0, half), Period(2, half, T)))
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    return Tracking(meta, xy, team, ball).validate()


def test_window_features_shapes_and_known_values():
    clean = _tracking()
    ball = clean.ball_xy.copy()
    size = int(WINDOW_S * FPS)
    ball[size + 100 : size + 350] = np.nan  # one 10 s gap inside the second window
    xy = clean.player_xy.copy()
    xy[: 2 * size, 0] = np.nan  # a home player missing for the first two windows
    tracking = Tracking(clean.meta, xy, clean.slot_team, ball).validate()
    features, blocks = window_features(tracking)
    n = sum(p.length // size for p in tracking.meta.periods)
    assert set(features) == set(GROUP_A + GROUP_B)
    assert all(v.shape == (n,) for v in features.values()) and blocks.shape == (n,)
    assert features["ball_missing"][0] == 0
    assert features["ball_missing"][1] == pytest.approx(250 / size)
    assert features["ball_gap_starts"][1] == 1 and features["ball_longest_gap_s"][1] == 10.0
    assert features["objects_min"][0] == 10 and features["objects_max"][0] == 11
    assert features["fragment_starts"][2] == 1  # the player comes back in the third window
    assert len(np.unique(blocks)) == 4  # two 5-minute blocks in each 6-minute half
    assert np.isfinite(matrix(features, GROUP_A)).all()


def test_cv_auc_separates_shifted_data_and_not_identical_data():
    rng = np.random.default_rng(0)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    y = (groups >= 20).astype(int)
    names = ("signal", "noise")
    shifted = np.stack([rng.normal(2.0 * y, 1.0), rng.normal(0, 1, n)], axis=1)
    result = cv_auc(shifted, y, groups, names, n_boot=200)
    assert result.auc > 0.85 and result.interval[0] < result.auc < result.interval[1]
    assert result.importance["signal"] > 0.2 > abs(result.importance["noise"])
    same = rng.normal(0, 1, (n, 2))
    assert abs(cv_auc(same, y, groups, names, n_boot=200).auc - 0.5) < 0.12
    boosted = cv_auc(shifted, y, groups, names, kind="boosting", n_boot=50)
    assert boosted.auc > 0.8
