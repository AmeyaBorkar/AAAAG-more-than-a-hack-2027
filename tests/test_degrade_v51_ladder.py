"""degrade-v5.1 player ladder (#73): dropout share, noise sigma, determinism, the suites."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aaaag.degrade import degrade
from aaaag.degrade.ladder import LadderModel, add_noise, degrade_ladder, dropout_mask
from aaaag.experiments import Suite
from aaaag.paths import CONFIGS_DIR
from aaaag.sources import idsse

RUNGS = ("mild", "medium", "severe", "medium-nearball")


def _model(**kw) -> LadderModel:
    base = dict(
        name="t", dropout_share=0.1, dropout_mean_s=2.0, noise_sigma_m=0.25, noise_smooth_s=0.25
    )
    return LadderModel(**{**base, **kw})


@pytest.mark.parametrize("rung", RUNGS)
def test_ladder_configs_load_and_are_physics_priors(rung):
    model = LadderModel.load(f"v51-{rung}")
    assert model.name == f"v51-{rung}"
    assert model.dropout_mean_s == 2.0 and model.noise_smooth_s == 0.25
    expected = {"mild": (0.05, 0.10), "medium": (0.10, 0.25), "severe": (0.20, 0.50)}
    share, sigma = expected[rung.split("-")[0]]
    assert (model.dropout_share, model.noise_sigma_m) == (share, sigma)
    assert model.near_ball_factor == (2.0 if rung.endswith("nearball") else 1.0)


def test_noise_has_the_declared_sigma_and_keeps_nan():
    rng = np.random.default_rng(0)
    xy = np.zeros((20000, 4, 2))
    xy[100:200, 1] = np.nan
    observed = ~np.isnan(xy[..., 0])
    out = add_noise(xy, observed, 0.25, 0.25 * 25, rng)
    assert np.isnan(out[100:200, 1]).all()
    assert abs(np.nanstd(out) - 0.25) < 0.01
    step = np.nanstd(np.diff(out[:, 0, 0]))
    assert step < 0.25 / 3  # smoothed: frame-to-frame changes are much smaller than sigma


@pytest.mark.parametrize("share", [0.05, 0.10, 0.20])
def test_dropout_share_matches_the_rung(small_bundle, share):
    tracking = small_bundle.tracking
    observed = ~np.isnan(tracking.player_xy[..., 0])
    shares = [
        dropout_mask(tracking, observed, _model(dropout_share=share), np.random.default_rng(s))[
            observed
        ].mean()
        for s in range(5)
    ]
    assert abs(np.mean(shares) - share) < 0.35 * share


def test_near_ball_factor_drops_more_near_the_ball(small_bundle):
    tracking = small_bundle.tracking
    observed = ~np.isnan(tracking.player_xy[..., 0])
    flat = dropout_mask(tracking, observed, _model(), np.random.default_rng(1))[observed].mean()
    near = dropout_mask(tracking, observed, _model(near_ball_factor=2.0), np.random.default_rng(1))[
        observed
    ].mean()
    assert near > flat


def test_ladder_is_deterministic_and_leaves_ball_and_teams(small_bundle):
    tracking = small_bundle.tracking
    model = LadderModel.load("v51-medium")
    a = degrade_ladder(tracking, model, np.random.default_rng([0, 5]))
    b = degrade_ladder(tracking, model, np.random.default_rng([0, 5]))
    np.testing.assert_array_equal(a.player_xy, b.player_xy)
    np.testing.assert_array_equal(a.ball_xy, tracking.ball_xy)
    np.testing.assert_array_equal(a.slot_team, tracking.slot_team)
    assert a.player_xy.dtype == np.float32
    assert np.isnan(a.player_xy).sum() > np.isnan(tracking.player_xy).sum()


def test_degrade_with_a_ladder_tags_the_match_and_keeps_v5_ball(small_bundle):
    v5 = degrade(small_bundle, version="5", seed=0)
    rung = degrade(small_bundle, version="5", seed=0, ladder="v51-severe")
    assert rung.meta.match_id == f"{small_bundle.meta.match_id}~d5v51-severes0"
    assert rung.extras["degrade"]["ladder"] == "v51-severe"
    np.testing.assert_array_equal(rung.tracking.ball_xy, v5.tracking.ball_xy)
    np.testing.assert_array_equal(rung.labels, small_bundle.labels)
    again = degrade(small_bundle, version="5", seed=0, ladder="v51-severe")
    assert again.fingerprint() == rung.fingerprint() != v5.fingerprint()


def test_ladder_name_must_match_its_file(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(
        "name: other\ndropout_share: 0.1\ndropout_mean_s: 2\nnoise_sigma_m: 0.1\n"
        "noise_smooth_s: 0.25\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected"):
        LadderModel.load("v51-x", path)


@pytest.mark.parametrize("rung", RUNGS)
def test_v51_suites_cover_the_dev_matches_with_five_seeds(rung):
    suite = Suite.load(CONFIGS_DIR / "suites" / f"idsse-dev-v51-{rung}.yaml")
    pairs = {(e.params["match_id"], e.params["degrade"]["seed"]) for e in suite.matches}
    assert pairs == {(m, s) for m in idsse.DEV_MATCHES for s in range(5)}
    assert len(pairs) == len(suite.matches)
    for entry in suite.matches:
        assert entry.params["degrade"]["version"] == "5"
        assert entry.params["degrade"]["ladder"] == f"v51-{rung}"
        assert "edge_jitter_s" not in entry.params["degrade"]


def test_zero_share_and_zero_noise_change_nothing(small_bundle):
    tracking = small_bundle.tracking
    out = degrade_ladder(
        tracking, _model(dropout_share=0.0, noise_sigma_m=0.0), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(out.player_xy, tracking.player_xy)
    assert replace(out, _fingerprint=None).fingerprint() == tracking.fingerprint()
