"""E71 dead_visible: the uncarried dead ball shown through true stoppages. Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from aaaag.ml.data import DegradeMix, TrainMatch, build_entry
from aaaag.ml.spec import TokenSpec
from aaaag.paths import CONFIGS_DIR


def _same_draw(got, ref) -> None:
    assert got.info == ref.info
    assert got.coupled.tobytes() == ref.coupled.tobytes()
    assert got.decoupled.tobytes() == ref.decoupled.tobytes()
    assert got.visible_coupled.tobytes() == ref.visible_coupled.tobytes()
    assert got.visible_decoupled.tobytes() == ref.visible_decoupled.tobytes()


def test_off_is_byte_identical_to_a_draw_without_the_argument(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    mix = DegradeMix()
    plain = build_entry(match, np.random.default_rng(3), TokenSpec(), mix)
    assert "dead_visible_stoppages" not in plain.info
    for off in (None, 0, 0.0):
        again = build_entry(match, np.random.default_rng(3), TokenSpec(), mix, dead_visible=off)
        _same_draw(again, plain)


def test_on_records_how_many_stoppages_changed(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    on = build_entry(match, np.random.default_rng(3), TokenSpec(), DegradeMix(), dead_visible=1.0)
    assert on.info["dead_visible_stoppages"] >= 0


def test_the_config_round_trip_drops_the_default():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, replace_train

    run = RunConfig.from_dict({})
    assert "dead_visible_p" not in run.to_dict()["train"]
    on = replace_train(run, dead_visible_p=0.5)
    assert RunConfig.from_dict(on.to_dict()).train.dead_visible_p == 0.5
    with pytest.raises(ValueError, match="dead_visible_p"):
        replace_train(run, dead_visible_p=1.5)


def test_e71_is_e56a_plus_dead_visible():
    e56 = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e56a.yaml").read_text(encoding="utf-8")
    )
    e71 = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e71.yaml").read_text(encoding="utf-8")
    )
    assert e71["train"].pop("dead_visible_p") == 0.5
    assert e71 == e56
