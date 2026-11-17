"""The optional `preprocess:` list (E127): absent keeps the digest; ball_flight_gate edits the bundle."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aaaag import pipeline, synthetic
from aaaag.features import ball_flight
from aaaag.pipeline import PipelineConfig

RAW = {
    "name": "t",
    "description": "",
    "evidence": [{"name": "nearest_player", "params": {}}],
    "decoder": {"name": "majority_vote", "params": {"half_window": 10}},
}
GATE = {"name": "ball_flight_gate", "params": {"pad_s": 0.2}}


def _jumpy_bundle():
    b = synthetic.generate(seed=5)
    ball = b.tracking.ball_xy.copy()
    s = b.meta.periods[0].start
    ball[s + 200 : s + 260, 0] += 8.0  # a jump out and one back
    t = dataclasses.replace(b.tracking, ball_xy=ball, _fingerprint=None).validate()
    return dataclasses.replace(b, tracking=t)


def test_absent_preprocess_keeps_the_dict_and_digest():
    plain = PipelineConfig.from_dict(RAW)
    assert "preprocess" not in plain.to_dict()
    empty = PipelineConfig.from_dict({**RAW, "preprocess": []})
    assert empty.digest() == plain.digest()
    gated = PipelineConfig.from_dict({**RAW, "preprocess": [GATE]})
    assert gated.digest() != plain.digest()
    assert gated.to_dict()["preprocess"] == [GATE]


def test_unknown_preprocess_step_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({**RAW, "preprocess": [{"name": "nope"}]})


def test_gate_runs_before_everything():
    b = _jumpy_bundle()
    assert ball_flight.flight_mask(b.tracking, pad_s=0.2).any()
    gated_cfg = PipelineConfig.from_dict({**RAW, "preprocess": [GATE]})
    plain_cfg = PipelineConfig.from_dict(RAW)
    via_config = pipeline.run(gated_cfg, b, use_cache=False)
    by_hand = pipeline.run(plain_cfg, ball_flight.gated_bundle(b, pad_s=0.2), use_cache=False)
    assert np.array_equal(via_config.labels, by_hand.labels)
    assert np.array_equal(via_config.proba, by_hand.proba)
    assert pipeline.preprocess(plain_cfg, b) is b
