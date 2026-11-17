"""`train.target_min_s` (E40): training targets with short possessions absorbed. Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from aaaag.contracts import A, H  # noqa: E402
from aaaag.ml.data import TrainMatch  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import TrainConfig, coarsen_targets  # noqa: E402
from aaaag.postprocess.min_spell import absorb_short  # noqa: E402
from aaaag.registry import SOURCES  # noqa: E402


def _match() -> TrainMatch:
    bundle = SOURCES.get("synthetic").fn(match_id="1")
    labels = bundle.labels.copy()
    start = bundle.meta.periods[0].start
    fps = bundle.meta.fps
    labels[start : start + 12 * fps] = H
    labels[start + 5 * fps : start + 5 * fps + fps // 2] = A  # a 0.5 s away touch inside home play
    bundle.labels = labels
    return TrainMatch.from_bundle(bundle, TokenSpec())


def test_coarsen_targets_absorbs_short_possessions_and_keeps_the_rest():
    match = _match()
    start = match.bundle.meta.periods[0].start
    fps = match.bundle.meta.fps
    out = coarsen_targets(match, 1.0)
    assert np.all(out.labels[start : start + 12 * fps] == H)
    assert np.array_equal(out.labels, absorb_short(match.labels, match.bundle.meta, 1.0))
    assert np.array_equal(out.bundle.labels, out.labels)
    seen = ~np.isnan(match.bundle.tracking.ball_xy[:, 0])
    assert np.array_equal(out.target_ok, seen & np.isin(out.labels, (H, A)) & match.played)
    assert np.any(match.labels[start : start + 12 * fps] == A)  # the input is not modified


def test_target_min_s_validation_and_default_round_trip():
    assert TrainConfig().target_min_s == 0.0
    assert TrainConfig(target_min_s=2).target_min_s == 2
    with pytest.raises(ValueError):
        TrainConfig(target_min_s=-1.0)
    with pytest.raises(ValueError):
        TrainConfig(target_min_s=True)
