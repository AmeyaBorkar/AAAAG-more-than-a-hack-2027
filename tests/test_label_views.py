"""The "interior" label view for event-timed extra matches (E80, #86). Synthetic arrays only."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchMeta, Period, X
from aaaag.ml.label_views import INTERIOR_S, change_mask, interior_view
from aaaag.registry import SOURCES

FPS = 25


def _meta(n: int, periods: tuple[Period, ...]) -> MatchMeta:
    return MatchMeta("m", "test", n, FPS, 105.0, 68.0, periods)


def _one_period(n: int = 1000) -> MatchMeta:
    return _meta(n, (Period(1, 0, n),))


def test_interior_masks_exactly_the_window_around_every_change_and_nothing_else():
    meta = _one_period()
    labels = np.full(1000, H, dtype=np.int8)
    labels[300:700] = A  # changes between 299|300 and 699|700
    out = interior_view(labels, meta, 2.0)  # 50 frames on each side
    expected = labels.copy()
    expected[250:350] = X
    expected[650:750] = X
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.int8


def test_a_possession_shorter_than_two_windows_vanishes():
    meta = _one_period()
    labels = np.full(1000, H, dtype=np.int8)
    labels[400:480] = A  # 3.2 s, under 2 x 2.0 s
    out = interior_view(labels, meta, 2.0)
    assert not (out == A).any()
    assert (out[:350] == H).all() and (out[530:] == H).all()
    assert (out[350:530] == X).all()


def test_dead_frames_become_x_and_dead_edges_are_changes():
    meta = _one_period()
    labels = np.full(1000, H, dtype=np.int8)
    labels[400:600] = D
    out = interior_view(labels, meta, 1.0)  # 25 frames
    assert not (out == D).any()
    assert (out[375:625] == X).all()
    assert (out[:375] == H).all() and (out[625:] == H).all()


def test_period_edges_are_not_changes_and_periods_do_not_leak():
    meta = _meta(1100, (Period(1, 0, 500), Period(2, 600, 1100)))
    labels = np.full(1100, X, dtype=np.int8)
    labels[0:500] = H
    labels[600:1100] = A
    labels[480:500] = A  # a change 20 frames before the end of the first half
    out = interior_view(labels, meta, 2.0)
    assert (out[:430] == H).all()  # the first frames of the half are kept: no change at its edge
    assert (out[430:500] == X).all()
    assert (out[600:1100] == A).all()  # the second half is untouched by the first half's change
    assert (out[500:600] == X).all()


def test_window_zero_is_the_team_view():
    meta = _one_period()
    rng = np.random.default_rng(0)
    labels = rng.choice(np.array([H, A, D], dtype=np.int8), size=1000)
    out = interior_view(labels, meta, 0.0)
    expected = labels.copy()
    expected[expected == D] = X
    np.testing.assert_array_equal(out, expected)
    assert not change_mask(labels, meta, 0).any()


def test_frames_without_a_seen_ball_become_x():
    meta = _one_period()
    labels = np.full(1000, H, dtype=np.int8)
    seen = np.ones(1000, dtype=bool)
    seen[100:200] = False
    out = interior_view(labels, meta, 2.0, seen=seen)
    assert (out[100:200] == X).all()
    assert (out[:100] == H).all() and (out[200:] == H).all()
    with pytest.raises(ValueError):
        interior_view(labels, meta, 2.0, seen=seen[:10])


def test_the_input_is_not_modified_and_a_negative_window_is_refused():
    meta = _one_period()
    labels = np.full(1000, H, dtype=np.int8)
    labels[500:] = D
    before = labels.copy()
    interior_view(labels, meta, 1.0)
    np.testing.assert_array_equal(labels, before)
    with pytest.raises(ValueError):
        interior_view(labels, meta, -1.0)


def test_extra_label_view_interior_uses_the_constant_and_the_bundles_own_ball():
    from dataclasses import replace

    from aaaag.ml.data import EXTRA_LABEL_MODES, extra_label_view

    assert "interior" in EXTRA_LABEL_MODES
    bundle = SOURCES.get("synthetic").fn(match_id="1")
    period = bundle.meta.periods[0]
    fps = bundle.meta.fps
    labels = bundle.labels.copy()
    labels[period.start : period.stop] = H
    cut = period.start + 30 * fps
    labels[cut : period.stop] = A
    ball = bundle.tracking.ball_xy.copy()
    ball[period.start : period.stop] = 0.0
    hidden = slice(period.start + 5 * fps, period.start + 6 * fps)
    ball[hidden] = np.nan
    bundle = replace(bundle, tracking=replace(bundle.tracking, ball_xy=ball, _fingerprint=None))
    out = extra_label_view(labels, bundle, "interior")
    half = int(round(INTERIOR_S * fps))
    assert (out[cut - half : cut + half] == X).all()
    assert out[cut - half - 1] == H and out[cut + half] == A
    assert (out[hidden] == X).all()
    assert out[period.start] == H
    # the existing modes are untouched
    assert extra_label_view(labels, bundle, "all") is labels
    np.testing.assert_array_equal(
        extra_label_view(labels, bundle, "team"), np.where(labels == D, X, labels)
    )
