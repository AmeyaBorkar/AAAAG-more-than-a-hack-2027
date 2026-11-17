"""Training data for the learned team model: degradation draws, ball masks, variant sets, weights.

Numpy only, on synthetic matches; the exam guard needs no data.
"""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H
from aaaag.ml.augment import hidden_ball_mask, hide_ball, with_ball
from aaaag.ml.data import (
    SEED_LOW,
    DegradeMix,
    TrainMatch,
    build_entry,
    build_set,
    class_weights,
    load_train_matches,
)
from aaaag.ml.spec import Col, TokenSpec
from aaaag.sources import idsse
from aaaag.synthetic import SyntheticConfig, generate

SPEC = TokenSpec()
SMALL = SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)


@pytest.fixture(scope="module")
def match(small_bundle) -> TrainMatch:
    return TrainMatch.from_bundle(small_bundle, SPEC)


def test_hidden_ball_mask_is_seeded_and_stays_inside_the_periods(small_bundle):
    meta = small_bundle.meta
    mask = hidden_ball_mask(meta.n_frames, meta.periods, meta.fps, np.random.default_rng(1))
    again = hidden_ball_mask(meta.n_frames, meta.periods, meta.fps, np.random.default_rng(1))
    other = hidden_ball_mask(meta.n_frames, meta.periods, meta.fps, np.random.default_rng(2))
    assert np.array_equal(mask, again) and not np.array_equal(mask, other)
    assert not mask[~meta.played_mask()].any()


def test_hidden_ball_mask_hides_a_wide_range_of_shares():
    from aaaag.contracts import Period

    periods = (Period(1, 0, 60 * 25 * 90),)  # one long period: 45 blocks
    shares = [
        hidden_ball_mask(periods[0].stop, periods, 25, np.random.default_rng(seed)).mean()
        for seed in range(12)
    ]
    assert 0.1 < np.mean(shares) < 0.45 and max(shares) < 0.9
    block = hidden_ball_mask(periods[0].stop, periods, 25, np.random.default_rng(5))
    per_block = block.reshape(-1, 120 * 25).mean(axis=1)
    assert (per_block == 0).any() and (per_block > 0.3).any()  # clear blocks and heavy ones


def test_ball_replacement_touches_nothing_else(small_bundle):
    tracking = small_bundle.tracking
    hidden = np.zeros(tracking.n_frames, dtype=bool)
    hidden[1000:1500] = True
    out = hide_ball(tracking, hidden)
    assert np.isnan(out.ball_xy[1000:1500]).all()
    assert np.array_equal(out.ball_xy[:1000], tracking.ball_xy[:1000], equal_nan=True)
    assert out.player_xy is tracking.player_xy and out.slot_team is tracking.slot_team
    assert with_ball(tracking, tracking.ball_xy).fingerprint() == tracking.fingerprint()


def test_degrade_mix_draws_are_seeded_and_follow_their_probabilities():
    mix = DegradeMix()
    draws = [mix.draw(np.random.default_rng(i)) for i in range(2000)]
    assert draws == [mix.draw(np.random.default_rng(i)) for i in range(2000)]
    assert all(d["version"] == "5" and d["seed"] >= SEED_LOW for d in draws)
    none = np.mean(["ladder" not in d for d in draws])
    assert 0.25 < none < 0.35  # ladder_p[0] = 0.30
    jittered = np.mean(["edge_jitter_s" in d for d in draws])
    assert 0.44 < jittered < 0.56
    assert DegradeMix(version=None).draw(np.random.default_rng(0)) == {}
    with pytest.raises(ValueError):
        DegradeMix(ladders=(None,), ladder_p=(0.5, 0.5))


def test_train_match_arrays_follow_the_clean_labels(match, small_bundle):
    played, labels = match.played, small_bundle.labels
    assert match.ball_target.shape == (small_bundle.meta.n_frames, 2)
    assert not np.isnan(match.ball_target).any()
    assert not match.target_ok[~played].any() and not match.target_ok[labels == D].any()
    assert match.target_ok[
        np.isin(labels, (H, A)) & played & small_bundle.tracking.ball_seen()
    ].all()
    assert set(np.unique(match.sign)) <= {-1.0, 1.0}


def test_a_training_bundle_needs_labels(small_bundle):
    from dataclasses import replace

    with pytest.raises(ValueError, match="labelled"):
        TrainMatch.from_bundle(replace(small_bundle, labels=None), SPEC)


def test_variants_report_what_the_input_ball_shows(match):
    entry = build_entry(match, np.random.default_rng(3), SPEC, DegradeMix(version=None))
    assert entry.coupled.dtype == np.float16 and entry.coupled.shape[1:] == (23, 14)
    ball_valid = entry.coupled[:, SPEC.ball_index, Col.VALID] > 0.5
    assert np.array_equal(ball_valid, entry.visible_coupled)
    assert np.array_equal(
        entry.decoupled[:, SPEC.ball_index, Col.VALID] > 0.5, entry.visible_decoupled
    )
    # the decoupled ball comes from the match's own ball track: it is never seen where that is missing
    clean_seen = match.bundle.tracking.ball_seen()
    assert not (entry.visible_decoupled & ~clean_seen).any()
    assert entry.info["draw"] == {}


def test_a_degraded_variant_hides_the_ball_through_the_stoppages(match):
    mix = DegradeMix(ladders=(None,), ladder_p=(1.0,), jitter_s=(0.0,), jitter_p=(1.0,))
    entry = build_entry(match, np.random.default_rng(4), SPEC, mix)
    dead = (match.labels == D) & match.played
    assert entry.info["draw"]["version"] == "5" and entry.info["draw"]["seed"] >= SEED_LOW
    assert entry.visible_coupled[dead].mean() < 0.2  # degrade-v5 blanks the stoppages
    assert entry.info["ball_hidden_coupled"] > 0.2


def test_sets_do_not_depend_on_the_thread_count(match):
    other = TrainMatch.from_bundle(generate(seed=8, config=SMALL), SPEC)
    mix = DegradeMix(version=None)
    one = build_set([match, other], 2, 0, SPEC, mix, workers=1)
    two = build_set([match, other], 2, 0, SPEC, mix, workers=2)
    for a, b in zip(one, two, strict=True):
        assert a.match_id == b.match_id
        assert np.array_equal(a.coupled, b.coupled) and np.array_equal(a.decoupled, b.decoupled)
    again = build_set([match, other], 3, 0, SPEC, mix)
    assert not np.array_equal(one[0].decoupled, again[0].decoupled)  # a new set, a new mask


def test_class_weights_treat_home_and_away_alike_and_average_one(match):
    weights = class_weights([match])
    assert weights[0] == weights[1] and weights.shape == (3,)
    played = match.played
    share = np.array([((match.labels == c) & played).sum() for c in range(3)], dtype=float)
    share = share / share.sum()
    share[:2] = share[:2].mean()
    assert float((share * weights).sum()) == pytest.approx(1.0)
    assert weights.argmin() == share.argmax()  # the commonest class weighs least


def test_only_dev_matches_load_for_training():
    for exam_match in idsse.EXAM_MATCHES:
        with pytest.raises(ValueError, match="only the IDSSE dev matches"):
            load_train_matches([exam_match], SPEC)
    with pytest.raises(ValueError):
        load_train_matches(["not-a-match"], SPEC)
