"""E158 alive_gap: label-free coupled-ball gaps on alive frames up to a target share. Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, Period, X
from aaaag.ml.augment import alive_gap_mask
from aaaag.ml.data import DegradeMix, TrainMatch, build_entry, build_set
from aaaag.ml.spec import TokenSpec

FPS = 25
N = 30_000  # 20 minutes
PERIODS = (Period(1, 0, 15_000), Period(2, 15_500, N))  # a half-time gap of 20 s


def _labels() -> np.ndarray:
    """Alternating 40 s alive (H then A) and 20 s dead spells, X outside the periods."""
    lab = np.full(N, X, dtype=np.int8)
    for period in PERIODS:
        t = np.arange(period.start, period.stop) - period.start
        phase = (t // FPS) % 60
        team = np.where(((t // FPS) // 60) % 2 == 0, H, A)
        lab[period.start : period.stop] = np.where(phase < 40, team, D)
    return lab


def _missing(labels: np.ndarray) -> np.ndarray:
    """Hidden on 95% of dead frames and 4% of alive ones (fixed, not random)."""
    idx = np.arange(N)
    missing = np.zeros(N, dtype=bool)
    missing[labels == D] = (idx[labels == D] % 20) != 0
    alive = np.isin(labels, (H, A))
    missing[alive] = (idx[alive] % 25) == 0
    return missing


def test_the_alive_missing_share_reaches_the_target_in_expectation():
    labels = _labels()
    missing = _missing(labels)
    alive = np.isin(labels, (H, A))
    start = missing[alive].mean()
    assert abs(start - 0.04) < 0.002
    shares = []
    for seed in range(40):
        gaps = alive_gap_mask(missing, labels, PERIODS, FPS, np.random.default_rng(seed), 0.12)
        shares.append((missing | gaps)[alive].mean())
    assert abs(np.mean(shares) - 0.12) < 0.01
    assert min(shares) > start  # every draw adds gaps


def test_dead_x_and_unplayed_frames_are_never_hidden():
    labels = _labels()
    missing = _missing(labels)
    for seed in range(10):
        gaps = alive_gap_mask(missing, labels, PERIODS, FPS, np.random.default_rng(seed), 0.5)
        assert gaps.any()
        assert not gaps[labels == D].any()
        assert not gaps[labels == X].any()
        assert not gaps[15_000:15_500].any()


def test_the_gaps_are_runs_from_the_renewal_distribution():
    labels = np.full(N, H, dtype=np.int8)  # all alive: the process shows unclipped
    missing = np.zeros(N, dtype=bool)
    gaps = alive_gap_mask(missing, labels, PERIODS, FPS, np.random.default_rng(3), 0.3)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], gaps.view(np.int8), [0]))))
    lengths = edges[1::2] - edges[::2]
    assert len(lengths) > 10
    assert np.median(lengths) >= 1 and lengths.max() < 600 * FPS


def test_nothing_is_drawn_or_hidden_when_the_share_is_already_reached():
    labels = _labels()
    missing = _missing(labels)
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    gaps = alive_gap_mask(missing, labels, PERIODS, FPS, rng, 0.03)
    assert not gaps.any()
    assert rng.bit_generator.state == state
    none = np.full(N, D, dtype=np.int8)
    assert not alive_gap_mask(missing, none, PERIODS, FPS, rng, 0.12).any()
    assert rng.bit_generator.state == state


def test_the_mask_is_seeded_by_the_generator():
    labels = _labels()
    missing = _missing(labels)
    one = alive_gap_mask(missing, labels, PERIODS, FPS, np.random.default_rng(7), 0.12)
    two = alive_gap_mask(missing, labels, PERIODS, FPS, np.random.default_rng(7), 0.12)
    other = alive_gap_mask(missing, labels, PERIODS, FPS, np.random.default_rng(8), 0.12)
    assert np.array_equal(one, two)
    assert not np.array_equal(one, other)


@pytest.mark.parametrize("target", [-0.1, 1.0, float("nan")])
def test_a_target_outside_zero_one_is_refused(target):
    labels = _labels()
    with pytest.raises(ValueError, match="target"):
        alive_gap_mask(_missing(labels), labels, PERIODS, FPS, np.random.default_rng(0), target)


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
    assert "alive_gap_frames" not in plain.info
    for off in (None, 0, 0.0):
        again = build_entry(match, np.random.default_rng(3), TokenSpec(), mix, alive_gap=off)
        _same_draw(again, plain)
    sets = build_set([match], 2, 11, TokenSpec(), mix, workers=1)
    sets_off = build_set([match], 2, 11, TokenSpec(), mix, workers=1, alive_gap=None)
    _same_draw(sets_off[0], sets[0])


def test_on_hides_only_alive_frames_of_the_coupled_variant(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    mix = DegradeMix()
    plain = build_entry(match, np.random.default_rng(3), TokenSpec(), mix)
    on = build_entry(match, np.random.default_rng(3), TokenSpec(), mix, alive_gap=0.6)
    assert on.info["alive_gap_frames"] > 0
    # the decoupled variant and every earlier draw are unchanged
    assert on.decoupled.tobytes() == plain.decoupled.tobytes()
    assert on.visible_decoupled.tobytes() == plain.visible_decoupled.tobytes()
    # the coupled ball only loses frames, and only alive ones
    lost = plain.visible_coupled & ~on.visible_coupled
    assert not (on.visible_coupled & ~plain.visible_coupled).any()
    assert int(lost.sum()) == on.info["alive_gap_frames"]
    assert np.isin(match.labels[lost], (H, A)).all()
    alive = np.isin(match.labels, (H, A)) & match.played
    assert (~on.visible_coupled)[alive].mean() > (~plain.visible_coupled)[alive].mean()


def test_a_match_without_degradation_is_left_alone(small_bundle):
    match = TrainMatch.from_bundle(small_bundle, TokenSpec())
    mix = DegradeMix()
    plain = build_entry(match, np.random.default_rng(3), TokenSpec(), mix, skip_degrade=True)
    on = build_entry(
        match, np.random.default_rng(3), TokenSpec(), mix, skip_degrade=True, alive_gap=0.6
    )
    _same_draw(on, plain)


def test_the_config_round_trip_drops_the_default():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, replace_train

    run = RunConfig.from_dict({})
    assert run.train.alive_gap_target == 0.0
    assert "alive_gap_target" not in run.to_dict()["train"]
    on = replace_train(run, alive_gap_target=0.12)
    assert RunConfig.from_dict(on.to_dict()).train.alive_gap_target == 0.12
    for bad in (1.0, -0.1, True, float("nan")):
        with pytest.raises(ValueError, match="alive_gap_target"):
            replace_train(run, alive_gap_target=bad)
