import numpy as np

from aaaag.contracts import A, D, H, X
from aaaag.synthetic import SyntheticConfig, generate

from .conftest import SMALL


def test_same_seed_same_match():
    a, b = generate(seed=3, config=SMALL), generate(seed=3, config=SMALL)
    assert a.tracking.fingerprint() == b.tracking.fingerprint()
    assert np.array_equal(a.labels, b.labels)


def test_different_seeds_differ():
    assert (
        generate(seed=1, config=SMALL).tracking.fingerprint()
        != generate(seed=2, config=SMALL).tracking.fingerprint()
    )


def test_labels_cover_periods_exactly(small_bundle):
    played = small_bundle.meta.played_mask()
    assert (small_bundle.labels[~played] == X).all()
    assert np.isin(small_bundle.labels[played], (H, A, D)).all()
    assert {H, A, D} <= set(np.unique(small_bundle.labels).tolist())


def test_observation_damage_is_ata_like(small_bundle):
    tracking, played = small_bundle.tracking, small_bundle.meta.played_mask()
    missing = 1.0 - tracking.ball_seen()[played].mean()
    assert 0.3 < missing < 0.65  # ~0.4 base rate plus extra dropout during stoppages
    objects = (~np.isnan(tracking.player_xy[played, :, 0])).sum(axis=1)
    assert 19 <= objects.mean() <= 22
    assert tracking.n_slots == 46


def test_config_overrides_flow_through():
    bundle = generate(
        seed=0, config=SyntheticConfig(minutes_per_half=1.0, gap_minutes=0.2, ball_missing_rate=0.1)
    )
    assert bundle.meta.n_frames == int(2 * 60 * 25 + 3 * 0.2 * 60 * 25)
