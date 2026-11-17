"""Optional training augmentations. Synthetic tracking only.

With every probability at zero, `build_entry` matches a copy of the pre-augment body, including
the random stream. The new warps are checked on hand-built tracks whose right answer is obvious.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, replace

import numpy as np
import pytest
import yaml

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.degrade import degrade
from aaaag.ml.augment import (
    AugmentMix,
    BallOffset,
    BallSmooth,
    GhostSlots,
    TeamStyle,
    _burst_mask,
    apply_augmentations,
    apply_ball_offset,
    apply_ball_smooth,
    apply_ghosts,
    apply_team_style,
    hidden_ball_mask,
    offset_ball_track,
    warp_player_xy,
    with_ball,
)
from aaaag.ml.data import DegradeMix, SetEntry, TrainMatch, build_entry, build_set
from aaaag.ml.geometry import symmetric
from aaaag.ml.spec import Col, TokenSpec
from aaaag.ml.windows import build_match_tokens
from aaaag.paths import CONFIGS_DIR
from aaaag.postprocess.touch_snap import ball_velocity_change

SPEC = TokenSpec()
OFF = AugmentMix(
    team_style=TeamStyle(p=0.0, stretch=0.5, shift_m=10.0),
    ball_offset=BallOffset(p=0.0, bias_m=5.0, lag_frames=8, noise_m=1.0),
    ghosts=GhostSlots(p=0.0, max_ghost_slots=8, duplicate_p=1.0),
)


def _tracking(positions, teams, ball, n=4, periods=None, *, speed=False) -> Tracking:
    pos = np.asarray(positions, dtype=np.float32)
    xy = np.broadcast_to(pos, (n, pos.shape[0], 2)).copy()
    ball_xy = np.broadcast_to(np.asarray(ball, dtype=np.float32), (n, 2)).copy()
    meta = MatchMeta("t", "test", n, 25, 105.0, 68.0, periods or (Period(1, 0, n),))
    player_speed = np.zeros((n, pos.shape[0]), np.float32) if speed else None
    return Tracking(meta, xy, np.asarray(teams, dtype=np.int8), ball_xy, player_speed=player_speed)


def _rng_blob(rng: np.random.Generator) -> bytes:
    return pickle.dumps(rng.bit_generator.state)


def _build_entry_without_augment(match, rng, spec, mix) -> SetEntry:
    """Body of `build_entry` before the optional warps existed."""
    draw = mix.draw(rng)
    clean = match.bundle
    degraded = degrade(clean, **draw) if draw else clean
    tracking = degraded.tracking
    meta = tracking.meta
    coupled = build_match_tokens(tracking, spec, sign=match.sign).astype(np.float16)
    hidden = hidden_ball_mask(meta.n_frames, meta.periods, meta.fps, rng)
    clean_ball = clean.tracking.ball_xy
    independent = with_ball(tracking, np.where(hidden[:, None], np.nan, clean_ball))
    decoupled = build_match_tokens(independent, spec, sign=match.sign).astype(np.float16)
    played = match.played
    info = {
        "draw": draw,
        "ball_hidden_coupled": float((~tracking.ball_seen())[played].mean()),
        "ball_hidden_decoupled": float((~independent.ball_seen())[played].mean()),
    }
    return SetEntry(
        match_id=match.match_id,
        coupled=coupled,
        decoupled=decoupled,
        visible_coupled=tracking.ball_seen() & played,
        visible_decoupled=independent.ball_seen() & played,
        info=info,
    )


def _same_entry(got: SetEntry, ref: SetEntry) -> None:
    assert got.match_id == ref.match_id and got.info == ref.info
    assert np.array_equal(got.coupled, ref.coupled)
    assert np.array_equal(got.decoupled, ref.decoupled)
    assert np.array_equal(got.visible_coupled, ref.visible_coupled)
    assert np.array_equal(got.visible_decoupled, ref.visible_decoupled)


@pytest.fixture(scope="module")
def match(small_bundle) -> TrainMatch:
    return TrainMatch.from_bundle(small_bundle, SPEC)


def test_disabled_augment_matches_today_and_the_random_stream(match):
    cases = (
        (0, DegradeMix(version=None), None),
        (1, DegradeMix(version=None), AugmentMix()),
        (2, DegradeMix(version=None), OFF),
        (3, DegradeMix(), OFF),
    )
    for seed, mix, augment in cases:
        rng_new, rng_old = np.random.default_rng(seed), np.random.default_rng(seed)
        got = build_entry(match, rng_new, SPEC, mix, augment)
        ref = _build_entry_without_augment(match, rng_old, SPEC, mix)
        _same_entry(got, ref)
        assert _rng_blob(rng_new) == _rng_blob(rng_old)


def test_team_style_is_the_stated_warp_and_spares_the_ball_and_the_near_players():
    tracking = _tracking(
        [(0.0, 0.0), (20.0, 0.0), (0.0, 3.0), (0.0, 30.0)],
        [HOME, HOME, AWAY, AWAY],
        (0.0, 0.0),
        n=3,
        speed=True,
    )
    scales = np.array([[1.5, 0.5], [0.8, 1.2]])
    shifts = np.array([[2.0, -1.0], [-3.0, 4.0]])
    out = warp_player_xy(
        tracking.player_xy, tracking.slot_team, tracking.ball_xy, tracking.meta, scales, shifts
    )
    assert np.array_equal(out[:, 0], tracking.player_xy[:, 0])  # 0 m from the ball
    assert np.array_equal(out[:, 2], tracking.player_xy[:, 2])  # 3 m
    assert np.allclose(out[:, 1], (27.0, -1.0), atol=1e-4)
    assert np.allclose(out[:, 3], (-3.0, 36.7), atol=1e-4)

    # w = 0 at 6 m, w = 1 at 12 m, w = smoothstep(0.5) = 0.5 at 9 m. c = (0+9+24)/3 = 11.
    ramp = _tracking([(0.0, 0.0), (9.0, 0.0), (24.0, 0.0), (6.0, 0.0)], [HOME] * 4, (0.0, 0.0))
    warped = warp_player_xy(
        ramp.player_xy,
        ramp.slot_team,
        ramp.ball_xy,
        ramp.meta,
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.zeros((2, 2)),
    )
    # The player at 6 m is shielded, but he changes c, so the exact check uses the other three
    # on their own track below. Here only the shield is asserted.
    assert warped[:, 3].tobytes() == ramp.player_xy[:, 3].tobytes()
    alone = _tracking([(0.0, 0.0), (9.0, 0.0), (24.0, 0.0)], [HOME] * 3, (0.0, 0.0))
    exact = warp_player_xy(
        alone.player_xy,
        alone.slot_team,
        alone.ball_xy,
        alone.meta,
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.zeros((2, 2)),
    )
    assert np.array_equal(exact[:, 0], alone.player_xy[:, 0])
    assert np.array_equal(exact[:, 1], np.tile(np.array([8.0, 0.0], np.float32), (4, 1)))
    assert np.array_equal(exact[:, 2], np.tile(np.array([37.0, 0.0], np.float32), (4, 1)))

    edge = _tracking([(12.0, 0.0)], [HOME], (0.0, 0.0))
    full = warp_player_xy(
        edge.player_xy,
        edge.slot_team,
        edge.ball_xy,
        edge.meta,
        np.ones((2, 2)),
        np.array([[4.0, -3.0], [0.0, 0.0]]),
    )
    assert np.array_equal(full, np.tile(np.array([16.0, -3.0], np.float32), (4, 1, 1)))
    shield = _tracking([(6.0, 0.0)], [HOME], (0.0, 0.0))
    held = warp_player_xy(
        shield.player_xy,
        shield.slot_team,
        shield.ball_xy,
        shield.meta,
        np.ones((2, 2)),
        np.array([[4.0, -3.0], [0.0, 0.0]]),
    )
    assert held.tobytes() == shield.player_xy.tobytes()

    # The anchor is the period mean, and a NaN ball uses the last finite one inside the period only.
    periods = (Period(1, 0, 4), Period(2, 4, 8))
    split = _tracking([(0.0, 0.0), (0.0, 0.0)], [HOME, HOME], (0.0, 0.0), n=8, periods=periods)
    split.player_xy[:4, 0, 0] = 10.0
    split.player_xy[:4, 1, 0] = 30.0
    split.player_xy[4:, 0, 0] = 10.0
    split.player_xy[4:, 1, 0] = 50.0
    split.ball_xy[:] = (-100.0, 0.0)
    anchored = warp_player_xy(
        split.player_xy,
        split.slot_team,
        split.ball_xy,
        split.meta,
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.zeros((2, 2)),
    )
    assert np.array_equal(anchored[:4, 0, 0], np.zeros(4))
    assert np.array_equal(anchored[:4, 1, 0], np.full(4, 40.0))
    assert np.array_equal(anchored[4:, 0, 0], np.full(4, -10.0))
    assert np.array_equal(anchored[4:, 1, 0], np.full(4, 70.0))

    filled = _tracking(
        [(2.0, 0.0), (30.0, 0.0)],
        [HOME, HOME],
        (0.0, 0.0),
        n=6,
        periods=(Period(1, 0, 3), Period(2, 4, 6)),
    )
    filled.ball_xy[:] = np.nan
    filled.ball_xy[0] = (0.0, 0.0)
    filled.player_xy[1, 0] = np.nan  # a missing near player stays missing
    moved = warp_player_xy(
        filled.player_xy,
        filled.slot_team,
        filled.ball_xy,
        filled.meta,
        np.ones((2, 2)),
        np.array([[5.0, 0.0], [0.0, 0.0]]),
    )
    assert moved[:, 0].tobytes() == filled.player_xy[:, 0].tobytes()
    assert np.allclose(moved[:3, 1, 0], 35.0)  # far from the forward-filled ball, including the gap
    assert np.array_equal(moved[3:, 1, 0], np.full(3, 30.0))  # gap frame and a period with no ball

    styled = _tracking(
        [(2.0, 0.0), (30.0, 0.0), (-2.0, 0.0), (-30.0, 0.0)],
        [HOME, HOME, AWAY, AWAY],
        (0.0, 0.0),
        n=5,
        speed=True,
    )
    original_speed = styled.player_speed
    changed = False
    for seed in range(8):
        out_t = apply_team_style(
            styled,
            styled.ball_xy,
            TeamStyle(p=1.0, stretch=0.15, shift_m=3.0),
            np.random.default_rng(seed),
        )
        assert out_t.ball_xy is styled.ball_xy
        assert out_t.slot_team is styled.slot_team and out_t.player_speed is original_speed
        assert out_t.player_xy[:, 0].tobytes() == styled.player_xy[:, 0].tobytes()
        assert out_t.player_xy[:, 2].tobytes() == styled.player_xy[:, 2].tobytes()
        changed = changed or not np.array_equal(out_t.player_xy[:, 1], styled.player_xy[:, 1])
    assert changed
    untouched = apply_team_style(
        styled,
        styled.ball_xy,
        TeamStyle(p=0.0, stretch=0.5, shift_m=10.0),
        np.random.default_rng(0),
    )
    assert untouched is styled


def test_ball_offset_shifts_both_tracks_and_leaves_players_and_gaps():
    ball = np.zeros((10, 2), np.float32)
    ball[:, 0] = np.arange(10)
    ball[3] = np.nan
    bias = np.array([1.0, -2.0])
    lagged = offset_ball_track(ball, bias, lag=2)
    assert np.isnan(lagged[:2]).all() and np.isnan(lagged[5]).all()
    assert np.allclose(lagged[2], (1.0, -2.0))
    assert np.allclose(lagged[6], (5.0, -2.0))
    back = offset_ball_track(ball, bias, lag=-2)
    assert np.isnan(back[-2:]).all() and np.isnan(back[1]).all()
    assert np.allclose(back[0], (3.0, -2.0))
    noisy = offset_ball_track(ball, np.zeros(2), 0, noise=np.ones((10, 2)))
    assert np.isnan(noisy[3]).all() and np.allclose(noisy[0], (1.0, 1.0))

    tracking = _tracking([(1.0, 2.0), (-4.0, 5.0)], [HOME, AWAY], (8.0, -1.0), n=12, speed=True)
    tracking.ball_xy[4] = np.nan
    clean = tracking.ball_xy.copy()
    clean[1::2] = np.nan
    z = np.zeros(12, np.float32)
    tracking = replace(tracking, ball_extra={"z": z})
    cfg = BallOffset(p=1.0, bias_m=2.0, lag_frames=0, noise_m=0.0)
    out, dec = apply_ball_offset(tracking, clean, cfg, np.random.default_rng(4))
    assert out.player_xy is tracking.player_xy and out.slot_team is tracking.slot_team
    assert out.player_speed is tracking.player_speed and out.ball_extra["z"] is z
    seen = tracking.ball_seen()
    delta = out.ball_xy[seen] - tracking.ball_xy[seen]
    assert np.allclose(delta, delta[0])
    assert 0.0 < np.linalg.norm(delta[0]) <= 2.0 + 1e-5
    both = seen & np.isfinite(clean[:, 0])
    assert np.allclose(dec[both] - clean[both], delta[0])
    assert np.isnan(out.ball_xy[~seen]).all()
    out.validate()

    pure = _tracking([(0.0, 0.0)], [HOME], (0.0, 0.0), n=9)
    pure.ball_xy[::3, 1] = np.linspace(-2.0, 2.0, 3)
    shifted, shifted_clean = apply_ball_offset(
        pure,
        pure.ball_xy,
        BallOffset(p=1.0, bias_m=0.0, lag_frames=4, noise_m=0.0),
        np.random.default_rng(1),
    )
    matched = [
        lag
        for lag in range(-4, 5)
        if np.array_equal(
            offset_ball_track(pure.ball_xy, np.zeros(2), lag), shifted.ball_xy, equal_nan=True
        )
    ]
    assert len(matched) == 1
    assert np.array_equal(
        offset_ball_track(pure.ball_xy, np.zeros(2), matched[0]), shifted_clean, equal_nan=True
    )
    same, same_ball = apply_ball_offset(
        pure, pure.ball_xy, BallOffset(p=0.0, bias_m=4.0), np.random.default_rng(0)
    )
    assert same is pure and same_ball is pure.ball_xy


def test_ghosts_only_add_columns_and_duplicates_sit_beside_one_teammate():
    tracking = _tracking([(0.0, 0.0), (30.0, 0.0)], [HOME, AWAY], (0.0, 0.0), n=40, speed=True)
    tracking.player_xy[3, 0] = np.nan
    raw = tracking.player_xy.copy()
    out = apply_ghosts(
        tracking, GhostSlots(p=1.0, max_ghost_slots=0, duplicate_p=1.0), np.random.default_rng(2)
    )
    assert out.player_xy[:, :2].tobytes() == raw.tobytes()
    assert out.slot_team[:2].tobytes() == tracking.slot_team.tobytes()
    assert out.slot_team[2:].tolist() == [HOME, AWAY]
    assert out.ball_xy is tracking.ball_xy and out.player_speed is None
    home = out.player_xy[:, 2]
    # frame 3 has no observed home slot, so the home duplicate is absent there
    assert np.isnan(home[3]).all()
    present = np.isfinite(home[:, 0])
    dist = np.linalg.norm(home[present] - np.array([0.0, 0.0]), axis=-1)
    assert np.all((dist >= 0.2 - 1e-4) & (dist <= 0.6 + 1e-4))
    assert np.all(np.linalg.norm(home[present] - np.array([30.0, 0.0]), axis=-1) > 20.0)
    out.validate()

    base = _tracking(
        [(0.0, 0.0)] * 4 + [(10.0, 0.0)] * 4, [HOME] * 4 + [AWAY] * 4, (0.0, 0.0), n=4000
    )
    found = None
    for seed in range(12):
        ghosts = apply_ghosts(
            base, GhostSlots(p=1.0, max_ghost_slots=4, duplicate_p=0.0), np.random.default_rng(seed)
        )
        if ghosts.n_slots > base.n_slots:
            found = ghosts
            break
    assert found is not None
    extra = found.player_xy[:, base.n_slots :]
    assert extra[:, :].shape[1] == found.n_slots - base.n_slots
    assert found.player_xy[:, : base.n_slots].tobytes() == base.player_xy.tobytes()
    finite = np.isfinite(extra[..., 0])
    assert finite.any() and (~finite).any()
    assert np.nanmax(np.abs(extra[..., 0])) <= 52.5 + 1e-3
    assert np.nanmax(np.abs(extra[..., 1])) <= 34.0 + 1e-3
    assert np.isin(found.slot_team[base.n_slots :], (HOME, AWAY)).all()
    assert found.ball_xy is base.ball_xy
    mask = _burst_mask(4000, 20.0, 60.0, True, np.random.default_rng(0))
    assert np.array_equal(mask, _burst_mask(4000, 20.0, 60.0, True, np.random.default_rng(0)))
    assert 0.15 < float(mask.mean()) < 0.45
    skipped = apply_ghosts(
        base, GhostSlots(p=0.0, max_ghost_slots=4, duplicate_p=1.0), np.random.default_rng(0)
    )
    assert skipped is base


def test_augmented_draws_are_deterministic_and_do_not_touch_the_clean_bundle(match):
    mix = AugmentMix(
        team_style=TeamStyle(p=1.0),
        ball_offset=BallOffset(p=1.0, lag_frames=0, noise_m=0.0),
        ghosts=GhostSlots(p=1.0, duplicate_p=0.3),
    )
    players = match.bundle.tracking.player_xy.copy()
    ball = match.bundle.tracking.ball_xy.copy()

    def run(seed: int) -> SetEntry:
        return build_entry(match, np.random.default_rng(seed), SPEC, DegradeMix(version=None), mix)

    one, two, other = run(5), run(5), run(6)
    assert np.array_equal(one.coupled, two.coupled)
    assert np.array_equal(one.decoupled, two.decoupled)
    assert not (
        np.array_equal(one.coupled, other.coupled)
        and np.array_equal(one.decoupled, other.decoupled)
    )
    assert np.array_equal(match.bundle.tracking.player_xy, players, equal_nan=True)
    assert np.array_equal(match.bundle.tracking.ball_xy, ball, equal_nan=True)
    both = one.visible_coupled & one.visible_decoupled
    assert int(both.sum()) > 100
    # Gap and ball-velocity features look past the frame, so a neighbour hidden only in the
    # decoupled variant changes them. Positions and the chosen players do not.
    ball_i = SPEC.ball_index
    pos = (Col.X, Col.Y, Col.VALID, Col.IS_BALL, Col.REF_BALL)
    players_differ = int((one.coupled[both, :ball_i] != one.decoupled[both, :ball_i]).sum())
    ball_differs = int(
        (one.coupled[both, ball_i][:, pos] != one.decoupled[both, ball_i][:, pos]).sum()
    )
    assert players_differ == 0 and ball_differs == 0

    plain = build_entry(match, np.random.default_rng(5), SPEC, DegradeMix(version=None))
    assert not np.array_equal(plain.coupled[:, ball_i], one.coupled[:, ball_i])
    assert not np.array_equal(plain.decoupled[:, ball_i], one.decoupled[:, ball_i])

    style = AugmentMix(team_style=TeamStyle(p=1.0, stretch=0.15, shift_m=3.0))
    styled = build_entry(match, np.random.default_rng(5), SPEC, DegradeMix(version=None), style)
    assert np.array_equal(plain.coupled[:, ball_i], styled.coupled[:, ball_i])
    assert not np.array_equal(plain.coupled[:, :ball_i], styled.coupled[:, :ball_i])
    assert not np.array_equal(plain.decoupled[:, :ball_i], styled.decoupled[:, :ball_i])

    workers_one = build_set(
        [match, match], 1, 9, SPEC, DegradeMix(version=None), workers=1, augment=mix
    )
    workers_two = build_set(
        [match, match], 1, 9, SPEC, DegradeMix(version=None), workers=2, augment=mix
    )
    for left, right in zip(workers_one, workers_two, strict=True):
        assert np.array_equal(left.coupled, right.coupled)
        assert np.array_equal(left.decoupled, right.decoupled)


def test_symmetries_hold_on_augmented_tracking(small_bundle):
    mix = AugmentMix(
        team_style=TeamStyle(p=1.0, stretch=0.15, shift_m=3.0),
        ball_offset=BallOffset(p=1.0, bias_m=2.0, lag_frames=4, noise_m=0.2),
        ghosts=GhostSlots(p=1.0, max_ghost_slots=4, duplicate_p=0.4),
    )
    tracking, _ = apply_augmentations(
        small_bundle.tracking,
        small_bundle.tracking.ball_xy,
        mix,
        np.random.default_rng(11),
    )
    tracking.validate()
    spec = TokenSpec()
    played = small_bundle.meta.played_mask()
    swapped = replace(
        tracking, slot_team=(1 - tracking.slot_team).astype(np.int8), _fingerprint=None
    )
    expected = symmetric(build_match_tokens(tracking, spec), spec, swap=True)
    got = build_match_tokens(swapped, spec)
    assert np.allclose(got[played], expected[played], atol=1e-5)
    mirror = np.array([1.0, -1.0], dtype=np.float32)
    flipped = replace(
        tracking,
        player_xy=tracking.player_xy * mirror,
        ball_xy=tracking.ball_xy * mirror,
        _fingerprint=None,
    )
    expected = symmetric(build_match_tokens(tracking, spec), spec, flip_y=True)
    assert np.allclose(build_match_tokens(flipped, spec), expected, atol=1e-5)


def test_augment_config_round_trips_and_old_dicts_default_off():
    assert AugmentMix.from_dict(None) == AugmentMix()
    assert AugmentMix.from_dict({}) == AugmentMix()
    partial = AugmentMix.from_dict({"team_style": {"p": 1}})
    assert partial.team_style == TeamStyle(p=1.0)
    assert partial.ball_offset == BallOffset() and partial.ghosts == GhostSlots()
    assert partial.ball_smooth == BallSmooth()
    assert AugmentMix.from_dict(asdict(partial)) == partial
    with pytest.raises(ValueError, match="unknown augment"):
        AugmentMix.from_dict({"team": {}})
    with pytest.raises(ValueError, match="team_style.p"):
        TeamStyle(p=1.5)

    base = yaml.safe_load((CONFIGS_DIR / "ml" / "team_baseline.yaml").read_text(encoding="utf-8"))
    raw = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_augment_example.yaml").read_text(encoding="utf-8")
    )
    aug = raw["train"].pop("augment")
    assert raw == base
    assert AugmentMix.from_dict(aug) == AugmentMix(
        team_style=TeamStyle(p=0.5, stretch=0.15, shift_m=3.0),
        ball_offset=BallOffset(p=0.5, bias_m=2.0, lag_frames=4, noise_m=0.0),
        ghosts=GhostSlots(p=0.5, max_ghost_slots=4, duplicate_p=0.0),
    )


def test_run_config_round_trips_including_a_config_without_augment():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, TrainConfig

    bare = TrainConfig.from_dict({"steps": 12, "degrade": {"version": None}, "augment": None})
    assert bare.steps == 12 and bare.degrade.version is None and bare.augment == AugmentMix()
    assert RunConfig.from_dict({"train": {"seed": 3}}).train.augment == AugmentMix()
    base = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_baseline.yaml")
    assert base.train.augment == AugmentMix()
    dumped = base.to_dict()
    assert dumped["train"]["augment"]["team_style"]["p"] == 0.0
    assert RunConfig.from_dict(dumped) == base
    del dumped["train"]["augment"]
    assert RunConfig.from_dict(dumped) == base
    example = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_augment_example.yaml")
    assert example.train.augment.ball_offset.lag_frames == 4
    restored = replace(example, train=replace(example.train, augment=AugmentMix()))
    assert restored == base


def test_a_long_match_augments_in_under_three_seconds():
    import time

    frames, slots = 146_000, 36
    xy = np.zeros((frames, slots, 2), np.float32)
    xy[..., 0] = np.linspace(-40.0, 40.0, slots, dtype=np.float32)
    xy[::17, ::5] = np.nan
    ball = np.zeros((frames, 2), np.float32)
    ball[::13] = np.nan
    teams = np.array([HOME] * (slots // 2) + [AWAY] * (slots - slots // 2), dtype=np.int8)
    meta = MatchMeta(
        "long",
        "test",
        frames,
        25,
        105.0,
        68.0,
        (Period(1, 0, frames // 2), Period(2, frames // 2, frames)),
    )
    tracking = Tracking(meta, xy, teams, ball, player_speed=np.zeros((frames, slots), np.float32))
    mix = AugmentMix(
        team_style=TeamStyle(p=1.0, stretch=0.15, shift_m=3.0),
        ball_offset=BallOffset(p=1.0, bias_m=2.0, lag_frames=4, noise_m=0.25),
        ghosts=GhostSlots(p=1.0, max_ghost_slots=4, duplicate_p=0.5),
    )
    started = time.perf_counter()
    out, decoupled = apply_augmentations(tracking, ball, mix, np.random.default_rng(0))
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"augmentations took {elapsed:.2f}s"
    assert out.n_slots >= slots + 2 and decoupled.shape == ball.shape
    out.validate()


def test_ball_smooth_off_matches_today_and_draws_nothing():
    tracking = _tracking([(0.0, 0.0), (12.0, -3.0)], [HOME, AWAY], (1.0, 2.0), n=30, speed=True)
    tracking.ball_xy[::4] = np.nan
    ball = tracking.ball_xy.copy()
    raw_players = tracking.player_xy.copy()

    rng = np.random.default_rng(7)
    untouched = np.random.default_rng(7)
    out, dec = apply_augmentations(tracking, ball, AugmentMix(), rng)
    assert out is tracking and dec is ball
    assert np.array_equal(tracking.player_xy, raw_players)
    assert rng.random() == untouched.random()

    rng_off = np.random.default_rng(8)
    rng_ref = np.random.default_rng(8)
    same, same_ball = apply_ball_smooth(
        tracking,
        ball,
        BallSmooth(p=0.0, sigma_frames=(6.0,), sigma_p=None),
        rng_off,
    )
    assert same is tracking and same_ball is ball
    assert rng_off.random() == rng_ref.random()

    # The new block sits between the offset and the ghosts. p = 0 must not shift that stream.
    mix = AugmentMix(
        team_style=TeamStyle(p=1.0, stretch=0.15, shift_m=3.0),
        ball_offset=BallOffset(p=1.0, bias_m=2.0, lag_frames=4, noise_m=0.2),
        ball_smooth=BallSmooth(p=0.0, sigma_frames=(2.0, 4.0, 6.0)),
        ghosts=GhostSlots(p=1.0, max_ghost_slots=2, duplicate_p=0.3),
    )
    rng_new, rng_old = np.random.default_rng(9), np.random.default_rng(9)
    got, got_ball = apply_augmentations(tracking, ball, mix, rng_new)
    old = apply_team_style(tracking, ball, mix.team_style, rng_old)
    old, old_ball = apply_ball_offset(old, ball, mix.ball_offset, rng_old)
    old = apply_ghosts(old, mix.ghosts, rng_old)
    assert np.array_equal(got.player_xy, old.player_xy, equal_nan=True)
    assert np.array_equal(got.ball_xy, old.ball_xy, equal_nan=True)
    assert np.array_equal(got.slot_team, old.slot_team)
    assert np.array_equal(got_ball, old_ball, equal_nan=True)
    assert rng_new.random() == rng_old.random()


def test_ball_smooth_rounds_a_long_corner_and_keeps_gaps_and_short_runs():
    sigma = 4.0
    n = 120
    ball = np.full((n, 2), np.nan, np.float32)
    # Long run [0, 80), corner at frame 40. 80 > 4*sigma.
    ball[:40, 0] = np.arange(40, dtype=np.float32)
    ball[:40, 1] = 0.0
    ball[40:80, 0] = 39.0
    ball[40:80, 1] = np.arange(40, dtype=np.float32)
    # Gap [80, 90). Short corner [90, 100), length 10 < 16.
    # Gap [100, 104) so it does not merge with the length-16 run [104, 120).
    ball[90:95, 0] = np.arange(5, dtype=np.float32)
    ball[90:95, 1] = 0.0
    ball[95:100, 0] = 4.0
    ball[95:100, 1] = np.arange(5, dtype=np.float32)
    ball[104:120, 0] = np.arange(16, dtype=np.float32)
    ball[104:120, 1] = np.array([0.0, 3.0] * 8, dtype=np.float32)

    z = np.zeros(n, np.float32)
    tracking = replace(
        _tracking([(1.0, 2.0)], [HOME], (0.0, 0.0), n=n, speed=True),
        ball_xy=ball.copy(),
        ball_extra={"z": z},
    )
    clean = ball.copy()
    clean[:80] += np.array([5.0, -2.0], np.float32)
    cfg = BallSmooth(p=1.0, sigma_frames=(sigma,))
    out, dec = apply_ball_smooth(tracking, clean, cfg, np.random.default_rng(0))

    raw_dv = ball_velocity_change(ball, tracking.meta.fps, 2)
    new_dv = ball_velocity_change(out.ball_xy, tracking.meta.fps, 2)
    assert np.isfinite(raw_dv[40]) and raw_dv[40] > 0.0
    assert new_dv[40] < raw_dv[40]
    assert np.isnan(out.ball_xy[80:90]).all()
    assert np.isnan(out.ball_xy[100:104]).all()
    assert np.array_equal(out.ball_xy[90:100], ball[90:100])
    assert np.array_equal(out.ball_xy[104:120], ball[104:120])
    assert not np.array_equal(out.ball_xy[:80], ball[:80])

    assert np.isnan(dec[80:90]).all()
    assert np.isnan(dec[100:104]).all()
    assert np.array_equal(dec[90:100], clean[90:100])
    assert np.array_equal(dec[104:120], clean[104:120])
    offset = np.array([5.0, -2.0], np.float32)
    assert np.allclose(dec[:80], out.ball_xy[:80] + offset, rtol=1e-5, atol=1e-4)
    assert not np.allclose(dec[:80], out.ball_xy[:80])
    again, twin = apply_ball_smooth(tracking, ball.copy(), cfg, np.random.default_rng(0))
    assert np.array_equal(again.ball_xy, out.ball_xy, equal_nan=True)
    assert np.array_equal(twin, again.ball_xy, equal_nan=True)
    weighted = BallSmooth(p=1.0, sigma_frames=(sigma,), sigma_p=(1.0,))
    via_p, via_p_ball = apply_ball_smooth(tracking, clean, weighted, np.random.default_rng(0))
    assert np.array_equal(via_p.ball_xy, out.ball_xy, equal_nan=True)
    assert np.array_equal(via_p_ball, dec, equal_nan=True)

    assert out.player_xy is tracking.player_xy
    assert out.slot_team is tracking.slot_team
    assert out.player_speed is tracking.player_speed
    assert out.ball_extra["z"] is z
    out.validate()

    # A continuous track split into two periods shorter than 4*sigma is not one long run.
    ramp = np.column_stack(
        (
            np.arange(20, dtype=np.float32),
            np.linspace(-3.0, 4.0, 20, dtype=np.float32),
        )
    )
    periods = (Period(1, 0, 10), Period(2, 10, 20))
    split = replace(
        _tracking([(0.0, 0.0)], [HOME], (0.0, 0.0), n=20, periods=periods),
        ball_xy=ramp.copy(),
    )
    held, held_ball = apply_ball_smooth(split, ramp.copy(), cfg, np.random.default_rng(1))
    assert np.array_equal(held.ball_xy, ramp)
    assert np.array_equal(held_ball, ramp)
    whole = replace(
        _tracking([(0.0, 0.0)], [HOME], (0.0, 0.0), n=20),
        ball_xy=ramp.copy(),
    )
    moved, _ = apply_ball_smooth(whole, ramp.copy(), cfg, np.random.default_rng(1))
    assert not np.array_equal(moved.ball_xy, ramp)


def test_ball_smooth_from_dict_validation():
    parsed = AugmentMix.from_dict(
        {"ball_smooth": {"p": 0.6, "sigma_frames": [2, 4, 6], "sigma_p": [0.2, 0.2, 0.6]}}
    )
    assert parsed.ball_smooth == BallSmooth(
        p=0.6, sigma_frames=(2.0, 4.0, 6.0), sigma_p=(0.2, 0.2, 0.6)
    )
    assert isinstance(parsed.ball_smooth.sigma_frames, tuple)
    assert isinstance(parsed.ball_smooth.sigma_p, tuple)
    assert AugmentMix.from_dict(asdict(parsed)).ball_smooth == parsed.ball_smooth

    with pytest.raises(ValueError, match="ball_smooth.p"):
        AugmentMix.from_dict({"ball_smooth": {"p": -0.1}})
    with pytest.raises(ValueError, match="ball_smooth.p"):
        AugmentMix.from_dict({"ball_smooth": {"p": 1.2}})
    with pytest.raises(ValueError, match="sigma_frames"):
        AugmentMix.from_dict({"ball_smooth": {"sigma_frames": [0, 4]}})
    with pytest.raises(ValueError, match="sigma_frames"):
        AugmentMix.from_dict({"ball_smooth": {"sigma_frames": [-2.0]}})
    with pytest.raises(ValueError, match="sigma_frames"):
        AugmentMix.from_dict({"ball_smooth": {"sigma_frames": []}})
    with pytest.raises(ValueError, match="sigma_p"):
        AugmentMix.from_dict({"ball_smooth": {"sigma_frames": [2, 4], "sigma_p": [1.0]}})
    with pytest.raises(ValueError, match="sigma_p"):
        AugmentMix.from_dict({"ball_smooth": {"sigma_frames": [2, 4], "sigma_p": [-0.2, 1.2]}})
    with pytest.raises(ValueError, match="sigma_p"):
        AugmentMix.from_dict({"ball_smooth": {"sigma_frames": [2, 4], "sigma_p": [0.2, 0.2]}})
    with pytest.raises(ValueError, match="unknown augment.ball_smooth"):
        AugmentMix.from_dict({"ball_smooth": {"p": 0.5, "scale": 1}})
    with pytest.raises(ValueError, match="unknown augment"):
        AugmentMix.from_dict({"ball_smooth": {"p": 1}, "scale": {}})


def test_ball_smooth_configs_load():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig

    expected = BallSmooth(p=0.6, sigma_frames=(2.0, 4.0, 6.0))
    base = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_switch.yaml")
    smooth = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_switch_smooth.yaml")
    assert smooth.train.augment.ball_smooth == expected
    assert smooth.train.augment.team_style == TeamStyle()
    assert base.train.augment.ball_smooth == BallSmooth()
    restored = replace(smooth, train=replace(smooth.train, augment=AugmentMix()))
    assert restored == base

    robust = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_switch_robust.yaml")
    robust_smooth = RunConfig.from_yaml(
        CONFIGS_DIR / "ml" / "team_cm_contact_switch_robust_smooth.yaml"
    )
    assert robust_smooth.train.augment.ball_smooth == expected
    assert robust.train.augment == AugmentMix()
    restored = replace(robust_smooth, train=replace(robust_smooth.train, augment=AugmentMix()))
    assert restored == robust
