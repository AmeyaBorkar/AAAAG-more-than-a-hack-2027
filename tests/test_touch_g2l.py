"""Guarded two-line impulse (`detect_touches_g2l`, `token.touch_detector: g2l`).

The detector is off unless the spec asks for it: "w2", or a missing key, leaves every token
tensor identical to today's. Synthetic tracks only. Torch pieces skip when torch is absent.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from aaaag.contracts import AWAY, HOME, MatchBundle, MatchMeta, Period, Tracking, X
from aaaag.ml.geometry import symmetric, team_swap_permutation
from aaaag.ml.spec import N_FEATURES, TokenSpec, contact_columns, path_columns, touch_dv_column
from aaaag.ml.windows import build_match_tokens
from aaaag.paths import CONFIGS_DIR
from aaaag.postprocess.touch_snap import detect_touches_g2l, touch_snap
from aaaag.synthetic import SyntheticConfig, generate

FPS = 25
# A 90 degree turn at this speed changes velocity by exactly 6 m/s.
SPEED = 6.0 / np.sqrt(2.0)
V_IN = np.array([SPEED, 0.0])
V_OUT = np.array([0.0, SPEED])
TRUE_DV = 6.0


def _meta(n: int, periods: tuple[Period, ...] | None = None) -> MatchMeta:
    return MatchMeta(
        "synthetic",
        "synthetic",
        n,
        FPS,
        105.0,
        68.0,
        periods or (Period(1, 0, n),),
    )


def _tracking(
    ball: np.ndarray,
    home: tuple[float, float] | np.ndarray,
    away: tuple[float, float] | np.ndarray,
    periods: tuple[Period, ...] | None = None,
) -> Tracking:
    n = len(ball)
    players = np.empty((n, 2, 2), dtype=np.float64)
    players[:, 0] = home
    players[:, 1] = away
    return Tracking(
        meta=_meta(n, periods),
        player_xy=players.astype(np.float32),
        slot_team=np.array([HOME, AWAY], dtype=np.int8),
        ball_xy=np.asarray(ball, dtype=np.float32),
    )


def _corner(n: int, frame: int, v_in: np.ndarray = V_IN, v_out: np.ndarray = V_OUT) -> np.ndarray:
    """Ball position, vertex at the origin on `frame`. Velocity switches after that frame."""
    t = np.arange(n, dtype=np.float64)
    dt = (t - frame) / float(FPS)
    ball = np.empty((n, 2), dtype=np.float64)
    before = t <= frame
    ball[before] = v_in * dt[before, None]
    ball[~before] = v_out * dt[~before, None]
    return ball


def _two_corners(n: int, first: int, second: int) -> np.ndarray:
    """Two 6 m/s kinks. The first vertex is the origin; the second is where that leg ends."""
    v3 = np.array([-SPEED, 0.0])
    ball = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        if i <= first:
            ball[i] = V_IN * ((i - first) / FPS)
        elif i <= second:
            ball[i] = V_OUT * ((i - first) / FPS)
        else:
            ball[i] = V_OUT * ((second - first) / FPS) + v3 * ((i - second) / FPS)
    return ball


def test_an_isolated_kink_recovers_the_impulse_and_the_player_at_the_vertex():
    frame = 250
    ball = _corner(500, frame)
    tracking = _tracking(ball, (0.0, 0.0), (40.0, 0.0))
    frames, teams, strength = detect_touches_g2l(tracking)
    assert len(frames) == 1
    assert abs(int(frames[0]) - frame) <= 1
    assert float(strength[0]) == pytest.approx(TRUE_DV, rel=0.05)
    assert int(teams[0]) == HOME


def test_gaussian_smoothing_keeps_most_of_the_impulse():
    """σ = 6 frames. The note's recovery at this scale is 0.914; the bar here is 85%."""
    frame = 250
    ball = gaussian_filter1d(_corner(500, frame), sigma=6.0, axis=0)
    tracking = _tracking(ball, (0.0, 0.0), (40.0, 0.0))
    frames, teams, strength = detect_touches_g2l(tracking)
    assert len(frames) == 1
    assert abs(int(frames[0]) - frame) <= 1
    assert float(strength[0]) >= 0.85 * TRUE_DV
    assert int(teams[0]) == HOME


def test_white_jitter_on_a_straight_track_stays_under_one_event_per_10_000_frames():
    rng = np.random.default_rng(0)
    n = 10_000
    t = np.arange(n, dtype=np.float64)
    ball = np.stack([t * (5.0 / FPS), np.zeros(n)], axis=1)
    ball += rng.normal(0.0, 0.10, size=ball.shape)
    tracking = _tracking(ball, (0.0, 30.0), (0.0, -30.0))
    frames, _, _ = detect_touches_g2l(tracking)
    assert len(frames) <= n / 10_000


def test_two_touches_forty_frames_apart_stay_two_events():
    first, second = 200, 240
    ball = _two_corners(500, first, second)
    tracking = _tracking(ball, ball[first], ball[second])
    frames, teams, _ = detect_touches_g2l(tracking)
    assert len(frames) == 2
    assert abs(int(frames[0]) - first) <= 1
    assert abs(int(frames[1]) - second) <= 1
    assert int(teams[0]) == HOME
    assert int(teams[1]) == AWAY


def test_a_guard_gap_is_bridged_and_a_fit_window_gap_is_not():
    """The vertex frame sits in the guard, so hiding it still leaves the event.

    The only frames that can form a run of 3 are the fit windows of t-1, t and t+1. Blanking
    one frame of either window drops the run under 3, so there is no event.
    """
    frame = 80
    n = 160
    full = _corner(n, frame)
    seen = np.zeros(n, dtype=bool)
    seen[frame - 14 : frame - 6] = True  # [t-14, t-7], the incoming windows
    seen[frame + 7 : frame + 15] = True  # [t+7, t+14], the outgoing windows
    ball = np.full((n, 2), np.nan)
    ball[seen] = full[seen]
    tracking = _tracking(ball, (0.0, 0.0), (40.0, 0.0))
    frames, teams, strength = detect_touches_g2l(tracking)
    assert list(map(int, frames)) == [frame]
    assert int(teams[0]) == HOME
    assert float(strength[0]) == pytest.approx(TRUE_DV, rel=0.05)

    # Thirteen unseen frames around the vertex, still inside the guard of the surviving run.
    gapped = full.copy()
    gapped[frame - 6 : frame + 7] = np.nan
    frames, teams, _ = detect_touches_g2l(_tracking(gapped, (0.0, 0.0), (40.0, 0.0)))
    assert len(frames) == 1 and abs(int(frames[0]) - frame) <= 1
    assert int(teams[0]) == HOME

    for index in (frame - 14, frame + 14):
        broken = ball.copy()
        broken[index] = np.nan
        assert len(detect_touches_g2l(_tracking(broken, (0.0, 0.0), (40.0, 0.0)))[0]) == 0


def test_a_period_boundary_is_not_a_touch():
    """Each period is a straight line. Joining them would be a 6 m/s kink; it must not fire."""
    n = 200
    t = np.arange(n, dtype=np.float64)
    ball = np.zeros((n, 2), dtype=np.float64)
    ball[:100, 0] = (t[:100] - 50) * (6.0 / FPS)
    ball[100:, 1] = (t[100:] - 150) * (6.0 / FPS)
    periods = (Period(1, 0, 100), Period(2, 100, 200))
    frames, _, _ = detect_touches_g2l(_tracking(ball, (0.0, 0.0), (30.0, 0.0), periods))
    assert len(frames) == 0


def test_w2_or_a_missing_key_matches_today():
    tracking = generate(
        seed=1, config=SyntheticConfig(minutes_per_half=0.4, gap_minutes=0.05)
    ).tracking
    specs = (
        {},
        {"touch_detector": "g2l"},  # contact is off, so the flag adds nothing
        {"contact_inputs": True},
        {"control_memory": True, "contact_inputs": True},
        {"contact_inputs": True, "ball_path_control": True},
        {"control_memory": True, "contact_inputs": True, "ball_path_control": True},
    )
    for kwargs in specs:
        spec = TokenSpec(**kwargs)
        raw = spec.to_dict()
        raw.pop("touch_detector")
        loaded = TokenSpec.from_dict(raw)
        assert loaded.touch_detector == "w2"
        today = build_match_tokens(tracking, spec)
        assert np.array_equal(
            today, build_match_tokens(tracking, replace(spec, touch_detector="w2"))
        )
        assert np.array_equal(today, build_match_tokens(tracking, loaded))
        if not spec.contact_inputs:
            assert (
                spec.n_features
                == TokenSpec(
                    control_memory=spec.control_memory, ball_path_control=spec.ball_path_control
                ).n_features
            )


def test_g2l_adds_touch_dv_and_fills_both_touch_columns_at_the_kink():
    frame = 250
    ball = _corner(500, frame)
    tracking = _tracking(ball, (0.0, 0.0), (40.0, 0.0))
    base = TokenSpec(tokens_per_team=1, control_memory=True, contact_inputs=True)
    spec = replace(base, touch_detector="g2l")
    # Column order with control memory, contact and g2l, path off:
    # 0-13 base, 14 CTRL_TEAM, 15 CTRL_AGE, 16 BALL_DV, 17 TOUCH_TEAM, 18 CTRL_NOW,
    # 19 ACCEL, 20 TOUCH_DV. Path columns, when on, start at 21.
    assert contact_columns(spec) == (16, 17, 18, 19)
    assert touch_dv_column(spec) == 20
    assert spec.n_features == 21
    wide = replace(spec, ball_path_control=True)
    assert path_columns(wide) == (21, 22, 23, 24)
    assert wide.n_features == N_FEATURES + 2 + 4 + 1 + 4
    bare = TokenSpec(contact_inputs=True, touch_detector="g2l")
    assert contact_columns(bare) == (14, 15, 16, 17) and touch_dv_column(bare) == 18

    plain = build_match_tokens(tracking, base)
    tokens = build_match_tokens(tracking, spec)
    assert tokens.shape[-1] == plain.shape[-1] + 1 == 21
    dv, touch, now, accel = contact_columns(spec)
    touch_dv = touch_dv_column(spec)
    assert np.array_equal(tokens[..., dv], plain[..., dv])
    assert np.array_equal(tokens[..., now], plain[..., now])
    assert np.array_equal(tokens[..., accel], plain[..., accel])
    assert not tokens[:, : spec.ball_index, touch_dv].any()
    assert not tokens[:, spec.ball_index, accel].any()

    frames, teams, strength = detect_touches_g2l(tracking)
    event = int(frames[0])
    sign = 1.0 if int(teams[0]) == HOME else -1.0
    scaled = float(np.clip(strength[0] / spec.velocity_scale, 0.0, 4.0))
    assert scaled > 0.0
    ball_tok = tokens[:, spec.ball_index]
    assert ball_tok[event, touch] == pytest.approx(sign)
    assert ball_tok[event, touch_dv] == pytest.approx(scaled)
    for age in range(0, 11):
        decay = float(np.exp(-age / 5.0))
        assert ball_tok[event + age, touch] == pytest.approx(sign * decay)
        assert ball_tok[event + age, touch_dv] == pytest.approx(scaled * decay)
    assert ball_tok[event - 1, touch] == 0.0 and ball_tok[event + 11, touch] == 0.0
    assert ball_tok[event + 11, touch_dv] == 0.0
    assert not np.isnan(tokens).any()


def test_a_team_swap_negates_touch_team_and_leaves_touch_dv():
    frame = 250
    tracking = _tracking(_corner(500, frame), (0.0, 0.0), (40.0, 0.0))
    spec = TokenSpec(
        tokens_per_team=1, control_memory=True, contact_inputs=True, touch_detector="g2l"
    )
    tokens = build_match_tokens(tracking, spec)
    _dv, touch, _now, _accel = contact_columns(spec)
    touch_dv = touch_dv_column(spec)
    ball = spec.ball_index
    assert tokens[:, ball, touch].max() > 0.5
    assert tokens[:, ball, touch_dv].max() > 0.5

    swapped_sym = symmetric(tokens, spec, swap=True)
    assert np.array_equal(swapped_sym[..., touch], -tokens[..., touch])
    assert np.array_equal(swapped_sym[..., touch_dv], tokens[..., touch_dv])
    swapped = replace(tracking, slot_team=np.array([AWAY, HOME], dtype=np.int8))
    assert np.allclose(build_match_tokens(swapped, spec), swapped_sym, atol=1e-5)

    mirror = np.array([1.0, -1.0], dtype=np.float32)
    flipped_tracking = replace(
        tracking, player_xy=tracking.player_xy * mirror, ball_xy=tracking.ball_xy * mirror
    )
    flipped_sym = symmetric(tokens, spec, flip_y=True)
    assert np.array_equal(flipped_sym[..., touch], tokens[..., touch])
    assert np.array_equal(flipped_sym[..., touch_dv], tokens[..., touch_dv])
    flipped = build_match_tokens(flipped_tracking, spec)
    assert np.allclose(flipped[..., [touch, touch_dv]], tokens[..., [touch, touch_dv]], atol=1e-5)
    assert np.allclose(flipped, flipped_sym, atol=1e-5)
    # The permutation is the one a team swap applies; touch columns live on the ball, which stays.
    assert team_swap_permutation(spec)[-1] == spec.ball_index

    for flip_y, swap in ((False, False), (True, False), (False, True), (True, True)):
        once = symmetric(tokens, spec, flip_y=flip_y, swap=swap)
        assert np.array_equal(symmetric(once, spec, flip_y=flip_y, swap=swap), tokens)


@pytest.mark.parametrize("value", ["g2L", "G2L", "", "w7", 1, None, True])
def test_touch_detector_rejects_anything_else(value):
    with pytest.raises(ValueError, match="touch_detector"):
        TokenSpec(touch_detector=value)


def test_g2l_snap_moves_a_turnover_onto_the_away_event():
    frame = 250
    n = 530
    ball = _corner(n, frame)
    periods = (Period(1, 0, 500),)
    tracking = _tracking(ball, (40.0, 0.0), (0.0, 0.0), periods)
    frames, teams, _ = detect_touches_g2l(tracking)
    assert len(frames) == 1 and int(teams[0]) == AWAY
    event = int(frames[0])
    labels = np.full(n, X, dtype=np.int8)
    labels[:500] = HOME
    boundary = event + 20
    labels[boundary:500] = AWAY
    bundle = MatchBundle(tracking)
    out = touch_snap(labels, bundle, detector="g2l")
    assert int(np.flatnonzero(out[:500] == AWAY)[0]) == event
    assert (out[:event] == HOME).all() and (out[event:500] == AWAY).all()
    assert (out[500:] == X).all()
    assert np.array_equal(touch_snap(labels, bundle), touch_snap(labels, bundle, detector="w2"))
    with pytest.raises(ValueError, match="detector"):
        touch_snap(labels, bundle, detector="w5")


def _drop(mapping: dict, *keys: str) -> dict:
    return {key: value for key, value in mapping.items() if key not in keys}


def test_g2l_configs_match_the_contact_switch_plus_the_declared_keys():
    import yaml

    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_switch.yaml").read_text(encoding="utf-8")
    )
    g2l = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_switch_g2l.yaml").read_text(encoding="utf-8")
    )
    smooth = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_switch_g2l_smooth.yaml").read_text(encoding="utf-8")
    )
    assert g2l["token"] == {**base["token"], "touch_detector": "g2l"}
    assert g2l["model"] == {**base["model"], "n_features": 21}
    assert _drop(g2l, "token", "model") == _drop(base, "token", "model")
    assert smooth["token"] == g2l["token"] and smooth["model"] == g2l["model"]
    assert _drop(smooth["train"], "augment") == g2l["train"]
    assert smooth["train"]["augment"]["ball_smooth"] == {"p": 0.6, "sigma_frames": [2.0, 4.0, 6.0]}

    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig

    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_switch_g2l.yaml")
    assert run.token.touch_detector == "g2l"
    assert run.token.contact_inputs and run.token.control_memory
    assert run.model.n_features == run.token.n_features == 21


def test_smooth_config_loads_when_augment_knows_ball_smooth():
    pytest.importorskip("torch")
    from aaaag.ml.augment import AugmentMix
    from aaaag.ml.train import RunConfig

    block = {"p": 0.6, "sigma_frames": [2.0, 4.0, 6.0]}
    try:
        AugmentMix.from_dict({"ball_smooth": block})
    except (TypeError, ValueError) as exc:
        if "ball_smooth" not in str(exc):
            raise
        pytest.skip(f"AugmentMix does not know ball_smooth yet: {exc}")
    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_switch_g2l_smooth.yaml")
    assert run.token.touch_detector == "g2l"
    assert run.model.n_features == run.token.n_features == 21


def test_a_few_training_steps_match_numpy_with_g2l(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.export import main
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.np_infer import load_numpy_weights
    from aaaag.ml.spec import FEATURE_VERSION
    from aaaag.ml.train import RunConfig, TrainConfig, train

    spec = TokenSpec(control_memory=True, contact_inputs=True, touch_detector="g2l")
    assert spec.n_features == 21
    run = RunConfig(
        token=spec,
        model=ModelConfig(
            d_model=16,
            n_heads=2,
            n_blocks=1,
            dilations=(1, 2, 4),
            dropout=0.0,
            n_features=spec.n_features,
        ),
        train=TrainConfig(
            seed=3,
            steps=2,
            refresh_steps=2,
            batch_size=2,
            window=64,
            warmup_steps=1,
            lr=3e-3,
            degrade=DegradeMix(version=None),
            amp=False,
        ),
    )
    match = TrainMatch.from_bundle(
        generate(seed=4, config=SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)), spec
    )
    final = train(
        run, [match], tmp_path, device="cpu", log=lambda _line: None, resume=False, threads=1
    )
    js = tmp_path / "g2l.json"
    main(["--weights", str(final), "--out", str(js)])
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["feature_version"] == FEATURE_VERSION == "1"
    assert doc["config"]["n_features"] == 21
    assert doc["token_spec"]["touch_detector"] == "g2l"
    assert doc["token_spec"]["contact_inputs"] is True
    loaded = load_checkpoint(final)
    net = load_numpy_weights(js)
    assert net.spec == spec and net.n_features == spec.n_features == loaded.model.cfg.n_features
    tokens = build_match_tokens(match.bundle.tracking, spec)
    assert tokens.shape[-1] == 21
    window = tokens[2000:2064]
    module = loaded.model.double().eval()
    with torch.inference_mode():
        ref = module(torch.from_numpy(np.ascontiguousarray(window)).double()[None])
    got = net.forward(window[None])
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-6
    assert np.max(np.abs(got["ball"] - ref["ball"].numpy())) <= 1e-6
    tweaked = window.copy()
    tweaked[:, spec.ball_index, touch_dv_column(spec)] += 1.0
    changed = net.forward(tweaked[None])["logits"]
    assert np.max(np.abs(changed - got["logits"])) > 0.0
