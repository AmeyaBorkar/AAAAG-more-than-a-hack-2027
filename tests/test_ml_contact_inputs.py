"""Optional ball-contact columns on the tokens (`token.contact_inputs`).

The flag is off by default: those tokens match the builder with the key absent, bit for bit, and a
spec dict from before the key existed still loads. On, four columns follow the control-memory pair
(or the base 14): BALL_DV, TOUCH_TEAM, CTRL_NOW on the ball token, ACCEL on the player tokens.
`contact_touch_w` and `contact_dv_w` default to 2 and 5. Absent, or set to those values, the
tensors match the windows the columns used before the keys existed.
Numpy only, except the torch forward, export and training smoke, which skip cleanly without torch.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.evidence.nearest_player import nearest_team_codes
from aaaag.evidence.possession.control import last_controller
from aaaag.ml.geometry import symmetric, team_swap_permutation
from aaaag.ml.spec import N_FEATURES, Col, TokenSpec, contact_columns
from aaaag.ml.windows import build_match_tokens, smoothed_velocity
from aaaag.paths import CONFIGS_DIR
from aaaag.synthetic import SyntheticConfig, generate

FPS = 25
SPEED = 6.0  # m/s, the ball's constant speed on each leg of the corner
TURN = 30  # frame of the 90 degree corner
ACCEL = 4.0  # m/s^2 of the scripted accelerating slot


def _meta(n: int, periods: tuple[Period, ...] | None = None) -> MatchMeta:
    return MatchMeta("t", "synthetic", n, FPS, 105.0, 68.0, periods or (Period(1, 0, n),))


def _static(ball, *, n, periods=None, home=(0.0, 0.0), away=(20.0, 0.0)) -> Tracking:
    xy = np.zeros((n, 2, 2), dtype=np.float32)
    xy[:, 0] = np.asarray(home, dtype=np.float32)
    xy[:, 1] = np.asarray(away, dtype=np.float32)
    teams = np.array([HOME, AWAY], dtype=np.int8)
    return Tracking(_meta(n, periods), xy, teams, np.asarray(ball, dtype=np.float32))


def _turn_tracking() -> Tracking:
    """Ball runs at 6 m/s, turns 90 degrees at frame 30, with a home player 0.4 m beside it.

    A second home slot accelerates at 4 m/s^2 in y from x = -20. The away slot stands at
    (25, -10). One period covers every frame.
    """
    n = 60
    step = SPEED / FPS
    ball = np.zeros((n, 2), np.float32)
    leg = (np.arange(n) - TURN) * step
    before = np.arange(n) <= TURN
    ball[before, 0] = leg[before]
    ball[~before, 1] = leg[~before]
    players = np.zeros((n, 3, 2), np.float32)
    players[:, 0, 0] = ball[:, 0] + 0.4
    players[:, 0, 1] = ball[:, 1]
    time = np.arange(n) / FPS
    players[:, 1, 0] = -20.0
    players[:, 1, 1] = (0.5 * ACCEL * time**2).astype(np.float32)
    players[:, 2] = (25.0, -10.0)
    teams = np.array([HOME, HOME, AWAY], dtype=np.int8)
    return Tracking(_meta(n), players, teams, ball)


def _accelerating(a: float, n: int = 50) -> Tracking:
    """One home slot with constant acceleration `a` m/s^2 along y. The ball is unseen."""
    time = np.arange(n) / FPS
    players = np.zeros((n, 1, 2), np.float32)
    players[:, 0, 0] = -15.0
    players[:, 0, 1] = (0.5 * a * time**2).astype(np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    teams = np.array([HOME], dtype=np.int8)
    return Tracking(_meta(n), players, teams, ball)


def _expected_ctrl_now(codes: np.ndarray, hold: int) -> np.ndarray:
    """Independent run walk: sign once `hold` consecutive in-reach frames end at t, else 0."""
    out = np.zeros(len(codes), np.float32)
    run_team, run_len = -2, 0
    for i, code in enumerate(codes.tolist()):
        code = int(code)
        if code >= 0 and code == run_team:
            run_len += 1
        elif code >= 0:
            run_team, run_len = code, 1
        else:
            run_team, run_len = -2, 0
        if run_len >= hold and code >= 0:
            out[i] = np.float32(1.0 if code == HOME else -1.0)
    return out


def _ctrl_now(tracking: Tracking, spec: TokenSpec) -> np.ndarray:
    codes = nearest_team_codes(tracking, max_distance_m=spec.control_reach_m)
    expect = np.zeros(tracking.n_frames, np.float32)
    for period in tracking.meta.periods:
        sl = slice(period.start, period.stop)
        expect[sl] = _expected_ctrl_now(codes[sl], spec.control_hold_frames)
    return expect


def _accel_from_smoothed(xy: np.ndarray, half: int, fps: int, speed_limit: float) -> np.ndarray:
    """|centred difference| / 8 of `smoothed_velocity`, clipped, 0 where it cannot be formed."""
    vel = smoothed_velocity(xy, half, fps)
    vel = vel.copy()
    vel[np.hypot(vel[:, 0], vel[:, 1]) > speed_limit] = np.nan
    n = len(vel)
    out = np.zeros(n, np.float32)
    if n <= 2 * half:
        return out
    dt = (2 * half) / fps
    left, right = vel[: n - 2 * half], vel[2 * half :]
    finite = np.isfinite(left).all(axis=-1) & np.isfinite(right).all(axis=-1)
    delta = (right - left) / dt
    mag = np.clip(np.hypot(delta[:, 0], delta[:, 1]) / 8.0, 0.0, 4.0)
    out[half : n - half][finite] = mag[finite].astype(np.float32)
    return out


def test_flag_off_matches_the_builder_with_the_key_absent():
    tracking = generate(
        seed=1, config=SyntheticConfig(minutes_per_half=0.4, gap_minutes=0.05)
    ).tracking
    plain = build_match_tokens(tracking)
    assert np.array_equal(plain, build_match_tokens(tracking, TokenSpec()))
    assert np.array_equal(plain, build_match_tokens(tracking, TokenSpec(contact_inputs=False)))
    bare = TokenSpec().to_dict()
    bare.pop("contact_inputs")
    assert TokenSpec.from_dict(bare) == TokenSpec()
    assert np.array_equal(plain, build_match_tokens(tracking, TokenSpec.from_dict(bare)))
    remembered = TokenSpec(control_memory=True).to_dict()
    remembered.pop("contact_inputs")
    loaded = TokenSpec.from_dict(remembered)
    assert loaded == TokenSpec(control_memory=True) and not loaded.contact_inputs
    memory = build_match_tokens(tracking, TokenSpec(control_memory=True))
    assert np.array_equal(memory, build_match_tokens(tracking, loaded))

    on = build_match_tokens(tracking, TokenSpec(contact_inputs=True))
    assert on.shape[-1] == N_FEATURES + 4
    assert contact_columns(TokenSpec(contact_inputs=True)) == (14, 15, 16, 17)
    assert np.array_equal(on[..., :N_FEATURES], plain)
    both_spec = TokenSpec(control_memory=True, contact_inputs=True)
    both = build_match_tokens(tracking, both_spec)
    assert both.shape[-1] == N_FEATURES + 6 == both_spec.n_features == 20
    assert contact_columns(both_spec) == (16, 17, 18, 19)
    assert np.array_equal(both[..., : N_FEATURES + 2], memory)
    assert not np.isnan(on).any() and not np.isnan(both).any()


def test_a_corner_beside_a_home_player_fills_the_ball_columns():
    tracking = _turn_tracking()
    spec = TokenSpec(tokens_per_team=2, contact_inputs=True)
    tokens = build_match_tokens(tracking, spec)
    dv, touch, now, accel = contact_columns(spec)
    ball = spec.ball_index
    assert tokens.shape[-1] == N_FEATURES + 4
    assert not tokens[:, :ball, [dv, touch, now]].any()
    assert not tokens[:, ball, accel].any()
    assert not tokens[:, 3, accel].any()  # padded away slot
    assert not tokens[:, 3, Col.VALID].any()

    p = tracking.ball_xy.astype(np.float64)
    w = 5
    vin = (p[TURN] - p[TURN - w]) * FPS / w
    vout = (p[TURN + w] - p[TURN]) * FPS / w
    expect_dv = float(np.clip(np.linalg.norm(vout - vin) / spec.velocity_scale, 0.0, 4.0))
    assert expect_dv > 0
    assert tokens[TURN, ball, dv] == pytest.approx(expect_dv, abs=1e-6)
    straight = TURN - 10
    vin_s = (p[straight] - p[straight - w]) * FPS / w
    vout_s = (p[straight + w] - p[straight]) * FPS / w
    # Float32 positions make a "straight" leg a few micrometres off a perfect line.
    assert np.linalg.norm(vout_s - vin_s) == pytest.approx(0.0, abs=1e-4)
    assert tokens[straight, ball, dv] == pytest.approx(0.0, abs=1e-5)

    column = tokens[:, ball, touch]
    assert column[TURN] == pytest.approx(1.0, abs=1e-6)
    for age in range(0, 11):
        assert column[TURN + age] == pytest.approx(float(np.exp(-age / 5.0)), abs=1e-6)
    assert column[TURN - 1] == 0.0 and column[TURN + 11] == 0.0

    assert np.array_equal(tokens[:, ball, now], _ctrl_now(tracking, spec))
    assert tokens[TURN, ball, now] == 1.0
    assert tokens[0, ball, now] == 0.0 and tokens[1, ball, now] == 0.0
    assert tokens[2, ball, now] == 1.0

    # The decay does not cross a period boundary (the single-period column above does).
    cut_meta = replace(
        tracking.meta,
        periods=(Period(1, 0, TURN + 2), Period(2, TURN + 2, tracking.n_frames)),
    )
    cut = build_match_tokens(replace(tracking, meta=cut_meta), spec)
    assert cut[TURN + 1, ball, touch] == pytest.approx(float(np.exp(-1 / 5.0)), abs=1e-6)
    assert cut[TURN + 2, ball, touch] == 0.0


def test_ctrl_now_is_the_live_hold_and_resets_each_period():
    n = 16
    ball = np.full((n, 2), 40.0, np.float32)
    ball[0:5] = (0.0, 0.0)
    ball[10] = (0.0, 0.0)  # one frame: shorter than the hold
    tracking = _static(ball, n=n, home=(0.0, 0.0), away=(15.0, 0.0))
    spec = TokenSpec(tokens_per_team=1, contact_inputs=True, control_hold_frames=3)
    now_col = contact_columns(spec)[2]
    now = build_match_tokens(tracking, spec)[:, spec.ball_index, now_col]
    assert now.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.array_equal(now, _ctrl_now(tracking, spec))
    codes = nearest_team_codes(tracking, max_distance_m=spec.control_reach_m)
    assert last_controller(codes, 3)[10] == HOME  # memory would still say home
    assert now[10] == 0.0
    immediate = replace(spec, control_hold_frames=1)
    now1 = build_match_tokens(tracking, immediate)[:, immediate.ball_index, now_col]
    assert now1[0] == 1.0 and now1[5] == 0.0 and now1[10] == 1.0

    held = _static(
        np.zeros((8, 2), np.float32),
        n=8,
        periods=(Period(1, 0, 4), Period(2, 4, 8)),
        home=(0.0, 0.0),
        away=(20.0, 0.0),
    )
    reset = build_match_tokens(held, spec)[:, spec.ball_index, now_col]
    assert reset.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert np.array_equal(reset, _ctrl_now(held, spec))


def test_acceleration_is_the_centred_velocity_difference_and_zero_when_unseen():
    mild = _accelerating(ACCEL)
    spec = TokenSpec(tokens_per_team=2, contact_inputs=True, max_player_speed=1.0e6)
    half = max(1, int(round(spec.velocity_half_s * FPS)))
    tokens = build_match_tokens(mild, spec)
    accel = contact_columns(spec)[3]
    expect = _accel_from_smoothed(mild.player_xy[:, 0], half, FPS, spec.max_player_speed)
    assert np.allclose(tokens[:, 0, accel], expect, atol=1e-5)
    assert tokens[30, 0, accel] == pytest.approx(ACCEL / 8.0, abs=1e-4)
    assert not tokens[:, 1, accel].any()  # padded unseen slot
    assert not tokens[:, spec.ball_index, accel].any()

    fast = _accelerating(80.0, n=40)
    gated = TokenSpec(tokens_per_team=1, contact_inputs=True)  # max_player_speed 12 m/s
    open_gate = replace(gated, max_player_speed=1.0e6)
    assert build_match_tokens(fast, gated)[20, 0, contact_columns(gated)[3]] == 0.0
    clipped = build_match_tokens(fast, open_gate)
    assert clipped[20, 0, contact_columns(open_gate)[3]] == pytest.approx(4.0, abs=1e-4)
    assert not clipped[:, open_gate.ball_index, contact_columns(open_gate)[3]].any()


def test_symmetries_negate_only_the_team_signs():
    tracking = _turn_tracking()
    for spec in (
        TokenSpec(tokens_per_team=2, contact_inputs=True),
        TokenSpec(tokens_per_team=2, contact_inputs=True, control_memory=True),
    ):
        tokens = build_match_tokens(tracking, spec)
        dv, touch, now, accel = contact_columns(spec)
        ball = spec.ball_index
        assert tokens[:, ball, touch].max() > 0.5
        assert tokens[:, ball, now].max() > 0.5
        assert tokens[:, ball, dv].max() > 0.5
        assert tokens[:, :, accel].max() > 0.2

        swapped_sym = symmetric(tokens, spec, swap=True)
        perm = team_swap_permutation(spec)
        assert np.array_equal(swapped_sym[..., dv], tokens[..., dv])
        assert np.array_equal(swapped_sym[..., touch], -tokens[..., touch])
        assert np.array_equal(swapped_sym[..., now], -tokens[..., now])
        assert np.array_equal(swapped_sym[..., accel], tokens[:, perm, accel])
        swapped = replace(tracking, slot_team=(1 - tracking.slot_team).astype(np.int8))
        assert np.allclose(build_match_tokens(swapped, spec), swapped_sym, atol=1e-5)

        mirror = np.array([1.0, -1.0], np.float32)
        flipped_tracking = replace(
            tracking, player_xy=tracking.player_xy * mirror, ball_xy=tracking.ball_xy * mirror
        )
        flipped_sym = symmetric(tokens, spec, flip_y=True)
        cols = [dv, touch, now, accel]
        assert np.array_equal(flipped_sym[..., cols], tokens[..., cols])
        flipped = build_match_tokens(flipped_tracking, spec)
        assert np.allclose(flipped, flipped_sym, atol=1e-5)
        assert np.allclose(flipped[..., cols], tokens[..., cols], atol=1e-5)
        assert not np.allclose(flipped[:, ball, Col.Y], tokens[:, ball, Col.Y])

        for flip_y, swap in ((False, False), (True, False), (False, True), (True, True)):
            once = symmetric(tokens, spec, flip_y=flip_y, swap=swap)
            assert np.array_equal(symmetric(once, spec, flip_y=flip_y, swap=swap), tokens)


def _touch_nonzero(tracking: Tracking, spec: TokenSpec) -> int:
    column = build_match_tokens(tracking, spec)[:, spec.ball_index, contact_columns(spec)[1]]
    return int(np.count_nonzero(column))


def test_absent_or_default_contact_windows_match_today():
    tracking = generate(
        seed=1, config=SyntheticConfig(minutes_per_half=0.4, gap_minutes=0.05)
    ).tracking
    bare = TokenSpec.from_dict({})
    assert bare.contact_touch_w == 2 and bare.contact_dv_w == 5
    for spec in (
        TokenSpec(contact_inputs=True),
        TokenSpec(contact_inputs=True, control_memory=True),
    ):
        explicit = replace(spec, contact_touch_w=2, contact_dv_w=5)
        today = build_match_tokens(tracking, spec)
        assert np.array_equal(today, build_match_tokens(tracking, explicit))
        raw = spec.to_dict()
        raw.pop("contact_touch_w")
        raw.pop("contact_dv_w")
        loaded = TokenSpec.from_dict(raw)
        assert loaded == spec
        assert np.array_equal(today, build_match_tokens(tracking, loaded))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contact_touch_w", 0),
        ("contact_touch_w", -2),
        ("contact_dv_w", 0),
        ("contact_touch_w", True),
        ("contact_dv_w", 2.0),
    ],
)
def test_contact_windows_reject_non_integers_and_values_below_one(field, value):
    with pytest.raises(ValueError, match=field):
        TokenSpec(**{field: value})


def test_a_wider_touch_window_fires_less_on_jitter_and_keeps_a_real_turn():
    """White jitter of 0.10 m, players still: w = 7 lights TOUCH_TEAM on far fewer frames.

    The same home player beside a real 90 degree change is still detected within 4 frames.
    """
    rng = np.random.default_rng(0)
    n = 800
    ball = rng.normal(0.0, 0.10, size=(n, 2)).astype(np.float32)
    jitter = _static(ball, n=n, home=(0.0, 0.0), away=(40.0, 0.0))
    narrow = TokenSpec(tokens_per_team=1, contact_inputs=True, contact_touch_w=2)
    wide = replace(narrow, contact_touch_w=7)
    noisy = _touch_nonzero(jitter, narrow)
    quiet = _touch_nonzero(jitter, wide)
    assert noisy > 0
    assert noisy >= 5 * quiet, (noisy, quiet)

    turn = _turn_tracking()
    spec = TokenSpec(tokens_per_team=2, contact_inputs=True, contact_touch_w=7)
    column = build_match_tokens(turn, spec)[:, spec.ball_index, contact_columns(spec)[1]]
    window = column[TURN - 4 : TURN + 5]
    assert window.max() > 0, window.tolist()

    # Off the corner the two windows disagree, so this locks contact_dv_w rather than the touch.
    frame = TURN + 3
    dv_spec = replace(spec, contact_dv_w=7)
    p = turn.ball_xy.astype(np.float64)
    w = dv_spec.contact_dv_w
    vin = (p[frame] - p[frame - w]) * FPS / w
    vout = (p[frame + w] - p[frame]) * FPS / w
    expect_dv = float(np.clip(np.linalg.norm(vout - vin) / dv_spec.velocity_scale, 0.0, 4.0))
    dv_col = contact_columns(dv_spec)[0]
    got_dv = build_match_tokens(turn, dv_spec)[frame, dv_spec.ball_index, dv_col]
    assert got_dv == pytest.approx(expect_dv, abs=1e-6)
    default_dv = build_match_tokens(turn, replace(spec, contact_dv_w=5))
    assert default_dv[frame, spec.ball_index, dv_col] != pytest.approx(expect_dv, abs=1e-3)


def test_ball_dv_is_zero_when_an_endpoint_is_unseen():
    tracking = _turn_tracking()
    missing = TURN + 8  # on the outgoing leg, off the origin, so a NaN filled with 0 would move
    ball = tracking.ball_xy.copy()
    ball[missing] = np.nan
    broken = replace(tracking, ball_xy=ball)
    spec = TokenSpec(tokens_per_team=1, contact_inputs=True)
    column = build_match_tokens(broken, spec)[:, spec.ball_index, contact_columns(spec)[0]]
    assert column[missing] == 0.0  # unseen at t
    assert column[missing - 5] == 0.0  # unseen at t+w
    assert column[missing + 5] == 0.0  # unseen at t-w


def test_robust_contact_config_loads():
    import yaml

    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_switch.yaml").read_text(encoding="utf-8")
    )
    robust = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_switch_robust.yaml").read_text(encoding="utf-8")
    )
    assert robust["token"] == {**base["token"], "contact_touch_w": 7, "contact_dv_w": 7}
    assert {key: value for key, value in robust.items() if key != "token"} == {
        key: value for key, value in base.items() if key != "token"
    }
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig

    robust_run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_switch_robust.yaml")
    assert robust_run.token.contact_touch_w == 7 and robust_run.token.contact_dv_w == 7


def test_the_contact_run_is_control_memory_plus_the_flag():
    import yaml

    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_control_memory.yaml").read_text(encoding="utf-8")
    )
    got = yaml.safe_load((CONFIGS_DIR / "ml" / "team_cm_contact.yaml").read_text(encoding="utf-8"))
    assert got["token"] == {**base["token"], "contact_inputs": True}
    assert got["model"] == {**base["model"], "n_features": 20}
    assert got["train"] == base["train"]
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig

    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact.yaml")
    assert run.token.contact_inputs and run.token.control_memory
    assert run.model.n_features == run.token.n_features == 20


def test_a_few_training_steps_match_numpy_to_1e6(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.export import main
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.np_infer import load_numpy_weights
    from aaaag.ml.spec import FEATURE_VERSION
    from aaaag.ml.train import RunConfig, TrainConfig, train

    spec = TokenSpec(control_memory=True, contact_inputs=True)
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
    js = tmp_path / "contact.json"
    main(["--weights", str(final), "--out", str(js)])
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["feature_version"] == FEATURE_VERSION == "1"
    assert doc["config"]["n_features"] == 20
    assert doc["token_spec"]["contact_inputs"] is True
    assert doc["token_spec"]["control_memory"] is True
    loaded = load_checkpoint(final)
    net = load_numpy_weights(js)
    assert net.spec == spec and net.n_features == spec.n_features == loaded.model.cfg.n_features
    tokens = build_match_tokens(match.bundle.tracking, spec)
    assert tokens.shape[-1] == 20
    assert np.any(tokens[..., list(contact_columns(spec))] != 0)
    window = tokens[2000:2064]
    module = loaded.model.double().eval()
    with torch.inference_mode():
        ref = module(torch.from_numpy(np.ascontiguousarray(window)).double()[None])
    got = net.forward(window[None])
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-6
    assert np.max(np.abs(got["ball"] - ref["ball"].numpy())) <= 1e-6
    tweaked = window.copy()
    tweaked[:, spec.ball_index, contact_columns(spec)[1]] += 1.0
    changed = net.forward(tweaked[None])["logits"]
    assert np.max(np.abs(changed - got["logits"])) > 0.0
