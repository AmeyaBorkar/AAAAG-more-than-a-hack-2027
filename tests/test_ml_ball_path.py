"""Optional ball-path control columns (`token.ball_path_control`).

The flag is off by default: those tokens match the builder with the key absent, bit for bit, with
contact inputs on or off. On, four columns follow the contact columns (or the control-memory pair,
or the base 14): PATH_SIGN, PATH_OUT, PATH_T on the ball token, PPCF on the player tokens.
Numpy only, except the torch forward, export and training smoke, which skip cleanly without torch.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.ml.geometry import attack_sign, symmetric, team_swap_permutation
from aaaag.ml.spec import N_FEATURES, Col, TokenSpec, path_columns
from aaaag.ml.windows import build_match_tokens, smoothed_velocity
from aaaag.paths import CONFIGS_DIR
from aaaag.synthetic import SyntheticConfig, generate

FPS = 25
# Ball rolling towards a home player standing on its line. Metres and seconds.
APPROACH_SPEED = 4.0
HOME_X = 10.0
AWAY_X = 3.5


def _meta(n: int, fps: int = FPS, periods: tuple[Period, ...] | None = None) -> MatchMeta:
    return MatchMeta("t", "synthetic", n, fps, 105.0, 68.0, periods or (Period(1, 0, n),))


def _tracking(
    ball: np.ndarray,
    players: np.ndarray,
    teams: np.ndarray,
    *,
    fps: int = FPS,
    periods: tuple[Period, ...] | None = None,
) -> Tracking:
    ball = np.asarray(ball, dtype=np.float32)
    players = np.asarray(players, dtype=np.float32)
    return Tracking(
        _meta(len(ball), fps, periods),
        players,
        np.asarray(teams, dtype=np.int8),
        ball,
    )


def _constant_slots(n: int, slots: list[tuple[float, float]]) -> np.ndarray:
    xy = np.zeros((n, len(slots), 2), dtype=np.float32)
    xy[:] = np.asarray(slots, dtype=np.float32)
    return xy


def _line(n: int, x0: float, speed: float, fps: int, y: float = 0.0) -> np.ndarray:
    """Exact when `speed / fps` is a binary fraction (the gap test uses 0.5)."""
    frames = np.arange(n, dtype=np.float64)
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = (x0 + speed * frames / fps).astype(np.float32)
    ball[:, 1] = np.float32(y)
    return ball


def _approach(n: int = 70) -> Tracking:
    """A ball at 4 m/s from x=0 towards one home player at x=10. No away team, so no rotation."""
    players = _constant_slots(n, [(HOME_X, 0.0)])
    return _tracking(_line(n, 0.0, APPROACH_SPEED, FPS), players, np.array([HOME]))


def _approach_with_away(n: int = 70) -> Tracking:
    """The same ball and home player, plus an away player standing much closer to the path."""
    players = _constant_slots(n, [(HOME_X, 0.0), (AWAY_X, 0.0)])
    return _tracking(_line(n, 0.0, APPROACH_SPEED, FPS), players, np.array([HOME, AWAY]))


def _off_pitch() -> Tracking:
    """Ball at x=50 moving at 25 m/s off the +x end. One home player far from that path."""
    n, f0, speed = 60, 30, 25.0
    frames = np.arange(n, dtype=np.float64)
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = (50.0 + speed * (frames - f0) / FPS).astype(np.float32)
    players = _constant_slots(n, [(-30.0, 20.0)])
    return _tracking(ball, players, np.array([HOME]))


def _add_unseen(tracking: Tracking) -> Tracking:
    """An extra away slot that is never observed, so it must take none of the control."""
    n = tracking.n_frames
    missing = np.full((n, 1, 2), np.nan, dtype=np.float32)
    players = np.concatenate([tracking.player_xy, missing], axis=1)
    teams = np.concatenate([tracking.slot_team, np.array([AWAY], dtype=np.int8)])
    return replace(tracking, player_xy=players, slot_team=teams, _fingerprint=None)


def _frames_with_a_path(tracking: Tracking, hold_s: float = 1.0) -> np.ndarray:
    """Frames whose ball was seen at most `hold_s` earlier inside the same period."""
    seen = tracking.ball_seen()
    on = np.zeros(tracking.n_frames, dtype=bool)
    max_age = int(np.floor(hold_s * tracking.meta.fps + 1e-9))
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        index = np.arange(hi - lo)
        last = np.maximum.accumulate(np.where(seen[lo:hi], index, -1))
        age = np.where(last >= 0, index - last, max_age + 1)
        on[lo:hi] = (last >= 0) & (age <= max_age)
    return on


def _mass(tokens: np.ndarray, spec: TokenSpec) -> np.ndarray:
    sign, out, _time, ppcf = path_columns(spec)
    del sign
    return tokens[..., ppcf].sum(axis=1) + tokens[:, spec.ball_index, out]


def _reference(
    p0: np.ndarray,
    v0: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    teams: np.ndarray,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> tuple[float, float, float, np.ndarray]:
    """One frame of the spec, in plain loops: sign, P_out, PATH_T, per-slot PPCF."""
    coef = np.pi / (np.sqrt(3.0) * 0.45)
    x_lim, y_lim = pitch[0] / 2.0 + 0.5, pitch[1] / 2.0 + 0.5
    observed = np.isfinite(position[:, 0])
    reaction = position + 0.7 * np.nan_to_num(velocity, nan=0.0)
    remaining = 1.0
    ppcf = np.zeros(len(position), dtype=np.float64)
    path_out = 0.0
    sum_q = 0.0
    sum_qt = 0.0
    stopped = False
    for k in range(26):
        tau = 0.1 * k
        point = p0 + v0 * tau
        if abs(float(point[0])) > x_lim or abs(float(point[1])) > y_lim:
            path_out = remaining
            remaining = 0.0
            stopped = True
            break
        prob = np.zeros(len(position), dtype=np.float64)
        for j in range(len(position)):
            if not observed[j]:
                continue
            dist = float(np.hypot(*(point - reaction[j])))
            gap = coef * (tau - (0.7 + dist / 5.0))
            if gap >= 0.0:
                prob[j] = 1.0 / (1.0 + np.exp(-gap))
            else:
                safe = np.exp(gap)
                prob[j] = safe / (1.0 + safe)
        total = float(prob.sum())
        hazard = 4.3 * total
        if hazard == 0.0:
            gained = 0.0
            share = np.zeros_like(prob)
        else:
            gained = remaining * (1.0 - np.exp(-hazard * 0.1))
            share = prob / total
        ppcf += gained * share
        sum_q += gained
        sum_qt += gained * tau
        remaining -= gained
    if not stopped:
        path_out = remaining
    path_t = 0.0 if sum_q == 0.0 else (sum_qt / max(sum_q, 1e-6)) / 2.5
    sign = float(ppcf[teams == HOME].sum() - ppcf[teams == AWAY].sum())
    return sign, float(path_out), float(path_t), ppcf


def _builder_inputs(tracking: Tracking, frame: int):
    """Position and gated velocity at `frame`, the same way the builder reads a seen ball."""
    assert np.all(attack_sign(tracking) == 1.0)
    fps = tracking.meta.fps
    half = max(1, int(round(0.2 * fps)))
    ball_v = smoothed_velocity(tracking.ball_xy, half, fps)
    player_v = smoothed_velocity(tracking.player_xy, half, fps)
    for velocity, limit in ((ball_v, 45.0), (player_v, 12.0)):
        velocity[np.hypot(velocity[..., 0], velocity[..., 1]) > limit] = np.nan
    p0 = tracking.ball_xy[frame].astype(np.float64)
    v0 = np.nan_to_num(ball_v[frame].astype(np.float64), nan=0.0)
    pos = tracking.player_xy[frame].astype(np.float64)
    vel = player_v[frame].astype(np.float64)
    return p0, v0, pos, vel


def test_flag_off_matches_the_builder_with_the_key_absent():
    tracking = generate(
        seed=1, config=SyntheticConfig(minutes_per_half=0.4, gap_minutes=0.05)
    ).tracking
    plain = build_match_tokens(tracking)
    assert np.array_equal(plain, build_match_tokens(tracking, TokenSpec()))
    bare = TokenSpec().to_dict()
    bare.pop("ball_path_control")
    assert TokenSpec.from_dict(bare) == TokenSpec()
    assert np.array_equal(plain, build_match_tokens(tracking, TokenSpec.from_dict(bare)))

    for kwargs in (
        {},
        {"contact_inputs": True},
        {"control_memory": True},
        {"control_memory": True, "contact_inputs": True},
    ):
        spec = TokenSpec(**kwargs)
        raw = spec.to_dict()
        raw.pop("ball_path_control")
        loaded = TokenSpec.from_dict(raw)
        assert loaded == spec and not loaded.ball_path_control
        built = build_match_tokens(tracking, spec)
        assert np.array_equal(built, build_match_tokens(tracking, loaded))
        assert np.array_equal(
            built, build_match_tokens(tracking, replace(spec, ball_path_control=False))
        )


def test_flag_on_adds_four_columns():
    tracking = generate(
        seed=2, config=SyntheticConfig(minutes_per_half=0.3, gap_minutes=0.05)
    ).tracking
    cases = (
        ({}, N_FEATURES, (14, 15, 16, 17)),
        ({"control_memory": True}, N_FEATURES + 2, (16, 17, 18, 19)),
        ({"contact_inputs": True}, N_FEATURES + 4, (18, 19, 20, 21)),
        ({"control_memory": True, "contact_inputs": True}, N_FEATURES + 6, (20, 21, 22, 23)),
    )
    for kwargs, width, columns in cases:
        off = TokenSpec(**kwargs)
        on = TokenSpec(**kwargs, ball_path_control=True)
        base = build_match_tokens(tracking, off)
        wide = build_match_tokens(tracking, on)
        assert base.shape[-1] == width == off.n_features
        assert wide.shape[-1] == width + 4 == on.n_features
        assert path_columns(on) == columns
        assert np.array_equal(wide[..., :width], base)
        assert not np.isnan(wide).any()


def test_a_ball_rolling_towards_a_home_player():
    tracking = _approach()
    spec = TokenSpec(tokens_per_team=1, ball_path_control=True)
    tokens = build_match_tokens(tracking, spec)
    sign, _out, time, ppcf = path_columns(spec)
    ball = spec.ball_index
    assert tokens.shape[-1] == N_FEATURES + 4
    assert not tokens[:, :ball, sign].any()
    assert not tokens[:, ball, ppcf].any()
    # Centred velocity is valid on these frames; the ball is still short of the player.
    frames = (10, 25, 40)
    assert all(tokens[frame, ball, sign] > 0.5 for frame in frames)
    times = [float(tokens[frame, ball, time]) for frame in frames]
    assert times[0] > times[1] > times[2]
    for frame in frames:
        assert tokens[frame, 0, Col.VALID] == 1.0
        assert tokens[frame, 0, Col.IS_AWAY] == 0.0
        assert tokens[frame, 0, ppcf] == tokens[frame, :, ppcf].max()
        assert tokens[frame, 0, ppcf] > 0.5


def test_an_away_player_closer_to_the_path_takes_the_sign():
    tracking = _approach_with_away()
    spec = TokenSpec(tokens_per_team=1, ball_path_control=True)
    tokens = build_match_tokens(tracking, spec)
    sign, _out, _time, ppcf = path_columns(spec)
    ball = spec.ball_index
    # Frame 15: the ball is just short of the away player and still far from home.
    assert tokens[15, ball, sign] < 0.0
    assert tokens[15, spec.tokens_per_team, ppcf] > tokens[15, 0, ppcf]


def test_a_ball_rolling_off_the_pitch_is_mostly_out():
    tracking = _off_pitch()
    spec = TokenSpec(tokens_per_team=1, ball_path_control=True)
    tokens = build_match_tokens(tracking, spec)
    _sign, out, _time, _ppcf = path_columns(spec)
    frame = 30  # x = 50, on the pitch, 25 m/s towards the goal line
    assert tokens[frame, spec.ball_index, Col.VX] == pytest.approx(25.0 / 8.0, abs=1e-4)
    assert tokens[frame, spec.ball_index, out] > 0.8


def test_a_short_gap_keeps_the_extrapolated_state():
    fps = 10  # 0.5 s, 1.0 s and 1.5 s are whole frames; 0.5 m per frame is exact in float32
    n, last = 70, 30
    speed = 5.0
    ball = _line(n, -5.0, speed, fps)
    players = _constant_slots(n, [(12.0, 0.0)])
    teams = np.array([HOME])
    full = _tracking(ball, players, teams, fps=fps)
    hidden = ball.copy()
    hidden[last + 1 :] = np.nan
    gappy = _tracking(hidden, players, teams, fps=fps)
    spec = TokenSpec(tokens_per_team=1, ball_path_control=True)
    cols = list(path_columns(spec))
    got = build_match_tokens(gappy, spec)
    expect = build_match_tokens(full, spec)
    for age, kept in ((5, True), (10, True), (11, False), (15, False)):
        frame = last + age
        if kept:
            assert np.allclose(got[frame, :, cols], expect[frame, :, cols], atol=1e-5)
            assert np.max(np.abs(expect[frame, :, cols])) > 0.2
        else:
            assert np.all(got[frame, :, cols] == 0.0)
            assert np.max(np.abs(expect[frame, :, cols])) > 0.2

    # A new period does not inherit the previous period's last ball, even one frame later.
    split = (Period(1, 0, last + 1), Period(2, last + 1, n))
    cut = _tracking(hidden, players, teams, fps=fps, periods=split)
    one = build_match_tokens(gappy, spec)
    two = build_match_tokens(cut, spec)
    assert np.max(np.abs(one[last + 1, :, cols])) > 0.2
    assert np.all(two[last + 1, :, cols] == 0.0)


def test_symmetries_negate_only_the_path_sign():
    tracking = _approach_with_away(n=50)
    # A nonzero y, so a mirror is not the identity on the coordinate columns.
    shifted = replace(
        tracking,
        player_xy=tracking.player_xy + np.array([0.0, 4.0], np.float32),
        ball_xy=tracking.ball_xy + np.array([0.0, 1.0], np.float32),
        _fingerprint=None,
    )
    specs = (
        TokenSpec(tokens_per_team=1, ball_path_control=True),
        TokenSpec(
            tokens_per_team=1, ball_path_control=True, contact_inputs=True, control_memory=True
        ),
    )
    for spec in specs:
        tokens = build_match_tokens(shifted, spec)
        sign, out, time, ppcf = path_columns(spec)
        cols = [sign, out, time, ppcf]
        assert np.max(np.abs(tokens[:, spec.ball_index, sign])) > 0.2

        swapped_sym = symmetric(tokens, spec, swap=True)
        perm = team_swap_permutation(spec)
        assert np.array_equal(swapped_sym[..., sign], -tokens[..., sign])
        assert np.array_equal(swapped_sym[..., out], tokens[..., out])
        assert np.array_equal(swapped_sym[..., time], tokens[..., time])
        assert np.array_equal(swapped_sym[..., ppcf], tokens[:, perm, ppcf])
        swapped = replace(
            shifted, slot_team=(1 - shifted.slot_team).astype(np.int8), _fingerprint=None
        )
        assert np.allclose(build_match_tokens(swapped, spec), swapped_sym, atol=1e-5)

        flipped_sym = symmetric(tokens, spec, flip_y=True)
        assert np.array_equal(flipped_sym[..., cols], tokens[..., cols])
        mirror = np.array([1.0, -1.0], np.float32)
        flipped_tracking = replace(
            shifted,
            player_xy=shifted.player_xy * mirror,
            ball_xy=shifted.ball_xy * mirror,
            _fingerprint=None,
        )
        flipped = build_match_tokens(flipped_tracking, spec)
        assert np.allclose(flipped, flipped_sym, atol=1e-5)
        assert np.allclose(flipped[..., cols], tokens[..., cols], atol=1e-5)
        assert not np.allclose(
            flipped[:, spec.ball_index, Col.Y], tokens[:, spec.ball_index, Col.Y]
        )

        for flip_y, swap in ((False, False), (True, False), (False, True), (True, True)):
            once = symmetric(tokens, spec, flip_y=flip_y, swap=swap)
            assert np.array_equal(symmetric(once, spec, flip_y=flip_y, swap=swap), tokens)


def test_control_probabilities_sum_to_one():
    scenes = (
        _approach(),
        _approach_with_away(),
        _off_pitch(),
        _add_unseen(_approach()),
    )
    # The gap scene is built inline so the 1.5 s hole is part of the same assertion.
    fps, n, last = 10, 70, 30
    ball = _line(n, -5.0, 5.0, fps)
    hidden = ball.copy()
    hidden[last + 1 :] = np.nan
    gappy = _tracking(hidden, _constant_slots(n, [(12.0, 0.0)]), np.array([HOME]), fps=fps)
    for tracking in (*scenes, gappy):
        spec = TokenSpec(tokens_per_team=4, ball_path_control=True)
        tokens = build_match_tokens(tracking, spec)
        mass = _mass(tokens, spec)
        on = _frames_with_a_path(tracking)
        assert on.any()
        assert np.allclose(mass[on], 1.0, atol=1e-5)
        assert np.all(mass[~on] == 0.0)
        assert not np.isnan(tokens).any()


def test_a_direct_integration_matches_the_builder():
    n, fps, frame = 50, 25, 25
    frames = np.arange(n, dtype=np.float64)
    players = np.zeros((n, 3, 2), dtype=np.float32)
    players[:, 0, 0] = (-8.0 + 2.0 * frames / fps).astype(np.float32)  # home, running
    players[:, 1] = np.nan  # unseen home slot
    players[:, 2] = (8.0, -1.0)  # away, standing off the line
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = (4.0 * (frames - frame) / fps).astype(np.float32)
    ball[:, 1] = 0.5
    teams = np.array([HOME, HOME, AWAY])
    tracking = _tracking(ball, players, teams, fps=fps)
    spec = TokenSpec(tokens_per_team=2, ball_path_control=True)
    tokens = build_match_tokens(tracking, spec)
    p0, v0, pos, vel = _builder_inputs(tracking, frame)
    ref_sign, ref_out, ref_t, ref_ppcf = _reference(p0, v0, pos, vel, teams)
    sign, out, time, ppcf = path_columns(spec)
    ball_i = spec.ball_index
    assert tokens[frame, ball_i, sign] == pytest.approx(ref_sign, abs=1e-5)
    assert tokens[frame, ball_i, out] == pytest.approx(ref_out, abs=1e-5)
    assert tokens[frame, ball_i, time] == pytest.approx(ref_t, abs=1e-5)
    assert tokens[frame, 0, ppcf] == pytest.approx(ref_ppcf[0], abs=1e-5)
    assert tokens[frame, spec.tokens_per_team, ppcf] == pytest.approx(ref_ppcf[2], abs=1e-5)
    assert ref_ppcf[1] == 0.0
    assert tokens[frame, 1, ppcf] == 0.0  # the unseen slot is not a valid token
    assert ref_sign == pytest.approx(float(ref_ppcf[0] - ref_ppcf[2]), abs=1e-12)
    assert ref_ppcf.sum() + ref_out == pytest.approx(1.0, abs=1e-12)

    off = _off_pitch()
    off_spec = TokenSpec(tokens_per_team=1, ball_path_control=True)
    off_tokens = build_match_tokens(off, off_spec)
    p0, v0, pos, vel = _builder_inputs(off, 30)
    ref_sign, ref_out, ref_t, ref_ppcf = _reference(p0, v0, pos, vel, off.slot_team)
    sign, out, time, ppcf = path_columns(off_spec)
    assert off_tokens[30, off_spec.ball_index, out] == pytest.approx(ref_out, abs=1e-5)
    assert off_tokens[30, off_spec.ball_index, sign] == pytest.approx(ref_sign, abs=1e-5)
    assert off_tokens[30, off_spec.ball_index, time] == pytest.approx(ref_t, abs=1e-5)
    assert off_tokens[30, 0, ppcf] == pytest.approx(ref_ppcf[0], abs=1e-5)
    assert ref_out > 0.8


def test_the_path_run_is_the_contact_run_plus_the_flag():
    import yaml

    base = yaml.safe_load((CONFIGS_DIR / "ml" / "team_cm_contact.yaml").read_text(encoding="utf-8"))
    got = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_path.yaml").read_text(encoding="utf-8")
    )
    assert got["token"] == {**base["token"], "ball_path_control": True}
    assert got["model"] == {**base["model"], "n_features": 24}
    assert got["train"] == base["train"]
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig

    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_path.yaml")
    assert run.token.ball_path_control and run.token.contact_inputs and run.token.control_memory
    assert run.model.n_features == run.token.n_features == 24


def test_a_few_training_steps_match_numpy_to_1e6(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.export import main
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.np_infer import load_numpy_weights, predict_logits_np
    from aaaag.ml.spec import FEATURE_VERSION
    from aaaag.ml.train import RunConfig, TrainConfig, train

    spec = TokenSpec(control_memory=True, contact_inputs=True, ball_path_control=True)
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
    js = tmp_path / "path.json"
    main(["--weights", str(final), "--out", str(js)])
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["feature_version"] == FEATURE_VERSION == "1"
    assert doc["config"]["n_features"] == 24
    assert doc["token_spec"]["ball_path_control"] is True
    assert doc["token_spec"]["contact_inputs"] is True
    assert doc["token_spec"]["control_memory"] is True
    loaded = load_checkpoint(final)
    net = load_numpy_weights(js)
    assert net.spec == spec and net.n_features == spec.n_features == loaded.model.cfg.n_features
    tokens = build_match_tokens(match.bundle.tracking, spec)
    assert tokens.shape[-1] == 24
    assert np.any(tokens[..., list(path_columns(spec))] != 0)
    window = tokens[2000:2064]
    module = loaded.model.double().eval()
    with torch.inference_mode():
        ref = module(torch.from_numpy(np.ascontiguousarray(window)).double()[None])
    got = net.forward(window[None])
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-6
    assert np.max(np.abs(got["ball"] - ref["ball"].numpy())) <= 1e-6
    tweaked = window.copy()
    tweaked[:, spec.ball_index, path_columns(spec)[0]] += 1.0
    changed = net.forward(tweaked[None])["logits"]
    assert np.max(np.abs(changed - got["logits"])) > 0.0
    prediction = predict_logits_np([js], match.bundle.tracking)
    played = match.bundle.meta.played_mask()
    assert prediction.logits.shape == (match.bundle.tracking.n_frames, 3)
    assert np.isfinite(prediction.logits[played & prediction.opinion]).all()
