"""Optional last-controller memory on the ball token (E14).

The flag is off by default: those tokens match the base builder bit for bit, and a spec dict from
before the keys existed still loads. On, the ball token gains the control rule's team and the age
of that run. Numpy only, except the torch forward, export and training smoke, which skip cleanly
without torch.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchBundle, MatchMeta, Period, Tracking
from aaaag.evidence.nearest_player import nearest_team_codes
from aaaag.evidence.possession.control import control_possession
from aaaag.ml.geometry import symmetric
from aaaag.ml.np_infer import dump_numpy_weights, load_numpy_weights, parameter_layout
from aaaag.ml.spec import FEATURE_VERSION, N_FEATURES, Col, TokenSpec
from aaaag.ml.windows import build_match_tokens
from aaaag.paths import CONFIGS_DIR

FPS = 25
OLD_SPEC = {
    "tokens_per_team": 11,
    "velocity_half_s": 0.2,
    "max_player_speed": 12.0,
    "max_ball_speed": 45.0,
    "velocity_scale": 8.0,
    "ref_scale_m": 20.0,
    "gap_cap_s": 30.0,
}


def _age(frames: float, fps: int = FPS) -> float:
    """log1p(seconds) / log1p(60), the CTRL_AGE formula, uncapped."""
    return float(np.log1p(frames / fps) / np.log1p(60.0))


def _run_starts(codes: np.ndarray, hold: int) -> np.ndarray:
    """Start index of the latest qualifying run, else -1. A run walk, not the builder's scan."""
    n = len(codes)
    starts = np.full(n, -1, dtype=np.int64)
    latest = -1
    i = 0
    while i < n:
        j = i + 1
        while j < n and codes[j] == codes[i]:
            j += 1
        if codes[i] >= 0 and j - i >= hold:
            latest = i
        if latest >= 0:
            starts[i:j] = latest
        i = j
    return starts


def _expected_age(codes: np.ndarray, hold: int, fps: int) -> np.ndarray:
    starts = _run_starts(codes, hold)
    seconds = (np.arange(len(codes)) - starts) / fps
    age = np.log1p(np.maximum(seconds, 0.0)) / np.log1p(60.0)
    return np.where(starts >= 0, np.minimum(age, 1.0), 0.0).astype(np.float32)


def _static(ball, *, n, fps=FPS, periods=None, home=(-10.0, 4.0), away=(10.0, -6.0)) -> Tracking:
    xy = np.zeros((n, 2, 2), dtype=np.float32)
    xy[:, 0] = np.asarray(home, dtype=np.float32)
    xy[:, 1] = np.asarray(away, dtype=np.float32)
    teams = np.array([HOME, AWAY], dtype=np.int8)
    meta = MatchMeta("t", "synthetic", n, fps, 105.0, 68.0, periods or (Period(1, 0, n),))
    return Tracking(meta, xy, teams, np.asarray(ball, dtype=np.float32))


def _scripted() -> Tracking:
    """Home at (-10, 4), away at (10, -6). One possession change, a flicker, a gap, a new period."""
    n = 50
    home = np.array([-10.0, 4.0], np.float32)
    away = np.array([10.0, -6.0], np.float32)
    far = np.array([40.0, 0.0], np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    ball[0:5] = far
    ball[5:13] = home
    ball[13:15] = away  # two frames: shorter than the hold, so it changes nothing
    ball[18:30] = away
    ball[30:43] = far  # 30:40 is between periods; 40:43 is the new period, still out of reach
    ball[43:50] = home
    return _static(ball, n=n, periods=(Period(1, 0, 30), Period(2, 40, 50)), home=home, away=away)


def _tiny_config(n_features: int, tokens_per_team: int = 2) -> dict:
    return {
        "tokens_per_team": tokens_per_team,
        "n_features": n_features,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1, 2],
        "kernel": 3,
        "dropout": 0.0,
    }


def _dump(path, config, spec, seed):
    rng = np.random.default_rng(seed)
    arrays = {
        name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in parameter_layout(config)
    }
    dump_numpy_weights(
        path,
        arch="team_net",
        config=config,
        token_spec=spec,
        arrays=arrays,
        feature_version=FEATURE_VERSION,
    )


def test_flag_off_matches_today_and_old_specs_round_trip(small_bundle):
    tracking = small_bundle.tracking
    default = build_match_tokens(tracking)
    assert np.array_equal(default, build_match_tokens(tracking, TokenSpec()))
    loaded = TokenSpec.from_dict(OLD_SPEC)
    assert loaded == TokenSpec() and loaded.n_features == N_FEATURES and not loaded.control_memory
    assert np.array_equal(default, build_match_tokens(tracking, loaded))
    assert default.shape == (tracking.n_frames, TokenSpec().n_tokens, N_FEATURES)
    assert FEATURE_VERSION == "1"
    wide = TokenSpec(control_memory=True, control_reach_m=1.5, control_hold_frames=4)
    assert TokenSpec.from_dict(wide.to_dict()) == wide
    assert TokenSpec.from_dict(TokenSpec().to_dict()) == TokenSpec()
    on = build_match_tokens(tracking, TokenSpec(control_memory=True))
    assert on.shape[-1] == N_FEATURES + 2
    assert np.array_equal(on[..., :N_FEATURES], default)  # the base columns are untouched
    assert not np.isnan(on).any()
    ball = TokenSpec().ball_index
    assert not on[:, :ball, Col.CTRL_TEAM].any() and not on[:, :ball, Col.CTRL_AGE].any()


def test_control_columns_match_the_rule_and_an_independent_run_start():
    tracking = _scripted()
    spec = TokenSpec(tokens_per_team=1, control_memory=True)
    tokens = build_match_tokens(tracking, spec)
    off = build_match_tokens(tracking, TokenSpec(tokens_per_team=1))
    assert tokens.shape[-1] == N_FEATURES + 2
    assert np.array_equal(tokens[..., :N_FEATURES], off)
    assert not np.isnan(tokens).any()
    assert np.array_equal(tokens, build_match_tokens(tracking, spec))
    ball = spec.ball_index
    assert not tokens[:, :ball, [Col.CTRL_TEAM, Col.CTRL_AGE]].any()
    assert not tokens[~tracking.meta.played_mask()].any()

    proba = control_possession(
        MatchBundle(tracking), reach_m=spec.control_reach_m, hold_frames=spec.control_hold_frames
    ).proba
    expect_team = np.zeros(tracking.n_frames, np.float32)
    expect_team[proba[:, 0] == 1] = 1.0
    expect_team[proba[:, 1] == 1] = -1.0
    assert np.array_equal(tokens[:, ball, Col.CTRL_TEAM], expect_team)

    codes = nearest_team_codes(tracking, max_distance_m=spec.control_reach_m)
    assert codes[[4, 5, 12, 13, 14, 15, 18, 29, 40, 43]].tolist() == [
        -1,
        HOME,
        HOME,
        AWAY,
        AWAY,
        -1,
        AWAY,
        AWAY,
        -1,
        HOME,
    ]
    expect_age = np.zeros(tracking.n_frames, np.float32)
    for period in tracking.meta.periods:
        sl = slice(period.start, period.stop)
        expect_age[sl] = _expected_age(codes[sl], spec.control_hold_frames, FPS)
    assert np.allclose(tokens[:, ball, Col.CTRL_AGE], expect_age, atol=1e-6, rtol=0)

    column = tokens[:, ball]
    assert column[4, Col.CTRL_TEAM] == 0 and column[4, Col.CTRL_AGE] == 0  # before any control
    assert column[5, Col.CTRL_TEAM] == 1 and column[5, Col.CTRL_AGE] == 0  # run dated from frame 5
    assert column[14, Col.CTRL_TEAM] == 1  # the two-frame away touch does not take the ball
    assert column[14, Col.CTRL_AGE] == pytest.approx(_age(9), abs=1e-6)
    assert column[17, Col.CTRL_TEAM] == 1  # held through the unseen-ball gap
    assert column[17, Col.CTRL_AGE] == pytest.approx(_age(12), abs=1e-6)
    assert column[18, Col.CTRL_TEAM] == -1 and column[18, Col.CTRL_AGE] == 0
    assert column[29, Col.CTRL_AGE] == pytest.approx(_age(11), abs=1e-6)
    assert column[40, Col.CTRL_TEAM] == 0 and column[40, Col.CTRL_AGE] == 0  # new period, no memory
    assert column[43, Col.CTRL_TEAM] == 1 and column[43, Col.CTRL_AGE] == 0
    assert column[49, Col.CTRL_AGE] == pytest.approx(_age(6), abs=1e-6)


def test_a_later_run_by_the_same_team_restarts_the_age():
    n = 10
    ball = np.full((n, 2), 30.0, np.float32)
    ball[:, 1] = 0.0
    ball[0:3] = (0.0, 0.0)
    ball[5:10] = (0.0, 0.0)
    tracking = _static(ball, n=n, home=(0.0, 0.0), away=(20.0, 0.0))
    spec = TokenSpec(tokens_per_team=1, control_memory=True)
    column = build_match_tokens(tracking, spec)[:, spec.ball_index]
    assert column[4, Col.CTRL_TEAM] == 1  # held through the gap, clock still running
    assert column[4, Col.CTRL_AGE] == pytest.approx(_age(4), abs=1e-6)
    assert column[5, Col.CTRL_TEAM] == 1 and column[5, Col.CTRL_AGE] == 0.0  # same team, new run
    assert column[9, Col.CTRL_AGE] == pytest.approx(_age(4), abs=1e-6)


def test_the_age_caps_at_sixty_seconds():
    n = 60 * FPS + 10
    ball = np.zeros((n, 2), np.float32)
    tracking = _static(ball, n=n, home=(0.0, 0.0), away=(20.0, 0.0))
    spec = TokenSpec(tokens_per_team=1, control_memory=True)
    age = build_match_tokens(tracking, spec)[:, spec.ball_index, Col.CTRL_AGE]
    assert age[0] == 0.0
    assert age[60 * FPS - 1] < 1.0
    assert age[60 * FPS] == pytest.approx(1.0, abs=1e-6)
    assert age[-1] == pytest.approx(1.0, abs=1e-6)


def test_reach_and_hold_follow_the_spec():
    n = 8
    ball = np.tile(np.array([1.5, 0.0], np.float32), (n, 1))
    tracking = _static(ball, n=n, home=(0.0, 0.0), away=(15.0, 0.0))
    tight = TokenSpec(tokens_per_team=1, control_memory=True, control_reach_m=1.0)
    wide = TokenSpec(tokens_per_team=1, control_memory=True, control_reach_m=2.0)
    patient = replace(wide, control_hold_frames=20)
    assert not build_match_tokens(tracking, tight)[:, tight.ball_index, Col.CTRL_TEAM].any()
    got = build_match_tokens(tracking, wide)
    assert (got[:, wide.ball_index, Col.CTRL_TEAM] == 1).all()
    assert not build_match_tokens(tracking, patient)[:, patient.ball_index, Col.CTRL_TEAM].any()
    proba = control_possession(MatchBundle(tracking), reach_m=2.0, hold_frames=3).proba
    assert (proba[:, 0] == 1).all()


def test_symmetries_with_control_memory_on(small_bundle):
    spec = TokenSpec(tokens_per_team=1, control_memory=True)
    tracking = _scripted()
    tokens = build_match_tokens(tracking, spec)
    played = tracking.meta.played_mask()
    ball = spec.ball_index
    seen_teams = set(np.unique(tokens[played, ball, Col.CTRL_TEAM]).tolist())
    assert seen_teams == {-1.0, 0.0, 1.0}

    swapped = replace(tracking, slot_team=(1 - tracking.slot_team).astype(np.int8))
    got = build_match_tokens(swapped, spec)
    assert np.allclose(got[played], symmetric(tokens, spec, swap=True)[played], atol=1e-6)
    assert np.allclose(got[played, ball, Col.CTRL_TEAM], -tokens[played, ball, Col.CTRL_TEAM])
    assert np.allclose(got[played, ball, Col.CTRL_AGE], tokens[played, ball, Col.CTRL_AGE])

    mirror = np.array([1.0, -1.0], np.float32)
    flipped_tracking = replace(
        tracking, player_xy=tracking.player_xy * mirror, ball_xy=tracking.ball_xy * mirror
    )
    flipped = build_match_tokens(flipped_tracking, spec)
    assert np.allclose(flipped, symmetric(tokens, spec, flip_y=True), atol=1e-6)
    assert np.allclose(flipped[:, ball, Col.CTRL_TEAM], tokens[:, ball, Col.CTRL_TEAM])
    assert np.allclose(flipped[:, ball, Col.CTRL_AGE], tokens[:, ball, Col.CTRL_AGE])
    assert not np.allclose(flipped[:, ball, Col.Y], tokens[:, ball, Col.Y])  # the mirror is real

    # Inside a period IS_AWAY is already 0/1, so a swap is an involution there. Outside the
    # periods the builder writes zeros and a swap still stamps the away flag, as it always has.
    inside = tokens[played]
    before = tokens.copy()
    for flip_y, swap in ((False, False), (True, False), (False, True), (True, True)):
        once = symmetric(inside, spec, flip_y=flip_y, swap=swap)
        assert np.array_equal(symmetric(once, spec, flip_y=flip_y, swap=swap), inside)
    assert np.array_equal(tokens, before)

    full = TokenSpec(control_memory=True)
    original = build_match_tokens(small_bundle.tracking, full)
    exchanged = replace(
        small_bundle.tracking,
        slot_team=(1 - small_bundle.tracking.slot_team).astype(np.int8),
        _fingerprint=None,
    )
    played = small_bundle.meta.played_mask()
    assert np.allclose(
        build_match_tokens(exchanged, full)[played],
        symmetric(original, full, swap=True)[played],
        atol=1e-6,
    )
    mirrored = replace(
        small_bundle.tracking,
        player_xy=small_bundle.tracking.player_xy * mirror,
        ball_xy=small_bundle.tracking.ball_xy * mirror,
        _fingerprint=None,
    )
    assert np.allclose(
        build_match_tokens(mirrored, full), symmetric(original, full, flip_y=True), atol=1e-6
    )


def test_the_control_memory_run_is_the_baseline_plus_the_flag():
    import yaml

    base = yaml.safe_load((CONFIGS_DIR / "ml" / "team_baseline.yaml").read_text(encoding="utf-8"))
    got = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_control_memory.yaml").read_text(encoding="utf-8")
    )
    assert got["token"] == {**base["token"], "control_memory": True}
    assert got["model"] == {**base["model"], "n_features": 16}
    assert got["train"] == base["train"]
    spec = TokenSpec.from_dict(got["token"])
    assert spec.control_memory and spec.control_reach_m == 1.0 and spec.control_hold_frames == 3
    assert spec.n_features == got["model"]["n_features"] == N_FEATURES + 2


def test_numpy_json_carries_the_new_width_and_old_specs(tmp_path):
    spec = TokenSpec(tokens_per_team=2, control_memory=True)
    config = _tiny_config(spec.n_features, spec.tokens_per_team)
    path = tmp_path / "wide.json"
    _dump(path, config, spec.to_dict(), seed=0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["feature_version"] == FEATURE_VERSION == "1"
    assert doc["config"]["n_features"] == N_FEATURES + 2
    assert doc["token_spec"]["control_memory"] is True
    embed = next(entry for entry in doc["parameters"] if entry["name"] == "embed.0.weight")
    assert embed["shape"][1] == N_FEATURES + 2
    net = load_numpy_weights(path)
    assert net.n_features == N_FEATURES + 2 and net.spec == spec
    tokens = build_match_tokens(_scripted(), spec)
    out = net.forward(tokens[:8][None])
    assert out["logits"].shape == (1, 8, 3) and out["ball"].shape == (1, 8, 2)
    assert np.isfinite(out["logits"]).all() and np.isfinite(out["ball"]).all()

    old = tmp_path / "old.json"
    _dump(old, _tiny_config(N_FEATURES, 2), OLD_SPEC | {"tokens_per_team": 2}, seed=1)
    loaded = load_numpy_weights(old)
    assert loaded.spec == TokenSpec(tokens_per_team=2) and loaded.n_features == N_FEATURES
    zeros = np.zeros((1, 4, loaded.n_tokens, N_FEATURES), np.float32)
    assert np.isfinite(loaded.forward(zeros)["logits"]).all()


def test_a_full_length_match_adds_little_cost():
    n = 146_000
    rng = np.random.default_rng(0)
    meta = MatchMeta("long", "synthetic", n, FPS, 105.0, 68.0, (Period(1, 0, n),))
    player = rng.normal(0, 20, size=(n, 22, 2)).astype(np.float32)
    teams = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    ball = rng.normal(0, 15, size=(n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    tracking = Tracking(meta, player, teams, ball)
    off, on = TokenSpec(), TokenSpec(control_memory=True)
    build_match_tokens(tracking, off)
    build_match_tokens(tracking, on)
    started = time.perf_counter()
    base = build_match_tokens(tracking, off)
    plain = time.perf_counter() - started
    started = time.perf_counter()
    wide = build_match_tokens(tracking, on)
    widened = time.perf_counter() - started
    extra = widened - plain
    print(f"CONTROL_MEMORY_EXTRA_S {extra:.4f} plain {plain:.4f} widened {widened:.4f}")
    assert base.shape[-1] == N_FEATURES and wide.shape[-1] == N_FEATURES + 2
    assert np.array_equal(base, wide[..., :N_FEATURES])
    assert not np.isnan(wide).any()
    assert extra < 3.0


def test_torch_forward_and_export_match_to_1e9(small_bundle, tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import save_checkpoint
    from aaaag.ml.export import export_file
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.np_infer import NumpyTeamNet

    spec = TokenSpec(control_memory=True)
    cfg = ModelConfig(
        d_model=16,
        n_heads=2,
        n_blocks=1,
        dilations=(1, 2, 4),
        dropout=0.0,
        n_features=spec.n_features,
    )
    torch.manual_seed(0)
    module = TeamNet(cfg).eval()
    pt, js = tmp_path / "wide.pt", tmp_path / "wide.json"
    save_checkpoint(pt, module, spec, {})
    export_file(pt, js)
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["feature_version"] == "1"
    assert doc["config"]["n_features"] == N_FEATURES + 2
    assert doc["token_spec"]["control_memory"] is True
    state = {}
    original = module.state_dict()
    for entry in doc["parameters"]:
        shape = tuple(entry["shape"])
        raw = np.frombuffer(bytes.fromhex(entry["data"]), dtype="<f4").reshape(shape).copy()
        assert raw.tobytes() == original[entry["name"]].detach().cpu().numpy().tobytes()
        state[entry["name"]] = torch.from_numpy(raw)
    assert state["embed.0.weight"].shape[1] == N_FEATURES + 2
    fresh = TeamNet(cfg).eval()
    fresh.load_state_dict(state)
    fresh = fresh.double()
    net = load_numpy_weights(js)
    assert isinstance(net, NumpyTeamNet) and net.n_features == spec.n_features and net.spec == spec
    window = build_match_tokens(small_bundle.tracking, spec)[1000:1128]
    batch = np.stack([window, window[::-1].copy()])
    with torch.inference_mode():
        ref = fresh(torch.from_numpy(np.ascontiguousarray(batch)).double())
    got = net.forward(batch)
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-9
    assert np.max(np.abs(got["ball"] - ref["ball"].numpy())) <= 1e-9


def test_a_short_training_run_accepts_the_wider_tokens(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.train import RunConfig, TrainConfig, train
    from aaaag.synthetic import SyntheticConfig, generate

    spec = TokenSpec(control_memory=True)
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
    loaded = load_checkpoint(final)
    assert loaded.spec == spec and loaded.model.cfg.n_features == N_FEATURES + 2
    assert loaded.payload["feature_version"] == "1"
    tokens = torch.from_numpy(build_match_tokens(match.bundle.tracking, spec)[2000:2064])
    out = loaded.model(tokens[None])
    assert out["logits"].shape == (1, 64, 3) and torch.isfinite(out["logits"]).all()


def test_an_old_checkpoint_without_the_control_keys_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint, load_payload, save_checkpoint
    from aaaag.ml.model import ModelConfig, TeamNet

    torch.manual_seed(0)
    model = TeamNet(ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1,), dropout=0.0))
    path = tmp_path / "old.pt"
    save_checkpoint(path, model, TokenSpec(), {})
    payload = load_payload(path)
    assert payload["feature_version"] == "1"
    for key in ("control_memory", "control_reach_m", "control_hold_frames"):
        payload["token_spec"].pop(key)
    torch.save(payload, path)
    loaded = load_checkpoint(path)
    assert loaded.spec == TokenSpec() and loaded.spec.n_features == N_FEATURES
    assert loaded.model.cfg.n_features == N_FEATURES
