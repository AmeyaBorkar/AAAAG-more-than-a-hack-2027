"""Optional gap-bracket columns on the ball token (#33, E73).

Off by default: the tokens match the builder without the flag bit for bit, for every combination
of the other flags, and a spec dict without the key still loads. On, six columns follow every
other block: while the ball is unseen, the controlling team at the last sighting and from the next
sighting on, and the ball position at both sightings. Numpy only, no data.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.ml.geometry import symmetric
from aaaag.ml.np_infer import dump_numpy_weights, load_numpy_weights, parameter_layout
from aaaag.ml.spec import (
    BRACKET_FEATURES,
    FEATURE_VERSION,
    N_FEATURES,
    TokenSpec,
    bracket_columns,
    path_columns,
)
from aaaag.ml.windows import build_match_tokens
from aaaag.paths import CONFIGS_DIR

FPS = 25
HOME_XY = (-10.0, 4.0)
AWAY_XY = (10.0, -6.0)
HALF_L, HALF_W = 52.5, 34.0


def _static(ball, *, n, periods) -> Tracking:
    xy = np.zeros((n, 2, 2), dtype=np.float32)
    xy[:, 0] = np.asarray(HOME_XY, dtype=np.float32)
    xy[:, 1] = np.asarray(AWAY_XY, dtype=np.float32)
    teams = np.array([HOME, AWAY], dtype=np.int8)
    meta = MatchMeta("t", "synthetic", n, FPS, 105.0, 68.0, periods)
    return Tracking(meta, xy, teams, np.asarray(ball, dtype=np.float32))


def _scripted() -> Tracking:
    """Home holds the ball, a 10-frame gap, away holds it; a gap open to the end of period 1;
    period 2 opens with a gap and then the ball sits far from everyone."""
    n = 60
    ball = np.full((n, 2), np.nan, np.float32)
    ball[0:6] = HOME_XY  # home control, 6 frames >= hold 3
    # 6:16 hidden
    ball[16:22] = AWAY_XY  # away control
    ball[22:25] = (30.0, 20.0)  # out of everyone's reach; 25:30 hidden up to the period's end
    # 30:40 between the periods; period 2 is 40:60
    # 40:45 hidden at the start of period 2
    ball[45:60] = (40.0, -10.0)  # nobody within reach
    return _static(ball, n=n, periods=(Period(1, 0, 30), Period(2, 40, 60)))


def _combos():
    for cm, contact, g2l, path in itertools.product((False, True), repeat=4):
        if g2l and not contact:
            continue
        yield {
            "tokens_per_team": 1,
            "control_memory": cm,
            "contact_inputs": contact,
            "touch_detector": "g2l" if g2l else "w2",
            "ball_path_control": path,
        }


def test_off_is_byte_identical_and_on_only_appends(small_bundle):
    assert FEATURE_VERSION == "1" and BRACKET_FEATURES == 6
    assert TokenSpec().gap_bracket is False
    raw = TokenSpec().to_dict()
    raw.pop("gap_bracket")
    assert TokenSpec.from_dict(raw) == TokenSpec()  # an old spec dict loads, flag off
    default = build_match_tokens(small_bundle.tracking)
    assert np.array_equal(default, build_match_tokens(small_bundle.tracking, TokenSpec()))
    tracking = _scripted()
    for kw in _combos():
        off = build_match_tokens(tracking, TokenSpec(**kw))
        on = build_match_tokens(tracking, TokenSpec(**kw, gap_bracket=True))
        assert on.shape[-1] == off.shape[-1] + BRACKET_FEATURES
        assert np.array_equal(on[..., : off.shape[-1]], off)  # every earlier column untouched
        spec = TokenSpec(**kw, gap_bracket=True)
        assert bracket_columns(spec) == tuple(range(off.shape[-1], off.shape[-1] + 6))
        if spec.ball_path_control:
            assert bracket_columns(spec)[0] == path_columns(spec)[3] + 1
        assert not np.isnan(on).any()
    with pytest.raises(ValueError):
        bracket_columns(TokenSpec())


def test_columns_carry_the_two_ends_of_each_gap():
    tracking = _scripted()
    spec = TokenSpec(tokens_per_team=1, gap_bracket=True)
    tok = build_match_tokens(tracking, spec, sign=np.ones(tracking.n_frames, np.float32))
    ball = spec.ball_index
    cols = bracket_columns(spec)
    br = tok[:, ball, cols[0] : cols[-1] + 1]  # (T, 6)
    assert not tok[:, :ball, cols[0] :].any()  # player tokens stay zero
    # the 6:16 gap: home at the last sighting, away from the next one on
    expected = [1.0, -1.0, HOME_XY[0] / HALF_L, HOME_XY[1] / HALF_W, AWAY_XY[0] / HALF_L]
    expected.append(AWAY_XY[1] / HALF_W)
    assert np.allclose(br[6:16], np.asarray(expected, np.float32))
    # seen frames are zero
    assert not br[0:6].any() and not br[16:25].any() and not br[45:60].any()
    # 25:30 is open to the end of period 1: the before side only, the last controller still away
    # (the ball out of reach does not end a held run), the position where it was last seen
    assert np.allclose(br[25:30, 0], -1.0) and not br[25:30, 1].any()
    assert np.allclose(br[25:30, 2:4], [30.0 / HALF_L, 20.0 / HALF_W])
    assert not br[25:30, 4:].any()
    # between the periods: zero
    assert not tok[30:40].any()
    # 40:45 opens period 2: the after side only; nobody ever controls in period 2, so team 0
    assert not br[40:45, 0].any() and not br[40:45, 2:4].any() and not br[40:45, 1].any()
    assert np.allclose(br[40:45, 4:], [40.0 / HALF_L, -10.0 / HALF_W])


def test_the_after_team_is_the_next_qualifying_run_not_a_flicker():
    n = 40
    ball = np.full((n, 2), np.nan, np.float32)
    ball[0:5] = HOME_XY
    ball[15:17] = AWAY_XY  # 2 frames, shorter than hold 3: not a run
    ball[17:20] = (30.0, 20.0)
    ball[20:40] = HOME_XY  # home again, long
    tracking = _static(ball, n=n, periods=(Period(1, 0, n),))
    spec = TokenSpec(tokens_per_team=1, gap_bracket=True)
    tok = build_match_tokens(tracking, spec, sign=np.ones(n, np.float32))
    team_after = tok[:, spec.ball_index, bracket_columns(spec)[1]]
    # the first qualifying run that ends at or after frame 15 is home's from 20 on
    assert np.allclose(team_after[5:15], 1.0)


def test_hidden_ball_cannot_leak_into_the_columns():
    tracking = _scripted()
    spec = TokenSpec(tokens_per_team=1, gap_bracket=True)
    a = build_match_tokens(tracking, spec)
    more = tracking.ball_xy.copy()
    more[16:22] = np.nan  # hide the away sighting too: the 6:25 stretch is now one gap
    b = build_match_tokens(replace(tracking, ball_xy=more, _fingerprint=None), spec)
    cols = bracket_columns(spec)
    ball = spec.ball_index
    # the after side of 6:16 now points at the next frame still seen, 22 (out of reach)
    assert np.allclose(b[6:22, ball, cols[1]], 0.0)
    assert not np.allclose(a[6:16, ball, cols[4]], b[6:16, ball, cols[4]])


def test_symmetries_match_the_mirrored_match(small_bundle):
    mirror = np.array([1.0, -1.0], np.float32)
    for tracking, spec in (
        (_scripted(), TokenSpec(tokens_per_team=1, control_memory=True, gap_bracket=True)),
        (
            small_bundle.tracking,
            TokenSpec(control_memory=True, contact_inputs=True, gap_bracket=True),
        ),
    ):
        tokens = build_match_tokens(tracking, spec)
        played = tracking.meta.played_mask()
        swapped = replace(
            tracking, slot_team=(1 - tracking.slot_team).astype(np.int8), _fingerprint=None
        )
        got = build_match_tokens(swapped, spec)
        assert np.allclose(got[played], symmetric(tokens, spec, swap=True)[played], atol=1e-6)
        flipped = replace(
            tracking,
            player_xy=tracking.player_xy * mirror,
            ball_xy=tracking.ball_xy * mirror,
            _fingerprint=None,
        )
        assert np.allclose(
            build_match_tokens(flipped, spec), symmetric(tokens, spec, flip_y=True), atol=1e-6
        )
        inside = tokens[played]
        for flip_y, swap in ((True, False), (False, True), (True, True)):
            once = symmetric(inside, spec, flip_y=flip_y, swap=swap)
            assert np.array_equal(symmetric(once, spec, flip_y=flip_y, swap=swap), inside)


def test_the_build_is_deterministic(small_bundle):
    spec = TokenSpec(gap_bracket=True, control_memory=True, contact_inputs=True)
    a = build_match_tokens(small_bundle.tracking, spec)
    b = build_match_tokens(small_bundle.tracking, spec)
    assert np.array_equal(a, b)


def test_the_e73_config_is_e56a_plus_the_flag():
    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e56a.yaml").read_text(encoding="utf-8")
    )
    got = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e73_bracket.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert got["token"] == {**base["token"], "gap_bracket": True}
    assert got["model"] == {**base["model"], "n_features": base["model"]["n_features"] + 6}
    assert got["train"] == base["train"]
    spec = TokenSpec.from_dict(got["token"])
    assert spec.n_features == got["model"]["n_features"] == 26


def test_the_numpy_path_reads_the_wider_tokens(tmp_path):
    spec = TokenSpec(tokens_per_team=1, control_memory=True, gap_bracket=True)
    config = {
        "tokens_per_team": 1,
        "n_features": spec.n_features,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1, 2],
        "kernel": 3,
        "dropout": 0.0,
    }
    rng = np.random.default_rng(0)
    arrays = {
        name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in parameter_layout(config)
    }
    path = tmp_path / "bracket.json"
    dump_numpy_weights(
        path,
        arch="team_net",
        config=config,
        token_spec=spec.to_dict(),
        arrays=arrays,
        feature_version=FEATURE_VERSION,
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["token_spec"]["gap_bracket"] is True
    net = load_numpy_weights(path)
    assert net.spec == spec and net.n_features == N_FEATURES + 2 + 6
    tokens = build_match_tokens(_scripted(), spec)
    out = net.forward(tokens[:20][None])
    assert out["logits"].shape == (1, 20, 3) and np.isfinite(out["logits"]).all()


def test_torch_forward_and_export_match_numpy_to_1e9(small_bundle, tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import save_checkpoint
    from aaaag.ml.export import export_file
    from aaaag.ml.model import ModelConfig, TeamNet

    spec = TokenSpec(control_memory=True, contact_inputs=True, gap_bracket=True)
    cfg = ModelConfig(
        d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0,
        n_features=spec.n_features,
    )  # fmt: skip
    torch.manual_seed(0)
    module = TeamNet(cfg).eval()
    pt, js = tmp_path / "bracket.pt", tmp_path / "bracket.json"
    save_checkpoint(pt, module, spec, {})
    export_file(pt, js)
    net = load_numpy_weights(js)
    assert net.spec == spec and net.n_features == 26
    window = build_match_tokens(small_bundle.tracking, spec)[1000:1128]
    batch = np.stack([window, window[::-1].copy()])
    with torch.inference_mode():
        ref = module.double()(torch.from_numpy(np.ascontiguousarray(batch)).double())
    got = net.forward(batch)
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-9


def test_a_short_training_run_accepts_the_bracket_tokens(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.train import RunConfig, TrainConfig, train
    from aaaag.synthetic import SyntheticConfig, generate

    spec = TokenSpec(control_memory=True, gap_bracket=True)
    run = RunConfig(
        token=spec,
        model=ModelConfig(
            d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0,
            n_features=spec.n_features,
        ),
        train=TrainConfig(
            seed=3, steps=2, refresh_steps=2, batch_size=2, window=64, warmup_steps=1, lr=3e-3,
            degrade=DegradeMix(version=None), amp=False, flip_y=True, swap_teams=True,
        ),
    )  # fmt: skip
    match = TrainMatch.from_bundle(
        generate(seed=4, config=SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)), spec
    )
    final = train(
        run, [match], tmp_path, device="cpu", log=lambda _line: None, resume=False, threads=1
    )
    loaded = load_checkpoint(final)
    assert loaded.spec == spec and loaded.model.cfg.n_features == N_FEATURES + 2 + 6
    tokens = torch.from_numpy(build_match_tokens(match.bundle.tracking, spec)[2000:2064])
    out = loaded.model(tokens[None])
    assert out["logits"].shape == (1, 64, 3) and torch.isfinite(out["logits"]).all()
