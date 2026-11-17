"""Optional possessor head on TeamNet.

Off by default: the network, the loss, the saved config and the random stream match a run without
the head. On, a per-token head is trained against an algorithmic slot target and its team sums
join the frame embedding. Target tests are numpy only. Anything that imports torch, the model or
the trainer starts with importorskip so CI without torch still runs the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchMeta, Period, X
from aaaag.ml.data import DegradeMix, TrainMatch, build_entry
from aaaag.ml.np_infer import dump_numpy_weights, load_numpy_weights, parameter_layout
from aaaag.ml.possessor import possessor_targets
from aaaag.ml.spec import FEATURE_VERSION, ROTATE_COLUMNS, Col, TokenSpec
from aaaag.paths import CONFIGS_DIR
from aaaag.synthetic import SyntheticConfig, generate

# Pitch 64 x 32 so half-length 32 and half-width 16 keep the metre placements binary-exact.
PITCH_L, PITCH_W = 64.0, 32.0
HALF_L, HALF_W = PITCH_L / 2.0, PITCH_W / 2.0
REACH = 2.0


def _meta(n: int, played_until: int | None = None) -> MatchMeta:
    stop = n if played_until is None else played_until
    return MatchMeta("t", "synthetic", n, 25, PITCH_L, PITCH_W, (Period(1, 0, stop),))


def _blank(n: int, k: int) -> np.ndarray:
    return np.zeros((n, 2 * k + 1, TokenSpec(tokens_per_team=k).n_features), np.float64)


def _put(tokens: np.ndarray, frame: int, slot: int, x_m: float, y_m: float, *, valid: float = 1.0):
    tokens[frame, slot, Col.X] = x_m / HALF_L
    tokens[frame, slot, Col.Y] = y_m / HALF_W
    tokens[frame, slot, Col.VALID] = valid


def _targets(tokens, labels, ball, sign, meta, k, reach=REACH):
    spec = TokenSpec(tokens_per_team=k)
    return possessor_targets(tokens, labels, ball, sign, meta, spec, reach)


def test_targets_follow_the_label_the_reach_and_validity():
    k = 3
    n = 10
    # Frames 8 and 9 sit outside the period.
    meta = _meta(n, played_until=8)
    tokens = _blank(n, k)
    ball = np.zeros((n, 2), np.float32)
    sign = np.ones(n, np.float32)
    labels = np.array([H, A, H, H, D, X, H, H, H, D], dtype=np.int8)

    _put(tokens, 0, 0, 0.0, 0.0)  # home, on the ball
    _put(tokens, 0, 1, 4.0, 0.0)
    _put(tokens, 1, 0, 1.0, 0.0)  # home is closer, but the label is away
    _put(tokens, 1, k, 0.0, 0.0)
    _put(tokens, 1, k + 1, 8.0, 0.0)
    _put(tokens, 2, 0, 2.25, 0.0)  # above the reach
    _put(tokens, 2, 1, 8.0, 0.0)
    ball[3] = np.nan  # clean ball missing
    _put(tokens, 3, 0, 0.0, 0.0)
    _put(tokens, 4, 0, 0.0, 0.0)  # dead, even with a player on the ball
    _put(tokens, 5, 0, 0.0, 0.0)  # X
    _put(tokens, 6, 0, 0.0, 0.0, valid=0.0)  # nearer, but invalid
    _put(tokens, 6, 1, 1.0, 0.0)
    _put(tokens, 7, 0, 0.0, 4.0)  # x-only would pick this; true distance is 4 m
    _put(tokens, 7, 1, 2.0, 0.0)  # true nearest, inside the reach of 3 m below
    _put(tokens, 8, 0, 0.0, 0.0)  # unplayed H
    _put(tokens, 9, 0, 0.0, 0.0)  # unplayed D

    got = _targets(tokens, labels, ball, sign, meta, k)
    assert got.dtype == np.int16 and got.shape == (n,)
    assert got[0] == 0
    assert got[1] == k  # an away slot, not the closer home slot
    assert got[2] == -1  # 2.25 m is above the 2 m reach
    assert got[3] == -1
    assert got[4] == 2 * k
    assert got[5] == -1
    assert got[6] == 1  # the invalid slot 0 is not chosen
    y_reach = _targets(tokens, labels, ball, sign, meta, k, reach=3.0)
    assert y_reach[7] == 1
    assert got[8] == -1 and got[9] == -1

    # Distance equal to the reach is kept. Sign puts the clean ball into the attack frame.
    edge = _blank(2, k)
    _put(edge, 0, 0, REACH, 0.0)
    _put(edge, 1, 2, 4.0, 0.0)
    edge_ball = np.array([[0.0, 0.0], [-4.0, 0.0]], np.float32)
    edge_sign = np.array([1.0, -1.0], np.float32)
    edge_labels = np.array([H, H], np.int8)
    edge_got = _targets(edge, edge_labels, edge_ball, edge_sign, _meta(2), k)
    assert edge_got.tolist() == [0, 2]
    assert _targets(edge.astype(np.float16), edge_labels, edge_ball, edge_sign, _meta(2), k)[0] == 0


def test_each_variant_picks_from_its_own_slots():
    k = 2
    meta = _meta(1)
    labels = np.array([H], np.int8)
    ball = np.zeros((1, 2), np.float32)
    sign = np.ones(1, np.float32)
    near_first = _blank(1, k)
    near_second = _blank(1, k)
    _put(near_first, 0, 0, 0.0, 0.0)
    _put(near_first, 0, 1, 5.0, 0.0)
    _put(near_second, 0, 0, 5.0, 0.0)
    _put(near_second, 0, 1, 0.0, 0.0)
    first = _targets(near_first, labels, ball, sign, meta, k)
    second = _targets(near_second, labels, ball, sign, meta, k)
    assert first[0] == 0 and second[0] == 1


@pytest.fixture(scope="module")
def train_match() -> TrainMatch:
    spec = TokenSpec()
    bundle = generate(seed=4, config=SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5))
    return TrainMatch.from_bundle(bundle, spec)


def test_build_entry_draws_targets_only_when_the_weight_is_positive(train_match):
    spec = TokenSpec()
    mix = DegradeMix(version=None)
    rng_off = np.random.default_rng(1)
    off = build_entry(train_match, rng_off, spec, mix)
    rng_on = np.random.default_rng(1)
    on = build_entry(train_match, rng_on, spec, mix, possessor_weight=0.2, possessor_reach_m=REACH)
    assert off.possessor_coupled is None and off.possessor_decoupled is None
    assert np.array_equal(off.coupled, on.coupled)
    assert np.array_equal(off.decoupled, on.decoupled)
    assert np.array_equal(off.visible_coupled, on.visible_coupled)
    assert rng_off.random() == rng_on.random()

    k = spec.tokens_per_team
    ball = train_match.bundle.tracking.ball_xy
    for tokens, target in (
        (on.coupled, on.possessor_coupled),
        (on.decoupled, on.possessor_decoupled),
    ):
        expect = possessor_targets(
            tokens, train_match.labels, ball, train_match.sign, train_match.bundle.meta, spec, REACH
        )
        assert target.dtype == np.int16 and np.array_equal(target, expect)
        played = train_match.played
        assert np.all(target[(train_match.labels == D) & played] == 2 * k)
        assert np.all(target[train_match.labels == X] == -1)
        assert np.all(target[~played] == -1)
        home = target[(train_match.labels == H) & played]
        home = home[home >= 0]
        away = target[(train_match.labels == A) & played]
        away = away[away >= 0]
        assert home.size and np.all(home < k)
        assert away.size and np.all((away >= k) & (away < 2 * k))


def test_numpy_layout_grows_only_when_the_flag_is_on():
    base = {
        "tokens_per_team": 11,
        "n_features": 14,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1],
        "kernel": 3,
    }
    off = parameter_layout(base)
    assert parameter_layout({**base, "possessor_head": False}) == off
    assert all(not name.startswith("possessor") for name, _shape in off)
    shapes = dict(off)
    assert shapes["frame.0.weight"] == (24,) and shapes["frame.1.weight"] == (8, 24)
    on = parameter_layout({**base, "possessor_head": True})
    on_names = [name for name, _shape in on]
    off_names = [name for name, _shape in off]
    assert on_names[: len(off_names)] == off_names
    assert on_names[len(off_names) :] == [
        "possessor.weight",
        "possessor.bias",
        "possessor_none.weight",
        "possessor_none.bias",
    ]
    on_shapes = dict(on)
    assert on_shapes["frame.0.weight"] == (27,) and on_shapes["frame.1.weight"] == (8, 27)
    assert on_shapes["possessor.weight"] == (1, 8)
    assert on_shapes["possessor_none.bias"] == (1,)


def test_numpy_forward_with_the_head_is_finite_and_uses_the_weights(tmp_path):
    config = {
        "tokens_per_team": 2,
        "n_features": 14,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1],
        "kernel": 3,
        "dropout": 0.0,
        "possessor_head": True,
    }
    spec = TokenSpec(tokens_per_team=2)
    rng = np.random.default_rng(0)
    arrays = {
        name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in parameter_layout(config)
    }
    path = tmp_path / "possessor.json"
    dump_numpy_weights(
        path,
        arch="team_net",
        config=config,
        token_spec=spec.to_dict(),
        arrays=arrays,
        feature_version=FEATURE_VERSION,
    )
    net = load_numpy_weights(path)
    tokens = rng.normal(size=(2, 6, spec.n_tokens, spec.n_features)).astype(np.float64)
    tokens[..., Col.VALID] = 1.0
    tokens[0, 1, 0, Col.VALID] = 0.0
    out = net.forward(tokens)
    assert set(out) == {"logits", "ball"}
    assert out["logits"].shape == (2, 6, 3) and np.isfinite(out["logits"]).all()
    assert np.isfinite(out["ball"]).all()
    zeroed = dict(arrays)
    for name in (
        "possessor.weight",
        "possessor.bias",
        "possessor_none.weight",
        "possessor_none.bias",
    ):
        zeroed[name] = np.zeros_like(arrays[name])
    dump_numpy_weights(
        tmp_path / "zero.json",
        arch="team_net",
        config=config,
        token_spec=spec.to_dict(),
        arrays=zeroed,
        feature_version=FEATURE_VERSION,
    )
    other = load_numpy_weights(tmp_path / "zero.json").forward(tokens)
    assert np.max(np.abs(other["logits"] - out["logits"])) > 0.0


def test_the_run_yaml_is_control_memory_plus_the_head():
    import yaml

    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_control_memory.yaml").read_text(encoding="utf-8")
    )
    got = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_possessor.yaml").read_text(encoding="utf-8")
    )
    assert got["token"] == base["token"]
    assert got["model"] == {**base["model"], "possessor_head": True}
    train = dict(got["train"])
    assert train.pop("possessor_weight") == 0.2
    assert train.pop("possessor_reach_m") == 2.0
    assert train == base["train"]


class _Zeros:
    """Draws that always fall under the 0.5 swap/flip threshold."""

    def random(self, size):
        return np.zeros(size)


def test_flag_off_matches_today_and_old_checkpoints_load(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint, load_payload, save_checkpoint
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.train import RunConfig, TrainConfig

    tiny = dict(
        d_model=8, n_heads=2, n_blocks=1, ffn_mult=2, dilations=(1, 2), kernel=3, dropout=0.1
    )
    torch.manual_seed(0)
    omitted = TeamNet(ModelConfig(**tiny)).eval()
    torch.manual_seed(0)
    explicit = TeamNet(ModelConfig(**tiny, possessor_head=False)).eval()
    left, right = omitted.state_dict(), explicit.state_dict()
    assert list(left) == list(right)
    for key in left:
        assert left[key].shape == right[key].shape
        assert torch.equal(left[key], right[key])
    x = torch.randn(2, 6, omitted.cfg.n_tokens, omitted.cfg.n_features)
    with torch.no_grad():
        a, b = omitted(x), explicit(x)
    assert set(a) == set(b) == {"logits", "ball"}
    assert torch.equal(a["logits"], b["logits"]) and torch.equal(a["ball"], b["ball"])
    assert "possessor_head" not in omitted.cfg.to_dict()

    old = {
        "tokens_per_team": 4,
        "n_features": 14,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1, 2],
        "kernel": 3,
        "dropout": 0.0,
    }
    assert "possessor_head" not in old
    torch.manual_seed(2)
    model = TeamNet(ModelConfig.from_dict(old)).eval()
    fresh = TeamNet(ModelConfig.from_dict(dict(old))).eval()
    fresh.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    probe = torch.randn(1, 4, model.cfg.n_tokens, model.cfg.n_features)
    with torch.no_grad():
        assert torch.equal(model(probe)["logits"], fresh(probe)["logits"])
    path = tmp_path / "old.pt"
    save_checkpoint(path, model, TokenSpec(tokens_per_team=4), {})
    payload = load_payload(path)
    assert "possessor_head" not in payload["model_config"]
    loaded = load_checkpoint(path)
    assert not loaded.model.cfg.possessor_head
    assert torch.equal(loaded.model.state_dict()["head.weight"], model.state_dict()["head.weight"])

    bare = RunConfig()
    assert "possessor_head" not in bare.to_dict()["model"]
    assert "possessor_weight" not in bare.to_dict()["train"]
    assert "possessor_reach_m" not in bare.to_dict()["train"]
    for name in ("team_baseline.yaml", "team_control_memory.yaml"):
        run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / name)
        dumped = run.to_dict()
        assert "possessor_head" not in dumped["model"]
        assert "possessor_weight" not in dumped["train"]
        assert "possessor_reach_m" not in dumped["train"]
        assert RunConfig.from_dict(dumped) == run
    with pytest.raises(ValueError, match="possessor"):
        RunConfig(train=TrainConfig(possessor_weight=0.2))
    RunConfig(model=ModelConfig(possessor_head=True), train=TrainConfig(possessor_weight=0.2))


def test_augment_swap_maps_the_target_and_keeps_the_player():
    torch = pytest.importorskip("torch")
    from aaaag.ml.train import TrainConfig, augment

    k = 4
    spec = TokenSpec(tokens_per_team=k)
    n_tokens = 2 * k + 1
    tokens = torch.zeros(1, 6, n_tokens, spec.n_features)
    for slot in range(2 * k):
        tokens[:, :, slot, Col.X] = (slot + 1) * 0.01
        tokens[:, :, slot, Col.Y] = (slot + 1) * 0.02
        tokens[:, :, slot, Col.SPEED] = (slot + 1) * 0.03
        tokens[:, :, slot, Col.VALID] = 1.0
        tokens[:, :, slot, Col.IS_AWAY] = float(slot >= k)
    possessor = torch.tensor([[0, k - 1, k, 2 * k - 1, 2 * k, -1]], dtype=torch.long)
    batch = {
        "tokens": tokens.clone(),
        "labels": torch.zeros(1, 6, dtype=torch.long),
        "ball": torch.zeros(1, 6, 2),
        "possessor": possessor.clone(),
    }
    original = batch["tokens"].clone()
    augment(batch, spec, TrainConfig(swap_teams=True, flip_y=False), _Zeros())
    expect = torch.tensor([[k, 2 * k - 1, 0, k - 1, 2 * k, -1]], dtype=torch.long)
    assert torch.equal(batch["possessor"], expect)
    for slot in range(2 * k):
        mapped = slot + k if slot < k else slot - k
        old = original[:, :, slot]
        new = batch["tokens"][:, :, mapped]
        for column in ROTATE_COLUMNS:
            assert torch.equal(new[..., column], -old[..., column])
        for column in (Col.SPEED, Col.VALID):
            assert torch.equal(new[..., column], old[..., column])

    flipped = {
        "tokens": tokens.clone(),
        "labels": torch.zeros(1, 6, dtype=torch.long),
        "ball": torch.zeros(1, 6, 2),
        "possessor": possessor.clone(),
    }
    y_before = flipped["tokens"][..., Col.Y].clone()
    augment(flipped, spec, TrainConfig(swap_teams=False, flip_y=True), _Zeros())
    assert torch.equal(flipped["possessor"], possessor)
    assert torch.equal(flipped["tokens"][..., Col.Y], -y_before)

    # The possessor map draws no random numbers, so tokens and labels match a batch without it.
    def _window(with_target: bool) -> dict:
        window = {
            "tokens": tokens.repeat(2, 1, 1, 1).clone(),
            "labels": torch.tensor([[0, 1, 2, 3, 0, 1], [1, 1, 0, 2, 3, 0]]),
            "ball": torch.arange(24, dtype=torch.float32).view(2, 6, 2),
        }
        if with_target:
            window["possessor"] = possessor.repeat(2, 1).clone()
        return window

    cfg = TrainConfig()  # swap and y-flip both on
    left_rng, right_rng = np.random.default_rng(5), np.random.default_rng(5)
    left = augment(_window(False), spec, cfg, left_rng)
    right = augment(_window(True), spec, cfg, right_rng)
    assert torch.equal(left["tokens"], right["tokens"])
    assert torch.equal(left["labels"], right["labels"])
    assert torch.equal(left["ball"], right["ball"])
    assert left_rng.random() == right_rng.random()


def test_loss_is_unchanged_at_weight_zero_and_adds_a_finite_term():
    torch = pytest.importorskip("torch")
    from aaaag.ml.spellweight import SpellWeight
    from aaaag.ml.train import DistillConfig, loss_terms

    torch.manual_seed(0)
    n, w, margin, k = 2, 40, 4, 3
    classes = 2 * k + 1
    labels = torch.randint(0, 3, (n, w))
    out = {"logits": torch.randn(n, w, 3), "ball": torch.randn(n, w, 2)}
    batch = {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.randn(n, w, 2),
    }
    batch["ball_ok"][:, 12] = True
    weights = torch.tensor([1.0, 1.2, 0.8])
    plain = loss_terms(out, batch, weights, margin, 0.3)
    off = loss_terms(out, batch, weights, margin, 0.3, possessor_weight=0.0)
    assert torch.equal(plain["loss"], off["loss"])
    assert set(plain) == set(off) == {"loss", "ce", "aux", "acc"}
    for key in plain:
        assert torch.equal(plain[key], off[key])

    target = torch.randint(0, classes, (n, w))
    target[:, :margin] = -1
    target[:, margin] = -1
    out_p = {**out, "possessor": torch.randn(n, w, classes)}
    batch_p = {**batch, "possessor": target}
    still = loss_terms(out_p, batch_p, weights, margin, 0.3, possessor_weight=0.0)
    assert torch.equal(still["loss"], plain["loss"]) and "pos" not in still
    on = loss_terms(out_p, batch_p, weights, margin, 0.3, possessor_weight=0.2)
    assert "pos" in on and torch.isfinite(on["pos"])
    assert torch.equal(on["loss"], plain["loss"] + 0.2 * on["pos"])
    assert not torch.equal(on["loss"], plain["loss"])

    ignored = {**batch_p, "possessor": torch.full((n, w), -1)}
    zero = loss_terms(out_p, ignored, weights, margin, 0.3, possessor_weight=0.2)
    assert zero["pos"].item() == 0.0 and torch.equal(zero["loss"], plain["loss"])

    spelled = loss_terms(
        out_p, batch_p, weights, margin, 0.0, spell=SpellWeight(gamma=0.5), possessor_weight=0.2
    )
    assert torch.isfinite(spelled["loss"]) and torch.isfinite(spelled["pos"])

    teacher = torch.randn(n, w, 3)
    distill = DistillConfig(teacher_runs=("teacher",), alpha=0.5, temperature=2.0)
    base = loss_terms(out, batch, weights, margin, 0.3, teacher_logits=teacher, distill=distill)
    distilled_off = loss_terms(
        out_p,
        batch_p,
        weights,
        margin,
        0.3,
        teacher_logits=teacher,
        distill=distill,
        possessor_weight=0.0,
    )
    assert torch.equal(distilled_off["loss"], base["loss"])
    assert "kl" in distilled_off and "pos" not in distilled_off
    distilled = loss_terms(
        out_p,
        batch_p,
        weights,
        margin,
        0.3,
        teacher_logits=teacher,
        distill=distill,
        possessor_weight=0.4,
    )
    assert "kl" in distilled and torch.isfinite(distilled["pos"])
    assert torch.equal(distilled["loss"], base["loss"] + 0.4 * distilled["pos"])


def test_numpy_matches_torch_to_1e6_with_the_head_on():
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import MASKED, ModelConfig, TeamNet
    from aaaag.ml.np_infer import NumpyTeamNet

    torch.manual_seed(0)
    cfg = ModelConfig(
        tokens_per_team=2,
        d_model=8,
        n_heads=2,
        n_blocks=1,
        ffn_mult=2,
        dilations=(1, 2),
        kernel=3,
        dropout=0.0,
        possessor_head=True,
    )
    module = TeamNet(cfg).eval()
    assert [name for name, _shape in parameter_layout(cfg.to_dict())] == list(module.state_dict())
    assert module.state_dict()["frame.1.weight"].shape == (8, 3 * 8 + 3)
    assert "possessor_head" in cfg.to_dict()
    module = module.double()
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in module.state_dict().items()}
    spec = TokenSpec(tokens_per_team=2)
    net = NumpyTeamNet(cfg.to_dict(), spec, arrays)
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(2, 5, spec.n_tokens, spec.n_features))
    tokens[..., Col.VALID] = 1.0
    tokens[0, 2, :2, Col.VALID] = 0.0  # empty home team on one frame
    tokens[1, 1, 3, Col.VALID] = 0.0
    with torch.inference_mode():
        ref = module(torch.from_numpy(np.ascontiguousarray(tokens)).double())
        ref_frames = module.encode_frames(
            torch.from_numpy(tokens.reshape(-1, spec.n_tokens, spec.n_features)).double()
        )
    got = net.forward(tokens)
    frames = net.encode_frames(tokens.reshape(-1, spec.n_tokens, spec.n_features))
    assert ref["possessor"].shape == (2, 5, 2 * spec.tokens_per_team + 1)
    assert "possessor" not in got
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-6
    assert np.max(np.abs(got["ball"] - ref["ball"].numpy())) <= 1e-6
    assert np.max(np.abs(frames - ref_frames.numpy())) <= 1e-6
    # The invalid player slot is masked; a valid slot is a real logit.
    assert ref["possessor"][1, 1, 3].item() == pytest.approx(MASKED, rel=0, abs=0)
    assert ref["possessor"][1, 1, 0].item() > -1e8


def test_a_few_steps_train_and_the_export_matches_numpy(tmp_path, train_match):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint, load_payload
    from aaaag.ml.data import DegradeMix
    from aaaag.ml.export import main
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.np_infer import load_numpy_weights
    from aaaag.ml.train import RunConfig, TrainConfig, train
    from aaaag.ml.windows import build_match_tokens

    spec = TokenSpec()
    run = RunConfig(
        token=spec,
        model=ModelConfig(
            d_model=16,
            n_heads=2,
            n_blocks=1,
            dilations=(1, 2, 4),
            dropout=0.0,
            possessor_head=True,
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
            possessor_weight=0.2,
            possessor_reach_m=2.0,
        ),
    )
    lines: list[str] = []
    final = train(
        run, [train_match], tmp_path, device="cpu", log=lines.append, resume=False, threads=1
    )
    assert any(" pos " in line for line in lines)
    payload = load_payload(final)
    assert payload["meta"]["run"]["model"]["possessor_head"] is True
    assert payload["meta"]["run"]["train"]["possessor_weight"] == 0.2
    assert np.isfinite(payload["meta"]["history"][0]["pos"])
    assert np.isfinite(payload["meta"]["history"][0]["loss"])
    loaded = load_checkpoint(final)
    assert loaded.model.cfg.possessor_head
    js = tmp_path / "possessor.json"
    main(["--weights", str(final), "--out", str(js)])
    net = load_numpy_weights(js)
    tokens = build_match_tokens(train_match.bundle.tracking, spec)
    window = np.ascontiguousarray(tokens[2000:2064])
    assert (window[..., Col.VALID] > 0.5).any()
    module = loaded.model.double().eval()
    with torch.inference_mode():
        ref = module(torch.from_numpy(window).double()[None])
    got = net.forward(window[None])
    assert ref["possessor"].shape == (1, 64, 2 * spec.tokens_per_team + 1)
    assert "possessor" not in got
    assert np.max(np.abs(got["logits"] - ref["logits"].numpy())) <= 1e-6
    assert np.max(np.abs(got["ball"] - ref["ball"].numpy())) <= 1e-6
