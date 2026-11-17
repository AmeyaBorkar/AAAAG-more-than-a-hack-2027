"""A dense touch head: one logit per frame for a ball touch, from the clean ball.

Off by default. With the head and the weight off, a tiny training run and its checkpoint keys
match a run that never mentions the head. Torch tests skip when torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from aaaag.contracts import A, D, H, MatchBundle, MatchMeta, Period, Tracking, X
from aaaag.ml.data import TrainMatch, touch_target, with_touch_target
from aaaag.ml.spec import TokenSpec
from aaaag.paths import CONFIGS_DIR
from aaaag.postprocess.touch_snap import detect_touches

FPS = 25
SPEED = 6.0 / FPS
TURNS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


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
    n: int, touches: dict[int, int], periods: tuple[Period, ...] | None = None
) -> Tracking:
    """Ball at 6 m/s with a 90 degree turn at each frame of `touches` (team 0 or 1 stands next to it)."""
    meta = _meta(n, periods)
    ball = np.zeros((n, 2), dtype=np.float32)
    position = np.zeros(2)
    turn = 0
    for t in range(n):
        if (t - 1) in touches:
            turn += 1
        position = position + np.array(TURNS[turn % 4]) * SPEED
        ball[t] = position
    players = np.full((n, 2, 2), 80.0, dtype=np.float32)
    for t, team in touches.items():
        lo, hi = max(0, t - 3), min(n, t + 4)
        players[lo:hi, team] = ball[lo:hi] + 0.5
    return Tracking(
        meta=meta, player_xy=players, slot_team=np.array([0, 1], dtype=np.int8), ball_xy=ball
    ).validate()


def _bundle(n: int = 30, gap: bool = False) -> MatchBundle:
    periods = (Period(1, 0, 12), Period(2, 18, n)) if gap else (Period(1, 0, n),)
    meta = _meta(n, periods)
    xy = np.zeros((n, 2, 2), np.float32)
    xy[:, 1, 0] = 8.0
    ball = np.zeros((n, 2), np.float32)
    teams = np.array([H, A], np.int8)
    return MatchBundle(Tracking(meta, xy, teams, ball).validate())


def _tiny_layout():
    spec = TokenSpec(tokens_per_team=1)
    return {
        "tokens_per_team": 1,
        "n_features": spec.n_features,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1],
        "kernel": 3,
        "dropout": 0.0,
    }, spec


# -- target and config (no torch) -----------------------------------------------------------------


def test_touch_target_marks_clean_touches_and_ignores_unseen_and_dead():
    n = 120
    played_until = 110
    periods = (Period(1, 0, played_until),)
    touches = {40: 0, 70: 1}  # home, then away: both teams count
    tracking = _tracking(n, touches, periods)
    tracking.ball_xy[100:108] = np.nan  # unseen, and not on a touch
    labels = np.full(n, H, dtype=np.int8)
    labels[10:16] = D
    labels[39] = D  # inside the tolerance of the touch at 40, still ignored
    labels[41] = D
    labels[90:96] = X
    labels[played_until:] = H  # unplayed, even with a seen ball
    bundle = MatchBundle(tracking, labels=labels).validate()
    match = TrainMatch.from_bundle(bundle, TokenSpec())
    assert match.touch is None

    tol = 2
    marked = with_touch_target(match, tol)
    assert match.touch is None
    target = marked.touch
    assert target is not None and target.dtype == np.float32 and target.shape == (n,)
    direct = touch_target(match, tol)
    assert np.array_equal(target, direct, equal_nan=True)

    frames, teams = detect_touches(tracking)
    found = dict(zip(map(int, frames), map(int, teams), strict=True))
    assert found[40] == 0 and found[70] == 1
    supervised = tracking.ball_seen() & np.isin(labels, (H, A)) & match.played
    near = np.zeros(n, dtype=bool)
    for frame in frames:
        lo = max(0, int(frame) - tol)
        near[lo : int(frame) + tol + 1] = True
    assert np.array_equal(target == 1, supervised & near)
    assert np.array_equal(target == 0, supervised & ~near)
    assert np.isnan(target[~supervised]).all()
    assert np.isnan(target[~tracking.ball_seen()]).all()
    assert np.isnan(target[labels == D]).all()
    assert np.isnan(target[labels == X]).all()
    assert np.isnan(target[~match.played]).all()
    assert target[40] == 1 and target[70] == 1
    # Dead frames inside the tolerance stay ignored; the alive edge of that neighbourhood is 1.
    assert np.isnan(target[39]) and np.isnan(target[41])
    distance = np.min(np.abs(np.arange(n)[:, None] - np.asarray(frames)[None, :]), axis=1)
    on_edge = np.flatnonzero(supervised & (distance == tol))
    past = np.flatnonzero(supervised & (distance == tol + 1))
    assert len(on_edge) > 0 and np.all(target[on_edge] == 1)
    assert len(past) > 0 and np.all(target[past] == 0)

    exact = touch_target(match, 0)
    assert exact[40] == 1 and np.isnan(exact[39])
    assert exact[38] == (1 if distance[38] == 0 else 0)


def test_e48_config_is_the_wide_recipe_plus_the_touch_head():
    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact_both_smooth_wide.yaml").read_text(encoding="utf-8")
    )
    got = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e48.yaml").read_text(encoding="utf-8")
    )
    assert got["token"] == base["token"]
    assert got["model"] == {**base["model"], "touch_head": True}
    train = dict(got["train"])
    assert train.pop("touch_weight") == 0.2
    assert train == base["train"]


def test_numpy_producer_attaches_touch_and_omits_it_when_off(tmp_path):
    from aaaag.evidence.possession.learned_team import learned_team
    from aaaag.ml.np_infer import dump_numpy_weights, parameter_layout, predict_logits_np
    from aaaag.ml.spec import FEATURE_VERSION

    cfg, spec = _tiny_layout()

    def write(path, config, seed):
        drawn = np.random.default_rng(seed)
        arrays = {
            name: (drawn.standard_normal(shape) * 0.02).astype(np.float32)
            for name, shape in parameter_layout(config)
        }
        dump_numpy_weights(
            path,
            arch="team_net",
            config=config,
            token_spec=spec.to_dict(),
            arrays=arrays,
            feature_version=FEATURE_VERSION,
        )

    bundle = _bundle(n=30, gap=True)
    off = tmp_path / "off.json"
    write(off, cfg, seed=0)
    plain = learned_team(bundle, weights=str(off), backend="numpy", views=4)
    assert plain.channels == {}
    assert predict_logits_np([off], bundle.tracking, views=4).touch is None

    on = tmp_path / "on.json"
    write(on, {**cfg, "touch_head": True}, seed=1)
    carrying = learned_team(bundle, weights=str(on), backend="numpy", views=4, output="joint")
    channel = carrying.channels["touch"]
    played = bundle.meta.played_mask()
    assert "switch" not in carrying.channels
    assert channel.shape == (bundle.meta.n_frames,) and channel.dtype == np.float32
    assert np.isnan(channel[~played]).all()
    assert np.isfinite(channel[played]).all()
    names = [name for name, _shape in parameter_layout(cfg)]
    headed = [name for name, _shape in parameter_layout({**cfg, "touch_head": True})]
    assert headed[: len(names)] == names
    assert headed[-2:] == ["touch.weight", "touch.bias"]

    both_cfg = {**cfg, "switch_head": True, "touch_head": True}
    both = tmp_path / "both.json"
    write(both, both_cfg, seed=2)
    stacked = learned_team(bundle, weights=str(both), backend="numpy", views=2)
    assert set(stacked.channels) == {"switch", "touch"}

    other = tmp_path / "other.json"
    write(other, {**cfg, "touch_head": True}, seed=3)
    one = predict_logits_np([on], bundle.tracking, views=4, chunk=8, workers=1)
    two = predict_logits_np([other], bundle.tracking, views=4, chunk=8, workers=1)
    mean = predict_logits_np([on, other], bundle.tracking, views=4, chunk=8, workers=1)
    assert one.touch is not None and two.touch is not None and mean.touch is not None
    expect = ((one.touch.astype(np.float64) + two.touch.astype(np.float64)) / 2).astype(np.float32)
    assert np.allclose(mean.touch, expect, equal_nan=True)
    pooled = predict_logits_np([on], bundle.tracking, views=4, chunk=8, workers=2)
    assert np.array_equal(pooled.touch, one.touch, equal_nan=True)
    assert np.array_equal(pooled.logits, one.logits, equal_nan=True)


def test_views_average_the_touch_logit_and_do_not_swap_it():
    from aaaag.ml.geometry import SWAP_CLASSES, symmetric
    from aaaag.ml.np_infer import VIEWS, NumpyTeamNet, _kept_scalars, parameter_layout
    from aaaag.ml.spec import Col

    cfg, spec = _tiny_layout()
    cfg = {**cfg, "touch_head": True, "switch_head": True}
    rng = np.random.default_rng(1)
    arrays = {
        name: (rng.standard_normal(shape) * 0.05).astype(np.float32)
        for name, shape in parameter_layout(cfg)
    }
    net = NumpyTeamNet(cfg, spec, arrays)
    window = rng.standard_normal((12, net.n_tokens, net.n_features))
    window[..., Col.VALID] = 1.0
    logits, scalars = _kept_scalars(net, spec, window, 0, 0, 12, 4, ("switch", "touch"))
    total_z = None
    total_s = None
    total_t = None
    for flip_y, swap in VIEWS[4]:
        x = symmetric(window, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else window
        forward = net.forward(x[None])
        z = forward["logits"][0]
        if swap:
            z = z[:, SWAP_CLASSES]
        total_z = z if total_z is None else total_z + z
        total_s = forward["switch"][0] if total_s is None else total_s + forward["switch"][0]
        total_t = forward["touch"][0] if total_t is None else total_t + forward["touch"][0]
    assert np.allclose(logits, (total_z / 4).astype(np.float32))
    assert np.allclose(scalars["switch"], (total_s / 4).astype(np.float32))
    assert np.allclose(scalars["touch"], (total_t / 4).astype(np.float32))


# -- torch ----------------------------------------------------------------------------------------


def test_flag_off_matches_an_explicit_false_and_old_dicts_still_load():
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.train import RunConfig, TrainConfig

    tiny = dict(
        d_model=8, n_heads=2, n_blocks=1, ffn_mult=2, dilations=(1, 2), kernel=3, dropout=0.0
    )
    torch.manual_seed(0)
    omitted = TeamNet(ModelConfig(**tiny)).eval()
    torch.manual_seed(0)
    explicit = TeamNet(ModelConfig(**tiny, touch_head=False)).eval()
    left, right = omitted.state_dict(), explicit.state_dict()
    assert list(left) == list(right)
    for key in left:
        assert torch.equal(left[key], right[key])
    x = torch.randn(2, 5, omitted.cfg.n_tokens, omitted.cfg.n_features)
    with torch.no_grad():
        a, b = omitted(x), explicit(x)
    assert set(a) == set(b) == {"logits", "ball"}
    assert torch.equal(a["logits"], b["logits"]) and torch.equal(a["ball"], b["ball"])
    assert "touch_head" not in omitted.cfg.to_dict()

    torch.manual_seed(0)
    headed = TeamNet(ModelConfig(**tiny, touch_head=True)).eval()
    shared = [key for key in left if key in headed.state_dict()]
    assert shared == list(left)
    for key in shared:
        assert torch.equal(left[key], headed.state_dict()[key])
    assert list(headed.state_dict())[-2:] == ["touch.weight", "touch.bias"]
    with torch.no_grad():
        out = headed(x)
    assert out["touch"].shape == (2, 5)
    assert torch.equal(out["logits"], a["logits"]) and torch.equal(out["ball"], a["ball"])

    torch.manual_seed(0)
    switch_only = TeamNet(ModelConfig(**tiny, switch_head=True)).eval()
    torch.manual_seed(0)
    both = TeamNet(ModelConfig(**tiny, switch_head=True, touch_head=True)).eval()
    assert list(both.state_dict())[-4:] == [
        "switch.weight",
        "switch.bias",
        "touch.weight",
        "touch.bias",
    ]
    for key in switch_only.state_dict():
        assert torch.equal(switch_only.state_dict()[key], both.state_dict()[key])

    bare = RunConfig()
    assert "touch_head" not in bare.to_dict()["model"]
    for key in ("touch_weight", "touch_tol_frames", "touch_pos_weight"):
        assert key not in bare.to_dict()["train"]
    for name in ("team_baseline.yaml", "team_control_memory.yaml"):
        dumped = RunConfig.from_yaml(CONFIGS_DIR / "ml" / name).to_dict()
        assert "touch_head" not in dumped["model"]
        for key in ("touch_weight", "touch_tol_frames", "touch_pos_weight"):
            assert key not in dumped["train"]
    with pytest.raises(ValueError, match="touch"):
        RunConfig(train=TrainConfig(touch_weight=0.2))
    loaded = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e48.yaml")
    assert loaded.model.touch_head and loaded.model.switch_head and loaded.model.possessor_head
    assert loaded.train.touch_weight == 0.2
    assert loaded.train.touch_tol_frames == 2 and loaded.train.touch_pos_weight == 10.0
    assert RunConfig.from_dict(loaded.to_dict()) == loaded


def test_windows_slice_the_target_and_augment_leaves_it():
    torch = pytest.importorskip("torch")
    from aaaag.ml.data import SetEntry
    from aaaag.ml.train import DeviceSet, TrainConfig, augment

    n = 8
    meta = _meta(n)
    tracking = Tracking(
        meta,
        np.zeros((n, 2, 2), np.float32),
        np.array([0, 1], np.int8),
        np.zeros((n, 2), np.float32),
    ).validate()
    labels = np.array([H, H, A, A, D, H, H, H], np.int8)
    touch = np.array([0, 1, 1, 0, np.nan, 0, 1, 0], np.float32)
    match = TrainMatch(
        match_id="syn",
        bundle=MatchBundle(tracking, labels=labels.copy()),
        labels=labels.copy(),
        sign=np.ones(n, np.float32),
        ball_target=np.zeros((n, 2), np.float32),
        target_ok=np.ones(n, dtype=bool),
        played=np.ones(n, dtype=bool),
        touch=touch.copy(),
    )
    entry = SetEntry(
        "syn",
        np.zeros((n, 1, 4), np.float16),
        np.zeros((n, 1, 4), np.float16),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )
    data = DeviceSet([entry], [match], torch.device("cpu"))
    batch = data.batch(np.random.default_rng(0), 2, n, decoupled=False)
    expect = torch.from_numpy(touch)
    assert torch.equal(batch["labels"][0], torch.from_numpy(labels).long())
    assert torch.equal(
        torch.nan_to_num(batch["touch"][0], nan=-2), torch.nan_to_num(expect, nan=-2)
    )

    spec = TokenSpec(tokens_per_team=1)

    def _window(with_target: bool) -> dict:
        window = {
            "tokens": torch.zeros(2, n, spec.n_tokens, spec.n_features),
            "labels": torch.tensor([[H, A, D, X, H, A, H, A], [A, A, H, H, D, X, H, A]]),
            "ball": torch.arange(2 * n * 2, dtype=torch.float32).view(2, n, 2),
        }
        window["tokens"][..., 7] = 1.0
        if with_target:
            window["touch"] = expect.repeat(2, 1).clone()
        return window

    cfg = TrainConfig()
    left_rng, right_rng = np.random.default_rng(5), np.random.default_rng(5)
    before = _window(True)["touch"].clone()
    left = augment(_window(False), spec, cfg, left_rng)
    right = augment(_window(True), spec, cfg, right_rng)
    assert torch.equal(left["tokens"], right["tokens"])
    assert torch.equal(left["labels"], right["labels"])
    assert torch.equal(left["ball"], right["ball"])
    assert left_rng.random() == right_rng.random()
    assert torch.equal(torch.nan_to_num(right["touch"], nan=-2), torch.nan_to_num(before, nan=-2))


def test_loss_is_unchanged_at_weight_zero_and_adds_a_finite_touch_term():
    torch = pytest.importorskip("torch")
    from aaaag.ml.train import DistillConfig, loss_terms

    torch.manual_seed(0)
    n, w, margin = 2, 24, 3
    labels = torch.randint(0, 4, (n, w))
    labels[:, 8:12] = torch.tensor([H, H, A, A])
    out = {"logits": torch.randn(n, w, 3), "ball": torch.randn(n, w, 2)}
    touch = torch.zeros(n, w)
    touch[:, 10] = 1
    touch[:, 4] = torch.nan
    batch = {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.randn(n, w, 2),
        "touch": touch,
    }
    batch["ball_ok"][:, 10] = True
    weights = torch.tensor([1.0, 1.2, 0.8])
    plain = loss_terms(out, batch, weights, margin, 0.3)
    off = loss_terms(out, batch, weights, margin, 0.3, touch_weight=0.0)
    assert torch.equal(plain["loss"], off["loss"])
    assert set(plain) == set(off) == {"loss", "ce", "aux", "acc"}
    headed = {**out, "touch": torch.randn(n, w), "switch": torch.randn(n, w)}
    still = loss_terms(headed, batch, weights, margin, 0.3, touch_weight=0.0)
    assert torch.equal(still["loss"], plain["loss"]) and "touch" not in still
    on = loss_terms(headed, batch, weights, margin, 0.3, touch_weight=0.2, touch_pos_weight=10.0)
    assert "touch" in on and torch.isfinite(on["touch"])
    assert torch.equal(on["loss"], plain["loss"] + 0.2 * on["touch"])
    assert not torch.equal(on["loss"], plain["loss"])

    # The margin is outside the loss: changing it does not move the term.
    moved = touch.clone()
    moved[:, :margin] = 1 - torch.nan_to_num(moved[:, :margin])
    moved[:, -margin:] = 1
    changed = loss_terms(headed, {**batch, "touch": moved}, weights, margin, 0.3, touch_weight=0.2)
    assert torch.equal(changed["touch"], on["touch"])

    ignored = {**batch, "touch": torch.full((n, w), torch.nan)}
    zero = loss_terms(headed, ignored, weights, margin, 0.3, touch_weight=0.2)
    assert zero["touch"].item() == 0.0
    assert torch.equal(zero["loss"], plain["loss"])

    both = loss_terms(
        headed,
        batch,
        weights,
        margin,
        0.3,
        switch_weight=0.2,
        switch_tol_frames=3,
        switch_pos_weight=20.0,
        touch_weight=0.2,
        touch_pos_weight=10.0,
    )
    assert torch.equal(both["loss"], plain["loss"] + 0.2 * both["sw"] + 0.2 * both["touch"])

    teacher = torch.randn(n, w, 3)
    distill = DistillConfig(teacher_runs=("teacher",), alpha=0.5, temperature=2.0)
    base = loss_terms(out, batch, weights, margin, 0.3, teacher_logits=teacher, distill=distill)
    distilled = loss_terms(
        headed,
        batch,
        weights,
        margin,
        0.3,
        teacher_logits=teacher,
        distill=distill,
        touch_weight=0.4,
    )
    assert "kl" in distilled and torch.isfinite(distilled["touch"])
    assert torch.equal(distilled["loss"], base["loss"] + 0.4 * distilled["touch"])


def test_numpy_engine_matches_torch_on_the_touch_logits_including_views():
    torch = pytest.importorskip("torch")
    from aaaag.ml.geometry import SWAP_CLASSES, symmetric
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.np_infer import VIEWS, NumpyTeamNet, _kept_scalars, parameter_layout
    from aaaag.ml.spec import Col

    torch.manual_seed(0)
    spec = TokenSpec(tokens_per_team=2)
    cfg = ModelConfig(
        tokens_per_team=2,
        n_features=spec.n_features,
        d_model=8,
        n_heads=2,
        n_blocks=1,
        ffn_mult=2,
        dilations=(1,),
        kernel=3,
        dropout=0.0,
        touch_head=True,
        switch_head=True,
    )
    module = TeamNet(cfg).eval().double()
    assert [name for name, _shape in parameter_layout(cfg.to_dict())] == list(module.state_dict())
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in module.state_dict().items()}
    net = NumpyTeamNet(cfg.to_dict(), spec, arrays)
    batch = torch.randn(2, 6, cfg.n_tokens, cfg.n_features)
    batch[..., Col.VALID] = 1.0
    with torch.inference_mode():
        ref = module(batch.double())
    got = net.forward(batch.numpy())
    for key in ("logits", "ball", "switch", "touch"):
        assert np.max(np.abs(got[key] - ref[key].numpy())) <= 1e-5

    window = batch[0].numpy()
    _logits, scalars = _kept_scalars(net, spec, window, 0, 0, window.shape[0], 4, ("touch",))
    total_t = None
    total_z = None
    x = torch.from_numpy(window).double()[None]
    with torch.inference_mode():
        for flip_y, swap in VIEWS[4]:
            inp = symmetric(x, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else x
            forward = module(inp)
            z = forward["logits"][0]
            z = z[:, SWAP_CLASSES] if swap else z
            total_z = z if total_z is None else total_z + z
            value = forward["touch"][0]
            total_t = value if total_t is None else total_t + value
    assert np.max(np.abs(scalars["touch"] - (total_t / 4).numpy())) <= 1e-5
    assert np.max(np.abs(_logits - (total_z / 4).numpy().astype(np.float32))) <= 1e-5


def _train_run(*, touch_head: bool | None = None, touch_weight: float | None = None):
    from aaaag.ml.data import DegradeMix
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.train import RunConfig, TrainConfig

    model = dict(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
    if touch_head is not None:
        model["touch_head"] = touch_head
    train = dict(
        seed=0,
        steps=2,
        refresh_steps=2,
        batch_size=2,
        window=64,
        warmup_steps=1,
        lr=3e-3,
        degrade=DegradeMix(version=None),
        amp=False,
    )
    if touch_weight is not None:
        train["touch_weight"] = touch_weight
    return RunConfig(token=TokenSpec(), model=ModelConfig(**model), train=TrainConfig(**train))


def _one_match():
    from aaaag.synthetic import SyntheticConfig, generate

    spec = TokenSpec()
    bundle = generate(seed=7, config=SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.1))
    return TrainMatch.from_bundle(bundle, spec)


def test_head_off_training_keeps_today_checkpoint_keys(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_payload
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.train import train

    match = _one_match()
    assert match.touch is None

    def once(out, **kwargs):
        lines: list[str] = []
        final = train(
            _train_run(**kwargs),
            [match],
            out,
            device="cpu",
            log=lines.append,
            resume=False,
            threads=1,
        )
        return final, lines

    plain, plain_log = once(tmp_path / "plain")
    explicit, explicit_log = once(tmp_path / "explicit", touch_head=False, touch_weight=0.0)
    assert match.touch is None
    assert all(" touch " not in line for line in plain_log + explicit_log)
    left, right = load_payload(plain), load_payload(explicit)
    assert list(left["state_dict"]) == list(right["state_dict"])
    for key in left["state_dict"]:
        assert torch.equal(left["state_dict"][key], right["state_dict"][key])
    fresh = TeamNet(
        ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
    )
    assert list(left["state_dict"]) == list(fresh.state_dict())
    assert "touch.weight" not in left["state_dict"] and "touch.bias" not in left["state_dict"]
    assert "touch_head" not in left["model_config"]
    for key in ("touch_weight", "touch_tol_frames", "touch_pos_weight"):
        assert key not in left["meta"]["run"]["train"]
        assert key not in right["meta"]["run"]["train"]
    assert "touch" not in left["meta"]["history"][0]


def test_a_few_steps_train_and_the_touch_term_is_finite(tmp_path):
    pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint, load_payload
    from aaaag.ml.train import train

    match = _one_match()
    lines: list[str] = []
    final = train(
        _train_run(touch_head=True, touch_weight=0.2),
        [match],
        tmp_path,
        device="cpu",
        log=lines.append,
        resume=False,
        threads=1,
    )
    assert match.touch is None  # the run builds the target on its own copy
    assert any(" touch " in line for line in lines)
    payload = load_payload(final)
    assert payload["model_config"]["touch_head"] is True
    assert payload["meta"]["run"]["train"]["touch_weight"] == 0.2
    assert payload["meta"]["run"]["train"]["touch_tol_frames"] == 2
    assert payload["meta"]["run"]["train"]["touch_pos_weight"] == 10.0
    row = payload["meta"]["history"][0]
    assert np.isfinite(row["touch"]) and np.isfinite(row["loss"])
    assert list(payload["state_dict"])[-2:] == ["touch.weight", "touch.bias"]
    loaded = load_checkpoint(final)
    assert loaded.model.cfg.touch_head
    assert "touch.weight" in loaded.model.state_dict()
