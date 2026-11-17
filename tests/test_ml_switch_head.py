"""A learned switch head and the bonus it gives an H↔A cut in the semi-Markov decoder.

Off by default. With the head and the weight off, the network, the loss, the evidence file and
the decoded labels stay as they are. Torch tests skip when torch is not installed. The contract,
fusion, cache and decoder tests are numpy only.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
import yaml

from aaaag.contracts import A, D, Evidence, H, MatchBundle, MatchMeta, Period, Tracking
from aaaag.decoders.dead_gated_hsmm import dead_gated_hsmm
from aaaag.decoders.hsmm import _period_switch_bonus, hsmm
from aaaag.fusion.factorised import factorised
from aaaag.paths import CONFIGS_DIR
from aaaag.pipeline import (
    PipelineConfig,
    Step,
    _evidence,
    _load_cached_evidence,
    _prepare_decoder,
    _save_cached_evidence,
)

FPS = 25


def _meta(
    n: int, source: str = "synthetic", periods: tuple[Period, ...] | None = None
) -> MatchMeta:
    return MatchMeta(
        "synthetic",
        source,
        n,
        FPS,
        105.0,
        68.0,
        periods or (Period(1, 0, n),),
    )


def _bundle(n: int = 40, source: str = "synthetic", gap: bool = False) -> MatchBundle:
    periods = (Period(1, 0, 12), Period(2, 18, n)) if gap else (Period(1, 0, n),)
    meta = _meta(n, source, periods)
    xy = np.zeros((n, 2, 2), np.float32)
    xy[:, 1, 0] = 8.0
    ball = np.zeros((n, 2), np.float32)
    teams = np.array([H, A], np.int8)
    return MatchBundle(Tracking(meta, xy, teams, ball).validate())


def _posterior_from_team_logit(lt: np.ndarray, p_dead: float = 0.02) -> np.ndarray:
    p_home = 1.0 / (1.0 + np.exp(-lt))
    alive = 1.0 - p_dead
    proba = np.stack([alive * p_home, alive * (1.0 - p_home), np.full(len(lt), p_dead)], axis=1)
    return proba.astype(np.float32)


def _ha_boundary(labels: np.ndarray) -> int | None:
    changes = np.flatnonzero((labels[:-1] == H) & (labels[1:] == A))
    if len(changes) == 0:
        return None
    return int(changes[0] + 1)


def _npy(path, name: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def _rows(n: int = 6) -> np.ndarray:
    return np.full((n, 3), 1.0 / 3.0, np.float32)


# -- contract, cache, fusion, decoder, producer (no torch) ---------------------------------------


def test_evidence_channels_validate_and_default_to_empty():
    n = 4
    proba = _rows(n)
    plain = Evidence("e", proba).validate(n)
    assert plain.channels == {}
    channel = np.arange(n, dtype=np.float32)
    carrying = Evidence("e", proba, {"switch": channel}).validate(n)
    assert np.array_equal(carrying.channels["switch"], channel)
    with pytest.raises(ValueError, match="channel"):
        Evidence("e", proba, {"switch": np.zeros((n, 1), np.float32)}).validate(n)
    with pytest.raises(ValueError, match="float32"):
        Evidence("e", proba, {"switch": np.zeros(n, np.float64)}).validate(n)


def test_cache_without_channels_matches_a_proba_only_archive(tmp_path):
    proba = _rows(5)
    proba[0] = np.nan
    path = tmp_path / "new.npz"
    _save_cached_evidence(path, Evidence("n", proba))
    plain = tmp_path / "plain.npz"
    np.savez(plain, proba=proba)
    assert zipfile.ZipFile(path).namelist() == ["proba.npy"]
    assert _npy(path, "proba.npy") == _npy(plain, "proba.npy")
    loaded = _load_cached_evidence(path, "n", 5)
    assert loaded.channels == {}
    assert np.array_equal(loaded.proba, proba, equal_nan=True)
    old = tmp_path / "old.npz"
    np.savez(old, proba=proba)
    from_old = _load_cached_evidence(old, "n", 5)
    assert from_old.channels == {}
    assert np.array_equal(from_old.proba, proba, equal_nan=True)


def test_cache_round_trip_keeps_channels_and_the_pipeline_uses_it(tmp_path, monkeypatch):
    proba = _rows(5)
    channel = np.linspace(-1.0, 2.0, 5, dtype=np.float32)
    path = tmp_path / "with.npz"
    _save_cached_evidence(path, Evidence("n", proba, {"switch": channel}))
    loaded = _load_cached_evidence(path, "n", 5)
    assert loaded.channels["switch"].dtype == np.float32
    assert np.array_equal(loaded.channels["switch"], channel)
    assert np.array_equal(loaded.proba, proba)

    monkeypatch.setenv("AAAAG_CACHE_DIR", str(tmp_path / "cache"))
    bundle = _bundle(source="metrica")
    first = _evidence(Step("nearest_player"), bundle, True)
    files = list((tmp_path / "cache").rglob("*.npz"))
    assert len(files) == 1
    assert zipfile.ZipFile(files[0]).namelist() == ["proba.npy"]
    again = _evidence(Step("nearest_player"), bundle, True)
    assert again.channels == {}
    assert np.array_equal(again.proba, first.proba, equal_nan=True)


def test_factorised_forwards_and_averages_the_switch_channel():
    n = 6
    meta = _meta(n)
    proba = _rows(n)
    left = np.array([1, 3, np.nan, 5, 7, 9], np.float32)
    right = np.array([3, 1, 4, np.nan, 1, 1], np.float32)
    dead = Evidence("ball", proba.copy())
    net = Evidence("net", proba.copy(), {"switch": left})
    rule = Evidence("rule", proba.copy(), {"switch": right})
    fused = factorised([dead, net, rule], meta, dead_from=["ball"], team_from=["net", "rule"])
    expect = np.array([2, 2, 4, 5, 4, 5], np.float32)
    assert np.allclose(fused.channels["switch"], expect, equal_nan=True)
    one = factorised([dead, net], meta, dead_from=["ball"], team_from=["net"])
    assert np.array_equal(one.channels["switch"], left, equal_nan=True)
    bare_net = Evidence("net", proba.copy())
    bare_rule = Evidence("rule", proba.copy())
    plain = factorised(
        [dead, bare_net, bare_rule], meta, dead_from=["ball"], team_from=["net", "rule"]
    )
    assert not hasattr(plain, "channels")
    assert np.array_equal(np.asarray(fused), plain, equal_nan=True)
    only_dead = Evidence("ball", proba.copy(), {"switch": np.ones(n, np.float32)})
    hidden = factorised([only_dead, bare_net], meta, dead_from=["ball"], team_from=["net"])
    assert not hasattr(hidden, "channels")


def test_prepare_decoder_is_a_no_op_without_channels():
    proba = _rows(4)
    params = {"refine_s": 0.0}
    got_params, got = _prepare_decoder(hsmm, params, proba)
    assert got_params is params and got is proba


def test_block_bonus_clips_and_treats_nan_as_zero():
    switch = np.array([0.0, 10.0, np.nan, -9.0, 1.0], np.float32)
    bonus = _period_switch_bonus(switch, block=3, weight=2.0)
    assert bonus[0] == 0.0
    assert bonus[1] == pytest.approx(2.0 * 6.9)
    assert bonus[2] == pytest.approx(2.0)


def test_weight_zero_or_no_channel_matches_the_decoder_without_the_argument():
    n = 360
    meta = _meta(n)
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    proba[::17] = np.nan
    switch = rng.normal(size=n).astype(np.float32)
    base = hsmm(proba, meta)
    assert np.array_equal(hsmm(proba, meta, switch_weight=0.0), base)
    assert np.array_equal(hsmm(proba, meta, switch_weight=0.0, switch=switch), base)
    assert np.array_equal(hsmm(proba, meta, switch_weight=2.0), base)
    gated = dead_gated_hsmm(proba, meta)
    assert np.array_equal(dead_gated_hsmm(proba, meta, switch_weight=0.0, switch=switch), gated)
    assert np.array_equal(dead_gated_hsmm(proba, meta, switch_weight=2.0), gated)


def test_a_strong_switch_logit_moves_the_boundary_onto_that_block():
    n, block = 250, 5
    meta = _meta(n)
    lt = np.empty(n)
    lt[:100] = 3.0
    lt[100:150] = np.linspace(3.0, -3.0, 50, endpoint=False)
    lt[150:] = -3.0
    proba = _posterior_from_team_logit(lt)
    kw = dict(
        lam_team=0.2,
        lam_dead=0.05,
        beta=0.0,
        refine_s=0.0,
        duration_prior="flat",
        block=block,
        min_spell_s=0.0,
    )
    base = hsmm(proba, meta, **kw)
    natural = _ha_boundary(base)
    assert natural is not None and natural % block == 0
    target = natural - 4 * block
    assert target >= 2 * block
    switch = np.zeros(n, np.float32)
    switch[target - block : target] = 20.0
    moved = hsmm(proba, meta, switch_weight=8.0, switch=switch, **kw)
    assert _ha_boundary(moved) == target
    gated_kw = dict(lam_team=0.2, beta=0.0, refine_s=0.0, duration_prior="flat", block=block)
    gated = dead_gated_hsmm(proba, meta, switch_weight=8.0, switch=switch, **gated_kw)
    assert _ha_boundary(gated) == target


def test_a_switch_logit_inside_confident_home_does_not_create_a_spell():
    n, block = 200, 5
    meta = _meta(n)
    proba = _posterior_from_team_logit(np.full(n, 5.0), p_dead=0.02)
    switch = np.zeros(n, np.float32)
    switch[80:85] = 20.0
    kw = dict(
        lam_team=0.6,
        lam_dead=0.1,
        beta=0.0,
        refine_s=0.0,
        duration_prior="flat",
        block=block,
    )
    base = hsmm(proba, meta, **kw)
    moved = hsmm(proba, meta, switch_weight=1.0, switch=switch, **kw)
    assert np.all(base == H)
    assert np.array_equal(moved, base)


def test_pipeline_passes_the_switch_channel(monkeypatch):
    from aaaag.fusion.factorised import _carry
    from aaaag.pipeline import run
    from aaaag.registry import DECODERS, FUSION, Component

    bundle = _bundle()
    fusion = FUSION.get("factorised")

    def wrapped(evidence, meta, **params):
        proba = fusion.fn(evidence, meta, **params)
        switch = np.full(meta.n_frames, 1.25, np.float32)
        return _carry(np.asarray(proba, dtype=np.float32), {"switch": switch})

    monkeypatch.setitem(
        FUSION._items,
        "factorised",
        Component(fusion.kind, fusion.name, fusion.version, wrapped, dict(fusion.example)),
    )
    decoder = DECODERS.get("hsmm")
    seen: dict = {}

    def spy(proba, meta, *, switch=None, switch_weight=0.0, **kwargs):
        seen["switch"] = switch
        seen["weight"] = switch_weight
        return decoder.fn(proba, meta, switch=switch, switch_weight=0.0, **kwargs)

    monkeypatch.setitem(
        DECODERS._items,
        "hsmm",
        Component(decoder.kind, decoder.name, decoder.version, spy, dict(decoder.example)),
    )
    config = PipelineConfig.from_dict(
        {
            "name": "switch-wire",
            "description": "",
            "evidence": [{"name": "nearest_player", "params": {}}],
            "fusion": {
                "name": "factorised",
                "params": {"dead_from": ["nearest_player"], "team_from": ["nearest_player"]},
            },
            "decoder": {
                "name": "hsmm",
                "params": {
                    "switch_weight": 1.0,
                    "refine_s": 0.0,
                    "beta": 0.0,
                    "duration_prior": "flat",
                },
            },
        }
    )
    run(config, bundle, use_cache=False)
    assert seen["weight"] == 1.0
    assert seen["switch"] is not None
    assert seen["switch"].shape == (bundle.meta.n_frames,)
    assert seen["switch"].dtype == np.float32


def test_numpy_producer_attaches_the_switch_channel_and_omits_it_when_off(tmp_path):
    from aaaag.evidence.possession.learned_team import learned_team
    from aaaag.ml.np_infer import dump_numpy_weights, parameter_layout
    from aaaag.ml.spec import FEATURE_VERSION, TokenSpec

    spec = TokenSpec(tokens_per_team=1)
    base = {
        "tokens_per_team": 1,
        "n_features": spec.n_features,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1],
        "kernel": 3,
        "dropout": 0.0,
    }
    rng = np.random.default_rng(0)

    def write(path, config):
        arrays = {
            name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
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
    write(off, base)
    plain = learned_team(bundle, weights=str(off), backend="numpy", views=1)
    assert plain.channels == {}
    on = tmp_path / "on.json"
    write(on, {**base, "switch_head": True})
    carrying = learned_team(bundle, weights=str(on), backend="numpy", views=2, output="joint")
    channel = carrying.channels["switch"]
    played = bundle.meta.played_mask()
    assert channel.shape == (bundle.meta.n_frames,) and channel.dtype == np.float32
    assert np.isnan(channel[~played]).all()
    assert np.isfinite(channel[played]).all()
    names = [name for name, _shape in parameter_layout(base)]
    headed = [name for name, _shape in parameter_layout({**base, "switch_head": True})]
    assert headed[: len(names)] == names
    assert headed[-2:] == ["switch.weight", "switch.bias"]
    # The shipped pipeline runs the numpy engine with a process pool: the pooled chunks must give
    # the serial switch channel and posterior exactly.
    pooled = learned_team(
        bundle, weights=str(on), backend="numpy", views=2, output="joint", chunk=8, workers=2
    )
    serial = learned_team(
        bundle, weights=str(on), backend="numpy", views=2, output="joint", chunk=8, workers=1
    )
    assert np.array_equal(pooled.proba, serial.proba, equal_nan=True)
    assert np.array_equal(pooled.channels["switch"], serial.channels["switch"], equal_nan=True)


def test_views_average_the_switch_logit_and_do_not_swap_it():
    from aaaag.ml.geometry import SWAP_CLASSES, symmetric
    from aaaag.ml.np_infer import VIEWS, NumpyTeamNet, _kept_switch, parameter_layout
    from aaaag.ml.spec import Col, TokenSpec

    spec = TokenSpec(tokens_per_team=1)
    cfg = {
        "tokens_per_team": 1,
        "n_features": spec.n_features,
        "d_model": 8,
        "n_heads": 2,
        "n_blocks": 1,
        "ffn_mult": 2,
        "dilations": [1],
        "kernel": 3,
        "dropout": 0.0,
        "switch_head": True,
    }
    rng = np.random.default_rng(1)
    arrays = {
        name: (rng.standard_normal(shape) * 0.05).astype(np.float32)
        for name, shape in parameter_layout(cfg)
    }
    net = NumpyTeamNet(cfg, spec, arrays)
    window = rng.standard_normal((12, net.n_tokens, net.n_features))
    window[..., Col.VALID] = 1.0
    logits, switch = _kept_switch(net, spec, window, 0, 0, 12, 4)
    total_z = None
    total_s = None
    for flip_y, swap in VIEWS[4]:
        x = symmetric(window, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else window
        forward = net.forward(x[None])
        z = forward["logits"][0]
        if swap:
            z = z[:, SWAP_CLASSES]
        total_z = z if total_z is None else total_z + z
        total_s = forward["switch"][0] if total_s is None else total_s + forward["switch"][0]
    assert np.allclose(logits, (total_z / 4).astype(np.float32))
    assert np.allclose(switch, (total_s / 4).astype(np.float32))


def test_switch_configs_extend_the_existing_ones():
    base = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_control_memory.yaml").read_text(encoding="utf-8")
    )
    got = yaml.safe_load((CONFIGS_DIR / "ml" / "team_cm_switch.yaml").read_text(encoding="utf-8"))
    assert got["token"] == base["token"]
    assert got["model"] == {**base["model"], "switch_head": True}
    train = dict(got["train"])
    assert train.pop("switch_weight") == 0.2
    assert train.pop("switch_tol_frames") == 3
    assert train.pop("switch_pos_weight") == 20.0
    assert train == base["train"]
    pipeline = yaml.safe_load(
        (CONFIGS_DIR / "pipelines" / "possession_learned_joint_v3.yaml").read_text(encoding="utf-8")
    )
    variant = yaml.safe_load(
        (CONFIGS_DIR / "pipelines" / "possession_learned_joint_v3_switch.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert variant["name"] == "possession_learned_joint_v3_switch"
    assert variant["evidence"] == pipeline["evidence"]
    assert variant["fusion"] == pipeline["fusion"]
    assert variant["postprocess"] == pipeline["postprocess"]
    params = dict(variant["decoder"]["params"])
    assert params.pop("switch_weight") == 1.0
    assert params == pipeline["decoder"]["params"]
    loaded = PipelineConfig.load(
        CONFIGS_DIR / "pipelines" / "possession_learned_joint_v3_switch.yaml"
    )
    assert loaded.decoder.params["switch_weight"] == 1.0


# -- torch: the head, the target, the loss, numpy agreement --------------------------------------


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
    explicit = TeamNet(ModelConfig(**tiny, switch_head=False)).eval()
    left, right = omitted.state_dict(), explicit.state_dict()
    assert list(left) == list(right)
    for key in left:
        assert torch.equal(left[key], right[key])
    x = torch.randn(2, 5, omitted.cfg.n_tokens, omitted.cfg.n_features)
    with torch.no_grad():
        a, b = omitted(x), explicit(x)
    assert set(a) == set(b) == {"logits", "ball"}
    assert torch.equal(a["logits"], b["logits"]) and torch.equal(a["ball"], b["ball"])
    assert "switch_head" not in omitted.cfg.to_dict()

    torch.manual_seed(0)
    headed = TeamNet(ModelConfig(**tiny, switch_head=True)).eval()
    shared = [key for key in left if key in headed.state_dict()]
    assert shared == list(left)
    for key in shared:
        assert torch.equal(left[key], headed.state_dict()[key])
    assert list(headed.state_dict())[-2:] == ["switch.weight", "switch.bias"]
    with torch.no_grad():
        out = headed(x)
    assert out["switch"].shape == (2, 5)
    assert torch.equal(out["logits"], a["logits"]) and torch.equal(out["ball"], a["ball"])

    bare = RunConfig()
    assert "switch_head" not in bare.to_dict()["model"]
    for key in ("switch_weight", "switch_tol_frames", "switch_pos_weight"):
        assert key not in bare.to_dict()["train"]
    for name in ("team_baseline.yaml", "team_control_memory.yaml"):
        dumped = RunConfig.from_yaml(CONFIGS_DIR / "ml" / name).to_dict()
        assert "switch_head" not in dumped["model"]
        for key in ("switch_weight", "switch_tol_frames", "switch_pos_weight"):
            assert key not in dumped["train"]
    with pytest.raises(ValueError, match="switch"):
        RunConfig(train=TrainConfig(switch_weight=0.2))
    RunConfig(model=ModelConfig(switch_head=True), train=TrainConfig(switch_weight=0.2))


def test_switch_target_marks_only_a_direct_change_and_is_symmetric():
    torch = pytest.importorskip("torch")
    from aaaag.ml.spec import Col, TokenSpec
    from aaaag.ml.train import TrainConfig, augment, switch_targets

    labels = torch.tensor([[H, H, H, A, A, D, D, A]])
    target = switch_targets(labels, tol=3, margin=0)[0]
    expect = torch.tensor([1, 1, 1, 1, 1, torch.nan, torch.nan, 0], dtype=torch.float32)
    assert torch.equal(torch.nan_to_num(target, nan=-2), torch.nan_to_num(expect, nan=-2))
    margin = switch_targets(labels, tol=3, margin=1)[0]
    expect_m = expect.clone()
    expect_m[0] = torch.nan
    expect_m[-1] = torch.nan
    assert torch.equal(torch.nan_to_num(margin, nan=-2), torch.nan_to_num(expect_m, nan=-2))
    other = torch.tensor([[A, A, H, H]])
    near = switch_targets(other, tol=0, margin=0)[0]
    assert torch.equal(near, torch.tensor([0, 1, 1, 0], dtype=torch.float32))
    bridged = switch_targets(torch.tensor([[H, D, A]]), tol=5, margin=0)[0]
    assert torch.equal(torch.nan_to_num(bridged, nan=-2), torch.tensor([0.0, -2.0, 0.0]))

    class _Zeros:
        def random(self, size=None):
            return np.zeros(() if size is None else size)

    spec = TokenSpec(tokens_per_team=2)
    tokens = torch.zeros(1, 8, spec.n_tokens, spec.n_features)
    tokens[..., Col.VALID] = 1.0
    batch = {"tokens": tokens, "labels": labels.clone(), "ball": torch.zeros(1, 8, 2)}
    before = switch_targets(batch["labels"], 3, margin=1)
    augment(batch, spec, TrainConfig(swap_teams=True, flip_y=True), _Zeros())
    after = switch_targets(batch["labels"], 3, margin=1)
    assert torch.equal(torch.nan_to_num(before, nan=-2), torch.nan_to_num(after, nan=-2))
    assert not torch.equal(batch["labels"], labels)


def test_loss_is_unchanged_at_weight_zero_and_adds_a_finite_sw_term():
    torch = pytest.importorskip("torch")
    from aaaag.ml.train import loss_terms

    torch.manual_seed(0)
    n, w, margin = 2, 24, 3
    labels = torch.randint(0, 4, (n, w))
    labels[:, 8:12] = torch.tensor([H, H, A, A])
    out = {"logits": torch.randn(n, w, 3), "ball": torch.randn(n, w, 2)}
    batch = {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.randn(n, w, 2),
    }
    batch["ball_ok"][:, 10] = True
    weights = torch.tensor([1.0, 1.2, 0.8])
    plain = loss_terms(out, batch, weights, margin, 0.3)
    off = loss_terms(out, batch, weights, margin, 0.3, switch_weight=0.0)
    assert torch.equal(plain["loss"], off["loss"])
    assert set(plain) == set(off) == {"loss", "ce", "aux", "acc"}
    for key in plain:
        assert torch.equal(plain[key], off[key])
    headed = {**out, "switch": torch.randn(n, w)}
    still = loss_terms(headed, batch, weights, margin, 0.3, switch_weight=0.0)
    assert torch.equal(still["loss"], plain["loss"]) and "sw" not in still
    on = loss_terms(
        headed,
        batch,
        weights,
        margin,
        0.3,
        switch_weight=0.2,
        switch_tol_frames=3,
        switch_pos_weight=20.0,
    )
    assert "sw" in on and torch.isfinite(on["sw"])
    assert torch.equal(on["loss"], plain["loss"] + 0.2 * on["sw"])
    assert not torch.equal(on["loss"], plain["loss"])


def test_numpy_engine_matches_torch_on_the_switch_logits():
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.np_infer import NumpyTeamNet, parameter_layout
    from aaaag.ml.spec import TokenSpec

    torch.manual_seed(0)
    cfg = ModelConfig(
        tokens_per_team=2,
        d_model=8,
        n_heads=2,
        n_blocks=1,
        ffn_mult=2,
        dilations=(1,),
        kernel=3,
        dropout=0.0,
        switch_head=True,
    )
    module = TeamNet(cfg).eval().double()
    assert [name for name, _shape in parameter_layout(cfg.to_dict())] == list(module.state_dict())
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in module.state_dict().items()}
    spec = TokenSpec(tokens_per_team=2)
    net = NumpyTeamNet(cfg.to_dict(), spec, arrays)
    batch = torch.randn(2, 6, cfg.n_tokens, cfg.n_features)
    batch[..., 0] = 1.0  # a valid flag somewhere in the first column is enough for a finite pass
    with torch.inference_mode():
        ref = module(batch.double())
    got = net.forward(batch.numpy())
    for key in ("logits", "ball", "switch"):
        assert np.max(np.abs(got[key] - ref[key].numpy())) <= 1e-6
