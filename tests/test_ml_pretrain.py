"""Self-supervised pretraining and a checkpoint init for TeamNet (E52).

Off by default: a run that leaves `pretrain` false and `init_from` unset matches one that does not
mention them. Torch tests import torch themselves. Synthetic matches only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aaaag.contracts import X  # noqa: E402
from aaaag.ml.checkpoint import load_payload, save_checkpoint  # noqa: E402
from aaaag.ml.data import (  # noqa: E402
    DegradeMix,
    SetEntry,
    TrainMatch,
    class_weights,
)
from aaaag.ml.model import ModelConfig, build_model  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import RunConfig, TrainConfig, seed_everything, train  # noqa: E402
from aaaag.paths import CONFIGS_DIR  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
PRE_MODEL = ModelConfig(
    d_model=16,
    n_heads=2,
    n_blocks=1,
    dilations=(1, 2, 4),
    dropout=0.0,
    possessor_head=True,
    switch_head=True,
    touch_head=True,
)
SHORT = SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.05)

# A labelled run would open the teacher path. Pretrain must not.
_TINY_PRETRAIN_YAML = """
model:
  d_model: 16
  n_heads: 2
  n_blocks: 1
  dilations: [1, 2, 4]
  dropout: 0.0
  possessor_head: true
  switch_head: true
  touch_head: true
train:
  seed: 3
  steps: 2
  refresh_steps: 2
  batch_size: 2
  window: 64
  warmup_steps: 1
  lr: 0.003
  degrade:
    version: null
  amp: false
  aux_weight: 0.2
  possessor_weight: 0.2
  switch_weight: 0.2
  touch_weight: 0.2
  pretrain: true
  extra_matches: [synthetic/1, synthetic/2]
  extra_labels: none
  distill:
    teacher_runs: [not-a-teacher-run]
    alpha: 0.4
"""


def _short(seed: int, *, extra: bool = False) -> TrainMatch:
    return TrainMatch.from_bundle(generate(seed=seed, config=SHORT), SPEC, extra=extra)


def _plain(**changes: object) -> RunConfig:
    fields = {
        "seed": 3,
        "steps": 2,
        "refresh_steps": 2,
        "batch_size": 2,
        "window": 64,
        "warmup_steps": 1,
        "lr": 3e-3,
        "degrade": DegradeMix(version=None),
        "amp": False,
    }
    fields.update(changes)
    return RunConfig(token=SPEC, model=TINY, train=TrainConfig(**fields))


def _fit(run: RunConfig, matches: list[TrainMatch], out) -> tuple[object, list[str]]:
    lines: list[str] = []
    final = train(run, matches, out, device="cpu", log=lines.append, resume=False, threads=1)
    return final, lines


def _equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    """One real `python -m aaaag.ml train` on two synthetic label-free extras."""
    import aaaag.ml.data as data_mod
    import aaaag.ml.train as train_mod
    from aaaag.ml.cli import main

    out = tmp_path_factory.mktemp("pretrain")
    cfg_path = out / "run.yaml"
    cfg_path.write_text(_TINY_PRETRAIN_YAML, encoding="utf-8")
    calls = {"weights": 0, "touch": 0, "idsse": 0}
    real_weights = train_mod.class_weights
    real_touch = train_mod.with_touch_target
    real_idsse = data_mod.load_train_matches

    def spy_weights(matches):
        calls["weights"] += 1
        return real_weights(matches)

    def spy_touch(*args, **kwargs):
        calls["touch"] += 1
        return real_touch(*args, **kwargs)

    def spy_idsse(*_args, **_kwargs):
        calls["idsse"] += 1
        raise AssertionError("IDSSE matches must not be loaded during pretraining")

    train_mod.class_weights = spy_weights
    train_mod.with_touch_target = spy_touch
    data_mod.load_train_matches = spy_idsse
    try:
        main(
            [
                "train",
                "--run",
                str(cfg_path),
                "--holdout",
                "none",
                "--out",
                str(out),
                "--device",
                "cpu",
                "--threads",
                "1",
                "--no-resume",
            ]
        )
    finally:
        train_mod.class_weights = real_weights
        train_mod.with_touch_target = real_touch
        data_mod.load_train_matches = real_idsse
    return {"final": out / "final.pt", "calls": calls}


def test_a_pretrain_run_writes_final_and_skips_the_labelled_losses(pretrained):
    final = pretrained["final"]
    assert final.is_file()
    assert pretrained["calls"] == {"weights": 0, "touch": 0, "idsse": 0}
    payload = load_payload(final)
    assert payload["meta"]["train_matches"] == ["SYN-1", "SYN-2"]
    saved = payload["meta"]["run"]["train"]
    assert saved["pretrain"] is True
    assert saved["extra_labels"] == "none"
    assert saved["touch_weight"] == 0.2  # configured, and treated as off for the step
    assert "distill" in saved
    row = payload["meta"]["history"][-1]
    assert "ce" not in row
    assert "sw" not in row and "touch" not in row and "kl" not in row and "swd" not in row
    # "pos" is the possessor term, the name the training log already uses.
    for key in ("loss", "aux", "pos"):
        assert np.isfinite(row[key])


def test_a_labelled_extra_and_lomo_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="extra_labels"):
        TrainConfig(
            pretrain=True,
            extra_labels="team",
            extra_matches=("synthetic/1",),
            aux_weight=0.2,
        )
    with pytest.raises(ValueError, match="aux_weight"):
        TrainConfig(
            pretrain=True,
            extra_labels="none",
            extra_matches=("synthetic/1",),
            aux_weight=0.0,
            possessor_weight=0.0,
        )
    calls = {"n": 0}

    def spy(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("a pretraining run must not load matches for leave-one-match-out")

    monkeypatch.setattr("aaaag.ml.lomo.load_train_matches", spy)
    monkeypatch.setattr("aaaag.ml.lomo.load_extra_matches", spy)
    from aaaag.ml.lomo import run_lomo

    run = RunConfig(
        train=TrainConfig(
            pretrain=True,
            extra_labels="none",
            extra_matches=("synthetic/1", "synthetic/2"),
            aux_weight=0.2,
        )
    )
    with pytest.raises(ValueError, match="no folds"):
        run_lomo(run, "J03WMX", tmp_path / "lomo", device="cpu", threads=1, log=lambda _line: None)
    assert calls["n"] == 0


def test_finetune_starts_from_the_pretrained_weights(pretrained, tmp_path, monkeypatch):
    box: dict = {}
    real_build = build_model

    def spy(arch, cfg):
        model = real_build(arch, cfg)

        def hook(module, _inputs):
            if "snap" not in box:
                box["snap"] = {
                    key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                }

        model.register_forward_pre_hook(hook)
        return model

    monkeypatch.setattr("aaaag.ml.train.build_model", spy)
    source = str(pretrained["final"])
    run = RunConfig(
        token=SPEC,
        model=PRE_MODEL,
        train=TrainConfig(
            seed=3,
            steps=1,
            refresh_steps=1,
            batch_size=2,
            window=64,
            warmup_steps=1,
            lr=3e-3,
            degrade=DegradeMix(version=None),
            amp=False,
            init_from=source,
        ),
    )
    final, _lines = _fit(run, [_short(7), _short(8)], tmp_path / "finetune")
    ckpt = load_payload(pretrained["final"])["state_dict"]
    snap = box["snap"]
    assert snap.keys() == ckpt.keys()
    # Before step 1 every parameter is the pretrained one, not a fresh init.
    assert all(torch.equal(snap[key], ckpt[key]) for key in snap)
    seed_everything(3)
    fresh = real_build("team_net", PRE_MODEL).state_dict()
    assert any(not torch.equal(snap[key], fresh[key]) for key in snap)
    saved = load_payload(final)
    assert saved["meta"]["run"]["train"]["init_from"] == source


def test_a_mismatching_architecture_or_token_spec_raises(tmp_path):
    ckpt = tmp_path / "init.pt"
    torch.manual_seed(0)
    save_checkpoint(ckpt, build_model("team_net", TINY), SPEC, {})
    run = _plain(init_from=str(ckpt))
    with pytest.raises(ValueError, match="architecture"):
        train(
            replace(run, model=replace(TINY, d_model=32)),
            [],
            tmp_path / "arch",
            device="cpu",
            log=lambda _line: None,
            resume=False,
            threads=1,
        )
    with pytest.raises(ValueError, match="token spec"):
        train(
            replace(run, token=TokenSpec(ref_scale_m=15.0)),
            [],
            tmp_path / "token",
            device="cpu",
            log=lambda _line: None,
            resume=False,
            threads=1,
        )
    missing = _plain(init_from="PRETRAIN_CHECKPOINT")
    with pytest.raises(FileNotFoundError, match="init_from"):
        train(
            missing,
            [],
            tmp_path / "missing",
            device="cpu",
            log=lambda _line: None,
            resume=False,
            threads=1,
        )


def test_flags_off_match_a_run_that_does_not_set_them(tmp_path):
    matches = [_short(7), _short(8)]
    plain_final, plain_lines = _fit(_plain(), matches, tmp_path / "plain")
    off_final, off_lines = _fit(_plain(pretrain=False, init_from=None), matches, tmp_path / "off")
    plain_payload, off_payload = load_payload(plain_final), load_payload(off_final)
    assert plain_payload["state_dict"].keys() == off_payload["state_dict"].keys()
    for key in plain_payload["state_dict"]:
        assert torch.equal(plain_payload["state_dict"][key], off_payload["state_dict"][key])
    assert plain_payload["meta"]["run"] == off_payload["meta"]["run"]
    assert "pretrain" not in plain_payload["meta"]["run"]["train"]
    assert "init_from" not in plain_payload["meta"]["run"]["train"]
    learned = ("loss", "ce", "aux", "acc")
    for left, right in zip(
        plain_payload["meta"]["history"], off_payload["meta"]["history"], strict=True
    ):
        assert {key: left[key] for key in learned} == {key: right[key] for key in learned}
    plain_last, off_last = (
        load_payload(plain_final.parent / "last.pt"),
        load_payload(off_final.parent / "last.pt"),
    )
    assert _equal(plain_last["optimizer"], off_last["optimizer"])
    plain_line = next(line for line in plain_lines if line.startswith("train "))
    off_line = next(line for line in off_lines if line.startswith("train "))
    assert plain_line == off_line and "pretrain" not in plain_line


def test_defaults_round_trip_and_e52_accepts_the_placeholder():
    assert "pretrain" not in RunConfig().to_dict()["train"]
    assert "init_from" not in RunConfig().to_dict()["train"]
    bare = RunConfig()
    explicit = RunConfig(train=TrainConfig(pretrain=False, init_from=None))
    assert bare.to_dict() == explicit.to_dict()
    kept = RunConfig(train=TrainConfig(init_from="some/final.pt"))
    assert kept.to_dict()["train"]["init_from"] == "some/final.pt"
    assert "pretrain" not in kept.to_dict()["train"]
    assert RunConfig.from_dict(kept.to_dict()) == kept

    base = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_both_smooth_wide.yaml")
    pre = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e52_pretrain.yaml")
    fine = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e52_finetune.yaml")
    e35 = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e35a.yaml")
    assert pre.model == base.model and pre.token == base.token
    assert pre.train.pretrain is True
    assert pre.train.steps == 3000
    assert pre.train.switch_weight == 0
    assert pre.train.aux_weight == base.train.aux_weight
    assert pre.train.possessor_weight == base.train.possessor_weight
    assert pre.train.extra_labels == "none"
    assert pre.train.extra_weight == 1.0
    assert pre.train.extra_degrade is False
    assert pre.train.extra_matches[:3] == ("metrica/1", "metrica/2", "metrica/3")
    assert pre.train.extra_matches[3:] == e35.train.extra_matches
    assert pre.train == replace(
        base.train,
        steps=3000,
        switch_weight=0.0,
        pretrain=True,
        extra_matches=pre.train.extra_matches,
        extra_labels="none",
        extra_weight=1.0,
        extra_degrade=False,
    )
    assert RunConfig.from_dict(pre.to_dict()) == pre
    assert fine.train.init_from == "PRETRAIN_CHECKPOINT"
    assert "pretrain" not in fine.to_dict()["train"]
    assert replace(fine.train, init_from=None) == base.train
    assert fine.model == base.model and fine.token == base.token
    assert RunConfig.from_dict(fine.to_dict()) == fine


def test_class_weights_and_sampling_survive_a_roster_with_no_labels():
    from aaaag.ml.train import DeviceSet

    blank = replace(_short(2), labels=np.full_like(_short(2).labels, X))
    assert np.array_equal(class_weights([blank]), np.ones(3, dtype=np.float64))
    quiet = replace(blank, played=np.zeros(blank.played.shape, dtype=bool))
    frames = quiet.labels.shape[0]
    tokens = np.zeros((frames, 1, 1), np.float16)
    seen = np.zeros(frames, dtype=bool)
    data = DeviceSet(
        [SetEntry(quiet.match_id, tokens, tokens, seen, seen)],
        [quiet],
        torch.device("cpu"),
    )
    assert np.isfinite(data.share).all()
    assert float(data.share.sum()) == pytest.approx(1.0)


_AGGREGATE = {
    "champion": {"spell_f1_mean": 0.1},
    "learned": {"spell_f1_mean": 0.2},
    "team": {
        "learned_acc": 0.3,
        "rule_acc": 0.4,
        "learned_acc_where_rule_votes": 0.5,
        "rule_acc_where_rule_votes": 0.6,
    },
}


def test_every_lomo_fold_starts_from_the_same_checkpoint(tmp_path, monkeypatch):
    from aaaag.ml.lomo import run_lomo

    ckpt = tmp_path / "pre.pt"
    torch.manual_seed(1)
    model = build_model("team_net", TINY)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.25)
    save_checkpoint(ckpt, model, SPEC, {})
    expected = next(model.parameters()).detach().cpu().clone()
    snaps: list[torch.Tensor] = []
    real_build = build_model

    def spy(arch, cfg):
        net = real_build(arch, cfg)

        def hook(module, _inputs):
            if not getattr(module, "_saw_init", False):
                module._saw_init = True
                snaps.append(next(module.parameters()).detach().cpu().clone())

        net.register_forward_pre_hook(hook)
        return net

    store: dict[str, TrainMatch] = {}

    def fake_load(ids, spec):
        loaded = {}
        for match_id in ids:
            if match_id not in store:
                bundle = generate(seed=100 + len(store), config=SHORT)
                store[match_id] = TrainMatch.from_bundle(bundle, spec)
            loaded[match_id] = store[match_id]
        return loaded

    monkeypatch.setattr("aaaag.ml.train.build_model", spy)
    monkeypatch.setattr("aaaag.ml.lomo.load_train_matches", fake_load)
    monkeypatch.setattr(
        "aaaag.ml.lomo.evaluate_match",
        lambda *_args, **_kwargs: {"rows": [], "aggregate": _AGGREGATE},
    )
    monkeypatch.setattr("aaaag.ml.lomo.make_monitor", lambda *_args, **_kwargs: lambda _model: {})
    run = _plain(steps=1, refresh_steps=1, init_from=str(ckpt))
    run_lomo(
        run,
        "J03WMX,J03WPY",
        tmp_path / "lomo",
        seeds=(0,),
        device="cpu",
        threads=1,
        log=lambda _line: None,
    )
    assert len(snaps) == 2
    assert torch.equal(snaps[0], expected)
    assert torch.equal(snaps[1], expected)
    fold = next((tmp_path / "lomo").glob("fold-*/final.pt"))
    assert load_payload(fold)["meta"]["run"]["train"]["init_from"] == str(ckpt)
