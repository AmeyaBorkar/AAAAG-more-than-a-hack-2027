"""`{fold}` in train.init_from loads that leave-one-match-out fold's own checkpoint.

A single training run has no fold, so the placeholder is an error. A path without it is
loaded exactly as written. Synthetic matches only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aaaag.ml.checkpoint import load_payload, save_checkpoint  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch  # noqa: E402
from aaaag.ml.model import ModelConfig, build_model  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import RunConfig, TrainConfig, resolve_init_from, train  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
SHORT = SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.05)
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


def test_resolve_replaces_every_placeholder_per_fold():
    template = "runs/{fold}/fold-{fold}/final.pt"
    assert resolve_init_from(template, "J03WMX") == "runs/J03WMX/fold-J03WMX/final.pt"
    assert resolve_init_from(template, "J03WPY") == "runs/J03WPY/fold-J03WPY/final.pt"


def test_resolve_without_placeholder_is_unchanged():
    path = "runs/e80/final.pt"
    assert resolve_init_from(path, None) == path
    assert resolve_init_from(path, "J03WMX") == path
    assert resolve_init_from(None, "J03WMX") is None
    assert resolve_init_from(None, None) is None


def test_resolve_outside_leave_one_match_out_raises():
    template = "runs/fold-{fold}/final.pt"
    with pytest.raises(ValueError, match="leave-one-match-out"):
        resolve_init_from(template, None)
    with pytest.raises(ValueError, match="leave-one-match-out"):
        resolve_init_from(template, "none")
    with pytest.raises(ValueError, match="leave-one-match-out"):
        resolve_init_from(template, "")
    with pytest.raises(ValueError, match="match id"):
        resolve_init_from(template, "a/b")


def test_train_rejects_placeholder_without_a_fold(tmp_path):
    run = RunConfig(train=TrainConfig(init_from=str(tmp_path / "fold-{fold}" / "final.pt")))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="leave-one-match-out"):
        train(run, [], out, device="cpu", resume=False, threads=1, log=lambda _line: None)
    assert not out.exists()


def test_train_passes_a_concrete_path_through_unchanged(tmp_path, monkeypatch):
    seen: list[str] = []

    def spy(_model, path, _run):
        seen.append(path)
        raise RuntimeError("stop after init")

    monkeypatch.setattr("aaaag.ml.train._load_init_weights", spy)
    path = str(tmp_path / "final.pt")
    run = RunConfig(
        token=SPEC,
        model=TINY,
        train=TrainConfig(init_from=path, steps=1, refresh_steps=1),
    )
    with pytest.raises(RuntimeError, match="stop after init"):
        train(
            run,
            [],
            tmp_path / "out",
            device="cpu",
            resume=False,
            threads=1,
            log=lambda _line: None,
            fold="J03WMX",
        )
    assert seen == [path]


def test_train_resolves_the_fold_it_is_given(tmp_path, monkeypatch):
    seen: list[str] = []

    def spy(_model, path, _run):
        seen.append(path)
        raise RuntimeError("stop after init")

    monkeypatch.setattr("aaaag.ml.train._load_init_weights", spy)
    template = str(tmp_path / "fold-{fold}" / "final.pt")
    run = RunConfig(
        token=SPEC,
        model=TINY,
        train=TrainConfig(init_from=template, steps=1, refresh_steps=1),
    )
    with pytest.raises(RuntimeError, match="stop after init"):
        train(
            run,
            [],
            tmp_path / "out",
            device="cpu",
            resume=False,
            threads=1,
            log=lambda _line: None,
            fold="J03WMX",
        )
    assert seen == [str(tmp_path / "fold-J03WMX" / "final.pt")]


def _shifted(path, delta: float):
    torch.manual_seed(0)
    model = build_model("team_net", TINY)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(delta)
    save_checkpoint(path, model, SPEC, {})
    return next(model.parameters()).detach().cpu().clone()


def test_lomo_each_fold_loads_its_own_checkpoint(tmp_path, monkeypatch):
    """Fold J03WMX starts from fold-J03WMX, and J03WPY from fold-J03WPY."""
    from aaaag.ml.lomo import run_lomo

    src = tmp_path / "earlier"
    expected = {
        "J03WMX": _shifted(src / "fold-J03WMX" / "final.pt", 1.25),
        "J03WPY": _shifted(src / "fold-J03WPY" / "final.pt", -2.5),
    }
    assert not torch.equal(expected["J03WMX"], expected["J03WPY"])
    template = str(src / "fold-{fold}" / "final.pt")
    snaps: list[torch.Tensor] = []
    real_build = build_model

    def spy(arch, cfg):
        net = real_build(arch, cfg)

        def hook(module, _inputs):
            if getattr(module, "_saw_init", False):
                return
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
    run = RunConfig(
        token=SPEC,
        model=TINY,
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
            init_from=template,
        ),
    )
    lines: list[str] = []
    run_lomo(
        run,
        "J03WMX,J03WPY",
        tmp_path / "lomo",
        seeds=(0,),
        device="cpu",
        threads=1,
        log=lines.append,
    )
    assert len(snaps) == 2
    assert torch.equal(snaps[0], expected["J03WMX"])
    assert torch.equal(snaps[1], expected["J03WPY"])
    assert f"init_from {template.replace('{fold}', 'J03WMX')}" in lines
    assert f"init_from {template.replace('{fold}', 'J03WPY')}" in lines
    for fold_id in ("J03WMX", "J03WPY"):
        saved = load_payload(tmp_path / "lomo" / f"fold-{fold_id}" / "final.pt")
        assert saved["meta"]["run"]["train"]["init_from"] == template
        assert saved["meta"]["holdout"] == fold_id


def test_a_missing_saved_key_counts_as_its_default():
    from aaaag.ml.spec import TokenSpec
    from aaaag.ml.train import with_defaults

    current = TokenSpec()
    saved = {k: v for k, v in current.to_dict().items() if k != "gap_bracket"}
    assert with_defaults(saved, current) == current.to_dict()
    assert with_defaults(None, current) is None
    changed = dict(saved, tokens_per_team=current.tokens_per_team + 1)
    assert with_defaults(changed, current) != current.to_dict()
