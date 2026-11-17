"""Train on extra matches only (`python -m aaaag.ml train --extra-only`).

The flag is off by default: a run that does not pass it still asks for the IDSSE dev roster.
`--extra-only` trains on `train.extra_matches` with the normal loss and writes a checkpoint
`train.init_from` can load. `train.matches_per_set` null tokenises every match. Torch tests
import torch themselves. Synthetic matches only.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aaaag.ml.checkpoint import load_payload  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch, build_set  # noqa: E402
from aaaag.ml.model import ModelConfig, build_model  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import (  # noqa: E402
    RunConfig,
    TrainConfig,
    _roster_for_set,
    seed_everything,
    train,
)
from aaaag.sources import idsse  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
SHORT = SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.05)
MODEL = ModelConfig(
    d_model=16,
    n_heads=2,
    n_blocks=1,
    dilations=(1, 2, 4),
    dropout=0.0,
    switch_head=True,
)
_YAML = """
model:
  d_model: 16
  n_heads: 2
  n_blocks: 1
  dilations: [1, 2, 4]
  dropout: 0.0
  switch_head: true
train:
  seed: 1
  steps: 2
  refresh_steps: 2
  batch_size: 2
  window: 64
  warmup_steps: 1
  lr: 0.003
  amp: false
  aux_weight: 0.2
  switch_weight: 0.2
  extra_matches: [synthetic/1, synthetic/2]
  degrade:
    version: null
"""
_PRETRAIN_YAML = """
train:
  pretrain: true
  extra_labels: none
  extra_matches: [synthetic/1]
  aux_weight: 0.2
  degrade:
    version: null
"""


def _short(seed: int) -> TrainMatch:
    return TrainMatch.from_bundle(generate(seed=seed, config=SHORT), SPEC)


def _plain(**changes: object) -> RunConfig:
    fields = {
        "seed": 3,
        "steps": 1,
        "refresh_steps": 1,
        "batch_size": 2,
        "window": 64,
        "warmup_steps": 1,
        "lr": 3e-3,
        "degrade": DegradeMix(version=None),
        "amp": False,
    }
    fields.update(changes)
    return RunConfig(token=SPEC, model=MODEL, train=TrainConfig(**fields))


def _fit(run: RunConfig, matches: list[TrainMatch], out, log=None, meta=None):
    return train(
        run,
        matches,
        out,
        device="cpu",
        log=log or (lambda _line: None),
        resume=False,
        threads=1,
        meta=meta,
    )


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


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def extra_only_run(tmp_path_factory):
    """One real `train --extra-only` on two synthetic extras. No IDSSE load."""
    import aaaag.ml.data as data_mod
    from aaaag.ml.cli import main

    out = tmp_path_factory.mktemp("extra-only")
    cfg = out / "run.yaml"
    cfg.write_text(_YAML, encoding="utf-8")
    calls = {"idsse": 0}
    real = data_mod.load_train_matches

    def spy(*_args, **_kwargs):
        calls["idsse"] += 1
        raise AssertionError("IDSSE matches must not be loaded for --extra-only")

    data_mod.load_train_matches = spy
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            main(
                [
                    "train",
                    "--run",
                    str(cfg),
                    "--holdout",
                    "none",
                    "--extra-only",
                    "--seed",
                    "4",
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
        data_mod.load_train_matches = real
    return {"out": out, "calls": calls, "log": buf.getvalue()}


def test_extra_only_trains_labelled_extras_and_skips_idsse(extra_only_run):
    assert extra_only_run["calls"]["idsse"] == 0
    final = extra_only_run["out"] / "final.pt"
    assert final.is_file()
    payload = load_payload(final)
    assert payload["meta"]["extra_only"] is True
    assert payload["meta"]["holdout"] is None
    assert payload["meta"]["train_matches"] == ["SYN-1", "SYN-2"]
    saved = payload["meta"]["run"]["train"]
    assert saved["seed"] == 4
    assert saved["extra_matches"] == ["synthetic/1", "synthetic/2"]
    assert saved.get("extra_labels", "all") == "all"
    assert "pretrain" not in saved
    assert "matches_per_set" not in saved
    row = payload["meta"]["history"][-1]
    for key in ("loss", "ce", "aux", "sw", "acc"):
        assert key in row
        assert np.isfinite(row[key])
    log = extra_only_run["log"]
    assert "(2 extra)" in log
    assert " pretrain" not in log


def test_init_from_loads_the_extra_only_checkpoint(extra_only_run, tmp_path, monkeypatch):
    source = extra_only_run["out"] / "final.pt"
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
    roster = [_short(7), _short(8)]
    lines: list[str] = []
    final = _fit(
        RunConfig(
            token=SPEC,
            model=MODEL,
            train=TrainConfig(
                seed=4,
                steps=1,
                refresh_steps=1,
                batch_size=2,
                window=64,
                warmup_steps=1,
                lr=3e-3,
                degrade=DegradeMix(version=None),
                amp=False,
                init_from=str(source),
            ),
        ),
        roster,
        tmp_path / "finetune",
        lines.append,
        meta={"holdout": None, "train_matches": [match.match_id for match in roster]},
    )
    ckpt = load_payload(source)["state_dict"]
    snap = box["snap"]
    assert snap.keys() == ckpt.keys()
    assert all(torch.equal(snap[key], ckpt[key]) for key in snap)
    seed_everything(4)
    fresh = real_build("team_net", MODEL).state_dict()
    assert any(not torch.equal(snap[key], fresh[key]) for key in snap)
    saved = load_payload(final)
    assert saved["meta"]["run"]["train"]["init_from"] == str(source)
    assert saved["meta"]["train_matches"] == ["SYN-7", "SYN-8"]
    assert any(line.startswith(f"init_from {source}") for line in lines)


def test_extra_only_rejects_an_empty_list_before_loading(tmp_path, monkeypatch):
    from aaaag.ml.cli import main

    calls = {"idsse": 0, "extra": 0}

    def spy_idsse(*_args, **_kwargs):
        calls["idsse"] += 1
        raise AssertionError("IDSSE matches must not be loaded")

    def spy_extra(*_args, **_kwargs):
        calls["extra"] += 1
        raise AssertionError("extra matches must not be loaded when the list is empty")

    monkeypatch.setattr("aaaag.ml.data.load_train_matches", spy_idsse)
    monkeypatch.setattr("aaaag.ml.data.load_extra_matches", spy_extra)
    cfg = _write(tmp_path, "empty.yaml", "train: {}\n")
    with pytest.raises(ValueError, match="empty"):
        main(
            [
                "train",
                "--run",
                str(cfg),
                "--holdout",
                "none",
                "--extra-only",
                "--out",
                str(tmp_path / "out"),
                "--device",
                "cpu",
                "--threads",
                "1",
            ]
        )
    assert calls == {"idsse": 0, "extra": 0}


def test_extra_only_rejects_label_free_pretrain(tmp_path, monkeypatch):
    from aaaag.ml.cli import main

    def spy(*_args, **_kwargs):
        raise AssertionError("a refused run must not load matches")

    monkeypatch.setattr("aaaag.ml.data.load_train_matches", spy)
    monkeypatch.setattr("aaaag.ml.data.load_extra_matches", spy)
    cfg = _write(tmp_path, "pretrain.yaml", _PRETRAIN_YAML)
    with pytest.raises(ValueError, match="label-free"):
        main(
            [
                "train",
                "--run",
                str(cfg),
                "--holdout",
                "none",
                "--extra-only",
                "--out",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )


def test_without_the_flag_the_dev_roster_is_still_requested(tmp_path, monkeypatch):
    from aaaag.ml.cli import main

    seen: list[list[str]] = []

    def spy(ids, _spec):
        seen.append(list(ids))
        raise RuntimeError("stop before an IDSSE file is read")

    monkeypatch.setattr("aaaag.ml.data.load_train_matches", spy)
    cfg = _write(tmp_path, "run.yaml", _YAML)
    with pytest.raises(RuntimeError, match="stop before"):
        main(
            [
                "train",
                "--run",
                str(cfg),
                "--holdout",
                "none",
                "--out",
                str(tmp_path / "out"),
                "--device",
                "cpu",
                "--threads",
                "1",
                "--no-resume",
            ]
        )
    assert seen == [list(idsse.DEV_MATCHES)]


@pytest.mark.parametrize("holdout", ["not-a-match", idsse.EXAM_MATCHES[0]])
def test_extra_only_holdout_must_be_a_dev_match(tmp_path, monkeypatch, holdout):
    from aaaag.ml.cli import main

    def spy(*_args, **_kwargs):
        raise AssertionError("a refused holdout must not load a match")

    monkeypatch.setattr("aaaag.ml.data.load_train_matches", spy)
    monkeypatch.setattr("aaaag.ml.data.load_extra_matches", spy)
    monkeypatch.setattr("aaaag.ml.lomo.make_monitor", spy)
    cfg = _write(tmp_path, "run.yaml", _YAML)
    with pytest.raises(ValueError, match="dev match"):
        main(
            [
                "train",
                "--run",
                str(cfg),
                "--holdout",
                holdout,
                "--extra-only",
                "--out",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )


def test_a_held_out_dev_match_is_monitored_and_not_trained_on(tmp_path, monkeypatch):
    from aaaag.ml.cli import main

    holdout = idsse.DEV_MATCHES[0]
    monitored: list[str] = []
    loaded = {"idsse": 0}

    def spy_idsse(*_args, **_kwargs):
        loaded["idsse"] += 1
        raise AssertionError("the held-out match must not be loaded for training")

    def fake_monitor(match_id, _run, _device):
        monitored.append(match_id)
        return lambda _model: {"holdout_team_acc": 0.25}

    monkeypatch.setattr("aaaag.ml.data.load_train_matches", spy_idsse)
    monkeypatch.setattr("aaaag.ml.lomo.make_monitor", fake_monitor)
    cfg = _write(tmp_path, "run.yaml", _YAML)
    out = tmp_path / "held"
    main(
        [
            "train",
            "--run",
            str(cfg),
            "--holdout",
            holdout,
            "--extra-only",
            "--seed",
            "4",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--no-resume",
        ]
    )
    assert loaded["idsse"] == 0
    assert monitored == [holdout]
    payload = load_payload(out / "final.pt")
    assert payload["meta"]["holdout"] == holdout
    assert payload["meta"]["train_matches"] == ["SYN-1", "SYN-2"]
    assert holdout not in payload["meta"]["train_matches"]
    assert payload["meta"]["history"][-1]["monitor_holdout_team_acc"] == 0.25
    assert "ce" in payload["meta"]["history"][-1]


def test_train_help_lists_extra_only(capsys):
    from aaaag.ml.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["train", "--help"])
    assert stopped.value.code == 0
    assert "--extra-only" in capsys.readouterr().out


def test_matches_per_set_is_off_unless_set():
    assert TrainConfig().matches_per_set is None
    assert "matches_per_set" not in RunConfig().to_dict()["train"]
    with pytest.raises(ValueError, match="matches_per_set"):
        TrainConfig(matches_per_set=0)
    with pytest.raises(ValueError, match="matches_per_set"):
        TrainConfig(matches_per_set=True)
    with pytest.raises(ValueError, match="matches_per_set"):
        TrainConfig(matches_per_set=1.5)
    kept = RunConfig(train=TrainConfig(matches_per_set=4))
    assert kept.to_dict()["train"]["matches_per_set"] == 4
    assert RunConfig.from_dict(kept.to_dict()).train.matches_per_set == 4


def test_seed_index_keeps_the_roster_draw_and_rejects_a_short_list():
    mix = DegradeMix(version=None)
    full = build_set([_short(1), _short(2)], 3, 9, SPEC, mix, workers=1)
    same = build_set([_short(1), _short(2)], 3, 9, SPEC, mix, workers=1, seed_index=None)
    shifted = build_set([_short(2)], 3, 9, SPEC, mix, workers=1, seed_index=[1])
    assert np.array_equal(full[0].decoupled, same[0].decoupled)
    assert np.array_equal(full[1].decoupled, same[1].decoupled)
    assert np.array_equal(full[1].decoupled, shifted[0].decoupled)
    assert not np.array_equal(full[0].decoupled, shifted[0].decoupled)
    with pytest.raises(ValueError, match="seed_index"):
        build_set([_short(1)], 0, 0, SPEC, mix, workers=1, seed_index=[])


def test_an_unset_cap_returns_the_same_roster():
    matches = [_short(1), _short(2), _short(3)]
    roster, index = _roster_for_set(matches, None, 1, 0)
    assert roster is matches and index is None
    roster, index = _roster_for_set(matches, 3, 1, 0)
    assert roster is matches and index is None
    roster, index = _roster_for_set(matches, 8, 1, 0)
    assert roster is matches and index is None
    left, one = _roster_for_set(matches, 1, 5, 2)
    again, two = _roster_for_set(matches, 1, 5, 2)
    assert one == two and one == sorted(one) and len(one) == 1
    assert left[0] is matches[one[0]]


def test_a_cap_at_least_the_roster_matches_leaving_it_unset(tmp_path):
    matches = [_short(7), _short(8)]
    bare_lines: list[str] = []
    cap_lines: list[str] = []
    bare = _fit(_plain(), matches, tmp_path / "bare", bare_lines.append)
    capped = _fit(_plain(matches_per_set=8), matches, tmp_path / "capped", cap_lines.append)
    bare_payload, cap_payload = load_payload(bare), load_payload(capped)
    for key in bare_payload["state_dict"]:
        assert torch.equal(bare_payload["state_dict"][key], cap_payload["state_dict"][key])
    # switch_weight stays 0, so neither history has "sw": that key is written only when the
    # weight is positive. A cap at least the roster must log the same keys as leaving it unset.
    learned = ("loss", "ce", "aux", "acc")
    for left, right in zip(
        bare_payload["meta"]["history"], cap_payload["meta"]["history"], strict=True
    ):
        assert set(left) == set(right)
        assert {key: left[key] for key in learned} == {key: right[key] for key in learned}
    bare_last = load_payload(bare.parent / "last.pt")
    cap_last = load_payload(capped.parent / "last.pt")
    assert _equal(bare_last["optimizer"], cap_last["optimizer"])
    bare_line = next(line for line in bare_lines if line.startswith("train "))
    cap_line = next(line for line in cap_lines if line.startswith("train "))
    assert bare_line == cap_line
    assert not any("each set builds" in line for line in cap_lines)


def test_a_smaller_cap_builds_that_many_matches_with_roster_seeds(tmp_path, monkeypatch):
    import aaaag.ml.data as data_mod
    import aaaag.ml.train as train_mod

    group = [_short(i) for i in range(4)]
    seen: list[tuple[list[str], object]] = []

    def spy(matches, *args, **kwargs):
        seen.append(([m.match_id for m in matches], kwargs.get("seed_index")))
        return data_mod.build_set(matches, *args, **kwargs)

    monkeypatch.setattr(train_mod, "build_set", spy)
    lines: list[str] = []
    _fit(
        _plain(steps=2, refresh_steps=1, matches_per_set=2),
        group,
        tmp_path / "subset",
        lines.append,
    )
    assert len(seen) == 2
    for set_index, (chosen, index) in enumerate(seen):
        roster, expect = _roster_for_set(group, 2, 3, set_index)
        assert chosen == [match.match_id for match in roster]
        assert index == expect
        assert index == sorted(index)
    assert any("each set builds 2 of 4 matches" in line for line in lines)
