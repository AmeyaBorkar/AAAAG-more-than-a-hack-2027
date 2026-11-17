"""Extra training matches (`train.extra_matches`).

Empty by default. A run that leaves the three fields at their defaults matches a run that does not
set them: same tensors, same checkpoint. Torch tests import torch themselves; the loader does not
need it. Synthetic matches only.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from aaaag.contracts import X
from aaaag.ml.augment import AugmentMix, TeamStyle
from aaaag.ml.data import (
    DegradeMix,
    SetEntry,
    TrainMatch,
    build_entry,
    build_set,
    class_weights,
    include_extra_matches,
    load_extra_matches,
    load_train_matches,
)
from aaaag.ml.spec import TokenSpec
from aaaag.registry import SOURCES
from aaaag.sources import idsse
from aaaag.synthetic import SyntheticConfig, generate

SPEC = TokenSpec()
SMALL = SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)
# Keys of configs/ml/team_baseline.yaml as RunConfig.to_dict writes them with the extras off.
BASELINE_TRAIN_KEYS = {
    "amp",
    "augment",
    "aux_weight",
    "batch_size",
    "decouple_prob",
    "degrade",
    "final_lr_frac",
    "flip_y",
    "grad_clip",
    "lr",
    "refresh_steps",
    "seed",
    "steps",
    "swap_teams",
    "warmup_steps",
    "weight_decay",
    "window",
}


@pytest.fixture(scope="module")
def dev_pair() -> list[TrainMatch]:
    return [TrainMatch.from_bundle(generate(seed=seed, config=SMALL), SPEC) for seed in (7, 8)]


@pytest.fixture(scope="module")
def extra_pair() -> list[TrainMatch]:
    return load_extra_matches(["synthetic/1", "synthetic/2"], SPEC)


def test_load_train_matches_still_refuses_anything_outside_the_dev_roster():
    for exam_match in idsse.EXAM_MATCHES:
        with pytest.raises(ValueError, match="only the IDSSE dev matches"):
            load_train_matches([exam_match], SPEC)
    with pytest.raises(ValueError, match="only the IDSSE dev matches"):
        load_train_matches(["not-a-match"], SPEC)


def test_idsse_and_bar_ath_entries_are_refused_before_any_source_call():
    with pytest.raises(ValueError, match="idsse"):
        load_extra_matches(["idsse/J03WQQ"], SPEC)
    with pytest.raises(ValueError, match="bar-ath"):
        load_extra_matches(["bar-ath/anything"], SPEC)
    with pytest.raises(ValueError, match="source/match_id"):
        load_extra_matches(["synthetic"], SPEC)
    with pytest.raises(ValueError, match="unknown extra-match source"):
        load_extra_matches(["no-such-source/1"], SPEC)


def test_synthetic_entries_load_in_order_with_labels_including_x(extra_pair):
    assert [match.match_id for match in extra_pair] == ["SYN-1", "SYN-2"]
    assert [match.extra for match in extra_pair] == [True, True]
    assert extra_pair[0].match_id != extra_pair[1].match_id
    for match in extra_pair:
        assert match.labels is not None
        assert match.labels.shape == (match.bundle.meta.n_frames,)
        assert set(np.unique(match.labels).tolist()) <= {0, 1, 2, 3}
        assert int((match.labels == X).sum()) > 0


def test_an_unlabelled_bundle_is_refused(monkeypatch):
    blank = replace(generate(seed=0, config=SMALL), labels=None)
    real_get = SOURCES.get

    def fake_get(name: str):
        if name == "blank":
            return SimpleNamespace(fn=lambda **_kwargs: blank)
        return real_get(name)

    monkeypatch.setattr("aaaag.ml.data.SOURCES.get", fake_get)
    with pytest.raises(ValueError, match="labelled"):
        load_extra_matches(["blank/1"], SPEC)


def test_include_appends_once_and_an_empty_config_returns_the_same_list(dev_pair, monkeypatch):
    calls = {"n": 0}
    real = load_extra_matches

    def spy(entries, spec, **kwargs):
        calls["n"] += 1
        return real(entries, spec, **kwargs)

    monkeypatch.setattr("aaaag.ml.data.load_extra_matches", spy)
    roster = list(dev_pair)
    assert include_extra_matches(roster, (), SPEC) is roster
    assert calls["n"] == 0
    once = include_extra_matches(roster, ("synthetic/1", "synthetic/2"), SPEC)
    twice = include_extra_matches(once, ("synthetic/1", "synthetic/2"), SPEC)
    assert calls["n"] == 1
    assert twice is once
    assert [match.extra for match in once] == [False, False, True, True]
    assert [match.match_id for match in once if match.extra] == ["SYN-1", "SYN-2"]
    with pytest.raises(ValueError, match="extra matches"):
        include_extra_matches(once + once[-1:], ("synthetic/1",), SPEC)


def test_class_weights_see_every_training_match_and_ignore_x(dev_pair, extra_pair):
    group = (dev_pair[0], extra_pair[0])
    both = class_weights(list(group))
    counts = np.zeros(3, dtype=np.float64)
    for match in group:
        for code in range(3):
            counts[code] += int(((match.labels == code) & match.played).sum())
    share = counts / counts.sum()
    share[:2] = share[:2].mean()
    weight = 1.0 / np.maximum(share, 1e-6)
    expected = weight / float((share * weight).sum())
    assert np.allclose(both, expected)
    # X is not a class: a match whose played frames are all X does not move the weights.
    as_x = replace_labels_played(extra_pair[0], X)
    assert np.allclose(class_weights([dev_pair[0], as_x]), class_weights([dev_pair[0]]))
    as_dead = replace_labels_played(extra_pair[0], 2)
    assert not np.allclose(class_weights([dev_pair[0], as_dead]), class_weights([dev_pair[0]]))


def replace_labels_played(match: TrainMatch, code: int) -> TrainMatch:
    labels = match.labels.copy()
    labels[match.played] = code
    return replace(match, labels=labels)


def test_extra_matches_skip_degrade_unless_asked_and_augment_still_runs(
    dev_pair, extra_pair, monkeypatch
):
    import aaaag.ml.data as data_mod

    base, extra = dev_pair[0], extra_pair[0]
    mix = DegradeMix()
    calls: list[str] = []
    real = data_mod.degrade

    def wrapped(bundle, **kwargs):
        calls.append(bundle.meta.match_id)
        return real(bundle, **kwargs)

    monkeypatch.setattr(data_mod, "degrade", wrapped)
    skipped = build_set([base, extra], 0, 3, SPEC, mix, workers=1, extra_degrade=False)
    assert calls == [base.match_id]
    calls.clear()
    degraded = build_set([base, extra], 0, 3, SPEC, mix, workers=1, extra_degrade=True)
    assert calls == [base.match_id, extra.match_id]
    assert np.array_equal(skipped[0].coupled, degraded[0].coupled)
    assert np.array_equal(skipped[0].decoupled, degraded[0].decoupled)
    assert skipped[1].info["draw"] == {}
    assert degraded[1].info["draw"]["version"] == "5"
    clean = build_entry(extra, np.random.default_rng([3, 0, 1]), SPEC, DegradeMix(version=None))
    assert np.array_equal(skipped[1].coupled, clean.coupled)
    assert np.array_equal(skipped[1].decoupled, clean.decoupled)
    assert not np.array_equal(degraded[1].coupled, clean.coupled)
    threaded = build_set([base, extra], 0, 3, SPEC, mix, workers=2, extra_degrade=False)
    assert np.array_equal(threaded[1].coupled, skipped[1].coupled)
    assert threaded[1].info["draw"] == {}
    styled = build_set(
        [base, extra],
        0,
        3,
        SPEC,
        mix,
        workers=1,
        extra_degrade=False,
        augment=AugmentMix(team_style=TeamStyle(p=1.0, stretch=0.5, shift_m=8.0)),
    )
    assert styled[1].info["draw"] == {}
    assert not np.array_equal(styled[1].coupled, skipped[1].coupled)


def test_sampling_weight_scales_only_extra_matches(dev_pair):
    torch = pytest.importorskip("torch")
    from aaaag.ml.train import DeviceSet

    home = dev_pair[0]
    away = replace(dev_pair[1], extra=True)
    plain_away = dev_pair[1]
    entries = [_blank_entry(home), _blank_entry(away)]
    device = torch.device("cpu")
    plain = DeviceSet(entries, [home, away], device)
    same = DeviceSet(entries, [home, away], device, extra_weight=1.0)
    weighted = DeviceSet(entries, [home, away], device, extra_weight=2.0)
    played = np.array([home.played.sum(), away.played.sum()], dtype=np.float64)
    assert np.array_equal(plain.share, played / played.sum())
    assert np.array_equal(same.share, plain.share)
    scaled = played * np.array([1.0, 2.0])
    assert np.allclose(weighted.share, scaled / scaled.sum(), rtol=0, atol=0)
    untouched = DeviceSet(entries, [home, plain_away], device, extra_weight=2.0)
    assert np.array_equal(untouched.share, played / played.sum())


def _blank_entry(match: TrainMatch) -> SetEntry:
    frames = match.labels.shape[0]
    tokens = np.zeros((frames, 1, 1), np.float16)
    seen = np.zeros(frames, dtype=bool)
    return SetEntry(match.match_id, tokens, tokens, seen, seen)


def test_config_defaults_round_trip_and_forbidden_sources_fail_at_construction():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig, TrainConfig
    from aaaag.paths import CONFIGS_DIR

    bare = RunConfig()
    explicit = RunConfig(train=TrainConfig(extra_matches=(), extra_degrade=False, extra_weight=1.0))
    assert bare.to_dict() == explicit.to_dict()
    for path in sorted((CONFIGS_DIR / "ml").glob("*.yaml")):
        if "extra_matches:" in path.read_text(encoding="utf-8"):
            continue  # an E35-style config sets them on purpose
        train = RunConfig.from_yaml(path).to_dict()["train"]
        assert "extra_matches" not in train
        assert "extra_degrade" not in train
        assert "extra_weight" not in train
    baseline = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_baseline.yaml").to_dict()["train"]
    assert set(baseline) == BASELINE_TRAIN_KEYS
    on = RunConfig(
        train=TrainConfig(
            extra_matches=("skillcorner/1234", "synthetic/1"),
            extra_degrade=True,
            extra_weight=0.5,
        )
    )
    dumped = on.to_dict()["train"]
    assert dumped["extra_matches"] == ["skillcorner/1234", "synthetic/1"]
    assert dumped["extra_degrade"] is True
    assert dumped["extra_weight"] == 0.5
    assert RunConfig.from_dict(on.to_dict()) == on
    loaded = TrainConfig.from_dict(
        {"extra_matches": ["synthetic/3"], "extra_weight": 2, "extra_degrade": False, "steps": 4}
    )
    assert loaded.extra_matches == ("synthetic/3",) and loaded.extra_weight == 2.0
    assert loaded.steps == 4 and loaded.extra_degrade is False
    assert "extra_degrade" not in RunConfig(train=loaded).to_dict()["train"]
    with pytest.raises(ValueError, match="idsse"):
        TrainConfig(extra_matches=("idsse/anywhere",))
    with pytest.raises(ValueError, match="bar-ath"):
        TrainConfig(extra_matches=("bar-ath/anywhere",))
    with pytest.raises(ValueError, match="extra_weight"):
        TrainConfig(extra_weight=float("nan"))


def _run(steps: int = 3, **changes):
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.train import RunConfig, TrainConfig

    tiny = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
    train = TrainConfig(
        seed=3,
        steps=steps,
        refresh_steps=steps,
        batch_size=2,
        window=64,
        warmup_steps=1,
        lr=3e-3,
        degrade=DegradeMix(version=None),
        amp=False,
        **changes,
    )
    return RunConfig(token=SPEC, model=tiny, train=train)


def _fit(run, matches, out, log, meta=None):
    from aaaag.ml.train import train

    return train(run, matches, out, device="cpu", log=log, threads=1, resume=False, meta=meta)


def _same_state(left: dict, right: dict) -> None:
    import torch

    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def _equal(left, right) -> bool:
    import torch

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


def test_empty_extra_matches_match_a_run_without_the_option(dev_pair, tmp_path):
    pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_payload

    plain_lines, empty_lines = [], []
    plain = tmp_path / "plain"
    empty = tmp_path / "empty"
    _fit(_run(), dev_pair, plain, plain_lines.append)
    _fit(
        _run(extra_matches=(), extra_degrade=False, extra_weight=1.0),
        dev_pair,
        empty,
        empty_lines.append,
    )
    plain_payload, empty_payload = (
        load_payload(plain / "final.pt"),
        load_payload(empty / "final.pt"),
    )
    _same_state(plain_payload["state_dict"], empty_payload["state_dict"])
    assert plain_payload["meta"]["run"] == empty_payload["meta"]["run"]
    assert "extra_matches" not in plain_payload["meta"]["run"]["train"]
    assert "extra_degrade" not in plain_payload["meta"]["run"]["train"]
    assert "extra_weight" not in plain_payload["meta"]["run"]["train"]
    plain_last, empty_last = load_payload(plain / "last.pt"), load_payload(empty / "last.pt")
    assert plain_last.keys() == empty_last.keys()
    _same_state(plain_last["state_dict"], empty_last["state_dict"])
    assert _equal(plain_last["optimizer"], empty_last["optimizer"])
    assert plain_last["meta"] == empty_last["meta"]
    assert plain_last["set_index"] == empty_last["set_index"]
    plain_line = next(line for line in plain_lines if line.startswith("train "))
    empty_line = next(line for line in empty_lines if line.startswith("train "))
    assert plain_line == empty_line and "extra" not in plain_line


def test_training_uses_the_extra_matches_and_logs_them_once(dev_pair, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_payload
    from aaaag.ml.train import build_set as real_build
    from aaaag.ml.train import class_weights as real_weights

    seen: dict = {}

    def spy_weights(matches):
        seen["flags"] = [match.extra for match in matches]
        return real_weights(matches)

    def spy_build(*args, **kwargs):
        seen["n"] = len(args[0])
        seen["extra_degrade"] = kwargs.get("extra_degrade", None)
        return real_build(*args, **kwargs)

    monkeypatch.setattr("aaaag.ml.train.class_weights", spy_weights)
    monkeypatch.setattr("aaaag.ml.train.build_set", spy_build)
    lines: list[str] = []
    final = _fit(
        _run(
            steps=2,
            extra_matches=("synthetic/1", "synthetic/2"),
            extra_weight=2.0,
            extra_degrade=False,
        ),
        dev_pair,
        tmp_path / "with-extra",
        lines.append,
        meta={"holdout": "none", "train_matches": [match.match_id for match in dev_pair]},
    )
    assert seen["flags"] == [False, False, True, True]
    assert seen["n"] == 4 and seen["extra_degrade"] is False
    train_lines = [line for line in lines if line.startswith("train ")]
    assert train_lines == [line for line in train_lines if "(2 extra)" in line]
    assert len(train_lines) == 1
    payload = load_payload(final)
    assert payload["meta"]["train_matches"][-2:] == ["SYN-1", "SYN-2"]
    assert payload["meta"]["run"]["train"]["extra_matches"] == ["synthetic/1", "synthetic/2"]
    assert payload["meta"]["run"]["train"]["extra_weight"] == 2.0
    assert "extra_degrade" not in payload["meta"]["run"]["train"]
    assert np.isfinite(payload["meta"]["history"][-1]["loss"])


def test_extra_matches_join_every_fold_and_are_never_held_out(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from aaaag.ml.lomo import resolve_lomo, run_lomo

    store: dict[str, TrainMatch] = {}

    def fake_load(ids, spec):
        loaded = {}
        for match_id in ids:
            if match_id not in store:
                bundle = generate(seed=1000 + len(store), config=SMALL)
                store[match_id] = TrainMatch.from_bundle(bundle, spec)
            loaded[match_id] = store[match_id]
        return loaded

    recorded: list[tuple[str, list[TrainMatch]]] = []
    evaluated: list[str] = []
    monitored: list[str] = []

    def fake_train(_run, matches, out_dir, **kwargs):
        recorded.append((kwargs["meta"]["holdout"], list(matches)))
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / "final.pt"
        final.write_bytes(b"unused")
        return final

    def fake_eval(match_id, _weights, **_kwargs):
        evaluated.append(match_id)
        return {"rows": [], "aggregate": _AGGREGATE}

    def fake_monitor(match_id, _run, _device):
        monitored.append(match_id)
        return lambda _model: {}

    monkeypatch.setattr("aaaag.ml.lomo.load_train_matches", fake_load)
    monkeypatch.setattr("aaaag.ml.train.train", fake_train)
    monkeypatch.setattr("aaaag.ml.lomo.evaluate_match", fake_eval)
    monkeypatch.setattr("aaaag.ml.lomo.make_monitor", fake_monitor)
    run = _run(extra_matches=("synthetic/1", "synthetic/2"))
    report = run_lomo(
        run,
        "all",
        tmp_path / "lomo",
        seeds=(0,),
        device="cpu",
        threads=1,
        log=lambda _line: None,
    )
    split = resolve_lomo("all")
    assert tuple(report["folds"]) == split.folds == tuple(idsse.DEV_MATCHES)
    assert evaluated == list(split.folds)
    assert monitored == list(split.folds)
    assert set(evaluated).isdisjoint({"SYN-1", "SYN-2"})
    assert [holdout for holdout, _roster in recorded] == list(split.folds)
    for holdout, roster in recorded:
        dev_ids = split.train_ids[holdout]
        dev_roster = [store[match_id] for match_id in dev_ids]
        assert all(match is not store[holdout] for match in roster)
        assert all(roster[i] is dev_roster[i] for i in range(len(dev_roster)))
        assert [match.extra for match in roster[len(dev_ids) :]] == [True, True]
        assert [match.match_id for match in roster if match.extra] == ["SYN-1", "SYN-2"]
        assert holdout not in {match.match_id for match in roster}


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


def test_extra_label_view_modes():
    from aaaag.contracts import D, H
    from aaaag.ml.data import extra_label_view

    bundle = SOURCES.get("synthetic").fn(match_id="1")
    labels = bundle.labels.copy()
    start = bundle.meta.periods[0].start
    fps = bundle.meta.fps
    # one short dead run (1 s) and one long one (4 s), separated by home play
    labels[start : start + 10 * fps] = H
    labels[start + 2 * fps : start + 3 * fps] = D
    labels[start + 5 * fps : start + 9 * fps] = D
    before = labels.copy()
    assert extra_label_view(labels, bundle, "all") is labels
    team = extra_label_view(labels, bundle, "team")
    assert not np.any(team == D) and np.array_equal(labels, before)
    core = extra_label_view(labels, bundle, "core")
    assert np.all(core[start + 2 * fps : start + 3 * fps] == X)  # short run dropped
    edge = int(0.5 * fps)
    long_run = core[start + 5 * fps : start + 9 * fps]
    assert np.all(long_run[:edge] == X) and np.all(long_run[-edge:] == X)
    assert np.all(long_run[edge:-edge] == D)
    assert np.all(core[start : start + 2 * fps] == H)
    with pytest.raises(ValueError):
        extra_label_view(labels, bundle, "nope")


def test_extra_labels_mode_is_validated():
    pytest.importorskip("torch")
    from aaaag.ml.train import TrainConfig

    with pytest.raises(ValueError):
        TrainConfig(extra_labels="nope")


def test_label_free_mode_drops_every_label():
    from aaaag.ml.data import extra_label_view

    bundle = SOURCES.get("synthetic").fn(match_id="1")
    out = extra_label_view(bundle.labels, bundle, "none")
    assert np.all(out == X) and out.shape == bundle.labels.shape


def test_team_agnostic_possessor_picks_the_nearest_slot_of_either_team():
    from aaaag.ml.possessor import possessor_targets
    from aaaag.ml.windows import build_match_tokens

    spec = TokenSpec()
    bundle = SOURCES.get("synthetic").fn(match_id="1")
    tokens = build_match_tokens(bundle.tracking, spec)
    tokens = tokens[0] if isinstance(tokens, tuple) else tokens
    sign = np.ones(bundle.meta.n_frames)
    labels = np.full(bundle.meta.n_frames, X, dtype=np.int8)
    free = possessor_targets(
        tokens, labels, bundle.tracking.ball_xy, sign, bundle.meta, spec, 2.0, team_agnostic=True
    )
    assert free.shape == (bundle.meta.n_frames,)
    assert np.all((free >= -1) & (free < 2 * spec.tokens_per_team))  # never the dead target
    assert np.all(
        possessor_targets(tokens, labels, bundle.tracking.ball_xy, sign, bundle.meta, spec, 2.0)
        == -1
    )
