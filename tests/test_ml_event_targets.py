"""Event targets for training (#86): `train.touch_source`, `train.event_switch_weight` and
`train.init_add_touch_head`, all off by default.

The provider events of a synthetic match come from a stub registered for the "synthetic" source in
`data.EVENT_SOURCES`, built from its labels: one on-ball event at the first frame of each alive
spell and every 25 frames inside it, a foul and a free kick inside some stoppages. Torch tests skip
when torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchBundle, MatchMeta, Period, Tracking, X
from aaaag.ml import data
from aaaag.ml.data import TrainMatch, event_switch_target, source_touch_events, touch_target
from aaaag.ml.spec import TokenSpec
from aaaag.sources.idsse_events import TOUCH_DTYPE, possession_changes, touch_frames


def stub_events(bundle: MatchBundle) -> np.ndarray:
    """On-ball events read off the labels, plus a foul and a set piece in every other stoppage."""
    labels = bundle.labels
    rows = []
    for period in bundle.meta.periods:
        seg = labels[period.start : period.stop]
        for t in range(len(seg)):
            frame = period.start + t
            if seg[t] in (H, A) and (t == 0 or seg[t - 1] != seg[t] or t % 25 == 0):
                rows.append((frame, int(seg[t]), "pass"))
        dead = np.flatnonzero(seg == D)
        starts = dead[np.concatenate(([True], np.diff(dead) > 1))] if len(dead) else dead
        for k, t in enumerate(starts):
            if k % 2 == 0:
                rows.append((period.start + int(t), 1, "foul"))
                rows.append((period.start + int(t) + 1, 0, "set_piece"))
    out = np.array(rows, dtype=TOUCH_DTYPE)
    return out[np.argsort(out["frame"], kind="stable")]


@pytest.fixture
def events_for_synthetic(monkeypatch):
    monkeypatch.setitem(data.EVENT_SOURCES, "synthetic", stub_events)


def _one_match() -> TrainMatch:
    from aaaag.synthetic import SyntheticConfig, generate

    bundle = generate(seed=7, config=SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.1))
    return TrainMatch.from_bundle(bundle, TokenSpec())


# -- targets (no torch) ---------------------------------------------------------------------------


def test_a_source_without_events_is_refused_clearly():
    match = _one_match()
    assert "synthetic" not in data.EVENT_SOURCES and "idsse" in data.EVENT_SOURCES
    with pytest.raises(ValueError, match="source 'synthetic' has no touch_events"):
        source_touch_events(match.bundle)
    with pytest.raises(ValueError, match="has no touch_events"):
        touch_target(match, 2, "events")
    with pytest.raises(ValueError, match="has no touch_events"):
        event_switch_target(match, 3)
    with pytest.raises(ValueError, match="touch_source"):
        touch_target(match, 2, "video")
    # The default source is today's target, with or without the argument.
    assert np.array_equal(
        touch_target(match, 2), touch_target(match, 2, "tracking"), equal_nan=True
    )


def test_event_touch_target_marks_on_ball_events_only(events_for_synthetic):
    match = _one_match()
    events = stub_events(match.bundle)
    target = touch_target(match, 1, "events")
    n = match.bundle.meta.n_frames
    near = np.zeros(n, dtype=bool)
    for frame in touch_frames(events):
        near[max(0, frame - 1) : frame + 2] = True
    supervised = match.bundle.tracking.ball_seen() & np.isin(match.labels, (H, A)) & match.played
    assert np.array_equal(target == 1, supervised & near)
    assert np.array_equal(target == 0, supervised & ~near)
    assert np.isnan(target[~supervised]).all()
    fouls = events["frame"][events["kind"] == "foul"]
    assert len(fouls) and not set(fouls.tolist()) & set(touch_frames(events).tolist())
    marked = data.with_touch_target(match, 1, "events")
    assert match.touch is None and np.array_equal(marked.touch, target, equal_nan=True)


def test_event_switch_target_marks_open_play_changes(events_for_synthetic):
    match = _one_match()
    events = stub_events(match.bundle)
    tol = 3
    target = event_switch_target(match, tol)
    changes = possession_changes(events, match.bundle.meta.periods)
    labels = match.labels
    direct = np.flatnonzero(np.isin(labels[:-1], (H, A)) & np.isin(labels[1:], (H, A)))
    direct = direct[labels[direct] != labels[direct + 1]] + 1
    assert len(changes) > 0  # some across unmarked stoppages: events alone cannot see those
    assert set(direct.tolist()) <= set(changes.tolist())
    n = match.bundle.meta.n_frames
    near = np.zeros(n, dtype=bool)
    for frame in changes:
        near[max(0, frame - tol) : frame + tol + 1] = True
    supervised = np.isin(labels, (H, A)) & match.played
    assert np.array_equal(target == 1, supervised & near)
    assert np.array_equal(target == 0, supervised & ~near)
    assert np.isnan(target[~supervised]).all()
    marked = data.with_event_switch_target(match, tol)
    assert match.event_switch is None
    assert np.array_equal(marked.event_switch, target, equal_nan=True)


def test_possession_changes_skip_stoppages():
    periods = (Period(1, 0, 100), Period(2, 120, 200))
    rows = [
        (10, H, "pass"),
        (20, A, "pass"),  # change
        (30, A, "reception"),
        (35, A, "foul"),
        (40, H, "set_piece"),  # through a foul and a set piece: no change
        (50, H, "pass"),
        (60, A, "out"),
        (70, A, "pass"),  # through an out: no change
        (80, -1, "pass"),  # no team: not an on-ball event of a team
        (90, H, "tackle"),  # change
        (130, A, "pass"),  # a new period: no change across the break
        (140, H, "clearance"),  # change
    ]
    events = np.array(rows, dtype=TOUCH_DTYPE)
    assert possession_changes(events, periods).tolist() == [20, 90, 140]


# -- torch ----------------------------------------------------------------------------------------


def _tiny_run(**train_overrides):
    from aaaag.ml.data import DegradeMix
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.train import RunConfig, TrainConfig

    heads = train_overrides.pop("heads", {})
    model = dict(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0, **heads)
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
        **train_overrides,
    )
    return RunConfig(token=TokenSpec(), model=ModelConfig(**model), train=TrainConfig(**train))


def test_config_defaults_validation_and_round_trip():
    pytest.importorskip("torch")
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.train import RunConfig, TrainConfig

    bare = RunConfig().to_dict()["train"]
    for key in ("touch_source", "event_switch_weight", "init_add_touch_head"):
        assert key not in bare
    assert TrainConfig().touch_source == "tracking" and TrainConfig().event_switch_weight == 0.0
    with pytest.raises(ValueError, match="touch_source"):
        TrainConfig(touch_source="video")
    with pytest.raises(ValueError, match="no effect with touch_weight 0"):
        TrainConfig(touch_source="events")
    with pytest.raises(ValueError, match="event_switch_weight"):
        TrainConfig(event_switch_weight=-0.1)
    with pytest.raises(ValueError, match="switch_head"):
        RunConfig(train=TrainConfig(event_switch_weight=0.2))
    with pytest.raises(ValueError, match="init_add_touch_head"):
        RunConfig(model=ModelConfig(touch_head=True), train=TrainConfig(init_add_touch_head=True))
    run = RunConfig(
        model=ModelConfig(switch_head=True, touch_head=True),
        train=TrainConfig(
            touch_weight=0.2,
            touch_source="events",
            event_switch_weight=0.3,
            init_from="x.pt",
            init_add_touch_head=True,
        ),
    )
    dumped = run.to_dict()["train"]
    assert dumped["touch_source"] == "events" and dumped["event_switch_weight"] == 0.3
    assert dumped["init_add_touch_head"] is True
    assert RunConfig.from_dict(run.to_dict()) == run


def test_loss_is_unchanged_at_weight_zero_and_adds_the_esw_term():
    torch = pytest.importorskip("torch")
    from aaaag.ml.train import loss_terms

    torch.manual_seed(0)
    n, w, margin = 2, 24, 3
    labels = torch.randint(0, 4, (n, w))
    out = {"logits": torch.randn(n, w, 3), "ball": torch.randn(n, w, 2)}
    headed = {**out, "switch": torch.randn(n, w)}
    target = torch.zeros(n, w)
    target[:, 10:13] = 1
    target[:, 4] = torch.nan
    batch = {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.randn(n, w, 2),
        "event_switch": target,
    }
    weights = torch.tensor([1.0, 1.2, 0.8])
    plain = loss_terms(out, batch, weights, margin, 0.3)
    off = loss_terms(headed, batch, weights, margin, 0.3, event_switch_weight=0.0)
    assert torch.equal(plain["loss"], off["loss"]) and "esw" not in off
    on = loss_terms(headed, batch, weights, margin, 0.3, event_switch_weight=0.4)
    assert torch.isfinite(on["esw"]) and torch.equal(on["loss"], plain["loss"] + 0.4 * on["esw"])
    both = loss_terms(
        headed, batch, weights, margin, 0.3, switch_weight=0.2, event_switch_weight=0.4
    )
    assert torch.equal(both["loss"], plain["loss"] + 0.2 * both["sw"] + 0.4 * both["esw"])
    ignored = {**batch, "event_switch": torch.full((n, w), torch.nan)}
    zero = loss_terms(headed, ignored, weights, margin, 0.3, event_switch_weight=0.4)
    assert zero["esw"].item() == 0.0
    missing = {k: v for k, v in batch.items() if k != "event_switch"}
    with pytest.raises(ValueError, match="event-switch target"):
        loss_terms(headed, missing, weights, margin, 0.3, event_switch_weight=0.4)


def test_windows_slice_the_event_switch_target_and_augment_leaves_it():
    torch = pytest.importorskip("torch")
    from aaaag.ml.data import SetEntry
    from aaaag.ml.train import DeviceSet, TrainConfig, augment

    n = 8
    meta = MatchMeta("syn", "synthetic", n, 25, 105.0, 68.0, (Period(1, 0, n),))
    tracking = Tracking(
        meta,
        np.zeros((n, 2, 2), np.float32),
        np.array([0, 1], np.int8),
        np.zeros((n, 2), np.float32),
    ).validate()
    labels = np.array([H, H, A, A, D, H, H, X], np.int8)
    switch = np.array([0, 1, 1, 0, np.nan, 0, 0, np.nan], np.float32)
    match = TrainMatch(
        match_id="syn",
        bundle=MatchBundle(tracking, labels=labels.copy()),
        labels=labels.copy(),
        sign=np.ones(n, np.float32),
        ball_target=np.zeros((n, 2), np.float32),
        target_ok=np.ones(n, dtype=bool),
        played=np.ones(n, dtype=bool),
        event_switch=switch.copy(),
    )
    entry = SetEntry(
        "syn",
        np.zeros((n, 1, 4), np.float16),
        np.zeros((n, 1, 4), np.float16),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )
    got = DeviceSet([entry], [match], torch.device("cpu")).batch(
        np.random.default_rng(0), 2, n, decoupled=False
    )
    expect = torch.nan_to_num(torch.from_numpy(switch), nan=-2)
    assert torch.equal(torch.nan_to_num(got["event_switch"][0], nan=-2), expect)
    plain = DeviceSet([entry], [TrainMatch(**{**match.__dict__, "event_switch": None})], "cpu")
    assert "event_switch" not in plain.batch(np.random.default_rng(0), 2, n, decoupled=False)

    spec = TokenSpec(tokens_per_team=1)
    window = {
        "tokens": torch.zeros(2, n, spec.n_tokens, spec.n_features),
        "labels": torch.tensor([[H, A, D, X, H, A, H, A], [A, A, H, H, D, X, H, A]]),
        "ball": torch.zeros(2, n, 2),
        "event_switch": torch.from_numpy(switch).repeat(2, 1),
    }
    before = window["event_switch"].clone()
    after = augment(window, spec, TrainConfig(), np.random.default_rng(5))
    assert torch.equal(
        torch.nan_to_num(after["event_switch"], nan=-2), torch.nan_to_num(before, nan=-2)
    )


def test_off_runs_keep_today_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_payload
    from aaaag.ml.train import train

    match = _one_match()

    def once(out, **overrides):
        lines: list[str] = []
        final = train(
            _tiny_run(heads={"switch_head": True}, switch_weight=0.2, **overrides),
            [match],
            out,
            device="cpu",
            log=lines.append,
            resume=False,
            threads=1,
        )
        return load_payload(final), lines

    plain, plain_log = once(tmp_path / "plain")
    explicit, explicit_log = once(tmp_path / "explicit", event_switch_weight=0.0)
    assert all(" esw " not in line for line in plain_log + explicit_log)
    for key in plain["state_dict"]:
        assert torch.equal(plain["state_dict"][key], explicit["state_dict"][key])
    assert plain["meta"]["run"] == explicit["meta"]["run"]
    assert "esw" not in plain["meta"]["history"][0]
    assert match.event_switch is None and match.touch is None


def test_a_few_steps_train_on_event_targets(tmp_path, events_for_synthetic):
    pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_payload
    from aaaag.ml.train import train

    match = _one_match()
    lines: list[str] = []
    final = train(
        _tiny_run(
            heads={"switch_head": True, "touch_head": True},
            switch_weight=0.2,
            touch_weight=0.2,
            touch_source="events",
            event_switch_weight=0.3,
        ),
        [match],
        tmp_path,
        device="cpu",
        log=lines.append,
        resume=False,
        threads=1,
    )
    assert match.touch is None and match.event_switch is None
    assert any(" esw " in line and " touch " in line for line in lines)
    payload = load_payload(final)
    row = payload["meta"]["history"][0]
    assert np.isfinite(row["esw"]) and np.isfinite(row["touch"]) and np.isfinite(row["loss"])
    saved = payload["meta"]["run"]["train"]
    assert saved["touch_source"] == "events" and saved["event_switch_weight"] == 0.3


def test_event_targets_refuse_a_source_without_events(tmp_path):
    pytest.importorskip("torch")
    from aaaag.ml.train import train

    run = _tiny_run(heads={"switch_head": True}, event_switch_weight=0.3)
    with pytest.raises(ValueError, match="has no touch_events"):
        train(run, [_one_match()], tmp_path, device="cpu", log=lambda _: None, threads=1)


def test_init_add_touch_head_loads_every_weight_but_the_new_head(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_payload
    from aaaag.ml.model import build_model
    from aaaag.ml.train import _load_init_weights, train

    base = train(
        _tiny_run(heads={"switch_head": True}),
        [_one_match()],
        tmp_path / "base",
        device="cpu",
        log=lambda _: None,
        resume=False,
        threads=1,
    )
    saved = load_payload(base)["state_dict"]
    heads = {"switch_head": True, "touch_head": True}
    strict = _tiny_run(heads=heads, init_from=str(base))
    torch.manual_seed(0)
    model = build_model(strict.arch, strict.model)
    with pytest.raises(ValueError, match="does not match"):
        _load_init_weights(model, str(base), strict)

    adding = _tiny_run(heads=heads, init_from=str(base), init_add_touch_head=True)
    torch.manual_seed(0)
    model = build_model(adding.arch, adding.model)
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    _load_init_weights(model, str(base), adding)
    state = model.state_dict()
    assert set(state) - set(saved) == {"touch.weight", "touch.bias"}
    for key in saved:
        assert torch.equal(state[key], saved[key])
    for key in ("touch.weight", "touch.bias"):
        assert torch.equal(state[key], fresh[key])

    # The flag does not loosen any other difference.
    other = _tiny_run(
        heads={**heads, "possessor_head": True}, init_from=str(base), init_add_touch_head=True
    )
    with pytest.raises(ValueError, match="does not match"):
        _load_init_weights(build_model(other.arch, other.model), str(base), other)
