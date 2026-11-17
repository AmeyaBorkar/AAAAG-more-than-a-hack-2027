"""Exponential moving average of the training weights (`train.ema_decay`).

Decay 0 is off: the weights, the optimizer state and the saved config match a run that does not
set the field. Torch only, CPU, synthetic matches; CI skips this file.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aaaag.ml.checkpoint import load_payload  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch  # noqa: E402
from aaaag.ml.model import ModelConfig  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import RunConfig, TrainConfig, train  # noqa: E402
from aaaag.paths import CONFIGS_DIR  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
SMALL = SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)


def _run(steps: int = 6, **train_changes) -> RunConfig:
    train_cfg = TrainConfig(
        seed=3,
        steps=steps,
        refresh_steps=3,
        batch_size=2,
        window=64,
        warmup_steps=1,
        lr=3e-3,
        degrade=DegradeMix(version=None),
        amp=False,
        **train_changes,
    )
    return RunConfig(token=SPEC, model=TINY, train=train_cfg)


def _train(run: RunConfig, matches: list[TrainMatch], out, **kwargs):
    log = kwargs.pop("log", lambda _line: None)
    resume = kwargs.pop("resume", False)
    return train(
        run,
        matches,
        out,
        device="cpu",
        log=log,
        threads=1,
        resume=resume,
        **kwargs,
    )


def _equal(left, right) -> bool:
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


def _same_state(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


@pytest.fixture(scope="module")
def matches():
    return [TrainMatch.from_bundle(generate(seed=seed, config=SMALL), SPEC) for seed in (7, 8)]


def test_decay_zero_matches_training_without_the_option(matches, tmp_path):
    pytest.importorskip("torch")
    plain_dir, zero_dir = tmp_path / "plain", tmp_path / "zero"
    _train(_run(steps=3), matches, plain_dir)
    _train(_run(steps=3, ema_decay=0.0), matches, zero_dir)
    assert not (plain_dir / "final_raw.pt").exists()
    assert not (zero_dir / "final_raw.pt").exists()
    plain, zero = load_payload(plain_dir / "final.pt"), load_payload(zero_dir / "final.pt")
    _same_state(plain["state_dict"], zero["state_dict"])
    assert plain["meta"]["run"] == zero["meta"]["run"]
    assert "ema_decay" not in plain["meta"]["run"]["train"]
    plain_last, zero_last = load_payload(plain_dir / "last.pt"), load_payload(zero_dir / "last.pt")
    assert "ema" not in plain_last and "ema" not in zero_last
    assert plain_last.keys() == zero_last.keys()
    _same_state(plain_last["state_dict"], zero_last["state_dict"])
    assert _equal(plain_last["optimizer"], zero_last["optimizer"])
    assert plain_last["meta"] == zero_last["meta"]
    assert plain_last["set_index"] == zero_last["set_index"]


def test_one_parameter_and_a_floating_buffer_follow_the_hand_recursion():
    torch = pytest.importorskip("torch")
    from aaaag.ml.train import _WeightEma

    class Box(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.25, -0.5]))
            self.register_buffer("scale", torch.tensor([2.0, 4.0]))
            self.register_buffer("ids", torch.arange(3))

    box = Box()
    decay = 0.5
    ema = _WeightEma(box, decay)
    hand = {name: value.detach().clone() for name, value in ema.shadow.items()}
    steps = (
        (torch.tensor([0.0, 1.0]), torch.tensor([3.0, 5.0])),
        (torch.tensor([2.0, -2.0]), torch.tensor([1.0, 1.0])),
        (torch.tensor([-1.0, 0.5]), torch.tensor([8.0, -1.0])),
    )
    for n, (weight, scale) in enumerate(steps):
        with torch.no_grad():
            box.weight.copy_(weight)
            box.scale.copy_(scale)
        ema.update(box)
        d = min(decay, (1.0 + n) / (10.0 + n))  # the warm-up
        hand["weight"] = d * hand["weight"] + (1.0 - d) * box.weight.detach()
        hand["scale"] = d * hand["scale"] + (1.0 - d) * box.scale.detach()
    assert ema.updates == len(steps)
    saved = ema.state_dict()
    assert set(saved) == {"weight", "scale"}
    assert torch.equal(saved["weight"], hand["weight"])
    assert torch.equal(saved["scale"], hand["scale"])
    with torch.no_grad():
        box.ids.add_(1)
    ema.copy_to(box)
    assert torch.equal(box.weight.detach(), hand["weight"])
    assert torch.equal(box.scale.detach(), hand["scale"])
    assert torch.equal(box.ids, torch.arange(3) + 1)  # integer buffers are not averaged


def test_final_is_the_ema_and_final_raw_is_the_last_iterate(matches, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import build_model
    from aaaag.ml.train import seed_everything

    decay = 0.5
    run = _run(steps=3, ema_decay=decay)
    off = tmp_path / "off"
    _train(_run(steps=3), matches, off)

    build_model(run.arch, run.model)  # import architectures before the seeded init
    seed_everything(run.train.seed)
    twin = build_model(run.arch, run.model)
    names = [name for name, _value in twin.named_parameters()]
    initial = {
        name: value.detach().cpu().clone()
        for name, value in twin.state_dict().items()
        if value.is_floating_point()
    }
    assert names == list(initial)
    snapshots: list[dict[str, torch.Tensor]] = []
    orig_step = torch.optim.AdamW.step

    def step(self, *args, **kwargs):
        result = orig_step(self, *args, **kwargs)
        values = [
            param.detach().to(dtype=torch.float32).cpu().clone()
            for group in self.param_groups
            for param in group["params"]
        ]
        assert len(values) == len(names)
        snapshots.append(dict(zip(names, values, strict=True)))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    on = tmp_path / "on"
    _train(run, matches, on)
    assert len(snapshots) == 3

    hand = {name: value.clone() for name, value in initial.items()}
    for n, snap in enumerate(snapshots):
        d = min(decay, (1.0 + n) / (10.0 + n))  # the warm-up
        for name in hand:
            hand[name] = d * hand[name] + (1.0 - d) * snap[name]

    final = load_payload(on / "final.pt")
    raw = load_payload(on / "final_raw.pt")["state_dict"]
    off_state = load_payload(off / "final.pt")["state_dict"]
    got = final["state_dict"]
    assert got.keys() == raw.keys() == off_state.keys() == hand.keys()
    _same_state(raw, off_state)
    _same_state(raw, snapshots[-1])
    _same_state(got, hand)
    assert any(not torch.equal(got[name], raw[name]) for name in got)
    last = load_payload(on / "last.pt")
    _same_state(last["state_dict"], raw)
    _same_state(last["ema"], hand)
    assert final["meta"]["run"]["train"]["ema_decay"] == decay
    assert "ema_decay" not in load_payload(off / "final.pt")["meta"]["run"]["train"]
    assert not (off / "final_raw.pt").exists()


def test_resuming_from_last_matches_an_uninterrupted_ema(matches, tmp_path):
    pytest.importorskip("torch")
    run = _run(steps=6, ema_decay=0.5)
    whole = tmp_path / "whole"
    _train(run, matches, whole)
    seen: list[int] = []

    def interrupt(_model):
        seen.append(1)
        if len(seen) == 2:
            raise RuntimeError("interrupted")
        return {}

    split = tmp_path / "split"
    with pytest.raises(RuntimeError, match="interrupted"):
        _train(run, matches, split, resume=False, monitor=interrupt)
    assert (split / "last.pt").is_file() and not (split / "final.pt").exists()
    assert "ema" in load_payload(split / "last.pt")
    lines: list[str] = []
    _train(run, matches, split, resume=True, log=lines.append)
    assert any("resumed" in line for line in lines)
    for name in ("final.pt", "final_raw.pt"):
        _same_state(
            load_payload(whole / name)["state_dict"], load_payload(split / name)["state_dict"]
        )
    _same_state(load_payload(whole / "last.pt")["ema"], load_payload(split / "last.pt")["ema"])
    assert load_payload(split / "last.pt")["ema_updates"] == 6


def test_existing_configs_omit_ema_decay_and_the_contact_ema_config_loads():
    pytest.importorskip("torch")
    import yaml

    for name in ("team_baseline.yaml", "team_cm_contact.yaml", "team_control_memory.yaml"):
        dumped = RunConfig.from_yaml(CONFIGS_DIR / "ml" / name).to_dict()
        assert "ema_decay" not in dumped["train"]
    assert RunConfig().to_dict() == RunConfig(train=TrainConfig(ema_decay=0.0)).to_dict()
    assert RunConfig().train.ema_decay == 0.0

    text = (CONFIGS_DIR / "ml" / "team_cm_contact_ema.yaml").read_text(encoding="utf-8")
    assert text.startswith("#") and "ema_decay" in text.split("token:", 1)[0]
    contact = yaml.safe_load(
        (CONFIGS_DIR / "ml" / "team_cm_contact.yaml").read_text(encoding="utf-8")
    )
    raw = yaml.safe_load(text)
    assert raw["train"].pop("ema_decay") == 0.998
    assert raw == contact
    loaded = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_ema.yaml")
    assert loaded.train.ema_decay == 0.998
    assert loaded.to_dict()["train"]["ema_decay"] == 0.998
    assert RunConfig.from_dict(loaded.to_dict()) == loaded
    without = loaded.to_dict()
    without["train"].pop("ema_decay")
    assert without == RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact.yaml").to_dict()
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(ema_decay=1.5)
