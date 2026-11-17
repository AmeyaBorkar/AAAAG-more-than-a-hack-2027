"""Knowledge distillation into one TeamNet (plan E12).

Torch only, on synthetic matches and a tiny model, CPU. `distill` absent, null, or `alpha` 0
matches a run that does not distill, including the weights.
"""

from __future__ import annotations

import time
from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F  # noqa: E402

from aaaag.ml.checkpoint import load_payload, save_checkpoint  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch  # noqa: E402
from aaaag.ml.model import ModelConfig, TeamNet, build_model  # noqa: E402
from aaaag.ml.spec import N_FEATURES, TokenSpec  # noqa: E402
from aaaag.ml.train import (  # noqa: E402
    HUBER_DELTA,
    DistillConfig,
    RunConfig,
    TrainConfig,
    loss_terms,
    mean_teacher_logits,
    seed_everything,
    teacher_paths,
    train,
)
from aaaag.ml.windows import build_match_tokens  # noqa: E402
from aaaag.paths import CONFIGS_DIR  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
SHORT = SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.1)
TEACHER_RUNS = (
    "C:/Users/ameya/aaaag-cache/runs/e0_seed0",
    "C:/Users/ameya/aaaag-cache/runs/e2_seed1",
    "C:/Users/ameya/aaaag-cache/runs/e2_seed2",
    "C:/Users/ameya/aaaag-cache/runs/e2_seed3",
    "C:/Users/ameya/aaaag-cache/runs/e2_seed4",
)


def _run(steps: int = 4, model: ModelConfig = TINY, **train_changes) -> RunConfig:
    cfg = dict(
        seed=3,
        steps=steps,
        refresh_steps=steps,
        batch_size=2,
        window=64,
        warmup_steps=1,
        lr=3e-3,
        degrade=DegradeMix(version=None),
        amp=False,
    )
    cfg.update(train_changes)
    return RunConfig(token=SPEC, model=model, train=TrainConfig(**cfg))


def _fit(run: RunConfig, matches, out, **kwargs):
    return train(
        run,
        matches,
        out,
        device="cpu",
        log=lambda _: None,
        resume=False,
        threads=1,
        **kwargs,
    )


def _weights(path):
    return load_payload(path)["state_dict"]


def _equal(one, two) -> None:
    assert one.keys() == two.keys()
    assert all(torch.equal(one[key], two[key]) for key in one)


def _checkpoint(path, seed: int, bias: list[float] | None = None):
    torch.manual_seed(seed)
    model = TeamNet(TINY)
    if bias is not None:
        with torch.no_grad():
            model.head.bias.copy_(torch.tensor(bias))
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_checkpoint(path, model, SPEC, {})


@pytest.fixture(scope="module")
def matches():
    return [TrainMatch.from_bundle(generate(seed=s, config=SHORT), SPEC) for s in (7, 8)]


def test_distill_absent_matches_an_explicit_null(matches, tmp_path):
    """The off switch is the same run twice: no distill field, and distill=None."""
    model = replace(TINY, dropout=0.1)
    absent = _fit(_run(model=model, refresh_steps=2), matches, tmp_path / "absent")
    explicit = _fit(
        _run(model=model, refresh_steps=2, distill=None), matches, tmp_path / "explicit"
    )
    _equal(_weights(absent), _weights(explicit))


def test_alpha_zero_matches_the_undistilled_run(matches, tmp_path):
    model = replace(TINY, dropout=0.1)
    teacher = _checkpoint(tmp_path / "e0" / "final.pt", seed=11)
    plain = _fit(_run(steps=4, model=model, refresh_steps=2), matches, tmp_path / "plain")
    zero = _fit(
        _run(
            steps=4,
            model=model,
            refresh_steps=2,
            distill=DistillConfig(teacher_runs=(str(teacher.parent),), alpha=0.0, temperature=3.0),
        ),
        matches,
        tmp_path / "zero",
        teachers=[teacher],
    )
    _equal(_weights(plain), _weights(zero))


def test_identical_teacher_has_zero_kl_and_zero_gradients():
    torch.manual_seed(0)
    student = TeamNet(TINY).train()
    teacher = TeamNet(TINY).eval()
    teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False)
    tokens = torch.randn(2, 40, SPEC.n_tokens, N_FEATURES)
    labels = torch.randint(0, 3, (2, 40))
    labels[0, 20] = 3
    batch = {
        "tokens": tokens,
        "labels": labels,
        "ball": torch.zeros(2, 40, 2),
        "ball_ok": torch.zeros(2, 40, dtype=torch.bool),
    }
    with torch.no_grad():
        teacher_logits = teacher(tokens)["logits"]
    out = student(tokens)
    # Train-mode and no-grad forwards may take different CPU kernels (Apple Silicon differs in the
    # last bits, #158), so "identical" means equal to float tolerance, not bitwise.
    torch.testing.assert_close(out["logits"], teacher_logits, rtol=0.0, atol=1e-5)
    distill = DistillConfig(teacher_runs=("teacher",), alpha=1.0, temperature=2.0)
    terms = loss_terms(
        out,
        batch,
        torch.ones(3),
        TINY.receptive_half,
        aux_weight=0.0,
        teacher_logits=teacher_logits,
        distill=distill,
    )
    assert terms["ce"].item() > 0.1
    assert abs(terms["kl"].item()) < 1e-8
    assert abs(terms["loss"].item()) < 1e-8
    terms["loss"].backward()
    # softmax(log_softmax) does not sum to 1 bitwise, so a KL of identical logits backprops
    # dust around 1e-11. A real cross-entropy gradient is orders of magnitude larger.
    for name, param in student.named_parameters():
        if name.startswith("ball_head"):
            assert param.grad is None  # the ball term is off, so a zero loss never touches it
        else:
            assert param.grad is not None
            assert param.grad.abs().max().item() < 1e-6


def test_identical_teacher_loss_is_zero_on_the_first_step(matches, tmp_path):
    build_model("team_net", TINY)  # import architectures before the seed is drawn
    seed_everything(3)
    teacher = build_model("team_net", TINY)
    path = save_checkpoint(tmp_path / "run" / "final.pt", teacher, SPEC, {})
    final = _fit(
        _run(
            steps=1,
            distill=DistillConfig(teacher_runs=(str(path.parent),), alpha=1.0, temperature=1.0),
            aux_weight=0.0,
            weight_decay=0.0,
        ),
        matches,
        tmp_path / "student",
        teachers=[path],
    )
    # The logged step is the loss before the update: identical weights => KL 0, and alpha 1
    # drops the hard-label term.
    history = load_payload(final)["meta"]["history"]
    assert history[-1]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert history[-1]["kl"] == pytest.approx(0.0, abs=1e-6)


def test_a_few_steps_move_logits_toward_the_teacher_mean(matches):
    """Two random teachers with a confident head bias, so their mean sits away from the origin.

    A kaiming init already emits small logits; the bias makes the ensemble mean a target a few
    steps can measurably approach. Hard labels agree with that class, and alpha is 0.7.
    """
    tokens_np = build_match_tokens(matches[0].bundle.tracking, SPEC)
    start = matches[0].bundle.meta.periods[0].start
    tokens = torch.from_numpy(np.ascontiguousarray(tokens_np[start : start + 64])).float()[None]
    teachers = []
    for seed, bias in ((1, [6.0, -3.0, -3.0]), (2, [4.0, -2.0, -2.0])):
        torch.manual_seed(seed)
        teacher = TeamNet(TINY).eval()
        with torch.no_grad():
            teacher.head.bias.copy_(torch.tensor(bias))
        teacher.requires_grad_(False)
        teachers.append(teacher)
    torch.manual_seed(0)
    student = TeamNet(TINY).train()
    width = tokens.shape[1]
    batch = {
        "tokens": tokens,
        "labels": torch.zeros(1, width, dtype=torch.long),
        "ball": torch.zeros(1, width, 2),
        "ball_ok": torch.zeros(1, width, dtype=torch.bool),
    }
    distill = DistillConfig(teacher_runs=("a", "b"), alpha=0.7, temperature=1.0)
    margin = TINY.receptive_half
    weights = torch.tensor([1.0, 1.0, 1.0])

    def distance() -> float:
        was_training = student.training
        student.eval()
        with torch.no_grad():
            logits = student(tokens)["logits"]
            target = mean_teacher_logits(teachers, tokens)
        inner = slice(margin, width - margin)
        value = float((logits[:, inner] - target[:, inner]).pow(2).mean())
        student.train(was_training)
        return value

    before = distance()
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-2, weight_decay=0.0)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            target = mean_teacher_logits(teachers, tokens)
        terms = loss_terms(
            student(tokens),
            batch,
            weights,
            margin,
            aux_weight=0.0,
            teacher_logits=target,
            distill=distill,
        )
        terms["loss"].backward()
        optimizer.step()
    after = distance()
    assert after < before, f"logit mse to the teacher mean {before:.4f} -> {after:.4f}"


def test_the_distillation_term_matches_the_formula_and_ignores_edges_and_x():
    b, w, margin = 2, 30, 5
    student = torch.randn(b, w, 3)
    teacher = torch.randn(b, w, 3)
    labels = torch.randint(0, 3, (b, w))
    labels[0, margin + 2] = 3
    weights = torch.tensor([0.5, 1.5, 3.0])
    ball = torch.ones(b, w, 2)
    ball_ok = torch.zeros(b, w, dtype=torch.bool)
    ball_ok[:, margin + 3] = True
    batch = {"tokens": torch.zeros(b, w, 1, 1), "labels": labels, "ball": ball, "ball_ok": ball_ok}
    out = {"logits": student, "ball": torch.zeros(b, w, 2)}
    alpha, temperature, aux_weight = 0.7, 2.0, 0.4
    distill = DistillConfig(teacher_runs=("r",), alpha=alpha, temperature=temperature)
    terms = loss_terms(
        out,
        batch,
        weights,
        margin,
        aux_weight,
        teacher_logits=teacher,
        distill=distill,
    )

    inner = slice(margin, w - margin)
    logits, target, y = student[:, inner], teacher[:, inner], labels[:, inner]
    ce = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), weight=weights, ignore_index=3)
    log_p = F.log_softmax(target / temperature, dim=-1)
    log_q = F.log_softmax(logits / temperature, dim=-1)
    per_frame = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)
    valid = y != 3
    index = y.clamp(min=0, max=2)
    frame_w = torch.where(valid, weights[index], torch.zeros(()))
    kl = (per_frame * frame_w).sum() / frame_w.sum()
    ok = ball_ok[:, inner]
    aux = F.huber_loss(out["ball"][:, inner][ok], ball[:, inner][ok], delta=HUBER_DELTA)
    expected = (1.0 - alpha) * ce + (alpha * temperature**2) * kl + aux_weight * aux
    assert float(terms["kl"]) == pytest.approx(float(kl))
    assert float(terms["ce"]) == pytest.approx(float(ce))
    assert float(terms["loss"]) == pytest.approx(float(expected))

    spoiled = teacher.clone()
    spoiled[:, :margin] = torch.randn_like(spoiled[:, :margin])
    spoiled[0, margin + 2] = torch.tensor([8.0, -4.0, 1.0])
    ignored = loss_terms(
        out, batch, weights, margin, aux_weight, teacher_logits=spoiled, distill=distill
    )
    assert torch.equal(terms["kl"], ignored["kl"])
    moved = teacher.clone()
    moved[0, margin + 4, 0] += 2.0  # one class; a shift of all three leaves softmax unchanged
    changed = loss_terms(
        out, batch, weights, margin, aux_weight, teacher_logits=moved, distill=distill
    )
    assert not torch.allclose(terms["kl"], changed["kl"])
    plain = loss_terms(out, batch, weights, margin, aux_weight)
    off = loss_terms(
        out,
        batch,
        weights,
        margin,
        aux_weight,
        teacher_logits=teacher,
        distill=replace(distill, alpha=0.0),
    )
    assert torch.equal(off["loss"], plain["loss"])


def test_the_teacher_target_is_the_mean_of_logits():
    torch.manual_seed(0)
    first, second = TeamNet(TINY).eval(), TeamNet(TINY).eval()
    tokens = torch.randn(2, 16, SPEC.n_tokens, N_FEATURES)
    with torch.no_grad():
        alone = first(tokens)["logits"]
        mean = (first(tokens)["logits"] + second(tokens)["logits"]) / 2
    one = mean_teacher_logits([first], tokens)
    two = mean_teacher_logits([first, second], tokens)
    assert torch.equal(one, alone)
    assert torch.equal(two, mean)
    assert not two.requires_grad


def test_config_round_trip_accepts_old_dicts(tmp_path):
    import yaml

    legacy = {"seed": 1, "steps": 8, "degrade": {"version": None}}
    assert TrainConfig.from_dict(legacy).distill is None
    assert TrainConfig.from_dict({**legacy, "distill": None}).distill is None
    assert RunConfig.from_dict({"train": {"seed": 4}}).train.distill is None
    bare = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_baseline.yaml")
    assert bare.train.distill is None
    dumped = bare.to_dict()
    assert "distill" not in dumped["train"]
    assert RunConfig.from_dict(dumped) == bare

    distill = DistillConfig(teacher_runs=TEACHER_RUNS, alpha=0.7, temperature=2.0)
    run = replace(bare, train=replace(bare.train, distill=distill))
    text = yaml.safe_dump(run.to_dict())
    assert RunConfig.from_dict(yaml.safe_load(text)) == run
    assert TrainConfig.from_dict({"distill": None, "steps": 3}).steps == 3

    loaded = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_distill.yaml")
    assert loaded.train.distill == DistillConfig(
        teacher_runs=TEACHER_RUNS, alpha=0.7, temperature=1.0
    )
    assert replace(loaded, train=replace(loaded.train, distill=None)) == bare
    with pytest.raises(ValueError, match="unknown distill"):
        DistillConfig.from_dict({"teacher_runs": ["r"], "beta": 1})
    with pytest.raises(ValueError, match="teacher_runs"):
        DistillConfig.from_dict({"alpha": 0.5})


def test_teacher_paths_for_a_fold_and_for_none(tmp_path):
    first, second = tmp_path / "e0_seed0", tmp_path / "e2_seed1"
    fold = first / "fold-J03WMX"
    fold.mkdir(parents=True)
    (fold / "final.pt").write_bytes(b"fold")
    (first / "fold-none").mkdir()
    (first / "fold-none" / "final.pt").write_bytes(b"not-this")
    (first / "final.pt").write_bytes(b"all")
    (second / "final.pt").parent.mkdir(parents=True)
    (second / "final.pt").write_bytes(b"all2")
    assert teacher_paths([first], "J03WMX") == [fold / "final.pt"]
    assert teacher_paths([first, second], None) == [first / "final.pt", second / "final.pt"]
    assert teacher_paths([first], "none") == [first / "final.pt"]
    with pytest.raises(FileNotFoundError, match="fold-MISSING"):
        teacher_paths([first, second], "MISSING")
    with pytest.raises(FileNotFoundError, match="holdout none") as missing:
        teacher_paths([tmp_path / "absent"], None)
    assert "final.pt" in str(missing.value)
    with pytest.raises(ValueError, match="match id"):
        teacher_paths([first], "a/b")


def test_train_rejects_a_missing_teacher_before_any_step(tmp_path):
    run = _run(distill=DistillConfig(teacher_runs=(str(tmp_path),), alpha=0.7))
    with pytest.raises(ValueError, match="no teacher"):
        train(run, [], tmp_path / "out", device="cpu", log=lambda _: None, resume=False)
    with pytest.raises(FileNotFoundError, match="missing teacher"):
        train(
            run,
            [],
            tmp_path / "out2",
            device="cpu",
            log=lambda _: None,
            resume=False,
            teachers=[tmp_path / "no-such.pt"],
        )


def test_the_train_command_resolves_none_and_a_fold(tmp_path, monkeypatch):
    from aaaag.ml.cli import _train

    calls: list[tuple] = []
    seen: list = []

    def fake_paths(runs, holdout):
        calls.append((tuple(str(path) for path in runs), holdout))
        return [tmp_path / "final.pt"]

    def fake_train(*_args, **kwargs):
        seen.append(kwargs["teachers"])
        return tmp_path / "final.pt"

    monkeypatch.setattr("aaaag.ml.train.teacher_paths", fake_paths)
    monkeypatch.setattr("aaaag.ml.train.train", fake_train)
    monkeypatch.setattr("aaaag.ml.data.load_train_matches", lambda ids, token: dict.fromkeys(ids))
    args = Namespace(
        run=CONFIGS_DIR / "ml" / "team_distill.yaml",
        out=tmp_path / "out",
        steps=None,
        seed=None,
        device="cpu",
        threads=1,
        holdout="none",
        no_resume=True,
    )
    _train(args)
    args.holdout = "J03WMX"
    _train(args)
    assert calls[0][0] == TEACHER_RUNS and calls[0][1] is None
    assert calls[1][0] == TEACHER_RUNS and calls[1][1] == "J03WMX"
    assert seen == [[tmp_path / "final.pt"], [tmp_path / "final.pt"]]


def test_five_teachers_add_a_few_forwards_not_a_frame_loop():
    """Extra cost of 5 teachers on the tiny model: one batched forward each, per step."""
    torch.manual_seed(0)
    student = TeamNet(TINY).train()
    teachers = []
    for seed in range(5):
        torch.manual_seed(100 + seed)
        teacher = TeamNet(TINY).eval()
        teacher.requires_grad_(False)
        teachers.append(teacher)
    tokens = torch.randn(2, 64, SPEC.n_tokens, N_FEATURES)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    def step(use_teachers: bool) -> None:
        optimizer.zero_grad(set_to_none=True)
        if use_teachers:
            with torch.no_grad():
                mean_teacher_logits(teachers, tokens)
        out = student(tokens)
        out["logits"].sum().backward()
        optimizer.step()

    for _ in range(2):
        step(False)
        step(True)
    repeats = 8
    started = time.perf_counter()
    for _ in range(repeats):
        step(False)
    base = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        step(True)
    with_teachers = time.perf_counter() - started
    ratio = with_teachers / base
    print(
        f"DISTILL_COST base_ms {base / repeats * 1000:.2f} "
        f"with5_ms {with_teachers / repeats * 1000:.2f} ratio {ratio:.2f}"
    )
    assert ratio < 15.0
