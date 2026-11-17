"""The truncated temporal MSE smoothing loss on the class head (`train.tmse_weight`).

MS-TCN's T-MSE (Farha and Gall, CVPR 2019). Weight 0 is off: the loss tensor, the returned dict,
the saved config and the trained weights match a run that does not set the field. Torch only, CPU,
small random tensors and synthetic matches.
"""

from __future__ import annotations

import hashlib

import pytest
import yaml

torch = pytest.importorskip("torch")

from aaaag.contracts import A, H  # noqa: E402
from aaaag.ml.checkpoint import load_payload  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch  # noqa: E402
from aaaag.ml.model import ModelConfig  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import (  # noqa: E402
    IGNORE,
    DistillConfig,
    RunConfig,
    TrainConfig,
    _tmse_loss,
    loss_terms,
    train,
)
from aaaag.paths import CONFIGS_DIR  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

N, W, MARGIN = 2, 32, 3
WEIGHTS = torch.tensor([1.0, 1.2, 0.8])


def _batch(labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    if labels is None:
        labels = torch.randint(0, 4, (N, W))
        labels[:, 8:12] = torch.tensor([H, H, A, A])
    batch = {
        "tokens": torch.zeros(N, W, 1),
        "labels": labels,
        "ball_ok": torch.zeros(N, W, dtype=torch.bool),
        "ball": torch.randn(N, W, 2),
    }
    batch["ball_ok"][:, 10] = True
    return batch


def _out(logits: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    if logits is None:
        logits = torch.randn(N, W, 3)
    return {"logits": logits, "ball": torch.randn(N, W, 2)}


def _reference(logits: torch.Tensor, labels: torch.Tensor, tau: float) -> float:
    """The term written out pair by pair, on the interior."""
    inner = slice(MARGIN, W - MARGIN)
    logp = torch.log_softmax(logits[:, inner].double(), dim=-1)
    lab = labels[:, inner]
    total, count = 0.0, 0
    for b in range(logp.shape[0]):
        for t in range(1, logp.shape[1]):
            if lab[b, t] == IGNORE or lab[b, t - 1] == IGNORE:
                continue
            step = (logp[b, t] - logp[b, t - 1]) ** 2
            total += float(torch.clamp(step, max=tau**2).mean())
            count += 1
    return total / count if count else 0.0


# (a) ------------------------------------------------------------------------------------------


def test_weight_zero_matches_omitting_the_argument():
    out, batch = _out(), _batch()
    plain = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3)
    off = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3, tmse_weight=0.0, tmse_tau=1.0)
    assert set(plain) == set(off) == {"loss", "ce", "aux", "acc"}
    for key in plain:
        assert torch.equal(plain[key], off[key])

    teacher = torch.randn(N, W, 3)
    distill = DistillConfig(teacher_runs=("teacher",), alpha=0.5, temperature=2.0)
    base = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3, teacher_logits=teacher, distill=distill)
    still = loss_terms(
        out, batch, WEIGHTS, MARGIN, 0.3, teacher_logits=teacher, distill=distill, tmse_weight=0.0
    )
    assert set(base) == set(still) and "tmse" not in still
    for key in base:
        assert torch.equal(base[key], still[key])


# (b) ------------------------------------------------------------------------------------------


def test_positive_weight_adds_a_non_negative_term_in_both_class_paths():
    out, batch = _out(), _batch()
    plain = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3)
    on = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3, tmse_weight=0.15, tmse_tau=4.0)
    assert "tmse" in on and set(on) == {*plain, "tmse"}
    assert torch.isfinite(on["tmse"]) and float(on["tmse"]) > 0.0
    assert not on["tmse"].requires_grad
    assert torch.equal(on["loss"], plain["loss"] + 0.15 * on["tmse"])
    assert float(on["tmse"]) == pytest.approx(_reference(out["logits"], batch["labels"], 4.0))

    teacher = torch.randn(N, W, 3)
    distill = DistillConfig(teacher_runs=("teacher",), alpha=0.5, temperature=2.0)
    base = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3, teacher_logits=teacher, distill=distill)
    mixed = loss_terms(
        out, batch, WEIGHTS, MARGIN, 0.3, teacher_logits=teacher, distill=distill, tmse_weight=0.15
    )
    assert "kl" in mixed and "tmse" in mixed
    assert torch.equal(mixed["tmse"], on["tmse"])
    assert torch.equal(mixed["loss"], base["loss"] + 0.15 * mixed["tmse"])


def test_no_class_loss_leaves_the_term_out():
    out, batch = _out(), _batch()
    bare = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3, class_loss=False)
    on = loss_terms(out, batch, WEIGHTS, MARGIN, 0.3, class_loss=False, tmse_weight=0.5)
    assert set(on) == set(bare) and "tmse" not in on
    assert torch.equal(on["loss"], bare["loss"])


def test_pairs_with_an_x_label_are_skipped_and_none_left_gives_zero():
    logits = torch.randn(N, W, 3)
    every_x = torch.full((N, W), IGNORE)
    assert float(_tmse_loss(logits, every_x, 0.2, 4.0)) == 0.0
    labels = torch.full((N, W), IGNORE)
    labels[:, 5:9] = H
    expected = torch.log_softmax(logits[:, 5:9], dim=-1).diff(dim=1).pow(2).mean()
    assert float(_tmse_loss(logits, labels, 0.2, 100.0)) == pytest.approx(float(expected))
    assert _tmse_loss(logits, labels, 0.0, 4.0) is None


def test_gradient_reaches_the_later_frame_only():
    logits = torch.randn(1, 2, 3, requires_grad=True)
    labels = torch.tensor([[H, H]])
    _tmse_loss(logits, labels, 1.0, 100.0).backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1]) > 0


# (c) ------------------------------------------------------------------------------------------


def test_constant_logits_over_time_give_zero():
    torch.manual_seed(2)
    constant = torch.randn(N, 1, 3).expand(N, W, 3).contiguous()
    on = loss_terms(_out(constant), _batch(), WEIGHTS, MARGIN, 0.3, tmse_weight=0.15)
    assert float(on["tmse"]) == 0.0
    plain = loss_terms(_out(constant), _batch(), WEIGHTS, MARGIN, 0.3)
    assert torch.equal(on["loss"], plain["loss"])


# (d) ------------------------------------------------------------------------------------------


def test_the_squared_step_is_capped_at_tau_squared():
    tau, scale = 4.0, 100.0
    # Alternate a confident H frame and a confident A frame. The two log-sum-exps are equal, so
    # the steps are exactly (-scale, +scale, 0): two classes are capped and the third is 0.
    logits = torch.zeros(N, W, 3)
    logits[:, 0::2, 0] = scale
    logits[:, 1::2, 1] = scale
    labels = torch.full((N, W), H)
    on = loss_terms(
        _out(logits), _batch(labels), WEIGHTS, MARGIN, 0.3, tmse_weight=1.0, tmse_tau=tau
    )
    assert float(on["tmse"]) == pytest.approx(2.0 * tau**2 / 3.0)

    leaf = logits.clone().requires_grad_(True)
    _tmse_loss(leaf, labels, 1.0, tau).backward()
    assert torch.count_nonzero(leaf.grad) == 0  # a capped step passes no gradient

    torch.manual_seed(3)
    wild = 50.0 * torch.randn(N, W, 3)
    for cap in (0.5, 2.0, 4.0):
        assert float(_tmse_loss(wild, labels, 1.0, cap)) <= cap**2 + 1e-6
    loose = float(_tmse_loss(wild, labels, 1.0, 1e3))
    assert loose > float(_tmse_loss(wild, labels, 1.0, 4.0))


# (e) ------------------------------------------------------------------------------------------


def _digest(run: RunConfig) -> str:
    text = yaml.safe_dump(run.to_dict(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_serialisation_without_the_keys_is_unchanged():
    names = (
        "team_baseline.yaml",
        "team_cm_contact.yaml",
        "team_cm_both_smooth_wide_e84f_prod_s0.yaml",
    )
    for name in names:
        loaded = RunConfig.from_yaml(CONFIGS_DIR / "ml" / name)
        dumped = loaded.to_dict()["train"]
        assert "tmse_weight" not in dumped and "tmse_tau" not in dumped
        explicit = RunConfig(
            token=loaded.token,
            model=loaded.model,
            train=TrainConfig.from_dict({**dumped, "tmse_weight": 0.0, "tmse_tau": 9.0}),
            arch=loaded.arch,
        )
        assert explicit.to_dict() == loaded.to_dict()
        assert _digest(explicit) == _digest(loaded)
        assert RunConfig.from_dict(loaded.to_dict()) == loaded
    assert RunConfig().to_dict() == RunConfig(train=TrainConfig(tmse_weight=0.0)).to_dict()

    on = RunConfig.from_dict({"train": {"tmse_weight": 0.15, "tmse_tau": 4.0}})
    assert on.to_dict()["train"]["tmse_weight"] == 0.15
    assert on.to_dict()["train"]["tmse_tau"] == 4.0
    assert RunConfig.from_dict(on.to_dict()) == on
    with pytest.raises(ValueError, match="tmse_weight"):
        TrainConfig(tmse_weight=-0.1)
    with pytest.raises(ValueError, match="tmse_tau"):
        TrainConfig(tmse_tau=0.0)


# the training loop ----------------------------------------------------------------------------

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
SMALL = SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)


def _run(**train_changes) -> RunConfig:
    train_cfg = TrainConfig(
        seed=3,
        steps=3,
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


def test_training_off_matches_the_plain_run_and_on_logs_the_term(tmp_path):
    matches = [TrainMatch.from_bundle(generate(seed=seed, config=SMALL), SPEC) for seed in (7, 8)]
    logs: dict[str, list[str]] = {}
    for tag, changes in (
        ("plain", {}),
        ("off", {"tmse_weight": 0.0}),
        ("on", {"tmse_weight": 0.2}),
    ):
        logs[tag] = []
        train(
            _run(**changes),
            matches,
            tmp_path / tag,
            device="cpu",
            log=logs[tag].append,
            threads=1,
        )
    plain = load_payload(tmp_path / "plain" / "final.pt")
    off = load_payload(tmp_path / "off" / "final.pt")
    assert plain["meta"]["run"] == off["meta"]["run"]
    assert "tmse_weight" not in plain["meta"]["run"]["train"]
    for key in plain["state_dict"]:
        assert torch.equal(plain["state_dict"][key], off["state_dict"][key])
    assert all("tmse" not in row for row in plain["meta"]["history"])

    on = load_payload(tmp_path / "on" / "final.pt")
    assert on["meta"]["run"]["train"]["tmse_weight"] == 0.2
    assert all(row["tmse"] >= 0.0 for row in on["meta"]["history"])
    assert any(" tmse " in line for line in logs["on"] if line.startswith("set "))
    assert not any(" tmse " in line for line in logs["plain"])
