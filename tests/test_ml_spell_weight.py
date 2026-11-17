"""Spell-balanced frame weights (plan E5, the cheap part).

Torch only, CPU, hand-made label windows. gamma 0 or an absent `spell_weight` is the plain loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aaaag.ml.spellweight import SpellWeight, run_lengths, spell_weights  # noqa: E402
from aaaag.ml.train import RunConfig, TrainConfig, loss_terms  # noqa: E402


def _batch(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    n, w = labels.shape
    return {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.zeros(n, w, 2),
    }


def test_run_lengths_and_edges() -> None:
    labels = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    length, at_edge = run_lengths(labels)
    assert length.tolist() == [[2, 2, 3, 3, 3, 2, 2, 1], [4, 4, 4, 4, 4, 4, 4, 4]]
    assert at_edge[0].tolist() == [True, True, False, False, False, False, False, True]
    assert at_edge[1].tolist() == [True, True, True, True, True, True, True, True]


def test_a_run_never_crosses_a_window_boundary() -> None:
    labels = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    length, _ = run_lengths(labels)
    assert length.tolist() == [[4, 4, 4, 4], [4, 4, 4, 4]]


def test_weights_follow_the_spell_length() -> None:
    cfg = SpellWeight(gamma=0.5, ref_s=4.0, min_s=1.0, fps=2)  # 2 frames per second
    # interior runs: H x2 (1 s), A x8 (4 s), D x4, H x6 (3 s), edges on both ends
    labels = torch.tensor([[1] + [0, 0] + [1] * 8 + [2] * 4 + [0] * 6 + [1]])
    w = spell_weights(labels, cfg)[0]
    assert w[0].item() == 1.0 and w[-1].item() == 1.0  # window edges
    assert w[1].item() == pytest.approx((4.0 / 1.0) ** 0.5)  # 1 s spell
    assert w[3].item() == pytest.approx(1.0)  # 4 s spell = ref
    assert all(w[11 + k].item() == 1.0 for k in range(4))  # dead frames keep weight 1
    assert w[16].item() == pytest.approx((4.0 / 3.0) ** 0.5)  # 3 s spell
    assert w.max().item() <= cfg.max_weight + 1e-6


def test_gamma_zero_is_ones() -> None:
    labels = torch.randint(0, 4, (3, 40))
    assert torch.equal(
        spell_weights(labels, SpellWeight(gamma=0.0)), torch.ones(3, 40, dtype=torch.float32)
    )


def test_loss_is_unchanged_when_off_and_changes_when_on() -> None:
    torch.manual_seed(0)
    n, w, margin = 4, 120, 10
    labels = torch.randint(0, 3, (n, w))
    labels[:, :30] = 0
    out = {"logits": torch.randn(n, w, 3), "ball": torch.zeros(n, w, 2)}
    weights = torch.tensor([1.0, 1.0, 0.5])
    plain = loss_terms(out, _batch(labels), weights, margin, 0.0)["loss"]
    off = loss_terms(out, _batch(labels), weights, margin, 0.0, spell=SpellWeight(gamma=0.0))[
        "loss"
    ]
    none = loss_terms(out, _batch(labels), weights, margin, 0.0, spell=None)["loss"]
    on = loss_terms(out, _batch(labels), weights, margin, 0.0, spell=SpellWeight(gamma=0.5))["loss"]
    assert torch.equal(plain, off) and torch.equal(plain, none)
    assert torch.isfinite(on) and not torch.equal(plain, on)


def test_all_equal_weights_reduce_to_the_plain_mean() -> None:
    # every interior frame sits in one long run of the same label: all weights equal, same mean
    n, w, margin = 2, 100, 10
    labels = torch.zeros(n, w, dtype=torch.long)
    out = {"logits": torch.randn(n, w, 3), "ball": torch.zeros(n, w, 2)}
    weights = torch.tensor([1.0, 1.0, 0.5])
    plain = loss_terms(out, _batch(labels), weights, margin, 0.0)["loss"]
    on = loss_terms(out, _batch(labels), weights, margin, 0.0, spell=SpellWeight(gamma=1.0))["loss"]
    assert torch.allclose(plain, on, atol=1e-6)


def test_config_round_trip() -> None:
    cfg = TrainConfig.from_dict({"steps": 5, "spell_weight": {"gamma": 0.5, "ref_s": 5.0}})
    assert cfg.spell_weight == SpellWeight(gamma=0.5, ref_s=5.0)
    run = RunConfig(train=cfg)
    assert run.to_dict()["train"]["spell_weight"]["ref_s"] == 5.0
    assert "spell_weight" not in RunConfig().to_dict()["train"]
    with pytest.raises(ValueError):
        SpellWeight.from_dict({"gamma": 0.5, "typo": 1})
    with pytest.raises(ValueError):
        SpellWeight(gamma=-1.0)
