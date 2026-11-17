"""Boundary-weighted frame weights for the team cross entropy (`train.boundary_weight`, E155).

A frame within `window_s` of a change of the training target label weighs `factor`, 1 elsewhere,
built per match and per period. Null is off: the loss tensor, the saved config and the trained
weights match a run that does not set the field. Numpy and torch on the CPU, hand-made labels and
synthetic matches.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aaaag.contracts import A, D, H, X  # noqa: E402
from aaaag.ml.boundaryweight import BoundaryWeight, boundary_mask, boundary_weights  # noqa: E402
from aaaag.ml.checkpoint import load_payload  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch  # noqa: E402
from aaaag.ml.model import ModelConfig  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.spellweight import SpellWeight, spell_weights  # noqa: E402
from aaaag.ml.train import (  # noqa: E402
    IGNORE,
    RunConfig,
    TrainConfig,
    loss_terms,
    train,
    with_boundary_weight,
)
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402


def _marked(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask).tolist()


# the mask ---------------------------------------------------------------------------------------


def test_a_change_marks_radius_frames_on_each_side() -> None:
    labels = np.array([H] * 10 + [A] * 10, dtype=np.int8)
    assert _marked(boundary_mask(labels, [(0, 20)], 3)) == [7, 8, 9, 10, 11, 12]
    # a dead run between two spells: two changes, overlapping stretches merge
    labels = np.array([H] * 8 + [D] * 2 + [A] * 10, dtype=np.int8)
    assert _marked(boundary_mask(labels, [(0, 20)], 2)) == [6, 7, 8, 9, 10, 11]
    # no change, or radius 0: nothing
    assert not boundary_mask(np.full(20, H, dtype=np.int8), [(0, 20)], 3).any()
    assert not boundary_mask(labels, [(0, 20)], 0).any()


def test_a_change_never_reaches_across_a_period_edge() -> None:
    # the label differs across the period edge at 10: not a change
    labels = np.array([H] * 10 + [A] * 10, dtype=np.int8)
    assert not boundary_mask(labels, [(0, 10), (10, 20)], 3).any()
    # a change at 8 in the first period is clipped at its stop and does not mark 10..12
    labels = np.array([H] * 8 + [A] * 12, dtype=np.int8)
    assert _marked(boundary_mask(labels, [(0, 10), (10, 20)], 5)) == [3, 4, 5, 6, 7, 8, 9]
    # a change at 2 is clipped at the period start
    labels = np.array([H] * 2 + [A] * 18, dtype=np.int8)
    assert _marked(boundary_mask(labels, [(0, 20)], 4)) == [0, 1, 2, 3, 4, 5]


def test_frames_outside_every_period_and_x_steps_are_never_marked() -> None:
    labels = np.array([X] * 5 + [H] * 5 + [A] * 5 + [X] * 5 + [H] * 5, dtype=np.int8)
    # periods 5..15 and 20..25: one change at 10; the X gap and the steps into it are not changes
    mask = boundary_mask(labels, [(5, 15), (20, 25)], 2)
    assert _marked(mask) == [8, 9, 10, 11]
    # an X step inside a period is not a change either
    labels = np.array([H] * 5 + [X] * 5 + [H] * 5, dtype=np.int8)
    assert not boundary_mask(labels, [(0, 15)], 3).any()


def test_a_team_swap_leaves_the_mask_unchanged() -> None:
    rng = np.random.default_rng(0)
    labels = rng.choice([H, A, D, X], size=400, p=[0.4, 0.4, 0.15, 0.05]).astype(np.int8)
    labels = np.repeat(labels[::8], 8)[:400]
    swapped = np.where(labels == H, A, np.where(labels == A, H, labels)).astype(np.int8)
    runs = [(0, 180), (200, 400)]
    assert np.array_equal(boundary_mask(labels, runs, 6), boundary_mask(swapped, runs, 6))


def test_weights_are_factor_near_a_change_and_one_elsewhere() -> None:
    cfg = BoundaryWeight()  # the E155 values
    assert (cfg.factor, cfg.window_s, cfg.radius(25)) == (3.0, 1.0, 25)
    labels = np.array([H] * 100 + [A] * 100, dtype=np.int8)
    w = boundary_weights(labels, [(0, 200)], 25, cfg)
    assert w.dtype == np.float32 and w.shape == (200,)
    assert np.flatnonzero(w == 3.0).tolist() == list(range(75, 125))
    assert np.all(w[:75] == 1.0) and np.all(w[125:] == 1.0)


# the loss ---------------------------------------------------------------------------------------

N, W, MARGIN = 3, 80, 6
CLASS_W = torch.tensor([1.0, 1.2, 0.6])


def _batch(labels: torch.Tensor, near: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    batch = {
        "tokens": torch.zeros(labels.shape[0], labels.shape[1], 1),
        "labels": labels,
        "ball_ok": torch.zeros(labels.shape, dtype=torch.bool),
        "ball": torch.zeros(labels.shape[0], labels.shape[1], 2),
    }
    if near is not None:
        batch["boundary"] = near
    return batch


def _labels() -> torch.Tensor:
    torch.manual_seed(0)
    labels = torch.randint(0, 4, (N, W // 5)).repeat_interleave(5, dim=1)
    labels[:, 20:40] = H
    return labels


def _near(labels: torch.Tensor) -> torch.Tensor:
    rows = [boundary_weights(row.numpy(), [(0, W)], 5, BoundaryWeight()) for row in labels]
    return torch.from_numpy(np.stack(rows))


def test_without_the_key_the_loss_is_the_plain_tensor() -> None:
    labels = _labels()
    out = {"logits": torch.randn(N, W, 3), "ball": torch.zeros(N, W, 2)}
    plain = loss_terms(out, _batch(labels), CLASS_W, MARGIN, 0.0)
    again = loss_terms(out, _batch(labels), CLASS_W, MARGIN, 0.0, spell=None)
    assert torch.equal(plain["loss"], again["loss"]) and plain.keys() == again.keys()
    ones = loss_terms(out, _batch(labels, torch.ones(N, W)), CLASS_W, MARGIN, 0.0)
    assert torch.allclose(plain["loss"], ones["loss"], atol=1e-6)
    on = loss_terms(out, _batch(labels, _near(labels)), CLASS_W, MARGIN, 0.0)
    assert torch.isfinite(on["loss"]) and not torch.allclose(plain["loss"], on["loss"])


@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_boundary_weights_multiply_the_spell_weights(gamma: float) -> None:
    labels = _labels()
    near = _near(labels)
    torch.manual_seed(1)
    logits = torch.randn(N, W, 3)
    out = {"logits": logits, "ball": torch.zeros(N, W, 2)}
    spell = SpellWeight(gamma=gamma, ref_s=2.0, min_s=0.4, fps=5)
    got = loss_terms(out, _batch(labels, near), CLASS_W, MARGIN, 0.0, spell=spell)["ce"]

    inner = slice(MARGIN, W - MARGIN)
    lab = labels[:, inner].reshape(-1)
    frame_w = spell_weights(labels, spell)[:, inner].reshape(-1) * near[:, inner].reshape(-1)
    logp = torch.log_softmax(logits[:, inner].reshape(-1, 3).double(), dim=-1)
    keep = lab != IGNORE
    idx = lab.clamp(max=2)
    w = CLASS_W.double()[idx] * frame_w.double() * keep
    ref = -(w * logp[torch.arange(lab.shape[0]), idx]).sum() / w.sum()
    assert got.item() == pytest.approx(ref.item(), rel=1e-5)


def test_config_round_trip() -> None:
    cfg = TrainConfig.from_dict({"boundary_weight": {"factor": 3.0, "window_s": 1.0}})
    assert cfg.boundary_weight == BoundaryWeight(factor=3.0, window_s=1.0)
    run = RunConfig(train=cfg)
    assert run.to_dict()["train"]["boundary_weight"] == {"factor": 3.0, "window_s": 1.0}
    assert RunConfig.from_dict(run.to_dict()) == run
    assert "boundary_weight" not in RunConfig().to_dict()["train"]
    explicit_null = RunConfig(train=TrainConfig.from_dict({"boundary_weight": None}))
    assert explicit_null.to_dict() == RunConfig().to_dict()
    with pytest.raises(ValueError, match="unknown boundary_weight"):
        BoundaryWeight.from_dict({"factor": 3.0, "typo": 1})
    with pytest.raises(ValueError, match="factor"):
        BoundaryWeight(factor=0.0)
    with pytest.raises(ValueError, match="window_s"):
        BoundaryWeight(window_s=-1.0)


# the training loop ------------------------------------------------------------------------------

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


def test_the_match_weights_follow_its_periods() -> None:
    match = TrainMatch.from_bundle(generate(seed=7, config=SMALL), SPEC)
    weighted = with_boundary_weight(match, BoundaryWeight())
    assert match.boundary is None  # not changed in place
    w = weighted.boundary
    assert w is not None and w.shape == match.labels.shape and w.dtype == np.float32
    assert set(np.unique(w).tolist()) == {1.0, 3.0}
    assert np.all(w[~match.played] == 1.0)
    expected = boundary_weights(
        match.labels, match.bundle.meta.runs(), match.bundle.meta.fps, BoundaryWeight()
    )
    assert np.array_equal(w, expected)


def test_training_off_matches_the_plain_run_and_on_changes_it(tmp_path) -> None:
    matches = [TrainMatch.from_bundle(generate(seed=seed, config=SMALL), SPEC) for seed in (7, 8)]
    for tag, changes in (
        ("plain", {}),
        ("off", {"boundary_weight": None}),
        ("on", {"boundary_weight": BoundaryWeight(factor=3.0, window_s=1.0)}),
    ):
        train(_run(**changes), matches, tmp_path / tag, device="cpu", log=lambda _: None, threads=1)
    assert all(match.boundary is None for match in matches)  # the caller's list is untouched
    plain = load_payload(tmp_path / "plain" / "final.pt")
    off = load_payload(tmp_path / "off" / "final.pt")
    on = load_payload(tmp_path / "on" / "final.pt")
    assert plain["meta"]["run"] == off["meta"]["run"]
    assert "boundary_weight" not in plain["meta"]["run"]["train"]
    for key in plain["state_dict"]:
        assert torch.equal(plain["state_dict"][key], off["state_dict"][key])
    assert on["meta"]["run"]["train"]["boundary_weight"] == {"factor": 3.0, "window_s": 1.0}
    assert any(
        not torch.equal(plain["state_dict"][key], on["state_dict"][key])
        for key in plain["state_dict"]
    )
