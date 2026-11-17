"""Spell weights on the A-B-A-merged view (`spell_weight.length_view: aba1.0`, E154).

The default view (provider) is the behaviour before the key existed: same config dict, same loss,
same checkpoint. The aba view measures each spell once per match on `aba_merge(labels, fps, 1.0)`,
for the weights only. Synthetic labels and matches; torch tests skip where torch is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.ml.spellweight import (
    SpellWeight,
    aba_merge,
    match_run_frames,
    match_spell_seconds,
)

H, A, D, X = 0, 1, 2, 3


def _reference_aba(labels: np.ndarray, fps: int, n_s: float) -> np.ndarray:
    """The loop of `aba_merge` in analysis/scripts/e106_hold_regret.py, copied as it is."""
    out = np.array(labels, copy=True)
    limit = n_s * fps
    live = np.isin(out, (H, A))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], live.astype(np.int8), [0]))))
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        seg = out[lo:hi]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = [0, *change.tolist()]
        runs = [[seg[s], s, e] for s, e in zip(starts, [*starts[1:], len(seg)], strict=True)]
        while True:
            best = None
            for i in range(1, len(runs) - 1):
                lab, s, e = runs[i]
                if runs[i - 1][0] == runs[i + 1][0] != lab and e - s < limit:
                    if best is None or e - s < runs[best][2] - runs[best][1]:
                        best = i
            if best is None:
                break
            left, _, right = runs[best - 1], runs[best], runs[best + 1]
            runs[best - 1 : best + 2] = [[left[0], left[1], right[2]]]
        for lab, s, e in runs:
            seg[s:e] = lab
        out[lo:hi] = seg
    return out


def _runs(spec: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([np.full(n, lab, dtype=np.int8) for lab, n in spec])


def _flicker(rng: np.random.Generator, n_runs: int) -> np.ndarray:
    """Random runs, mostly live, with many runs under a second at 10 fps and a few dead gaps."""
    labs = rng.choice([H, A, D, X], size=n_runs, p=[0.42, 0.42, 0.1, 0.06])
    lens = np.where(
        rng.random(n_runs) < 0.6, rng.integers(1, 10, n_runs), rng.integers(10, 60, n_runs)
    )
    return np.repeat(labs, lens).astype(np.int8)


def test_merge_matches_the_e106_loop_on_random_flicker() -> None:
    rng = np.random.default_rng(154)
    for _ in range(300):
        labels = _flicker(rng, int(rng.integers(1, 80)))
        for fps, n_s in ((10, 1.0), (10, 0.5), (25, 1.0), (10, 3.0)):
            assert np.array_equal(aba_merge(labels, fps, n_s), _reference_aba(labels, fps, n_s))


def test_shortest_interior_goes_first() -> None:
    # 1 fps, N 5 s: H5 A2 H1 A4 H6. H1 is the shortest interior run and merges first (A2 H1 A4 -> A7),
    # after which A7 is not shorter than 5 s. Merging A2 first would have given one H18 run.
    labels = _runs([(H, 5), (A, 2), (H, 1), (A, 4), (H, 6)])
    got = aba_merge(labels, 1, 5.0)
    assert np.array_equal(got, _runs([(H, 5), (A, 7), (H, 6)]))
    assert np.array_equal(got, _reference_aba(labels, 1, 5.0))


def test_dead_frames_end_a_live_run_and_stay() -> None:
    # 10 fps, N 1 s: the A3 flicker inside the first possession goes; the A3 after a dead gap is an
    # outer run of its live segment and stays; dead and X frames never change.
    labels = _runs([(H, 12), (A, 3), (H, 12), (D, 5), (A, 3), (H, 20), (X, 4)])
    got = aba_merge(labels, 10, 1.0)
    assert np.array_equal(got, _runs([(H, 27), (D, 5), (A, 3), (H, 20), (X, 4)]))
    # a one-second interior run is not shorter than N and stays
    edge = _runs([(H, 12), (A, 10), (H, 12)])
    assert np.array_equal(aba_merge(edge, 10, 1.0), edge)


def test_match_lengths_take_the_merged_spell() -> None:
    labels = _runs([(H, 12), (A, 3), (H, 12), (D, 5), (A, 3), (H, 20)])
    frames = match_run_frames(labels)
    assert frames.tolist() == [12] * 12 + [3] * 3 + [12] * 12 + [5] * 5 + [3] * 3 + [20] * 20
    seconds = match_spell_seconds(labels, 10.0, "aba1.0")
    assert seconds.dtype == np.float32 and seconds.shape == labels.shape
    assert np.allclose(seconds[:27], 2.7) and np.allclose(seconds[32:35], 0.3)
    provider = match_spell_seconds(labels, 10.0, "provider")
    assert np.allclose(provider, frames / 10.0)
    assert match_run_frames(np.zeros(0, dtype=np.int8)).size == 0
    with pytest.raises(ValueError):
        match_spell_seconds(labels, 10.0, "aba2.0")


def test_config_default_and_validation() -> None:
    assert SpellWeight().length_view == "provider"
    assert SpellWeight.from_dict({"gamma": 0.25}) == SpellWeight(gamma=0.25)
    aba = SpellWeight.from_dict({"gamma": 0.25, "length_view": "aba1.0"})
    assert aba.length_view == "aba1.0" and aba.match_level
    assert not SpellWeight(gamma=0.25).match_level
    assert not SpellWeight(gamma=0.0, length_view="aba1.0").match_level
    with pytest.raises(ValueError):
        SpellWeight(length_view="aba")
    with pytest.raises(ValueError):
        SpellWeight.from_dict({"gamma": 0.25, "length_view": "aba0.5"})


# -- torch -----------------------------------------------------------------------------------------


def _torch():
    return pytest.importorskip("torch")


def _batch(torch, labels, spell_s=None):
    n, w = labels.shape
    out = {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.zeros(n, w, 2),
    }
    if spell_s is not None:
        out["spell_s"] = spell_s
    return out


def test_aba_weights_read_the_match_lengths() -> None:
    torch = _torch()
    from aaaag.ml.spellweight import spell_weights

    cfg = SpellWeight(gamma=0.5, ref_s=4.0, min_s=1.0, fps=2, length_view="aba1.0")
    labels = torch.tensor([[H, H, A, A, D, D, H, X]])
    seconds = torch.tensor([[9.0, 9.0, 0.5, 2.0, 1.0, 1.0, 16.0, 3.0]])
    w = spell_weights(labels, cfg, seconds)[0]
    assert w[0].item() == pytest.approx((4.0 / 9.0) ** 0.5)  # no window edge rule
    assert w[2].item() == pytest.approx(2.0)  # 0.5 s is clamped to min_s
    assert w[3].item() == pytest.approx(2.0**0.5)
    assert w[4].item() == 1.0 and w[5].item() == 1.0 and w[7].item() == 1.0  # dead and X
    assert w[6].item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        spell_weights(labels, cfg)
    off = SpellWeight(gamma=0.0, length_view="aba1.0")
    assert torch.equal(spell_weights(labels, off), torch.ones(1, 8))
    # the provider view does not read the lengths
    provider = SpellWeight(gamma=0.5, ref_s=4.0, min_s=1.0, fps=2)
    assert torch.equal(spell_weights(labels, provider, seconds), spell_weights(labels, provider))


def test_loss_wiring() -> None:
    torch = _torch()
    from aaaag.ml.train import loss_terms

    torch.manual_seed(0)
    n, w, margin = 2, 60, 5
    labels = torch.full((n, w), H)
    labels[:, 15:18] = A
    labels[:, 30:40] = D
    labels[:, 40:52] = A
    out = {"logits": torch.randn(n, w, 3), "ball": torch.zeros(n, w, 2)}
    weights = torch.tensor([1.0, 1.1, 0.6])
    provider = SpellWeight(gamma=0.5, ref_s=6.0, min_s=0.5, fps=25)
    plain = loss_terms(out, _batch(torch, labels), weights, margin, 0.0, spell=provider)
    # the provider view ignores lengths a batch may carry
    carried = torch.full((n, w), 0.04)
    same = loss_terms(out, _batch(torch, labels, carried), weights, margin, 0.0, spell=provider)
    assert set(plain) == set(same)
    for key in plain:
        assert torch.equal(plain[key], same[key])
    # aba with match lengths equal to the window's run lengths and no edge run differs from the
    # provider loss only by the edge rule; give every frame the window length to compare exactly
    aba = SpellWeight(gamma=0.5, ref_s=6.0, min_s=0.5, fps=25, length_view="aba1.0")
    with pytest.raises(ValueError):
        loss_terms(out, _batch(torch, labels), weights, margin, 0.0, spell=aba)
    from aaaag.ml.spellweight import run_lengths

    length, at_edge = run_lengths(labels)
    seconds = length.float() / 25.0
    seconds = torch.where(at_edge, torch.full_like(seconds, 6.0), seconds)  # weight 1 at an edge
    got = loss_terms(out, _batch(torch, labels, seconds), weights, margin, 0.0, spell=aba)
    assert torch.allclose(got["loss"], plain["loss"], rtol=0.0, atol=1e-6)
    longer = loss_terms(
        out, _batch(torch, labels, torch.full((n, w), 6.0)), weights, margin, 0.0, spell=aba
    )
    unweighted = loss_terms(out, _batch(torch, labels), weights, margin, 0.0)
    assert torch.allclose(longer["loss"], unweighted["loss"], rtol=0.0, atol=1e-6)


def test_run_config_omits_the_default_view() -> None:
    _torch()
    from aaaag.ml.train import RunConfig, TrainConfig

    bare = RunConfig(train=TrainConfig(spell_weight=SpellWeight(gamma=0.25)))
    explicit = RunConfig(
        train=TrainConfig.from_dict(
            {"spell_weight": {"gamma": 0.25, "length_view": "provider"}},
        )
    )
    assert bare.to_dict() == explicit.to_dict()
    assert "length_view" not in bare.to_dict()["train"]["spell_weight"]
    on = RunConfig(train=TrainConfig(spell_weight=SpellWeight(gamma=0.25, length_view="aba1.0")))
    assert on.to_dict()["train"]["spell_weight"]["length_view"] == "aba1.0"
    assert RunConfig.from_dict(on.to_dict()) == on
    assert RunConfig.from_dict(bare.to_dict()) == bare


# -- the training loop, end to end ----------------------------------------------------------------


def _train_tiny(tmp_path, name: str, spell: SpellWeight):
    torch = _torch()
    from aaaag.ml.checkpoint import load_payload
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.model import ModelConfig
    from aaaag.ml.spec import TokenSpec
    from aaaag.ml.train import RunConfig, TrainConfig, train
    from aaaag.synthetic import SyntheticConfig, generate

    spec = TokenSpec()
    matches = [
        TrainMatch.from_bundle(
            generate(seed=seed, config=SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)), spec
        )
        for seed in (7, 8)
    ]
    run = RunConfig(
        token=spec,
        model=ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0),
        train=TrainConfig(
            seed=3,
            steps=3,
            refresh_steps=3,
            batch_size=2,
            window=64,
            warmup_steps=1,
            lr=3e-3,
            degrade=DegradeMix(version=None),
            amp=False,
            spell_weight=spell,
        ),
    )
    out = tmp_path / name
    train(run, matches, out, device="cpu", log=lambda _line: None, threads=1)
    del torch
    return load_payload(out / "final.pt"), load_payload(out / "last.pt"), matches


def test_provider_view_is_byte_identical_and_aba_trains(tmp_path) -> None:
    torch = _torch()
    plain, plain_last, plain_matches = _train_tiny(tmp_path, "plain", SpellWeight(gamma=0.25))
    explicit, explicit_last, _ = _train_tiny(
        tmp_path, "explicit", SpellWeight.from_dict({"gamma": 0.25, "length_view": "provider"})
    )
    assert plain["state_dict"].keys() == explicit["state_dict"].keys()
    for key in plain["state_dict"]:
        assert torch.equal(plain["state_dict"][key], explicit["state_dict"][key])
    assert plain["meta"]["run"] == explicit["meta"]["run"]
    assert "length_view" not in plain["meta"]["run"]["train"]["spell_weight"]
    for left, right in zip(plain_last["history"], explicit_last["history"], strict=True):
        assert {k: left[k] for k in ("loss", "ce")} == {k: right[k] for k in ("loss", "ce")}
    assert all(getattr(m, "spell_s", None) is None for m in plain_matches)

    aba, aba_last, aba_matches = _train_tiny(
        tmp_path, "aba", SpellWeight(gamma=0.25, length_view="aba1.0")
    )
    assert aba["meta"]["run"]["train"]["spell_weight"]["length_view"] == "aba1.0"
    for match in aba_matches:
        assert match.spell_s.shape == match.labels.shape
        assert np.allclose(
            match.spell_s,
            match_spell_seconds(match.labels, float(match.bundle.meta.fps), "aba1.0"),
        )
    assert all(np.isfinite(row["loss"]) for row in aba_last["history"])
    assert any(
        not torch.equal(plain["state_dict"][key], aba["state_dict"][key])
        for key in plain["state_dict"]
    )
