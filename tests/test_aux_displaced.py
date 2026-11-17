"""`train.aux_displaced_m` (#86): the auxiliary ball loss on seen fed balls far from the target.

Synthetic only. The mask is built per match and per variant in `data.build_entry` from the ball
each variant's tokens are built from, against the clean provider ball behind
`TrainMatch.ball_target`, both in pitch metres. 0 stores nothing and leaves every array, the
random stream and the serialised config as they were.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from aaaag.ml import data as data_mod  # noqa: E402
from aaaag.ml.augment import AugmentMix, BallOffset, with_ball  # noqa: E402
from aaaag.ml.checkpoint import load_payload  # noqa: E402
from aaaag.ml.data import (  # noqa: E402
    DegradeMix,
    TrainMatch,
    build_entry,
    build_set,
    displaced_ball_mask,
)
from aaaag.ml.model import ModelConfig  # noqa: E402
from aaaag.ml.spec import TokenSpec  # noqa: E402
from aaaag.ml.train import (  # noqa: E402
    DeviceSet,
    RunConfig,
    TrainConfig,
    replace_train,
    train,
)
from aaaag.paths import CONFIGS_DIR  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
SHORT = SyntheticConfig(minutes_per_half=1.0, gap_minutes=0.1)
OFFSET = AugmentMix(ball_offset=BallOffset(p=1.0, bias_m=4.0, lag_frames=3, noise_m=0.5))
CPU = torch.device("cpu")
WINDOW = 64


@pytest.fixture(scope="module")
def match() -> TrainMatch:
    return TrainMatch.from_bundle(generate(seed=11, config=SHORT), SPEC)


def _state(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state, sort_keys=True, default=str)


def _same_arrays(got, ref) -> None:
    assert got.info == ref.info
    for name in ("coupled", "decoupled", "visible_coupled", "visible_decoupled"):
        assert getattr(got, name).tobytes() == getattr(ref, name).tobytes(), name


def _indexed(match: TrainMatch) -> TrainMatch:
    """`match` whose ball target holds the frame index, so a batch window names its start."""
    index = np.arange(len(match.labels), dtype=np.float32)
    target = np.stack([index, np.zeros_like(index)], axis=1)
    return replace(match, ball_target=target)


def _windows(data: DeviceSet, decoupled: bool, seed: int = 0):
    batch = data.batch(np.random.default_rng(seed), 16, WINDOW, decoupled=decoupled)
    starts = batch["ball"][:, 0, 0].long().tolist()
    return batch["ball_ok"].numpy(), starts


def _full(data: DeviceSet, kind: str) -> np.ndarray:
    return data.aux_mask(kind, 0, 0, len(data.labels[0])).numpy()


# -- (a) 0 is today's mask ------------------------------------------------------------------------


def test_zero_stores_nothing_draws_nothing_and_keeps_every_array(match):
    rng_plain, rng_zero = np.random.default_rng(3), np.random.default_rng(3)
    plain = build_entry(match, rng_plain, SPEC, DegradeMix(), OFFSET)
    zero = build_entry(match, rng_zero, SPEC, DegradeMix(), OFFSET, aux_displaced_m=0.0)
    assert plain.displaced_coupled is None and plain.displaced_decoupled is None
    assert zero.displaced_coupled is None and zero.displaced_decoupled is None
    _same_arrays(zero, plain)
    assert _state(rng_zero) == _state(rng_plain)
    # On: the same draw, the same tokens and masks, and no extra random number either.
    rng_on = np.random.default_rng(3)
    on = build_entry(match, rng_on, SPEC, DegradeMix(), OFFSET, aux_displaced_m=2.0)
    assert _state(rng_on) == _state(rng_plain)
    for name in ("coupled", "decoupled", "visible_coupled", "visible_decoupled"):
        assert getattr(on, name).tobytes() == getattr(plain, name).tobytes(), name


def test_zero_masks_equal_ball_ok_and_not_visible(match):
    entry = build_entry(match, np.random.default_rng(4), SPEC, DegradeMix(), OFFSET)
    indexed = _indexed(match)
    data = DeviceSet([entry], [indexed], CPU)
    assert data.displaced is None
    for decoupled, kind in ((False, "coupled"), (True, "decoupled")):
        visible = getattr(entry, f"visible_{kind}")
        expected = match.target_ok & ~visible
        assert np.array_equal(_full(data, kind), expected)
        ok, starts = _windows(data, decoupled)
        for row, start in zip(ok, starts, strict=True):
            assert np.array_equal(row, expected[start : start + WINDOW])


# -- (b) a 5 m displacement on chosen seen frames, threshold 2 m ----------------------------------


def _shift_fed_balls(monkeypatch, coupled, decoupled, small) -> None:
    """Move the fed balls after the augmentations: 5 m on `coupled` / `decoupled`, 1 m on `small`.

    Draws no random number, so the rest of the draw (the decoupled hiding included) is unchanged.
    The clean provider ball, the target, is copied and never written.
    """
    real = data_mod.apply_augmentations

    def shifted(tracking, clean_ball, mix, rng):
        tracking, ball = real(tracking, clean_ball, mix, rng)
        fed = np.array(tracking.ball_xy, dtype=np.float32, copy=True)
        fed[coupled] += np.float32([3.0, 4.0])
        fed[small] += np.float32([0.6, 0.8])
        other = np.array(ball, dtype=np.float32, copy=True)
        other[decoupled] += np.float32([-4.0, 3.0])
        other[small] += np.float32([0.8, -0.6])
        return with_ball(tracking, fed), other

    monkeypatch.setattr(data_mod, "apply_augmentations", shifted)


def _pick(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    where = np.flatnonzero(mask)
    assert len(where) >= count
    return np.sort(rng.choice(where, size=count, replace=False))


def test_exactly_the_displaced_seen_frames_are_added(match, monkeypatch):
    mix = DegradeMix(version=None)  # the fed ball is the clean ball until the shift below
    base = build_entry(match, np.random.default_rng(5), SPEC, mix)
    pick = np.random.default_rng(0)
    ok_c = base.visible_coupled & match.target_ok
    ok_d = base.visible_decoupled & match.target_ok
    coupled = _pick(ok_c, 40, pick)
    dead_seen = _pick(base.visible_coupled & ~match.target_ok, 10, pick)  # moved, never supervised
    decoupled = _pick(ok_d, 40, pick)
    hidden = _pick(~base.visible_decoupled & match.target_ok & base.visible_coupled, 10, pick)
    taken = np.zeros(len(match.labels), dtype=bool)
    taken[np.concatenate([coupled, dead_seen, decoupled, hidden])] = True
    small = _pick(ok_c & ok_d & ~taken, 40, pick)  # 1 m: under the threshold
    moved_c = np.concatenate([coupled, dead_seen])
    moved_d = np.concatenate([decoupled, hidden])
    _shift_fed_balls(monkeypatch, moved_c, moved_d, small)

    off = build_entry(match, np.random.default_rng(5), SPEC, mix)
    on = build_entry(match, np.random.default_rng(5), SPEC, mix, aux_displaced_m=2.0)
    # The shift hides nothing and draws nothing, but the coupled and decoupled balls moved.
    assert off.visible_coupled.tobytes() == base.visible_coupled.tobytes()
    assert off.visible_decoupled.tobytes() == base.visible_decoupled.tobytes()
    assert not np.array_equal(off.coupled, base.coupled)
    assert not np.array_equal(off.decoupled, base.decoupled)
    _same_arrays_except_info(on, off)

    n = len(match.labels)
    want_c = np.zeros(n, dtype=bool)
    want_c[moved_c] = True
    want_d = np.zeros(n, dtype=bool)
    want_d[decoupled] = True  # `hidden` frames are NaN in the decoupled input: not displaced
    assert np.array_equal(on.displaced_coupled, want_c)
    assert np.array_equal(on.displaced_decoupled, want_d)

    plain = DeviceSet([off], [match], CPU)
    wide = DeviceSet([on], [match], CPU)
    for kind, added in (("coupled", coupled), ("decoupled", decoupled)):
        before, after = _full(plain, kind), _full(wide, kind)
        assert not (before & ~after).any()  # nothing is removed
        assert np.array_equal(np.flatnonzero(after & ~before), added)  # dead_seen is not alive


def test_the_batch_reads_the_widened_mask(match, monkeypatch):
    base = build_entry(match, np.random.default_rng(6), SPEC, DegradeMix(version=None))
    moved = _pick(base.visible_coupled & match.target_ok, 200, np.random.default_rng(1))
    _shift_fed_balls(monkeypatch, moved, moved, moved[:0])
    on = build_entry(
        match, np.random.default_rng(6), SPEC, DegradeMix(version=None), aux_displaced_m=2.0
    )
    data = DeviceSet([on], [_indexed(match)], CPU)
    for decoupled, kind in ((False, "coupled"), (True, "decoupled")):
        visible = getattr(on, f"visible_{kind}")
        displaced = getattr(on, f"displaced_{kind}")
        expected = match.target_ok & (~visible | displaced)
        assert np.array_equal(_full(data, kind), expected)
        ok, starts = _windows(data, decoupled, seed=2)
        for row, start in zip(ok, starts, strict=True):
            assert np.array_equal(row, expected[start : start + WINDOW])


def test_the_mask_is_rebuilt_with_each_set(match):
    sets = [
        build_set([match], index, 9, SPEC, DegradeMix(), workers=1, augment=OFFSET)
        for index in (0, 1)
    ]
    on = [
        build_set(
            [match], index, 9, SPEC, DegradeMix(), workers=1, augment=OFFSET, aux_displaced_m=1.0
        )
        for index in (0, 1)
    ]
    for plain, wide in zip(sets, on, strict=True):
        assert plain[0].displaced_coupled is None
        _same_arrays_except_info(wide[0], plain[0])
        for kind in ("coupled", "decoupled"):
            displaced = getattr(wide[0], f"displaced_{kind}")
            assert not (displaced & ~getattr(wide[0], f"visible_{kind}")).any()
    assert not np.array_equal(on[0][0].displaced_decoupled, on[1][0].displaced_decoupled)


def _same_arrays_except_info(got, ref) -> None:
    for name in ("coupled", "decoupled", "visible_coupled", "visible_decoupled"):
        assert getattr(got, name).tobytes() == getattr(ref, name).tobytes(), name


def test_displaced_ball_mask_is_strict_and_nan_is_false():
    target = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
    fed = np.array([[2.0, 0.0], [3.0, 4.0], [np.nan, np.nan], [9.0, 9.0], [1.0, 1.5]])
    got = displaced_ball_mask(fed.astype(np.float32), target.astype(np.float32), 2.0)
    assert got.tolist() == [False, True, False, False, False]


# -- (c) the serialised config --------------------------------------------------------------------


def _digest(run: RunConfig) -> str:
    return hashlib.sha256(json.dumps(run.to_dict(), sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize(
    "name", ["team_baseline.yaml", "team_cm_both_smooth_wide_e84f_prod_s0.yaml"]
)
def test_a_yaml_without_the_key_serialises_as_before(name):
    raw = yaml.safe_load((CONFIGS_DIR / "ml" / name).read_text(encoding="utf-8"))
    assert "aux_displaced_m" not in raw.get("train", {})
    run = RunConfig.from_dict(raw)
    assert run.train.aux_displaced_m == 0.0
    off = run.to_dict()
    assert "aux_displaced_m" not in off["train"]
    assert _digest(replace_train(run, aux_displaced_m=0.0)) == _digest(run)
    assert _digest(RunConfig.from_dict(off)) == _digest(run)
    # On, the key is the only difference, and it round-trips.
    on = replace_train(run, aux_displaced_m=2.0).to_dict()
    assert on["train"].pop("aux_displaced_m") == 2.0
    assert on == off
    assert RunConfig.from_dict(replace_train(run, aux_displaced_m=2.0).to_dict()) == replace_train(
        run, aux_displaced_m=2.0
    )


def test_the_default_config_has_no_key_and_bad_values_are_refused():
    assert TrainConfig().aux_displaced_m == 0.0
    assert "aux_displaced_m" not in RunConfig.from_dict({}).to_dict()["train"]
    assert TrainConfig(aux_displaced_m=2).aux_displaced_m == 2.0
    assert isinstance(TrainConfig(aux_displaced_m=2).aux_displaced_m, float)
    for bad in (-1.0, float("nan"), float("inf"), True, "2"):
        with pytest.raises(ValueError, match="aux_displaced_m"):
            TrainConfig(aux_displaced_m=bad)


# -- end to end -----------------------------------------------------------------------------------


def test_a_short_training_run_with_the_option_on(match, tmp_path):
    tiny = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
    cfg = TrainConfig(
        seed=3,
        steps=4,
        refresh_steps=2,
        batch_size=2,
        window=WINDOW,
        warmup_steps=1,
        degrade=DegradeMix(version=None),
        augment=OFFSET,
        amp=False,
        aux_displaced_m=1.0,
    )
    run = RunConfig(token=SPEC, model=tiny, train=cfg)
    final = train(run, [match], tmp_path, device="cpu", log=lambda _: None, resume=False, threads=1)
    meta = load_payload(final)["meta"]
    assert meta["run"]["train"]["aux_displaced_m"] == 1.0
    assert len(meta["history"]) == 2  # one entry per refreshed set
    assert all(np.isfinite(row["aux"]) for row in meta["history"])
