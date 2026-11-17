"""Process-pool numpy inference: same bits as the serial loop, including on one core.

Torch is not used. `workers=1` is the serial path; a spawn pool runs the chunks otherwise.
BLAS in each worker is pinned to one thread, which is what makes the counts comparable, so the
serial side of every comparison is pinned the same way.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from aaaag.contracts import Evidence, MatchBundle, MatchMeta, Period, Tracking
from aaaag.evidence.possession.learned_team import learned_team
from aaaag.ml.np_infer import (
    _FAIL_ENV,
    dump_numpy_weights,
    load_numpy_weights,
    parameter_layout,
    predict_logits_np,
)
from aaaag.ml.spec import FEATURE_VERSION, TokenSpec

SPEC = TokenSpec()
TINY = {
    "tokens_per_team": 11,
    "n_features": 14,
    "d_model": 16,
    "n_heads": 2,
    "n_blocks": 1,
    "ffn_mult": 2,
    "dilations": [1, 2, 4],
    "kernel": 3,
    "dropout": 0.0,
}
DEFAULT = {
    "tokens_per_team": 11,
    "n_features": 14,
    "d_model": 64,
    "n_heads": 4,
    "n_blocks": 2,
    "ffn_mult": 2,
    "dilations": [1, 2, 4, 8, 16, 32, 64],
    "kernel": 3,
    "dropout": 0.1,
}
# Two periods and a gap, several chunks at this size: 6 chunks, so 2 and 3 workers both work.
CHUNK = 250
PERIODS = (Period(1, 0, 700), Period(2, 900, 1600))


def _write(path, config, seed, spec=SPEC):
    rng = np.random.default_rng(seed)
    arrays = {
        name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in parameter_layout(config)
    }
    dump_numpy_weights(
        path,
        arch="team_net",
        config=config,
        token_spec=spec.to_dict(),
        arrays=arrays,
        feature_version=FEATURE_VERSION,
    )


def _tracking(n: int, periods: tuple[Period, ...], seed: int) -> Tracking:
    """Same synthetic layout as the numpy timing test: 22 slots, a missing ball, no labels."""
    rng = np.random.default_rng(seed)
    meta = MatchMeta("T", "synthetic", n, 25, 105.0, 68.0, periods).validate()
    player = rng.normal(0, 15, size=(n, 22, 2)).astype(np.float32)
    player[rng.random((n, 22)) < 0.05] = np.nan
    teams = np.zeros(22, dtype=np.int8)
    teams[11:] = 1
    ball = rng.normal(0, 10, size=(n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    return Tracking(meta, player, teams, ball).validate()


def _predict(paths, tracking, *, views, workers, chunk=CHUNK):
    with threadpool_limits(limits=1):
        return predict_logits_np(paths, tracking, chunk=chunk, views=views, workers=workers)


def _assert_same_bits(left: np.ndarray, right: np.ndarray) -> None:
    """Bit identity, including NaN payloads. `array_equal` alone treats NaN as unequal."""
    assert left.shape == right.shape and left.dtype == right.dtype
    assert np.array_equal(left, right, equal_nan=True)
    assert np.array_equal(
        np.ascontiguousarray(left).view(np.uint32),
        np.ascontiguousarray(right).view(np.uint32),
    )


def _assert_pool_closed() -> None:
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline:
        if not mp.active_children():
            return
        time.sleep(0.05)
    raise AssertionError(f"leaked worker processes: {mp.active_children()}")


@pytest.fixture(scope="module")
def tiny_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("np-parallel") / "tiny.json"
    _write(path, TINY, seed=1)
    return path


@pytest.fixture(scope="module")
def ensemble_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("np-ensemble")
    first, second = root / "a.json", root / "b.json"
    _write(first, TINY, seed=1)
    _write(second, TINY, seed=2)
    return first, second


@pytest.fixture(scope="module")
def match():
    return _tracking(1600, PERIODS, seed=0)


def test_workers_do_not_change_a_bit(tiny_path, match):
    serial = {views: _predict([tiny_path], match, views=views, workers=1) for views in (1, 2, 4)}
    assert np.isnan(serial[1].logits).any() and np.isfinite(serial[1].logits).any()
    for views in (1, 2, 4):
        for workers in (2, 3):
            got = _predict([tiny_path], match, views=views, workers=workers)
            _assert_same_bits(serial[views].logits, got.logits)
            assert np.array_equal(serial[views].opinion, got.opinion)
    for workers in (0, -3):
        got = _predict([tiny_path], match, views=1, workers=workers)
        _assert_same_bits(serial[1].logits, got.logits)
    _assert_pool_closed()


def test_ensemble_workers_do_not_change_a_bit(ensemble_paths, match):
    paths = list(ensemble_paths)
    serial = _predict(paths, match, views=1, workers=1)
    for workers in (2, 3):
        got = _predict(paths, match, views=1, workers=workers)
        _assert_same_bits(serial.logits, got.logits)
        assert np.array_equal(serial.opinion, got.opinion)
    _assert_pool_closed()


def test_shorter_than_one_chunk(tiny_path):
    tracking = _tracking(30, (Period(1, 0, 30),), seed=4)
    serial = _predict([tiny_path], tracking, views=4, workers=1, chunk=2048)
    assert serial.logits.shape == (30, 3)
    assert np.isfinite(serial.logits[serial.opinion]).all()
    assert serial.opinion.any()
    for workers in (2, 3):
        got = _predict([tiny_path], tracking, views=4, workers=workers, chunk=2048)
        _assert_same_bits(serial.logits, got.logits)
        assert np.array_equal(serial.opinion, got.opinion)
    _assert_pool_closed()


def test_in_memory_net_stays_on_the_serial_loop(tiny_path, match):
    model = load_numpy_weights(tiny_path)
    serial = _predict([tiny_path], match, views=1, workers=1)
    memory = _predict([model], match, views=1, workers=3)
    _assert_same_bits(serial.logits, memory.logits)
    _assert_pool_closed()


def test_worker_error_surfaces_and_the_pool_shuts_down(tiny_path, monkeypatch):
    tracking = _tracking(30, (Period(1, 0, 30),), seed=5)
    monkeypatch.setenv(_FAIL_ENV, "boom")
    with pytest.raises(RuntimeError, match="worker failed: boom"):
        _predict([tiny_path], tracking, views=1, workers=2, chunk=64)
    _assert_pool_closed()


def test_parallel_run_restores_the_parent_blas_env(tiny_path):
    previous = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "3"
    try:
        tracking = _tracking(30, (Period(1, 0, 30),), seed=6)
        _predict([tiny_path], tracking, views=1, workers=2, chunk=2048)
        assert os.environ.get("OMP_NUM_THREADS") == "3"
    finally:
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous
    _assert_pool_closed()


def test_numpy_producer_workers_match(tiny_path, match):
    bundle = MatchBundle(tracking=match)
    with threadpool_limits(limits=1):
        one = learned_team(bundle, weights=str(tiny_path), backend="numpy", workers=1, chunk=CHUNK)
        two = learned_team(bundle, weights=str(tiny_path), backend="numpy", workers=2, chunk=CHUNK)
    assert isinstance(one, Evidence) and isinstance(two, Evidence)
    one.validate(match.n_frames)
    two.validate(match.n_frames)
    _assert_same_bits(one.proba, two.proba)
    assert one.name == two.name == "learned_team"
    _assert_pool_closed()


@pytest.mark.slow
def test_parallel_speedup_on_sixty_thousand_frames(tmp_path):
    n = 60_000
    path = tmp_path / "full.json"
    _write(path, DEFAULT, seed=6)
    tracking = _tracking(n, (Period(1, 0, n),), seed=0)
    timings: dict[int, float] = {}
    preds = {}
    for workers in (1, 2, 4):
        started = time.perf_counter()
        preds[workers] = _predict([path], tracking, views=1, workers=workers, chunk=2048)
        timings[workers] = time.perf_counter() - started
    t1, t2, t4 = timings[1], timings[2], timings[4]
    print(
        f"SPEED numpy parallel {n} frames one view: "
        f"1 worker {t1:.2f} s, 2 workers {t2:.2f} s ({t1 / t2:.2f}x), "
        f"4 workers {t4:.2f} s ({t1 / t4:.2f}x)"
    )
    _assert_same_bits(preds[1].logits, preds[2].logits)
    _assert_same_bits(preds[1].logits, preds[4].logits)
    assert preds[1].opinion.sum() > n // 2
    # The speedup is printed, not asserted: a shared two-core CI runner can make 4 workers as slow
    # as 1, and a test must not depend on wall-clock time. Bit identity is the contract.
    _assert_pool_closed()
