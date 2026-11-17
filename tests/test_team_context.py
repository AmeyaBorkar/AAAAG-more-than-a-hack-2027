"""team_context.frame_features on the synthetic small_bundle."""

from __future__ import annotations

import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.features.team_context import (
    FEATURE_VERSION,
    feature_groups,
    feature_names,
    frame_features,
)


def _mismatches(a: np.ndarray, b: np.ndarray) -> int:
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return int((~same).sum())


def _assert_equal(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        if a.shape != b.shape:
            raise AssertionError(f"shape {a.shape} vs {b.shape}")
        raise AssertionError(f"value mismatches {_mismatches(a, b)} of {a.size}")


def _assert_bits(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"shape {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
        raise AssertionError(f"not bit-identical, value mismatches {_mismatches(a, b)}")


def _column_key(col: np.ndarray) -> bytes:
    x = np.ascontiguousarray(col, dtype=np.float32).copy()
    nans = np.isnan(x)
    x[nans] = np.float32(np.nan)
    neg0 = np.signbit(x) & ~nans & (x == 0)
    x[neg0] = np.float32(0.0)
    return x.tobytes()


def _is_column_permutation(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    left = Counter(_column_key(a[:, i]) for i in range(a.shape[1]))
    right = Counter(_column_key(b[:, i]) for i in range(b.shape[1]))
    return left == right


def _one_thread():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1)


def test_small_bundle_contract(small_bundle):
    tracking = small_bundle.tracking
    names, groups = feature_names(), feature_groups()
    if FEATURE_VERSION != "1" or len(names) != 109 or len(groups) != 109:
        raise AssertionError("schema")
    if len(set(names)) != 109 or set(groups) != {"G1", "G2", "G3", "G4", "G5", "G6"}:
        raise AssertionError("names or groups")
    if not isinstance(names, tuple) or not isinstance(groups, tuple):
        raise AssertionError("names and groups must be tuples")
    first = frame_features(tracking)
    second = frame_features(tracking)
    if first.names != names or first.groups != groups:
        raise AssertionError("frame_features schema drifted")
    if first.X.shape != (tracking.n_frames, 109) or first.X.dtype != np.float32:
        raise AssertionError(f"shape {first.X.shape} {first.X.dtype}")
    n_inf = int(np.isinf(first.X).sum())
    if n_inf:
        raise AssertionError(f"{n_inf} infinities")
    _assert_bits(first.X, second.X)


def test_period_1_ignores_period_2_tracking(small_bundle):
    tracking = small_bundle.tracking
    periods = tracking.meta.periods
    if [p.index for p in periods] != [1, 2]:
        raise AssertionError("fixture periods")
    before = frame_features(tracking).X
    player = tracking.player_xy.copy()
    ball = tracking.ball_xy.copy()
    start, stop = periods[1].start, periods[1].stop
    player[start:stop, :, 0] += np.float32(30)
    ball[start:stop, 0] += np.float32(5)
    altered = replace(tracking, player_xy=player, ball_xy=ball, _fingerprint=None)
    after = frame_features(altered).X
    p1 = slice(periods[0].start, periods[0].stop)
    _assert_bits(before[p1], after[p1])
    if before[start:stop].tobytes() == after[start:stop].tobytes():
        raise AssertionError("period 2 did not change")
    _assert_bits(frame_features(tracking).X, before)


def test_missing_ball_frames_have_nan_g1(small_bundle):
    tracking = small_bundle.tracking
    feats = frame_features(tracking)
    missing = (~tracking.ball_seen()) & tracking.meta.played_mask()
    if int(missing.sum()) == 0:
        raise AssertionError("fixture has no missing ball inside periods")
    g1 = [i for i, group in enumerate(feats.groups) if group == "G1"]
    if not g1:
        raise AssertionError("no G1 columns")
    block = feats.X[missing][:, g1]
    n_bad = int((~np.isnan(block)).sum())
    if n_bad:
        raise AssertionError(f"{n_bad} non-NaN G1 values on missing-ball frames")


def test_swapping_slot_teams_recomputes(small_bundle):
    tracking = small_bundle.tracking
    original = frame_features(tracking)
    flipped = np.empty_like(tracking.slot_team)
    flipped[tracking.slot_team == HOME] = AWAY
    flipped[tracking.slot_team == AWAY] = HOME
    swapped = replace(tracking, slot_team=flipped, _fingerprint=None)
    once = frame_features(swapped)
    twice = frame_features(swapped)
    _assert_bits(once.X, twice.X)
    if np.array_equal(original.X, once.X, equal_nan=True):
        raise AssertionError("swap left features unchanged")
    if _is_column_permutation(original.X, once.X):
        raise AssertionError("swap was only a column permutation")
    names = original.names
    h, a, bx = names.index("h_d1"), names.index("a_d1"), names.index("ball_X")
    _assert_bits(once.X[:, h], original.X[:, a])
    _assert_bits(once.X[:, a], original.X[:, h])
    _assert_equal(once.X[:, bx], -original.X[:, bx])


@pytest.mark.slow
def test_frame_features_handles_146k_frames_within_four_seconds():
    n_frames, n_slots = 146_000, 36
    gap, half = 2_000, 71_000
    meta = MatchMeta(
        match_id="speed",
        source="synthetic",
        n_frames=n_frames,
        fps=25,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(
            Period(1, gap, gap + half),
            Period(2, 2 * gap + half, 2 * gap + 2 * half),
        ),
    )
    rng = np.random.default_rng(0)
    player = rng.standard_normal((n_frames, n_slots, 2)).astype(np.float32)
    ball = rng.standard_normal((n_frames, 2)).astype(np.float32)
    ball[rng.random(n_frames) < 0.41] = np.nan
    player[rng.random((n_frames, n_slots)) < 0.05] = np.nan
    teams = np.array([HOME] * 18 + [AWAY] * 18, dtype=np.int8)
    tracking = Tracking(meta, player, teams, ball).validate()
    with _one_thread():
        started = time.perf_counter()
        out = frame_features(tracking)
        elapsed = time.perf_counter() - started
    if out.X.shape != (n_frames, 109) or out.X.dtype != np.float32:
        raise AssertionError(f"shape {out.X.shape} {out.X.dtype}")
    if elapsed > 4.0:
        raise AssertionError(f"{elapsed:.3f}s")
