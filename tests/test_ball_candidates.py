"""The video ball candidate fill (#86) on synthetic tracking and synthetic candidate files."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aaaag import synthetic
from aaaag.contracts import Period
from aaaag.features import ball_candidates as bc

FPS = 25.0
PERIODS = (Period(1, 0, 400),)


def _line(n=400, speed=10.0):
    ball = np.zeros((n, 2))
    ball[:, 0] = np.arange(n) * speed / FPS
    return ball


def _cands(truth, frames, conf=0.9, k=3):
    xy = np.full((len(truth), k, 2), np.nan)
    cf = np.zeros((len(truth), k))
    xy[frames, 0] = truth[frames]
    cf[frames, 0] = conf
    return xy, cf


def test_chain_fills_a_gap_along_the_candidates():
    truth = _line()
    ball = truth.copy()
    ball[100:140] = np.nan
    xy, cf = _cands(truth, np.arange(100, 140, 5))
    out, stats = bc.fill_gaps(ball, xy, cf, PERIODS, FPS)
    assert stats["gaps_filled"] == 1 and stats["frames_filled"] == 40
    assert np.abs(out - truth).max() < 1e-9


def test_decoy_candidate_far_from_the_path_is_ignored():
    truth = _line()
    ball = truth.copy()
    ball[100:140] = np.nan
    xy, cf = _cands(truth, np.arange(100, 140, 5))
    xy[120, 1] = truth[120] + [30.0, 0.0]  # a confident decoy that needs 750 m/s
    cf[120, 1] = 0.99
    out, _ = bc.fill_gaps(ball, xy, cf, PERIODS, FPS)
    assert np.abs(out - truth).max() < 1e-9


def test_low_confidence_or_unbridged_gap_is_left_alone():
    truth = _line()
    ball = truth.copy()
    ball[100:140] = np.nan
    xy, cf = _cands(truth, np.arange(100, 140, 5), conf=0.3)
    out, stats = bc.fill_gaps(ball, xy, cf, PERIODS, FPS)
    assert stats["gaps_filled"] == 0 and np.isnan(out[100:140]).all()
    xy, cf = _cands(truth, np.arange(100, 140, 5))
    cf[120:125] = 0  # a hole of 10 frames between nodes, longer than max_skip_s = 0.4 s
    cf[115:130] = 0
    out, stats = bc.fill_gaps(ball, xy, cf, PERIODS, FPS)
    assert stats["gaps_filled"] == 0


def test_long_gaps_and_open_edges_are_never_touched():
    truth = _line()
    ball = truth.copy()
    ball[:20] = np.nan  # open at the period start
    ball[50:350] = np.nan  # 12 s: over max_gap_s
    xy, cf = _cands(truth, np.arange(0, 400, 5))
    out, stats = bc.fill_gaps(ball, xy, cf, PERIODS, FPS)
    assert stats["gaps"] == 2 and stats["gaps_filled"] == 0
    assert np.isnan(out[:20]).all() and np.isnan(out[50:350]).all()


def test_seen_frames_never_change():
    truth = _line()
    ball = truth.copy()
    ball[100:140] = np.nan
    xy, cf = _cands(truth, np.arange(0, 400, 3))
    xy[:, 0] += 5.0  # candidates disagree with the seen ball everywhere
    out, _ = bc.fill_gaps(ball, xy, cf, PERIODS, FPS)
    seen = np.isfinite(ball[:, 0])
    assert (out[seen] == ball[seen]).all()


def _write(tmp_path, truth, frames, flags=0):
    path = tmp_path / "cand.npz"
    n = len(frames)
    np.savez(
        path,
        frame=frames.astype(np.int32),
        x_m=truth[frames, 0].astype(np.float32),
        y_m=truth[frames, 1].astype(np.float32),
        conf=np.full(n, 0.9, np.float32),
        flags=np.full(n, flags, np.uint8),
    )
    return path


def test_load_pins_the_hash_and_drops_off_ground_rows(tmp_path):
    truth = _line()
    path = _write(tmp_path, truth, np.arange(100, 140))
    with pytest.raises(ValueError, match="sha256"):
        bc.load(path, 400, sha256="0" * 64)
    xy, cf = bc.load(path, 400, sha256=bc.file_sha256(path))
    assert cf[100:140, 0].min() > 0 and cf[:100].max() == 0
    flagged = _write(tmp_path, truth, np.arange(100, 140), flags=bc.FLAG_OFF_GROUND)
    _, cf = bc.load(flagged, 400, sha256=bc.file_sha256(flagged))
    assert cf.max() == 0


def test_filled_bundle_end_to_end_and_off_by_default(tmp_path):
    b = synthetic.generate(seed=5)
    start = b.meta.periods[0].start
    ball = b.tracking.ball_xy.astype(np.float64).copy()
    ball[start : start + 400] = [3.0, 4.0]  # a seen, still ball
    clean = ball.copy()
    ball[start + 200 : start + 240] = np.nan
    t = dataclasses.replace(b.tracking, ball_xy=ball.astype(np.float32), _fingerprint=None)
    b = dataclasses.replace(b, tracking=t.validate())
    frames = np.arange(start + 200, start + 240)
    path = _write(tmp_path, clean.astype(np.float32), np.arange(start + 190, start + 250))
    out = bc.filled_bundle(b, path=str(path), sha256=bc.file_sha256(path))
    assert np.isfinite(out.tracking.ball_xy[frames]).all()
    assert out.tracking.fingerprint() != b.tracking.fingerprint()
    # a file with nothing usable returns the very same bundle
    empty = _write(tmp_path, ball.astype(np.float32), np.arange(0))
    assert bc.filled_bundle(b, path=str(empty), sha256=bc.file_sha256(empty)) is b


def test_pipeline_step_is_registered_and_absent_by_default():
    from aaaag import pipeline

    assert "ball_candidates_fill" in pipeline.PREPROCESS_VERSIONS
