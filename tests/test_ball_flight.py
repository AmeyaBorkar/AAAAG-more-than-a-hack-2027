"""The E127 ball flight gate on synthetic tracking."""

from __future__ import annotations

import dataclasses

import numpy as np

from aaaag import synthetic
from aaaag.features import ball_flight


def _with_ball(bundle, ball):
    t = dataclasses.replace(bundle.tracking, ball_xy=ball.astype(np.float32), _fingerprint=None)
    return t.validate()


def test_no_params_is_identity():
    b = synthetic.generate(seed=1)
    assert ball_flight.gated_bundle(b) is b
    assert ball_flight.gated(b.tracking) is b.tracking


def test_slow_ball_is_untouched():
    b = synthetic.generate(seed=2)
    start = b.meta.periods[0].start
    ball = np.full((b.meta.n_frames, 2), np.nan)
    n = 200
    ball[start : start + n, 0] = np.arange(n) * 0.4  # 10 m/s at 25 fps
    ball[start : start + n, 1] = 0.0
    t = _with_ball(b, ball)
    assert not ball_flight.flight_mask(t, pad_s=0.2).any()


def test_jump_is_hidden_with_pad_and_merge():
    b = synthetic.generate(seed=3)
    fps = b.meta.fps
    start = b.meta.periods[0].start
    ball = np.zeros((b.meta.n_frames, 2))
    ball[:, 0] = 1.0
    i, j = start + 100, start + 100 + 2 * fps  # a jump out and one back, 2 s apart
    ball[i:j, 0] += 5.0
    t = _with_ball(b, ball)
    pad = int(round(0.2 * fps))
    m = ball_flight.flight_mask(t, pad_s=0.2)
    assert m[i - 1 - pad : i + 1 + pad].all() and m[j - 1 - pad : j + 1 + pad].all()
    assert not m[i + pad + 2 : j - pad - 2].any()
    merged = ball_flight.flight_mask(t, pad_s=0.2, merge_s=3.0, max_s=4.0)
    assert merged[i - 1 - pad : j + 1 + pad].all()
    assert merged.sum() == (j + pad) - (i - 1 - pad) + 1
    short = ball_flight.flight_mask(t, pad_s=0.2, merge_s=3.0, max_s=1.0)
    assert (short == m).all()
    g = ball_flight.gated(t, pad_s=0.2)
    assert np.isnan(g.ball_xy[m]).all() and np.isfinite(g.ball_xy[~m]).all()
    assert g.fingerprint() != t.fingerprint()


def test_unseen_frames_stay_unseen_and_gaps_over_limit_are_not_jumps():
    b = synthetic.generate(seed=4)
    fps = b.meta.fps
    start = b.meta.periods[0].start
    ball = np.zeros((b.meta.n_frames, 2))
    ball[start + 50 : start + 50 + 2 * fps] = np.nan  # a 2 s gap, then the ball 30 m away
    ball[start + 50 + 2 * fps :, 0] = 30.0
    t = _with_ball(b, ball)
    m = ball_flight.flight_mask(t, pad_s=0.2)
    assert not m[np.isnan(ball[:, 0])].any()
    assert not m.any()
