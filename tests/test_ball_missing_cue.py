"""#62: the opt-in `ball_missing` cue. NaN where the ball is seen, the run-length bins, and weight 0
reproducing the ball-state producer exactly. Synthetic matches only; nothing here selects."""

from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import MatchBundle
from aaaag.evidence.ballstate import cues, model

ARM_A = {"bounds_s": [0.5, 2.0, 5.0, 10.0], "llr": [-5.0, -1.0, 2.0, 3.0, 6.0], "closed": "right"}


def _with_gaps(bundle: MatchBundle, gaps: list[tuple[int, int]]) -> MatchBundle:
    ball = bundle.tracking.ball_xy.copy()
    for start, stop in gaps:
        ball[start:stop] = np.nan
    tracking = replace(bundle.tracking, ball_xy=ball, _fingerprint=None)
    return MatchBundle(tracking=tracking, labels=bundle.labels).validate()


def _gaps(bundle: MatchBundle, lengths: list[int]) -> list[tuple[int, int]]:
    """Gaps of the given lengths (frames) inside the first period, well apart."""
    period = bundle.meta.periods[0]
    out, t = [], period.start + 100
    for n in lengths:
        out.append((t, t + n))
        t += n + 200
    assert t < period.stop
    return out


def _clean(bundle: MatchBundle) -> MatchBundle:
    """The bundle with every ball position filled, so only the test's gaps are unseen."""
    ball = bundle.tracking.ball_xy.copy()
    missing = np.isnan(ball).any(axis=1)
    ball[missing] = 0.0
    tracking = replace(bundle.tracking, ball_xy=ball, _fingerprint=None)
    return MatchBundle(tracking=tracking, labels=bundle.labels).validate()


def test_seen_frames_get_no_opinion(small_bundle):
    bundle = _with_gaps(_clean(small_bundle), _gaps(small_bundle, [50]))
    llr = cues.ball_missing_llr(bundle.tracking, **ARM_A)
    seen = bundle.tracking.ball_seen()
    assert np.isnan(llr[seen]).all()
    assert np.isfinite(llr[~seen & bundle.meta.played_mask()]).all()


def test_a_clean_track_never_fires(small_bundle):
    assert np.isnan(cues.ball_missing_llr(_clean(small_bundle).tracking, **ARM_A)).all()


def test_run_length_bins_right_closed(small_bundle):
    assert small_bundle.meta.fps == 25
    # 0.48 s, 0.52 s, 2.0 s (stays in bin 1), 2.04 s, 5.0 s (stays in bin 2), 10.0 s, 12 s
    gaps = _gaps(small_bundle, [12, 13, 50, 51, 125, 250, 300])
    llr = cues.ball_missing_llr(_with_gaps(_clean(small_bundle), gaps).tracking, **ARM_A)
    expected = [-5.0, -1.0, -1.0, 2.0, 2.0, 3.0, 6.0]
    for (start, stop), want in zip(gaps, expected, strict=True):
        assert np.allclose(llr[start:stop], want)


def test_left_closed_one_second_rule(small_bundle):
    fps = small_bundle.meta.fps
    gaps = _gaps(small_bundle, [fps - 1, fps, 3 * fps])
    rule = {"bounds_s": [1.0], "llr": [None, 10.0], "closed": "left"}
    llr = cues.ball_missing_llr(_with_gaps(_clean(small_bundle), gaps).tracking, **rule)
    (a0, b0), (a1, b1), (a2, b2) = gaps
    assert np.isnan(llr[a0:b0]).all()
    assert np.allclose(llr[a1:b1], 10.0) and np.allclose(llr[a2:b2], 10.0)


def test_bad_parameters_are_rejected(small_bundle):
    with pytest.raises(ValueError):
        cues.ball_missing_llr(small_bundle.tracking, bounds_s=[1.0], llr=[1.0])
    with pytest.raises(ValueError):
        cues.ball_missing_llr(small_bundle.tracking, closed="both")


def test_weight_zero_reproduces_the_producer(small_bundle):
    bundle = _with_gaps(small_bundle, _gaps(small_bundle, [80, 400]))
    default = model.ballstate(bundle).proba
    params_only = model.ballstate(bundle, ball_missing=ARM_A).proba
    assert np.array_equal(default, params_only, equal_nan=True)
    on = model.ballstate(bundle, weights={"ball_missing": 1.0}, ball_missing=ARM_A).proba
    assert not np.array_equal(default, on, equal_nan=True)


def test_a_long_missing_ball_raises_p_dead(small_bundle):
    bundle = _with_gaps(_clean(small_bundle), _gaps(small_bundle, [20 * small_bundle.meta.fps]))
    (start, stop) = _gaps(small_bundle, [20 * small_bundle.meta.fps])[0]
    off = model.ballstate(bundle).proba[start:stop, 2]
    on = model.ballstate(bundle, weights={"ball_missing": 1.0}, ball_missing=ARM_A).proba
    assert np.nanmean(on[start:stop, 2]) > np.nanmean(off)
