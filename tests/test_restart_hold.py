"""The E131 restart hold on synthetic labels."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag import synthetic
from aaaag.registry import POSTPROCESS

H, A, D = 0, 1, 2


def _labels(bundle):
    lab = np.asarray(bundle.labels).copy()
    s = bundle.meta.periods[0].start
    lab[s : s + 100] = D
    lab[s + 100 : s + 110] = A
    lab[s + 110 : s + 200] = H
    return lab, s


def test_majority_absorbs_the_minority_after_a_stoppage():
    b = synthetic.generate(seed=1)
    lab, s = _labels(b)
    out = POSTPROCESS.get("restart_hold").fn(lab, b, hold_s=2.0)
    assert (out[s : s + 100] == D).all()
    assert (out[s + 100 : s + 110] == H).all()
    assert out.dtype == lab.dtype and len(out) == len(lab)


def test_first_label_mode_and_zero_hold():
    b = synthetic.generate(seed=1)
    lab, s = _labels(b)
    fn = POSTPROCESS.get("restart_hold").fn
    hold = int(round(2.0 * b.meta.fps))
    first = fn(lab, b, hold_s=2.0, team="first")
    assert (first[s + 100 : s + 100 + hold] == A).all()
    assert (fn(lab, b, hold_s=0.0) == lab).all()
    with pytest.raises(ValueError):
        fn(lab, b, team="last")


def test_short_stoppage_is_left_alone():
    b = synthetic.generate(seed=2)
    lab = np.asarray(b.labels).copy()
    s = b.meta.periods[0].start
    lab[s : s + 20] = D  # under 2 s at 25 fps
    lab[s + 20 : s + 25] = A
    lab[s + 25 : s + 100] = H
    out = POSTPROCESS.get("restart_hold").fn(lab, b, hold_s=2.0, min_dead_s=2.0)
    assert (out[s + 20 : s + 25] == A).all()
