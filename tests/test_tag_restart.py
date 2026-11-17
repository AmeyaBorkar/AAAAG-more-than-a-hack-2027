"""tag_restart: set-piece tag windows as dead-channel evidence (#86, P2). Synthetic matches only."""

from __future__ import annotations

import dataclasses

import numpy as np

from aaaag.contracts import Evidence, TagEvent
from aaaag.evidence.ballstate.tag_restart import tag_restart, tag_windows
from aaaag.fusion.factorised import factorised


def _tag(code: str, start: int, stop: int, kind: str | None) -> TagEvent:
    values = {"Half": "1", "Team": "N/A"}
    if kind is not None:
        values["Type"] = kind
    return TagEvent(code, start, stop, None, 1, values)


def _with_tags(bundle, tags):
    return dataclasses.replace(bundle, tags=tags)


def test_no_tags_gives_no_opinion(small_bundle):
    ev = tag_restart(_with_tags(small_bundle, None), llr=1.5)
    ev.validate(small_bundle.meta.n_frames)
    assert np.isnan(ev.proba).all()


def test_window_frames_get_the_bonus_and_a_team_tie(small_bundle):
    tags = [_tag("SET PIECES", 100, 150, "Throw-in"), _tag("SET PIECES", 400, 430, "Corner Kick")]
    ev = tag_restart(_with_tags(small_bundle, tags), llr=1.5)
    ev.validate(small_bundle.meta.n_frames)
    inside = np.zeros(small_bundle.meta.n_frames, dtype=bool)
    inside[100:150] = inside[400:430] = True
    p = 1.0 / (1.0 + np.exp(-1.5))
    np.testing.assert_allclose(ev.proba[inside, 2], p, rtol=1e-6)
    np.testing.assert_allclose(ev.proba[inside, 0], ev.proba[inside, 1])
    assert np.isnan(ev.proba[~inside]).all()


def test_kickoffs_other_codes_and_untyped_tags_are_ignored(small_bundle):
    tags = [
        _tag("SET PIECES", 100, 150, "Kick Off (Start)"),
        _tag("SET PIECES", 200, 250, "Kick Off (After Goal)"),
        _tag("BUILD UP", 300, 350, "Throw-in"),
        _tag("SET PIECES", 400, 450, None),
    ]
    assert not tag_windows(_with_tags(small_bundle, tags), ("throw-in",)).any()
    assert np.isnan(tag_restart(_with_tags(small_bundle, tags)).proba).all()


def test_windows_are_clipped_to_the_match(small_bundle):
    n = small_bundle.meta.n_frames
    tags = [_tag("SET PIECES", -10, 5, "Goal Kick"), _tag("SET PIECES", n - 3, n + 50, "Free Kick")]
    mask = tag_windows(_with_tags(small_bundle, tags), ("goal kick", "free kick"))
    assert mask[:5].all() and mask[n - 3 :].all() and mask.sum() == 8


def test_factorised_adds_llr_inside_and_nothing_outside(small_bundle):
    n = small_bundle.meta.n_frames
    rng = np.random.default_rng(0)
    p_dead = rng.uniform(0.05, 0.95, n)
    p_home = rng.uniform(0.05, 0.95, n)
    net = Evidence.from_factors("learned_team", p_dead, p_home)
    tags = [_tag("SET PIECES", 100, 150, "Free Kick")]
    tag = tag_restart(_with_tags(small_bundle, tags), llr=1.2)
    meta = small_bundle.meta
    base = factorised([net], meta, dead_from=["learned_team"], team_from=["learned_team"])
    both = factorised(
        [net, tag], meta, dead_from=["learned_team", "tag_restart"], team_from=["learned_team"]
    )
    outside = np.ones(n, dtype=bool)
    outside[100:150] = False
    np.testing.assert_array_equal(both[outside], base[outside])

    def logit(p):
        return np.log(p) - np.log1p(-p)

    shift = logit(both[100:150, 2].astype(np.float64)) - logit(base[100:150, 2].astype(np.float64))
    np.testing.assert_allclose(shift, 1.2, atol=1e-3)
    alive_share = both[100:150, 0] / (both[100:150, 0] + both[100:150, 1])
    base_share = base[100:150, 0] / (base[100:150, 0] + base[100:150, 1])
    np.testing.assert_allclose(alive_share, base_share, rtol=1e-4)
