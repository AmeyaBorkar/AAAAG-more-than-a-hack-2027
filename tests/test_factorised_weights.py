"""factorised fusion: per-producer weights on the team log-odds (#86, E13)."""

from __future__ import annotations

import numpy as np

from aaaag.contracts import Evidence, MatchMeta, Period
from aaaag.fusion.factorised import factorised

N = 400


def meta() -> MatchMeta:
    return MatchMeta(
        match_id="synthetic",
        source="synthetic",
        n_frames=N,
        fps=25,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(Period(1, 0, N),),
    )


def evidences() -> list[Evidence]:
    rng = np.random.default_rng(0)
    p_home = rng.uniform(0.05, 0.95, N).astype(np.float32)
    p_dead = rng.uniform(0.01, 0.2, N).astype(np.float32)
    net = Evidence.from_factors("net", p_dead, p_home)
    rule_team = (rng.random(N) < 0.5).astype(np.int8)
    rule_team[:30] = -1  # the rule abstains before its first control
    rule = Evidence.from_codes("rule", rule_team)
    return [net, rule]


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p) - np.log1p(-p)


def test_weight_one_is_the_old_behaviour_and_zero_removes_the_producer():
    ev = evidences()
    kw = {"dead_from": ["net"], "team_from": ["net", "rule"]}
    plain = factorised(ev, meta(), **kw)
    ones = factorised(ev, meta(), team_weights={"net": 1.0, "rule": 1.0}, **kw)
    assert np.array_equal(plain, ones, equal_nan=True)
    only_net = factorised(ev, meta(), dead_from=["net"], team_from=["net"])
    zero = factorised(ev, meta(), team_weights={"rule": 0.0}, **kw)
    assert np.allclose(zero[30:], only_net[30:], atol=1e-6, equal_nan=True)


def test_weighted_sum_matches_a_hand_computation():
    ev = evidences()
    out = factorised(
        ev, meta(), dead_from=["net"], team_from=["net", "rule"], team_weights={"rule": 0.1}
    )
    net, rule = ev[0].proba.astype(np.float64), ev[1].proba.astype(np.float64)
    z = logit(net[:, 0] / (net[:, 0] + net[:, 1]))
    has_rule = ~np.isnan(rule[:, 0])
    z = z + np.where(has_rule, 0.1 * logit(np.nan_to_num(rule[:, 0], nan=0.5)), 0.0)
    p_home = 1.0 / (1.0 + np.exp(-z))
    got = out[:, 0] / (out[:, 0] + out[:, 1])
    assert np.allclose(got, p_home, atol=1e-5)


def test_weights_accept_positions_and_unknown_names_raise():
    ev = evidences()
    by_name = factorised(ev, meta(), dead_from=[0], team_from=[0, 1], team_weights={"rule": 0.25})
    by_index = factorised(ev, meta(), dead_from=[0], team_from=[0, 1], team_weights={1: 0.25})
    assert np.array_equal(by_name, by_index, equal_nan=True)
    try:
        factorised(ev, meta(), dead_from=[0], team_from=[0, 1], team_weights={"ghost": 0.1})
    except ValueError as error:
        assert "ghost" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an unknown producer name must raise")
