"""The restarting side as the team opinion after a restart (#31): factorised `team_first_from`."""

import numpy as np

from aaaag import pipeline
from aaaag.contracts import A, D, Evidence, H, MatchMeta, Period
from aaaag.decoders.dead_gated_vote import dead_gated_vote
from aaaag.fusion.factorised import factorised
from aaaag.paths import CONFIGS_DIR
from aaaag.pipeline import PipelineConfig


def _meta(n: int, pad: int = 5) -> MatchMeta:
    return MatchMeta("T", "synthetic", n, 25, 105.0, 68.0, (Period(1, pad, n - pad),)).validate()


def test_team_first_replaces_the_pooled_team_only_where_it_has_an_opinion():
    team = Evidence.from_codes("nearest_player", np.array([H, H, H, -1, H, A], dtype=np.int8))
    p_dead = np.array([0.1, 0.1, 0.1, 0.1, np.nan, 0.2])
    lean = np.array([0.5, 0.25, 0.25, 0.75, np.nan, 0.5])  # the restarting side on frames 1-3
    ballstate = Evidence.from_factors("ballstate", p_dead, lean)
    meta = _meta(6, pad=0)
    pooled = factorised([team, ballstate], meta, dead_from=["ballstate"])
    first = factorised(
        [team, ballstate], meta, dead_from=["ballstate"], team_first_from=["ballstate"]
    )
    Evidence("fused", first).validate(6)
    assert np.allclose(first[:, 2][:4], pooled[:, 2][:4])  # P(dead) is untouched
    assert pooled[1, 0] > pooled[1, 1]  # summed, the 0.25 lean loses to the one-hot
    assert first[1, 1] > first[1, 0] and first[2, 1] > first[2, 0]  # first: the lean decides
    assert first[3, 0] > first[3, 1]  # and it speaks where the nearest player is silent
    assert np.allclose(first[[0, 5]], pooled[[0, 5]], equal_nan=True)  # a 0.5 tie is no opinion
    assert np.allclose(first[4], pooled[4], equal_nan=True)  # NaN is no opinion


def test_team_first_off_or_silent_reproduces_v1_byte_for_byte():
    rng = np.random.default_rng(0)
    n = 500
    codes = rng.choice(np.array([H, A, -1], dtype=np.int8), size=n)
    team = Evidence.from_codes("nearest_player", codes)
    ballstate = Evidence.from_factors("ballstate", rng.uniform(0, 1, n), np.full(n, 0.5))
    meta = _meta(n, pad=0)
    base = factorised([team, ballstate], meta, dead_from=["ballstate"])
    for first in (None, [], ["ballstate"]):
        out = factorised([team, ballstate], meta, dead_from=["ballstate"], team_first_from=first)
        assert np.array_equal(out, base, equal_nan=True)


def test_the_restarting_side_seeds_the_first_second_of_play():
    # stoppage 100-200; at the restart a home taker plays it, but an away marker is nearest the
    # ball for 1.6 s, then home has it
    n = 400
    meta = _meta(n)
    codes = np.full(n, A, dtype=np.int8)
    codes[100:200] = H  # the taker stands at the set ball (dead: no vote either way)
    codes[240:] = H
    team = Evidence.from_codes("nearest_player", codes)
    p_dead = np.full(n, 0.01)
    p_dead[100:200] = 0.99
    lean = np.full(n, 0.5)
    lean[200:225] = 0.75  # restarts.py: home restarts, for restart_window_s = 1 s
    ballstate = Evidence.from_factors("ballstate", p_dead, lean)
    params = {"half_window": 50, "min_dead_s": 2.0, "min_alive_s": 1.0}
    v1 = dead_gated_vote(factorised([team, ballstate], meta, dead_from=[1]), meta, **params)
    seeded = dead_gated_vote(
        factorised([team, ballstate], meta, dead_from=[1], team_first_from=[1]), meta, **params
    )
    assert (v1[100:200] == D).all() and (seeded[100:200] == D).all()
    assert v1[200] == A  # without the side, the marker's votes win the first frames
    assert (seeded[200:225] == H).all()  # with it, play starts in the restarting side's hands
    assert np.array_equal(seeded[:200], v1[:200]) and np.array_equal(seeded[300:], v1[300:])


def test_ballstate_v1_team_runs_and_is_deterministic(small_bundle):
    config = PipelineConfig.load(CONFIGS_DIR / "pipelines" / "ballstate_v1_team.yaml")
    first = pipeline.run(config, small_bundle)
    second = pipeline.run(config, small_bundle)
    assert np.array_equal(first.labels, second.labels)
    v1 = pipeline.run(
        PipelineConfig.load(CONFIGS_DIR / "pipelines" / "ballstate_v1.yaml"), small_bundle
    )
    assert np.array_equal(first.labels == D, v1.labels == D)  # the dead mask is v1's
