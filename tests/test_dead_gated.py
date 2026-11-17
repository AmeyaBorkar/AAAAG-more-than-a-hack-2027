"""The factorised fusion and the dead-gated vote decoder (added with the ball-state lane, #ballstate)."""

import numpy as np

from aaaag import pipeline
from aaaag.contracts import A, D, Evidence, H, MatchMeta, Period, X
from aaaag.decoders.dead_gated_vote import dead_gated_vote
from aaaag.evaluation.score import evaluate
from aaaag.fusion.factorised import factorised
from aaaag.paths import CONFIGS_DIR
from aaaag.pipeline import PipelineConfig


def _meta(n: int, pad: int = 5) -> MatchMeta:
    return MatchMeta("T", "synthetic", n, 25, 105.0, 68.0, (Period(1, pad, n - pad),)).validate()


def test_factorised_takes_each_factor_from_its_own_side():
    n = 6
    team = Evidence.from_codes("nearest_player", np.array([H, A, -1, H, -1, D], dtype=np.int8))
    dead = Evidence.from_factors(
        "ballstate",
        np.array([0.9, 0.1, 0.8, np.nan, np.nan, 0.2]),
        np.full(n, 0.5),
    )
    fused = factorised([team, dead], _meta(n, pad=0), dead_from=["ballstate"])
    Evidence("fused", fused).validate(n)
    assert np.allclose(
        fused[:, 2][[0, 1, 2]], [0.9, 0.1, 0.8], atol=1e-6
    )  # the one-hot can't veto D
    assert fused[0, 0] > fused[0, 1] and fused[1, 1] > fused[1, 0]  # team from nearest_player
    assert np.isclose(fused[2, 0], fused[2, 1])  # no team opinion: a tie
    assert fused[3, 2] < 1e-6 and fused[3, 0] > 0.99  # no dead opinion: p_dead_unknown (0)
    assert np.isnan(fused[4]).all()  # neither: no opinion
    assert np.isclose(fused[5, 0], fused[5, 1])  # a one-hot D says nothing about the team


def test_gated_vote_keeps_stoppage_edges_and_does_not_vote_across_them():
    n = 310
    meta = _meta(n)
    proba = np.zeros((n, 3), dtype=np.float32)
    proba[:, 0] = 1.0  # home everywhere ...
    proba[100:160] = [0.0, 0.0, 1.0]  # ... a 2.4 s stoppage
    proba[160:170] = [0.0, 1.0, 0.0]  # away restarts, briefly seen
    proba[170:260] = [0.5, 0.5, 0.0]  # then nobody votes
    proba[50:55, 2] = 0.9  # a 0.2 s blip of D: shorter than min_dead_s, dropped
    proba[50:55, :2] = 0.05
    labels = dead_gated_vote(proba, meta, half_window=50, min_dead_s=1.0)
    assert (labels[:5] == X).all() and (labels[-5:] == X).all()
    assert (labels[5:100] == H).all()
    assert (labels[100:160] == D).all()  # exact edges, no +/-50 shrinkage
    assert (labels[160:210] == A).all()  # home votes before the stoppage don't leak in; fill
    assert labels[300] == H


def test_gated_vote_run_without_votes_takes_the_summed_margin():
    n = 120
    proba = np.zeros((n, 3), dtype=np.float32)
    proba[:, :2] = [0.45, 0.55]  # every frame leans away, but a 0.1 margin is still a vote
    labels = dead_gated_vote(proba, _meta(n), half_window=10)
    assert (labels[5:-5] == A).all()


def test_ballstate_pipeline_runs_and_finds_stoppages(small_bundle):
    config = PipelineConfig.load(CONFIGS_DIR / "pipelines" / "ballstate_v1.yaml")
    first = pipeline.run(config, small_bundle)
    second = pipeline.run(config, small_bundle)
    assert np.array_equal(first.labels, second.labels)
    baseline = PipelineConfig.load(CONFIGS_DIR / "pipelines" / "nearest_vote_per_period.yaml")
    scores = evaluate(small_bundle.labels, first.labels)
    base = evaluate(small_bundle.labels, pipeline.run(baseline, small_bundle).labels)
    assert scores["macro_f1"] > base["macro_f1"]  # plumbing check, never a selection
