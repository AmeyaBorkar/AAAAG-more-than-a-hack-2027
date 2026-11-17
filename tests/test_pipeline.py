import numpy as np

from aaaag import experiments, pipeline
from aaaag.contracts import X
from aaaag.paths import CONFIGS_DIR
from aaaag.pipeline import PipelineConfig
from aaaag.submission import to_csv_bytes

PER_PERIOD = CONFIGS_DIR / "pipelines" / "nearest_vote_per_period.yaml"


def test_every_shipped_pipeline_config_loads():
    for path in sorted((CONFIGS_DIR / "pipelines").glob("*.yaml")):
        config = PipelineConfig.load(path)
        assert len(config.digest()) == 64


def test_digest_ignores_name_but_not_params():
    base = PipelineConfig.load(PER_PERIOD).to_dict()
    renamed = dict(base, name="other", description="other")
    changed = dict(base, decoder={"name": "majority_vote", "params": {"half_window": 51}})
    digest = PipelineConfig.from_dict(base).digest()
    assert PipelineConfig.from_dict(renamed).digest() == digest
    assert PipelineConfig.from_dict(changed).digest() != digest


def test_pipeline_is_deterministic(small_bundle):
    config = PipelineConfig.load(PER_PERIOD)
    first = pipeline.run(config, small_bundle)
    second = pipeline.run(config, small_bundle)
    assert to_csv_bytes(first.labels) == to_csv_bytes(second.labels)
    assert (first.labels[~small_bundle.meta.played_mask()] == X).all()


def test_smoothing_beats_raw_on_synthetic_structure(small_bundle):
    from aaaag.evaluation.score import evaluate

    raw = PipelineConfig.load(CONFIGS_DIR / "pipelines" / "official_baseline_raw.yaml")
    smooth = PipelineConfig.load(PER_PERIOD)
    score_raw = evaluate(small_bundle.labels, pipeline.run(raw, small_bundle).labels)[
        "spell_f1_mean"
    ]
    score_smooth = evaluate(small_bundle.labels, pipeline.run(smooth, small_bundle).labels)[
        "spell_f1_mean"
    ]
    assert score_smooth > score_raw
    assert score_smooth > 0.05


def test_suite_evaluation_records_and_ranks(tmp_path):
    suite = experiments.Suite(
        "tiny", "", (pipeline.Step("synthetic", {"seed": 0, "minutes_per_half": 2.0}),)
    )
    config = PipelineConfig.load(PER_PERIOD)
    results = experiments.evaluate_suite(config, suite)
    assert results["n_matches"] == 1 and 0.0 <= results["mean"]["spell_f1_mean"] <= 1.0
    path = experiments.write_record(config, suite, results, member="tester", directory=tmp_path)
    rows = experiments.leaderboard(directory=tmp_path)
    assert path.is_file() and rows[0]["pipeline"] == config.name
    assert np.isclose(rows[0]["mean"], results["mean"]["spell_f1_mean"])
