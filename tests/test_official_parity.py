"""Our reimplementation of the official baseline must agree with the organisers' own code.

If these hold, the pipeline infrastructure (parsing, slot-team mapping, frame alignment, decoding,
CSV writing) is right, and every improvement we measure is an improvement of the method.
"""

import numpy as np
import pytest

from aaaag import pipeline
from aaaag.contracts import LABELS, D, Evidence
from aaaag.decoders.majority_vote import majority_vote
from aaaag.evidence.nearest_player import nearest_team_codes
from aaaag.io import atd
from aaaag.official import BASELINE_RESULTS, load_official_baseline, pm
from aaaag.paths import CONFIGS_DIR
from aaaag.synthetic import SyntheticConfig, generate, write_epts

official = load_official_baseline()


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_majority_vote_equals_official_smoothing(window, small_bundle):
    rng = np.random.default_rng(window)
    codes = rng.choice([-1, 0, 1], size=small_bundle.meta.n_frames, p=[0.4, 0.3, 0.3]).astype(
        np.int8
    )
    codes[2000:2600] = -1  # a long gap, so empty windows (-> D) are exercised
    expected = official.smooth(codes, window)
    evidence = Evidence.from_codes("x", np.where(codes < 0, D, codes))
    ours = majority_vote(
        evidence.proba, small_bundle.meta, half_window=window, per_period=False, outside="keep"
    )
    assert (np.array(LABELS)[ours] == expected).all()


def test_nearest_player_equals_official_on_epts_files(tmp_path):
    bundle = generate(seed=5, config=SyntheticConfig(minutes_per_half=1.0, gap_minutes=0.2))
    meta_path, raw_path = write_epts(bundle.tracking, tmp_path)
    expected = official.nearest_player_labels(
        str(raw_path),
        official.team_of_each_track(str(meta_path)),
        bundle.meta.n_frames,
        (105.0, 68.0),
    )
    parsed = atd.load_tracking(meta_path, raw_path, bundle.meta)
    ours = nearest_team_codes(parsed)
    mismatched = int((ours != expected).sum())
    assert mismatched == 0, f"{mismatched} frames differ from the official baseline"


@pytest.mark.data
@pytest.mark.parametrize("name", ["raw", "smooth25", "smooth50", "smooth125"])
def test_official_baselines_reproduced_on_bar_ath(name):
    from aaaag.sources.bar_ath import load_bar_ath

    config = pipeline.PipelineConfig.load(
        CONFIGS_DIR / "pipelines" / f"official_baseline_{name}.yaml"
    )
    labels = pipeline.run(config, load_bar_ath()).labels
    expected = pm.label_codes(pm.read_label_csv(BASELINE_RESULTS / f"baseline_{name}.csv"))
    mismatched = int((labels != expected).sum())
    assert mismatched <= 20, f"{mismatched} frames differ from the official baseline_{name}.csv"
