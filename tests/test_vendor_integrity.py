"""The organisers' scorer is the ground truth of every number we report: it must be byte-exact."""

from aaaag import official


def test_vendored_files_are_byte_exact():
    assert official.verify_vendor() == []


def test_manifest_covers_the_scorer_and_public_truth():
    files = official.upstream_manifest()["files"]
    for required in (
        "games.py",
        "evaluation/possession_metrics.py",
        "evaluation/run_evaluation.py",
        "evaluation/validate_submission.py",
        "evaluation/gt/BAR-ATH_possession_gt_public.csv",
        "RULES.md",
    ):
        assert required in files


def test_official_constants():
    assert official.N_FRAMES == 199_319
    assert official.FPS == 25
    assert official.PITCH == (105.3, 68.0)
    periods = official.graded_periods()
    assert [(p.start, p.stop) for p in periods] == [(11185, 90652), (110808, 190198)]
    assert sum(p.length for p in periods) == 158_857
