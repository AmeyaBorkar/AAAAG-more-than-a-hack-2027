"""The submission file and the public-window smoke test, using only public files."""

import hashlib

import numpy as np

from aaaag.contracts import A, D, H, X
from aaaag.evaluation.smoke import public_smoke
from aaaag.official import BASELINE_RESULTS, N_FRAMES, graded_periods, pm
from aaaag.submission import to_csv_bytes, validate_official, write_submission


def _labels() -> np.ndarray:
    labels = np.full(N_FRAMES, X, dtype=np.int8)
    for period in graded_periods():
        labels[period.start : period.stop] = H
        labels[period.start + 500 : period.start + 900] = A
        labels[period.start + 1200 : period.start + 1300] = D
    return labels


def test_written_file_passes_the_official_validator(tmp_path):
    path = tmp_path / "AAAAG.csv"
    digest = write_submission(_labels(), path)
    ok, report = validate_official(path)
    assert ok, report
    data = path.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    assert data.startswith(b"frame,label\n0,X\n1,X\n") and b"\r" not in data
    assert data.count(b"\n") == N_FRAMES + 1


def test_csv_bytes_are_deterministic():
    assert to_csv_bytes(_labels()) == to_csv_bytes(_labels().copy())


def _baseline(name: str) -> np.ndarray:
    return pm.label_codes(pm.read_label_csv(BASELINE_RESULTS / f"{name}.csv"))


def test_smoke_passes_every_official_baseline():
    for name in ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"):
        report = public_smoke(_baseline(name))
        assert report.passed, (name, report.checks)


def test_smoke_catches_swapped_teams_and_constant_guesses():
    labels = _baseline("baseline_smooth50")
    swapped = labels.copy()
    swapped[labels == H], swapped[labels == A] = A, H
    assert not public_smoke(swapped).passed
    assert not public_smoke(np.full(N_FRAMES, D, dtype=np.int8)).passed


def test_submit_parses_the_match_cold(monkeypatch):
    """`submit` never reads a tracking cache: the organisers' fresh clone has none (#131)."""
    import argparse

    import pytest

    from aaaag import cli
    from aaaag.registry import SOURCES

    seen = {}

    class Stop(Exception):
        pass

    def load(**kwargs):
        seen.update(kwargs)
        raise Stop

    real_get = SOURCES.get
    monkeypatch.setattr(
        SOURCES,
        "get",
        lambda name: argparse.Namespace(fn=load) if name == "bar-ath" else real_get(name),
    )
    with pytest.raises(Stop):
        cli.cmd_submit(argparse.Namespace(config="configs/submission.yaml"))
    assert seen == {"use_cache": False}
