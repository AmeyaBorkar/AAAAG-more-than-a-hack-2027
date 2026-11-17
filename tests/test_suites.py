"""Every suite in configs/suites/ loads, names registered sources, and keeps the IDSSE exam sealed."""

from __future__ import annotations

import pytest

from aaaag.experiments import Suite
from aaaag.paths import CONFIGS_DIR
from aaaag.registry import SOURCES
from aaaag.sources import idsse

SUITES = sorted((CONFIGS_DIR / "suites").glob("*.yaml"))


@pytest.mark.parametrize("path", SUITES, ids=[p.stem for p in SUITES])
def test_suite_loads_with_registered_sources(path):
    suite = Suite.load(path)
    for entry in suite.matches:
        SOURCES.get(entry.name)
        if entry.name == "idsse":
            assert entry.params["match_id"] in idsse.MATCHES


@pytest.mark.parametrize("path", SUITES, ids=[p.stem for p in SUITES])
def test_exam_matches_only_in_an_explicit_exam_suite(path):
    for entry in Suite.load(path).matches:
        if entry.name == "idsse" and entry.params["match_id"] in idsse.EXAM_MATCHES:
            assert entry.params.get("exam") is True, f"{path.name} lists a sealed exam match"


def test_idsse_dev_clean_is_exactly_the_dev_matches():
    suite = Suite.load(CONFIGS_DIR / "suites" / "idsse-dev-clean.yaml")
    assert [entry.params["match_id"] for entry in suite.matches] == list(idsse.DEV_MATCHES)


def test_loader_refuses_exam_matches_without_the_flag():
    for match_id in idsse.EXAM_MATCHES:
        with pytest.raises(PermissionError, match="sealed exam"):
            idsse.load_idsse(match_id=match_id)
        with pytest.raises(PermissionError, match="sealed exam"):
            idsse.load_files(match_id, "meta.xml", "positions.xml")


@pytest.mark.parametrize(
    ("name", "matches"), [("idsse-dev", idsse.DEV_MATCHES), ("idsse-exam", idsse.EXAM_MATCHES)]
)
def test_degraded_idsse_suites_cover_every_match_with_five_seeds(name, matches):
    suite = Suite.load(CONFIGS_DIR / "suites" / f"{name}.yaml")
    pairs = {(e.params["match_id"], e.params["degrade"]["seed"]) for e in suite.matches}
    assert pairs == {(m, s) for m in matches for s in range(5)}
    assert len(pairs) == len(suite.matches)
    assert all(e.params["degrade"]["version"] == "2" for e in suite.matches)
