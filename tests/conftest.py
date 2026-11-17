"""Shared fixtures. CI never has the match data, so everything here runs on synthetic matches;
tests marked `data` run only where the BAR-ATH bundle is present (`pytest -m data`)."""

from __future__ import annotations

import pytest

from aaaag.sources.bar_ath import asset_paths
from aaaag.synthetic import SyntheticConfig, generate

SMALL = SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)


@pytest.fixture(scope="session")
def small_bundle():
    return generate(seed=7, config=SMALL)


def has_match_data() -> bool:
    paths = asset_paths()
    return paths["atd_meta"].is_file() and paths["atd_raw"].is_file()


def pytest_collection_modifyitems(config, items):
    if has_match_data():
        return
    skip = pytest.mark.skip(reason="BAR-ATH data not present (docs/ONBOARDING.md section 4)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
