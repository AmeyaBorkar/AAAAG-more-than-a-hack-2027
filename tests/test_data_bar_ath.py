"""Day-one checks on the real bundle: our reader must reproduce the organisers' published counts.

Run with `pytest -m data` once the data is in ./data (or $MTAH_DATA_DIR). Every assertion compares
an aggregate against a number published in DATA.md -- nothing here prints or stores frame-level data.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def bundle():
    from aaaag.sources.bar_ath import load_bar_ath

    return load_bar_ath()


def test_layout_matches_the_published_description():
    from aaaag.io import atd
    from aaaag.sources.bar_ath import asset_paths

    facts = atd.describe_layout(asset_paths()["atd_meta"])
    assert facts["n_slots"] == 46  # DATA.md: 46 track slots for 22 players


def test_ball_presence_matches_data_md(bundle):
    seen = bundle.tracking.ball_seen()
    played = bundle.meta.played_mask()
    assert int(seen.sum()) == 113_488  # DATA.md: 56.9% of all frames
    assert int(seen[played].sum()) == 94_214  # DATA.md: 59.3% of graded frames


def test_tracked_objects_match_data_md(bundle):
    played = bundle.meta.played_mask()
    objects = (~np.isnan(bundle.tracking.player_xy[played, :, 0])).sum(axis=1)
    assert abs(objects.mean() - 20.5) < 0.1  # DATA.md: mean over graded frames is 20.5


def test_smart_tagging_matches_data_md(bundle):
    tags = bundle.tags
    assert tags is not None and len(tags) == 638
    assert sum(tag.team is None for tag in tags) == 36  # the N/A instances
    assert sum(tag.code == "SET PIECES" for tag in tags) == 59
