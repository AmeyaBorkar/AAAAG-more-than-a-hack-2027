"""PFF event touches on synthetic fixtures. No real match files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME

N1, N2 = 90, 60  # native 30 fps rows per period: 75 and 50 frames on the 25 fps grid


def _row(period, i, game_id=None, poss_id=None):
    return {
        "frameNum": i,
        "period": period,
        "periodElapsedTime": i / 30.0,
        "game_event_id": game_id,
        "possession_event_id": poss_id,
    }


def _tracking():
    rows = []
    for i in range(N1):
        game = 1 if i < 10 else (2 if 40 <= i < 50 else None)
        poss = 7 if 30 <= i < 40 else None
        rows.append(_row(1, i, game, poss))
    for i in range(N2):
        rows.append(_row(2, i, 3 if i < 5 else None, 8 if 15 <= i < 25 else None))
    rows.append(_row(3, 0, 9, 9))  # extra time is not kept
    return rows


def _records():
    return [
        {
            "gameEventId": 1,
            "possessionEventId": None,
            "gameEvents": {"gameEventType": "FIRSTKICKOFF", "homeTeam": True, "period": 1},
            "possessionEvents": None,
        },
        {
            "gameEventId": 2,
            "possessionEventId": 7,
            "gameEvents": {"gameEventType": "OTB", "homeTeam": True, "period": 1},
            "possessionEvents": {"possessionEventType": "PA", "passOutcomeType": "C"},
        },
        {
            "gameEventId": 3,
            "possessionEventId": 8,
            "gameEvents": {"homeTeam": False, "period": 2, "gameEventType": "OTB"},
            "possession_event_type": "SH",
        },
        {  # id never on a kept row
            "gameEventId": 3,
            "possessionEventId": 99,
            "gameEvents": {"homeTeam": False, "period": 2},
            "possessionEvents": {"possessionEventType": "CL"},
        },
        {  # extra time
            "gameEventId": 9,
            "possessionEventId": 9,
            "gameEvents": {"homeTeam": True, "period": 3},
            "possessionEvents": {"possessionEventType": "PA"},
        },
        {  # no team
            "gameEventId": 2,
            "possessionEventId": 7,
            "possessionEvents": {"possessionEventType": "PA"},
        },
    ]


def _write(root: Path, gid: int, events=True):
    base = root / "pff"
    for sub in ("Tracking Data", "Metadata", "Rosters", "Event Data"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in _tracking())
    (base / "Tracking Data" / f"{gid}.jsonl").write_text(text, encoding="utf-8")
    (base / "Metadata" / f"{gid}.json").write_text("[]", encoding="utf-8")
    (base / "Rosters" / f"{gid}.json").write_text("[]", encoding="utf-8")
    if events:
        (base / "Event Data" / f"{gid}.json").write_text(json.dumps(_records()), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "external"
    root.mkdir()
    monkeypatch.setenv("AAAAG_EXTERNAL_DIR", str(root))
    monkeypatch.setenv("AAAAG_CACHE_DIR", str(tmp_path / "cache"))
    import aaaag.sources.pff as pff
    import aaaag.sources.pff_events as ev

    return pff, ev, root


def _rows(events):
    return [(int(e["frame"]), int(e["team"]), str(e["kind"])) for e in events]


def test_alignment_kinds_and_drops(env):
    pff, ev, root = env
    gid = pff.game_ids("train")[0]
    _write(root, int(gid))
    out, stats = ev.touch_events_with_stats(gid)
    assert out.dtype == np.dtype([("frame", "<i8"), ("team", "<i1"), ("kind", "<U16")])
    assert _rows(out) == [
        (0, HOME, "set_piece"),  # first row of game event 1
        (25, HOME, "pass"),  # possession 7 opens at 1000 ms
        (33, HOME, "reception"),  # and closes at 1300 ms: (1300 + 20) // 40
        (75 + 13, AWAY, "shot"),  # period 2 starts after the 75 frames of period 1
    ]
    assert np.all(np.diff(out["frame"]) >= 0)
    assert stats["unmatched"] == 1
    assert stats["outside_period"] == 1
    assert stats["no_team"] == 1
    assert stats["n_frames"] == 125
    assert stats["pass:home"] == 1 and stats["shot:away"] == 1


def test_probe_aggregates_and_cache(env):
    pff, ev, root = env
    gid = pff.game_ids("train")[0]
    _write(root, int(gid))
    text = ev.probe_events([gid])
    assert "with events 1" in text
    assert "pass" in text and "share within 2 frames" in text
    # 4 distinct frames, each +-2 frames: the set piece at frame 0 is clipped to 3
    assert ev.share_near(ev.touch_events(gid), 125) == 3 + 5 + 5 + 5
    again = ev.touch_events(gid)  # second call is served from the cache
    assert _rows(again)[1] == (25, HOME, "pass")


def test_refuses_the_sealed_test_split(env):
    pff, ev, root = env
    test_gid = pff.game_ids("test")[0]
    _write(root, int(test_gid))
    with pytest.raises(ValueError, match="sealed test split"):
        ev.touch_events(test_gid)
    assert len(ev.touch_events(test_gid, allow_test=True)) == 4


def test_missing_events_and_unknown_id(env):
    pff, ev, root = env
    gid = pff.game_ids("train")[0]
    _write(root, int(gid), events=False)
    with pytest.raises(FileNotFoundError, match="Event Data"):
        ev.touch_events(gid)
    with pytest.raises(ValueError, match="not a PFF game id"):
        ev.touch_events("1")
    assert "failed or no events 1" in ev.probe_events([gid])


def test_flat_naming_and_foul():
    import aaaag.sources.pff_events as ev

    scan = {
        "span": {1: (0, 2000)},
        "first": {("game", "5"): (1, 400), ("poss", "5"): (1, 400)},
        "last": {("poss", "5"): (1, 600)},
    }
    rec = {
        "game_event_id": 5,
        "possession_event_id": 5,
        "game_event_type": "OUT",
        "home_team": "false",
        "possession_event_type": "CH",
        "fouls": [{"foul_type": "X"}],
    }
    out, _ = ev.build_touch_events([rec], scan)
    assert _rows(out) == [
        (10, AWAY, "foul"),
        (10, AWAY, "out"),
        (10, AWAY, "tackle"),
    ]
