"""PFF source on synthetic fixtures. No real match files."""

from __future__ import annotations

import bz2
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from aaaag.contracts import A, D, H, X, validate_labels
from aaaag.registry import SOURCES

ROOT_NAME = "pff"


def _player(jersey, x, y, visibility="VISIBLE", confidence="HIGH"):
    return {
        "jerseyNum": jersey,
        "confidence": confidence,
        "visibility": visibility,
        "x": x,
        "y": y,
    }


def _ball(x, y, visibility="VISIBLE", z=0.0):
    return {"visibility": visibility, "x": x, "y": y, "z": z}


def _event(kind, home_ball=None, start_frame=None, end_frame=None, home_team=None):
    return {
        "game_id": 1,
        "game_event_type": kind,
        "player_id": None,
        "team_id": None,
        "start_time": None,
        "end_time": None,
        "duration": None,
        "home_team": home_team,
        "sequence": None,
        "home_ball": home_ball,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def _frame(
    index,
    seconds,
    *,
    period=1,
    home=None,
    away=None,
    home_s=None,
    away_s=None,
    balls=None,
    ball_s=None,
    event=None,
    event_id=None,
    omit=(),
):
    if home is None:
        home = [_player(9, 0.0, 0.0)]
    row = {
        "version": 2,
        "gameRefId": 1,
        "generatedTime": "2022-01-01T00:00:00Z",
        "smoothedTime": seconds,
        "videoTimeMs": int(round(float(seconds) * 1000)),
        "frameNum": index,
        "period": period,
        "periodElapsedTime": seconds,
        "periodGameClockTime": seconds,
        "homePlayers": home,
        "awayPlayers": away or [],
        "homePlayersSmoothed": home if home_s is None else home_s,
        "awayPlayersSmoothed": away or [] if away_s is None else away_s,
        "balls": [] if balls is None else balls,
        "ballsSmoothed": ball_s,
        "game_event_id": event_id,
        "possession_event_id": None,
        "game_event": event,
        "possession_event": None,
    }
    for key in omit:
        row.pop(key, None)
    return row


def _meta(gid="1", **overrides):
    meta = {
        "id": int(gid) if str(gid).isdigit() else gid,
        "homeTeam": {"id": 1, "name": "SENTINELFC"},
        "awayTeam": {"id": 2, "name": "Away FC"},
        "homeTeamStartLeft": True,
        "startPeriod1": 0,
        "endPeriod1": 45 * 60,
        "startPeriod2": 45 * 60,
        "endPeriod2": 90 * 60,
        "period1": 45,
        "period2": 45,
        "stadium": {"pitches": [{"length": 105, "width": 68}]},
        "videos": [{"fps": 29.97}],
    }
    meta.update(overrides)
    return [meta]


def _roster(*extra):
    rows = [
        {
            "team": {"id": 1},
            "player": {"id": 10, "nickname": "SECRET_PLAYER"},
            "shirtNumber": 9,
            "positionGroupType": "F",
        },
        {
            "team": {"id": 2},
            "player": {"id": 20, "nickname": "OTHER"},
            "shirtNumber": 4,
            "positionGroupType": "M",
        },
    ]
    rows.extend(extra)
    return rows


def _write(
    root: Path, gid, frames, *, meta=None, roster=None, events=None, bz2=False, also_jsonl=False
):
    base = root / ROOT_NAME
    tracking = base / "Tracking Data"
    tracking.mkdir(parents=True, exist_ok=True)
    (base / "Metadata").mkdir(parents=True, exist_ok=True)
    (base / "Rosters").mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in frames)
    jsonl = tracking / f"{gid}.jsonl"
    if bz2:
        with bz2.open(tracking / f"{gid}.jsonl.bz2", "wt", encoding="utf-8") as handle:
            handle.write(text)
        if also_jsonl:
            jsonl.write_text(text, encoding="utf-8")
    else:
        jsonl.write_text(text, encoding="utf-8")
    (base / "Metadata" / f"{gid}.json").write_text(
        json.dumps(meta if meta is not None else _meta(gid)), encoding="utf-8"
    )
    (base / "Rosters" / f"{gid}.json").write_text(
        json.dumps(roster if roster is not None else _roster()), encoding="utf-8"
    )
    if events is not None:
        (base / "Event Data").mkdir(parents=True, exist_ok=True)
        (base / "Event Data" / f"{gid}.json").write_text(json.dumps(events), encoding="utf-8")
    return base


@pytest.fixture
def pff_env(tmp_path, monkeypatch):
    root = tmp_path / "external"
    root.mkdir()
    monkeypatch.setenv("AAAAG_EXTERNAL_DIR", str(root))
    monkeypatch.setenv("AAAAG_CACHE_DIR", str(tmp_path / "cache"))
    import aaaag.sources.pff as pff

    return pff, root


def _load(pff, gid, **kwargs):
    return pff.load_pff(match_id=str(gid), **kwargs)


def test_split_rule_on_a_synthetic_id_list():
    import aaaag.sources.pff as pff

    ids = [str(i) for i in range(64, 0, -1)]
    ordered = [str(i) for i in range(1, 65)]
    test = pff.game_ids("test", ids=ids)
    train = pff.game_ids("train", ids=ids)
    assert test == ordered[3::4]
    assert len(test) == 16
    assert train == [gid for index, gid in enumerate(ordered) if index % 4 != 3]
    assert set(test).isdisjoint(train)
    assert pff.game_ids("all", ids=ids) == ordered
    assert "16 of 64" in pff.SPLIT_RULE
    with pytest.raises(ValueError, match="split"):
        pff.game_ids("dev", ids=ids)


def test_source_is_registered():
    component = SOURCES.get("pff")
    assert component.name == "pff"
    assert component.version == "2"
    assert component.example == {}


def test_degrade_and_bad_variant_raise():
    import aaaag.sources.pff as pff

    with pytest.raises(ValueError, match="degrade"):
        pff.load_pff(match_id="1", degrade={"version": "1"})
    with pytest.raises(ValueError, match="variant"):
        pff.load_pff(match_id="1", variant="cleaned")


def test_2997_fps_lands_on_the_25_fps_grid(pff_env):
    pff, root = pff_env
    xs = [0.0, 10.0, 20.0, 30.0]
    frames = [
        _frame(
            i,
            i / 29.97,
            home=[_player(9, xs[i], 1.0)],
            balls=[_ball(xs[i], -2.0)],
            ball_s={"x": xs[i], "y": -2.0, "z": 0.0},
        )
        for i in range(4)
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.meta.fps == 25
    assert bundle.meta.n_frames == 3
    assert bundle.meta.source == "pff"
    ball = bundle.tracking.ball_xy[:, 0]
    assert ball[0] == pytest.approx(0.0)
    assert ball[1] == pytest.approx(12.058823529411764, abs=1e-5)
    assert ball[2] == pytest.approx(23.939393939393938, abs=1e-5)
    # y is not rescaled or flipped: the sample y is -2 at every native frame.
    assert bundle.tracking.ball_xy[0, 1] == pytest.approx(-2.0)
    bundle.validate()
    validate_labels(bundle.labels, bundle.meta.n_frames)


def test_no_interpolation_across_a_long_gap_or_a_period(pff_env):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, period=1, home=[_player(9, 0.0, 0.0)], balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.04, period=1, home=[_player(9, 10.0, 0.0)], balls=[_ball(10.0, 0.0)]),
        _frame(2, 0.06, period=2, home=[_player(9, 100.0, 0.0)], balls=[_ball(100.0, 0.0)]),
        _frame(3, 0.10, period=2, home=[_player(9, 110.0, 0.0)], balls=[_ball(110.0, 0.0)]),
        _frame(4, 0.0, period=None, home=[_player(9, 4242.5, 0.0)], balls=[_ball(4242.5, 0.0)]),
        _frame(5, 0.0, period=4, home=[_player(9, 4242.5, 1.0)], balls=[_ball(4242.5, 1.0)]),
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert [(p.index, p.start, p.stop) for p in bundle.meta.periods] == [(1, 0, 2), (2, 2, 4)]
    assert bundle.tracking.ball_xy[:, 0].tolist() == pytest.approx([0.0, 10.0, 100.0, 110.0])
    assert not np.any(np.isclose(bundle.tracking.ball_xy[:, 0], 4242.5, equal_nan=False))
    assert bundle.extras["pff"]["native_frames"] == 6
    assert bundle.extras["pff"]["kept_native"] == {"1": 2, "2": 2}

    _write(
        root,
        "3813",
        [
            _frame(0, 0.0, home=[_player(9, 0.0, 0.0)], balls=[_ball(0.0, 0.0)]),
            _frame(1, 0.08, home=[_player(9, 8.0, 0.0)], balls=[_ball(8.0, 0.0)]),
        ],
    )
    gap = _load(pff, "3813", use_cache=False)
    assert gap.meta.n_frames == 3
    assert np.isnan(gap.tracking.ball_xy[1]).all()
    assert gap.tracking.ball_xy[0, 0] == pytest.approx(0.0)
    assert gap.tracking.ball_xy[2, 0] == pytest.approx(8.0)

    _write(
        root,
        "3814",
        [
            _frame(0, 0.0, home=[_player(9, 0.0, 0.0)], balls=[_ball(0.0, 0.0)]),
            _frame(1, 0.05, home=[_player(9, 10.0, 0.0)], balls=[_ball(10.0, 0.0)]),
        ],
    )
    bridged = _load(pff, "3814", use_cache=False)
    assert bridged.tracking.ball_xy[1, 0] == pytest.approx(8.0, abs=1e-5)

    _write(
        root,
        "3816",
        [
            _frame(0, 0.0, home=[_player(9, 0.0, 0.0)], balls=[_ball(0.0, 0.0)]),
            _frame(1, 0.06, home=[_player(9, 10.0, 0.0)], balls=[_ball(10.0, 0.0)]),
        ],
    )
    too_far = _load(pff, "3816", use_cache=False)
    assert np.isnan(too_far.tracking.ball_xy[1]).all()


def test_missing_sample_blocks_interpolation(pff_env):
    pff, root = pff_env
    frames = []
    for i in range(3):
        balls = [] if i == 1 else [_ball(float(i), 0.0)]
        frames.append(
            _frame(i, i / 29.97, home=[_player(9, float(i), 0.0)], balls=balls, ball_s=None)
        )
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.meta.n_frames == 2
    assert bundle.tracking.ball_xy[0, 0] == pytest.approx(0.0)
    assert np.isnan(bundle.tracking.ball_xy[1]).all()
    assert np.isfinite(bundle.tracking.player_xy[1, 0, 0])


def test_period_starts_at_its_first_frame(pff_env):
    pff, root = pff_env
    _write(
        root,
        "3812",
        [
            _frame(0, 10.0, home=[_player(9, 1.0, 2.0)], balls=[_ball(1.0, 2.0)]),
            _frame(1, 10.04, home=[_player(9, 3.0, 4.0)], balls=[_ball(3.0, 4.0)]),
        ],
    )
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.meta.n_frames == 2
    assert bundle.tracking.ball_xy[:, 0].tolist() == pytest.approx([1.0, 3.0])


def test_axes_are_not_rescaled(pff_env):
    pff, root = pff_env
    _write(
        root,
        "3812",
        [_frame(0, 0.0, home=[_player(9, -12.5, 33.25)], balls=[_ball(60.0, -40.0)])],
        meta=_meta("1", stadium={"pitches": [{"length": 100, "width": 64}]}),
    )
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.meta.pitch_length == 100
    assert bundle.meta.pitch_width == 64
    assert bundle.tracking.player_xy[0, 0].tolist() == pytest.approx([-12.5, 33.25])
    assert bundle.tracking.ball_xy[0].tolist() == pytest.approx([60.0, -40.0])
    assert bundle.extras["pff"]["pitch_path"] == "stadium.pitches[0]"
    assert bundle.extras["pff"]["fps_path"] == "videos[0].fps"


def test_pitch_and_fps_fallbacks(pff_env):
    pff, root = pff_env
    meta = _meta("1")
    meta[0].pop("stadium")
    meta[0].pop("videos")
    frames = [_frame(i, i / 29.97, balls=[_ball(0.0, 0.0)]) for i in range(4)]
    _write(root, "3812", frames, meta=meta)
    bundle = _load(pff, "3812", use_cache=False)
    stats = bundle.extras["pff"]
    assert bundle.meta.pitch_length == 105
    assert bundle.meta.pitch_width == 68
    assert stats["pitch_path"] == "default"
    assert stats["fps_path"] == "measured"
    assert stats["measured_fps"] == pytest.approx(29.97, abs=1e-6)

    meta = _meta("2", fps=30)
    _write(root, "3813", [_frame(0, 0.0, balls=[_ball(0.0, 0.0)])], meta=meta)
    assert _load(pff, "3813", use_cache=False).extras["pff"]["fps_path"] == "fps"


@pytest.mark.parametrize(
    "kind", ["FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"]
)
def test_kickoff_and_otb_set_alive(pff_env, kind):
    pff, root = pff_env
    _write(
        root, "3812", [_frame(0, 0.0, event=_event(kind, home_ball=True), balls=[_ball(0.0, 0.0)])]
    )
    assert _load(pff, "3812", use_cache=False).labels.tolist() == [H]


@pytest.mark.parametrize("kind", ["OUT", "END"])
def test_out_and_end_set_dead_and_owner_persists(pff_env, kind):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, event=_event("FIRSTKICKOFF", home_ball=True), balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.04, event=_event(kind), balls=[_ball(0.0, 0.0)]),
        _frame(2, 0.08, event=_event("OTB"), balls=[_ball(0.0, 0.0)]),
    ]
    _write(root, "3812", frames)
    assert _load(pff, "3812", use_cache=False).labels.tolist() == [H, D, H]


def test_owner_can_change_while_dead_and_period_resets(pff_env):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, event=_event("OUT", home_ball=False), balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.04, event=_event("OTB"), balls=[_ball(0.0, 0.0)]),
        _frame(2, 0.0, period=2, balls=[_ball(0.0, 0.0)]),
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.labels.tolist() == [D, A, D]
    assert bundle.extras["pff"]["possessions"] == 1


def test_alive_without_an_owner_is_x(pff_env):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, event=_event("OTB"), balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.04, balls=[_ball(0.0, 0.0)]),
    ]
    _write(root, "3812", frames)
    assert _load(pff, "3812", use_cache=False).labels.tolist() == [X, X]


def test_ignored_events_do_not_change_state(pff_env):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, event=_event("FIRSTKICKOFF", home_ball=True), balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.04, event=_event("G"), balls=[_ball(0.0, 0.0)]),
        _frame(2, 0.08, event=_event("OFF"), balls=[_ball(0.0, 0.0)]),
        _frame(3, 0.12, event=_event("ON"), balls=[_ball(0.0, 0.0)]),
        _frame(4, 0.16, event=_event("SUB", home_ball=False), balls=[_ball(0.0, 0.0)]),
    ]
    _write(root, "3812", frames)
    assert _load(pff, "3812", use_cache=False).labels.tolist() == [H, H, H, H, A]


def test_label_tie_uses_the_earlier_sample(pff_env):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, event=_event("FIRSTKICKOFF", home_ball=True), balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.08, event=_event("OTB", home_ball=False), balls=[_ball(8.0, 0.0)]),
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.labels.tolist() == [H, H, A]
    assert np.isnan(bundle.tracking.ball_xy[1]).all()


def test_vid_frames_are_x(pff_env):
    pff, root = pff_env
    frames = [
        _frame(0, 0.0, event=_event("FIRSTKICKOFF", home_ball=True), balls=[_ball(0.0, 0.0)]),
        _frame(1, 0.04, event=_event("VID", start_frame=1, end_frame=2), balls=[_ball(0.0, 0.0)]),
        _frame(2, 0.08, balls=[_ball(0.0, 0.0)]),
        _frame(3, 0.12, balls=[_ball(0.0, 0.0)]),
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.labels.tolist() == [H, X, X, H]
    assert bundle.extras["pff"]["vid_events"] == 1
    assert bundle.extras["pff"]["vid_frames"] == 2


def test_setpiece_backfill_comes_from_the_event_file(pff_env):
    pff, root = pff_env
    frames = [
        _frame(
            0,
            0.0,
            event=_event("FIRSTKICKOFF", home_ball=True),
            event_id=1,
            balls=[_ball(0.0, 0.0)],
        ),
        _frame(1, 0.5, balls=[_ball(1.0, 0.0)]),
        _frame(2, 1.0, event=_event("OTB", home_ball=True), event_id=10, balls=[_ball(2.0, 0.0)]),
    ]
    events = [{"id": 10, "details": {"setPieceType": "CORNER"}}]
    _write(root, "3812", frames, events=events)
    bundle = _load(pff, "3812", use_cache=False)
    labels = bundle.labels
    assert bundle.extras["pff"]["setpiece_rule"] is True
    assert bundle.extras["pff"]["setpiece_field"] == "details.setPieceType"
    assert bundle.extras["pff"]["setpiece_counts"] == {"CORNER": 1}
    assert labels[0] == H
    assert labels[6] == H
    assert labels[7] == D
    assert labels[12] == D
    assert labels[19] == H
    assert labels[25] == H

    _write(root, "3813", frames, events=[{"id": 10, "details": {"setPieceType": "O"}}])
    assert np.all(_load(pff, "3813", use_cache=False).labels == H)

    short = [
        _frame(
            0, 0.0, event=_event("FIRSTKICKOFF", home_ball=True), event_id=1, balls=[_ball(0, 0)]
        ),
        _frame(1, 0.4, balls=[_ball(0, 0)]),
        _frame(2, 0.8, event=_event("OTB", home_ball=True), event_id=10, balls=[_ball(0, 0)]),
    ]
    _write(root, "3814", short, events=events)
    assert np.all(_load(pff, "3814", use_cache=False).labels == H)


def test_no_setpiece_rule_without_a_field(pff_env):
    pff, root = pff_env
    frames = [
        _frame(
            0, 0.0, event=_event("FIRSTKICKOFF", home_ball=True), event_id=1, balls=[_ball(0, 0)]
        ),
        _frame(1, 0.5, balls=[_ball(0, 0)]),
        _frame(2, 1.0, event=_event("OTB", home_ball=True), event_id=10, balls=[_ball(0, 0)]),
    ]
    _write(root, "3812", frames)
    bare = _load(pff, "3812", use_cache=False)
    assert bare.extras["pff"]["setpiece_rule"] is False
    assert np.all(bare.labels == H)

    _write(root, "3813", frames, events=[{"id": 10, "kind": "CORNER"}])
    skipped = _load(pff, "3813", use_cache=False)
    assert skipped.extras["pff"]["setpiece_rule"] is False
    assert np.all(skipped.labels == H)


def test_raw_ball_visibility_and_smoothed_shapes(pff_env):
    pff, root = pff_env
    frames = [
        _frame(
            0,
            0.0,
            home=[_player(9, 1.0, 2.0)],
            home_s=[_player(9, 8.0, 9.0)],
            balls=[_ball(1.0, 2.0, "VISIBLE"), _ball(9.0, 9.0, "ESTIMATED")],
            ball_s={"x": 3.0, "y": 4.0, "z": 1.0},
        ),
        _frame(
            1,
            0.04,
            balls=[_ball(5.0, 6.0, "ESTIMATED")],
            ball_s=[{"x": 7.0, "y": 8.0, "z": 0.0}],
        ),
        _frame(2, 0.08, balls=[_ball(1.0, 1.0, "VISIBLE"), _ball(2.0, 2.0, "VISIBLE")], ball_s=[]),
        _frame(3, 0.12, balls=[], ball_s=None),
    ]
    _write(root, "3812", frames)
    raw = _load(pff, "3812", variant="raw", use_cache=False)
    assert raw.tracking.ball_xy[:, 0].tolist()[:1] == pytest.approx([1.0])
    assert np.isnan(raw.tracking.ball_xy[1:]).all()
    assert raw.tracking.player_xy[0, 0].tolist() == pytest.approx([1.0, 2.0])
    counts = raw.extras["pff"]["ball_raw_counts"]
    assert counts == {"visible": 2, "estimated": 1, "missing": 1}

    smoothed = _load(pff, "3812", variant="smoothed", use_cache=False)
    assert smoothed.tracking.ball_xy[0].tolist() == pytest.approx([3.0, 4.0])
    assert smoothed.tracking.ball_xy[1].tolist() == pytest.approx([7.0, 8.0])
    assert np.isnan(smoothed.tracking.ball_xy[2:]).all()
    assert smoothed.tracking.player_xy[0, 0].tolist() == pytest.approx([8.0, 9.0])


def test_missing_variant_keys_do_not_fall_back(pff_env):
    pff, root = pff_env
    raw_only = _frame(0, 0.0, balls=[_ball(1.0, 2.0)], ball_s={"x": 8.0, "y": 9.0, "z": 0.0})
    for key in ("homePlayersSmoothed", "awayPlayersSmoothed", "ballsSmoothed"):
        raw_only.pop(key)
    _write(root, "3812", [raw_only])
    assert _load(pff, "3812", variant="raw", use_cache=False).tracking.ball_xy[
        0, 0
    ] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not falling back"):
        _load(pff, "3812", variant="smoothed", use_cache=False)

    smooth_only = _frame(0, 0.0, balls=[_ball(1.0, 2.0)], ball_s={"x": 8.0, "y": 9.0, "z": 0.0})
    for key in ("homePlayers", "awayPlayers", "balls"):
        smooth_only.pop(key)
    _write(root, "3813", [smooth_only])
    assert _load(pff, "3813", variant="smoothed", use_cache=False).tracking.ball_xy[
        0, 0
    ] == pytest.approx(8.0)
    with pytest.raises(ValueError, match="homePlayers"):
        _load(pff, "3813", variant="raw", use_cache=False)


def test_unknown_jerseys_are_dropped(pff_env):
    pff, root = pff_env
    frames = [
        _frame(
            0,
            0.0,
            home=[_player(9, 1.0, 2.0), _player(99, 50.0, 50.0)],
            away=[_player(4, 3.0, 4.0)],
            balls=[_ball(0.0, 0.0)],
        )
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    assert bundle.tracking.slot_team.tolist() == [0, 1]
    assert bundle.tracking.player_xy[0, 0].tolist() == pytest.approx([1.0, 2.0])
    assert bundle.tracking.player_xy[0, 1].tolist() == pytest.approx([3.0, 4.0])
    assert not np.any(np.isclose(bundle.tracking.player_xy, 50.0, equal_nan=False))
    assert bundle.extras["pff"]["unknown_jerseys_raw"] == 1


def test_duplicate_jerseys_prefer_visible_then_confidence_then_first(pff_env):
    pff, root = pff_env
    frames = [
        _frame(
            0,
            0.0,
            home=[
                _player(9, 1.0, 2.0, "ESTIMATED", "HIGH"),
                _player(9, 3.0, 4.0, "VISIBLE", "LOW"),
            ],
            home_s=[_player(9, 0.0, 0.0)],
            balls=[_ball(0.0, 0.0)],
        ),
        _frame(
            1,
            0.04,
            home=[
                _player(9, 5.0, 6.0, "ESTIMATED", "MEDIUM"),
                _player(9, 7.0, 8.0, "ESTIMATED", "HIGH"),
            ],
            home_s=[_player(9, 0.0, 0.0)],
            balls=[_ball(0.0, 0.0)],
        ),
        _frame(
            2,
            0.08,
            home=[
                _player(9, 9.0, 10.0, "VISIBLE", "HIGH"),
                _player(9, 11.0, 12.0, "VISIBLE", "HIGH"),
            ],
            home_s=[_player(9, 0.0, 0.0)],
            balls=[_ball(0.0, 0.0)],
        ),
    ]
    _write(root, "3812", frames)
    bundle = _load(pff, "3812", use_cache=False)
    got = bundle.tracking.player_xy[:, 0, :]
    assert got[0].tolist() == pytest.approx([3.0, 4.0])
    assert got[1].tolist() == pytest.approx([7.0, 8.0])
    assert got[2].tolist() == pytest.approx([9.0, 10.0])
    stats = bundle.extras["pff"]
    assert stats["duplicate_players_raw"] == 3
    assert stats["duplicate_players_smoothed"] == 0
    assert stats["player_estimated_raw"] == pytest.approx(0.5)
    assert stats["player_estimated_smoothed"] == pytest.approx(0.0)


def test_cache_round_trip(pff_env, monkeypatch):
    pff, root = pff_env
    _write(
        root, "3812", [_frame(0, 0.0, balls=[_ball(1.0, 2.0)], ball_s={"x": 3.0, "y": 4.0, "z": 0})]
    )
    first = _load(pff, "3812", variant="raw")
    smoothed = _load(pff, "3812", variant="smoothed")
    assert first.tracking.ball_xy[0, 0] == pytest.approx(1.0)
    assert smoothed.tracking.ball_xy[0, 0] == pytest.approx(3.0)
    cache = pff._cache_path("3812", "raw")
    assert cache.is_file() and cache.with_suffix(".json").is_file()

    def boom(*_args, **_kwargs):
        raise AssertionError("cache hit must not parse the tracking file again")

    monkeypatch.setattr(pff, "_parse_tracking", boom)
    second = _load(pff, "3812", variant="raw")
    again = _load(pff, "3812", variant="smoothed")
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.tracking.ball_xy, second.tracking.ball_xy, equal_nan=True)
    assert np.array_equal(first.tracking.player_xy, second.tracking.player_xy, equal_nan=True)
    assert first.extras["pff"]["possessions"] == second.extras["pff"]["possessions"]
    assert again.tracking.ball_xy[0, 0] == pytest.approx(3.0)


def test_cache_stamp_changes_with_the_file(pff_env):
    pff, root = pff_env
    _write(root, "3812", [_frame(0, 0.0, balls=[_ball(1.0, 2.0)])])
    first = _load(pff, "3812")
    _write(root, "3812", [_frame(0, 0.0, home=[_player(9, 6.0, 7.0)], balls=[_ball(6.0, 7.0)])])
    second = _load(pff, "3812")
    assert first.tracking.ball_xy[0, 0] == pytest.approx(1.0)
    assert second.tracking.ball_xy[0, 0] == pytest.approx(6.0)


def test_bz2_is_preferred_over_plain_jsonl(pff_env):
    pff, root = pff_env
    plain = [_frame(0, 0.0, home=[_player(9, 1.0, 0.0)], balls=[_ball(1.0, 0.0)])]
    _write(root, "3816", plain, also_jsonl=False)
    # Rewrite the bz2 with a different coordinate and leave the jsonl in place.
    text = json.dumps(_frame(0, 0.0, home=[_player(9, 2.0, 0.0)], balls=[_ball(2.0, 0.0)])) + "\n"
    tracking = root / ROOT_NAME / "Tracking Data"
    (tracking / "3816.jsonl").write_text(
        json.dumps(_frame(0, 0.0, home=[_player(9, 1.0, 0.0)], balls=[_ball(1.0, 0.0)])) + "\n",
        encoding="utf-8",
    )
    with bz2.open(tracking / "3816.jsonl.bz2", "wt", encoding="utf-8") as handle:
        handle.write(text)
    bundle = _load(pff, "3816", use_cache=False)
    assert bundle.tracking.ball_xy[0, 0] == pytest.approx(2.0)


def test_probe_and_list_print_aggregates_only(pff_env):
    pff, root = pff_env
    frames = [
        _frame(
            i,
            i / 29.97,
            event=_event("FIRSTKICKOFF", home_ball=True) if i == 0 else None,
            event_id=1 if i == 0 else None,
            balls=[_ball(0.0, 0.0)],
        )
        for i in range(4)
    ]
    frames.append(
        _frame(9, 0.0, period=3, home=[_player(9, 4242.5, 1.0)], balls=[_ball(4242.5, 1.0)])
    )
    _write(root, "3817", frames, events=[{"id": 1, "setpiece_type": "O"}])
    _write(root, "3813", [_frame(0, 0.0, balls=[_ball(0.0, 0.0)])])
    env = dict(**__import__("os").environ)
    listed = subprocess.run(
        [sys.executable, "-m", "aaaag.sources.pff", "--list"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert listed.stdout.splitlines() == ["3813", "3817"]
    probed = subprocess.run(
        [sys.executable, "-m", "aaaag.sources.pff", "--probe", "--games", "1"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    text = probed.stdout
    assert "pff 3813 " in text
    assert "pff 3817 " not in text
    assert "native_frames=" in text
    assert "measured_fps=" in text
    assert "possessions=" in text
    assert "ball_visible=" in text
    assert "tracking_keys" in text
    assert "game_event" in text
    assert "metadata_keys" in text
    assert "homeTeam" in text
    assert "roster_keys" in text
    assert "shirtNumber" in text
    assert "SECRET_PLAYER" not in text
    assert "SENTINELFC" not in text
    assert "4242.5" not in text
    assert probed.stderr == ""


def test_parse_speed_on_150k_lines(pff_env):
    pff, root = pff_env
    gid = "3812"
    base = root / ROOT_NAME
    tracking = base / "Tracking Data"
    tracking.mkdir(parents=True)
    (base / "Metadata").mkdir()
    (base / "Rosters").mkdir()
    home = ",".join(
        f'{{"jerseyNum":{j},"confidence":"HIGH","visibility":"VISIBLE","x":1,"y":2}}'
        for j in range(1, 12)
    )
    away = ",".join(
        f'{{"jerseyNum":{j},"confidence":"HIGH","visibility":"VISIBLE","x":3,"y":4}}'
        for j in range(1, 12)
    )
    roster = []
    for jersey in range(1, 12):
        roster.append(
            {
                "team": {"id": 1},
                "player": {"id": jersey, "nickname": "n"},
                "shirtNumber": jersey,
                "positionGroupType": "M",
            }
        )
        roster.append(
            {
                "team": {"id": 2},
                "player": {"id": 100 + jersey, "nickname": "n"},
                "shirtNumber": jersey,
                "positionGroupType": "M",
            }
        )
    (base / "Metadata" / f"{gid}.json").write_text(json.dumps(_meta(gid)), encoding="utf-8")
    (base / "Rosters" / f"{gid}.json").write_text(json.dumps(roster), encoding="utf-8")
    path = tracking / f"{gid}.jsonl"
    n = 150_000
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for i in range(n):
            handle.write(
                f'{{"period":1,"frameNum":{i},"periodElapsedTime":{i / 29.97:.5f},'
                f'"homePlayers":[{home}],"awayPlayers":[{away}],'
                f'"homePlayersSmoothed":[{home}],"awayPlayersSmoothed":[{away}],'
                f'"balls":[{{"visibility":"VISIBLE","x":0,"y":0,"z":0}}],'
                f'"ballsSmoothed":{{"x":0,"y":0,"z":0}},"game_event":null}}\n'
            )
    started = time.perf_counter()
    bundle = _load(pff, gid, variant="raw")
    uncached = time.perf_counter() - started
    assert bundle.meta.n_frames > 100_000
    assert bundle.tracking.player_xy.shape[1] == 22

    def boom(*_args, **_kwargs):
        raise AssertionError("cached load parsed the file")

    import aaaag.sources.pff as pff_mod

    original = pff_mod._parse_tracking
    pff_mod._parse_tracking = boom
    try:
        started = time.perf_counter()
        second = _load(pff, gid, variant="raw")
        cached = time.perf_counter() - started
    finally:
        pff_mod._parse_tracking = original
    assert np.array_equal(bundle.labels, second.labels)
    print(
        f"\nPFF_PARSE_SECONDS uncached={uncached:.2f} cached={cached:.2f} frames={bundle.meta.n_frames}"
    )
    assert uncached < 180, f"uncached load took {uncached:.1f}s"
    assert cached < 5, f"cached load took {cached:.1f}s"
