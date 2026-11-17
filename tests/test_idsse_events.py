"""IDSSE event data on a hand-written DFL fixture: parsing, clock sync, half lag, refinement,
the elastic sync, receptions, outs, possession changes, pins, the exam gate and the cache (#86).

The fixture is synthetic: four static players on a diamond, a ball passed round it every 40
frames (each player holds it a few frames, a different number each time), and an event file whose
clock is 0.8 s early in the first half and 0.4 s late in the second. A second fixture, built as
arrays, plays two 20-minute halves with irregular kicks and an event clock that drifts by 4 s
across each half.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, A, H, MatchBundle, MatchMeta, Period, Tracking
from aaaag.sources import idsse, idsse_events

HOME_ID, AWAY_ID = "DFL-CLU-HOME", "DFL-CLU-AWAY"
PEOPLE = {
    "P-H1": (HOME_ID, (-15.0, 0.0)),
    "P-H2": (HOME_ID, (0.0, 15.0)),
    "P-A1": (AWAY_ID, (15.0, 0.0)),
    "P-A2": (AWAY_ID, (0.0, -15.0)),
}
CYCLE = ("P-H1", "P-H2", "P-A1", "P-A2")
N_HALF = 1500
STEP = 40
HOLDS = (4, 7, 10, 13, 16)  # frames the ball rests at a player before the kick, by kick index
LOCAL = timezone(timedelta(hours=2))
SECTIONS = {
    "firstHalf": (10000, datetime(2022, 10, 15, 13, 30, tzinfo=LOCAL)),
    "secondHalf": (100000, datetime(2022, 10, 15, 14, 30, tzinfo=LOCAL)),
}
OFFSET_S = {"firstHalf": -0.8, "secondHalf": 0.4}  # event clock minus tracking clock
GAP = 25
MATCH = "TST001"

MATCH_INFO = f"""<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-{MATCH}" HomeTeamId="{HOME_ID}" GuestTeamId="{AWAY_ID}"
  HomeTeamName="Home FC" GuestTeamName="Away FC" />
<Environment PitchX="105.00" PitchY="68.00" />
</MatchInformation></PutDataRequest>
"""


def kicks() -> list[int]:
    return list(range(20, N_HALF - 20, STEP))


def team_of(person: str) -> int:
    return HOME if PEOPLE[person][0] == HOME_ID else AWAY


def kicker(i: int) -> str:
    return CYCLE[i % len(CYCLE)]


def hold(i: int) -> int:
    return HOLDS[i % len(HOLDS)]


def ball_path() -> tuple[np.ndarray, np.ndarray]:
    """(N_HALF, 2) ball positions and (N_HALF,) owner (1 home, 2 away) of one half."""
    frames = kicks()
    xy = np.zeros((N_HALF, 2))
    owner = np.ones(N_HALF, dtype=int)
    xy[:] = PEOPLE[kicker(0)][1]
    for i, kick in enumerate(frames):
        here = np.array(PEOPLE[kicker(i)][1])
        xy[kick - hold(i) : kick + 1] = here
        owner[kick - hold(i) :] = 1 if team_of(kicker(i)) == HOME else 2
        # Every kick sends the ball on, the last one too (towards a pass the half never logs).
        there = np.array(PEOPLE[kicker(i + 1)][1])
        arrive = kick + STEP - hold(i + 1)
        for t in range(kick + 1, min(arrive, N_HALF)):
            xy[t] = here + (t - kick) / (arrive - kick) * (there - here)
        xy[arrive:] = there
    return xy, owner


def _stamp(section: str, frame: float, offset_s: float = 0.0, utc: bool = False) -> str:
    when = SECTIONS[section][1] + timedelta(seconds=frame / 25 + offset_s)
    return (when.astimezone(UTC) if utc else when).isoformat(timespec="milliseconds")


def positions_xml() -> str:
    xy, owner = ball_path()
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest><Positions>"]
    for section, (first, _) in SECTIONS.items():
        for person, (team, (x, y)) in PEOPLE.items():
            rows = "\n".join(
                f'<Frame N="{first + t}" T="{_stamp(section, t)}" X="{x:.2f}" Y="{y:.2f}" '
                'S="0.00" />'
                for t in range(N_HALF)
            )
            parts.append(
                f'<FrameSet GameSection="{section}" MatchId="DFL-MAT-{MATCH}" TeamId="{team}" '
                f'PersonId="{person}">\n{rows}\n</FrameSet>'
            )
        rows = "\n".join(
            f'<Frame N="{first + t}" T="{_stamp(section, t)}" X="{xy[t, 0]:.3f}" '
            f'Y="{xy[t, 1]:.3f}" Z="0.11" S="0.00" BallPossession="{owner[t]}" BallStatus="1" />'
            for t in range(N_HALF)
        )
        parts.append(
            f'<FrameSet GameSection="{section}" MatchId="DFL-MAT-{MATCH}" TeamId="BALL" '
            f'PersonId="DFL-OBJ-BALL">\n{rows}\n</FrameSet>'
        )
    parts.append("</Positions></PutDataRequest>")
    return "\n".join(parts)


# Second-half kicks that are not plain passes: index -> what the event file says.
SPECIAL = {
    6: "throw_in",  # P-A1 throws in to P-A2; the out, decided earlier, goes to HOME
    9: "shot",  # P-H2, saved
    13: "clearance",  # P-H2
    14: "tackle",  # P-A1 wins it
    15: "claim",  # P-A2 claims the ball
}
FOUL_AFTER = 17  # a foul by AWAY 10 frames after second-half kick 17, before kick 18
DELETED_AT = 11  # a first-half pass logged twice; the twin is deleted
DECIDED_BEFORE = 15  # the throw-in's DecisionTimestamp, frames before the throw


def _play(i: int, frames: list[int]) -> str:
    person = kicker(i)
    receiver = kicker(i + 1) if i + 1 < len(frames) else None
    done = receiver is not None and team_of(receiver) == team_of(person)
    evaluation = "successfullyCompleted" if done else "unsuccessful"
    recipient = f' Recipient="{receiver}"' if done else ""
    team = PEOPLE[person][0]
    return (
        f'<Play Team="{team}" Player="{person}" Evaluation="{evaluation}"{recipient}><Pass/></Play>'
    )


def events_xml() -> str:
    frames = kicks()
    rows = []
    serial = 0

    def event(section: str, frame: float, body: str, event_id: str | None = None) -> None:
        nonlocal serial
        serial += 1
        stamp = _stamp(section, frame, OFFSET_S[section], utc=True)
        rows.append(f'<Event EventId="{event_id or serial}" EventTime="{stamp}">{body}</Event>')

    for section in SECTIONS:
        for i, kick in enumerate(frames):
            person, team = kicker(i), PEOPLE[kicker(i)][0]
            special = SPECIAL.get(i) if section == "secondHalf" else None
            if i == 0:
                body = (
                    f'<KickOff GameSection="{section}" Team="{team}">{_play(i, frames)}</KickOff>'
                )
            elif special == "throw_in":
                decided = _stamp(section, kick - DECIDED_BEFORE, OFFSET_S[section], utc=True)
                body = f'<ThrowIn Team="{team}" DecisionTimestamp="{decided}">{_play(i, frames)}</ThrowIn>'
            elif special == "shot":
                body = f'<ShotAtGoal Team="{team}" Player="{person}"><SavedShot/></ShotAtGoal>'
            elif special == "clearance":
                body = (
                    f'<OtherBallAction Team="{team}" Player="{person}" DefensiveClearance="true"/>'
                )
            elif special == "tackle":
                body = (
                    f'<TacklingGame WinnerTeam="{team}" Winner="{person}" LoserTeam="{HOME_ID}" '
                    'WinnerResult="ballClaimed" PossessionChange="true"/>'
                )
            elif special == "claim":
                body = f'<BallClaiming Team="{team}" Player="{person}" Type="BallClaimed"/>'
            else:
                body = _play(i, frames)
            event(section, kick, body)
            if section == "firstHalf" and i == DELETED_AT:
                event(section, kick + 3, _play(i, frames), event_id="TWIN")
                event(section, kick + 4, '<Delete EventId="TWIN"/>', event_id="DEL")
            if section == "secondHalf" and i == FOUL_AFTER:
                event(section, kick + 10, f'<Foul TeamFouler="{AWAY_ID}" Fouler="P-A1"/>')
        event(section, 30, f'<Substitution Team="{HOME_ID}" PlayerIn="X" PlayerOut="Y"/>')
    # Logged during half-time: outside both periods.
    event("secondHalf", -25 * 600, _play(1, frames))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<PutDataRequest>\n'
        + "\n".join(rows)
        + ("\n</PutDataRequest>")
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("idsse_events")
    meta, positions, events = folder / "meta.xml", folder / "positions.xml", folder / "events.xml"
    meta.write_text(MATCH_INFO, encoding="utf-8")
    positions.write_text(positions_xml(), encoding="utf-8")
    events.write_text(events_xml(), encoding="utf-8")
    return meta, positions, events


@pytest.fixture(scope="module")
def built(files):
    meta, positions, events = files
    bundle = idsse.load_files(MATCH, meta, positions, gap_frames=GAP)
    info = idsse.read_match_info(meta)
    scan = idsse_events.scan_positions(positions.read_bytes(), info)
    table, report = idsse_events.build_table(events.read_bytes(), info, scan, bundle)
    return bundle, info, scan, table, report


def _true_frame(section: str, frame: int, bundle) -> int:
    k = list(SECTIONS).index(section)
    return bundle.meta.periods[k].start + frame


def test_scan_reads_the_ball_clock_and_the_slot_order(files, built):
    bundle, _info, scan, _, _ = built
    assert scan.sections == ("firstHalf", "secondHalf")
    assert [len(t) for t in scan.times] == [N_HALF, N_HALF]
    first = SECTIONS["firstHalf"][1].timestamp()
    assert scan.times[0][0] == pytest.approx(first)
    assert np.allclose(np.diff(scan.times[1]), 0.04)
    assert scan.people == ("P-H1", "P-H2", "P-A1", "P-A2")
    assert np.array_equal(scan.slot_team, bundle.tracking.slot_team)


def test_kinds_teams_and_the_half_lags(built):
    bundle, _, _, table, report = built
    assert [lag["lag_frames"] for lag in report["lags"]] == [20, -10]
    for lag in report["lags"]:
        assert lag["on_touch_at_lag"] == 1.0 and lag["on_touch_at_zero"] < 0.5
    inside = table[table["frame"] >= 0]
    assert set(inside["kind"]) <= set(idsse_events.KINDS)
    # Every kick lands on its true frame after the lag and the refinement.
    for k, section in enumerate(SECTIONS):
        period = bundle.meta.periods[k]
        mine = inside[(inside["period"] == period.index) & (inside["kind"] != "reception")]
        on_ball = mine[np.isin(mine["kind"], idsse_events.TOUCH_KINDS)]
        expect = [_true_frame(section, kick, bundle) for kick in kicks()]
        assert sorted(on_ball["frame"].tolist()) == expect
        assert np.array_equal(on_ball["frame"], on_ball["lag_frame"])
    second = bundle.meta.periods[1].start
    frames = kicks()

    def at(i: int) -> np.void:
        rows = inside[(inside["frame"] == second + frames[i]) & (inside["kind"] != "reception")]
        assert len(rows) == 1
        return rows[0]

    assert (at(0)["kind"], at(0)["detail"], at(0)["team"]) == ("set_piece", "KickOff", H)
    assert (at(6)["kind"], at(6)["detail"], at(6)["team"]) == ("set_piece", "ThrowIn", A)
    assert (at(9)["kind"], at(9)["detail"], at(9)["team"], at(9)["goal"]) == (
        "shot",
        "SavedShot",
        H,
        False,
    )
    assert (at(13)["kind"], at(13)["team"]) == ("clearance", H)
    assert (at(14)["kind"], at(14)["team"], bool(at(14)["ok"])) == ("tackle", A, True)
    assert (at(15)["kind"], at(15)["team"]) == ("other_touch", A)
    assert (at(1)["kind"], at(1)["team"]) == ("pass", H)
    outs = inside[inside["kind"] == "out"]
    assert outs["frame"].tolist() == [second + frames[6] - DECIDED_BEFORE]
    assert outs["team"].tolist() == [H]  # the other team from the throw-in's
    fouls = inside[inside["kind"] == "foul"]
    assert fouls["frame"].tolist() == [second + frames[FOUL_AFTER] + 10]
    assert fouls["team"].tolist() == [A]


def test_receptions_follow_completed_passes(built):
    bundle, _, _, table, report = built
    frames = kicks()
    found = table[table["kind"] == "reception"]
    expect = []
    for section in SECTIONS:
        for i in range(len(frames) - 1):
            special = SPECIAL.get(i) if section == "secondHalf" else None
            passes = special in (None, "throw_in")
            if passes and team_of(kicker(i)) == team_of(kicker(i + 1)):
                frame = _true_frame(section, frames[i + 1] - hold(i + 1), bundle)
                expect.append((frame, team_of(kicker(i))))
    assert sorted(zip(found["frame"].tolist(), found["team"].tolist(), strict=True)) == expect
    assert report["receptions"]["found"] == len(expect)
    assert report["receptions"]["completed_passes"] == len(expect)


def test_deletes_ignored_tags_and_events_outside_the_periods(built):
    _, _, _, table, report = built
    assert report["deleted_ids"] == 2  # the marker's own id and the twin it names
    assert report["ignored_by_tag"] == {"Substitution": 2}
    assert report["outside_periods"] == {"pass": 1}
    assert (table["frame"] < 0).sum() == 1
    assert table["frame"][-1] == -1 and table["lag_frame"][-1] == -1
    inside = table["frame"][table["frame"] >= 0]
    assert (np.diff(inside) >= 0).all()


def test_align_off_keeps_the_raw_clock_and_refine_off_keeps_the_lag(files, built):
    meta, positions, events = files
    bundle, info, scan, table, _ = built
    raw, report = idsse_events.build_table(
        events.read_bytes(), info, scan, bundle, sync="lag", align=False, refine_s=0.0
    )
    assert report["sync"] == "lag" and "elastic" not in report
    assert [lag["lag_frames"] for lag in report["lags"]] == [0, 0]
    assert report["refine"]["moved"] == 0
    # Nothing refined: every logged event keeps its clock frame; a reception is on a touch.
    assert np.array_equal(raw["synced"], raw["kind"] == "reception")
    passes = raw[(raw["kind"] == "pass") & (raw["period"] == 1)]
    aligned = table[(table["kind"] == "pass") & (table["period"] == 1)]
    assert sorted((passes["frame"] + 20).tolist()) == sorted(aligned["frame"].tolist())
    again, _ = idsse_events.build_table(events.read_bytes(), info, scan, bundle)
    assert again.tobytes() == table.tobytes()


def test_the_lag_sync_gives_the_same_frames_here_and_flags_its_refined_events(files, built):
    _, _, events = files
    bundle, info, scan, table, report = built
    assert report["sync"] == "elastic" and "refine" not in report
    lag, lag_report = idsse_events.build_table(events.read_bytes(), info, scan, bundle, sync="lag")
    assert lag_report["lags"] == report["lags"]
    assert lag_report["refine"]["moved"] == 0 and lag_report["refine"]["candidates"] > 0
    assert np.array_equal(lag["frame"], table["frame"])
    assert np.array_equal(lag["kind"], table["kind"])
    on_ball = np.isin(table["kind"], idsse_events.TOUCH_KINDS) & (table["frame"] >= 0)
    assert table["synced"][on_ball].all() and lag["synced"][on_ball].all()
    assert not table["synced"][~on_ball].any()  # fouls, outs, the half-time pass: clock frames
    with pytest.raises(ValueError, match="sync must be one of"):
        idsse_events.build_table(events.read_bytes(), info, scan, bundle, sync="nearest")


def test_elastic_without_the_clock_still_lands_every_kick(files, built):
    _, _, events = files
    bundle, info, scan, table, _ = built
    raw, report = idsse_events.build_table(events.read_bytes(), info, scan, bundle, align=False)
    assert [lag["lag_frames"] for lag in report["lags"]] == [0, 0]
    assert all(half["drift_s"] is None for half in report["elastic"]["periods"])

    # The assignment alone absorbs the 0.8 s and 0.4 s offsets: every on-ball event is on its kick.
    def on_ball(version: np.ndarray) -> np.ndarray:
        return version[np.isin(version["kind"], idsse_events.TOUCH_KINDS) & (version["frame"] >= 0)]

    assert on_ball(raw)[["frame", "kind", "team"]].tolist() == (
        on_ball(table)[["frame", "kind", "team"]].tolist()
    )
    assert on_ball(raw)["synced"].all()


def test_touch_events_dtype_and_possession_changes(built):
    bundle, _, _, table, _ = built
    events = idsse_events.as_touch_events(table)
    assert events.dtype == idsse_events.TOUCH_DTYPE == idsse_events.TOUCH_EVENT_DTYPE
    assert events.dtype == np.dtype([("frame", "<i8"), ("team", "<i1"), ("kind", "<U16")])
    assert (events["frame"] >= 0).all() and (np.diff(events["frame"]) >= 0).all()
    assert set(np.unique(events["team"]).tolist()) == {H, A}
    assert set(events["kind"].tolist()) <= set(idsse_events.KINDS)
    touches = idsse_events.touch_frames(events)
    assert (np.diff(touches) > 0).all()
    changes = idsse_events.possession_changes(events, bundle.meta.periods)
    frames = kicks()
    expect = []
    for section in SECTIONS:
        for i in range(1, len(frames)):
            if team_of(kicker(i)) == team_of(kicker(i - 1)):
                continue
            if section == "secondHalf" and (SPECIAL.get(i) == "throw_in" or i == FOUL_AFTER + 1):
                continue  # through a set piece or a foul: not an open-play change
            expect.append(_true_frame(section, frames[i], bundle))
    assert changes.tolist() == expect
    # A set piece of the same team starts a new spell but is not a change.
    empty = np.zeros(0, dtype=idsse_events.TOUCH_DTYPE)
    assert idsse_events.possession_changes(empty, bundle.meta.periods).tolist() == []


def test_touch_events_keep_team_rows_inside_the_periods_once():
    rows = [
        (10, H, "pass"),
        (10, H, "pass"),  # two events snapped onto one touch: one row
        (10, A, "tackle"),
        (12, idsse_events.NO_TEAM, "set_piece"),  # a dropped ball: no team, left out
        (15, A, "reception"),
        (-1, H, "pass"),  # outside every period: left out
    ]
    table = np.zeros(len(rows), dtype=idsse_events.EVENT_DTYPE)
    for i, (frame, team, kind) in enumerate(rows):
        table[i]["frame"], table[i]["team"], table[i]["kind"] = frame, team, kind
    got = idsse_events.as_touch_events(table)
    assert got.tolist() == [(10, H, "pass"), (10, A, "tackle"), (15, A, "reception")]
    empty = idsse_events.as_touch_events(np.zeros(0, dtype=idsse_events.EVENT_DTYPE))
    assert empty.dtype == idsse_events.TOUCH_DTYPE and len(empty) == 0


def test_parse_reads_naive_times_as_utc():
    naive = idsse_events.posix_seconds("2022-10-15T11:30:00.000")
    aware = idsse_events.posix_seconds("2022-10-15T13:30:00.000+02:00")
    assert naive == aware == datetime(2022, 10, 15, 11, 30, tzinfo=UTC).timestamp()


def test_pins_downloads_and_the_exam_gate(tmp_path, monkeypatch):
    raw = b"<PutDataRequest/>"
    blob = idsse_events.git_blob_id(raw)
    source = tmp_path / "remote.xml"
    source.write_bytes(raw)
    dest = tmp_path / "local" / "events.xml"
    idsse_events.download_pinned(source.as_uri(), dest, len(raw), blob)
    assert dest.read_bytes() == raw
    bad = tmp_path / "local" / "bad.xml"
    with pytest.raises(ValueError, match="pinned blob"):
        idsse_events.download_pinned(source.as_uri(), bad, len(raw), "0" * 40)
    assert not bad.exists() and not bad.with_name(bad.name + ".part").exists()
    with pytest.raises(ValueError, match="Fetch it again"):
        idsse_events.check_pin("J03WMX", raw)
    for match in idsse.EXAM_MATCHES:
        with pytest.raises(PermissionError, match="sealed exam"):
            idsse_events.touch_events(match)
        with pytest.raises(PermissionError, match="sealed exam"):
            idsse_events.fetch_events(match)
    names = {idsse_events.event_file_name(m) for m in idsse.MATCHES}
    assert len(names) == 7 and all(n.startswith("DFL_03_02_events_raw_DFL-COM-") for n in names)


def git_blob_reference(raw: bytes) -> str:
    """The id `git hash-object` prints, written independently of the module."""
    import hashlib

    header = f"blob {len(raw)}".encode() + b"\x00"
    return hashlib.sha1(header + raw).hexdigest()  # noqa: S324


def test_git_blob_id_is_gits_object_id():
    for raw in (b"", b"hello\n", bytes(range(256))):
        assert idsse_events.git_blob_id(raw) == git_blob_reference(raw)
    assert idsse_events.git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


@pytest.fixture
def installed(files, tmp_path, monkeypatch):
    """The fixture installed as an IDSSE match with pins, in a private external and cache dir."""
    import hashlib

    meta, positions, events = files
    monkeypatch.setenv("AAAAG_EXTERNAL_DIR", str(tmp_path / "external"))
    monkeypatch.setenv("AAAAG_CACHE_DIR", str(tmp_path / "cache"))
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (meta, positions)}
    monkeypatch.setitem(idsse.MATCHES, MATCH, ("000009", sha[meta], sha[positions]))
    raw = events.read_bytes()
    monkeypatch.setitem(idsse_events.EVENT_FILES, MATCH, (len(raw), idsse_events.git_blob_id(raw)))
    monkeypatch.setattr(idsse_events, "_MEMO", {})
    paths = idsse.file_paths(MATCH)
    paths["meta"].parent.mkdir(parents=True)
    paths["meta"].write_bytes(meta.read_bytes())
    paths["positions"].write_bytes(positions.read_bytes())
    idsse_events.event_path(MATCH).write_bytes(raw)
    return tmp_path


def test_event_table_caches_and_serves_the_same_bytes(installed, built, monkeypatch):
    _, _, _, direct, _ = built
    table, report = idsse_events.event_table(MATCH, gap_frames=GAP)
    assert table.tobytes() == direct.tobytes()
    assert [lag["lag_frames"] for lag in report["lags"]] == [20, -10]
    stored = sorted((installed / "cache" / "idsse_events").glob(f"{MATCH}-*.npz"))
    assert len(stored) == 2  # the positions scan and the table
    monkeypatch.setattr(idsse_events, "_MEMO", {})
    cached, again = idsse_events.event_table(MATCH, gap_frames=GAP)
    assert cached.tobytes() == table.tobytes() and again == report
    table[0]["frame"] = -5  # callers get copies
    assert idsse_events.event_table(MATCH, gap_frames=GAP)[0].tobytes() == cached.tobytes()
    events = idsse_events.touch_events(MATCH)
    assert np.array_equal(events, idsse_events.as_touch_events(cached))
    store = installed / "cache" / "idsse_events"
    # refine_s is a lag parameter: the elastic entry ignores it.
    same, _ = idsse_events.event_table(MATCH, gap_frames=GAP, refine_s=0.0)
    assert same.tobytes() == cached.tobytes()
    assert len(sorted(store.glob(f"{MATCH}-v2-*.npz"))) == 1
    # The lag sync and its parameters are other entries; on this fixture refinement moves nothing.
    lag, lag_report = idsse_events.event_table(MATCH, gap_frames=GAP, sync="lag")
    other, other_report = idsse_events.event_table(MATCH, gap_frames=GAP, sync="lag", refine_s=0.0)
    assert other_report["refine"]["radius_frames"] == 0 and lag_report["refine"]["moved"] == 0
    assert np.array_equal(other["frame"], cached["frame"])
    assert np.array_equal(lag["frame"], cached["frame"])
    assert np.array_equal(idsse_events.touch_events(MATCH, sync="lag"), events)
    assert len(sorted(store.glob(f"{MATCH}-v2-*.npz"))) == 3
    with pytest.raises(ValueError, match="sync must be one of"):
        idsse_events.event_table(MATCH, gap_frames=GAP, sync="nearest")


def test_a_bundle_of_the_match_is_used_and_a_wrong_one_refused(installed, built):
    bundle = built[0]
    events = idsse_events.touch_events(MATCH, bundle=bundle)
    assert len(events) and events.dtype == idsse_events.TOUCH_DTYPE
    from dataclasses import replace

    renamed = replace(bundle.meta, match_id="J03WMX")
    wrong = replace(bundle, tracking=replace(bundle.tracking, meta=renamed))
    with pytest.raises(ValueError, match="clean idsse bundle"):
        idsse_events.touch_events(MATCH, bundle=wrong)


def test_a_wrong_event_file_is_refused(installed, monkeypatch):
    idsse_events.event_path(MATCH).write_bytes(b"<PutDataRequest/>")
    with pytest.raises(ValueError, match="Fetch it again"):
        idsse_events.event_table(MATCH, gap_frames=GAP, use_cache=False)


def test_probe_prints_aggregates(installed, capsys):
    idsse_events.main(["--probe", MATCH])
    out = capsys.readouterr().out
    assert "blob ok" in out and "period 1: lag +0.80 s" in out and "period 2: lag -0.40 s" in out
    assert "label team = event team" in out and "open-play possession changes" in out
    assert "elastic sync: contacts" in out and "pooled over 1 matches" in out
    for name in ("raw clock", "half lag", "lag + refine", "elastic"):
        assert f"contact ({name})" in out
    assert "  elastic  matched 1.000 of" in out and "  lag      matched 1.000 of" in out
    assert len(out.splitlines()) < 40  # a summary, not a dump
    with pytest.raises(SystemExit):
        idsse_events.main([MATCH])  # neither --fetch nor --probe


def test_training_targets_read_the_idsse_events(installed, built):
    from aaaag.ml.data import TrainMatch, event_switch_target, touch_target
    from aaaag.ml.spec import TokenSpec

    bundle = built[0]
    match = TrainMatch.from_bundle(bundle, TokenSpec())
    events = idsse_events.touch_events(MATCH, bundle=bundle)
    tol = 1
    target = touch_target(match, tol, "events")
    marked = np.zeros(bundle.meta.n_frames, dtype=bool)
    for frame in idsse_events.touch_frames(events):
        marked[max(0, frame - tol) : frame + tol + 1] = True
    supervised = np.isin(match.labels, (H, A)) & match.played
    assert np.array_equal(target == 1, supervised & marked)
    assert np.array_equal(target == 0, supervised & ~marked)
    assert np.isnan(target[~supervised]).all()
    assert not np.array_equal(target, touch_target(match, tol), equal_nan=True)

    switch = event_switch_target(match, 2)
    changes = idsse_events.possession_changes(events, bundle.meta.periods)
    assert len(changes) > 0
    near = np.zeros(bundle.meta.n_frames, dtype=bool)
    for frame in changes:
        near[frame - 2 : frame + 3] = True
    assert np.array_equal(switch == 1, supervised & near)
    assert np.array_equal(switch == 0, supervised & ~near)
    assert np.isnan(switch[~supervised]).all()


# -- the elastic sync: the assignment, and a drifting event clock ----------------------------------


def _options(*pairs: tuple[int, float]) -> tuple[np.ndarray, np.ndarray]:
    index = np.array([i for i, _ in pairs], dtype=np.int64)
    return index, np.array([score for _, score in pairs], dtype=np.float64)


def test_monotone_assign_keeps_the_order_and_one_event_per_contact():
    assign = idsse_events.monotone_assign
    # Greedy would give event 0 contact 2 and event 1 nothing better than 3; the best total agrees.
    assert assign([_options((0, 1.0), (2, 3.0)), _options((1, 2.0), (3, 0.5))]).tolist() == [2, 3]
    # Crossing is not allowed: the larger score wins, the other event stays unmatched.
    assert assign([_options((2, 3.0)), _options((1, 2.5))]).tolist() == [2, -1]
    assert assign([_options((2, 2.0)), _options((1, 2.5))]).tolist() == [-1, 1]
    # One contact, two events: on a tie the earlier event keeps it.
    assert assign([_options((1, 2.0)), _options((1, 2.0))]).tolist() == [1, -1]
    # A score of 0 or less is never taken; an event without options stays unmatched.
    assert assign([_options((0, 0.0)), _options((1, -1.0)), _options()]).tolist() == [-1, -1, -1]
    assert assign([]).tolist() == []
    # Separate blocks (contacts 0, 1 and 4, 5) give what one table would.
    got = assign([_options((0, 1.0)), _options((1, 1.0)), _options((5, 1.0)), _options((4, 2.0))])
    assert got.tolist() == [0, 1, -1, 4]
    # A shifted clock: every event scores its successor's contact a little higher. With nothing in
    # the way the chain moves by one; when the last event can only take its own contact, one
    # contact per event and the order hold the whole chain in place.
    chain = [_options((i, 1.0), (i + 1, 1.2)) for i in range(5)]
    assert assign(chain).tolist() == [1, 2, 3, 4, 5]
    chain[-1] = _options((4, 1.0))
    assert assign(chain).tolist() == [0, 1, 2, 3, 4]


DRIFT_N = 25 * 60 * 20  # a 20-minute half
DRIFT_S = 2.0  # the event clock runs from -2 s to +2 s off its half offset (back again in half 2)
STEPS = (37, 43, 51, 41, 47, 39, 45)  # irregular kicks: no lag one cycle off fits as well
SPECTATOR = ("P-H3", (-40.0, 30.0))  # a home player far from every ball position
PHANTOM_AFTER = 200  # first half: an other_touch credited to the spectator, after this kick


def _drift_kicks(n: int) -> list[int]:
    out, frame = [], 20
    while frame < n - 60:
        out.append(frame)
        frame += STEPS[len(out) % len(STEPS)]
    return out


def _drift_ball(n: int, frames: list[int]) -> np.ndarray:
    """(n, 2) ball positions: kick i leaves kicker(i) at frames[i] and rests at kicker(i + 1) for
    hold(i + 1) frames before its kick."""
    xy = np.zeros((n, 2))
    xy[:] = PEOPLE[kicker(0)][1]
    for i, kick in enumerate(frames):
        here, there = np.array(PEOPLE[kicker(i)][1]), np.array(PEOPLE[kicker(i + 1)][1])
        following = frames[i + 1] if i + 1 < len(frames) else kick + STEPS[0]
        arrive = following - hold(i + 1)
        t = np.arange(kick + 1, min(arrive, n))
        xy[t] = here + ((t - kick) / (arrive - kick))[:, None] * (there - here)
        xy[arrive:] = there
    return xy


def _drift_offset(section: str, frame: int, n: int) -> float:
    sign = 1.0 if section == "firstHalf" else -1.0
    return OFFSET_S[section] + sign * DRIFT_S * (2.0 * frame / (n - 1) - 1.0)


@pytest.fixture(scope="module")
def drifting():
    """(bundle, info, scan, events bytes, kick frames, phantom frame) of the drifting fixture."""
    n = DRIFT_N
    frames = _drift_kicks(n)
    people = {**PEOPLE, SPECTATOR[0]: (HOME_ID, SPECTATOR[1])}
    slots = sorted(people, key=lambda p: (0 if people[p][0] == HOME_ID else 1, p))
    slot_team = np.array([H if people[p][0] == HOME_ID else A for p in slots], dtype=np.int8)
    periods = (Period(1, 0, n), Period(2, n + GAP, 2 * n + GAP))
    T = 2 * n + GAP
    player_xy = np.full((T, len(slots), 2), np.nan, dtype=np.float32)
    ball_xy = np.full((T, 2), np.nan, dtype=np.float32)
    for period in periods:
        player_xy[period.start : period.stop] = np.array([people[p][1] for p in slots])
        ball_xy[period.start : period.stop] = _drift_ball(n, frames)
    meta = MatchMeta(MATCH, "idsse", T, 25, 105.0, 68.0, periods)
    bundle = MatchBundle(Tracking(meta, player_xy, slot_team, ball_xy)).validate()
    info = idsse.MatchInfo(HOME_ID, AWAY_ID, "Home FC", "Away FC", 105.0, 68.0)
    times = tuple(start.timestamp() + np.arange(n) / 25 for _, start in SECTIONS.values())
    scan = idsse_events.PositionsScan(tuple(SECTIONS), times, tuple(slots), slot_team)

    rows = []

    def event(section: str, frame: int, body: str) -> None:
        offset = _drift_offset(section, frame, n)
        stamp = _stamp(section, frame, offset, utc=True)
        rows.append(f'<Event EventId="{len(rows) + 1}" EventTime="{stamp}">{body}</Event>')

    phantom = (frames[PHANTOM_AFTER] + frames[PHANTOM_AFTER + 1]) // 2
    for section in SECTIONS:
        for i, kick in enumerate(frames):
            team = PEOPLE[kicker(i)][0]
            play = _play(i, frames)
            body = (
                f'<KickOff GameSection="{section}" Team="{team}">{play}</KickOff>'
                if not i
                else play
            )
            event(section, kick, body)
            if section == "firstHalf" and i == PHANTOM_AFTER:
                spectator = f'<OtherBallAction Team="{HOME_ID}" Player="{SPECTATOR[0]}"/>'
                event(section, phantom, spectator)
    raw = '<?xml version="1.0"?>\n<PutDataRequest>\n' + "\n".join(rows) + "\n</PutDataRequest>"
    return bundle, info, scan, raw.encode(), frames, phantom


def test_elastic_recovers_a_drifting_clock_and_keeps_the_order(drifting):
    bundle, info, scan, raw, frames, _ = drifting
    table, report = idsse_events.build_table(raw, info, scan, bundle)
    slot = {p: s for s, p in enumerate(scan.people)}
    for k, period in enumerate(bundle.meta.periods):
        mine = table[
            (table["period"] == period.index) & np.isin(table["kind"], ("pass", "set_piece"))
        ]
        expect = period.start + np.array(frames)
        assert len(mine) == len(expect) and mine["synced"].all()
        # Every kick within 2 frames of its true frame (here: on it), each by its own kicker, in the
        # logged order: the table is sorted by frame, and the kickers follow the cycle.
        assert np.abs(mine["frame"] - expect).max() <= idsse_events.SYNC_TOL_FRAMES
        assert mine["actor"].tolist() == [slot[kicker(i)] for i in range(len(frames))]
        # The clock drifted by 4 s across the half; the local lag followed it.
        drift = report["elastic"]["periods"][k]["drift_s"]
        assert drift is not None and drift[2] - drift[0] >= 3.5
        shift = (mine["frame"] - mine["lag_frame"]) / 25
        assert shift.max() - shift.min() >= 3.5  # the half lag alone was up to 2 s off
    # One lag per half cannot follow it: refinement within +-0.5 s finds a minority of the kicks.
    lag, _ = idsse_events.build_table(raw, info, scan, bundle, sync="lag")
    passes = lag[np.isin(lag["kind"], ("pass", "set_piece")) & (lag["frame"] >= 0)]
    true = np.concatenate([period.start + np.array(frames) for period in bundle.meta.periods])
    gap = idsse_events._nearest_gap(passes["frame"], true)
    assert (gap <= idsse_events.SYNC_TOL_FRAMES).mean() < 0.5


def test_elastic_leaves_an_event_without_a_contact_unmatched(drifting):
    bundle, info, scan, raw, frames, phantom = drifting
    table, report = idsse_events.build_table(raw, info, scan, bundle)
    spectator = scan.people.index(SPECTATOR[0])
    rows = table[table["actor"] == spectator]
    assert len(rows) == 1 and rows[0]["kind"] == "other_touch"
    # No contact within reach: unmatched, on its clock frame (the half lag plus the local drift),
    # which follows the drifting clock to within a few frames.
    assert not rows[0]["synced"] and rows[0]["period"] == 1
    assert abs(int(rows[0]["frame"]) - phantom) <= 6
    first = report["elastic"]["periods"][0]
    assert first["events"] == len(frames) + 1 and first["matched"] == len(frames)
    # Its neighbours keep their kicks.
    for i in (PHANTOM_AFTER, PHANTOM_AFTER + 1):
        hit = table[(table["frame"] == frames[i]) & np.isin(table["kind"], ("pass", "set_piece"))]
        assert len(hit) == 1 and hit[0]["synced"]


def test_control_changes_count_controlled_actions_only():
    rows = [  # frame, team, kind, ok
        (10, H, "pass", True),
        (20, A, "tackle", True),  # a tackle is not control
        (25, A, "other_touch", False),  # nor a deflection
        (30, A, "pass", False),  # nor a failed pass
        (40, A, "pass", True),  # the new team's first controlled action: change at 40
        (50, A, "reception", True),
        (60, H, "clearance", False),
        (70, H, "shot", False),  # a shot is control: change at 70
        (80, NO_TEAM_ROW, "pass", True),  # no team: ignored
        (90, A, "pass", True),  # change at 90
        (5, A, "pass", True),  # a different period below
    ]
    table = np.zeros(len(rows), dtype=idsse_events.EVENT_DTYPE)
    for i, (frame, team, kind, ok) in enumerate(rows):
        table[i]["frame"], table[i]["team"], table[i]["kind"], table[i]["ok"] = (
            frame,
            team,
            kind,
            ok,
        )
    periods = (Period(1, 0, 100),)
    got = idsse_events.control_changes(table, periods)
    assert got.tolist() == [10, 40, 70, 90]  # the A pass at 5 starts the half; H takes over at 10
    assert idsse_events.control_changes(table[:0], periods).tolist() == []
    stats = idsse_events.control_stats(got, np.array([12, 41, 200]))
    assert stats["events"] == 4 and stats["labels"] == 3
    assert stats["label_hit"] == [1, 2, 2, 2]  # within 1, 3, 5, 10 frames
    assert stats["offset"].tolist() == [-2, -1, -110]


NO_TEAM_ROW = idsse_events.NO_TEAM
