"""IDSSE event data (DFL `events_raw`) on the `idsse` frame axis: touches, possession changes, restarts.

External data only: DFL / Sportec IDSSE (Bassek et al. 2025, CC BY 4.0), fetched from the Hugging
Face mirror at the revision `idsse.REVISION` pins. Nothing here reads BAR-ATH.

    python -m aaaag.sources.idsse_events --fetch            # the five dev matches' event files
    python -m aaaag.sources.idsse_events --probe            # aggregates only (needs idsse fetched)

**Integrity.** The mirror stores these small files outside Git LFS, so its listing publishes no
sha256 for them. Each is pinned by its size and its git blob id at the revision (the SHA-1 of
``b"blob <size>\\0" + bytes``, the content hash Git and the Hub publish), checked on every fetch and
on every parse. `--fetch` prints each verified file's sha256 for the record.

**The file.** One ``<Event EventId=... EventTime="ISO 8601">`` per logged action. Its first child
names the type and nests the detail, as kloppy 3.18's Sportec event deserializer reads it:

- ``Play`` (Team, Player, Evaluation, Recipient) with a Pass or Cross child: **pass**;
- ``KickOff``, ``ThrowIn``, ``GoalKick``, ``CornerKick``, ``FreeKick``, ``Penalty`` wrapping the
  Play or ShotAtGoal that takes it: **set_piece** (the kick, credited to the kicker's team).
  ``ThrowIn``, ``GoalKick`` and ``CornerKick`` also give an **out** at their ``DecisionTimestamp``,
  credited to the other team: Laws 15-17 award those restarts to the opponents of the team that
  last touched the ball. ``RefereeBall`` (dropped ball) is a set_piece with no team;
- ``ShotAtGoal`` (Team, Player) with an outcome child (SuccessfulShot, SavedShot, ...): **shot**;
- ``TacklingGame`` (WinnerTeam, Winner, WinnerResult, PossessionChange): **tackle**, credited to
  the winner;
- ``OtherBallAction`` (Team, Player, DefensiveClearance): **clearance** when DefensiveClearance is
  true, else **other_touch**; ``BallClaiming`` and ``BallDeflection``: **other_touch**;
- ``Foul`` (TeamFouler, Fouler): **foul**, credited to the fouling team;
- **reception** is not logged by DFL. For a completed pass (Evaluation successfullyCompleted or
  successful) that names a Recipient, it is the first tracked contact (below: a kinematic touch
  for `sync="lag"`, any contact for `sync="elastic"`) after the pass and before the next logged
  event at which the recipient's track is within `REACH_M` of the ball. It is credited to the
  passing team, and is absent when there is no such contact;
- everything else (Offside, Caution, Substitution, FinalWhistle, ...) gives no output and is
  counted by tag in the report. A ``Delete`` marker cancels an EventId: the marker and every
  event with that id are dropped.

**Output.** `touch_events(match_id)` has the contract of `aaaag.sources.pff_events.touch_events`
(PR #217): TOUCH_DTYPE rows inside the periods, team HOME or AWAY, kind from KINDS, sorted by
frame, identical rows merged. `event_table` keeps everything else (EVENT_DTYPE): rows without a
team, rows outside the periods (frame -1), the lag-corrected frame, whether the sync put the event
on a tracked contact, and the DFL detail.

**Synchronisation: how an event gets a frame.** Two methods: `sync="elastic"` (the default) and
`sync="lag"` (the first version, kept for comparison). Both start the same way, per game section
(firstHalf, secondHalf, then the extra-time halves, in playing order; each is one `idsse` period):

1. *Clock.* Every ball ``Frame`` of the positions file carries ``T``, its wall-clock time. The
   section's ball frames are its period on the frame axis (``idsse.to_bundle``: row =
   period.start + N - N_first). An event time (EventTime; DecisionTimestamp for an out) maps to
   the row of the ball frame nearest in time, the 25 Hz grid extended past either end. It belongs
   to the section whose time span, widened by `ALIGN_MAX_LAG_S` each side, contains it. A time
   without a UTC offset is read as UTC.
2. *Half lag.* The DFL event clock runs a few seconds off the tracking clock, by a different amount
   each half (on the dev matches: -0.68 to +1.60 s, `analysis/findings/2026-11-06-ameya-wf1-
   control_semantics.md`; raw stamps up to 2.4 s early, `...-wf1-synthesis.md` E3). A kick is a
   kinematic touch of the clean ball: `postprocess.touch_snap.detect_touches` (a ball velocity
   change of at least 4 m/s with a player within `REACH_M`), credited to the nearest player's team.
   Every event of a half moves by one lag: the one in +-`ALIGN_MAX_LAG_S` that maximises, over
   the half's passes, set-piece kicks and shots, a triangular kernel of the distance to the
   nearest kinematic touch of their own team (1 on it, 0 beyond `SYNC_TOL_FRAMES`): a
   cross-correlation of the two touch trains. Ties go to the smaller |lag|. A half with fewer than
   `ALIGN_MIN_EVENTS` such kicks keeps lag 0. `align=False` skips this step (and, for
   `sync="elastic"`, step 4 too: every event then searches around its raw clock frame).
   (The passer-to-ball distance used in the WF1 analysis has a flat minimum while a player holds
   the ball, so its argmin is not sharp; a touch is.) The frame after this step is `lag_frame`,
   the lag-corrected frame, in both methods.

`sync="lag"` then refines each event on its own:

3. *Refinement* (`refine_s` > 0). An on-ball event whose actor has a track moves to the nearest
   kinematic touch within +-`refine_s` of its lagged frame (inside the period) at which that track
   is within `REACH_M` of the ball; without one it stays. Ties go to the earlier frame. `REFINE_S`
   is 0.5 s, the residual per-event jitter of the DFL stamps after the half lag (WF1 synthesis E3,
   measured on the IDSSE dev matches).

On the dev matches that left only 0.43-0.66 of the refined events within 2 frames of a tracked
contact: the DFL clock drifts inside a half. `sync="elastic"` absorbs that drift and assigns the
on-ball events to tracked contacts jointly, in the spirit of ELASTIC (Kim et al. 2025, "ELASTIC:
event-tracking data synchronization in soccer without annotated event locations", arXiv
2508.09238) and ETSY (Van Roy et al. 2023): candidate contacts from the tracking alone, a score per
event and candidate, then an order-keeping assignment (a Needleman-Wunsch-type alignment, the way
DataBallPy aligns IDSSE). `refine_s` is unused.

3. *Contacts.* Every kinematic touch (above), and every soft contact more than `SYNC_TOL_FRAMES`
   from all of them: a velocity-change peak of at least `SOFT_THETA` m/s while a player is within
   `CLOSE_M` of the ball (a cushioned reception, a dribble touch). `contact_frames`.
4. *Local clock.* The lag is re-estimated around every kick (pass, set piece or shot of a known
   player or team): `local_lags` takes the lag within +-`ALIGN_MAX_LAG_S` of the half lag that puts
   the most kicks within +-`DRIFT_WINDOW_S` on, or nearest to, a contact at which their own player
   is within `REACH_M` (the half-lag kernel, per player instead of per team). Ties go to the lag
   nearest the half lag. A kick with fewer than `DRIFT_MIN_EVENTS` kicks in its window, or with no
   hit at all, has no opinion. An event's *clock frame* is its raw frame plus the local lag,
   interpolated linearly in time between kicks and held flat beyond the first and the last; a
   half without any opinion keeps its half lag. The report gives the drift (local lag minus half
   lag) per half.
5. *Scores.* An on-ball event (pass, shot, tackle, clearance, other_touch, set_piece) can take a
   contact within +-`WINDOW_S` of its clock frame, inside its period, at which its player is
   within `REACH_M` of the ball. Its player is the actor's track; else the nearest track of the
   event's team; else the nearest track. The score of a contact, in nats, is
   ``MATCH_NATS - (dt / TIME_SIGMA_S)**2 / 2 - (d / REACH_SIGMA_M)**2 / 2 + kick``: dt is the time
   from the clock frame (s), d the player-ball distance (m), and kick, for a pass, shot,
   set_piece or clearance only, is clip(rise / `KICK_RISE_MPS`, -1, 1), where rise is the ball's
   speed over the next `KICK_FRAMES` frames minus its speed over the previous `KICK_FRAMES` (a kick
   speeds the ball up, a reception slows it; an unseen ball counts 0). A contact on time and at
   the player's track scores `MATCH_NATS`; one 2 sigma off in time with nothing else for it
   scores 0.
6. *Assignment.* Per half, `monotone_assign` picks the contacts that maximise the summed score of
   the matched events, with the matched contacts strictly increasing in the logged order (raw
   time, then file order): the order is kept, a contact takes at most one event, and a local
   shift costs nothing beyond its time term. An unmatched event scores 0, so no event takes a
   contact whose score is 0 or less.
7. *Unmatched.* An on-ball event without a contact keeps its clock frame (the half lag plus the
   local drift, not the bare half lag: that is the better estimate of its time) with `synced`
   False; fouls and outs take their clock frame too.

Both methods end the same way:

- *Receptions* (above), searched after the sync, on the final frames.
- `synced` is True for an event placed on a tracked contact: matched (elastic), found by the
  refinement (lag), and every reception. `lag_frame` keeps the lag-corrected frame.
- *Periods.* An event whose final frame is outside its period is dropped from `touch_events`
  (frame -1 in `event_table`) and counted by the probe.

Players: ``PersonId`` maps to the bundle's slot in `idsse.to_bundle`'s order (home first, then
away, each sorted by PersonId), read from the positions file's FrameSet headers.

Signals: the event file, the matchinformation team ids, the positions file's ball-frame times and
FrameSet headers, and the clean IDSSE bundle (ball and player tracks).

Where it fails: events are logged by hand, so an untagged touch is missing and a team can be
wrong; a touch too soft for the detectors, or made more than `REACH_M` from the player's track (a
header, a tracking error), is not found, so the event keeps its clock frame (lag: its lagged
frame) or the reception is absent; DFL possession (the labels) is not the same object as the event
sequence (#14). Per method:

- lag: one lag per half leaves any drift inside the half, and refinement snaps a dribble to
  whichever of its touches is nearest in time;
- elastic: a wrong player in the log makes the right contact infeasible (the event stays unmatched
  or takes another contact of that player); in a dribble the kick term prefers the touch that
  speeds the ball up, which is usually the pass but can be a hard dribble touch; two events logged
  on one touch (a tackle won and the next pass in one touch) cannot share it, so one moves to a
  neighbouring contact or stays unmatched; the local clock follows drift on the scale of
  `DRIFT_WINDOW_S`, so a jump in the DFL clock is smeared over that window and a long stoppage is
  bridged by interpolation; the score's scales come from physics and the measured jitter, not from
  a fit, because no frame-annotated IDSSE events exist here to fit them; in strictly periodic play
  (synthetic only) a lag one cycle off fits as well, and the tie goes to the lag nearest the half
  lag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import sys
import threading
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..contracts import AWAY, HOME, A, H, MatchBundle, Period, Tracking
from ..paths import cache_dir, external_dir
from ..postprocess.touch_snap import detect_touches
from . import idsse

VERSION = "2"  # bump whenever the table produced from the same files changes (2: elastic sync)
SCAN_VERSION = "1"  # bump whenever `scan_positions` changes
FPS = idsse.FPS
NO_TEAM = -1

KINDS = (
    "pass",
    "reception",
    "shot",
    "tackle",
    "clearance",
    "other_touch",
    "set_piece",
    "foul",
    "out",
)
TOUCH_KINDS = ("pass", "reception", "shot", "tackle", "clearance", "other_touch", "set_piece")
STOP_KINDS = ("foul", "out", "set_piece")  # a possession change across one of these is a restart
TOUCH_DTYPE = np.dtype([("frame", "<i8"), ("team", "<i1"), ("kind", "<U16")])
TOUCH_EVENT_DTYPE = TOUCH_DTYPE  # the name `aaaag.sources.pff_events` uses for the same dtype
EVENT_DTYPE = np.dtype(
    [
        ("frame", "<i8"),  # final frame on the bundle axis; -1 outside every period
        ("team", "<i1"),  # HOME, AWAY or NO_TEAM
        ("kind", "<U16"),  # one of KINDS
        ("detail", "<U24"),  # the DFL element: restart type, shot outcome, WinnerResult, ...
        ("ok", "?"),  # completed pass, successful restart pass, or a tackle that changed possession
        ("goal", "?"),  # a shot (direct or from a restart) whose outcome is SuccessfulShot
        ("period", "<i1"),  # 1-based period index; 0 outside
        ("lag_frame", "<i8"),  # frame after the half lag (the lag-corrected frame); -1 outside
        ("actor", "<i4"),  # slot of the acting player (the recipient for a reception); -1 unknown
        ("synced", "?"),  # the frame is a tracked contact the sync chose; False: the clock frame
    ]
)

SYNCS = ("elastic", "lag")
SYNC = "elastic"
ALIGN_MAX_LAG_S = 4.0  # covers the largest measured offset (2.4 s) with room for the exam matches
ALIGN_MIN_EVENTS = 20  # fewer kicks in a half than this: lag 0
ALIGN_KINDS = ("pass", "set_piece", "shot")
SYNC_TOL_FRAMES = 2  # an event "on" a touch: the detector's own half-window (w = 2)
REFINE_S = 0.5
REFINE_KINDS = ("pass", "shot", "tackle", "clearance", "other_touch", "set_piece")
REACH_M = 2.5  # the reach of the kinematic touch detector (postprocess.touch_snap.detect_touches)

# sync="elastic" (module docstring, steps 3-7). Scales in seconds, metres and m/s, set from physics
# and the measured stamp jitter; nothing here is fitted to a score.
SYNC_KINDS = REFINE_KINDS  # the logged on-ball events a sync moves; receptions are found after it
KICK_KINDS = ("pass", "shot", "set_piece", "clearance")  # the ball leaves the player faster
SOFT_THETA = 2.0  # m/s: a soft contact's velocity-change peak, half the kinematic touch's 4 m/s
CLOSE_M = 1.5  # ... with a player this close: at the foot, not merely near
DRIFT_WINDOW_S = 60.0  # the kicks within +-60 s vote on the local lag (drift is slower than that)
DRIFT_MIN_EVENTS = 10  # fewer voting kicks than this: no local opinion, interpolate
WINDOW_S = 5.0  # an on-ball event looks for its contact within +-5 s of its clock frame
TIME_SIGMA_S = REFINE_S  # stamp jitter around the clock frame (WF1 E3: 0.5 s after the half lag)
REACH_SIGMA_M = 1.0  # player track to ball at a contact: about a leg's length
KICK_FRAMES = 5  # ball speed over 0.2 s before and after a contact
KICK_RISE_MPS = 5.0  # a 5 m/s speed rise is a full kick cue (+1 nat); as large a fall, -1
MATCH_NATS = 2.0  # a contact 2 sigma off in time, with nothing else for it, scores 0

SUCCESS = ("successfullyCompleted", "successful")
RESTARTS = ("KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty")
OUT_RESTARTS = ("ThrowIn", "GoalKick", "CornerKick")
GOAL = "SuccessfulShot"

# match id -> (events file size in bytes, git blob id) at idsse.REVISION, from the mirror's listing.
EVENT_FILES = {
    "J03WMX": (776578, "bc2f5265763786ac8c05cd68595acc614deee50c"),
    "J03WN1": (612114, "67b982007b8e3f5de83dcd104779acc92b76a813"),
    "J03WOH": (635747, "c651487e6d0defbc033f87fadb1b5ebfcad75037"),
    "J03WOY": (683018, "9d2bde37ff79a5a9f993882bc1d83aef70a51d8e"),
    "J03WPY": (698898, "c8065cd1464a67a0312a7788bc6da6442084d141"),
    "J03WQQ": (732186, "081d5489af3437a7e3dbe7219226136bf61f7fd4"),
    "J03WR9": (681050, "1c9cc3875850b25837d37d7789554f3112ed1b76"),
}
assert set(EVENT_FILES) == set(idsse.MATCHES), "one pinned event file per IDSSE match"

_T = re.compile(rb' T="([^"]*)"')
_LOCK = threading.Lock()
_MEMO: dict[tuple[str, str], tuple[np.ndarray, dict]] = {}


# -- files ----------------------------------------------------------------------------------------


def event_file_name(match_id: str) -> str:
    competition = idsse._entry(match_id)[0]
    return f"DFL_03_02_events_raw_DFL-COM-{competition}_DFL-MAT-{match_id}.xml"


def event_path(match_id: str) -> Path:
    return external_dir() / "idsse" / event_file_name(match_id)


def git_blob_id(raw: bytes) -> str:
    """The git object id of a file's bytes: SHA-1 of ``b"blob <size>\\0" + raw``."""
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()  # noqa: S324 (git's id)


def check_pin(match_id: str, raw: bytes) -> None:
    """Raise unless `raw` is the pinned event file of `match_id` (size and git blob id)."""
    size, blob = EVENT_FILES[match_id]
    if len(raw) != size or git_blob_id(raw) != blob:
        raise ValueError(
            f"{event_file_name(match_id)}: {len(raw)} bytes, blob {git_blob_id(raw)}; pinned "
            f"{size} bytes, blob {blob}. Fetch it again: python -m aaaag.sources.idsse_events "
            f"--fetch {match_id}"
        )


def download_pinned(url: str, dest: Path, size: int, blob: str) -> Path:
    """Fetch `url` to `dest`. Wrong bytes raise and leave nothing at `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (pinned URL)
            raw = response.read()
        if len(raw) != size or git_blob_id(raw) != blob:
            raise ValueError(f"{dest.name}: downloaded bytes do not match the pinned blob {blob}")
        partial.write_bytes(raw)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def fetch_events(match_id: str, *, exam: bool = False, force: bool = False) -> Path:
    """The event file of one match from the pinned mirror revision. A valid file is reused."""
    idsse._check_exam(match_id, exam)
    size, blob = EVENT_FILES[match_id]
    path = event_path(match_id)
    if not force and path.is_file():
        raw = path.read_bytes()
        if len(raw) == size and git_blob_id(raw) == blob:
            return path
    url = idsse.MIRROR_URL.format(revision=idsse.REVISION, name=path.name)
    return download_pinned(url, path, size, blob)


# -- parsing --------------------------------------------------------------------------------------


def posix_seconds(stamp: str) -> float:
    """Seconds since the epoch of an ISO 8601 time; a time without an offset is UTC."""
    value = datetime.fromisoformat(stamp.strip())
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


@dataclass(frozen=True)
class PositionsScan:
    """What the event mapping needs from the positions file, without its frames."""

    sections: tuple[str, ...]  # ball sections, in playing order
    times: tuple[np.ndarray, ...]  # per section: (n,) float64 POSIX seconds of each ball frame
    people: tuple[str, ...]  # PersonId of each slot, in idsse.to_bundle's order
    slot_team: np.ndarray  # (S,) int8


def scan_positions(raw: bytes | mmap.mmap, info: idsse.MatchInfo) -> PositionsScan:
    """Ball-frame times per section and the slot order, from one pass over the FrameSet headers."""
    heads = list(idsse._FRAMESET.finditer(raw))
    team_of = {info.home_id: HOME, info.away_id: AWAY}
    times: dict[str, np.ndarray] = {}
    people = set()
    for i, head in enumerate(heads):
        attrs = {k.decode(): v.decode() for k, v in idsse._ATTR.findall(head.group(1))}
        team = attrs.get("TeamId")
        if team == idsse.BALL_TEAM:
            start = head.end()
            stop = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
            stamps = _T.findall(raw, start, stop)
            if not stamps:
                raise ValueError(f"ball FrameSet {attrs} has no frame times (T)")
            times[attrs["GameSection"]] = np.array(
                [posix_seconds(s.decode()) for s in stamps], dtype=np.float64
            )
        elif team in team_of:
            people.add((team_of[team], attrs["PersonId"]))
    sections = tuple(s for s in idsse.SECTIONS if s in times)
    ordered = sorted(people)
    return PositionsScan(
        sections=sections,
        times=tuple(times[s] for s in sections),
        people=tuple(person for _, person in ordered),
        slot_team=np.array([team for team, _ in ordered], dtype=np.int8),
    )


def _chain(element: ET.Element) -> list[tuple[str, dict[str, str]]]:
    """(tag, attributes) from `element` down its first children, as kloppy reads an event."""
    out = []
    node: ET.Element | None = element
    while node is not None:
        out.append((node.tag, dict(node.attrib)))
        node = node[0] if len(node) else None
    return out


def parse_events(raw: bytes) -> tuple[list[dict], dict]:
    """Every live <Event> as {order, time, chain}, in file order, and parse counts.

    A ``Delete`` child cancels its own event's EventId and, when it carries one, the EventId it
    names: every event with a cancelled id is dropped, the marker included.
    """
    root = ET.fromstring(raw)
    elements = list(root.iter("Event"))
    deleted: set[str] = set()
    for ev in elements:
        marker = ev.find("Delete")
        if marker is not None:
            deleted.update(v for v in (ev.get("EventId"), marker.get("EventId")) if v)
    rows, untimed = [], 0
    for order, ev in enumerate(elements):
        if ev.get("EventId") in deleted or ev.find("Delete") is not None:
            continue
        if not len(ev):
            continue
        stamp = ev.get("EventTime")
        if stamp is None:
            untimed += 1
            continue
        rows.append({"order": order, "time": posix_seconds(stamp), "chain": _chain(ev[0])})
    counts = {"events": len(elements), "deleted_ids": len(deleted), "untimed": untimed}
    return rows, counts


def _team(value: str | None, info: idsse.MatchInfo) -> int:
    if value == info.home_id:
        return H
    if value == info.away_id:
        return A
    return NO_TEAM


def _record(order, time, kind, team, detail="", ok=False, goal=False, actor="", recipient=""):
    return {
        "order": order,
        "time": time,
        "kind": kind,
        "team": team,
        "detail": detail,
        "ok": ok,
        "goal": goal,
        "actor": actor,
        "recipient": recipient,
    }


def classify(row: dict, info: idsse.MatchInfo) -> list[dict]:
    """The output records of one parsed event (none, one, or a set_piece plus its out)."""
    chain = row["chain"]
    tag, attrs = chain[0]
    inner_tag, inner = chain[1] if len(chain) > 1 else ("", {})
    order, time = 4 * row["order"], row["time"]
    if tag in RESTARTS:
        team = _team(inner.get("Team") or attrs.get("Team"), info)
        actor = inner.get("Player") or attrs.get("Player", "")
        ok = inner_tag == "Play" and inner.get("Evaluation") in SUCCESS
        outcome = chain[2][0] if inner_tag == "ShotAtGoal" and len(chain) > 2 else ""
        recipient = inner.get("Recipient", "") if ok else ""
        out = [
            _record(order + 1, time, "set_piece", team, tag, ok, outcome == GOAL, actor, recipient)
        ]
        decided = attrs.get("DecisionTimestamp")
        if tag in OUT_RESTARTS and decided:
            other = NO_TEAM if team == NO_TEAM else 1 - team
            out.insert(0, _record(order, posix_seconds(decided), "out", other, tag))
        return out
    if tag == "RefereeBall":
        team = _team(inner.get("Team") or attrs.get("Team"), info)
        actor = inner.get("Player") or attrs.get("Player", "")
        return [_record(order + 1, time, "set_piece", team, tag, actor=actor)]
    if tag == "Play":
        ok = attrs.get("Evaluation") in SUCCESS
        detail = inner_tag or "Pass"
        recipient = attrs.get("Recipient", "") if ok else ""
        return [
            _record(
                order + 1,
                time,
                "pass",
                _team(attrs.get("Team"), info),
                detail,
                ok,
                actor=attrs.get("Player", ""),
                recipient=recipient,
            )
        ]
    if tag == "ShotAtGoal":
        return [
            _record(
                order + 1,
                time,
                "shot",
                _team(attrs.get("Team"), info),
                inner_tag,
                inner_tag == GOAL,
                inner_tag == GOAL,
                attrs.get("Player", ""),
            )
        ]
    if tag == "TacklingGame":
        return [
            _record(
                order + 1,
                time,
                "tackle",
                _team(attrs.get("WinnerTeam"), info),
                attrs.get("WinnerResult", ""),
                attrs.get("PossessionChange") == "true",
                actor=attrs.get("Winner", ""),
            )
        ]
    if tag == "OtherBallAction":
        kind = "clearance" if attrs.get("DefensiveClearance") == "true" else "other_touch"
        team = _team(attrs.get("Team"), info)
        return [_record(order + 1, time, kind, team, tag, actor=attrs.get("Player", ""))]
    if tag in ("BallClaiming", "BallDeflection"):
        team = _team(attrs.get("Team"), info)
        detail = attrs.get("Type") or tag
        return [
            _record(order + 1, time, "other_touch", team, detail, actor=attrs.get("Player", ""))
        ]
    if tag == "Foul":
        team = _team(attrs.get("TeamFouler"), info)
        return [_record(order + 1, time, "foul", team, tag, actor=attrs.get("Fouler", ""))]
    return []


# -- the table ------------------------------------------------------------------------------------


def _section_rows(
    times: np.ndarray, section: np.ndarray, scan: PositionsScan, periods
) -> np.ndarray:
    """(N,) int64 frame of each time on its section's clock; the 25 Hz grid extended past the ends."""
    frames = np.full(len(times), -1, dtype=np.int64)
    for k, (clock, period) in enumerate(zip(scan.times, periods, strict=True)):
        rows = np.flatnonzero(section == k)
        if not len(rows):
            continue
        t = times[rows]
        right = np.clip(np.searchsorted(clock, t), 1, len(clock) - 1) if len(clock) > 1 else None
        if right is None:
            index = np.zeros(len(t), dtype=np.int64)
        else:
            left = right - 1
            nearer_right = np.abs(clock[right] - t) < np.abs(t - clock[left])
            index = np.where(nearer_right, right, left).astype(np.int64)
        before, after = t < clock[0], t > clock[-1]
        index[before] = np.floor((t[before] - clock[0]) * FPS + 0.5).astype(np.int64)
        index[after] = (
            len(clock) - 1 + np.floor((t[after] - clock[-1]) * FPS + 0.5).astype(np.int64)
        )
        frames[rows] = period.start + index
    return frames


def _distance(tracking, frames: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Distance (m, float64) between slot `slots` and the ball at `frames`; inf where either is
    unseen. Shapes broadcast; `frames` must already be valid rows."""
    player = tracking.player_xy[frames, slots].astype(np.float64)
    ball = tracking.ball_xy[frames].astype(np.float64)
    d = np.sqrt(((player - ball) ** 2).sum(axis=-1))
    return np.where(np.isfinite(d), d, np.inf)


def _by_nearness(radius: int) -> np.ndarray:
    """-radius .. radius ordered by |offset|, then offset: argmin ties go to the nearest frame."""
    offsets = np.arange(-radius, radius + 1)
    return offsets[np.lexsort((offsets, np.abs(offsets)))]


def touch_kernel(n: int, frames: np.ndarray, tol: int) -> np.ndarray:
    """(n,) float64 triangular kernel of the distance d to the nearest of `frames`:
    (tol + 1 - d) / (tol + 1) within `tol` frames, 0 beyond. 1 on a touch."""
    if not len(frames):
        return np.zeros(n, dtype=np.float64)
    grid = np.arange(n)
    ref = np.sort(np.asarray(frames, dtype=np.int64))
    right = np.clip(np.searchsorted(ref, grid), 0, len(ref) - 1)
    left = np.clip(right - 1, 0, len(ref) - 1)
    d = np.minimum(np.abs(grid - ref[left]), np.abs(grid - ref[right]))
    return np.clip(tol + 1 - d, 0, None) / (tol + 1)


def half_lag(
    kernel: dict[int, np.ndarray],
    frames: np.ndarray,
    teams: np.ndarray,
    period: Period,
    max_lag: int,
    min_events: int,
) -> dict:
    """The frame lag of one half: the lag in +-max_lag that maximises the summed `kernel[team]`
    ((T,) `touch_kernel` of that team's kinematic touches) over the kicks (`frames`, of `teams`):
    the most kicks on, or nearest to, a touch of their own team. Ties go to the smaller |lag|,
    then the negative one."""
    keep = (frames >= period.start + max_lag) & (frames < period.stop - max_lag)
    keep &= np.isin(teams, (H, A))
    n = int(keep.sum())
    report = {"period": period.index, "n": n, "lag_frames": 0, "on_touch_at_lag": None}
    report["on_touch_at_zero"] = None
    if n < min_events or max_lag <= 0:
        return report
    lags = _by_nearness(max_lag)
    rows = frames[keep][:, None] + lags[None, :]
    score = np.zeros(rows.shape, dtype=np.float64)
    for team in (H, A):
        mine = teams[keep] == team
        score[mine] = kernel[team][rows[mine]]
    best = int(np.argmax(score.sum(axis=0)))
    zero = int(np.flatnonzero(lags == 0)[0])
    report["lag_frames"] = int(lags[best])
    report["on_touch_at_lag"] = float((score[:, best] > 0).mean())
    report["on_touch_at_zero"] = float((score[:, zero] > 0).mean())
    return report


# -- elastic sync ---------------------------------------------------------------------------------


def contact_frames(
    tracking: Tracking, touches: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(frames, kinematic) of the tracked ball contacts, sorted: every kinematic touch
    (`detect_touches` at its defaults; `touches` when already computed), kinematic True, and every
    soft contact (`detect_touches` at `SOFT_THETA` m/s within `CLOSE_M`) more than
    `SYNC_TOL_FRAMES` from all of them, kinematic False."""
    strong = detect_touches(tracking, reach_m=REACH_M)[0] if touches is None else touches
    strong = np.asarray(strong, dtype=np.int64)
    soft = detect_touches(tracking, theta=SOFT_THETA, reach_m=CLOSE_M)[0].astype(np.int64)
    if len(strong) and len(soft):
        soft = soft[_nearest_gap(soft, strong) > SYNC_TOL_FRAMES]
    frames = np.concatenate([strong, soft])
    kinematic = np.concatenate([np.ones(len(strong), dtype=bool), np.zeros(len(soft), dtype=bool)])
    order = np.argsort(frames, kind="stable")
    return frames[order], kinematic[order]


def kick_rise(ball_xy: np.ndarray, fps: float, k: int = KICK_FRAMES) -> np.ndarray:
    """(T,) float64: the ball's speed over the next `k` frames minus its speed over the previous
    `k` (m/s). NaN where the ball is unseen at t - k, t or t + k, or the window leaves the array."""
    p = np.asarray(ball_xy, dtype=np.float64)
    n = len(p)
    out = np.full(n, np.nan)
    if n > 2 * k:
        mid = p[k : n - k]
        after = np.sqrt(((p[2 * k :] - mid) ** 2).sum(axis=1))
        before = np.sqrt(((mid - p[: n - 2 * k]) ** 2).sum(axis=1))
        out[k : n - k] = (after - before) * fps / k
    return out


def _reach(
    tracking: Tracking, frames: np.ndarray, teams: np.ndarray, actors: np.ndarray
) -> np.ndarray:
    """(E, W) float64 distance (m) from the ball at `frames` (E, W) to event e's player: the actor's
    track when `actors[e]` >= 0, else the nearest track of team `teams[e]`, else the nearest track.
    inf where the ball or every candidate track is unseen."""
    out = np.full(frames.shape, np.inf)
    known = np.flatnonzero(actors >= 0)
    if len(known):
        out[known] = _distance(tracking, frames[known], actors[known][:, None])
    rest = np.flatnonzero(actors < 0)
    step = max(1, 4096 // max(1, frames.shape[1]))
    for begin in range(0, len(rest), step):
        rows = rest[begin : begin + step]
        at = frames[rows]
        player = tracking.player_xy[at].astype(np.float64)  # (r, W, S, 2)
        ball = tracking.ball_xy[at].astype(np.float64)[:, :, None, :]
        d = np.sqrt(((player - ball) ** 2).sum(axis=-1))  # (r, W, S)
        team = teams[rows]
        allowed = (tracking.slot_team[None, :] == team[:, None]) | ~np.isin(team, (H, A))[:, None]
        d = np.where(allowed[:, None, :] & np.isfinite(d), d, np.inf)
        out[rows] = d.min(axis=2) if d.shape[2] else np.inf
    return out


def local_lags(
    tracking: Tracking,
    is_contact: np.ndarray,
    frames: np.ndarray,
    teams: np.ndarray,
    actors: np.ndarray,
    period: Period,
    centre: int,
    max_lag: int,
    window: int,
    min_events: int,
) -> np.ndarray:
    """(n,) float64 local lag (frames) at each kick of one period, at raw `frames` (sorted).

    Each kick votes for every lag in `centre` +- `max_lag` with the half-lag kernel (3, 2, 1 for a
    contact 0, 1, 2 frames away, `SYNC_TOL_FRAMES`) of the nearest contact (`is_contact`, inside
    the period) at which its own player (`_reach`) is within `REACH_M`. A kick's local lag is the
    lag with the most votes from the kicks within +-`window` frames of it; ties go to the lag
    nearest `centre`, then the smaller. NaN (no opinion) when that window holds fewer than
    `min_events` kicks or no vote at all. Exact: integer votes."""
    n = len(frames)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    tol = SYNC_TOL_FRAMES
    offsets = centre + np.arange(-max_lag - tol, max_lag + tol + 1)
    grid = frames[:, None] + offsets[None, :]
    inside = (grid >= period.start) & (grid < period.stop)
    grid = np.clip(grid, 0, len(is_contact) - 1)
    hit = inside & is_contact[grid]
    rows, cols = np.nonzero(hit)
    if len(rows):
        d = _reach(tracking, grid[rows, cols][:, None], teams[rows], actors[rows])[:, 0]
        hit[rows, cols] = d <= REACH_M
    width = 2 * max_lag + 1
    votes = np.zeros((n, width), dtype=np.int64)
    for delta in range(-tol, tol + 1):
        weight = tol + 1 - abs(delta)
        np.maximum(votes, hit[:, tol + delta : tol + delta + width] * weight, out=votes)
    total = np.vstack([np.zeros((1, width), dtype=np.int64), np.cumsum(votes, axis=0)])
    lo = np.searchsorted(frames, frames - window, side="left")
    hi = np.searchsorted(frames, frames + window, side="right")
    sums = total[hi] - total[lo]
    lags = centre + np.arange(-max_lag, max_lag + 1)
    prefer = np.lexsort((lags, np.abs(lags - centre)))
    best = prefer[np.argmax(sums[:, prefer], axis=1)]
    ok = (hi - lo >= min_events) & (sums[np.arange(n), best] > 0)
    out[ok] = lags[best[ok]]
    return out


def monotone_assign(options: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """(N,) int64: for events in their logged order, each with (candidate indices, scores) into one
    sorted list of contacts, the candidate each event takes, or -1 (unmatched).

    The choice maximises the summed score of the matched events subject to the matched candidate
    indices increasing strictly with the event order: events keep their order and a contact takes
    at most one event. An unmatched event scores 0, so a candidate scoring 0 or less is never
    taken. A Needleman-Wunsch-type table V[i, j] (the best total of the first i events using the
    first j candidates), filled one event at a time with a running maximum and read back from the
    end; exact ties go to the earlier candidate, then to leaving the later event unmatched.
    Events whose candidate ranges cannot interact (every earlier range ends before every later
    one starts) are solved separately, so the table stays small."""
    n = len(options)
    out = np.full(n, -1, dtype=np.int64)
    pruned = []
    for index, score in options:
        index, score = np.asarray(index, dtype=np.int64), np.asarray(score, dtype=np.float64)
        keep = score > 0
        pruned.append((index[keep], score[keep]))
    live = np.array([i for i, (index, _) in enumerate(pruned) if len(index)], dtype=np.int64)
    if not len(live):
        return out
    first = np.array([pruned[i][0].min() for i in live])
    last = np.array([pruned[i][0].max() + 1 for i in live])
    ends = np.maximum.accumulate(last)
    starts = np.minimum.accumulate(first[::-1])[::-1]
    cuts = np.flatnonzero(ends[:-1] <= starts[1:]) + 1
    for block in np.split(np.arange(len(live)), cuts):
        events = live[block]
        c0 = int(first[block].min())
        m = int(last[block].max()) - c0
        table = np.zeros((len(events) + 1, m + 1))
        for r, i in enumerate(events, start=1):
            index, score = pruned[i]
            gain = np.full(m + 1, -np.inf)
            gain[index - c0 + 1] = table[r - 1, index - c0] + score
            np.maximum.accumulate(np.maximum(table[r - 1], gain), out=table[r])
        r, j = len(events), m
        while r > 0:
            if j > 0 and table[r, j] == table[r, j - 1]:
                j -= 1
            elif table[r, j] == table[r - 1, j]:
                r -= 1
            else:
                out[events[r - 1]] = c0 + j - 1
                r -= 1
                j -= 1
    return out


def _options(
    tracking: Tracking,
    contacts: np.ndarray,
    rise: np.ndarray,
    clock: np.ndarray,
    kinds: np.ndarray,
    teams: np.ndarray,
    actors: np.ndarray,
    period: Period,
    window: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per event: (indices into `contacts`, scores) of the contacts it may take (module docstring,
    step 5): within +-`window` frames of its clock frame, inside `period`, its player within
    `REACH_M`, score above 0."""
    n = len(clock)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0))
    lo = np.searchsorted(contacts, np.maximum(clock - window, period.start), side="left")
    hi = np.searchsorted(contacts, np.minimum(clock + window, period.stop - 1), side="right")
    width = int(np.max(hi - lo, initial=0))
    if width <= 0 or not len(contacts):
        return [empty] * n
    index = lo[:, None] + np.arange(width)[None, :]
    ok = index < hi[:, None]
    index = np.minimum(index, len(contacts) - 1)
    at = contacts[index]
    d = _reach(tracking, at, teams, actors)
    dt = (at - clock[:, None]) / FPS
    score = MATCH_NATS - 0.5 * (dt / TIME_SIGMA_S) ** 2 - 0.5 * (d / REACH_SIGMA_M) ** 2
    kick = np.clip(np.nan_to_num(rise[at] / KICK_RISE_MPS, nan=0.0), -1.0, 1.0)
    score = score + np.where(np.isin(kinds, KICK_KINDS)[:, None], kick, 0.0)
    ok &= (d <= REACH_M) & (score > 0)
    return [(index[r][ok[r]], score[r][ok[r]]) for r in range(n)]


def elastic_sync(
    tracking: Tracking,
    raw: np.ndarray,
    section: np.ndarray,
    half_lags: list[int],
    kind: np.ndarray,
    team: np.ndarray,
    actor: np.ndarray,
    order: np.ndarray,
    *,
    align: bool = True,
    touches: np.ndarray | None = None,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(frames, synced, is_contact, report): steps 3-7 of the module docstring for every event of
    every period. `raw` is each event's frame on the raw clock, `section` its period index (-1:
    none), `half_lags` each period's half lag in frames. `frames` is the contact an on-ball event
    was matched to, else the event's clock frame; `is_contact` (T,) marks the contacts. The local
    lag is searched within +-`max_lag` frames of the half lag (default `ALIGN_MAX_LAG_S`)."""
    periods = tracking.meta.periods
    contacts, kinematic = contact_frames(tracking, touches)
    is_contact = np.zeros(tracking.n_frames, dtype=bool)
    is_contact[contacts] = True
    rise = kick_rise(tracking.ball_xy, FPS)
    if max_lag is None:
        max_lag = int(round(ALIGN_MAX_LAG_S * FPS))
    window = int(round(WINDOW_S * FPS))
    drift_window = int(round(DRIFT_WINDOW_S * FPS))
    frames = raw.copy()
    synced = np.zeros(len(raw), dtype=bool)
    halves = []
    for k, period in enumerate(periods):
        mine = np.flatnonzero(section == k)
        half = int(half_lags[k]) if align else 0
        lag = np.full(len(mine), float(half))
        info = {"period": period.index, "kicks": 0, "voting": 0, "drift_s": None}
        kicks = mine[np.isin(kind[mine], ALIGN_KINDS)]
        kicks = kicks[(actor[kicks] >= 0) | np.isin(team[kicks], (H, A))]
        if align and len(kicks):
            kicks = kicks[np.lexsort((order[kicks], raw[kicks]))]
            local = local_lags(
                tracking,
                is_contact,
                raw[kicks],
                team[kicks],
                actor[kicks],
                period,
                half,
                max_lag,
                drift_window,
                DRIFT_MIN_EVENTS,
            )
            voted = np.isfinite(local)
            info["kicks"], info["voting"] = int(len(kicks)), int(voted.sum())
            if voted.any():
                lag = np.interp(raw[mine], raw[kicks][voted], local[voted])
                drift = (local[voted] - half) / FPS
                info["drift_s"] = [round(float(v), 2) for v in np.percentile(drift, [0, 50, 100])]
        frames[mine] = raw[mine] + np.rint(lag).astype(np.int64)
        on = mine[np.isin(kind[mine], SYNC_KINDS)]
        on = on[np.lexsort((order[on], raw[on]))]
        options = _options(
            tracking, contacts, rise, frames[on], kind[on], team[on], actor[on], period, window
        )
        pick = monotone_assign(options)
        hit = pick >= 0
        frames[on[hit]] = contacts[pick[hit]]
        synced[on[hit]] = True
        info["events"], info["matched"] = int(len(on)), int(hit.sum())
        info["on_kinematic"] = int(kinematic[pick[hit]].sum())
        halves.append(info)
    report = {
        "contacts": {"kinematic": int(kinematic.sum()), "soft": int((~kinematic).sum())},
        "periods": halves,
    }
    return frames, synced, is_contact, report


def build_table(
    raw_events: bytes,
    info: idsse.MatchInfo,
    scan: PositionsScan,
    bundle: MatchBundle,
    *,
    sync: str = SYNC,
    align: bool = True,
    refine_s: float = REFINE_S,
    max_lag_s: float = ALIGN_MAX_LAG_S,
    min_align: int = ALIGN_MIN_EVENTS,
) -> tuple[np.ndarray, dict]:
    """The event table (EVENT_DTYPE) of one match on `bundle`'s frame axis, and a report of
    aggregates. `sync` is "elastic" or "lag" (module docstring); `refine_s` is used by "lag" only.
    Pure: the same inputs give the same bytes."""
    if sync not in SYNCS:
        raise ValueError(f"sync must be one of {SYNCS}, got {sync!r}")
    tracking = bundle.tracking
    periods = bundle.meta.periods
    if len(scan.sections) != len(periods):
        raise ValueError(f"{len(scan.sections)} ball sections but {len(periods)} periods")
    for clock, period in zip(scan.times, periods, strict=True):
        if len(clock) != period.length:
            raise ValueError(f"{len(clock)} ball-frame times for a period of {period.length}")
        if len(clock) > 1 and not (np.diff(clock) >= 0).all():
            raise ValueError("ball-frame times are not in order")
    if not np.array_equal(scan.slot_team, tracking.slot_team):
        raise ValueError("the positions file's players do not match the bundle's slots")
    if refine_s < 0 or max_lag_s < 0:
        raise ValueError("refine_s and max_lag_s must be >= 0")
    slot_of = {person: slot for slot, person in enumerate(scan.people)}

    parsed, counts = parse_events(raw_events)
    ignored: dict[str, int] = {}
    records = []
    for row in parsed:
        got = classify(row, info)
        if not got:
            tag = row["chain"][0][0]
            ignored[tag] = ignored.get(tag, 0) + 1
        records.extend(got)
    n = len(records)
    times = np.array([r["time"] for r in records], dtype=np.float64)
    max_lag = int(round(max_lag_s * FPS))
    section = np.full(n, -1, dtype=np.int64)
    for k, clock in enumerate(scan.times):
        inside = (times >= clock[0] - max_lag_s) & (times <= clock[-1] + max_lag_s)
        section[(section < 0) & inside] = k
    frames = _section_rows(times, section, scan, periods)
    kind = np.array([r["kind"] for r in records], dtype="<U16")
    team = np.array([r["team"] for r in records], dtype=np.int64)
    actor = np.array([slot_of.get(r["actor"], -1) for r in records], dtype=np.int64)
    starts = np.array([periods[k].start if k >= 0 else 0 for k in section], dtype=np.int64)
    stops = np.array([periods[k].stop if k >= 0 else 0 for k in section], dtype=np.int64)

    # Kinematic touches of the clean ball: what a kick or a reception looks like in the tracking.
    T = tracking.n_frames
    touch_frames_, touch_teams = detect_touches(tracking, reach_m=REACH_M)
    is_touch = np.zeros(T, dtype=bool)
    is_touch[touch_frames_] = True
    kernel = {t: touch_kernel(T, touch_frames_[touch_teams == t], SYNC_TOL_FRAMES) for t in (H, A)}

    raw_frames = frames.copy()
    lags = []
    for k, period in enumerate(periods):
        mine = section == k
        kicks = mine & np.isin(kind, ALIGN_KINDS) if align else np.zeros(n, dtype=bool)
        got = half_lag(kernel, frames[kicks], team[kicks], period, max_lag, min_align)
        frames[mine] += got["lag_frames"]
        lags.append({**got, "lag_s": got["lag_frames"] / FPS})
    lagged = frames.copy()
    order = np.array([r["order"] for r in records], dtype=np.int64)
    synced = np.zeros(n, dtype=bool)
    sync_report: dict = {}

    if sync == "lag":
        inside = (section >= 0) & (frames >= starts) & (frames < stops)
        radius = int(round(refine_s * FPS))
        moved = 0
        candidates = np.flatnonzero(inside & (actor >= 0) & np.isin(kind, REFINE_KINDS))
        if radius > 0 and len(candidates):
            offsets = _by_nearness(radius)
            window = frames[candidates][:, None] + offsets[None, :]
            valid = (window >= starts[candidates][:, None]) & (window < stops[candidates][:, None])
            window = np.clip(window, 0, T - 1)
            d = _distance(tracking, window, actor[candidates][:, None])
            ok = valid & is_touch[window] & (d <= REACH_M)
            found = ok.any(axis=1)
            target = window[np.arange(len(candidates)), np.argmax(ok, axis=1)]
            moved = int((found & (target != frames[candidates])).sum())
            frames[candidates[found]] = target[found]
            synced[candidates[found]] = True
        sync_report["refine"] = {
            "radius_frames": radius,
            "candidates": int(len(candidates)),
            "moved": moved,
        }
        on_contact = is_touch
    else:
        frames, synced, on_contact, sync_report["elastic"] = elastic_sync(
            tracking,
            raw_frames,
            section,
            [lag["lag_frames"] for lag in lags],
            kind,
            team,
            actor,
            order,
            align=align,
            touches=touch_frames_,
            max_lag=max_lag,
        )
        inside = (section >= 0) & (frames >= starts) & (frames < stops)

    # Receptions: after each completed pass, before the next logged event of the same period.
    recipient = np.array([slot_of.get(r["recipient"], -1) for r in records], dtype=np.int64)
    completed = np.flatnonzero(inside & (recipient >= 0) & np.isin(kind, ("pass", "set_piece")))
    receptions = []
    live = np.flatnonzero(inside)
    sequence = live[np.lexsort((order[live], frames[live]))]
    position = {int(i): p for p, i in enumerate(sequence)}
    for i in completed:
        p = position[int(i)]
        first = frames[i] + 1
        last = stops[i] - 1
        if p + 1 < len(sequence):
            last = min(last, frames[sequence[p + 1]] - 1)
        if last < first:
            continue
        rows = np.arange(first, last + 1)
        rows = rows[on_contact[rows]]
        if not len(rows):
            continue
        d = _distance(tracking, rows, np.full(len(rows), recipient[i]))
        reached = np.flatnonzero(d <= REACH_M)
        if len(reached):
            receptions.append((int(i), int(rows[reached[0]])))

    period_no = np.array([periods[k].index if k >= 0 else 0 for k in section], dtype=np.int64)
    table = np.zeros(n + len(receptions), dtype=EVENT_DTYPE)
    table["frame"][:n] = np.where(inside, frames, -1)
    table["lag_frame"][:n] = np.where(inside, lagged, -1)
    table["team"][:n] = [r["team"] for r in records]
    table["kind"][:n] = kind
    table["detail"][:n] = [r["detail"] for r in records]
    table["ok"][:n] = [r["ok"] for r in records]
    table["goal"][:n] = [r["goal"] for r in records]
    table["period"][:n] = np.where(inside, period_no, 0)
    table["actor"][:n] = actor
    table["synced"][:n] = synced & inside
    sort_order = np.concatenate([order, np.zeros(len(receptions), dtype=np.int64)])
    for j, (i, frame) in enumerate(receptions):
        row = n + j
        table[row] = (frame, table["team"][i], "reception", "", True, False, 0, frame, 0, True)
        table["period"][row] = table["period"][i]
        table["actor"][row] = recipient[i]
        sort_order[row] = order[i] + 2
    outside = table["frame"] < 0
    table = table[np.lexsort((sort_order, table["frame"], outside))]

    out_counts: dict[str, int] = {}
    for name in KINDS:
        hit = int(((table["kind"] == name) & (table["frame"] < 0)).sum())
        if hit:
            out_counts[name] = hit
    report = {
        **counts,
        "records": int(n),
        "ignored_by_tag": dict(sorted(ignored.items())),
        "outside_periods": out_counts,
        "lags": lags,
        "sync": sync,
        **sync_report,
        "receptions": {"completed_passes": int(len(completed)), "found": len(receptions)},
    }
    return table, report


# -- the cached API -------------------------------------------------------------------------------


def _atomic_write(path: Path, write) -> None:
    """`write(handle)` into a private temporary file, then rename it onto `path`: parallel training
    jobs that build the same entry never read a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        with open(partial, "wb") as handle:
            write(handle)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _save(store: Path, sidecar: dict, **arrays: np.ndarray) -> None:
    """The arrays first, the JSON sidecar second, each atomically: a reader that finds the sidecar
    finds the arrays it describes."""
    _atomic_write(store, lambda handle: np.savez(handle, **arrays))
    text = json.dumps(sidecar, indent=1).encode("utf-8")
    _atomic_write(store.with_suffix(".json"), lambda handle: handle.write(text))


def _gap_frames(bundle: MatchBundle) -> int:
    periods = bundle.meta.periods
    gaps = {b.start - a.stop for a, b in zip(periods[:-1], periods[1:], strict=True)}
    if len(gaps) > 1:
        raise ValueError(f"{bundle.meta.match_id}: periods are not evenly spaced ({sorted(gaps)})")
    return gaps.pop() if gaps else 25


def _scan_cached(match_id: str, info: idsse.MatchInfo) -> PositionsScan:
    path = idsse.file_paths(match_id)["positions"]
    stat = path.stat()
    stamp = {"version": SCAN_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    store = cache_dir() / "idsse_events" / f"{match_id}-scan-v{SCAN_VERSION}.npz"
    sidecar = store.with_suffix(".json")
    if store.is_file() and sidecar.is_file():
        if json.loads(sidecar.read_text(encoding="utf-8")).get("stamp") == stamp:
            with np.load(store, allow_pickle=False) as saved:
                lengths = saved["lengths"]
                edges = np.concatenate(([0], np.cumsum(lengths)))
                return PositionsScan(
                    sections=tuple(str(s) for s in saved["sections"]),
                    times=tuple(
                        saved["times"][edges[k] : edges[k + 1]] for k in range(len(lengths))
                    ),
                    people=tuple(str(p) for p in saved["people"]),
                    slot_team=saved["slot_team"],
                )
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        scan = scan_positions(raw, info)
    _save(
        store,
        {"stamp": stamp},
        sections=np.array(scan.sections, dtype="<U32"),
        lengths=np.array([len(t) for t in scan.times], dtype=np.int64),
        times=np.concatenate(scan.times) if scan.times else np.zeros(0),
        people=np.array(scan.people, dtype="<U64"),
        slot_team=scan.slot_team,
    )
    return scan


def event_table(
    match_id: str,
    *,
    bundle: MatchBundle | None = None,
    exam: bool = False,
    sync: str = SYNC,
    align: bool = True,
    refine_s: float = REFINE_S,
    gap_frames: int | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, dict]:
    """(EVENT_DTYPE table, report) of one IDSSE match on the `idsse` frame axis, synchronised by
    `sync` ("elastic", the default, or "lag"; `refine_s` is a "lag" parameter only).

    `bundle`, when given, must be that match's clean `idsse` bundle; the sealed exam is then
    allowed, because loading that bundle already needed `exam=True`. Without it, the exam needs
    `exam=True` here too, and the bundle is loaded only when the cache misses. The cache key is the
    pinned file hashes plus the parameters, so a cache built on one machine serves another.
    """
    if sync not in SYNCS:
        raise ValueError(f"sync must be one of {SYNCS}, got {sync!r}")
    if bundle is not None:
        if bundle.meta.source != "idsse" or bundle.meta.match_id != match_id:
            raise ValueError(
                f"event_table({match_id!r}) needs that match's clean idsse bundle, got "
                f"{bundle.meta.source}/{bundle.meta.match_id}"
            )
        gap_frames = _gap_frames(bundle)
    else:
        idsse._check_exam(match_id, exam)
    gap = 25 if gap_frames is None else int(gap_frames)
    stamp = {
        "version": VERSION,
        "events_blob": EVENT_FILES[match_id][1],
        "positions_sha256": idsse.MATCHES[match_id][2],
        "gap_frames": gap,
        "sync": sync,
        "align": bool(align),
        "refine_s": float(refine_s) if sync == "lag" else None,
    }
    key = json.dumps(stamp, sort_keys=True)
    with _LOCK:
        table, report = _table_locked(match_id, stamp, key, bundle, exam, use_cache)
    if bundle is not None and table["frame"].max(initial=-1) >= bundle.meta.n_frames:
        raise ValueError(f"{match_id}: an event frame is past the bundle's end")
    return table.copy(), json.loads(json.dumps(report))


def _table_locked(
    match_id: str,
    stamp: dict,
    key: str,
    bundle: MatchBundle | None,
    exam: bool,
    use_cache: bool,
) -> tuple[np.ndarray, dict]:
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    store = cache_dir() / "idsse_events" / f"{match_id}-v{VERSION}-{digest}.npz"
    if (match_id, key) in _MEMO:
        return _MEMO[(match_id, key)]
    if use_cache and store.is_file() and store.with_suffix(".json").is_file():
        saved = json.loads(store.with_suffix(".json").read_text(encoding="utf-8"))
        if saved.get("stamp") == stamp:
            with np.load(store, allow_pickle=False) as arrays:
                table = arrays["table"]
            _MEMO[(match_id, key)] = (table, saved["report"])
            return table, saved["report"]
    sync, align, gap = stamp["sync"], stamp["align"], stamp["gap_frames"]
    refine_s = REFINE_S if stamp["refine_s"] is None else stamp["refine_s"]
    if bundle is None:
        bundle = idsse.load_idsse(match_id=match_id, gap_frames=gap, exam=exam)
    paths = idsse.file_paths(match_id)
    path = event_path(match_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"IDSSE event file not found: {path}\nFetch it with: "
            f"python -m aaaag.sources.idsse_events --fetch {match_id}"
        )
    raw = path.read_bytes()
    check_pin(match_id, raw)
    info = idsse.read_match_info(paths["meta"])
    scan = _scan_cached(match_id, info)
    table, report = build_table(raw, info, scan, bundle, sync=sync, align=align, refine_s=refine_s)
    if table["frame"].max(initial=-1) >= bundle.meta.n_frames:
        raise ValueError(f"{match_id}: an event frame is past the bundle's end")
    if use_cache:
        _save(store, {"stamp": stamp, "report": report}, table=table)
    _MEMO[(match_id, key)] = (table, report)
    return table, report


def as_touch_events(table: np.ndarray) -> np.ndarray:
    """The rows of an event table inside the periods and credited to a team, as TOUCH_DTYPE, in
    table order (so sorted by frame). Identical (frame, team, kind) rows are merged, keeping the
    first: two events that refinement snapped onto one touch are one touch. A row without a team
    (a dropped ball, an event naming no club of the match) is left out, as in
    `aaaag.sources.pff_events`; `event_table` keeps it."""
    keep = (table["frame"] >= 0) & np.isin(table["team"], (H, A))
    rows = table[keep]
    out = np.zeros(len(rows), dtype=TOUCH_DTYPE)
    out["frame"] = rows["frame"]
    out["team"] = rows["team"]
    out["kind"] = rows["kind"]
    if len(out):
        _, first = np.unique(out, return_index=True)
        out = out[np.sort(first)]
    return out


def touch_events(
    match_id: str,
    *,
    bundle: MatchBundle | None = None,
    exam: bool = False,
    sync: str = SYNC,
    align: bool = True,
    refine_s: float = REFINE_S,
) -> np.ndarray:
    """(N,) TOUCH_DTYPE `[("frame", "<i8"), ("team", "<i1"), ("kind", "<U16")]`, sorted by frame:
    frame on the `idsse` frame axis (inside a period), team HOME or AWAY (`contracts`), kind one
    of KINDS; identical rows merged (`as_touch_events`). The same contract as
    `aaaag.sources.pff_events.touch_events`. See the module docstring for the frame
    synchronisation (`sync`, default "elastic") and `event_table` for `bundle` and `exam`."""
    table, _ = event_table(
        match_id, bundle=bundle, exam=exam, sync=sync, align=align, refine_s=refine_s
    )
    return as_touch_events(table)


# -- generic helpers on TOUCH_DTYPE arrays (any source) ------------------------------------------


def touch_frames(events: np.ndarray) -> np.ndarray:
    """Sorted unique frames of the on-ball events (TOUCH_KINDS), any team."""
    on = np.isin(events["kind"], TOUCH_KINDS) & (events["frame"] >= 0)
    return np.unique(events["frame"][on]).astype(np.int64)


def possession_changes(events: np.ndarray, periods) -> np.ndarray:
    """Sorted frames where possession changes in open play, by the events alone.

    Inside each period, take the on-ball events of a known team in frame order (ties keep the
    input order). A change is a consecutive pair whose teams differ; it sits on the frame of the
    second event, the new team's first touch, and does not count when a foul, an out or a set
    piece (any team) lies in (first frame, second frame]: that change goes through a stoppage.
    """
    frames = events["frame"]
    on = np.isin(events["kind"], TOUCH_KINDS) & np.isin(events["team"], (H, A)) & (frames >= 0)
    stops = np.sort(frames[np.isin(events["kind"], STOP_KINDS) & (frames >= 0)])
    out = []
    for period in periods:
        mine = np.flatnonzero(on & (frames >= period.start) & (frames < period.stop))
        mine = mine[np.argsort(frames[mine], kind="stable")]
        if len(mine) < 2:
            continue
        f, team = frames[mine], events["team"][mine]
        prev, nxt = f[:-1], f[1:]
        crossed = np.searchsorted(stops, nxt, side="right") - np.searchsorted(
            stops, prev, side="right"
        )
        change = (team[:-1] != team[1:]) & (crossed == 0)
        out.append(nxt[change])
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(out)).astype(np.int64)


def control_changes(table: np.ndarray, periods) -> np.ndarray:
    """Sorted frames where the team in CONTROL changes, by the events alone (the DFL possession
    rule: the label changes when the other team gains control).

    Controlled actions are a completed pass (its start), a reception (the target frame of a
    completed pass) and a shot, each of a known team and inside a period. Tackles, clearances,
    other touches (blocks, deflections, claims), set pieces, fouls, outs and failed passes never
    count. Per period, in frame order (ties keep table order), a change sits on the first
    controlled action of the team that differs from the previous controlled action's team."""
    frames = table["frame"]
    kind = table["kind"]
    controlled = ((kind == "pass") & table["ok"]) | (kind == "reception") | (kind == "shot")
    on = controlled & np.isin(table["team"], (H, A)) & (frames >= 0)
    out = []
    for period in periods:
        mine = np.flatnonzero(on & (frames >= period.start) & (frames < period.stop))
        mine = mine[np.argsort(frames[mine], kind="stable")]
        if len(mine) < 2:
            continue
        team = table["team"][mine]
        out.append(frames[mine][1:][team[:-1] != team[1:]])
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(out)).astype(np.int64)


def _signed_nearest(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signed offset (nearest reference frame minus frame) per frame; empty without a reference."""
    if not len(reference) or not len(frames):
        return np.zeros(0, dtype=np.int64)
    reference = np.sort(reference)
    right = np.clip(np.searchsorted(reference, frames), 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    a, b = reference[left] - frames, reference[right] - frames
    return np.where(np.abs(a) <= np.abs(b), a, b).astype(np.int64)


CONTROL_WINDOWS = (1, 3, 5, 10)


def control_stats(changes: np.ndarray, label_changes: np.ndarray) -> dict:
    """Aggregates comparing event-derived and label possession changes: the counts, per window the
    numbers of label changes with an event change within it and vice versa, and the signed offsets
    (event change minus label change) of each label change to its nearest event change."""
    offset = _signed_nearest(label_changes, changes)
    back = _nearest_gap(changes, label_changes)
    return {
        "events": int(len(changes)),
        "labels": int(len(label_changes)),
        "label_hit": [int((np.abs(offset) <= w).sum()) for w in CONTROL_WINDOWS],
        "event_hit": [int((back <= w).sum()) for w in CONTROL_WINDOWS],
        "offset": offset,
    }


def pool_control(stats: list[dict]) -> dict:
    """`control_stats` of several matches as one."""
    return {
        "events": sum(s["events"] for s in stats),
        "labels": sum(s["labels"] for s in stats),
        "label_hit": [sum(s["label_hit"][i] for s in stats) for i in range(len(CONTROL_WINDOWS))],
        "event_hit": [sum(s["event_hit"][i] for s in stats) for i in range(len(CONTROL_WINDOWS))],
        "offset": np.concatenate([s["offset"] for s in stats]),
    }


def print_control(name: str, s: dict) -> None:
    lab = " ".join(
        f"{w}:{_share(h, s['labels'])}"
        for w, h in zip(CONTROL_WINDOWS, s["label_hit"], strict=True)
    )
    ev = " ".join(
        f"{w}:{_share(h, s['events'])}"
        for w, h in zip(CONTROL_WINDOWS, s["event_hit"], strict=True)
    )
    off = s["offset"]
    if len(off):
        p10, q25, q50, q75, p90 = np.percentile(off, [10, 25, 50, 75, 90])
        dist = f"median {q50:+.1f} IQR {q25:+.1f} to {q75:+.1f} p10 {p10:+.1f} p90 {p90:+.1f}"
    else:
        dist = "n/a"
    print(
        f"  control-rule ({name}): changes events {s['events']} vs labels {s['labels']}; label "
        f"changes within w frames of an event change [{lab}]; event changes within w of a label "
        f"change [{ev}]; signed offset (event minus label, frames) {dist}"
    )


def probe_control(match_id: str, *, exam: bool = False) -> dict:
    """Print `control_stats` of the control-rule changes against the label changes, with the
    elastic and the lag sync; returns both for pooling."""
    bundle = idsse.load_idsse(match_id=match_id, exam=exam)
    labels = bundle.labels
    pair = np.flatnonzero(
        np.isin(labels[:-1], (H, A)) & np.isin(labels[1:], (H, A)) & (labels[:-1] != labels[1:])
    )
    out = {}
    for sync in SYNCS:
        table, _ = event_table(match_id, bundle=bundle, sync=sync)
        changes = control_changes(table, bundle.meta.periods)
        out[sync] = control_stats(changes, pair + 1)
        print(match_id, end="")
        print_control(sync, out[sync])
    return out


# -- probe ----------------------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> str:
    if not len(values):
        return "n 0"
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return f"n {len(values)}, median {q50:.1f} (IQR {q25:.1f} to {q75:.1f})"


def _nearest_gap(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """|frame - nearest reference frame| per frame; empty when there is no reference."""
    if not len(reference) or not len(frames):
        return np.zeros(0, dtype=np.int64)
    reference = np.sort(reference)
    right = np.clip(np.searchsorted(reference, frames), 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    return np.minimum(np.abs(frames - reference[left]), np.abs(frames - reference[right]))


def sync_stats(table: np.ndarray, contacts: np.ndarray, labels: np.ndarray) -> dict:
    """Aggregates of one event table: the logged on-ball events inside the periods (`SYNC_KINDS`)
    and how many the sync put on a contact (`synced`); the on-ball events, receptions included,
    within 2 frames of a tracked contact (`contacts`); on live label frames, how often the label
    team is the event team; and each matched event's shift from its lag-corrected frame (frames,
    an array: print only its quantiles)."""
    inside = table["frame"] >= 0
    movable = inside & np.isin(table["kind"], SYNC_KINDS)
    matched = movable & table["synced"]
    on = inside & np.isin(table["kind"], TOUCH_KINDS)
    gap = _nearest_gap(table["frame"][on], contacts)
    events = as_touch_events(table)
    events = events[np.isin(events["kind"], TOUCH_KINDS)]
    seen = labels[events["frame"]]
    live = np.isin(seen, (H, A))
    return {
        "movable": int(movable.sum()),
        "matched": int(matched.sum()),
        "on_ball": int(on.sum()),
        "near": int((gap <= 2).sum()),
        "live": int(live.sum()),
        "agree": int((seen[live] == events["team"][live]).sum()),
        "shift": (table["frame"] - table["lag_frame"])[matched].astype(np.int64),
    }


def pool_stats(stats: list[dict]) -> dict:
    """`sync_stats` of several matches as one: counts summed, shifts concatenated."""
    out = {key: sum(s[key] for s in stats) for key in stats[0] if key != "shift"}
    out["shift"] = np.concatenate([s["shift"] for s in stats])
    return out


def _share(num: int, den: int) -> str:
    return f"{num / den:.3f}" if den else "n/a"


def print_stats(name: str, s: dict) -> None:
    print(
        f"  {name:<8} matched {_share(s['matched'], s['movable'])} of {s['movable']}; within 2 "
        f"frames of a contact {_share(s['near'], s['on_ball'])}; label team = event team "
        f"{_share(s['agree'], s['live'])}; shift from the lag frame (matched), frames: "
        f"{_quantiles(s['shift'])}"
    )


def probe(match_id: str, *, exam: bool = False) -> dict:
    """Print aggregates for one match: counts per kind and team, frames outside the periods, the
    half lags, the elastic drift, the gap to the nearest tracked ball-player contact, and, per
    sync method, `sync_stats` and agreement with the labels. Returns the `sync_stats` per method
    ("lag", "elastic") for pooling."""
    raw = event_path(match_id).read_bytes()
    check_pin(match_id, raw)
    bundle = idsse.load_idsse(match_id=match_id, exam=exam)
    table, report = event_table(match_id, bundle=bundle)
    refined, lag_report = event_table(match_id, bundle=bundle, sync="lag")
    aligned, _ = event_table(match_id, bundle=bundle, sync="lag", refine_s=0.0)
    unaligned, _ = event_table(match_id, bundle=bundle, sync="lag", align=False, refine_s=0.0)
    print(f"{match_id}: {len(raw)} bytes, blob ok, sha256 {hashlib.sha256(raw).hexdigest()}")
    print(
        f"  events {report['events']}, deleted ids {report['deleted_ids']}, untimed "
        f"{report['untimed']}, output records {report['records']} + receptions "
        f"{report['receptions']['found']} of {report['receptions']['completed_passes']} "
        "completed passes (elastic)"
    )
    print(f"  ignored by tag: {report['ignored_by_tag']}")
    print("  kind            home   away   none  outside   (elastic)")
    for kind in KINDS:
        rows = table[table["kind"] == kind]
        inside = rows[rows["frame"] >= 0]
        cells = [int((inside["team"] == t).sum()) for t in (H, A, NO_TEAM)]
        print(
            f"  {kind:<14}{cells[0]:>6} {cells[1]:>6} {cells[2]:>6} {int((rows['frame'] < 0).sum()):>8}"
        )
    for lag in report["lags"]:
        at, zero = lag["on_touch_at_lag"], lag["on_touch_at_zero"]
        detail = "" if at is None else f", kicks on a touch {at:.3f} (at lag 0: {zero:.3f})"
        edge = " AT THE SEARCH EDGE" if abs(lag["lag_frames"]) >= ALIGN_MAX_LAG_S * FPS else ""
        print(
            f"  period {lag['period']}: lag {lag['lag_s']:+.2f} s from {lag['n']} kicks{detail}{edge}"
        )
    refine = lag_report["refine"]
    print(
        f"  lag sync, refined: {refine['moved']} of {refine['candidates']} moved "
        f"(+-{refine['radius_frames']} frames)"
    )
    elastic = report["elastic"]
    found = elastic["contacts"]
    print(f"  elastic sync: contacts {found['kinematic']} kinematic + {found['soft']} soft")
    for half in elastic["periods"]:
        drift = half["drift_s"]
        if drift is None:
            span = "no local clock"
        else:
            span = f"drift vs the half lag {drift[0]:+.2f} to {drift[2]:+.2f} s"
            span += f" (median {drift[1]:+.2f})"
        print(
            f"  period {half['period']}: matched {half['matched']} of {half['events']} "
            f"({half['on_kinematic']} on a kinematic touch); local lag from {half['voting']} of "
            f"{half['kicks']} kicks, {span}"
        )
    contacts, _ = detect_touches(bundle.tracking)
    versions = (
        ("raw clock", unaligned),
        ("half lag", aligned),
        ("lag + refine", refined),
        ("elastic", table),
    )
    for name, version in versions:
        on = version[np.isin(version["kind"], TOUCH_KINDS) & (version["frame"] >= 0)]
        gap = _nearest_gap(on["frame"], contacts)
        share = float((gap <= 2).mean()) if len(gap) else float("nan")
        print(
            f"  gap to the nearest tracked contact ({name}), frames: {_quantiles(gap)}; "
            f"within 2 frames {share:.3f}"
        )
    labels = bundle.labels
    stats = {
        "lag": sync_stats(refined, contacts, labels),
        "elastic": sync_stats(table, contacts, labels),
    }
    for name, s in stats.items():
        print_stats(name, s)
    events = as_touch_events(table)
    changes = possession_changes(events, bundle.meta.periods)
    pair = np.flatnonzero(
        np.isin(labels[:-1], (H, A)) & np.isin(labels[1:], (H, A)) & (labels[:-1] != labels[1:])
    )
    label_changes = pair + 1
    if len(changes) and len(label_changes):
        hit = float((_nearest_gap(changes, label_changes) <= 3).mean())
        back = float((_nearest_gap(label_changes, changes) <= 3).mean())
        print(
            f"  open-play possession changes (elastic): events {len(changes)}, labels "
            f"{len(label_changes)}; event changes within 3 frames of a label change {hit:.3f}, "
            f"label changes within 3 frames of an event change {back:.3f}"
        )
    else:
        print(f"  open-play possession changes: events {len(changes)}, labels {len(label_changes)}")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("matches", nargs="*", help="match ids (default: the five dev matches)")
    parser.add_argument("--fetch", action="store_true", help="download and verify the event files")
    parser.add_argument("--probe", action="store_true", help="print aggregates per match")
    parser.add_argument(
        "--probe-control",
        action="store_true",
        help="control-rule possession changes against the label changes (aggregates)",
    )
    parser.add_argument("--exam", action="store_true", help="allow the sealed exam matches")
    args = parser.parse_args(argv)
    if not (args.fetch or args.probe or args.probe_control):
        parser.error("pass --fetch, --probe and/or --probe-control")
    matches = args.matches or list(idsse.DEV_MATCHES)
    pooled: dict[str, list[dict]] = {}
    control: dict[str, list[dict]] = {}
    for match_id in matches:
        if args.fetch:
            path = fetch_events(match_id, exam=args.exam)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{match_id}: ok ({path.stat().st_size} bytes, sha256 {digest})")
        if args.probe:
            for name, stats in probe(match_id, exam=args.exam).items():
                pooled.setdefault(name, []).append(stats)
        if args.probe_control:
            for name, stats in probe_control(match_id, exam=args.exam).items():
                control.setdefault(name, []).append(stats)
    if pooled:
        print(f"pooled over {len(matches)} matches:")
        for name, stats in pooled.items():
            print_stats(name, pool_stats(stats))
    if control:
        print(f"control-rule pooled over {len(matches)} matches:")
        for name, stats in control.items():
            print_control(name, pool_control(stats))


if __name__ == "__main__":
    main(sys.argv[1:])
