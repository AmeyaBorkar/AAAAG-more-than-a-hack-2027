"""PFF touch and pass timing from the event file: `touch_events(game_id)`.

Supervises the touch head during PFF pretraining with PFF's own event data instead of
touches detected in the tracking. Nothing is fetched; it reads `Event Data/{id}.json` and
the tracking rows through the paths of `aaaag.sources.pff`, and only for the 48 train-split
games. The 16 test games are refused (`allow_test=True` exists for label-free evaluation
tooling, never for training).

Output: a structured array, sorted by frame, dtype `TOUCH_EVENT_DTYPE`
`[("frame", "<i8"), ("team", "<i1"), ("kind", "<U16")]`. `frame` is an index on the reader's
resampled 25 fps axis (the same axis as `load_pff(...).tracking`, halves adjacent). `team` is
`contracts.HOME` or `AWAY`. `kind` is one of `KINDS`. Identical rows are merged.

Event-to-frame alignment. The event file's clock is never trusted. Every tracking row carries
`game_event_id` and `possession_event_id`; a possession event happens at the FIRST kept
tracking row that carries its id (the moment PFF opened the event, i.e. the contact for a
pass, shot, clearance or challenge). That native row has a `periodElapsedTime`; it is placed
on the resampled grid exactly as the reader does it: per period the grid is
t0 + k * 40 ms from the first to the last kept row, and the event takes the nearest k
(an exact tie goes to the later frame). Periods are adjacent in the order 1, 2. Events whose
period is not 1 or 2, or whose id never appears on a kept tracking row, are dropped and
counted, never guessed.

Kinds, from the record (names are matched case-insensitively and ignoring `_`, so
`possessionEventType` and `possession_event_type` are the same, on the record or one level
down):
- possession type PA (pass), CR (cross) -> `pass`; a completed one (`passOutcomeType` C or
  COMPLETE, or a receiver id when the outcome is absent) also emits a `reception` of the
  same team at the LAST tracking row carrying that possession-event id;
- SH -> `shot`; CL -> `clearance`; CH and TC -> `tackle` (PFF's challenge/tackle codes);
  any other non-empty type (BC ball carry, RE rebound, ...) -> `other_touch`;
- a game event with a set-piece type that is not open play, or a kick-off type -> `set_piece`
  at that game event's first row; game event OUT -> `out` at its first row;
- a non-empty `fouls` / `foulType` field -> `foul` at the possession event frame if there
  is one, else the game event frame.
Team is the `homeTeam` flag of the record (true = HOME). A record with no readable flag is
dropped and counted.

Where it fails: the event-file schema is not published here, so field names are matched
loosely and unknown possession types fall into `other_touch`. A reception sits at the end of
the pass event span, which is the ball arriving as PFF coded it, not a measured foot contact.
Run `--probe-events` first and read the per-kind counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ..contracts import AWAY, HOME
from ..paths import cache_dir
from . import pff

EVENTS_VERSION = "1"
TOUCH_EVENT_DTYPE = np.dtype([("frame", "<i8"), ("team", "<i1"), ("kind", "<U16")])
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
_POSSESSION_KIND = {
    "pa": "pass",
    "pass": "pass",
    "cr": "pass",
    "cross": "pass",
    "sh": "shot",
    "shot": "shot",
    "cl": "clearance",
    "clearance": "clearance",
    "ch": "tackle",
    "tc": "tackle",
    "challenge": "tackle",
    "tackle": "tackle",
}
_NO_POSSESSION = frozenset({"", "none", "null", "nan", "nonevent"})
_KICKOFFS = frozenset({"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"})
_COMPLETE = frozenset({"c", "complete", "completed"})
_POSS_ID_KEYS = ("possession_event_id", "possessionEventId")
_NORM = re.compile(r"[^a-z0-9]")


def _norm(name) -> str:
    return _NORM.sub("", str(name).casefold())


def _flat(record: dict) -> dict[str, object]:
    """Normalised key -> value over the record and one nested level. Outer scalars win."""
    out: dict[str, object] = {}
    nested = []
    for key, value in record.items():
        items = value if isinstance(value, list) else [value]
        inner = next((item for item in items if isinstance(item, dict)), None)
        if inner is None:
            out.setdefault(_norm(key), value)
        else:
            nested.append(inner)
    for inner in nested:
        for key, value in inner.items():
            out.setdefault(_norm(key), value)
    return out


def _text(value) -> str:
    if value is None or isinstance(value, bool | dict | list):
        return ""
    return str(value).strip()


def _present(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, list | dict | str):
        return len(value) > 0 and str(value).strip().casefold() not in {"null", "none"}
    return True


# --------------------------------------------------------------------------- scan


def _scan_rows(path: Path) -> dict:
    """First/last kept native row (period, time_ms) per event id, and each period's span."""
    first: dict[tuple[str, str], tuple[int, int]] = {}
    last: dict[tuple[str, str], tuple[int, int]] = {}
    span: dict[int, list[int]] = {}
    with pff._open_text(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON") from exc
            period = pff._period(row.get("period")) if isinstance(row, dict) else None
            if period not in (1, 2):
                continue
            try:
                ms = int(np.rint(pff._elapsed(row.get("periodElapsedTime")) * 1000.0))
            except ValueError:
                continue
            bounds = span.setdefault(period, [ms, ms])
            bounds[0], bounds[1] = min(bounds[0], ms), max(bounds[1], ms)
            ids = [("game", pff._key(row.get("game_event_id")))]
            for key in _POSS_ID_KEYS:
                if row.get(key) is not None:
                    ids.append(("poss", pff._key(row[key])))
                    break
            for ident in ids:
                if not ident[1]:
                    continue
                cur = first.get(ident)
                if cur is None or (period, ms) < cur:
                    first[ident] = (period, ms)
                cur = last.get(ident)
                if cur is None or (period, ms) > cur:
                    last[ident] = (period, ms)
    return {"first": first, "last": last, "span": {p: tuple(v) for p, v in span.items()}}


def _offsets(span: dict[int, tuple[int, int]]) -> dict[int, int]:
    """Start index of each kept period on the resampled axis (the reader's `Period.start`)."""
    out, cursor = {}, 0
    for pid in sorted(span):
        out[pid] = cursor
        cursor += len(pff._grid_ms(*span[pid]))
    return out


def _frame_of(where: tuple[int, int], span, offsets) -> int:
    period, ms = where
    t0, t1 = span[period]
    n = len(pff._grid_ms(t0, t1))
    k = (ms - t0 + pff.STEP_OUT_MS // 2) // pff.STEP_OUT_MS
    return offsets[period] + int(min(max(k, 0), n - 1))


# ------------------------------------------------------------------------- records


def build_touch_events(records: list[dict], scan: dict) -> tuple[np.ndarray, dict]:
    """Pure core: event records + `_scan_rows` output -> (events array, aggregate stats)."""
    span, first, last = scan["span"], scan["first"], scan["last"]
    offsets = _offsets(span)
    drops: Counter[str] = Counter()
    rows: set[tuple[int, int, str]] = set()

    def put(where, team: int, kind: str) -> None:
        rows.add((_frame_of(where, span, offsets), team, kind))

    for record in records:
        flat = _flat(record)
        period = pff._period(flat.get("period"))
        if period is not None and period not in (1, 2):
            drops["outside_period"] += 1
            continue
        home = pff._home_flag(flat.get("hometeam"))
        if home is None:
            drops["no_team"] += 1
            continue
        team = HOME if home else AWAY
        game_where = first.get(("game", pff._event_join_id(record)))
        poss_id = ""
        for key in _POSS_ID_KEYS:
            if record.get(key) is not None:
                poss_id = pff._key(record[key])
                break
        poss_where = first.get(("poss", poss_id)) if poss_id else None
        poss_type = _text(flat.get("possessioneventtype")).casefold()
        game_type = pff._event_type(flat.get("gameeventtype"))
        emitted = False

        if poss_type not in _NO_POSSESSION:
            if poss_where is None:
                drops["unmatched"] += 1
            else:
                kind = _POSSESSION_KIND.get(poss_type, "other_touch")
                put(poss_where, team, kind)
                emitted = True
                if kind == "pass":
                    outcome = _text(flat.get("passoutcometype")).casefold()
                    receiver = any(
                        _present(flat.get(k))
                        for k in ("receiverplayerid", "targetplayerid", "receiverid")
                    )
                    if (outcome in _COMPLETE or (not outcome and receiver)) and (
                        ("poss", poss_id) in last
                    ):
                        put(last[("poss", poss_id)], team, "reception")
        if any(_present(flat.get(k)) for k in ("fouls", "foultype")):
            where = poss_where or game_where
            if where is None:
                drops["unmatched"] += 1
            else:
                put(where, team, "foul")
                emitted = True
        setpiece = next((flat[k] for k in flat if "setpiece" in k and _present(flat[k])), None)
        if game_type in _KICKOFFS or (setpiece is not None and pff._is_setpiece(setpiece)):
            if game_where is None:
                drops["unmatched"] += 1
            else:
                put(game_where, team, "set_piece")
                emitted = True
        elif game_type == "OUT":
            if game_where is None:
                drops["unmatched"] += 1
            else:
                put(game_where, team, "out")
                emitted = True
        if not emitted:
            drops["no_touch"] += 1

    ordered = sorted(rows, key=lambda row: (row[0], row[1], row[2]))
    out = np.empty(len(ordered), dtype=TOUCH_EVENT_DTYPE)
    for i, row in enumerate(ordered):
        out[i] = row
    stats: dict[str, int] = {"records": len(records), **drops}
    for _, team, kind in ordered:
        stats[f"{kind}:{'home' if team == HOME else 'away'}"] = (
            stats.get(f"{kind}:{'home' if team == HOME else 'away'}", 0) + 1
        )
    stats["n_frames"] = sum(len(pff._grid_ms(*span[p])) for p in span)
    return out, stats


def _guard(game_id: str, allow_test: bool) -> str:
    gid = str(game_id)
    if gid not in pff._GAME_ID_SET:
        raise ValueError(f"{gid!r} is not a PFF game id")
    if not allow_test and gid in set(pff.game_ids("test")):
        raise ValueError(f"PFF game {gid} is in the sealed test split; it is never trained on")
    return gid


def _stamp(paths: dict) -> dict:
    return {
        "version": EVENTS_VERSION,
        "files": [pff._stat(paths["tracking"]), pff._stat(paths["events"])],
    }


def touch_events_with_stats(game_id: str, *, allow_test: bool = False) -> tuple[np.ndarray, dict]:
    """`touch_events` plus the aggregate counts; cached under `cache/pff-{id}/`."""
    gid = _guard(game_id, allow_test)
    paths = pff.file_paths(gid)
    if paths["events"] is None:
        raise FileNotFoundError(f"PFF game {gid} has no Event Data file")
    stamp = _stamp(paths)
    cache = cache_dir() / f"pff-{gid}" / f"pff-touch-events-v{EVENTS_VERSION}.npz"
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if json.loads(str(z["stamp"])) == stamp:
                    return z["events"].astype(TOUCH_EVENT_DTYPE), json.loads(str(z["stats"]))
        except (OSError, ValueError, KeyError):
            pass
    payload = json.loads(paths["events"].read_text(encoding="utf-8-sig"))
    out, stats = build_touch_events(pff._event_records(payload), _scan_rows(paths["tracking"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, events=out, stamp=json.dumps(stamp), stats=json.dumps(stats))
    return out, stats


def touch_events(game_id: str, *, allow_test: bool = False) -> np.ndarray:
    """Touch and pass events of one PFF game on the resampled 25 fps index.

    Returns the `TOUCH_EVENT_DTYPE` array described in the module docstring, sorted by
    frame. Refuses the 16 test-split games unless `allow_test=True` (evaluation tooling only).
    """
    return touch_events_with_stats(game_id, allow_test=allow_test)[0]


# ---------------------------------------------------------------------------- probe


def share_near(events: np.ndarray, n_frames: int, tol: int = 2) -> int:
    """Number of frames within `tol` frames of any event."""
    frames = np.unique(events["frame"])
    if len(frames) == 0:
        return 0
    grid = np.arange(n_frames)
    lo = np.searchsorted(frames, grid - tol, side="left")
    hi = np.searchsorted(frames, grid + tol, side="right")
    return int(np.count_nonzero(hi > lo))


def format_events_probe(rows: list[dict]) -> str:
    """Aggregates over per-game rows (`error` rows are counted, not detailed)."""
    ok = [r for r in rows if "stats" in r]
    total: Counter[str] = Counter()
    near = frames = 0
    for r in ok:
        total.update({k: v for k, v in r["stats"].items() if k != "n_frames"})
        frames += r["stats"]["n_frames"]
        near += r["near"]
    lines = [
        f"games requested {len(rows)}, with events {len(ok)}, failed or no events "
        f"{len(rows) - len(ok)}",
        "kind        home      away",
    ]
    for kind in KINDS:
        lines.append(f"{kind:<10} {total[kind + ':home']:>6}    {total[kind + ':away']:>6}")
    drops = ("records", "outside_period", "unmatched", "no_team", "no_touch")
    lines.append("  ".join(f"{k} {total[k]}" for k in drops))
    share = near / frames if frames else float("nan")
    lines.append(f"frames {frames}, share within 2 frames of an event {share:.4f}")
    for r in rows:
        if "error" in r:
            lines.append(f"  {r['id']}: {r['error']}")
    return "\n".join(lines) + "\n"


def probe_events(ids: list[str]) -> str:
    rows = []
    for gid in ids:
        try:
            events, stats = touch_events_with_stats(gid)
            rows.append({"id": gid, "stats": stats, "near": share_near(events, stats["n_frames"])})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            rows.append({"id": gid, "error": f"{type(exc).__name__}: {exc}"})
    return format_events_probe(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PFF event touches (train split only).")
    parser.add_argument("--probe-events", action="store_true")
    parser.add_argument("--games", default=None, help="'all' or the first N train ids")
    args = parser.parse_args(argv)
    if not args.probe_events:
        parser.print_help()
        raise SystemExit(2)
    ids = pff._limit_ids(pff.game_ids("train", installed_only=True), args.games)
    sys.stdout.write(probe_events(ids))


if __name__ == "__main__":
    main()
