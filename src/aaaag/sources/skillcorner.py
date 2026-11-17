"""SkillCorner open data: broadcast tracking at 10 fps, resampled to 25 fps.

github.com/SkillCorner/opendata (MIT; credit SkillCorner). Files live under
`$AAAAG_EXTERNAL_DIR/skillcorner/data/`, the repository's `data/` tree:
`matches.json`, then `matches/{id}/{id}_match.json`, `{id}_tracking_extrapolated.jsonl`
and `{id}_phases_of_play.csv`. Nothing here is fetched; drop the tree in place.

Tracking is one JSON object per line. Coordinates are already metres with the origin at
the pitch centre, so they are not rescaled. `frame` is contiguous across the file.
`timestamp` is match time at 0.1 s (`HH:MM:SS.ss` or a number of seconds) and is null
on the few frames whose `period` is null. Those frames sit before the first half and
between the halves.

Resampling, per contiguous run of the same `period` (a null period is its own run), so
nothing is interpolated across a period or across the outside frames:
- sample times come from `timestamp`, parsed to milliseconds;
- a run with no timestamps uses the frame index, 100 ms apart;
- the 25 fps grid is t0, t0+40 ms, ... up to the last sample;
- a position is linearly interpolated only between two consecutive samples that are
  both valid and at most 150 ms apart (one 10 fps step, plus a little slack);
- everywhere else the position is NaN;
- the label is the label of the nearest 10 fps sample in the run (an exact tie takes
  the earlier sample).

`detected_only=True` (the default) drops a player sample whose `is_detected` is not
true, which is the extrapolated point the camera did not see. The ball is NaN unless
`is_detected` is true, whatever `detected_only` is. Slots are the metadata players
with a home or away `team_id`, in file order, and carry no names. A tracking id that
is not in the metadata is dropped.

Labels, int8, derived here:
- a frame whose `period` is null is outside the halves and is X;
- inside a period, a frame whose `frame` is not in any phases-of-play interval
  `[frame_start, frame_end)` is D. SkillCorner only records a phase while the ball is
  in play, and `duration` equals `(frame_end - frame_start) / 10`, so the end frame
  is exclusive. Continuous play has the next `frame_start` equal to this `frame_end`;
- otherwise `possession.group` `"home team"` is H and `"away team"` is A;
- in play with no group is X, which training ignores.

`{id}_dynamic_events.csv` is not a ball-out interval (it is actions and possessions),
so it is ignored. If the phases file is missing, or has no `frame_start`/`frame_end`
column, or no positive-length interval, there is no out-of-play signal and no frame
is labelled D.

`bundle.extras["label_share"]` is the fraction of 25 fps frames in each of H, A, D, X.
`bundle.extras["provenance"]["dead_ball"]` is `phases_of_play [frame_start, frame_end)`
or `none`.

Where it fails: about 3% of identities are wrong (SkillCorner's own figure); a phase
boundary is the ball going out, not a referee's whistle; an in-play frame with a null
group stays X rather than a guess. The parsed cache is
`cache/skillcorner-{id}/skillcorner-v{PARSER_VERSION}.npz` plus a json sidecar.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import (
    AWAY,
    HOME,
    LABELS,
    A,
    D,
    H,
    MatchBundle,
    MatchMeta,
    Period,
    Tracking,
    X,
)
from ..degrade import degrade as degrade_bundle
from ..paths import cache_dir, external_dir
from ..registry import SOURCES

PARSER_VERSION = "1"  # bump whenever the arrays produced for the same files change
FPS = 25
STEP_IN_MS = 100  # 10 fps
STEP_OUT_MS = 40  # 25 fps
MAX_GAP_MS = 150  # one 10 fps step; a skipped frame (200 ms) is a gap

PROVENANCE = {
    "dataset": "SkillCorner Open Data",
    "repository": "github.com/SkillCorner/opendata",
    "licence": "MIT",
    "labeller": (
        "SkillCorner broadcast tracking; labels derived here from possession.group "
        "and phases of play"
    ),
}
_DEAD_BALL = "phases_of_play [frame_start, frame_end)"
_NO_DEAD_BALL = "none"


@dataclass
class _Sample:
    frame: int
    period: int | None
    ms: int | None
    ball: tuple[float, float] | None
    group: int | None
    players: dict[str, tuple[float, float]]


def data_root() -> Path:
    """`$AAAAG_EXTERNAL_DIR/skillcorner/data` (the opendata `data/` directory)."""
    return external_dir() / "skillcorner" / "data"


def file_paths(match_id: str) -> dict[str, Path]:
    """meta, tracking and phases paths for one match. Phases may be absent."""
    match_id = str(match_id)
    folder = data_root() / "matches" / match_id
    return {
        "meta": folder / f"{match_id}_match.json",
        "tracking": folder / f"{match_id}_tracking_extrapolated.jsonl",
        "phases": folder / f"{match_id}_phases_of_play.csv",
    }


def _match_ids() -> tuple[str, ...]:
    path = data_root() / "matches.json"
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(raw, list):
        return ()
    ids: list[str] = []
    for item in raw:
        if isinstance(item, dict) and item.get("id") is not None:
            ids.append(str(item["id"]))
        elif isinstance(item, int | str) and not isinstance(item, bool):
            ids.append(str(item))
    return tuple(ids)


# Read once at import. Empty when the tree is not installed, so importing this module
# never requires the download. `load_skillcorner` itself reads paths at call time.
MATCHES: tuple[str, ...] = _match_ids()


@SOURCES.register("skillcorner", version=PARSER_VERSION)
def load_skillcorner(
    *,
    match_id: str,
    use_cache: bool = True,
    degrade: dict | None = None,
    detected_only: bool = True,
) -> MatchBundle:
    """One SkillCorner match on the 25 fps contract, with derived labels.

    `degrade` ({version, seed, players}) applies aaaag.degrade after the cache, so the
    cache stores the clean parse. No `example`: the contract tests skip this source
    when the files are not installed.
    """
    match_id = str(match_id)
    paths = file_paths(match_id)
    missing = [str(paths[name]) for name in ("meta", "tracking") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "SkillCorner files not found:\n  "
            + "\n  ".join(missing)
            + "\nPlace the opendata tree at $AAAAG_EXTERNAL_DIR/skillcorner/data/"
        )
    cache = _cache_path(match_id) if use_cache else None
    bundle = _load_files(match_id, paths, detected_only=detected_only, cache_path=cache)
    if degrade is not None:
        bundle = degrade_bundle(bundle, **degrade)
    return bundle


def _load_files(
    match_id: str,
    paths: dict[str, Path],
    *,
    detected_only: bool,
    cache_path: Path | None,
) -> MatchBundle:
    phases_path = paths["phases"]
    stamp = {
        "parser_version": PARSER_VERSION,
        "detected_only": bool(detected_only),
        "files": [
            _stat(paths["meta"]),
            _stat(paths["tracking"]),
            _stat(phases_path) if phases_path.is_file() else None,
        ],
    }
    if cache_path is not None and (cached := _read_cache(cache_path, stamp, match_id)) is not None:
        return cached
    meta_raw = json.loads(paths["meta"].read_text(encoding="utf-8"))
    roster, slot_team, home, away, pitch_length, pitch_width = _roster(meta_raw)
    samples = _parse_tracking(paths["tracking"], detected_only=detected_only)
    phases = _read_phases(phases_path)
    bundle = _assemble(
        match_id,
        samples,
        roster,
        slot_team,
        home=home,
        away=away,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        phases=phases,
        detected_only=detected_only,
    )
    if cache_path is not None:
        _write_cache(cache_path, stamp, bundle)
    return bundle


def _assemble(
    match_id: str,
    samples: list[_Sample],
    roster: list[str],
    slot_team: np.ndarray,
    *,
    home: str,
    away: str,
    pitch_length: float,
    pitch_width: float,
    phases: tuple[np.ndarray, np.ndarray] | None,
    detected_only: bool,
) -> MatchBundle:
    slot_of = {player_id: index for index, player_id in enumerate(roster)}
    player_parts: list[np.ndarray] = []
    ball_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    periods: list[Period] = []
    cursor = 0
    for run in _runs(samples):
        times = _times_ms(run)
        grid = _grid_ms(int(times[0]), int(times[-1]))
        xy, valid, ball, ball_valid = _dense(run, slot_of)
        nearest = _nearest_index(times, grid)
        player_parts.append(_resample_xy(times, xy, valid, grid))
        ball_parts.append(_resample_xy(times, ball[:, None, :], ball_valid[:, None], grid)[:, 0, :])
        label_parts.append(_labels_at_10(run, phases)[nearest])
        n_out = len(grid)
        if run[0].period is not None:
            periods.append(Period(int(run[0].period), cursor, cursor + n_out))
        cursor += n_out
    if not periods:
        raise ValueError(f"{match_id}: tracking has no numbered period")
    meta = _meta(
        match_id,
        cursor,
        tuple(periods),
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        home=home,
        away=away,
    )
    tracking = Tracking(
        meta=meta,
        player_xy=np.concatenate(player_parts, axis=0),
        slot_team=slot_team,
        ball_xy=np.concatenate(ball_parts, axis=0),
    )
    provenance = {
        **PROVENANCE,
        "match_id": match_id,
        "detected_only": bool(detected_only),
        "dead_ball": _DEAD_BALL if phases is not None else _NO_DEAD_BALL,
    }
    bundle = MatchBundle(
        tracking=tracking,
        labels=np.concatenate(label_parts, axis=0),
        extras={"provenance": provenance},
    )
    return _finish(bundle)


def _finish(bundle: MatchBundle) -> MatchBundle:
    bundle.extras = {
        **bundle.extras,
        "label_share": _label_share(bundle.labels),
    }
    return bundle.validate()


def _label_share(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(LABELS))
    total = float(counts[: len(LABELS)].sum())
    return {name: float(counts[index] / total) for index, name in enumerate(LABELS)}


def _runs(samples: list[_Sample]) -> list[list[_Sample]]:
    """Contiguous same-period blocks, in frame order. A null period is its own block."""
    runs: list[list[_Sample]] = []
    for sample in samples:
        if not runs or runs[-1][0].period != sample.period:
            runs.append([sample])
        else:
            runs[-1].append(sample)
    for run in runs:
        if all(sample.ms is not None for sample in run) and any(
            int(run[i].ms) > int(run[i + 1].ms) for i in range(len(run) - 1)
        ):
            run.sort(key=lambda sample: (int(sample.ms or 0), sample.frame))
    return runs


def _times_ms(run: list[_Sample]) -> np.ndarray:
    frames = np.fromiter((sample.frame for sample in run), dtype=np.int64, count=len(run))
    known = np.array([sample.ms is not None for sample in run])
    if bool(known.all()):
        return np.fromiter((sample.ms for sample in run), dtype=np.int64, count=len(run))
    if not bool(known.any()):
        return (frames - int(frames[0])) * STEP_IN_MS
    times = np.zeros(len(run), dtype=np.int64)
    times[known] = [sample.ms for sample in run if sample.ms is not None]
    known_idx = np.flatnonzero(known)
    for index in np.flatnonzero(~known):
        place = int(np.searchsorted(known_idx, index))
        if place <= 0:
            anchor = int(known_idx[0])
        elif place >= len(known_idx):
            anchor = int(known_idx[-1])
        else:
            left = int(known_idx[place - 1])
            right = int(known_idx[place])
            anchor = left if index - left <= right - index else right
        times[int(index)] = int(times[anchor]) + (int(frames[index]) - int(frames[anchor])) * (
            STEP_IN_MS
        )
    return times


def _grid_ms(t0: int, t1: int) -> np.ndarray:
    if t1 < t0:
        t0, t1 = t1, t0
    n_out = (t1 - t0) // STEP_OUT_MS + 1
    return t0 + np.arange(n_out, dtype=np.int64) * STEP_OUT_MS


def _nearest_index(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    nxt = np.clip(np.searchsorted(times, grid, side="left"), 0, len(times) - 1)
    prv = np.clip(nxt - 1, 0, len(times) - 1)
    use_prev = np.abs(times[prv] - grid) <= np.abs(times[nxt] - grid)
    return np.where(use_prev, prv, nxt)


def _resample_xy(
    times: np.ndarray, xy: np.ndarray, valid: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """(M, S, 2) float32. Interpolate only across one valid 10 fps step; else NaN."""
    n_samples, n_slots, _ = xy.shape
    left = np.searchsorted(times, grid, side="right") - 1
    left_c = np.clip(left, 0, n_samples - 1)
    right = np.clip(left + 1, 0, n_samples - 1)
    exact = (left >= 0) & (left < n_samples) & (times[left_c] == grid)
    has_right = (left >= 0) & ((left + 1) < n_samples)
    dt = np.zeros(len(grid), dtype=np.int64)
    dt[has_right] = times[right[has_right]] - times[left_c[has_right]]
    span = has_right & (dt > 0) & (dt <= MAX_GAP_MS) & ~exact
    alpha = np.zeros(len(grid), dtype=np.float64)
    alpha[span] = (grid[span] - times[left_c[span]]) / dt[span]

    left_xy = xy[left_c]
    right_xy = xy[right]
    left_ok = valid[left_c]
    right_ok = valid[right]
    take_exact = exact[:, None] & left_ok
    take_span = span[:, None] & left_ok & right_ok
    out = np.full((len(grid), n_slots, 2), np.nan, dtype=np.float32)
    if np.any(take_exact):
        out[take_exact] = left_xy[take_exact]
    if np.any(take_span):
        blended = (1.0 - alpha)[:, None, None] * left_xy + alpha[:, None, None] * right_xy
        out[take_span] = blended[take_span]
    return out


def _dense(
    run: list[_Sample], slot_of: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(run)
    n_slots = len(slot_of)
    xy = np.full((n_samples, n_slots, 2), np.nan, dtype=np.float64)
    valid = np.zeros((n_samples, n_slots), dtype=bool)
    ball = np.full((n_samples, 2), np.nan, dtype=np.float64)
    ball_valid = np.zeros(n_samples, dtype=bool)
    for index, sample in enumerate(run):
        if sample.ball is not None:
            ball[index] = sample.ball
            ball_valid[index] = True
        for player_id, pos in sample.players.items():
            slot = slot_of.get(player_id)
            if slot is None:
                continue
            xy[index, slot] = pos
            valid[index, slot] = True
    return xy, valid, ball, ball_valid


def _labels_at_10(
    samples: list[_Sample], phases: tuple[np.ndarray, np.ndarray] | None
) -> np.ndarray:
    labels = np.full(len(samples), X, dtype=np.int8)
    played = np.array([sample.period is not None for sample in samples])
    if phases is None:
        dead = np.zeros(len(samples), dtype=bool)
    else:
        frames = np.fromiter(
            (sample.frame for sample in samples), dtype=np.int64, count=len(samples)
        )
        dead = played & ~_covers(frames, phases[0], phases[1])
    groups = np.full(len(samples), -1, dtype=np.int8)
    for index, sample in enumerate(samples):
        if sample.group is not None:
            groups[index] = sample.group
    labels[dead] = D
    alive = played & ~dead
    labels[alive & (groups == H)] = H
    labels[alive & (groups == A)] = A
    return labels


def _covers(frames: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """True where `frame` lies in any half-open [start, end). Overlaps are allowed."""
    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = np.maximum.accumulate(ends[order])
    index = np.searchsorted(starts, frames, side="right") - 1
    covered = np.zeros(len(frames), dtype=bool)
    valid = index >= 0
    covered[valid] = frames[valid] < ends[index[valid]]
    return covered


def _parse_tracking(path: Path, *, detected_only: bool) -> list[_Sample]:
    samples: list[_Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                continue
            try:
                frame = _frame_id(row.get("frame"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line_no}: bad frame") from exc
            samples.append(
                _Sample(
                    frame=frame,
                    period=_period(row.get("period")),
                    ms=_timestamp_ms(row.get("timestamp")),
                    ball=_ball_xy(row.get("ball_data")),
                    group=_group(row.get("possession")),
                    players=_player_map(row.get("player_data"), detected_only=detected_only),
                )
            )
    if not samples:
        raise ValueError(f"{path.name}: no frames")
    samples.sort(key=lambda sample: sample.frame)
    return samples


def _read_phases(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(starts, ends) or None when the file is not a clean in-play signal."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None
        fields = {(name or "").strip().casefold(): name for name in reader.fieldnames}
        start_key = fields.get("frame_start", fields.get("start_frame"))
        end_key = fields.get("frame_end", fields.get("end_frame"))
        if start_key is None or end_key is None:
            return None
        starts: list[int] = []
        ends: list[int] = []
        for row in reader:
            try:
                start = int(float(row[start_key]))
                end = int(float(row[end_key]))
            except (TypeError, ValueError):
                continue
            if end > start:
                starts.append(start)
                ends.append(end)
    if not starts:
        return None
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _roster(
    meta: dict,
) -> tuple[list[str], np.ndarray, str, str, float, float]:
    home = meta.get("home_team") or {}
    away = meta.get("away_team") or {}
    home_id = _key(home.get("id") if isinstance(home, dict) else home)
    away_id = _key(away.get("id") if isinstance(away, dict) else away)
    if not home_id or not away_id:
        raise ValueError("match metadata needs home_team.id and away_team.id")
    player_ids: list[str] = []
    teams: list[int] = []
    seen: set[str] = set()
    for player in meta.get("players") or []:
        if not isinstance(player, dict):
            continue
        player_id = _key(player.get("id", player.get("player_id")))
        team = player.get("team_id")
        if team is None and isinstance(player.get("team"), dict):
            team = player["team"].get("id")
        team_id = _key(team)
        if not player_id or player_id in seen:
            continue
        if team_id == home_id:
            side = HOME
        elif team_id == away_id:
            side = AWAY
        else:
            continue
        seen.add(player_id)
        player_ids.append(player_id)
        teams.append(side)
    if not player_ids:
        raise ValueError("match metadata has no player with a home or away team_id")
    return (
        player_ids,
        np.asarray(teams, dtype=np.int8),
        _side_name(home, "home"),
        _side_name(away, "away"),
        *_pitch(meta),
    )


def _pitch(meta: dict) -> tuple[float, float]:
    length = meta.get("pitch_length")
    width = meta.get("pitch_width")
    pitch = meta.get("pitch") if isinstance(meta.get("pitch"), dict) else {}
    if length is None:
        length = pitch.get("length", pitch.get("pitch_length"))
    if width is None:
        width = pitch.get("width", pitch.get("pitch_width"))
    if length is None or width is None:
        raise ValueError("match metadata needs pitch_length and pitch_width")
    length_f, width_f = float(length), float(width)
    if not np.isfinite(length_f) or not np.isfinite(width_f) or length_f <= 0 or width_f <= 0:
        raise ValueError("pitch_length and pitch_width must be positive")
    return length_f, width_f


def _side_name(side: object, fallback: str) -> str:
    if isinstance(side, dict):
        name = side.get("name") or side.get("short_name")
        if name:
            return str(name)
    if isinstance(side, str) and side:
        return side
    return fallback


def _player_map(raw: object, *, detected_only: bool) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, list):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        player_id = _key(item.get("player_id", item.get("id")))
        if not player_id:
            continue
        xy = _xy(item)
        if xy is None:
            continue
        if detected_only and not _detected(item.get("is_detected")):
            continue
        out[player_id] = xy
    return out


def _ball_xy(raw: object) -> tuple[float, float] | None:
    if not isinstance(raw, dict) or not _detected(raw.get("is_detected")):
        return None
    return _xy(raw)


def _group(raw: object) -> int | None:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("group") or "").strip().casefold()
    if text == "home team":
        return H
    if text == "away team":
        return A
    return None


def _xy(raw: dict) -> tuple[float, float] | None:
    x, y = raw.get("x"), raw.get("y")
    if x is None or y is None:
        return None
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf) or not np.isfinite(yf):
        return None
    return xf, yf


def _detected(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return value is True or value == 1


def _frame_id(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("frame is missing")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return int(str(value).strip())


def _period(value: object) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        period = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return period if period >= 1 else None


def _timestamp_ms(value: object) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, int | float):
        if not np.isfinite(value):
            return None
        return int(np.rint(float(value) * 1000))
    text = str(value).strip()
    if not text or text.casefold() == "null":
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            seconds = int(parts[0]) * 60 + float(parts[1])
        else:
            seconds = float(text)
    except ValueError:
        return None
    if not np.isfinite(seconds):
        return None
    return int(np.rint(seconds * 1000))


def _key(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return "" if text.casefold() == "none" else text


def _meta(
    match_id: str,
    n_frames: int,
    periods: tuple[Period, ...],
    *,
    pitch_length: float,
    pitch_width: float,
    home: str,
    away: str,
) -> MatchMeta:
    return MatchMeta(
        match_id=match_id,
        source="skillcorner",
        n_frames=n_frames,
        fps=FPS,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        periods=periods,
        home=home,
        away=away,
    ).validate()


def _cache_path(match_id: str) -> Path:
    return cache_dir() / f"skillcorner-{match_id}" / f"skillcorner-v{PARSER_VERSION}.npz"


def _read_cache(path: Path, stamp: dict, match_id: str) -> MatchBundle | None:
    sidecar = path.with_suffix(".json")
    if not (path.is_file() and sidecar.is_file()):
        return None
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if saved.get("stamp") != stamp:
        return None
    info = saved["meta"]
    with np.load(path, allow_pickle=False) as arrays:
        periods = tuple(
            Period(int(index), int(start), int(stop)) for index, start, stop in arrays["periods"]
        )
        labels = arrays["labels"]
        meta = _meta(
            match_id,
            int(labels.shape[0]),
            periods,
            pitch_length=float(info["pitch_length"]),
            pitch_width=float(info["pitch_width"]),
            home=str(info["home"]),
            away=str(info["away"]),
        )
        tracking = Tracking(
            meta=meta,
            player_xy=arrays["player_xy"],
            slot_team=arrays["slot_team"],
            ball_xy=arrays["ball_xy"],
        )
        bundle = MatchBundle(
            tracking=tracking,
            labels=labels,
            extras={"provenance": saved["provenance"]},
        )
    return _finish(bundle)


def _write_cache(path: Path, stamp: dict, bundle: MatchBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracking = bundle.tracking
    np.savez(
        path,
        player_xy=tracking.player_xy,
        slot_team=tracking.slot_team,
        ball_xy=tracking.ball_xy,
        labels=bundle.labels,
        periods=np.array([(p.index, p.start, p.stop) for p in bundle.meta.periods], dtype=np.int64),
    )
    sidecar = {
        "stamp": stamp,
        "provenance": bundle.extras["provenance"],
        "meta": {
            "pitch_length": bundle.meta.pitch_length,
            "pitch_width": bundle.meta.pitch_width,
            "home": bundle.meta.home,
            "away": bundle.meta.away,
        },
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]
