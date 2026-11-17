"""PFF FC World Cup 2022 tracking: 64 broadcast matches at 29.97 fps (train split for training, test split sealed).

Nothing here is fetched. Drop PFF's delivery under `$AAAAG_EXTERNAL_DIR/pff/` and keep it
out of git. Tracking is `{id}.jsonl.bz2` (plain `.jsonl` is accepted), metadata `{id}.json`
in a folder whose name contains "meta", rosters `{id}.json` in a folder whose name contains
"roster", and optional events `{id}.json` in a folder whose name contains "event"
(`Event Data/{id}.json`). Game ids are the tracking stems, and they are the 64 integers in `GAME_IDS`. `SPLIT_RULE`
holds out every fourth id of that list, whether or not the files are installed.
`load_extra_matches` refuses those test ids, so only the train split can enter
training. `degrade` on this loader is still refused: raw PFF is already a broadcast feed,
and training skips a second degradation with `extra_degrade: false`.

Spec (Tracking Data Specification v2.2), one JSON object per line: `version`, `gameRefId`,
`generatedTime`, `smoothedTime`, `videoTimeMs`, `frameNum` (video frames from 0), `period`
(1 and 2; 3 and 4 are extra time and are dropped, as is null), `periodElapsedTime` (seconds
since the half started), `periodGameClockTime`, `homePlayers`, `homePlayersSmoothed`,
`awayPlayers`, `awayPlayersSmoothed`, `balls`, `ballsSmoothed`, `game_event_id`,
`possession_event_id`, `game_event`, `possession_event`.

Players are `{jerseyNum, confidence, visibility, x, y}`. `confidence` is LOW / MEDIUM / HIGH.
`visibility` is VISIBLE or ESTIMATED (an off-screen player placed from context). Both
visibilities are kept. A jersey that appears twice for one team on one frame keeps the
VISIBLE sample, then the higher confidence, then the earlier one.

`balls` is a list of `{visibility, x, y, z}`. The raw ball is the single VISIBLE entry;
zero or several VISIBLE entries are NaN, and an ESTIMATED ball is NaN. That missingness is
the point: the match we will submit is missing its ball on about 41% of frames. `ballsSmoothed`
is a dict in kloppy's reader and a list in the spec; both are accepted. The smoothed ball is
not an independent measurement. PFF builds it from its own events, putting the ball on the
carrier and interpolating linearly between events, and our labels are derived from those same
events. Variant "smoothed" therefore leaks the label source. Score "raw" unless the question
is specifically about that leak.

Coordinates are metres, origin at the pitch centre. x runs along the touchlines, increasing
left to right as seen by the main camera (the length axis). y increases bottom to top (the
width axis). `io/atd.py` maps a normalised pitch onto that same centred frame, x along the
length and y across the width, and the contract does not flip at half-time. Nothing is
swapped, flipped or rescaled. `homeTeamStartLeft` is ignored.

Resampling, per kept period, from `periodElapsedTime`, as in skillcorner.py with a 50 ms gap
(one 29.97 fps step is about 33 ms, so a skipped frame is not bridged):
- the 25 fps grid is t0, t0+40 ms, ... up to the last sample;
- a position is linearly interpolated only between two consecutive samples that are both
  valid and at most 50 ms apart;
- everywhere else the position is NaN, including across a sample whose ball or player is
  missing and across the break between periods;
- the label is the label of the nearest native sample, and an exact tie takes the earlier.

Labels, int8, are event-derived and approximate, like Metrica's. They are not a referee
ball-state feed. On the native frames of each period, before resampling:
- the period starts DEAD with no owner;
- `game_event_type` OUT or END sets DEAD from that frame;
- FIRSTKICKOFF, SECONDKICKOFF, THIRDKICKOFF, FOURTHKICKOFF or OTB sets ALIVE;
- G (hits the post, bar or flag and stays in play), OFF, ON and SUB change nothing;
- `home_ball` not null sets the owner (true = home), which persists, including across DEAD;
- DEAD is D, ALIVE with an owner is H or A, ALIVE with no owner yet is X;
- VID (no video) labels `start_frame`..`end_frame` inclusive as X, after the rules below,
  so those frames are not scored; the running state is left unchanged;
- a set piece marks D from the frame after the previous game event to the frame before an
  OTB, when the gap is at least 1 s. The type comes from `Event Data/{id}.json`, joined on
  the game-event id. The file's schema is not specified: a key whose name contains
  "setpiece" (case-insensitive), on the event object or one level down, is the type. Null
  and open play (`O` / open / in play) do not trigger it. If the file or the key is absent
  the rule is skipped. This is the Metrica idea (the stoppage before a restart) and it is
  wrong when game events are sparse: the probe's event counts are how that is checked.
  On the real files the rule changes no frame, by design rather than by a bug. A game event
  repeats on every tracking row it spans, so "the previous game event" is that event's last
  row. In two train games, all 199 set-piece OTBs came at least 1 s after it (median about
  18 s). About 94% followed an OUT and the rest a substitution inside the same stoppage, so
  those frames are already D. The rule stays as a guard for files that lack the OUT.

`labels` selects the owner clock. `sequence` (the default) is the rules above: `home_ball`
on a game event sets the owner. `control` keeps the same dead-ball, kick-off, END, VID and
set-piece rules, but the owner changes only at an OTB, at that event's `start_frame`, to
`home_team` (true = home, false = away) and is carried forward. A missing `start_frame`
uses the frame the event is attached to. `control_touch` is `control` with each alive
H↔A owner change moved to the nearest raw-tracking touch of the new team within ±1.0 s
(`detect_touches` on the raw variant, even when the returned tracking is smoothed). With
no such touch the change stays on the event frame. The probe reports how many changes
moved and the median and IQR of that shift, in seconds.

`quality="strict"` (default `none`) then sets X, which the loss ignores, on three kinds of
frame: VID ranges (already X); frames where either team has fewer than `min_visible`
(default 10) VISIBLE players in the raw lists, so a replay, a close-up or another camera
is not trained on (ESTIMATED does not count); and ±`edge_s` (default 0.5 s) around each
dead/alive edge, because stoppage timing is the least certain part of an event label. The
probe counts each kind. The cache file name includes the label version, the quality mode
and those two thresholds.

Slots are the roster, in file order, for the home and away team ids, matched by
`jerseyNum == shirtNumber`. Any other jersey is dropped. Halves are placed adjacent, each
starting at its first kept frame. The parsed cache is
`cache/pff-{id}/pff-v{PARSER_VERSION}-{variant}-{labels}-{quality}-m{min_visible}-e{edge}.npz`
plus a json sidecar.

Ball height (`ball_z_axis`, optional, off unless asked for): the bundle keeps no z. With
`keep_ball_z` the parser also stores the `z` of the same single VISIBLE raw ball, and
`ball_z_axis` resamples it onto the bundle's 25 fps axis exactly as the raw ball's x and y
are resampled (a sample whose z is missing is NaN). It is versioned by `BALL_Z_VERSION`,
apart from `PARSER_VERSION`, so the bundle caches stay valid. `pff_ballz` caches it per game
and refuses the test split; `train.airborne_extra` reads it there.

Where it fails: event boundaries are not a whistle; a set-piece backfill can swallow open
play if the previous game event is far away; two VISIBLE balls become a gap; ESTIMATED ball
tracks are dropped on purpose; player identities are not kept, only shirt numbers as slot
keys. PFF's z is estimated from broadcast video and its accuracy is not documented here.
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from collections import Counter
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
from ..paths import cache_dir, external_dir
from ..registry import SOURCES

PARSER_VERSION = "2"  # bump whenever the arrays produced for the same files change
# The optional ball height (`ball_z_axis`). Bump it when the heights produced for the same files
# change; it never enters the bundle cache, so a run without the height reads the same caches.
BALL_Z_VERSION = "1"
FPS = 25
STEP_OUT_MS = 40
MAX_GAP_MS = 50  # one 29.97 fps step (~33 ms); a skipped frame (~67 ms) stays a gap
DEFAULT_PITCH = (105.0, 68.0)
SPLIT_RULE = (
    "Sort GAME_IDS as integers. Index % 4 == 3 is test (16 of 64), the rest is train. "
    "Fixed before any result: the split is that list, never which files are installed, "
    "and the test games are the only PFF games barred from training."
)
# The 64 World Cup game ids, already sorted as integers. The split is this list.
GAME_IDS = (
    3812,
    3813,
    3814,
    3815,
    3816,
    3817,
    3818,
    3819,
    3820,
    3821,
    3822,
    3823,
    3824,
    3825,
    3826,
    3827,
    3828,
    3829,
    3830,
    3831,
    3832,
    3833,
    3834,
    3835,
    3836,
    3837,
    3838,
    3839,
    3840,
    3841,
    3842,
    3843,
    3844,
    3845,
    3846,
    3847,
    3848,
    3849,
    3850,
    3851,
    3852,
    3853,
    3854,
    3855,
    3856,
    3857,
    3858,
    3859,
    10502,
    10503,
    10504,
    10505,
    10506,
    10507,
    10508,
    10509,
    10510,
    10511,
    10512,
    10513,
    10514,
    10515,
    10516,
    10517,
)
_GAME_ID_STRS = tuple(str(gid) for gid in GAME_IDS)
_GAME_ID_SET = frozenset(_GAME_ID_STRS)

_ALIVE = frozenset({"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"})
_DEAD = frozenset({"OUT", "END"})
_OPEN_PLAY = frozenset(
    {"", "o", "open", "openplay", "open_play", "open play", "inplay", "in_play", "in play", "null"}
)
_VARIANT_KEYS = {
    "raw": ("homePlayers", "awayPlayers", "balls"),
    "smoothed": ("homePlayersSmoothed", "awayPlayersSmoothed", "ballsSmoothed"),
}
_CONFIDENCE = {"high": 3, "medium": 2, "low": 1}
_JOIN_KEYS = ("game_event_id", "gameEventId", "id", "event_id", "eventId")
_LABEL_MODES = ("sequence", "control", "control_touch")
_QUALITY_MODES = ("none", "strict")
_TOUCH_WINDOW_S = 1.0  # control_touch: nearest raw touch of the new team

PROVENANCE = {
    "dataset": "PFF FC World Cup 2022",
    "licence": "PFF terms; training on the train split authorised by Ameya on 11 Nov (ADR 2026-11-12)",
    "labeller": (
        "event-derived from game_event and home_ball; approximate, like Metrica, "
        "not a referee ball-state feed"
    ),
    "smoothed_ball": (
        "PFF places ballsSmoothed on the event ball-carrier and interpolates between "
        "events, so variant smoothed leaks these labels"
    ),
}


def data_root() -> Path:
    """`$AAAAG_EXTERNAL_DIR/pff`."""
    return external_dir() / "pff"


def assign_split(ids: list[str], split: str) -> list[str]:
    """Apply `SPLIT_RULE` to an explicit id list. Ids must be integers."""
    if split not in {"train", "test", "all"}:
        raise ValueError(f"split must be 'train', 'test' or 'all', got {split!r}")
    ordered = sorted((str(gid) for gid in ids), key=_int_id)
    if split == "all":
        return ordered
    rest = split == "train"
    return [gid for index, gid in enumerate(ordered) if (index % 4 != 3) is rest]


def game_ids(
    split: str = "all", ids: list[str] | None = None, *, installed_only: bool = False
) -> list[str]:
    """Ids in `split` (`train`, `test` or `all`) from `GAME_IDS`.

    Pass `ids` to apply `SPLIT_RULE` to that list instead. `installed_only` keeps the ids
    whose tracking file is on disk, in the same order. The split itself does not change
    when files are missing.
    """
    chosen = assign_split(list(_GAME_ID_STRS) if ids is None else list(ids), split)
    if installed_only:
        present = set(list_game_ids())
        chosen = [gid for gid in chosen if gid in present]
    return chosen


def _require_catalogue_id(match_id: str) -> str:
    text = str(match_id)
    if text not in _GAME_ID_SET:
        raise ValueError(
            f"PFF game id {text} is not in GAME_IDS ({len(GAME_IDS)} World Cup matches). "
            "The loader only accepts those ids."
        )
    return text


def list_game_ids() -> list[str]:
    """Tracking stems, sorted as integers."""
    return sorted(discover_games(), key=_int_id)


def discover_games() -> dict[str, dict[str, Path | None | list[Path]]]:
    """tracking / meta / roster / events paths keyed by game id. Empty when not installed."""
    root = data_root()
    found: dict[str, dict[str, Path | None | list[Path]]] = {}
    if not root.is_dir():
        return found
    tracking: dict[str, Path] = {}
    meta: dict[str, list[Path]] = {}
    roster: dict[str, list[Path]] = {}
    events: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        gid = _tracking_id(path.name)
        if gid is not None:
            tracking[gid] = _prefer_tracking(tracking.get(gid), path)
            continue
        role = _json_role(path, root)
        if role == "meta":
            meta.setdefault(path.stem, []).append(path)
        elif role == "roster":
            roster.setdefault(path.stem, []).append(path)
        elif role == "events":
            events.setdefault(path.stem, []).append(path)
    for gid in tracking:
        found[gid] = {
            "tracking": tracking[gid],
            "meta": meta.get(gid, []),
            "roster": roster.get(gid, []),
            "events": events.get(gid, []),
        }
    return found


def file_paths(match_id: str) -> dict[str, Path | None]:
    """Resolved files for one game. Events may be None. Raises when required files are absent."""
    match_id = str(match_id)
    games = discover_games()
    if match_id not in games:
        raise FileNotFoundError(
            f"PFF game {match_id} not found under {data_root()} "
            f"({len(games)} tracking file(s) discovered)"
        )
    entry = games[match_id]
    missing = [
        name
        for name in ("meta", "roster")
        if not entry[name]  # type: ignore[arg-type]
    ]
    if missing:
        raise FileNotFoundError(f"PFF game {match_id} is missing {', '.join(missing)}")
    for name in ("meta", "roster", "events"):
        paths = entry[name]
        if isinstance(paths, list) and len(paths) > 1:
            raise ValueError(f"PFF game {match_id}: {len(paths)} {name} files")
    event_list = entry["events"]
    return {
        "tracking": entry["tracking"],  # type: ignore[return-value]
        "meta": entry["meta"][0],  # type: ignore[index]
        "roster": entry["roster"][0],  # type: ignore[index]
        "events": event_list[0] if event_list else None,  # type: ignore[index]
    }


@SOURCES.register("pff", version=PARSER_VERSION)
def load_pff(
    *,
    match_id: str,
    variant: str = "raw",
    degrade: dict | None = None,
    use_cache: bool = True,
    labels: str = "sequence",
    quality: str = "none",
    min_visible: int = 10,
    edge_s: float = 0.5,
) -> MatchBundle:
    """One PFF match on the 25 fps contract, with event-derived labels.

    `variant` is `raw` (default) or `smoothed`. A file that lacks that variant's keys
    raises; the other variant is never substituted. `labels` is `sequence` (default),
    `control` or `control_touch`. `quality` is `none` (default) or `strict`. `degrade`
    must be None: this loader returns the feed as delivered, and training turns its own
    degradation off for PFF. `smoothed` repeats PFF's event ball and leaks the labels
    (see the module docstring). `control_touch` still reads the raw arrays for the touch.
    No `example`: the contract tests skip this source until files are installed.
    """
    if degrade is not None:
        raise ValueError("PFF is loaded as delivered; degrade must be None")
    variant = _check_variant(variant)
    labels = _check_labels(labels)
    quality = _check_quality(quality)
    min_visible = _check_min_visible(min_visible)
    edge_s = _check_edge_s(edge_s)
    match_id = _require_catalogue_id(match_id)
    paths = file_paths(match_id)
    cache = (
        _cache_path(match_id, variant, labels, quality, min_visible, edge_s) if use_cache else None
    )
    return _load_files(
        match_id,
        paths,
        variant=variant,
        cache_path=cache,
        labels=labels,
        quality=quality,
        min_visible=min_visible,
        edge_s=edge_s,
    )


def _load_files(
    match_id: str,
    paths: dict[str, Path | None],
    *,
    variant: str,
    cache_path: Path | None,
    labels: str,
    quality: str,
    min_visible: int,
    edge_s: float,
) -> MatchBundle:
    events_path = paths["events"]
    stamp = {
        "parser_version": PARSER_VERSION,
        "variant": variant,
        "labels": labels,
        "quality": quality,
        "min_visible": min_visible,
        "edge_s": edge_s,
        "files": [
            _stat(paths["tracking"]),  # type: ignore[arg-type]
            _stat(paths["meta"]),  # type: ignore[arg-type]
            _stat(paths["roster"]),  # type: ignore[arg-type]
            _stat(events_path) if events_path is not None else None,
        ],
    }
    if cache_path is not None and (cached := _read_cache(cache_path, stamp, match_id)) is not None:
        return cached
    meta_raw = json.loads(paths["meta"].read_text(encoding="utf-8-sig"))  # type: ignore[union-attr]
    roster_raw = json.loads(paths["roster"].read_text(encoding="utf-8-sig"))  # type: ignore[union-attr]
    meta_obj = _meta_object(meta_raw)
    pitch, pitch_path = _pitch(meta_obj)
    meta_fps, fps_path = _meta_fps(meta_obj)
    home_id, away_id, home, away = _teams(meta_obj)
    slots = _roster_slots(roster_raw, home_id, away_id)
    setpiece_of, setpiece_field, setpiece_counts, apply_setpiece = _load_events(events_path)
    parsed = _parse_tracking(
        paths["tracking"],  # type: ignore[arg-type]
        slots,
        variant=variant,
        setpiece_of=setpiece_of,
        apply_setpiece=apply_setpiece,
        store_raw_players=labels == "control_touch" and variant != "raw",
    )
    if fps_path == "measured":
        meta_fps = parsed["measured_fps"]
    bundle = _assemble(
        match_id,
        parsed,
        slots,
        variant=variant,
        home=home,
        away=away,
        pitch=pitch,
        pitch_path=pitch_path,
        meta_fps=meta_fps,
        fps_path=fps_path,
        setpiece_field=setpiece_field,
        setpiece_counts=setpiece_counts,
        apply_setpiece=apply_setpiece,
        labels_mode=labels,
        quality=quality,
        min_visible=min_visible,
        edge_s=edge_s,
    )
    if cache_path is not None:
        _write_cache(cache_path, stamp, bundle)
    return bundle


def _assemble(
    match_id: str,
    parsed: dict,
    slots: list[tuple[object, int]],
    *,
    variant: str,
    home: str,
    away: str,
    pitch: tuple[float, float],
    pitch_path: str,
    meta_fps: float | None,
    fps_path: str,
    setpiece_field: str | None,
    setpiece_counts: dict[str, int],
    apply_setpiece: bool,
    labels_mode: str,
    quality: str,
    min_visible: int,
    edge_s: float,
) -> MatchBundle:
    slot_team = np.asarray([team for _, team in slots], dtype=np.int8)
    player_parts: list[np.ndarray] = []
    ball_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    vis_home_parts: list[np.ndarray] = []
    vis_away_parts: list[np.ndarray] = []
    raw_player_parts: list[np.ndarray] = []
    raw_ball_parts: list[np.ndarray] = []
    periods: list[Period] = []
    raw_25 = _nan_shares()
    smooth_25 = _nan_shares()
    need_raw = labels_mode == "control_touch" and variant != "raw"
    cursor = 0
    for pid in (1, 2):
        idx = _period_rows(parsed, pid)
        if len(idx) == 0:
            continue
        time_s = parsed["time_s"][idx]
        time_ms = parsed["time_ms"][idx]
        frame_num = parsed["frame_num"][idx]
        local = {int(global_i): pos for pos, global_i in enumerate(idx)}
        events = sorted(
            (local[event[0]], *event[1:]) for event in parsed["events"] if event[0] in local
        )
        labels = _labels_period(
            time_s,
            frame_num,
            events,
            parsed["vid_ranges"],
            apply_setpiece=apply_setpiece,
            mode=labels_mode,
        )
        players, ball, out_labels, grid = _resample_period(
            time_s,
            time_ms,
            parsed["players"][idx],
            _variant_ball(parsed, variant)[idx],
            labels,
            pid,
        )
        nearest = _nearest_index(time_ms, grid)
        player_parts.append(players)
        ball_parts.append(ball)
        label_parts.append(out_labels)
        vis_home_parts.append(parsed["vis_home"][idx][nearest])
        vis_away_parts.append(parsed["vis_away"][idx][nearest])
        if need_raw:
            raw_players, raw_ball, _raw_labels, _raw_grid = _resample_period(
                time_s,
                time_ms,
                _raw_players(parsed)[idx],
                _variant_ball(parsed, "raw")[idx],
                labels,
                pid,
            )
            raw_player_parts.append(raw_players)
            raw_ball_parts.append(raw_ball)
        periods.append(Period(pid, cursor, cursor + len(out_labels)))
        cursor += len(out_labels)
        raw_25 = _add_share(raw_25, _resampled_missing(parsed, "raw", idx, time_s, time_ms, pid))
        smooth_25 = _add_share(
            smooth_25, _resampled_missing(parsed, "smoothed", idx, time_s, time_ms, pid)
        )
    if not periods:
        raise ValueError(f"{match_id}: tracking has no period 1 or 2 frames")
    meta = _match_meta(
        match_id,
        cursor,
        tuple(periods),
        pitch_length=pitch[0],
        pitch_width=pitch[1],
        home=home,
        away=away,
    )
    out_labels = np.concatenate(label_parts, axis=0)
    if labels_mode == "control_touch":
        if variant == "raw":
            touch_players = np.concatenate(player_parts, axis=0)
            touch_ball = np.concatenate(ball_parts, axis=0)
        else:
            touch_players = np.concatenate(raw_player_parts, axis=0)
            touch_ball = np.concatenate(raw_ball_parts, axis=0)
        touch_tracking = Tracking(
            meta=meta,
            player_xy=touch_players,
            slot_team=slot_team,
            ball_xy=touch_ball,
        )
        out_labels, touch_stats = _snap_owner_changes(out_labels, touch_tracking)
    else:
        touch_stats = _touch_stats_none()
    vis_home = np.concatenate(vis_home_parts, axis=0)
    vis_away = np.concatenate(vis_away_parts, axis=0)
    low_visible = (vis_home < min_visible) | (vis_away < min_visible)
    edge = _dead_alive_edge_mask(out_labels, meta.periods, meta.fps, edge_s)
    if quality == "strict":
        out_labels = np.array(out_labels, copy=True)
        out_labels[low_visible | edge] = X
    tracking = Tracking(
        meta=meta,
        player_xy=np.concatenate(player_parts, axis=0),
        slot_team=slot_team,
        ball_xy=np.concatenate(ball_parts, axis=0),
    )
    stats = {
        "native_frames": parsed["native_frames"],
        "measured_fps": parsed["measured_fps"],
        "meta_fps": meta_fps,
        "fps_path": fps_path,
        "pitch_path": pitch_path,
        "kept_native": parsed["kept_native"],
        "ball_raw_counts": parsed["ball_raw_counts"],
        "ball_smoothed_missing_native": parsed["ball_smoothed_missing_native"],
        "ball_missing_raw_25fps": _share_ratio(raw_25),
        "ball_smoothed_missing_25fps": _share_ratio(smooth_25),
        "unknown_jerseys_raw": parsed["unknown_raw"],
        "unknown_jerseys_smoothed": parsed["unknown_smoothed"],
        "duplicate_players_raw": parsed["dups_raw"],
        "duplicate_players_smoothed": parsed["dups_smoothed"],
        "player_estimated_raw": parsed["estimated_raw"],
        "player_estimated_smoothed": parsed["estimated_smoothed"],
        "game_event_type": parsed["game_event_type"],
        "setpiece_field": setpiece_field,
        "setpiece_counts": setpiece_counts,
        "setpiece_rule": bool(apply_setpiece),
        "vid_events": parsed["vid_events"],
        "vid_frames": parsed["vid_frames"],
        "label_version": labels_mode,
        "quality": quality,
        "min_visible": min_visible,
        "edge_s": edge_s,
        "quality_low_visible": int(low_visible.sum()),
        "quality_edge": int(edge.sum()),
        "quality_applied": quality == "strict",
        **touch_stats,
    }
    provenance = {
        **PROVENANCE,
        "match_id": match_id,
        "variant": variant,
        "labels": labels_mode,
        "quality": quality,
        "setpiece_rule": bool(apply_setpiece),
        "dead_ball": (
            "OUT/END, VID as X, set-piece backfill from the event file"
            if apply_setpiece
            else "OUT/END and VID as X (no set-piece field)"
        ),
    }
    bundle = MatchBundle(
        tracking=tracking,
        labels=out_labels,
        extras={"provenance": provenance, "pff": json.loads(json.dumps(stats))},
    )
    return _finish(bundle)


def _period_rows(parsed: dict, pid: int) -> np.ndarray:
    """Native rows of period `pid` in time order, ties by frame number: the order resampled."""
    idx = np.flatnonzero(parsed["period"] == pid)
    if len(idx) == 0:
        return idx
    return idx[np.lexsort((parsed["frame_num"][idx], parsed["time_ms"][idx]))]


def ball_z_axis(match_id: str) -> tuple[np.ndarray, tuple[Period, ...]]:
    """(T,) float32 raw-ball height in metres on the 25 fps axis, and that axis's periods.

    The axis is the one `load_pff` builds from the same tracking file, whatever its variant,
    labels or quality: periods 1 and 2, each from its first to its last kept native row. A
    native sample is valid when the single VISIBLE raw ball has finite x, y and z; the height
    is interpolated like the ball's x and y (`_resample_period`) and is NaN elsewhere, so a
    frame with a ball but no z is NaN. No roster is read and no player is kept. Not cached and
    not split-checked: `pff_ballz.ball_z` does both.
    """
    match_id = _require_catalogue_id(match_id)
    paths = file_paths(match_id)
    parsed = _parse_tracking(
        paths["tracking"],  # type: ignore[arg-type]
        [],
        variant="raw",
        setpiece_of={},
        apply_setpiece=False,
        keep_ball_z=True,
    )
    ball = _variant_ball(parsed, "raw")
    height = parsed["ball_raw_z"]
    parts: list[np.ndarray] = []
    periods: list[Period] = []
    cursor = 0
    for pid in (1, 2):
        idx = _period_rows(parsed, pid)
        if len(idx) == 0:
            continue
        valid = np.isfinite(ball[idx, 0]) & np.isfinite(height[idx])
        z = np.where(valid, height[idx], np.nan)
        pair = np.stack([z, z], axis=1)
        _players, out, _labels, _grid = _resample_period(
            parsed["time_s"][idx],
            parsed["time_ms"][idx],
            pair[:, None, :],
            pair,
            np.zeros(len(idx), dtype=np.int8),
            pid,
        )
        parts.append(np.asarray(out[:, 0], dtype=np.float32))
        periods.append(Period(pid, cursor, cursor + len(out)))
        cursor += len(out)
    if not periods:
        raise ValueError(f"{match_id}: tracking has no period 1 or 2 frames")
    return np.concatenate(parts), tuple(periods)


def _finish(bundle: MatchBundle) -> MatchBundle:
    stats = dict(bundle.extras.get("pff") or {})
    stats["possessions"] = _possession_count(bundle.labels)
    bundle.extras = {
        **bundle.extras,
        "label_share": _label_share(bundle.labels),
        "pff": stats,
    }
    return bundle.validate()


def _variant_ball(parsed: dict, variant: str) -> np.ndarray:
    ball = parsed["ball_raw" if variant == "raw" else "ball_smoothed"]
    if ball is None:
        raise ValueError(f"variant {variant!r} ball was not parsed")
    return ball


def _resampled_missing(parsed, which: str, idx, time_s, time_ms, pid) -> tuple[int, int] | None:
    ball = parsed["ball_raw" if which == "raw" else "ball_smoothed"]
    if ball is None:
        return None
    _players, out, _labels, _n = _resample_period(
        time_s,
        time_ms,
        ball[idx][:, None, :],
        ball[idx],
        np.zeros(len(idx), dtype=np.int8),
        pid,
    )
    del _players, _labels
    missing = int(np.isnan(out[:, 0]).sum())
    return missing, int(len(out))


def _resample_period(time_s, time_ms, players, ball, labels, pid):
    """Players here may be (N, S, 2) or, for the ball-only share, (N, 1, 2)."""
    span = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    if span > 3 * 3600:
        raise ValueError(
            f"period {pid} spans {span:.0f}s; periodElapsedTime is assumed to be seconds"
        )
    t0, t1 = int(time_ms[0]), int(time_ms[-1])
    grid = _grid_ms(t0, t1)
    if len(grid) > max(10_000, len(time_ms) * 5):
        raise ValueError(
            f"period {pid} would resample {len(time_ms)} samples to {len(grid)} frames; "
            "periodElapsedTime is assumed to be seconds"
        )
    valid = np.isfinite(players[..., 0])
    ball_valid = np.isfinite(ball[..., 0])
    out_players = _resample_xy(time_ms, players, valid, grid)
    out_ball = _resample_xy(time_ms, ball[:, None, :], ball_valid[:, None], grid)[:, 0, :]
    nearest = _nearest_index(time_ms, grid)
    return out_players, out_ball, labels[nearest], grid


def _labels_period(
    time_s, frame_num, events, vid_ranges, *, apply_setpiece: bool, mode: str = "sequence"
) -> np.ndarray:
    """Native-frame labels. `control_touch` matches `control` here; the touch snap is later."""
    n = len(time_s)
    if not events and not vid_ranges:
        return np.full(n, D, dtype=np.int8)
    owner_mode = "control" if mode == "control_touch" else mode
    owner_at: list[tuple[int, int, int]] = []
    for seq, event in enumerate(events):
        index, _etype, home_ball, _sp, home_team, start_frame = event
        if owner_mode == "sequence":
            if home_ball is True:
                owner_at.append((int(index), seq, H))
            elif home_ball is False:
                owner_at.append((int(index), seq, A))
        elif _etype == "OTB":
            if home_team is True or home_team is False:
                at = _frame_index(frame_num, start_frame, int(index))
                owner_at.append((at, seq, H if home_team else A))
    owner_at.sort()
    labels = np.empty(n, dtype=np.int8)
    alive = False
    owner = -1
    cursor = 0
    owner_cursor = 0
    for index in range(n):
        while cursor < len(events) and events[cursor][0] == index:
            etype = events[cursor][1]
            cursor += 1
            if etype in _DEAD:
                alive = False
            elif etype in _ALIVE:
                alive = True
        while owner_cursor < len(owner_at) and owner_at[owner_cursor][0] == index:
            owner = owner_at[owner_cursor][2]
            owner_cursor += 1
        if not alive:
            labels[index] = D
        elif owner < 0:
            labels[index] = X
        else:
            labels[index] = owner
    if apply_setpiece:
        previous = None
        for index, etype, _home_ball, sp, *_rest in events:
            if (
                previous is not None
                and etype == "OTB"
                and _is_setpiece(sp)
                and float(time_s[index] - time_s[previous]) >= 1.0
                and previous + 1 < index
            ):
                labels[previous + 1 : index] = D
            previous = index
    if vid_ranges:
        mask = np.zeros(n, dtype=bool)
        for start, end in vid_ranges:
            mask |= (frame_num >= start) & (frame_num <= end)
        labels[mask] = X
    return labels


def _frame_index(frame_num: np.ndarray, start_frame: int | None, fallback: int) -> int:
    """Native index of `start_frame` inside this period. Missing uses `fallback`."""
    if start_frame is None or len(frame_num) == 0:
        return fallback
    hits = np.flatnonzero(frame_num == start_frame)
    if len(hits):
        return int(hits[0])
    after = np.flatnonzero(frame_num >= start_frame)
    if len(after):
        return int(after[0])
    return int(len(frame_num) - 1)


def _touch_stats_none() -> dict:
    return {
        "touch_changes": None,
        "touch_moved": None,
        "touch_shift_median_s": None,
        "touch_shift_iqr_s": None,
    }


def _shift_summary(shifts: list[float]) -> tuple[float | None, float | None]:
    if not shifts:
        return None, None
    arr = np.asarray(shifts, dtype=np.float64)
    q75, q25 = np.percentile(arr, [75, 25])
    return float(np.median(arr)), float(q75 - q25)


def _nearest_touch(candidates: np.ndarray, frame: int, window: int) -> int | None:
    if not len(candidates):
        return None
    i = int(np.searchsorted(candidates, frame))
    best, best_d = None, window + 1
    for j in (i - 1, i):
        if 0 <= j < len(candidates):
            distance = abs(int(candidates[j]) - frame)
            if distance < best_d:
                best, best_d = int(candidates[j]), distance
    return best


def _snap_owner_changes(labels: np.ndarray, tracking: Tracking) -> tuple[np.ndarray, dict]:
    """Move each alive H↔A boundary onto the nearest touch of the new team within ±1.0 s.

    A boundary with no such touch stays put and is not part of the shift distribution.
    The shift is the signed move in seconds after the one-frame-per-spell constraint.
    """
    from ..postprocess.touch_snap import detect_touches

    frames, teams = detect_touches(tracking)
    meta = tracking.meta
    window = int(round(_TOUCH_WINDOW_S * meta.fps))
    frames = np.asarray(frames, dtype=np.int64)
    teams = np.asarray(teams)
    by_team = {team: frames[teams == team] for team in (H, A)}
    out = np.array(labels, copy=True)
    shifts: list[float] = []
    changes = 0
    moved = 0
    for period in meta.periods:
        lo, hi = period.start, period.stop
        seg = labels[lo:hi]
        if len(seg) < 2:
            continue
        breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        if not len(breaks):
            continue
        starts = np.concatenate(([0], breaks)) + lo
        codes = seg[starts - lo]
        new = starts.copy()
        for k in range(1, len(starts)):
            boundary = int(starts[k])
            before, after = int(codes[k - 1]), int(codes[k])
            if before not in (H, A) or after not in (H, A) or before == after:
                continue
            changes += 1
            hit = _nearest_touch(by_team.get(after, frames[:0]), boundary, window)
            if hit is None:
                continue
            upper = int(starts[k + 1]) - 1 if k + 1 < len(starts) else hi - 1
            placed = min(max(hit, int(new[k - 1]) + 1), upper)
            new[k] = placed
            if placed != boundary:
                moved += 1
                shifts.append((placed - boundary) / float(meta.fps))
        stops = np.concatenate((new[1:], [hi]))
        for k in range(len(new)):
            out[int(new[k]) : int(stops[k])] = codes[k]
    median, iqr = _shift_summary(shifts)
    return out.astype(np.int8), {
        "touch_changes": changes,
        "touch_moved": moved,
        "touch_shift_median_s": median,
        "touch_shift_iqr_s": iqr,
    }


def _dead_alive_edge_mask(labels: np.ndarray, periods, fps: int, edge_s: float) -> np.ndarray:
    """Frames within ±`edge_s` of a D↔H/A boundary, clipped to the period.

    The boundary frame is the first frame of the new state. `round(edge_s * fps)` frames
    on either side of it are marked, including the boundary frame itself.
    """
    mask = np.zeros(len(labels), dtype=bool)
    half = int(round(float(edge_s) * fps))
    if half <= 0:
        return mask
    alive = (H, A)
    for period in periods:
        seg = labels[period.start : period.stop]
        if len(seg) < 2:
            continue
        changes = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for i in changes.tolist():
            prev, cur = int(seg[i - 1]), int(seg[i])
            if not ((prev == D and cur in alive) or (cur == D and prev in alive)):
                continue
            lo = period.start + max(0, i - half)
            hi = period.start + min(len(seg), i + half + 1)
            mask[lo:hi] = True
    return mask


def _raw_players(parsed: dict) -> np.ndarray:
    players = parsed.get("players_raw")
    if players is None:
        raise ValueError(
            "control_touch needs the raw tracking arrays (homePlayers, awayPlayers, balls)"
        )
    return players


def _parse_tracking(
    path: Path,
    slots: list[tuple[object, int]],
    *,
    variant: str,
    setpiece_of: dict[str, object],
    apply_setpiece: bool,
    store_raw_players: bool = False,
    keep_ball_z: bool = False,
) -> dict:
    """Native rows of periods 1 and 2. `keep_ball_z` also stores the raw ball's z
    (`ball_raw_z`, NaN where the raw ball or its z is missing); False stores None."""
    home_at, away_at = _slot_maps(slots)
    required = _VARIANT_KEYS[variant]
    other = "smoothed" if variant == "raw" else "raw"
    other_keys = _VARIANT_KEYS[other]
    other_ok = True
    n_slots = len(slots)

    native_frames = 0
    event_counts: Counter[str] = Counter()
    vid_events = 0
    vid_ranges: list[tuple[int, int]] = []
    times_by_period: dict[int, list[float]] = {}
    ball_counts = {"visible": 0, "estimated": 0, "missing": 0}
    smooth_missing = 0
    smooth_seen = 0
    unknown = {"raw": set(), "smoothed": set()}
    dups = {"raw": 0, "smoothed": 0}
    estimated = {"raw": [0, 0], "smoothed": [0, 0]}  # estimated, total
    events: list[tuple] = []
    kept_native = {"1": 0, "2": 0}

    cap = 1024
    period = np.empty(cap, dtype=np.int8)
    time_s = np.empty(cap, dtype=np.float64)
    time_ms = np.empty(cap, dtype=np.int64)
    frame_num = np.empty(cap, dtype=np.int64)
    players = np.full((cap, n_slots, 2), np.nan, dtype=np.float64)
    players_raw = (
        np.full((cap, n_slots, 2), np.nan, dtype=np.float64) if store_raw_players else None
    )
    vis_home = np.zeros(cap, dtype=np.int16)
    vis_away = np.zeros(cap, dtype=np.int16)
    ball_raw = np.full((cap, 2), np.nan, dtype=np.float64)
    ball_smooth = np.full((cap, 2), np.nan, dtype=np.float64)
    ball_z = np.full(cap, np.nan, dtype=np.float64) if keep_ball_z else None
    raw_ball_ok = True
    smooth_ball_ok = True
    n = 0

    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_no}: tracking line must be an object")
            native_frames += 1
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(
                    f"{path.name}:{line_no}: variant {variant!r} requires {missing}; "
                    "not falling back to the other variant"
                )
            if any(key not in row for key in other_keys):
                other_ok = False
            ge = row.get("game_event")
            etype = None
            home_ball = None
            home_team = None
            start_frame = None
            if ge is not None:
                if not isinstance(ge, dict):
                    raise ValueError(f"{path.name}:{line_no}: game_event must be an object or null")
                etype = _event_type(ge.get("game_event_type"))
                event_counts[_count_key(ge.get("game_event_type"))] += 1
                home_ball = _home_flag(ge.get("home_ball"))
                home_team = _home_flag(ge.get("home_team"))
                start_frame = _optional_frame(ge.get("start_frame"))
                if etype == "VID":
                    vid_events += 1
                    vid_ranges.append(_vid_span(ge, row.get("frameNum")))
            period_id = _period(row.get("period"))
            elapsed = row.get("periodElapsedTime")
            if period_id is not None and elapsed is not None and not isinstance(elapsed, bool):
                try:
                    times_by_period.setdefault(period_id, []).append(float(elapsed))
                except (TypeError, ValueError):
                    pass
            if period_id not in (1, 2):
                continue
            if n == cap:
                cap *= 2
                period = _grow(period, n, cap)
                time_s = _grow(time_s, n, cap)
                time_ms = _grow(time_ms, n, cap)
                frame_num = _grow(frame_num, n, cap)
                players = _grow_nan(players, n, cap)
                if players_raw is not None:
                    players_raw = _grow_nan(players_raw, n, cap)
                vis_home = _grow(vis_home, n, cap)
                vis_away = _grow(vis_away, n, cap)
                ball_raw = _grow_nan(ball_raw, n, cap)
                ball_smooth = _grow_nan(ball_smooth, n, cap)
                if ball_z is not None:
                    ball_z = _grow_nan(ball_z, n, cap)
            try:
                seconds = _elapsed(row.get("periodElapsedTime"))
                number = _frame_num(row.get("frameNum"))
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
            period[n] = period_id
            time_s[n] = seconds
            time_ms[n] = int(np.rint(seconds * 1000.0))
            frame_num[n] = number
            kept_native[str(period_id)] += 1
            vis_home[n] = 0
            vis_away[n] = 0

            raw_xy, raw_class = _raw_ball(row.get("balls"))
            ball_counts[raw_class] += 1
            if raw_xy is not None:
                ball_raw[n] = raw_xy
                if ball_z is not None:
                    height = _raw_ball_z(row.get("balls"))
                    if height is not None:
                        ball_z[n] = height
            smooth_xy = _smooth_ball(row.get("ballsSmoothed"))
            smooth_seen += 1
            if smooth_xy is None:
                smooth_missing += 1
            else:
                ball_smooth[n] = smooth_xy

            which_players = {
                "raw": (row.get("homePlayers"), row.get("awayPlayers")),
                "smoothed": (row.get("homePlayersSmoothed"), row.get("awayPlayersSmoothed")),
            }
            for name, (home_list, away_list) in which_players.items():
                if name != variant and not other_ok:
                    continue
                if name == variant:
                    target = players[n]
                elif name == "raw" and players_raw is not None:
                    target = players_raw[n]
                else:
                    target = None
                dropped, visible = _place_players(
                    home_list, home_at, HOME, target, unknown[name], estimated[name]
                )
                dups[name] += dropped
                if name == "raw":
                    vis_home[n] = visible
                dropped, visible = _place_players(
                    away_list, away_at, AWAY, target, unknown[name], estimated[name]
                )
                dups[name] += dropped
                if name == "raw":
                    vis_away[n] = visible

            if ge is not None:
                joined = setpiece_of.get(_key(row.get("game_event_id")))
                events.append((n, etype or "", home_ball, joined, home_team, start_frame))
            n += 1

    if native_frames == 0:
        raise ValueError(f"{path.name}: no frames")
    if n == 0:
        raise ValueError(f"{path.name}: no period 1 or 2 frames")
    if not other_ok:
        if other == "raw":
            raw_ball_ok = False
        else:
            smooth_ball_ok = False
    vid_frames = 0
    if vid_ranges:
        kept_fn = frame_num[:n]
        mask = np.zeros(n, dtype=bool)
        for start, end in vid_ranges:
            mask |= (kept_fn >= start) & (kept_fn <= end)
        vid_frames = int(mask.sum())
    return {
        "period": period[:n],
        "time_s": time_s[:n],
        "time_ms": time_ms[:n],
        "frame_num": frame_num[:n],
        "players": players[:n],
        "players_raw": None if players_raw is None else players_raw[:n],
        "vis_home": vis_home[:n],
        "vis_away": vis_away[:n],
        "ball_raw": ball_raw[:n] if raw_ball_ok else None,
        "ball_smoothed": ball_smooth[:n] if smooth_ball_ok else None,
        "ball_raw_z": None if ball_z is None else ball_z[:n],
        "events": events,
        "vid_ranges": vid_ranges,
        "native_frames": native_frames,
        "measured_fps": _measured_fps(times_by_period),
        "kept_native": {key: value for key, value in kept_native.items() if value},
        "ball_raw_counts": None if not raw_ball_ok else dict(ball_counts),
        "ball_smoothed_missing_native": (
            None if not smooth_ball_ok or smooth_seen == 0 else smooth_missing / smooth_seen
        ),
        "unknown_raw": None if variant != "raw" and not other_ok else len(unknown["raw"]),
        "unknown_smoothed": (
            None if variant != "smoothed" and not other_ok else len(unknown["smoothed"])
        ),
        "dups_raw": None if variant != "raw" and not other_ok else dups["raw"],
        "dups_smoothed": None if variant != "smoothed" and not other_ok else dups["smoothed"],
        "estimated_raw": _estimated_share(estimated["raw"], variant == "raw" or other_ok),
        "estimated_smoothed": _estimated_share(
            estimated["smoothed"], variant == "smoothed" or other_ok
        ),
        "game_event_type": dict(event_counts),
        "vid_events": vid_events,
        "vid_frames": vid_frames,
        "apply_setpiece": apply_setpiece,
    }


def _place_players(items, mapping, team, xy_row, unknown: set, estimated_pair: list[int]):
    """Write the chosen sample per jersey into `xy_row` when it is not None.

    Returns `(extras dropped, VISIBLE players)`. A jersey with any VISIBLE sample counts once,
    including a jersey that is not on the roster. ESTIMATED-only jerseys do not count.
    """
    if items is None:
        return 0, 0
    if not isinstance(items, list):
        raise ValueError("player list must be a list")
    best: dict[object, tuple] = {}
    sizes: dict[object, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        jersey = _jersey(item.get("jerseyNum"))
        pos = _xy(item)
        if jersey is None or pos is None:
            continue
        estimated_pair[1] += 1
        if str(item.get("visibility") or "").strip().casefold() == "estimated":
            estimated_pair[0] += 1
        sizes[jersey] = sizes.get(jersey, 0) + 1
        rank = (
            1 if str(item.get("visibility") or "").strip().casefold() == "visible" else 0,
            _CONFIDENCE.get(str(item.get("confidence") or "").strip().casefold(), 0),
            -index,
        )
        current = best.get(jersey)
        if current is None or rank > current[0]:
            best[jersey] = (rank, pos)
    dropped = 0
    visible = 0
    for jersey, (rank, pos) in best.items():
        if rank[0] == 1:
            visible += 1
        dropped += max(0, sizes[jersey] - 1)
        slot = mapping.get(jersey)
        if slot is None:
            unknown.add((team, jersey))
            continue
        if xy_row is not None:
            xy_row[slot, 0] = pos[0]
            xy_row[slot, 1] = pos[1]
    return dropped, visible


def _raw_ball(value) -> tuple[tuple[float, float] | None, str]:
    if not isinstance(value, list) or not value:
        return None, "missing"
    visible: list[dict] = []
    estimated = False
    for item in value:
        if not isinstance(item, dict):
            continue
        flag = str(item.get("visibility") or "").strip().casefold()
        if flag == "visible":
            visible.append(item)
        elif flag == "estimated":
            estimated = True
    if len(visible) == 1:
        return _xy(visible[0]), "visible"
    if len(visible) > 1:
        return None, "visible"
    if estimated:
        return None, "estimated"
    return None, "missing"


def _raw_ball_z(value) -> float | None:
    """z (m) of the single VISIBLE raw ball, the entry `_raw_ball` keeps. None otherwise."""
    if not isinstance(value, list):
        return None
    visible = [
        item
        for item in value
        if isinstance(item, dict)
        and str(item.get("visibility") or "").strip().casefold() == "visible"
    ]
    if len(visible) != 1:
        return None
    z = visible[0].get("z")
    if z is None or isinstance(z, bool):
        return None
    try:
        height = float(z)
    except (TypeError, ValueError):
        return None
    return height if np.isfinite(height) else None


def _smooth_ball(value) -> tuple[float, float] | None:
    """`ballsSmoothed` is a dict in kloppy and a list in the spec."""
    if isinstance(value, dict):
        return _xy(value)
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            return _xy(value[0])
        return None
    return None


def _load_events(path: Path | None):
    """(id → setpiece value, field path or None, value counts, apply rule)."""
    if path is None or not path.is_file():
        return {}, None, {}, False
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    records = _event_records(payload)
    field = _setpiece_field(records)
    if field is None:
        return {}, None, {}, False
    counts: Counter[str] = Counter()
    joined: dict[str, object] = {}
    for record in records:
        value = _path_get(record, field)
        counts[_count_key(value)] += 1
        gid = _event_join_id(record)
        if gid and gid not in joined:
            joined[gid] = value
    return joined, field, dict(counts), True


def _event_records(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("event file must be a JSON list or object")
    candidates = []
    for value in payload.values():
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            candidates.append([item for item in value if isinstance(item, dict)])
    if not candidates:
        return [payload]

    def score(records: list[dict]) -> tuple[int, int]:
        points = 0
        for record in records[:30]:
            if _event_join_id(record):
                points += 2
            if _setpiece_paths(record):
                points += 5
        return points, len(records)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _setpiece_field(records: list[dict]) -> str | None:
    counts: Counter[str] = Counter()
    for record in records:
        for path in _setpiece_paths(record):
            counts[path] += 1
    if not counts:
        return None
    return sorted(counts, key=lambda path: (-counts[path], path))[0]


def _setpiece_paths(record: dict) -> list[str]:
    found = []
    for key, value in record.items():
        if "setpiece" in str(key).casefold():
            found.append(str(key))
        if isinstance(value, dict):
            for key2 in value:
                if "setpiece" in str(key2).casefold():
                    found.append(f"{key}.{key2}")
    return found


def _path_get(record: dict, path: str):
    if "." not in path:
        return record.get(path)
    head, tail = path.split(".", 1)
    child = record.get(head)
    if isinstance(child, dict):
        return child.get(tail)
    return None


def _event_join_id(record: dict) -> str:
    for key in _JOIN_KEYS:
        if key in record and record[key] is not None:
            return _key(record[key])
    return ""


def _vid_span(event: dict, frame_num) -> tuple[int, int]:
    start = event.get("start_frame")
    end = event.get("end_frame")
    try:
        start_i = _frame_num(start) if start is not None else _frame_num(frame_num)
    except ValueError:
        start_i = _frame_num(frame_num)
    try:
        end_i = _frame_num(end) if end is not None else start_i
    except ValueError:
        end_i = start_i
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    return start_i, end_i


def _is_setpiece(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    text = str(value).strip().casefold()
    return text not in _OPEN_PLAY


def _event_type(value) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _count_key(value) -> str:
    if value is None:
        return "null"
    return str(value).strip()


def _estimated_share(pair: list[int], ok: bool) -> float | None:
    if not ok:
        return None
    _estimated, total = pair
    if total == 0:
        return None
    return _estimated / total


def _measured_fps(times_by_period: dict[int, list[float]]) -> float | None:
    deltas = []
    for samples in times_by_period.values():
        if len(samples) < 2:
            continue
        ordered = np.sort(np.asarray(samples, dtype=np.float64))
        diff = np.diff(ordered)
        diff = diff[diff > 0]
        if len(diff):
            deltas.append(diff)
    if not deltas:
        return None
    median = float(np.median(np.concatenate(deltas)))
    if median <= 0:
        return None
    return 1.0 / median


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
    """(M, S, 2) float32. Interpolate only across one valid step of at most 50 ms."""
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


def _meta_object(payload) -> dict:
    if isinstance(payload, list):
        if not payload or not isinstance(payload[0], dict):
            raise ValueError("metadata must be a JSON list with an object at [0]")
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError("metadata must be a JSON list or object")


def _teams(meta: dict) -> tuple[str, str, str, str]:
    home = meta.get("homeTeam")
    away = meta.get("awayTeam")
    home_id = _team_id(home, meta.get("homeTeamId"))
    away_id = _team_id(away, meta.get("awayTeamId"))
    if not home_id or not away_id:
        raise ValueError("metadata needs homeTeam.id and awayTeam.id")
    return home_id, away_id, _team_name(home, "home"), _team_name(away, "away")


def _team_id(value, alt) -> str:
    if isinstance(value, dict) and value.get("id") is not None:
        return _key(value.get("id"))
    if alt is not None:
        return _key(alt)
    if isinstance(value, int | str) and not isinstance(value, bool):
        return _key(value)
    return ""


def _team_name(value, fallback: str) -> str:
    if isinstance(value, dict):
        name = value.get("name") or value.get("shortName") or value.get("id")
        return str(name) if name is not None else fallback
    if isinstance(value, str) and value:
        return value
    return fallback


def _pitch(meta: dict) -> tuple[tuple[float, float], str]:
    stadium = meta.get("stadium") if isinstance(meta.get("stadium"), dict) else {}
    pitches = stadium.get("pitches") if isinstance(stadium, dict) else None
    if isinstance(pitches, list) and pitches and isinstance(pitches[0], dict):
        got = _pair(pitches[0].get("length"), pitches[0].get("width"))
        if got is not None:
            return got, "stadium.pitches[0]"
    if isinstance(stadium, dict):
        got = _pair(stadium.get("pitchLength"), stadium.get("pitchWidth"))
        if got is not None:
            return got, "stadium.pitchLength"
        got = _pair(stadium.get("length"), stadium.get("width"))
        if got is not None:
            return got, "stadium.length"
    got = _pair(meta.get("pitchLength"), meta.get("pitchWidth"))
    if got is not None:
        return got, "pitchLength"
    return DEFAULT_PITCH, "default"


def _pair(length, width) -> tuple[float, float] | None:
    if length is None or width is None:
        return None
    try:
        length_f, width_f = float(length), float(width)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(length_f) or not np.isfinite(width_f) or length_f <= 0 or width_f <= 0:
        return None
    return length_f, width_f


def _meta_fps(meta: dict) -> tuple[float | None, str]:
    if _finite_number(meta.get("fps")) is not None:
        return _finite_number(meta.get("fps")), "fps"
    videos = meta.get("videos")
    if isinstance(videos, dict) and _finite_number(videos.get("fps")) is not None:
        return _finite_number(videos.get("fps")), "videos.fps"
    if isinstance(videos, list):
        for item in videos:
            if isinstance(item, dict) and _finite_number(item.get("fps")) is not None:
                return _finite_number(item.get("fps")), "videos[0].fps"
    return None, "measured"


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _roster_slots(payload, home_id: str, away_id: str) -> list[tuple[object, int]]:
    if not isinstance(payload, list):
        raise ValueError("roster must be a JSON list")
    slots: list[tuple[object, int]] = []
    seen: set[tuple[int, object]] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        team = entry.get("team")
        team_id = team.get("id") if isinstance(team, dict) else team
        jersey = _jersey(entry.get("shirtNumber"))
        if jersey is None:
            continue
        side = HOME if _key(team_id) == home_id else AWAY if _key(team_id) == away_id else None
        if side is None or (side, jersey) in seen:
            continue
        seen.add((side, jersey))
        slots.append((jersey, side))
    if not slots:
        raise ValueError("roster has no player with a home or away team id")
    return slots


def _slot_maps(slots: list[tuple[object, int]]):
    home_at: dict[object, int] = {}
    away_at: dict[object, int] = {}
    for index, (jersey, team) in enumerate(slots):
        target = home_at if team == HOME else away_at
        if jersey not in target:
            target[jersey] = index
    return home_at, away_at


def _possession_count(labels: np.ndarray) -> int:
    from ..official import pm

    return len(pm.possession_spells(np.asarray(labels), pm.valid_runs(np.asarray(labels))))


def _label_share(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(LABELS))
    total = float(counts[: len(LABELS)].sum())
    return {name: float(counts[index] / total) for index, name in enumerate(LABELS)}


def _match_meta(
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
        source="pff",
        n_frames=n_frames,
        fps=FPS,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        periods=periods,
        home=home,
        away=away,
    ).validate()


def _cache_path(
    match_id: str,
    variant: str,
    labels: str = "sequence",
    quality: str = "none",
    min_visible: int = 10,
    edge_s: float = 0.5,
) -> Path:
    edge = f"{float(edge_s):.3f}".replace(".", "p")
    name = f"pff-v{PARSER_VERSION}-{variant}-{labels}-{quality}-m{int(min_visible)}-e{edge}.npz"
    return cache_dir() / f"pff-{match_id}" / name


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
        meta = _match_meta(
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
            extras={"provenance": saved["provenance"], "pff": saved["pff"]},
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
        "pff": bundle.extras["pff"],
        "meta": {
            "pitch_length": bundle.meta.pitch_length,
            "pitch_width": bundle.meta.pitch_width,
            "home": bundle.meta.home,
            "away": bundle.meta.away,
        },
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def _grow(buf: np.ndarray, n: int, cap: int) -> np.ndarray:
    grown = np.empty((cap, *buf.shape[1:]), dtype=buf.dtype)
    grown[:n] = buf[:n]
    return grown


def _grow_nan(buf: np.ndarray, n: int, cap: int) -> np.ndarray:
    grown = np.full((cap, *buf.shape[1:]), np.nan, dtype=buf.dtype)
    grown[:n] = buf[:n]
    return grown


def _nan_shares() -> list[int]:
    return [0, 0]


def _add_share(total: list[int], part: tuple[int, int] | None) -> list[int]:
    if part is None:
        return total
    total[0] += part[0]
    total[1] += part[1]
    return total


def _share_ratio(part: list[int]) -> float | None:
    if part[1] == 0:
        return None
    return part[0] / part[1]


def _open_text(path: Path):
    if path.name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def _tracking_id(name: str) -> str | None:
    if name.endswith(".jsonl.bz2"):
        return name[: -len(".jsonl.bz2")]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return None


def _prefer_tracking(old: Path | None, new: Path) -> Path:
    if old is None:
        return new
    new_bz2 = new.name.endswith(".jsonl.bz2")
    old_bz2 = old.name.endswith(".jsonl.bz2")
    if new_bz2 and not old_bz2:
        return new
    if old_bz2 and not new_bz2:
        return old
    return old if str(old) <= str(new) else new


def _json_role(path: Path, root: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    parts = [part.casefold() for part in path.relative_to(root).parts[:-1]]
    if not parts:
        return None
    event = any("event" in part for part in parts)
    roster = any("roster" in part for part in parts)
    meta = any("meta" in part for part in parts)
    if event and not roster:
        return "events"
    if roster and not meta:
        return "roster"
    if meta and not roster:
        return "meta"
    parent = path.parent.name.casefold()
    if "roster" in parent:
        return "roster"
    if "event" in parent:
        return "events"
    if "meta" in parent:
        return "meta"
    return None


def _check_variant(variant: str) -> str:
    if variant not in _VARIANT_KEYS:
        raise ValueError(f"variant must be 'raw' or 'smoothed', got {variant!r}")
    return variant


def _check_labels(labels: str) -> str:
    if labels not in _LABEL_MODES:
        raise ValueError(f"labels must be one of {_LABEL_MODES}, got {labels!r}")
    return labels


def _check_quality(quality: str) -> str:
    if quality not in _QUALITY_MODES:
        raise ValueError(f"quality must be one of {_QUALITY_MODES}, got {quality!r}")
    return quality


def _check_min_visible(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"min_visible must be an int >= 1, got {value!r}")
    return value


def _check_edge_s(value) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"edge_s must be a number >= 0, got {value!r}")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"edge_s must be a number >= 0, got {value!r}")
    return number


def _optional_frame(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return _frame_num(value)
    except (TypeError, ValueError):
        return None


def _int_id(value: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PFF game id {value!r} is not an integer") from exc


def _period(value) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        period = int(value)
    except (TypeError, ValueError):
        return None
    return period if period >= 1 else None


def _frame_num(value) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("frameNum is missing")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    return int(str(value).strip())


def _elapsed(value) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("periodElapsedTime is missing")
    seconds = float(value)
    if not np.isfinite(seconds):
        raise ValueError("periodElapsedTime is not finite")
    return seconds


def _jersey(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return text


def _xy(raw: dict) -> tuple[float, float] | None:
    x, y = raw.get("x"), raw.get("y")
    if x is None or y is None or isinstance(x, bool) or isinstance(y, bool):
        return None
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf) or not np.isfinite(yf):
        return None
    return xf, yf


def _home_flag(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def _key(value) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return "" if text.casefold() == "none" else text


def key_paths(obj) -> list[str]:
    """Top-level keys and one nested level. No values."""
    if isinstance(obj, list):
        obj = next((item for item in obj if isinstance(item, dict)), None)
    if not isinstance(obj, dict):
        return []
    found: list[str] = []
    for key in sorted(obj, key=str):
        found.append(str(key))
        child = obj[key]
        if isinstance(child, dict):
            for key2 in sorted(child, key=str):
                found.append(f"{key}.{key2}")
        elif isinstance(child, list):
            first = next((item for item in child if isinstance(item, dict)), None)
            if isinstance(first, dict):
                for key2 in sorted(first, key=str):
                    found.append(f"{key}[].{key2}")
    return found


def _header_keys(paths: dict[str, Path | None]) -> dict[str, list[str]]:
    tracking = _first_object(paths["tracking"]) if paths.get("tracking") else None
    meta = (
        _meta_object(json.loads(paths["meta"].read_text(encoding="utf-8-sig")))
        if paths.get("meta")
        else None
    )  # type: ignore[union-attr]
    roster_payload = (
        json.loads(paths["roster"].read_text(encoding="utf-8-sig")) if paths.get("roster") else None  # type: ignore[union-attr]
    )
    roster = (
        roster_payload[0] if isinstance(roster_payload, list) and roster_payload else roster_payload
    )
    events = None
    if paths.get("events") is not None:
        events = json.loads(paths["events"].read_text(encoding="utf-8-sig"))  # type: ignore[union-attr]
    return {
        "tracking": key_paths(tracking),
        "metadata": key_paths(meta),
        "roster": key_paths(roster),
        "events": key_paths(events) if events is not None else [],
    }


def _first_object(path: Path | None):
    if path is None:
        return None
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                return row if isinstance(row, dict) else None
    return None


def _fmt_float(value, digits=4) -> str:
    if value is None:
        return "na"
    return f"{float(value):.{digits}f}"


def _fmt_counts(counts: dict | None) -> str:
    if not counts:
        return "none"
    return " ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _fmt_keys(keys: list[str]) -> str:
    return ",".join(keys) if keys else "none"


def _ball_shares(counts: dict | None) -> dict[str, float] | None:
    if not counts:
        return None
    total = sum(counts.values())
    if total <= 0:
        return None
    return {name: counts[name] / total for name in ("visible", "estimated", "missing")}


def format_probe(games: list[dict]) -> str:
    """Aggregate text for `--probe`. No coordinates, names or per-frame rows."""
    lines: list[str] = []
    for game in games:
        if game.get("error"):
            lines.append(f"pff {game['id']} ERROR {game['error']}")
            lines.append(f"  tracking_keys {_fmt_keys(game.get('tracking_keys') or [])}")
            lines.append(f"  metadata_keys {_fmt_keys(game.get('metadata_keys') or [])}")
            lines.append(f"  roster_keys {_fmt_keys(game.get('roster_keys') or [])}")
            lines.append(f"  event_keys {_fmt_keys(game.get('event_keys') or [])}")
            continue
        stats = game["stats"]
        kept = stats.get("kept_native") or {}
        kept_text = ",".join(f"{key}:{kept[key]}" for key in sorted(kept, key=int))
        shares = _ball_shares(stats.get("ball_raw_counts"))
        labels = game["labels"]
        lines.append(
            " ".join(
                [
                    f"pff {game['id']}",
                    f"native_frames={stats['native_frames']}",
                    f"measured_fps={_fmt_float(stats.get('measured_fps'), 3)}",
                    f"fps_path={stats.get('fps_path')}",
                    f"pitch_path={stats.get('pitch_path')}",
                    f"frames_25fps={game['frames_25fps']}",
                    f"kept={kept_text or 'none'}",
                    f"possessions={stats.get('possessions')}",
                    f"unknown_jerseys_raw={_fmt_num(stats.get('unknown_jerseys_raw'))}",
                    f"unknown_jerseys_smoothed={_fmt_num(stats.get('unknown_jerseys_smoothed'))}",
                    f"duplicate_players_raw={_fmt_num(stats.get('duplicate_players_raw'))}",
                    f"duplicate_players_smoothed={_fmt_num(stats.get('duplicate_players_smoothed'))}",
                    f"player_estimated_raw={_fmt_float(stats.get('player_estimated_raw'))}",
                    f"player_estimated_smoothed={_fmt_float(stats.get('player_estimated_smoothed'))}",
                    f"setpiece_rule={'yes' if stats.get('setpiece_rule') else 'no'}",
                    f"setpiece_field={stats.get('setpiece_field') or 'none'}",
                    f"vid_events={stats.get('vid_events')}",
                    f"vid_frames={stats.get('vid_frames')}",
                ]
            )
        )
        if shares is None:
            ball = "ball_visible=na ball_estimated=na ball_missing=na"
        else:
            ball = (
                f"ball_visible={shares['visible']:.4f} "
                f"ball_estimated={shares['estimated']:.4f} "
                f"ball_missing={shares['missing']:.4f}"
            )
        lines.append(
            "  "
            + ball
            + f" ball_smoothed_missing={_fmt_float(stats.get('ball_smoothed_missing_native'))}"
            + f" ball_missing_25fps={_fmt_float(stats.get('ball_missing_raw_25fps'))}"
            + f" ball_smoothed_missing_25fps={_fmt_float(stats.get('ball_smoothed_missing_25fps'))}"
        )
        lines.append("  labels " + " ".join(f"{name}={labels[name]:.4f}" for name in LABELS))
        lines.append(_quality_line(stats))
        lines.append(f"  game_event_type {_fmt_counts(stats.get('game_event_type'))}")
        lines.append(f"  setpiece_counts {_fmt_counts(stats.get('setpiece_counts'))}")
        lines.append(f"  tracking_keys {_fmt_keys(game.get('tracking_keys') or [])}")
        lines.append(f"  metadata_keys {_fmt_keys(game.get('metadata_keys') or [])}")
        lines.append(f"  roster_keys {_fmt_keys(game.get('roster_keys') or [])}")
        lines.append(f"  event_keys {_fmt_keys(game.get('event_keys') or [])}")
    lines.append(_format_total(games))
    return "\n".join(lines) + "\n"


def _fmt_num(value) -> str:
    return "na" if value is None else str(int(value))


def _fmt_opt_int(value) -> str:
    return "na" if value is None else str(int(value))


def _quality_line(stats: dict) -> str:
    return (
        "  "
        f"label_version={stats.get('label_version', 'sequence')} "
        f"quality={stats.get('quality', 'none')} "
        f"low_visible={_fmt_opt_int(stats.get('quality_low_visible'))} "
        f"edge={_fmt_opt_int(stats.get('quality_edge'))} "
        f"touch_changes={_fmt_opt_int(stats.get('touch_changes'))} "
        f"touch_moved={_fmt_opt_int(stats.get('touch_moved'))} "
        f"touch_shift_median_s={_fmt_float(stats.get('touch_shift_median_s'))} "
        f"touch_shift_iqr_s={_fmt_float(stats.get('touch_shift_iqr_s'))}"
    )


def _format_total(games: list[dict]) -> str:
    ok = [game for game in games if not game.get("error")]
    if not ok:
        return f"pff TOTAL games=0 errors={len(games)}"
    native = sum(game["stats"]["native_frames"] for game in ok)
    frames = sum(game["frames_25fps"] for game in ok)
    fps_num = 0.0
    fps_den = 0
    for game in ok:
        fps = game["stats"].get("measured_fps")
        if fps is not None:
            fps_num += float(fps) * game["stats"]["native_frames"]
            fps_den += game["stats"]["native_frames"]
    labels = {name: 0.0 for name in LABELS}
    for game in ok:
        for name in LABELS:
            labels[name] += game["labels"][name] * game["frames_25fps"]
    if frames:
        labels = {name: value / frames for name, value in labels.items()}
    ball = Counter()
    for game in ok:
        counts = game["stats"].get("ball_raw_counts") or {}
        ball.update(counts)
    ball_total = sum(ball.values())
    shares = (
        {name: ball[name] / ball_total for name in ("visible", "estimated", "missing")}
        if ball_total
        else None
    )
    possessions = sum(int(game["stats"].get("possessions") or 0) for game in ok)
    vid_events = sum(int(game["stats"].get("vid_events") or 0) for game in ok)
    vid_frames = sum(int(game["stats"].get("vid_frames") or 0) for game in ok)
    event_counts: Counter[str] = Counter()
    set_counts: Counter[str] = Counter()
    for game in ok:
        event_counts.update(game["stats"].get("game_event_type") or {})
        set_counts.update(game["stats"].get("setpiece_counts") or {})
    keys = {
        name: sorted({key for game in ok for key in (game.get(name) or [])})
        for name in ("tracking_keys", "metadata_keys", "roster_keys", "event_keys")
    }
    head = (
        f"pff TOTAL games={len(ok)} errors={len(games) - len(ok)} native_frames={native} "
        f"measured_fps={_fmt_float(fps_num / fps_den if fps_den else None, 3)} "
        f"frames_25fps={frames} possessions={possessions} "
        f"vid_events={vid_events} vid_frames={vid_frames}"
    )
    if shares is None:
        ball_line = "ball_visible=na ball_estimated=na ball_missing=na"
    else:
        ball_line = (
            f"ball_visible={shares['visible']:.4f} ball_estimated={shares['estimated']:.4f} "
            f"ball_missing={shares['missing']:.4f}"
        )
    label_line = "labels " + " ".join(f"{name}={labels[name]:.4f}" for name in LABELS)
    return "\n".join(
        [
            head,
            f"  {ball_line}",
            f"  {label_line}",
            f"  game_event_type {_fmt_counts(dict(event_counts))}",
            f"  setpiece_counts {_fmt_counts(dict(set_counts))}",
            "  "
            + _quality_line(
                {
                    "label_version": "mixed",
                    "quality": "mixed",
                    "quality_low_visible": sum(
                        int(game["stats"].get("quality_low_visible") or 0) for game in ok
                    ),
                    "quality_edge": sum(int(game["stats"].get("quality_edge") or 0) for game in ok),
                    "touch_changes": _sum_optional(ok, "touch_changes"),
                    "touch_moved": _sum_optional(ok, "touch_moved"),
                    "touch_shift_median_s": None,
                    "touch_shift_iqr_s": None,
                }
            ).lstrip(),
            f"  tracking_keys {_fmt_keys(keys['tracking_keys'])}",
            f"  metadata_keys {_fmt_keys(keys['metadata_keys'])}",
            f"  roster_keys {_fmt_keys(keys['roster_keys'])}",
            f"  event_keys {_fmt_keys(keys['event_keys'])}",
        ]
    )


def _sum_optional(games: list[dict], key: str):
    values = [game["stats"].get(key) for game in games]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def probe_games(
    ids: list[str],
    *,
    labels: str = "sequence",
    quality: str = "none",
    min_visible: int = 10,
    edge_s: float = 0.5,
) -> str:
    """Load each id (variant raw) and return the aggregate report."""
    rows = []
    discovered = discover_games()
    for gid in ids:
        entry = discovered.get(gid)
        paths = None
        keys = {"tracking": [], "metadata": [], "roster": [], "events": []}
        try:
            if entry is None:
                raise FileNotFoundError(f"PFF game {gid} not found")
            paths = file_paths(gid)
            keys = _header_keys(paths)
            bundle = load_pff(
                match_id=gid,
                variant="raw",
                labels=labels,
                quality=quality,
                min_visible=min_visible,
                edge_s=edge_s,
            )
            rows.append(
                {
                    "id": gid,
                    "stats": bundle.extras["pff"],
                    "labels": bundle.extras["label_share"],
                    "frames_25fps": bundle.meta.n_frames,
                    "tracking_keys": keys["tracking"],
                    "metadata_keys": keys["metadata"],
                    "roster_keys": keys["roster"],
                    "event_keys": keys["events"],
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "id": gid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "tracking_keys": keys["tracking"],
                    "metadata_keys": keys["metadata"],
                    "roster_keys": keys["roster"],
                    "event_keys": keys["events"],
                }
            )
    return format_probe(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PFF FC World Cup 2022 source.")
    parser.add_argument("--list", action="store_true", help="print game ids, one per line")
    parser.add_argument(
        "--probe", action="store_true", help="print aggregate schema and label stats"
    )
    parser.add_argument(
        "--probe-events",
        action="store_true",
        help="aggregate touch-event counts over the train split (see pff_events)",
    )
    parser.add_argument("--games", default=None, help="'all' or the first N ids in integer order")
    parser.add_argument("--labels", default="sequence", choices=_LABEL_MODES)
    parser.add_argument("--quality", default="none", choices=_QUALITY_MODES)
    parser.add_argument("--min-visible", type=int, default=10)
    parser.add_argument("--edge-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.probe_events:
        from .pff_events import main as events_main

        events_main(["--probe-events"] + (["--games", args.games] if args.games else []))
        return
    if not args.list and not args.probe:
        parser.print_help()
        raise SystemExit(2)
    ids = _limit_ids(game_ids("all", installed_only=True), args.games)
    if args.list:
        for gid in ids:
            print(gid)
    if args.probe:
        sys.stdout.write(
            probe_games(
                ids,
                labels=args.labels,
                quality=args.quality,
                min_visible=args.min_visible,
                edge_s=args.edge_s,
            )
        )


def _limit_ids(ids: list[str], spec: str | None) -> list[str]:
    if spec is None or spec == "all":
        return ids
    if not str(spec).isdigit():
        raise SystemExit(f"--games must be 'all' or a non-negative integer, got {spec!r}")
    return ids[: int(spec)]


if __name__ == "__main__":
    main()
