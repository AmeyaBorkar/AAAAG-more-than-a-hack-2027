"""Fast reader for Metrica's FIFA/EPTS tracking export (the ATD), with an on-disk cache.

Raw line format: ``<frame_id>:<slot>;<slot>;...;<slot>:<ball>`` where every slot group and the ball
group hold comma-separated channels (players: x, y, speed) and a missing observation is ``NaN``.

Metadata XML: Teams (first = home, the EPTS convention the official baseline uses), one Player per
track slot, PlayerChannel elements mapping channel ids (``player7_x``) to players, and one or more
DataFormatSpecification blocks, each covering a frame range and listing PlayerChannelRef /
BallChannelRef in line order. Metrica's public sample game 3 has eleven such blocks with different
player sets, so layouts are resolved per block and mapped onto global slots (Player order); the
competition file is documented as 46 slots throughout, in which case this reduces to the official
baseline's mapping.

Speed: the official baseline streams the file in Python (~1-2 min). Here separators are translated
to commas in one C-level pass and each block goes through pandas' C parser (seconds), after a
cheap check that every line has the expected field count. Anything irregular falls back to a
strict line-by-line parser, so a format surprise costs time, never correctness. Parsed arrays are
cached as .npz keyed on source size + mtime + PARSER_VERSION.

Output: a `Tracking` in contract units -- metres, pitch-centred, float32, NaN for missing, indexed
by video frame (= frame_id - 1).
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import AWAY, HOME, MatchMeta, Tracking

PARSER_VERSION = "2"  # bump whenever the arrays produced for the same files change
_TO_COMMA = bytes.maketrans(b":;", b",,")
_SPEED_NAMES = ("speed", "s", "velocity", "v")


@dataclass(frozen=True)
class SpecLayout:
    """One DataFormatSpecification: which global slot each line group is, and the channel names."""

    start_frame: int | None  # inclusive frame_id range; None = whole file
    end_frame: int | None
    slots: tuple[int, ...]
    player_channels: tuple[str, ...]
    ball_channels: tuple[str, ...]

    @property
    def n_columns(self) -> int:
        return 1 + len(self.slots) * len(self.player_channels) + len(self.ball_channels)


@dataclass(frozen=True)
class AtdMetadata:
    team_names: tuple[str, str]
    slot_team: np.ndarray  # (S,) int8, global slots in Player order
    specs: tuple[SpecLayout, ...]  # empty when the XML declares no resolvable layout
    frame_rate: int | None


def read_metadata(meta_path: str | Path) -> AtdMetadata:
    """Teams, slot -> team, and per-block layouts from the EPTS metadata XML."""
    root = ET.parse(meta_path).getroot()
    teams = [
        (team.get("id"), (team.findtext("Name") or "").strip()) for team in _iter(root, "Team")
    ]
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(teams)}")
    players = [(player.get("id"), player.get("teamId")) for player in _iter(root, "Player")]
    if not players:
        raise ValueError(f"{meta_path}: no Player entries")
    slot_of = {player_id: slot for slot, (player_id, _) in enumerate(players)}
    slot_team = np.array(
        [HOME if team == teams[0][0] else AWAY for _, team in players], dtype=np.int8
    )

    channel_map = {
        ch.get("id"): (ch.get("playerId"), (ch.get("channelId") or "").lower())
        for ch in _iter(root, "PlayerChannel")
    }
    specs = []
    for spec in _iter(root, "DataFormatSpecification"):
        layout = _spec_layout(spec, channel_map, slot_of)
        if layout is None:
            specs = []
            break
        specs.append(layout)

    rate = next((el.text for el in _iter(root, "FrameRate") if el.text), None)
    return AtdMetadata(
        team_names=(teams[0][1], teams[1][1]),
        slot_team=slot_team,
        specs=tuple(specs),
        frame_rate=int(float(rate)) if rate else None,
    )


def _spec_layout(spec: ET.Element, channel_map: dict, slot_of: dict) -> SpecLayout | None:
    """Layout of one block, or None if its channel refs cannot be resolved to players."""
    refs = [ref.get("playerChannelId") for ref in _iter(spec, "PlayerChannelRef")]
    slots: list[int] = []
    channels: list[str] = []
    for ref in refs:
        player_id, channel = channel_map.get(ref, (None, None))
        if player_id not in slot_of:
            return None
        slot = slot_of[player_id]
        if not slots or slots[-1] != slot:
            slots.append(slot)
        if len(slots) == 1:
            channels.append(channel)
    if not slots or len(refs) != len(slots) * len(channels):
        return None
    ball = tuple((ref.get("channelId") or "").lower() for ref in _iter(spec, "BallChannelRef"))
    start, end = spec.get("startFrame"), spec.get("endFrame")
    return SpecLayout(
        start_frame=int(start) if start else None,
        end_frame=int(end) if end else None,
        slots=tuple(slots),
        player_channels=tuple(channels),
        ball_channels=ball or ("x", "y"),
    )


def infer_layout(raw_path: str | Path, n_slots: int) -> SpecLayout:
    """Single-block layout from the first data line, for XMLs that declare none."""
    with open(raw_path, encoding="ascii") as handle:
        line = next((text.strip() for text in handle if text.strip()), None)
    if line is None:
        raise ValueError(f"{raw_path}: empty file")
    parts = line.split(":")
    if len(parts) != 3:
        raise ValueError(f"{raw_path}: expected 'frame:players:ball', got {len(parts)} parts")
    groups = parts[1].split(";")
    if len(groups) != n_slots:
        raise ValueError(
            f"{raw_path}: {len(groups)} slot groups but the XML lists {n_slots} players"
        )
    n_player, n_ball = len(groups[0].split(",")), len(parts[2].split(","))
    return SpecLayout(
        start_frame=None,
        end_frame=None,
        slots=tuple(range(n_slots)),
        player_channels=("x", "y", "speed", *(f"c{i}" for i in range(3, n_player)))[:n_player],
        ball_channels=("x", "y", *(f"c{i}" for i in range(2, n_ball)))[:n_ball],
    )


def parse_blocks(raw: bytes, specs: tuple[SpecLayout, ...]) -> list[tuple[SpecLayout, np.ndarray]]:
    """[(layout, (N, n_columns) float64 table)] -- frame_id, slot channels, ball channels."""
    if len(specs) == 1:
        return [(specs[0], _parse_block(raw, specs[0]))]
    lines = [line for line in raw.splitlines() if line.strip()]
    ids = np.fromiter(
        (int(line[: line.index(b":")]) for line in lines), dtype=np.int64, count=len(lines)
    )
    out, used = [], np.zeros(len(lines), dtype=bool)
    for spec in specs:
        low = ids.min() if spec.start_frame is None else spec.start_frame
        high = ids.max() if spec.end_frame is None else spec.end_frame
        pick = (ids >= low) & (ids <= high) & ~used
        if pick.any():
            used |= pick
            block = b"\n".join(lines[i] for i in np.flatnonzero(pick)) + b"\n"
            out.append((spec, _parse_block(block, spec)))
    if not used.all():
        raise ValueError(
            f"{int((~used).sum())} lines fall outside every DataFormatSpecification range"
        )
    return out


def _parse_block(raw: bytes, spec: SpecLayout) -> np.ndarray:
    table = _parse_fast(raw, spec.n_columns)
    return table if table is not None else _parse_slow(raw, spec.n_columns)


def _parse_fast(raw: bytes, n_columns: int) -> np.ndarray | None:
    """pandas C parser on comma-translated bytes, or None if the lines are not uniform."""
    n_lines = sum(1 for line in raw.splitlines() if line.strip())
    n_separators = raw.count(b":") + raw.count(b";") + raw.count(b",")
    if n_lines == 0 or n_separators != n_lines * (n_columns - 1):
        return None
    try:
        frame = pd.read_csv(
            io.BytesIO(raw.translate(_TO_COMMA)),
            header=None,
            names=list(range(n_columns)),
            dtype=np.float64,
            engine="c",
        )
    except (ValueError, pd.errors.ParserError):
        return None
    return frame.to_numpy(dtype=np.float64)


def _parse_slow(raw: bytes, n_columns: int) -> np.ndarray:
    """Line-by-line reference parser. Strict: a malformed line is an error, never a skip."""
    rows = []
    for number, line in enumerate(raw.decode("ascii").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) != 3:
            raise ValueError(f"line {number}: expected 3 ':'-separated parts, got {len(parts)}")
        fields = [parts[0]]
        for group in parts[1].split(";"):
            fields.extend(group.split(","))
        fields.extend(parts[2].split(","))
        if len(fields) != n_columns:
            raise ValueError(f"line {number}: {len(fields)} fields, expected {n_columns}")
        rows.append([float(value) for value in fields])
    return np.asarray(rows, dtype=np.float64).reshape(-1, n_columns)


def to_tracking(
    blocks: list[tuple[SpecLayout, np.ndarray]], metadata: AtdMetadata, meta: MatchMeta
) -> Tracking:
    """Place rows by frame_id (video frame = frame_id - 1) on global slots, in centred metres."""
    T, S = meta.n_frames, len(metadata.slot_team)
    L, W = meta.pitch_length, meta.pitch_width
    player_xy = np.full((T, S, 2), np.nan, dtype=np.float32)
    has_speed = any(_index(spec.player_channels, _SPEED_NAMES) is not None for spec, _ in blocks)
    player_speed = np.full((T, S), np.nan, dtype=np.float32) if has_speed else None
    ball_xy = np.full((T, 2), np.nan, dtype=np.float32)
    ball_extra: dict[str, np.ndarray] = {}
    filled = np.zeros(T, dtype=bool)

    for spec, table in blocks:
        video = table[:, 0].astype(np.int64) - 1
        if (video < 0).any() or (video >= T).any():
            raise ValueError(f"frame ids outside 1..{T}")
        if filled[video].any() or len(np.unique(video)) != len(video):
            raise ValueError("duplicate frame ids in raw file")
        filled[video] = True

        g, c = len(spec.slots), len(spec.player_channels)
        players = table[:, 1 : 1 + g * c].reshape(-1, g, c)
        rows, cols = video[:, None], np.asarray(spec.slots)[None, :]
        ix, iy = spec.player_channels.index("x"), spec.player_channels.index("y")
        player_xy[rows, cols, 0] = (players[:, :, ix] - 0.5) * L
        player_xy[rows, cols, 1] = (players[:, :, iy] - 0.5) * W
        i_speed = _index(spec.player_channels, _SPEED_NAMES)
        if i_speed is not None:
            player_speed[rows, cols] = players[:, :, i_speed]

        ball = table[:, 1 + g * c :]
        bx, by = spec.ball_channels.index("x"), spec.ball_channels.index("y")
        ball_xy[video, 0] = (ball[:, bx] - 0.5) * L
        ball_xy[video, 1] = (ball[:, by] - 0.5) * W
        for i, name in enumerate(spec.ball_channels):
            if i not in (bx, by):
                ball_extra.setdefault(name, np.full(T, np.nan, dtype=np.float32))[video] = ball[
                    :, i
                ]

    return Tracking(
        meta=meta,
        player_xy=player_xy,
        slot_team=metadata.slot_team.astype(np.int8),
        ball_xy=ball_xy,
        player_speed=player_speed,
        ball_extra=ball_extra,
    ).validate()


def load_tracking(
    meta_path: str | Path,
    raw_path: str | Path,
    meta: MatchMeta,
    cache_path: str | Path | None = None,
) -> Tracking:
    """Parse (or load from cache) one ATD pair into a validated Tracking."""
    meta_path, raw_path = Path(meta_path), Path(raw_path)
    stamp = {
        "parser_version": PARSER_VERSION,
        "meta": _stat(meta_path),
        "raw": _stat(raw_path),
        "match": repr(meta),
    }
    if cache_path is not None:
        cached = _read_cache(Path(cache_path), stamp, meta)
        if cached is not None:
            return cached

    metadata = read_metadata(meta_path)
    specs = metadata.specs or (infer_layout(raw_path, len(metadata.slot_team)),)
    tracking = to_tracking(parse_blocks(raw_path.read_bytes(), specs), metadata, meta)
    if cache_path is not None:
        _write_cache(Path(cache_path), stamp, tracking)
    return tracking


def describe_layout(meta_path: str | Path) -> dict:
    """Aggregate facts about the declared layout -- safe to print in an agent session."""
    metadata = read_metadata(meta_path)
    return {
        "n_slots": len(metadata.slot_team),
        "home_slots": int((metadata.slot_team == HOME).sum()),
        "away_slots": int((metadata.slot_team == AWAY).sum()),
        "n_blocks": len(metadata.specs),
        "player_channels": sorted({ch for s in metadata.specs for ch in s.player_channels}),
        "ball_channels": sorted({ch for s in metadata.specs for ch in s.ball_channels}),
        "slot_order_matches_player_order": all(
            s.slots == tuple(range(len(s.slots))) for s in metadata.specs
        ),
        "frame_rate": metadata.frame_rate,
    }


def _read_cache(path: Path, stamp: dict, meta: MatchMeta) -> Tracking | None:
    sidecar = path.with_suffix(".json")
    if not (path.is_file() and sidecar.is_file()):
        return None
    if json.loads(sidecar.read_text(encoding="utf-8")) != stamp:
        return None
    with np.load(path, allow_pickle=False) as arrays:
        prefix = "ball_extra__"
        extra = {key[len(prefix) :]: arrays[key] for key in arrays.files if key.startswith(prefix)}
        return Tracking(
            meta=meta,
            player_xy=arrays["player_xy"],
            slot_team=arrays["slot_team"],
            ball_xy=arrays["ball_xy"],
            player_speed=arrays["player_speed"] if "player_speed" in arrays.files else None,
            ball_extra=extra,
        ).validate()


def _write_cache(path: Path, stamp: dict, tracking: Tracking) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "player_xy": tracking.player_xy,
        "slot_team": tracking.slot_team,
        "ball_xy": tracking.ball_xy,
    }
    if tracking.player_speed is not None:
        arrays["player_speed"] = tracking.player_speed
    for name, channel in tracking.ball_extra.items():
        arrays[f"ball_extra__{name}"] = channel
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def _iter(root: ET.Element, local_name: str):
    """Elements by local tag name, ignoring any XML namespace."""
    for element in root.iter():
        tag = element.tag
        if isinstance(tag, str) and (tag == local_name or tag.endswith("}" + local_name)):
            yield element


def _index(names: tuple[str, ...], wanted: tuple[str, ...]) -> int | None:
    return next((i for i, name in enumerate(names) if name in wanted), None)
