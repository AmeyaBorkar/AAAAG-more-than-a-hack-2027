"""IDSSE (DFL / Sportec open data, Bassek et al. 2025, CC BY 4.0): seven labelled OTHER matches.

Signals: the provider's per-frame `BallStatus` (1 alive, 0 dead) and `BallPossession` (1 home,
2 guest) on the ball's FrameSet -- the same mechanism our ground truth is built from -- plus TRACAB
optical positions of players and ball at 25 Hz. Labels: D where the ball is dead, else H/A from the
owning team, X outside the periods. An alive frame with no owner is labelled D and counted in
`extras["provenance"]["alive_without_owner"]` (0 on J03WMX).

Positions file: ``<FrameSet GameSection=... TeamId=... PersonId=...>`` per person and half, each a
run of ``<Frame N=... X=... Y=... S=... />`` with X/Y already in pitch-centred metres and S in km/h.
Frame N counts from 10000 in the first half and 100000 in the second. Every attribute is pulled
with one C-level regex pass per FrameSet (about 15 s for a 400 MB match, against about 3 min
through kloppy's per-frame objects), then placed on the frame axis with numpy.

Frame axis: the halves back to back, separated by `gap_frames` of X so no spell spans half-time.
Orientation as delivered. Team level only: referees are dropped and players become anonymous slots
(home first, then away, each sorted by PersonId; a substitute gets its own slot).

Where it fails or differs from BAR-ATH:
- the ball is never missing and freezes during stoppages (static on every dead frame), so it is a
  clean upper bound: degrade it (degrade-v1) before using it for selection;
- DFL possession is not the challenge's "controlled" possession: spells are shorter (#14);
- one provider and camera system for all seven matches.

Fetch first (pinned mirror revision, sha256-checked), then load:

    python -m aaaag.sources.idsse J03WMX J03WPY      # or no ids for all seven
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import AWAY, HOME, A, D, H, MatchBundle, MatchMeta, Period, Tracking, X
from ..degrade import degrade as degrade_bundle
from ..paths import cache_dir, external_dir
from ..registry import SOURCES

PARSER_VERSION = "1"  # bump whenever the arrays produced for the same files change
FPS = 25
KMH_TO_MPS = 1.0 / 3.6

REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"  # huggingface.co/datasets/pysport/idsse-data
MIRROR_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{revision}/{name}"
PROVENANCE = {
    "dataset": "IDSSE: Integrated Dataset of Synchronized Spatiotemporal and Event Data in Elite "
    "Soccer (Bassek et al. 2025, Scientific Data)",
    "doi": "10.6084/m9.figshare.28196177",
    "mirror": "huggingface.co/datasets/pysport/idsse-data",
    "revision": REVISION,
    "licence": "CC BY 4.0",
    "labeller": "Sportec Solutions for the DFL: TRACAB optical tracking with provider BallStatus "
    "and BallPossession",
}

# match id -> (competition, matchinformation sha256, positions sha256)
MATCHES = {
    "J03WMX": (
        "000001",
        "a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9",
        "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49",
    ),
    "J03WN1": (
        "000001",
        "5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c",
        "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec",
    ),
    "J03WOH": (
        "000002",
        "f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c",
        "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f",
    ),
    "J03WOY": (
        "000002",
        "4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9",
        "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b",
    ),
    "J03WPY": (
        "000002",
        "566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420",
        "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1",
    ),
    "J03WQQ": (
        "000002",
        "e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41",
        "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f",
    ),
    "J03WR9": (
        "000002",
        "429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01",
        "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269",
    ),
}

# Sealed exam (docs/decisions/2026-10-30-seal-the-idsse-exam-j03wqq-j03wn1.md): loaded only with
# exam=True, at the re-plan checkpoint and the freeze. Never for statistics, tuning or priors.
EXAM_MATCHES = ("J03WQQ", "J03WN1")
DEV_MATCHES = tuple(m for m in MATCHES if m not in EXAM_MATCHES)

SECTIONS = ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")  # in playing order
BALL_TEAM = "BALL"
REFEREE_TEAM = "referee"

_FRAMESET = re.compile(rb"<FrameSet ([^>]*)>")
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
_FRAME = re.compile(rb"<Frame ")
_COLUMNS = {
    name: re.compile(rb" " + name.encode() + rb'="([^"]*)"')
    for name in ("N", "X", "Y", "S", "BallPossession", "BallStatus")
}


@dataclass(frozen=True)
class MatchInfo:
    home_id: str
    away_id: str
    home: str
    away: str
    pitch_length: float
    pitch_width: float


def file_names(match_id: str) -> dict[str, str]:
    competition = _entry(match_id)[0]
    suffix = f"DFL-COM-{competition}_DFL-MAT-{match_id}.xml"
    return {
        "meta": f"DFL_02_01_matchinformation_{suffix}",
        "positions": f"DFL_04_03_positions_raw_observed_{suffix}",
    }


def file_paths(match_id: str) -> dict[str, Path]:
    folder = external_dir() / "idsse"
    return {kind: folder / name for kind, name in file_names(match_id).items()}


def fetch(match_id: str, *, force: bool = False) -> dict[str, Path]:
    """Download one match from the pinned mirror revision; every file is sha256-checked.

    A file already on disk is re-hashed, and fetched again if it no longer matches the pin.
    """
    _, meta_sha, positions_sha = _entry(match_id)
    paths = file_paths(match_id)
    for kind, sha in (("meta", meta_sha), ("positions", positions_sha)):
        if force or not paths[kind].is_file() or file_sha256(paths[kind]) != sha:
            url = MIRROR_URL.format(revision=REVISION, name=paths[kind].name)
            download(url, paths[kind], sha)
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str) -> Path:
    """Stream `url` to `dest`. A wrong hash raises and leaves nothing at `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"{dest.name}: sha256 {digest.hexdigest()} != pinned {sha256}")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def read_match_info(path: str | Path) -> MatchInfo:
    root = ET.parse(path).getroot()
    general = next(root.iter("General")).attrib
    environment = next(root.iter("Environment")).attrib
    return MatchInfo(
        home_id=general["HomeTeamId"],
        away_id=general["GuestTeamId"],
        home=general["HomeTeamName"],
        away=general["GuestTeamName"],
        pitch_length=float(environment["PitchX"]),
        pitch_width=float(environment["PitchY"]),
    )


def parse_positions(raw: bytes) -> list[dict]:
    """One dict per FrameSet: its header attributes plus numpy columns, without copying `raw`."""
    heads = list(_FRAMESET.finditer(raw))
    framesets = []
    for i, head in enumerate(heads):
        start, stop = head.end(), heads[i + 1].start() if i + 1 < len(heads) else len(raw)
        attrs = {k.decode(): v.decode() for k, v in _ATTR.findall(head.group(1))}
        is_ball = attrs["TeamId"] == BALL_TEAM
        names = ("N", "X", "Y", "S", "BallPossession", "BallStatus") if is_ball else "NXYS"
        n_frames = len(_FRAME.findall(raw, start, stop))
        columns = {}
        for name in names:
            values = _COLUMNS[name].findall(raw, start, stop)
            if len(values) != n_frames:
                raise ValueError(f"FrameSet {attrs}: {len(values)} {name} for {n_frames} frames")
            columns[name] = np.array(values, dtype=np.bytes_)
        framesets.append({"attrs": attrs, **columns})
    return framesets


def to_bundle(
    match_id: str, info: MatchInfo, framesets: list[dict], gap_frames: int
) -> MatchBundle:
    """Place every FrameSet on the frame axis (halves back to back, X gaps between them)."""
    balls = {
        fs["attrs"]["GameSection"]: fs for fs in framesets if fs["attrs"]["TeamId"] == BALL_TEAM
    }
    unknown = set(balls) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{match_id}: unknown GameSection {sorted(unknown)}")

    periods, period_of, first_n, cursor = [], {}, {}, 0
    for section in (s for s in SECTIONS if s in balls):
        n = balls[section]["N"].astype(np.int64)
        if len(n) == 0 or not (np.diff(n) == 1).all():
            raise ValueError(f"{match_id} {section}: ball frames are not one contiguous run")
        if periods:
            cursor += gap_frames
        first_n[section] = int(n[0]) - cursor
        periods.append(Period(len(periods) + 1, cursor, cursor + len(n)))
        period_of[section] = periods[-1]
        cursor += len(n)
    if not periods:
        raise ValueError(f"{match_id}: no ball FrameSet")
    T = cursor

    team_of = {info.home_id: HOME, info.away_id: AWAY}
    people = sorted(
        {
            (team_of[fs["attrs"]["TeamId"]], fs["attrs"]["PersonId"])
            for fs in framesets
            if fs["attrs"]["TeamId"] in team_of
        }
    )
    other = {fs["attrs"]["TeamId"] for fs in framesets} - set(team_of) - {BALL_TEAM, REFEREE_TEAM}
    if other:
        raise ValueError(f"{match_id}: FrameSets of unknown teams {sorted(other)}")
    slot_of = {person: slot for slot, (_, person) in enumerate(people)}
    S = len(people)

    player_xy = np.full((T, S, 2), np.nan, dtype=np.float32)
    player_speed = np.full((T, S), np.nan, dtype=np.float32)
    ball_xy = np.full((T, 2), np.nan, dtype=np.float32)
    labels = np.full(T, X, dtype=np.int8)
    alive_without_owner = 0
    for fs in framesets:
        attrs = fs["attrs"]
        if attrs["TeamId"] == REFEREE_TEAM or len(fs["N"]) == 0:
            continue
        section = attrs["GameSection"]
        if section not in period_of:
            raise ValueError(f"{match_id}: FrameSet {attrs} in a section without a ball")
        rows = fs["N"].astype(np.int64) - first_n[section]
        period = period_of[section]
        if rows.min() < period.start or rows.max() >= period.stop:
            raise ValueError(f"{match_id}: FrameSet {attrs} has frames outside its half")
        xy = np.stack([fs["X"].astype(np.float32), fs["Y"].astype(np.float32)], axis=1)
        if attrs["TeamId"] == BALL_TEAM:
            ball_xy[rows] = xy
            status = fs["BallStatus"].astype(np.int8)
            owner = fs["BallPossession"].astype(np.int8)
            alive = status == 1
            alive_without_owner += int((alive & ~np.isin(owner, (1, 2))).sum())
            labels[rows] = np.where(alive & (owner == 1), H, np.where(alive & (owner == 2), A, D))
        else:
            slot = slot_of[attrs["PersonId"]]
            player_xy[rows, slot] = xy
            player_speed[rows, slot] = fs["S"].astype(np.float32) * KMH_TO_MPS

    meta = MatchMeta(
        match_id=match_id,
        source="idsse",
        n_frames=T,
        fps=FPS,
        pitch_length=info.pitch_length,
        pitch_width=info.pitch_width,
        periods=tuple(periods),
        home=info.home,
        away=info.away,
    ).validate()
    tracking = Tracking(
        meta=meta,
        player_xy=player_xy,
        slot_team=np.array([team for team, _ in people], dtype=np.int8),
        ball_xy=ball_xy,
        player_speed=player_speed,
    ).validate()
    provenance = {**PROVENANCE, "match_id": match_id, "alive_without_owner": alive_without_owner}
    return MatchBundle(tracking=tracking, labels=labels, extras={"provenance": provenance})


def load_files(
    match_id: str,
    meta_path: str | Path,
    positions_path: str | Path,
    *,
    gap_frames: int = 25,
    cache_path: str | Path | None = None,
    exam: bool = False,
) -> MatchBundle:
    """Parse (or load from cache) one match's matchinformation + positions pair."""
    _check_exam(match_id, exam)
    meta_path, positions_path = Path(meta_path), Path(positions_path)
    stamp = {
        "parser_version": PARSER_VERSION,
        "meta": _stat(meta_path),
        "positions": _stat(positions_path),
        "gap_frames": gap_frames,
    }
    info = read_match_info(meta_path)
    if cache_path is not None:
        cached = _read_cache(Path(cache_path), stamp, match_id, info)
        if cached is not None:
            return cached
    framesets = parse_positions(positions_path.read_bytes())
    bundle = to_bundle(match_id, info, framesets, gap_frames).validate()
    if cache_path is not None:
        _write_cache(Path(cache_path), stamp, bundle)
    return bundle


@SOURCES.register("idsse", version=PARSER_VERSION)
def load_idsse(
    *,
    match_id: str,
    gap_frames: int = 25,
    use_cache: bool = True,
    exam: bool = False,
    degrade: dict | None = None,
) -> MatchBundle:
    """A labelled IDSSE match (fetch it first: `python -m aaaag.sources.idsse MATCH_ID`).

    The sealed exam matches load only with `exam=True`, so a loop over every match fails loudly.
    `degrade` ({version, seed}) applies aaaag.degrade; the match id becomes `<id>~d<v>s<seed>`.
    """
    _check_exam(match_id, exam)
    if os.environ.get(PARSED_ONLY_ENV) == "1":
        bundle = _load_parsed_only(match_id, gap_frames)
        return bundle if degrade is None else degrade_bundle(bundle, **degrade)
    paths = file_paths(match_id)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "IDSSE files not found:\n  " + "\n  ".join(missing) + "\nFetch them with: "
            f"python -m aaaag.sources.idsse {match_id}"
        )
    cache = (
        cache_dir() / match_id / f"idsse-v{PARSER_VERSION}-gap{gap_frames}.npz"
        if use_cache
        else None
    )
    bundle = load_files(
        match_id,
        paths["meta"],
        paths["positions"],
        gap_frames=gap_frames,
        cache_path=cache,
        exam=exam,
    )
    if degrade is not None:
        bundle = degrade_bundle(bundle, **degrade)
    return bundle


def _check_exam(match_id: str, exam: bool) -> None:
    if match_id in EXAM_MATCHES and not exam:
        raise PermissionError(
            f"{match_id} is a sealed exam match (docs/decisions/2026-10-30-seal-the-idsse-exam-"
            "j03wqq-j03wn1.md): pass exam=True only at the re-plan checkpoint or the freeze"
        )


def _entry(match_id: str) -> tuple[str, str, str]:
    if match_id not in MATCHES:
        raise KeyError(f"unknown IDSSE match {match_id!r}; known: {', '.join(MATCHES)}")
    return MATCHES[match_id]


PARSED_ONLY_ENV = "AAAAG_IDSSE_PARSED_ONLY"


def _load_parsed_only(match_id: str, gap_frames: int) -> MatchBundle:
    """Rented-GPU mode (`AAAAG_IDSSE_PARSED_ONLY=1`): a parsed cache built on another machine,
    trusted without the raw positions file (its stamp holds that machine's file sizes and mtimes,
    so it can never validate here). The cache must come from this parser version and gap; the
    small matchinformation file is still read. Saves shipping 2.5 GB of XML to every box."""
    paths = file_paths(match_id)
    cache = cache_dir() / match_id / f"idsse-v{PARSER_VERSION}-gap{gap_frames}.npz"
    sidecar = cache.with_suffix(".json")
    if not (paths["meta"].is_file() and cache.is_file() and sidecar.is_file()):
        raise FileNotFoundError(
            f"{match_id}: parsed-only mode needs {paths['meta']}, {cache} and {sidecar}"
        )
    stamp = json.loads(sidecar.read_text(encoding="utf-8")).get("stamp") or {}
    if stamp.get("parser_version") != PARSER_VERSION or stamp.get("gap_frames") != gap_frames:
        raise ValueError(
            f"{match_id}: cache from parser {stamp.get('parser_version')} and gap "
            f"{stamp.get('gap_frames')}, expected {PARSER_VERSION} and {gap_frames}"
        )
    bundle = _read_cache(cache, stamp, match_id, read_match_info(paths["meta"]))
    if bundle is None:
        raise ValueError(f"{match_id}: unreadable parsed cache {cache}")
    return bundle


def _read_cache(path: Path, stamp: dict, match_id: str, info: MatchInfo) -> MatchBundle | None:
    sidecar = path.with_suffix(".json")
    if not (path.is_file() and sidecar.is_file()):
        return None
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if saved.get("stamp") != stamp:
        return None
    with np.load(path, allow_pickle=False) as arrays:
        periods = tuple(
            Period(i + 1, int(start), int(stop))
            for i, (start, stop) in enumerate(arrays["periods"])
        )
        meta = MatchMeta(
            match_id=match_id,
            source="idsse",
            n_frames=int(arrays["labels"].shape[0]),
            fps=FPS,
            pitch_length=info.pitch_length,
            pitch_width=info.pitch_width,
            periods=periods,
            home=info.home,
            away=info.away,
        )
        tracking = Tracking(
            meta=meta,
            player_xy=arrays["player_xy"],
            slot_team=arrays["slot_team"],
            ball_xy=arrays["ball_xy"],
            player_speed=arrays["player_speed"],
        )
        bundle = MatchBundle(
            tracking=tracking, labels=arrays["labels"], extras={"provenance": saved["provenance"]}
        )
    return bundle.validate()


def _write_cache(path: Path, stamp: dict, bundle: MatchBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracking = bundle.tracking
    np.savez(
        path,
        player_xy=tracking.player_xy,
        slot_team=tracking.slot_team,
        ball_xy=tracking.ball_xy,
        player_speed=tracking.player_speed,
        labels=bundle.labels,
        periods=np.array([(p.start, p.stop) for p in bundle.meta.periods], dtype=np.int64),
    )
    sidecar = {"stamp": stamp, "provenance": bundle.extras["provenance"]}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def main(argv: list[str] | None = None) -> None:
    for match_id in (argv if argv is not None else sys.argv[1:]) or list(MATCHES):
        paths = fetch(match_id)
        sizes = ", ".join(
            f"{kind} {path.stat().st_size / 1e6:.1f} MB" for kind, path in paths.items()
        )
        print(f"{match_id}: ok ({sizes})")


if __name__ == "__main__":
    main()
