"""Metrica Sports sample data: three anonymised matches tracked by the provider of the ATD.

github.com/metrica-sports/sample-data, pinned to one commit and sha256-checked. There is no formal
licence; the README asks users to "be responsible" and to acknowledge the source when the data is
used publicly (decision 2026-11-05-degrade-from-external-data-only-and-keep-metrica). Credit
Metrica Sports in the technical document; never redistribute the files.

Why it matters: Metrica produced BAR-ATH's tracking (the ATD is a Metrica EPTS export, io/atd.py),
so these games show the same kind of tracking failures. In particular the ball is really missing
on about a third of their frames. That makes them external evidence for the degrade simulator.

Tracking:
- games 1 and 2 are two CSVs (home, away) with normalised x, y per player and for the ball;
- game 3 is EPTS (metadata XML plus raw text, eleven layout blocks), read with io/atd.py;
- both become pitch-centred metres on an assumed 105 x 68 m pitch (the files carry no size), with
  the same axes as io/atd.py; frames as delivered (video frame = frame id - 1), halves adjacent.

Periods: games 1 and 2 from the CSV's Period column; game 3 from its events (first event start to
last event end of each half).

Labels, built from the events and *approximate* (EXTERNAL_DATA.md):
- D from the end of the event before each SET PIECE to the SET PIECE's start (ball out, foul,
  goal, offside: any stoppage that ends in a restart), and before each period's first event;
- otherwise the team of the latest PASS / CARRY / RECOVERY / SHOT / BALL LOST / SET PIECE that
  started at or before the frame;
- X outside the periods.

Where it fails: event frames are hand-tagged, so label boundaries are off by a few frames;
stoppages without a recorded restart stay alive; contested balls go to the last team that acted.

Fetch first, then load:

    python -m aaaag.sources.metrica          # all three games
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import AWAY, HOME, A, D, H, MatchBundle, MatchMeta, Period, Tracking, X
from ..degrade import degrade as degrade_bundle
from ..io import atd
from ..paths import cache_dir, external_dir
from ..registry import SOURCES
from .idsse import download, file_sha256

PARSER_VERSION = "1"  # bump whenever the arrays produced for the same files change
FPS = 25
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"  # github.com/metrica-sports/sample-data
RAW_URL = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/{revision}/data/{game}/{name}"
)
PROVENANCE = {
    "dataset": "Metrica Sports sample data",
    "repository": "github.com/metrica-sports/sample-data",
    "revision": REVISION,
    "licence": "none stated; README: be responsible, acknowledge the source",
    "labeller": "Metrica Sports: automatic tracking, hand-tagged events (labels derived here)",
}

# game -> {file name: sha256}
FILES = {
    "1": {
        "Sample_Game_1_RawTrackingData_Home_Team.csv": "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
        "Sample_Game_1_RawTrackingData_Away_Team.csv": "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
        "Sample_Game_1_RawEventsData.csv": "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    },
    "2": {
        "Sample_Game_2_RawTrackingData_Home_Team.csv": "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
        "Sample_Game_2_RawTrackingData_Away_Team.csv": "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
        "Sample_Game_2_RawEventsData.csv": "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    },
    "3": {
        "Sample_Game_3_metadata.xml": "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
        "Sample_Game_3_tracking.txt": "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
        "Sample_Game_3_events.json": "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    },
}
GAMES = tuple(FILES)
OWNING = ("PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE")
SKIPPED = ("CARD",)  # bookkeeping, not play: never the event "before" a restart


def file_paths(game: str) -> dict[str, Path]:
    folder = external_dir() / "metrica"
    return {name: folder / name for name in _files(game)}


def fetch(game: str, *, force: bool = False) -> dict[str, Path]:
    """Download one game at the pinned revision; every file is sha256-checked."""
    paths = file_paths(game)
    for name, sha in _files(game).items():
        path = paths[name]
        if force or not path.is_file() or file_sha256(path) != sha:
            url = RAW_URL.format(revision=REVISION, game=f"Sample_Game_{game}", name=name)
            download(url, path, sha)
    return paths


def read_events(game: str, path: str | Path) -> pd.DataFrame:
    """Events as columns type, period, start, end (video frames), team (HOME/AWAY), subtypes."""
    if game == "3":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        home_id = raw["metadata"]["teams"][0]["id"] if "teams" in raw.get("metadata", {}) else None
        rows = []
        for e in raw["data"]:
            team_id = e["team"]["id"]
            subtypes = e["subtypes"] or []
            subtypes = [subtypes] if isinstance(subtypes, dict) else subtypes
            rows.append(
                {
                    "type": e["type"]["name"],
                    "period": int(e["period"]),
                    "start": int(e["start"]["frame"]) - 1,
                    "end": int(e["end"]["frame"]) - 1,
                    "team": HOME if team_id == (home_id or "FIFATMA") else AWAY,
                    "subtypes": "-".join(s["name"] for s in subtypes),
                }
            )
        events = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(path)
        events = pd.DataFrame(
            {
                "type": raw["Type"].str.upper(),
                "period": raw["Period"].astype(int),
                "start": raw["Start Frame"].astype(int) - 1,
                "end": raw["End Frame"].astype(int) - 1,
                "team": np.where(raw["Team"].str.lower() == "home", HOME, AWAY),
                "subtypes": raw["Subtype"].fillna("").str.upper(),
            }
        )
    events["end"] = np.maximum(events["end"], events["start"])  # some ends are 0 (instant events)
    return events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)


def read_csv_tracking(home_path: str | Path, away_path: str | Path) -> tuple[np.ndarray, ...]:
    """(player_xy (T,S,2), slot_team (S,), ball_xy (T,2), period (T,)) from the two CSVs."""
    xys, teams, ball, period = [], [], None, None
    for path, team in ((home_path, HOME), (away_path, AWAY)):
        names = pd.read_csv(path, skiprows=2, nrows=0).columns
        table = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=np.float64)
        frames = table[:, 1].astype(np.int64)
        if not (frames == np.arange(1, len(frames) + 1)).all():
            raise ValueError(f"{Path(path).name}: frames are not 1..T")
        players = [i for i, n in enumerate(names) if str(n).startswith("Player")]
        xys.append(np.stack([table[:, players], table[:, [i + 1 for i in players]]], axis=-1))
        teams += [team] * len(players)
        i_ball = list(names).index("Ball")
        team_ball = table[:, [i_ball, i_ball + 1]]
        if ball is not None and not np.array_equal(ball, team_ball, equal_nan=True):
            raise ValueError("home and away files disagree on the ball")
        ball, period = team_ball, table[:, 0].astype(np.int64)
    if len(xys[0]) != len(xys[1]):
        raise ValueError("home and away files have different lengths")
    player_xy = np.concatenate(xys, axis=1)
    return player_xy, np.array(teams, dtype=np.int8), ball, period


def to_metres(xy: np.ndarray) -> np.ndarray:
    """Normalised [0, 1] coordinates to pitch-centred metres, axes as in io/atd.py."""
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH_LENGTH
    out[..., 1] = (xy[..., 1] - 0.5) * PITCH_WIDTH
    return out


def event_labels(events: pd.DataFrame, periods: tuple[Period, ...], n_frames: int) -> np.ndarray:
    """(T,) int8 H/A/D/X from the events (rules in the module docstring)."""
    labels = np.full(n_frames, X, dtype=np.int8)
    owning = events[events["type"].isin(OWNING)]
    starts = owning["start"].to_numpy()
    teams = owning["team"].to_numpy()
    frames = np.arange(n_frames)
    latest = np.searchsorted(starts, frames, side="right") - 1
    owner = np.where(latest >= 0, teams[np.maximum(latest, 0)], -1)
    played = np.zeros(n_frames, dtype=bool)
    for period in periods:
        played[period.start : period.stop] = True
    labels[played & (owner == HOME)] = H
    labels[played & (owner == AWAY)] = A
    labels[played & (owner < 0)] = D

    play = events[~events["type"].isin(SKIPPED)]
    for period in periods:
        local = play[(play["start"] >= period.start) & (play["start"] < period.stop)]
        if len(local):
            labels[period.start : int(local["start"].iloc[0])] = D  # before the first event
        ends = local["end"].to_numpy()
        for i in np.flatnonzero((local["type"] == "SET PIECE").to_numpy()):
            restart = int(local["start"].iloc[i])
            before = ends[:i][ends[:i] <= restart]
            if len(before):
                labels[int(before.max()) : restart] = D
    return labels


def load_files(game: str, paths: dict[str, Path], *, cache_path: Path | None = None) -> MatchBundle:
    """Parse (or load from cache) one game into a labelled MatchBundle."""
    names = list(_files(game))
    stamp = {"parser_version": PARSER_VERSION, "files": [_stat(paths[n]) for n in names]}
    if cache_path is not None and (cached := _read_cache(cache_path, stamp, game)) is not None:
        return cached
    events = read_events(game, paths[names[2]])
    match_id = f"metrica-{game}"
    if game == "3":
        bounds = events.groupby("period").agg(start=("start", "min"), stop=("end", "max"))
        n_frames = _count_lines(paths[names[1]])
        periods = tuple(
            Period(int(p), int(row.start), min(int(row.stop) + 1, n_frames))
            for p, row in bounds.iterrows()
        )
        meta = _meta(match_id, n_frames, periods)
        tracking = atd.load_tracking(paths[names[0]], paths[names[1]], meta)
        tracking = Tracking(
            meta=meta,
            player_xy=tracking.player_xy,
            slot_team=tracking.slot_team,
            ball_xy=tracking.ball_xy,
            player_speed=tracking.player_speed,
        )
    else:
        player_xy, slot_team, ball, period_col = read_csv_tracking(paths[names[0]], paths[names[1]])
        n_frames = len(ball)
        periods = tuple(
            Period(
                int(p),
                int(np.flatnonzero(period_col == p)[0]),
                int(np.flatnonzero(period_col == p)[-1]) + 1,
            )
            for p in np.unique(period_col)
        )
        meta = _meta(match_id, n_frames, periods)
        tracking = Tracking(
            meta=meta, player_xy=to_metres(player_xy), slot_team=slot_team, ball_xy=to_metres(ball)
        )
    labels = event_labels(events, meta.periods, meta.n_frames)
    bundle = MatchBundle(
        tracking=tracking.validate(),
        labels=labels,
        extras={"provenance": {**PROVENANCE, "game": game}},
    ).validate()
    if cache_path is not None:
        _write_cache(cache_path, stamp, bundle)
    return bundle


@SOURCES.register("metrica", version=PARSER_VERSION)
def load_metrica(
    *, game: str | int, use_cache: bool = True, degrade: dict | None = None
) -> MatchBundle:
    """A Metrica sample game with event-derived labels (fetch first: `python -m
    aaaag.sources.metrica`). `degrade` ({version, seed, players}) applies aaaag.degrade."""
    game = str(game)
    paths = file_paths(game)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Metrica files not found:\n  "
            + "\n  ".join(missing)
            + "\nFetch them with: python -m aaaag.sources.metrica"
        )
    cache = (
        cache_dir() / f"metrica-{game}" / f"metrica-v{PARSER_VERSION}.npz" if use_cache else None
    )
    bundle = load_files(game, paths, cache_path=cache)
    if degrade is not None:
        bundle = degrade_bundle(bundle, **degrade)
    return bundle


def _files(game: str) -> dict[str, str]:
    if game not in FILES:
        raise KeyError(f"unknown Metrica game {game!r}; known: {', '.join(GAMES)}")
    return FILES[game]


def _meta(match_id: str, n_frames: int, periods: tuple[Period, ...]) -> MatchMeta:
    return MatchMeta(
        match_id=match_id,
        source="metrica",
        n_frames=n_frames,
        fps=FPS,
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        periods=periods,
    ).validate()


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))


def _read_cache(path: Path, stamp: dict, game: str) -> MatchBundle | None:
    sidecar = path.with_suffix(".json")
    if not (path.is_file() and sidecar.is_file()):
        return None
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if saved.get("stamp") != stamp:
        return None
    with np.load(path, allow_pickle=False) as arrays:
        periods = tuple(Period(i + 1, int(a), int(b)) for i, (a, b) in enumerate(arrays["periods"]))
        meta = _meta(f"metrica-{game}", int(arrays["labels"].shape[0]), periods)
        tracking = Tracking(
            meta=meta,
            player_xy=arrays["player_xy"],
            slot_team=arrays["slot_team"],
            ball_xy=arrays["ball_xy"],
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
        labels=bundle.labels,
        periods=np.array([(p.start, p.stop) for p in bundle.meta.periods], dtype=np.int64),
    )
    sidecar = {"stamp": stamp, "provenance": bundle.extras["provenance"]}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def main(argv: list[str] | None = None) -> None:
    for game in (argv if argv is not None else sys.argv[1:]) or list(GAMES):
        paths = fetch(str(game))
        total = sum(p.stat().st_size for p in paths.values()) / 1e6
        print(f"metrica game {game}: ok ({total:.1f} MB)")


if __name__ == "__main__":
    main()
