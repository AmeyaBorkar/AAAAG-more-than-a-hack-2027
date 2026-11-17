"""The competition match, FC Barcelona vs Athletic Club, tactical camera.

The bundle carries tracking and Smart Tagging and NEVER labels: the public five-minute ground
truth is only ever read by aaaag.evaluation.smoke, so no stage can train or tune on it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .. import official
from ..contracts import AWAY, HOME, MatchBundle, MatchMeta
from ..io import atd, smart_tagging
from ..paths import CONFIGS_DIR, cache_dir, data_dir
from ..registry import SOURCES

MATCH_CONFIG = CONFIGS_DIR / "matches" / "bar-ath.yaml"


def match_config() -> dict:
    with open(MATCH_CONFIG, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bar_ath_meta() -> MatchMeta:
    """Constants from the organisers' games.py: frame count, fps, measured pitch, graded halves."""
    config = match_config()
    return MatchMeta(
        match_id=official.GAME,
        source="bar-ath",
        n_frames=official.N_FRAMES,
        fps=official.FPS,
        pitch_length=float(official.PITCH[0]),
        pitch_width=float(official.PITCH[1]),
        periods=official.graded_periods(),
        home=config["home"],
        away=config["away"],
    ).validate()


def asset_paths() -> dict[str, Path]:
    """Tactical-source asset paths under $MTAH_DATA_DIR or ./data (built by the organisers' code)."""
    paths = official.games.asset_paths(official.GAME, data_dir=str(data_dir()))
    return {name: Path(path) for name, path in paths.items()}


def team_aliases() -> dict[int, list[str]]:
    aliases = match_config()["team_aliases"]
    return {HOME: list(aliases["home"]), AWAY: list(aliases["away"])}


@SOURCES.register("bar-ath", version="1")
def load_bar_ath(*, with_tags: bool = True, use_cache: bool = True) -> MatchBundle:
    meta = bar_ath_meta()
    paths = asset_paths()
    missing = [str(paths[name]) for name in ("atd_meta", "atd_raw") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "match data not found:\n  " + "\n  ".join(missing) + "\nUnpack the bundle outside "
            "the repository and point MTAH_DATA_DIR at it (docs/ONBOARDING.md section 4)."
        )
    cache = (
        cache_dir() / meta.match_id / f"tracking-v{atd.PARSER_VERSION}.npz" if use_cache else None
    )
    tracking = atd.load_tracking(paths["atd_meta"], paths["atd_raw"], meta, cache_path=cache)
    tags = None
    if with_tags and paths["smart_tagging"].is_file():
        tags = smart_tagging.load_tags(
            paths["smart_tagging"], meta.fps, meta.n_frames, team_aliases()
        )
    return MatchBundle(tracking=tracking, tags=tags).validate()
