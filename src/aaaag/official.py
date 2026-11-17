"""The organisers' code, loaded from the byte-exact copy in vendor/mtah2027.

Everything that produces a score goes through `pm` (possession_metrics), so our numbers are the
organisers' numbers by construction. Never re-implement a metric: wrap it.

The vendored scripts import each other as top-level modules (`import possession_metrics`,
`from games import ...`), so they are registered in sys.modules under those names.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from .contracts import Period
from .paths import VENDOR_ROOT, check_layout

check_layout()

VALIDATOR = VENDOR_ROOT / "evaluation" / "validate_submission.py"
SCORER = VENDOR_ROOT / "evaluation" / "run_evaluation.py"
PUBLIC_GT = VENDOR_ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
BASELINE_RESULTS = VENDOR_ROOT / "evaluation" / "results"


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


games = _load("games", VENDOR_ROOT / "games.py")
pm = _load("possession_metrics", VENDOR_ROOT / "evaluation" / "possession_metrics.py")

GAME: str = games.GAME
N_FRAMES: int = games.GAMES[GAME]["n_video_frames"]
FPS: int = games.GAMES[GAME]["fps"]
PITCH: tuple[float, float] = tuple(games.GAMES[GAME]["pitch"])  # metres, measured by organisers


def graded_periods(game: str = GAME) -> tuple[Period, ...]:
    """The graded halves as half-open Periods, from the organisers' own games.half_video_blocks."""
    return tuple(
        Period(index=half, start=first, stop=last + 1)
        for half, first, last in games.half_video_blocks(game)
    )


def load_official_baseline() -> ModuleType:
    """starter/baseline.py as a module (for parity tests). Imported under a namespaced name."""
    return _load("mtah2027_baseline", VENDOR_ROOT / "starter" / "baseline.py")


def upstream_manifest() -> dict:
    return json.loads((VENDOR_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))


def verify_vendor() -> list[str]:
    """Vendored files whose bytes differ from UPSTREAM.json. Empty list = byte-exact."""
    bad = []
    for relative, digest in upstream_manifest()["files"].items():
        path = VENDOR_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(relative)
    return bad
