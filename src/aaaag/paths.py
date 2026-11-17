"""Filesystem locations. The one module that spells a path out, so nothing else has to.

The package must be installed in editable mode (`pip install -e .`): paths are resolved from the
source tree, because the vendored scorer, configs and registries live beside the code.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor" / "mtah2027"
CONFIGS_DIR = REPO_ROOT / "configs"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"  # records in experiments/<member>/
FINDINGS_DIR = REPO_ROOT / "analysis" / "findings"
DECISIONS_DIR = REPO_ROOT / "docs" / "decisions"
HANDOVER_DIR = REPO_ROOT / "docs" / "handover"
STATUS_DIR = REPO_ROOT / "docs" / "status"
TEAM_DIR = REPO_ROOT / "team"


def data_dir() -> Path:
    """Root of the gated match bundle: $MTAH_DATA_DIR, else ./data (the upstream convention)."""
    env = os.environ.get("MTAH_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data"


def external_dir() -> Path:
    """Downloaded datasets of OTHER matches: $AAAAG_EXTERNAL_DIR, else ./external_data."""
    env = os.environ.get("AAAAG_EXTERNAL_DIR")
    return Path(env) if env else REPO_ROOT / "external_data"


def cache_dir() -> Path:
    """Derived per-frame artifacts (parsed tracking, evidence): $AAAAG_CACHE_DIR, else ./cache."""
    env = os.environ.get("AAAAG_CACHE_DIR")
    return Path(env) if env else REPO_ROOT / "cache"


def outputs_dir() -> Path:
    """Submissions, figures and reports. Gitignored: everything in it is derived from match data."""
    return REPO_ROOT / "outputs"


def check_layout() -> None:
    """Fail loudly if the package is not running from a source checkout."""
    if not (VENDOR_ROOT / "games.py").is_file():
        raise RuntimeError(
            f"vendored challenge code not found at {VENDOR_ROOT}. Install the package in editable "
            "mode from the repository root: python -m pip install -e . --no-deps"
        )
