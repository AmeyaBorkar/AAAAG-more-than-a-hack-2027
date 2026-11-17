"""Locating and pinning weight files without importing torch.

The registered `learned_team` producer must load in the repository's normal environment, which has
no torch, so the file checks it does before it needs the framework live here: resolve a path
against the repository, hash the bytes, and compare against the pin a pipeline config carries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from ..paths import REPO_ROOT

MIN_PIN = 8  # hex characters of a SHA-256 that count as a pin


def resolve(path: str | Path) -> Path:
    """`path` as given if absolute, else relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value: str | Sequence[str] | None) -> list[str]:
    """A config value that may be one string, a list of strings or None, as a list."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def check_files(paths: Sequence[Path], pins: Sequence[str]) -> None:
    """Raise unless every file exists and, where a pin is given, its SHA-256 starts with it.

    A pin is what keeps the evidence cache honest: the cache key holds the pipeline parameters, not
    the bytes of the file they point to, so an overwritten checkpoint with the same name would
    otherwise serve stale evidence silently.
    """
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"learned_team: weights not found: {path}")
    if pins and len(pins) != len(paths):
        raise ValueError(f"learned_team: {len(pins)} sha256 pins for {len(paths)} weight files")
    for path, pin in zip(paths, pins, strict=False):
        if len(pin) < MIN_PIN:
            raise ValueError(
                f"learned_team: sha256 pin {pin!r} is shorter than {MIN_PIN} characters"
            )
        actual = sha256_file(path)
        if not actual.startswith(pin.lower()):
            raise ValueError(f"learned_team: {path.name} has sha256 {actual}, pinned {pin}")
