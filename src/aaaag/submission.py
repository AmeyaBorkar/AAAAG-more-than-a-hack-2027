"""The submission file: written byte-identically on every OS, checked by the official validator."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from .contracts import LABELS, validate_labels
from .official import VALIDATOR

TEAM = "AAAAG"  # the CSV and the technical document must be named after the team (SUBMISSION.md)


def to_csv_bytes(labels: np.ndarray) -> bytes:
    """'frame,label' header, one row per frame, '\\n' line endings, ASCII. Deterministic bytes."""
    letters = np.array(LABELS)[labels.astype(np.int64)]
    body = "\n".join(f"{frame},{letter}" for frame, letter in enumerate(letters.tolist()))
    return f"frame,label\n{body}\n".encode("ascii")


def write_submission(labels: np.ndarray, path: str | Path) -> str:
    """Write the CSV and return its SHA-256 (the fingerprint we record for reproducibility)."""
    validate_labels(labels, len(labels))
    data = to_csv_bytes(labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def validate_official(path: str | Path) -> tuple[bool, str]:
    """Run the organisers' validate_submission.py on the file, exactly as they will."""
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0, (completed.stdout + completed.stderr).strip()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
