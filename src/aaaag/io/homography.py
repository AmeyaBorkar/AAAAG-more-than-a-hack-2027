"""Per-frame homography: normalised pitch coordinates (0-1) -> image pixels.

Our Tracking is in centred metres, so convert back before projecting:
    x_norm = x_m / pitch_length + 0.5,  y_norm = y_m / pitch_width + 0.5
`valid` says whether a matrix exists; `reprojection_error` how much to trust it (lower is better).
Main use: humans debugging errors on the video (agents must not view match frames; AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MATRIX_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def load_homography(path: str | Path, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(matrices (T,3,3) float64, reprojection_error (T,), valid (T,) bool), indexed by video frame."""
    table = pd.read_csv(path)
    missing = [
        c for c in ["frame", "reprojection_error", "valid", *MATRIX_COLUMNS] if c not in table
    ]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frames = table["frame"].to_numpy()
    if not np.array_equal(frames, np.arange(n_frames)):
        raise ValueError(f"{path}: frames are not 0..{n_frames - 1} in order")
    matrices = table[MATRIX_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    return matrices, table["reprojection_error"].to_numpy(), table["valid"].to_numpy().astype(bool)


def project(
    matrix: np.ndarray, x_norm: np.ndarray, y_norm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised pitch coordinates -> pixels. NaN in, NaN out."""
    x = np.asarray(x_norm, dtype=np.float64)
    y = np.asarray(y_norm, dtype=np.float64)
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return (
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / w,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / w,
    )
