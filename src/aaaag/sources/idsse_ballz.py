"""The IDSSE ball's height (Z), which `idsse.to_bundle` does not keep, on a bundle's frame axis.

External data only (DFL / Sportec, CC BY 4.0). Read from the same raw positions file as the
bundle, with the same placement: row = period.start + (N - N_first) for each ball FrameSet, the
sections in playing order. Cached as one small .npz per match under `cache_dir()/idsse_ballz`,
keyed by the raw file's size. Thread-safe (training builds matches on threads).

Where it fails: a match whose ball FrameSets do not match the bundle's periods raises; nothing is
guessed.
"""

from __future__ import annotations

import mmap
import re
import threading

import numpy as np

from ..paths import cache_dir
from . import idsse

_Z = re.compile(rb' Z="([^"]*)"')
_N = re.compile(rb' N="([^"]*)"')
_LOCK = threading.Lock()
_MEMO: dict[str, np.ndarray] = {}


def _parse(match_id: str, n_frames: int, periods) -> np.ndarray:
    path = idsse.file_paths(match_id)["positions"]
    z = np.full(n_frames, np.nan, dtype=np.float32)
    present: list[tuple[str, np.ndarray, np.ndarray]] = []
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        heads = list(idsse._FRAMESET.finditer(raw))
        for i, head in enumerate(heads):
            attrs = {k.decode(): v.decode() for k, v in idsse._ATTR.findall(head.group(1))}
            if attrs.get("TeamId") != idsse.BALL_TEAM:
                continue
            start = head.end()
            stop = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
            n = np.array(_N.findall(raw, start, stop), dtype=np.int64)
            zz = np.array(_Z.findall(raw, start, stop), dtype=np.float32)
            if len(n) != len(zz):
                raise ValueError(f"{match_id}: {len(zz)} Z for {len(n)} ball frames")
            present.append((attrs["GameSection"], n, zz))
    order = {s: k for k, s in enumerate(idsse.SECTIONS)}
    present.sort(key=lambda p: order[p[0]])
    if len(present) != len(periods):
        raise ValueError(f"{match_id}: {len(present)} ball sections, {len(periods)} periods")
    for (_, n, zz), period in zip(present, periods, strict=True):
        rows = period.start + (n - n[0])
        if rows.min() < period.start or rows.max() >= period.stop:
            raise ValueError(f"{match_id}: ball Z rows outside their period")
        z[rows] = zz
    return z


def ball_z(match_id: str, bundle) -> np.ndarray:
    """(T,) float32 ball height in metres on `bundle`'s frame axis, NaN where there is none."""
    with _LOCK:
        if match_id in _MEMO:
            return _MEMO[match_id]
        size = idsse.file_paths(match_id)["positions"].stat().st_size
        folder = cache_dir() / "idsse_ballz"
        cached = folder / f"{match_id}_{size}.npz"
        if cached.is_file():
            with np.load(cached) as data:
                z = data["z"]
        else:
            z = _parse(match_id, bundle.meta.n_frames, bundle.meta.periods)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, z=z)
        if len(z) != bundle.meta.n_frames:
            raise ValueError(
                f"{match_id}: cached Z has {len(z)} frames, bundle {bundle.meta.n_frames}"
            )
        _MEMO[match_id] = z
        return z
