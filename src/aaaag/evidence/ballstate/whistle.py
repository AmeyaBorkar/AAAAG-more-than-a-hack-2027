"""Referee whistles as a dead-ball onset prior (numpy only; OFF by default).

A referee whistle marks the start of most stoppages. `tools/whistle_detect/` detects whistles in
the match audio offline and maps them onto the tactical frame axis (an .npz of times, no audio).
This producer reads that file. With no file (`times_path=None`, the default),
it returns all-NaN evidence, so it changes nothing and a pipeline that omits it is byte-identical.

A whistle is a dead-ball BOUNDARY cue, not only an onset: referees also whistle restarts
(kick-offs, penalties, some free kicks). So by default (`mode="boundary"`) the producer gives no
opinion on the state (proba all NaN) and only emits the channel `whistle_boundary` (1.0 on the
whistle's onset frame, 0.0 on other covered frames, NaN where uncovered) for a decoder to weigh in
either direction. `mode="onset"` additionally gives a dead prior as follows.

Output of `mode="onset"`: D with probability `0.5 + (p_dead - 0.5) * confidence` over `hold_s` seconds from each
whistle's onset (shifted by `lag_s`), NaN everywhere else: the absence of a whistle says nothing,
and frames the audio does not cover (broadcast cuts, no audio stream) are NaN by construction.
Channels (both modes): `whistle_boundary` as above, and `whistle_onset` (the same values).

Uses: precomputed whistle frames only. Fails when whistles are not tied to stoppages (fouls the
referee waves on, kick-off and restart whistles, whistling fans), and when the audio/video offset
is wrong: `lag_s` exists for an offset measured on external matches, never on BAR-ATH labels.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Evidence, MatchBundle
from ...registry import EVIDENCE


def whistle_arrays(
    n_frames: int, whistles: np.ndarray, covered: np.ndarray, fps: float, *, lag_s: float,
    hold_s: float, p_dead: float, min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(p_dead (T,), onset channel (T,)) float32 from (N, 3) whistle frames and (K, 2) coverage."""
    dead = np.full(n_frames, np.nan, dtype=np.float32)
    onset = np.full(n_frames, np.nan, dtype=np.float32)
    for lo, hi in covered:
        onset[max(0, int(lo)) : min(n_frames, int(hi))] = 0.0
    shift = int(round(lag_s * fps))
    hold = max(1, int(round(hold_s * fps)))
    for first, _last, conf in whistles[np.argsort(whistles[:, 0], kind="stable")]:
        if conf < min_confidence:
            continue
        a = int(first) + shift
        if not 0 <= a < n_frames or np.isnan(onset[a]):
            continue
        onset[a] = 1.0
        z = min(n_frames, a + hold)
        value = np.float32(0.5 + (p_dead - 0.5) * min(1.0, float(conf)))
        window = dead[a:z]
        dead[a:z] = np.where(np.isnan(window), value, np.maximum(window, value))
    return dead, onset


@EVIDENCE.register("ballstate_whistle", version="1", example={"times_path": None})
def ballstate_whistle(
    bundle: MatchBundle,
    *,
    times_path: str | None = None,
    mode: str = "boundary",
    lag_s: float = 0.0,
    hold_s: float = 4.0,
    p_dead: float = 0.7,
    min_confidence: float = 0.5,
) -> Evidence:
    """See the module docstring. `times_path` is the .npz written by map_to_frames.py."""
    n = bundle.meta.n_frames
    if times_path is None:
        dead = onset = np.full(n, np.nan, dtype=np.float32)
    else:
        with np.load(times_path, allow_pickle=False) as z:
            whistles, covered, fps = z["whistles"], z["covered"], float(z["fps"])
        dead, onset = whistle_arrays(
            n, whistles, covered, fps, lag_s=lag_s, hold_s=hold_s, p_dead=p_dead,
            min_confidence=min_confidence,
        )  # fmt: skip
    if mode not in ("boundary", "onset"):
        raise ValueError(f"mode must be 'boundary' or 'onset', got {mode!r}")
    if mode == "boundary":
        dead = np.full(n, np.nan, dtype=np.float32)
    ev = Evidence.from_factors("ballstate_whistle", dead, np.full(n, 0.5))
    ev.channels["whistle_onset"] = onset.astype(np.float32)
    ev.channels["whistle_boundary"] = onset.astype(np.float32)
    return ev
