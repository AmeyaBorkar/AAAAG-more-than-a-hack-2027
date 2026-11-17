"""Crowd-noise cues for the dead ball (numpy only; OFF by default).

`tools/whistle_detect/crowd.py` measures the crowd in the match audio offline and
`map_to_frames.py --kind crowd` maps it onto the tactical frame axis (an .npz of times and a
loudness z score, no audio). This producer reads that file. With no file (`times_path=None`, the
default) every output is NaN, so it changes nothing and a pipeline that omits it is byte-identical.

Channels (all (T,) float32, NaN where the audio does not cover, e.g. broadcast cuts or no audio):
- `crowd_roar`: a dead prior. After a sustained roar starts (a goal; also big chances, which keep
  play alive, hence a weak prior) the ball is more likely dead until the next restart cue or a
  cap: value `0.5 + (p_dead - 0.5) * confidence` from the roar onset (shifted by `lag_s`) until
  the first restart cue at least `min_dead_s` later, or `cap_s`; NaN elsewhere on covered frames
  (no roar says nothing).
- `crowd_spike`: a boundary cue with no state opinion: 1.0 on the onset frame of a sharp rise
  in crowd loudness, 0.0 on other covered frames. Both directions possible (fouls and the ball
  going out start a stoppage, a shot or a through ball starts excitement in live play).
- `crowd_excitement`: the continuous crowd-loudness z score against a rolling baseline, shifted
  by `lag_s`; a feature for a decoder, not a probability.

Restart cue: the onsets of `restart_times_path`, a whistle .npz from the same tooling (kick-off
and restart whistles); without it only the cap ends the prior. `mode="boundary"` (default) gives
no state opinion (proba NaN, channels only); `mode="onset"` also puts `crowd_roar` into the D
column.

Uses: precomputed crowd frames only. Fails when: commentary, music or PA is loud; the crowd is
biased towards the home side; reactions lag the event by 1-3 s (`lag_s` exists, measured on
external audio only, never on the competition labels); roars that do not end play; a wrong audio/video
offset. No decoder reads these channels yet.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Evidence, MatchBundle
from ...registry import EVIDENCE


def _covered_mask(n: int, covered: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for lo, hi in covered:
        mask[max(0, int(lo)) : min(n, int(hi))] = True
    return mask


def crowd_arrays(
    n_frames: int, roars: np.ndarray, spikes: np.ndarray, excitement: np.ndarray,
    covered: np.ndarray, restarts: np.ndarray, fps: float, *, lag_s: float, p_dead: float,
    cap_s: float, min_dead_s: float, min_roar_confidence: float, min_spike_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """(crowd_roar, crowd_spike, crowd_excitement), each (T,) float32."""
    cover = _covered_mask(n_frames, covered)
    shift = int(round(lag_s * fps))
    cap, min_dead = max(1, int(round(cap_s * fps))), int(round(min_dead_s * fps))
    roar = np.full(n_frames, np.nan, dtype=np.float32)
    spike = np.where(cover, 0.0, np.nan).astype(np.float32)
    restarts = np.sort(np.asarray(restarts, dtype=np.float64))
    for first, _last, conf in roars[np.argsort(roars[:, 0], kind="stable")]:
        a = int(first) + shift
        if conf < min_roar_confidence or not 0 <= a < n_frames or not cover[a]:
            continue
        later = restarts[restarts >= a + min_dead]
        z = min(n_frames, a + cap, int(later[0]) if len(later) else n_frames)
        value = np.float32(0.5 + (p_dead - 0.5) * min(1.0, float(conf)))
        window = roar[a:z]
        roar[a:z] = np.where(np.isnan(window), value, np.maximum(window, value))
    roar[~cover] = np.nan
    for frame, conf in spikes:
        a = int(frame) + shift
        if conf >= min_spike_confidence and 0 <= a < n_frames and cover[a]:
            spike[a] = 1.0
    exc = np.full(n_frames, np.nan, dtype=np.float32)
    src = np.asarray(excitement, dtype=np.float32)[:n_frames]
    if shift >= 0:
        exc[shift : shift + len(src)] = src[: max(0, n_frames - shift)]
    else:
        exc[: max(0, len(src) + shift)] = src[-shift:]
    exc[~cover] = np.nan
    return roar, spike, exc


@EVIDENCE.register("ballstate_crowd", version="1", example={"times_path": None})
def ballstate_crowd(
    bundle: MatchBundle,
    *,
    times_path: str | None = None,
    restart_times_path: str | None = None,
    mode: str = "boundary",
    lag_s: float = 0.0,
    p_dead: float = 0.65,
    cap_s: float = 45.0,
    min_dead_s: float = 5.0,
    min_roar_confidence: float = 0.5,
    min_spike_confidence: float = 0.4,
) -> Evidence:
    """See the module docstring. `times_path` is the .npz written by
    `map_to_frames.py --kind crowd`; `restart_times_path` an optional whistle .npz."""
    if mode not in ("boundary", "onset"):
        raise ValueError(f"mode must be 'boundary' or 'onset', got {mode!r}")
    n = bundle.meta.n_frames
    nan = np.full(n, np.nan, dtype=np.float32)
    roar, spike, exc = nan, nan, nan
    if times_path is not None:
        with np.load(times_path, allow_pickle=False) as z:
            roars, spikes, excitement = z["roars"], z["spikes"], z["excitement"]
            covered, fps = z["covered"], float(z["fps"])
        restarts = np.zeros(0)
        if restart_times_path is not None:
            with np.load(restart_times_path, allow_pickle=False) as w:
                restarts = w["whistles"][:, 0]
        roar, spike, exc = crowd_arrays(
            n, roars, spikes, excitement, covered, restarts, fps, lag_s=lag_s, p_dead=p_dead,
            cap_s=cap_s, min_dead_s=min_dead_s, min_roar_confidence=min_roar_confidence,
            min_spike_confidence=min_spike_confidence,
        )  # fmt: skip
    dead = roar.copy() if mode == "onset" else nan
    ev = Evidence.from_factors("ballstate_crowd", dead, np.full(n, 0.5))
    ev.channels["crowd_roar"] = roar
    ev.channels["crowd_spike"] = spike
    ev.channels["crowd_excitement"] = exc
    return ev
