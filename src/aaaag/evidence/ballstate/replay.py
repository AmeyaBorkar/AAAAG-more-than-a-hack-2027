"""Dead-ball prior from broadcast replays and other non-live content (#86). OFF by default.

Reads a precomputed JSON written by `tools/replay_detect` (`run map`): tactical-frame intervals
where the broadcast was showing a replay, slow motion or a close-up, with a confidence per
interval and a `covered` list of tactical frames where the detector had an opinion at all.

    {"version": 1, "n_frames": T, "fps": 25.0,
     "intervals": [{"start", "stop", "kind", "conf", ...}], "covered": [[start, stop], ...]}

Output, per frame:
- inside a non-live interval: P(dead) = 0.5 + (p_dead - 0.5) * conf * kind_weight, the strongest
  interval winning (a prior, not a measurement: set `p_dead` from external validation);
- on a covered frame outside every interval: `p_dead_live` if given, else NaN;
- elsewhere, and outside the played periods: NaN. A broadcast that never showed the moment is no
  evidence, and neither is a detector that was not running there.

The team split is left to the possession producers (P(home | alive) = 0.5).

Reproducibility: the file is derived from the video, which no submission build may need to read.
Commit the JSON (a few kB of intervals, no frames) and pin it with `sha256`; the build then
needs only numpy and this file. With `path=None` (the default) the component is all-NaN, so a
pipeline that does not name it is byte-identical to one that never had it. How a decoder consumes
it: through fusion like any producer's D column (e.g. a log-odds weight on D), or as a transition
prior that lowers the alive->dead switch cost inside covered intervals; see the PR.

Uses: broadcast shot structure only. Fails when a replay is shown during live play (after a
miss, a foul that play continues from) or a long time after the incident, and when the alignment
placed the interval at the wrong time.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ...contracts import Evidence, MatchBundle
from ...registry import EVIDENCE

KIND_WEIGHT = {"replay_logo": 1.0, "replay_slowmo": 0.6, "closeup": 0.5}


@EVIDENCE.register("ballstate_replay", version="1", example={})
def ballstate_replay(
    bundle: MatchBundle,
    *,
    path: str | None = None,
    sha256: str | None = None,
    p_dead: float = 0.9,
    p_dead_live: float | None = None,
    min_conf: float = 0.3,
    kind_weight: dict[str, float] | None = None,
) -> Evidence:
    n = bundle.meta.n_frames
    dead = np.full(n, np.nan, dtype=np.float32)
    if path is not None:
        with open(path, "rb") as f:
            raw = f.read()
        if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError(f"{path}: sha256 does not match the pinned value")
        doc = json.loads(raw)
        if int(doc["n_frames"]) != n:
            raise ValueError(f"{path}: n_frames {doc['n_frames']} != bundle {n}")
        if p_dead_live is not None:
            for a, z in doc.get("covered", []):
                dead[max(0, a) : min(n, z)] = p_dead_live
        weights = KIND_WEIGHT if kind_weight is None else kind_weight
        strength = np.zeros(n, dtype=np.float32)
        for iv in doc["intervals"]:
            s = float(iv["conf"]) * weights.get(iv["kind"], 0.0)
            if float(iv["conf"]) < min_conf or s <= 0:
                continue
            a, z = max(0, int(iv["start"])), min(n, int(iv["stop"]))
            strength[a:z] = np.maximum(strength[a:z], s)
        hit = strength > 0
        dead[hit] = 0.5 + (p_dead - 0.5) * strength[hit]
        dead[~bundle.meta.played_mask()] = np.nan
    return Evidence.from_factors("ballstate_replay", dead, np.full(n, 0.5))
