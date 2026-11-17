"""Last-controller team possession (#58): the team of the last player who controlled the ball.

`nearest_player` names the team of whoever is nearest the ball on every seen frame, so a pass that
runs past an opponent flips it, and the decoder's +-2 s vote then has to smooth those flips away,
merging short real spells with them. Here possession changes only when the other team *controls*
the ball, and it is held through passes, flights and ball gaps in between.

- **Control:** on a seen frame, the nearest tracked player is within `reach_m` of the ball
  (playing distance, a physical prior).
- **Change of team:** a run of at least `hold_frames` consecutive control frames by one team. A
  shorter run (a deflection, the ball rolling past a player) changes nothing. The new possession is
  dated from the first frame of the qualifying run, not from its `hold_frames`-th frame: the
  decoders are non-causal, and the run's first touch is when the team took the ball.
- **Between controls:** P(home | alive) is the team of the last qualifying run, held inside the
  period through flights and ball gaps. Before a period's first qualifying run there is no opinion.

Signals: ball position, player positions and teams. Output: one-hot (1, 0, 0) or (0, 1, 0) rows,
like `nearest_player`, so it speaks only to the team factor of `factorised` fusion, plus NaN rows
where it has no opinion. The ball-state lane's P(dead) and restarting side are untouched.

Where it fails: the ball seen but no player within reach for a whole possession (long gaps in the
player tracks, a carrier the tracker lost); two opponents both within reach in a duel, where the
nearest one flickers; a long unseen-ball stretch with a change of possession inside it (the old
team is held until the next control; `gap_possession`, #33, is the complement there); tracking
noise that puts the carrier beyond `reach_m` (BAR-ATH's tracking is noisier than IDSSE's).
The defaults are priors fixed before any run (#58), not fitted; tune them only on external data.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Evidence, MatchBundle
from ...registry import EVIDENCE
from ..nearest_player import nearest_team_codes


@EVIDENCE.register("control_possession", version="1", example={})
def control_possession(
    bundle: MatchBundle,
    *,
    reach_m: float = 1.0,
    hold_frames: int = 3,
) -> Evidence:
    """One-hot team of the last controlling team, per period; NaN before its first control."""
    tracking = bundle.tracking
    codes = nearest_team_codes(tracking, max_distance_m=reach_m)  # team in control, else -1
    team = np.full(tracking.n_frames, -1, dtype=np.int8)
    for period in tracking.meta.periods:
        lo, hi = period.start, period.stop
        team[lo:hi] = last_controller(codes[lo:hi], max(1, int(hold_frames)))
    return Evidence.from_codes("control_possession", team)


def last_controller(codes: np.ndarray, hold_frames: int) -> np.ndarray:
    """(n,) int8: the team of the latest run of >= `hold_frames` equal control codes started at or
    before each frame, else -1. `codes` is the controlling team per frame, -1 for no control."""
    n = len(codes)
    out = np.full(n, -1, dtype=np.int8)
    if not n:
        return out
    change = np.flatnonzero(np.diff(codes.astype(np.int16)) != 0) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    qualifying = (codes[starts] >= 0) & (stops - starts >= hold_frames)
    if not qualifying.any():
        return out
    marks = np.full(n, -1, dtype=np.int64)
    marks[starts[qualifying]] = starts[qualifying]
    latest = np.maximum.accumulate(marks)  # the latest qualifying run start at or before t
    has = latest >= 0
    out[has] = codes[latest[has]]
    return out
