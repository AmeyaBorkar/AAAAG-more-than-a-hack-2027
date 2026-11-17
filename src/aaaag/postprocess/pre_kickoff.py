"""Dead before the opening kick-off of each period (E55, #86).

No team has the ball before a period kicks off. The organisers' FAQ names one such stretch on the
target: the second half is graded from its first tracked frame, about two minutes before its
kick-off, and every one of those graded frames is `D`. The network sees players walking into
position near a ball and calls a team there, which gives spurious possessions.

Where the kick-off frame comes from (`source`):
- `anchors`: the organisers' kick-off anchors in `games.py` (`kickoff`, `kickoff_2nd`: "the ball is
  struck on this frame"), looked up by the match id. Rule 2 of AGENTS.md allows only that
  metadata. A match with no anchors (every external match) is returned unchanged.
- `detect`: the opening kick-off found from the formation by `ballstate.kickoff.detect_kickoffs`
  (ball-free; both teams standing in their own halves). It is a check on the anchors, and the
  fallback for a match without them.
- `anchors_or_detect`: anchors where they exist, the detector elsewhere.

Uses: the period bounds and one kick-off frame per period; never the labels around it. Fails when
the anchor or the detected formation is wrong. A kick that falls outside its period is ignored.
"""

from __future__ import annotations

import numpy as np

from ..contracts import D, MatchBundle, Period
from ..registry import POSTPROCESS

SOURCES = ("anchors", "detect", "anchors_or_detect")
ANCHOR_KEYS = {1: "kickoff", 2: "kickoff_2nd"}  # games.py names, by period index


def anchor_kicks(match_id: str) -> dict[int, int]:
    """{period index: video frame of the kick} from the organisers' anchors; {} if none."""
    from ..official import games

    entry = games.GAMES.get(match_id)
    if entry is None:
        return {}
    return {
        index: int(games.anchor_video_frame(entry[key]))
        for index, key in ANCHOR_KEYS.items()
        if key in entry
    }


def detected_kicks(bundle: MatchBundle) -> dict[int, int]:
    """{period index: kick frame} of each period's opening kick-off found from the formation."""
    from ..evidence.ballstate.kickoff import detect_kickoffs

    return {ko.period: int(ko.kick) for ko in detect_kickoffs(bundle.tracking) if ko.opening}


def dead_before(
    labels: np.ndarray, periods: tuple[Period, ...], kicks: dict[int, int]
) -> np.ndarray:
    """`labels` with `D` from each period's start up to (not including) its kick frame."""
    out = np.asarray(labels).astype(np.int8, copy=True)
    for period in periods:
        kick = kicks.get(period.index)
        if kick is not None and period.start < kick < period.stop:
            out[period.start : kick] = D
    return out


def kicks_for(bundle: MatchBundle, source: str) -> dict[int, int]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if source == "anchors":
        return anchor_kicks(bundle.meta.match_id)
    if source == "detect":
        return detected_kicks(bundle)
    return anchor_kicks(bundle.meta.match_id) or detected_kicks(bundle)


@POSTPROCESS.register("pre_kickoff_dead", version="1", example={})
def pre_kickoff_dead(
    labels: np.ndarray, bundle: MatchBundle, *, source: str = "anchors"
) -> np.ndarray:
    """`D` from each period's start to its opening kick-off (see the module docstring)."""
    return dead_before(labels, bundle.meta.periods, kicks_for(bundle, source))
