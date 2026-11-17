"""Smart Tagging (SportsCode XML) -> TagEvents on the frame axis.

Time base, verified by the organisers: <start>/<end> are seconds from the first frame of the
tactical video, so frame = round(seconds * fps). Intervals are half-open like everything else here.

Remember what this file is (DATA.md): automatic, with false positives and negatives; intervals are
phases, not possessions; set pieces are tagged as a window AROUND the restart, not the instant; and
`Team` is "N/A" on 36 instances including every throw-in. Treat a tag as evidence, never an answer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..contracts import AWAY, HOME, TagEvent

HALF_NAMES = {"1st half": 1, "2nd half": 2}


def load_tags(
    path: str | Path,
    fps: int,
    n_frames: int,
    team_aliases: dict[int, Iterable[str]],
) -> list[TagEvent]:
    """Every instance in file order. `team_aliases` maps HOME/AWAY to the names the XML may use."""
    lookup = {name.strip().lower(): team for team, names in team_aliases.items() for name in names}
    events = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {
            (label.findtext("group") or "").strip(): (label.findtext("text") or "").strip()
            for label in instance.findall("label")
        }
        team_name = tags.get("Team", "").strip().lower()
        events.append(
            TagEvent(
                code=(instance.findtext("code") or "").strip().upper(),
                start=int(np.clip(round(start_s * fps), 0, n_frames)),
                stop=int(np.clip(round(end_s * fps), 0, n_frames)),
                team=lookup.get(team_name),
                half=HALF_NAMES.get(tags.get("Half", "").strip().lower()),
                tags=tags,
            )
        )
    return events


def unmapped_team_names(path: str | Path, team_aliases: dict[int, Iterable[str]]) -> dict[str, int]:
    """Team tag values that map to no side, with counts. Expect only 'n/a'."""
    known = {name.strip().lower() for names in team_aliases.values() for name in names}
    counts: dict[str, int] = {}
    for instance in ET.parse(path).iter("instance"):
        for label in instance.findall("label"):
            if (label.findtext("group") or "").strip() == "Team":
                value = (label.findtext("text") or "").strip().lower()
                if value not in known:
                    counts[value] = counts.get(value, 0) + 1
    return counts


def coverage_mask(events: Iterable[TagEvent], n_frames: int) -> np.ndarray:
    """(T,) int16: how many of the given events cover each frame. Vectorised via a difference array."""
    delta = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        if event.stop > event.start:
            delta[event.start] += 1
            delta[event.stop] -= 1
    return np.cumsum(delta[:-1]).astype(np.int16)


def team_coverage(events: Iterable[TagEvent], n_frames: int, codes: Iterable[str]) -> np.ndarray:
    """(T, 2) int16 coverage counts of the given codes per side (column HOME, column AWAY)."""
    wanted = {code.upper() for code in codes}
    events = [event for event in events if event.code in wanted]
    out = np.zeros((n_frames, 2), dtype=np.int16)
    for team in (HOME, AWAY):
        out[:, team] = coverage_mask((e for e in events if e.team == team), n_frames)
    return out
