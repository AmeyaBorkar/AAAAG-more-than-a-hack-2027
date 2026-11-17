"""The ball-state lane's main producer: P(dead) per frame, plus who restarts after a stoppage.

Cues (cues.py, kickoff.py) give per-frame log-likelihood ratios. They are summed with weights,
tempered to about one independent observation per `evidence_s` (consecutive frames are far from
independent), and decoded by a two-state alive/dead HMM, forward-backward per period (hmm.py).
The transition rates come from mean stoppage and alive-play durations in seconds. The posterior is
non-causal and calibrated as well as the cues are.

Who restarts: each stoppage end of the posterior is typed and attributed by restarts.py (kick-off
formation, else the ball's spot: corner, goal kick, throw-in, free kick), and for
`restart_window_s` after it the team split leans `restart_team_p` towards that side. Elsewhere it
is 0.5 / 0.5, which leaves it to the possession lane. A "last touch before the ball left the
pitch" rule was tried and dropped: on the IDSSE dev matches it named the restarting side no better
than chance.
"""

from __future__ import annotations

import numpy as np

from ...contracts import HOME, Evidence, MatchBundle, Tracking
from ...registry import EVIDENCE
from . import cues
from .hmm import posterior_dead
from .kickoff import Kickoff, detect_kickoffs, kickoff_llr
from .restarts import Restart, detect_restarts
from .snap import snap_to_ball_gaps

DEFAULT_WEIGHTS = {
    "slowdown": 1.0,
    "off_pitch": 1.0,
    "static_ball": 1.0,
    "ball_in_flight": 1.0,
    "restart_spot": 1.0,
    "kickoff": 1.0,
    "ball_missing": 0.0,  # opt-in (#62): the missing ball as dead-ball evidence
}


def cue_llrs(
    tracking: Tracking, kickoffs: list[Kickoff], ball_missing: dict | None = None
) -> dict[str, np.ndarray]:
    """Every cue's (T,) per-frame LLR (NaN = no opinion), with each cue's physical defaults;
    `ball_missing` passes parameters to cues.ball_missing_llr."""
    return {
        "slowdown": cues.slowdown_llr(tracking),
        "off_pitch": cues.off_pitch_llr(tracking),
        "static_ball": cues.static_ball_llr(tracking),
        "ball_in_flight": cues.ball_in_flight_llr(tracking),
        "restart_spot": cues.restart_spot_llr(tracking),
        "kickoff": kickoff_llr(tracking, kickoffs),
        "ball_missing": cues.ball_missing_llr(tracking, **(ball_missing or {})),
    }


def dead_posterior(
    tracking: Tracking,
    llrs: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    evidence_s: float,
    mean_alive_s: float,
    mean_dead_s: float,
    prior_dead: float,
) -> np.ndarray:
    """(T,) P(dead) inside the periods, NaN outside."""
    meta = tracking.meta
    unknown = set(weights) - set(llrs)
    if unknown:
        raise ValueError(f"unknown ballstate cue weights {sorted(unknown)}; known: {sorted(llrs)}")
    total = np.zeros(meta.n_frames, dtype=np.float64)
    for name, llr in llrs.items():
        weight = float(weights.get(name, 0.0))
        if weight:
            total += weight * np.nan_to_num(llr.astype(np.float64), nan=0.0)
    total /= max(evidence_s * meta.fps, 1.0)
    out = np.full(meta.n_frames, np.nan, dtype=np.float64)
    for period in meta.periods:
        out[period.start : period.stop] = posterior_dead(
            total[period.start : period.stop],
            p_alive_to_dead=1.0 / (mean_alive_s * meta.fps),
            p_dead_to_alive=1.0 / (mean_dead_s * meta.fps),
            prior_dead=prior_dead,
        )
    return out


def restart_teams(
    tracking: Tracking, events: list[Restart], *, window_s: float, team_p: float
) -> np.ndarray:
    """(T,) P(home | alive): `team_p` towards the restarting side for `window_s` after each
    attributed restart, else 0.5."""
    window = int(round(window_s * tracking.meta.fps))
    home = np.full(tracking.n_frames, 0.5, dtype=np.float32)
    for event in events:
        if event.team is not None:
            home[event.frame : event.frame + window] = team_p if event.team == HOME else 1 - team_p
    return home


@EVIDENCE.register("ballstate", version="1", example={})
def ballstate(
    bundle: MatchBundle,
    *,
    weights: dict[str, float] | None = None,
    evidence_s: float = 1.0,
    mean_alive_s: float = 35.0,
    mean_dead_s: float = 25.0,
    prior_dead: float = 0.5,
    restart_window_s: float = 1.0,
    restart_team_p: float = 0.75,
    ball_missing: dict | None = None,
    snap_to_ball_gaps_s: float | None = None,
) -> Evidence:
    """P(dead) from collective slowdown, ball off the pitch, ball at rest or in flight, restart
    spots and kick-off formations, decoded per period by a two-state HMM; plus the restarting team.

    Uses: slot positions (always), the ball where seen (never its absence). Fails in slow build-up
    play (looks like a stoppage), on short stoppages with the ball unseen, and lags a restart by up
    to a second when only the players' acceleration marks it. `weights` overrides DEFAULT_WEIGHTS
    per cue (0 switches a cue off), for ablations. `ball_missing` sets the parameters of the
    opt-in missing-ball cue (#62), which only counts with a nonzero `weights["ball_missing"]`.
    `snap_to_ball_gaps_s` (#66, default None: off) moves each stoppage's edges onto the unseen-ball
    run that overlaps it, by at most that many seconds (snap.py), before the restarts are typed.
    """
    tracking = bundle.tracking
    kickoffs = detect_kickoffs(tracking)
    p_dead = dead_posterior(
        tracking,
        cue_llrs(tracking, kickoffs, ball_missing),
        {**DEFAULT_WEIGHTS, **(weights or {})},
        evidence_s=evidence_s,
        mean_alive_s=mean_alive_s,
        mean_dead_s=mean_dead_s,
        prior_dead=prior_dead,
    )
    if snap_to_ball_gaps_s is not None:
        p_dead = snap_to_ball_gaps(
            p_dead, tracking.ball_seen(), tracking.meta, max_s=float(snap_to_ball_gaps_s)
        )
    events = detect_restarts(tracking, p_dead, kickoffs)
    p_home = restart_teams(tracking, events, window_s=restart_window_s, team_p=restart_team_p)
    return Evidence.from_factors("ballstate", p_dead, p_home)
