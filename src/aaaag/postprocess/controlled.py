"""Controlled possession: keep a short spell when the team shows control, absorb it otherwise.

The organisers change possession only when the other team gains *controlled* possession. A touch
without control, a deflection or an aerial duel without subsequent control is not a change, and a
loose or contested ball stays with the last team. Provider labels flip on any touch. `absorb_short`
drops every spell under `hold_s`, including the short spells that were controlled. This rule keeps
a spell that lasts `hold_s` or that shows a completed pass, and absorbs the rest with the same
merge as `absorb_short`. `pass_model` adds a logistic estimate of that pass, for a ball track
whose smoothing would hide the receiver.
"""

from __future__ import annotations

import numpy as np

from ..contracts import A, D, H, MatchBundle, MatchMeta, Tracking, X
from ..registry import POSTPROCESS
from .min_spell import _merge_run, _stretches
from .pass_check import load_pass_model, pass_probability, span_features


def _forward_fill_x(labels: np.ndarray, meta: MatchMeta) -> np.ndarray:
    """X inside a period takes the previous non-X label. A leading X stays X.

    The scorer forward-fills an abstention before it cuts spells, so a ball in flight does not
    split a possession. Frames outside the periods are left as they are.
    """
    out = np.asarray(labels, dtype=np.int8).copy()
    for period in meta.periods:
        seg = out[period.start : period.stop]
        known_at = np.where(seg != X, np.arange(len(seg)), -1)
        np.maximum.accumulate(known_at, out=known_at)
        take = known_at >= 0
        filled = seg.copy()
        filled[take] = seg[known_at[take]]
        out[period.start : period.stop] = filled
    return out


def _in_reach(tracking: Tracking, reach_m: float) -> dict[int, np.ndarray]:
    """Per team, ``(T, n_slots)`` bool: the observed ball is within ``reach_m`` of that slot.

    One broadcast over frames and slots, the nearest-teammate distances (the minimum along the
    slot axis is the nearest teammate). No Python loop over frames. A NaN ball or a NaN slot is
    never within reach, so it contributes no pass evidence.
    """
    ball = np.asarray(tracking.ball_xy, dtype=np.float64)
    players = np.asarray(tracking.player_xy, dtype=np.float64)
    dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    near = np.isfinite(dist) & (dist <= reach_m)
    teams = np.asarray(tracking.slot_team)
    return {team: near[:, teams == team] for team in (H, A)}


def _team_confidence(proba: np.ndarray) -> dict[int, np.ndarray]:
    """Per team, (T,) P(team | alive) from a (T, 3) posterior over (H, A, D); NaN with no opinion."""
    p = np.asarray(proba, dtype=np.float64)
    alive = p[:, 0] + p[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        home = np.where(alive > 1e-6, p[:, 0] / alive, np.nan)
    return {H: home, A: 1.0 - home}


def _model_keeps(run: list[int], model: dict | None, tracking: Tracking | None) -> bool:
    """True when the logistic check calls this short spell a completed pass.

    ``model is None`` is false, which leaves the geometric rule as the only pass evidence.
    """
    if model is None:
        return False
    if tracking is None:
        raise ValueError("pass_model requires tracking")
    features = span_features(tracking, int(run[0]), int(run[1]), int(run[2]))
    return pass_probability(features, model) >= float(model["threshold"])


def _counts(
    run: list[int],
    hold_frames: float,
    near: dict[int, np.ndarray],
    min_players: int,
    ball: np.ndarray | None = None,
    min_sep_m: float = 0.0,
    conf: dict[int, np.ndarray] | None = None,
    keep_conf: float = 0.0,
    model: dict | None = None,
    tracking: Tracking | None = None,
) -> bool:
    """True when the spell is long enough, or a pass check says it was controlled.

    With `min_sep_m` above 0 the geometric pass must also cover distance: the ball positions at
    the first contact of each of those teammates must span at least `min_sep_m` (a crowd round
    one ball is not a pass). Zero keeps the plain distinct-teammate count. `model`, when set,
    also keeps the spell when its logistic score reaches the model's threshold.
    """
    if not run[1] - run[0] < hold_frames:
        return True
    if keep_conf > 0.0 and conf is not None:
        team_p = conf[int(run[2])][run[0] : run[1]]
        if team_p.size and np.isfinite(team_p).any() and float(np.nanmean(team_p)) >= keep_conf:
            return True
    window = near[int(run[2])][run[0] : run[1]]
    if window.size != 0:
        touched = window.any(axis=0)
        if int(np.count_nonzero(touched)) >= min_players:
            if min_sep_m <= 0.0 or ball is None:
                return True
            first = np.argmax(window[:, touched], axis=0) + run[0]
            spots = ball[first]
            span = np.hypot(*(spots[:, None, :] - spots[None, :, :]).transpose(2, 0, 1))
            if bool(span.max() >= min_sep_m):
                return True
    return _model_keeps(run, model, tracking)


def controlled_possession(
    labels: np.ndarray,
    tracking: Tracking,
    *,
    hold_s: float = 1.5,
    reach_m: float = 1.0,
    min_players: int = 2,
    min_sep_m: float = 0.0,
    proba: np.ndarray | None = None,
    keep_conf: float = 0.0,
    pass_model: str | None = None,
    pass_model_sha256: str | None = None,
    keep_restart: bool = False,
) -> np.ndarray:
    """``labels`` with uncontrolled spells absorbed into the previous possession.

    ``keep_restart`` (E61, off by default) also keeps the first spell of every alive stretch that
    starts right after a stoppage: the organisers' rules give a restart "a clear possessor", so a
    short spell that opens play (a throw-in lost within a second) is not handed to the next team.
    The first alive stretch of each period (its opening kick-off) is left to the rule as before.

    Inside each period, X takes the previous label first (a leading X stays X) and D breaks an
    alive stretch. A spell is kept when it lasts at least ``hold_s`` seconds, or when at least
    ``min_players`` distinct slots of its team each have the ball within ``reach_m`` on some frame
    of the spell (a completed pass: two teammates inside the radius count, including on one frame).
    A spell that does not count is absorbed like ``absorb_short``: the shortest uncounted spell
    goes first (the earliest on a tie); ``_merge_run`` joins it to the previous spell, which then
    also takes the next spell when that spell is the same team; a first spell of the stretch joins
    the next one. The stretch is re-evaluated after every merge, so a merged spell that is now
    long enough, or that now contains a pass, is kept. A stretch of one spell is left as it is.
    Dead frames are never rewritten. The result is int8 and the same shape as ``labels``.

    ``pass_model``, when set, is a path to a JSON logistic file, relative to the repository root
    unless it is absolute. A short spell is also kept when ``pass_probability`` of its
    ``span_features`` is at least the file's ``threshold``. The geometric check and ``keep_conf``
    still keep a spell on their own. ``pass_model_sha256`` pins the file. With ``pass_model``
    left as None the labels are the geometric rule alone (and ``keep_conf``, when set).

    Signals: the labels, ``tracking.player_xy``, ``tracking.slot_team``, ``tracking.ball_xy``, and
    the periods and frame rate on ``tracking.meta``. The logistic check reads that same tracking.

    Where it fails: a pass to a teammate with the ball unseen gives no evidence, and a dribble
    shorter than ``hold_s`` with one player is absorbed. With a pass model, a short spell is
    still absorbed when the geometric check, ``keep_conf`` and the logistic score all decline.
    """
    if labels.shape != (tracking.n_frames,):
        raise ValueError(
            f"labels length {labels.shape} does not match tracking length {tracking.n_frames}"
        )
    if pass_model is None:
        if pass_model_sha256 is not None:
            raise ValueError("pass_model_sha256 is set but pass_model is not")
        model = None
    else:
        model = load_pass_model(pass_model, pass_model_sha256)
    hold_frames = float(hold_s) * tracking.meta.fps
    near = _in_reach(tracking, float(reach_m))
    ball = np.asarray(tracking.ball_xy, dtype=np.float64)
    conf = _team_confidence(proba) if (keep_conf > 0.0 and proba is not None) else None
    min_players = int(min_players)
    out = _forward_fill_x(labels, tracking.meta)
    # The first alive stretch of each period opens on its kick-off, not on a stoppage seen in play:
    # it is never a keep_restart stretch (on BAR-ATH the second half's opening is pre_kickoff_dead's
    # D, and the walk-in tail after the kick is not a restart possession).
    openers = set()
    for period in tracking.meta.periods:
        alive = np.flatnonzero(np.isin(out[period.start : period.stop], (H, A)))
        if len(alive):
            openers.add(period.start + int(alive[0]))
    for runs in _stretches(out, tracking.meta):
        restart = (
            bool(keep_restart)
            and runs[0][0] not in openers
            and runs[0][0] > 0
            and int(out[runs[0][0] - 1]) == D
        )
        while len(runs) > 1:
            weak = [
                (run[1] - run[0], index)
                for index, run in enumerate(runs)
                if not (restart and index == 0)
                and not _counts(
                    run,
                    hold_frames,
                    near,
                    min_players,
                    ball,
                    min_sep_m,
                    conf,
                    keep_conf,
                    model,
                    tracking,
                )
            ]
            if not weak:
                break
            _merge_run(runs, min(weak)[1])
        for start, stop, code in runs:
            out[start:stop] = code
    return out


@POSTPROCESS.register(
    "controlled_spell",
    version="1",
    example={"hold_s": 1.5, "reach_m": 1.0, "min_players": 2, "min_sep_m": 0.0},
)
def controlled_spell(
    labels: np.ndarray,
    bundle: MatchBundle,
    *,
    hold_s: float = 1.5,
    reach_m: float = 1.0,
    min_players: int = 2,
    min_sep_m: float = 0.0,
    ball_deblur: str | None = None,
    ball_deblur_sha256: str | None = None,
    keep_conf: float = 0.0,
    proba: np.ndarray | None = None,
    pass_model: str | None = None,
    pass_model_sha256: str | None = None,
    keep_restart: bool = False,
) -> np.ndarray:
    """Keep a possession that lasts ``hold_s`` or shows a completed pass; absorb the rest.

    ``keep_restart`` (E61) keeps the spell that opens play after each stoppage.

    With ``keep_conf`` above 0 (E47) a short spell is also kept when the fused posterior's mean
    P(its team | alive) over the spell is at least ``keep_conf``; the pipeline passes ``proba``.

    ``pass_model``, when set, is a JSON logistic file. A short spell is also kept when
    ``pass_probability`` of ``span_features`` on the tracking this call sees is at least the
    file's threshold, including after ``ball_deblur`` has replaced the ball. ``pass_model_sha256``
    pins that file. With ``pass_model`` left as None the labels are the call without it.

    Pipeline YAML uses this the same way as ``min_spell``. See ``controlled_possession``.
    `ball_deblur`, when set, is a JSON weight file: the pass check reads `deblur_ball` of the
    bundle's ball and does not modify the bundle. `ball_deblur_sha256` pins that file.
    """
    tracking = bundle.tracking
    if ball_deblur is not None:
        from ..ml.np_deblur import deblurred_tracking

        tracking = deblurred_tracking(tracking, ball_deblur, ball_deblur_sha256)
    return controlled_possession(
        labels,
        tracking,
        hold_s=float(hold_s),
        reach_m=float(reach_m),
        min_players=int(min_players),
        min_sep_m=float(min_sep_m),
        proba=proba,
        keep_conf=float(keep_conf),
        pass_model=pass_model,
        pass_model_sha256=pass_model_sha256,
        keep_restart=bool(keep_restart),
    )
