"""Ball-state lane: the HMM scan, the cues' honesty rules, kick-offs, restarts, and the producers on
synthetic matches. Synthetic scores check plumbing and that an obvious signal is picked up; they
never select anything (docs/VALIDATION.md)."""

from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, D, H, MatchBundle, MatchMeta, Period, Tracking
from aaaag.evidence.ballstate import cues, kickoff, model, restarts
from aaaag.evidence.ballstate.hmm import posterior_dead, posterior_dead_reference
from aaaag.registry import EVIDENCE

HMM = {"p_alive_to_dead": 0.01, "p_dead_to_alive": 0.02, "prior_dead": 0.3}


@pytest.mark.parametrize("n", [1, 2, 3, 8, 37, 1000])
def test_scan_forward_backward_equals_the_textbook_loop(n):
    rng = np.random.default_rng(n)
    llr = rng.normal(0.0, 2.0, n)
    llr[::4] = np.nan  # no evidence
    assert np.allclose(posterior_dead(llr, **HMM), posterior_dead_reference(llr, **HMM), atol=1e-9)


def test_scan_survives_long_runs_of_strong_evidence():
    llr = np.concatenate([np.full(5000, 40.0), np.full(5000, -40.0)])
    p = posterior_dead(llr, **HMM)
    assert np.isfinite(p).all()
    assert p[:4990].min() > 0.999 and p[5010:].max() < 0.001


def _without_ball(bundle: MatchBundle) -> MatchBundle:
    tracking = replace(
        bundle.tracking,
        ball_xy=np.full_like(bundle.tracking.ball_xy, np.nan),
        _fingerprint=None,
    )
    return MatchBundle(tracking=tracking, labels=bundle.labels).validate()


def test_a_missing_ball_is_never_evidence(small_bundle):
    blind = _without_ball(small_bundle)
    for llr in (
        cues.off_pitch_llr(blind.tracking),
        cues.static_ball_llr(blind.tracking),
        cues.ball_in_flight_llr(blind.tracking),
        cues.restart_spot_llr(blind.tracking),
    ):
        assert np.isnan(llr).all()
    assert EVIDENCE.get("ballstate_off_pitch").fn(blind).coverage() == 0.0

    # without a ball, the main producer equals the players-only producer on the full bundle
    ball_cues = {"off_pitch": 0, "static_ball": 0, "ball_in_flight": 0, "restart_spot": 0}
    players_only = model.ballstate(small_bundle, weights=ball_cues).proba[:, 2]
    no_ball = model.ballstate(blind).proba[:, 2]
    assert np.allclose(no_ball, players_only, equal_nan=True)


def test_ball_cues_ignore_jumps_and_never_clip_to_the_pitch(small_bundle):
    tracking = small_bundle.tracking
    ball = tracking.ball_xy.copy()
    played = np.flatnonzero(small_bundle.meta.played_mask() & tracking.ball_seen())
    jump, outside = played[len(played) // 3], played[len(played) // 2]
    ball[jump] += 80.0  # a one-frame 80 m jump: a tracking glitch
    ball[outside] = [0.0, tracking.meta.pitch_width / 2 + 3.0]
    raw = replace(tracking, ball_xy=ball, _fingerprint=None)
    speed = cues.ball_speed(raw, window_s=0.2)
    assert np.nanmax(speed) <= cues.MAX_BALL_SPEED
    assert cues.off_pitch_mask(raw)[outside]
    cleaned_ball = ball.copy()
    cleaned_ball[jump] = np.nan  # what the data lane's cleaning will do
    cleaned = replace(tracking, ball_xy=cleaned_ball, _fingerprint=None)
    for fn in (cues.static_ball_llr, cues.ball_in_flight_llr, cues.restart_spot_llr):
        a, b = fn(raw), fn(cleaned)
        far = np.abs(np.arange(len(a)) - jump) > 30
        assert np.array_equal(np.isnan(a[far]), np.isnan(b[far]))


def test_ballstate_finds_the_synthetic_stoppages(small_bundle):
    p = EVIDENCE.get("ballstate").fn(small_bundle).proba[:, 2]
    played = small_bundle.meta.played_mask()
    assert np.isnan(p[~played]).all()
    dead = small_bundle.labels[played] == D
    pred = p[played] >= 0.5
    precision = (pred & dead).sum() / pred.sum()
    recall = (pred & dead).sum() / dead.sum()
    assert precision > 0.6 and recall > 0.8


def test_ballstate_is_deterministic(small_bundle):
    first = EVIDENCE.get("ballstate").fn(small_bundle).proba
    second = EVIDENCE.get("ballstate").fn(small_bundle).proba
    assert np.array_equal(first, second, equal_nan=True)


def test_unknown_cue_weights_are_rejected(small_bundle):
    with pytest.raises(ValueError, match="unknown ballstate cue"):
        model.ballstate(small_bundle, weights={"offpitch": 1.0})


# -- kick-offs ---------------------------------------------------------------------------------------


def _kickoff_match(setup_s: float = 12.0, play_s: float = 30.0, fps: int = 25) -> MatchBundle:
    """One period: a still kick-off formation (home in the -x half, its kicker on the spot), then
    both teams running about the whole pitch. The ball is never seen: kick-offs are ball-free."""
    rng = np.random.default_rng(3)
    setup, n = int(setup_s * fps), int((setup_s + play_s) * fps)
    meta = MatchMeta("KO", "synthetic", n, fps, 105.0, 68.0, (Period(1, 0, n),)).validate()
    home = np.column_stack([-rng.uniform(5, 40, 11), rng.uniform(-30, 30, 11)])
    home[0] = [0.4, 0.2]  # the kicker
    away = np.column_stack([rng.uniform(10, 40, 11), rng.uniform(-30, 30, 11)])
    xy = np.repeat(np.concatenate([home, away])[None], n, axis=0)
    xy[:setup] += rng.normal(0, 0.02, xy[:setup].shape)
    t = np.arange(n - setup)[:, None] / fps
    phase = rng.uniform(0, 2 * np.pi, 22)
    xy[setup:, :, 0] = 30 * np.sin(0.25 * t + phase)  # ~7 m/s peaks, crossing the halfway line
    xy[setup:, :, 1] = 20 * np.cos(0.2 * t + phase)
    xy[setup:, :11, 0] -= 8.0  # keep home's centroid on its own side, as in real play
    xy[setup:, 11:, 0] += 8.0
    tracking = Tracking(
        meta=meta,
        player_xy=xy.astype(np.float32),
        slot_team=np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8),
        ball_xy=np.full((n, 2), np.nan, dtype=np.float32),
    ).validate()
    labels = np.where(np.arange(n) < setup, D, H).astype(np.int8)
    return MatchBundle(tracking=tracking, labels=labels).validate()


def test_kickoff_formation_is_found_without_a_ball():
    bundle = _kickoff_match()
    found = kickoff.detect_kickoffs(bundle.tracking)
    assert len(found) == 1
    ko = found[0]
    assert ko.opening and ko.team == HOME
    assert abs(ko.kick - 12 * 25) <= 25  # within a second of the real kick

    ev = EVIDENCE.get("ballstate_kickoff").fn(bundle).validate(bundle.meta.n_frames)
    assert (ev.proba[: ko.kick, 2] > 0.9).all()  # everything before the opening kick-off is dead
    assert ev.proba[ko.kick, 0] > ev.proba[ko.kick, 1]  # then the kicking side has the ball
    assert np.isnan(ev.proba[ko.kick + 50 :]).all()  # and no opinion afterwards

    p = model.ballstate(bundle).proba[:, 2]
    assert p[: ko.kick - 25].min() > 0.5 and p[ko.kick + 50 :].mean() < 0.5


def test_no_kickoff_in_open_play(small_bundle):
    # synthetic open play never puts both teams wholly in their own halves with a kicker on the spot
    assert kickoff.detect_kickoffs(small_bundle.tracking) == []


# -- restarts --------------------------------------------------------------------------------------


def test_restart_types_and_sides():
    fps, n = 25, 25 * 40
    meta = MatchMeta("RS", "synthetic", n, fps, 105.0, 68.0, (Period(1, 0, n),)).validate()
    xy = np.zeros((n, 4, 2), dtype=np.float32)
    xy[:, 0], xy[:, 1] = [-30.0, 0.0], [-20.0, 5.0]  # home, which defends the -x goal
    xy[:, 2], xy[:, 3] = [30.0, 0.0], [20.0, -5.0]  # away
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    p_dead = np.zeros(n)
    events = {  # restart frame -> (ball spot, the nearest player's new spot)
        5 * fps: ((51.5, 33.0), (52.0, 33.5)),  # corner at the away goal: home attacks it
        15 * fps: ((-48.0, 2.0), (-47.0, 2.0)),  # inside home's goal area: home's goal kick
        25 * fps: ((10.0, 33.8), (10.5, 34.3)),  # touchline: the nearest player throws
    }
    slot = {5 * fps: 0, 15 * fps: 2, 25 * fps: 3}  # who stands nearest the ball
    for frame, (spot, taker) in events.items():
        p_dead[frame - 4 * fps : frame] = 1.0
        ball[frame - 2 * fps : frame + fps] = spot
        xy[frame - 2 * fps : frame + fps, slot[frame]] = taker
    tracking = Tracking(
        meta=meta,
        player_xy=xy,
        slot_team=np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8),
        ball_xy=ball,
    ).validate()
    found = restarts.detect_restarts(tracking, p_dead, [])
    assert [(r.frame, r.kind, r.team) for r in found] == [
        (5 * fps, "corner", HOME),
        (15 * fps, "goal_kick", HOME),
        (25 * fps, "throw_in", AWAY),
    ]


def test_restarts_without_a_ball_are_unknown(small_bundle):
    blind = _without_ball(small_bundle)
    p = model.ballstate(blind).proba[:, 2]
    found = restarts.detect_restarts(blind.tracking, p, [])
    assert found and all(r.kind == "unknown" and r.team is None for r in found)
