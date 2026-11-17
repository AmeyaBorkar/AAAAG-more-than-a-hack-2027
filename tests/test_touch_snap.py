"""The post-processing stage and the gaining-team touch snap."""

from __future__ import annotations

import numpy as np

from aaaag import pipeline
from aaaag.contracts import A, D, H, MatchBundle, MatchMeta, Period, Tracking, X
from aaaag.pipeline import PipelineConfig
from aaaag.postprocess.touch_snap import detect_touches, snap_turnovers
from aaaag.registry import POSTPROCESS

FPS = 25
N = 1000
SPEED = 6.0 / FPS  # 6 m/s
TURNS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def meta() -> MatchMeta:
    return MatchMeta(
        match_id="synthetic",
        source="synthetic",
        n_frames=N + 100,
        fps=FPS,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(Period(1, 0, N),),
    )


def tracking_with_touches(touches: dict[int, int]) -> Tracking:
    """The ball moves at 6 m/s and turns by 90 degrees at each frame of `touches`, where a slot of
    the given team (0 = HOME, 1 = AWAY) stands next to it; the other slot is always far away."""
    m = meta()
    ball = np.zeros((m.n_frames, 2), dtype=np.float32)
    position = np.zeros(2)
    turn = 0
    for t in range(m.n_frames):
        if (t - 1) in touches:  # the corner vertex of the path is the ball position at t
            turn += 1
        position = position + np.array(TURNS[turn % 4]) * SPEED
        ball[t] = position
    players = np.full((m.n_frames, 2, 2), 80.0, dtype=np.float32)
    for t, team in touches.items():
        lo, hi = max(0, t - 3), t + 4
        players[lo:hi, team] = ball[lo:hi] + 0.5
    return Tracking(
        meta=m, player_xy=players, slot_team=np.array([0, 1], dtype=np.int8), ball_xy=ball
    )


def test_registered_and_the_example_runs_on_a_synthetic_match():
    component = POSTPROCESS.get("touch_snap")
    assert component.version == "1"
    from aaaag.synthetic import SyntheticConfig, generate

    bundle = generate(seed=2, config=SyntheticConfig(minutes_per_half=2.0, gap_minutes=0.5))
    labels = np.full(bundle.meta.n_frames, X, dtype=np.int8)
    for lo, hi in bundle.meta.runs():
        labels[lo:hi] = H
        labels[lo + (hi - lo) // 2 : hi] = A
    out = component.fn(labels, bundle, **component.example)
    assert out.dtype == np.int8 and out.shape == labels.shape
    assert (out[~bundle.meta.played_mask()] == X).all()


def test_a_turnover_moves_to_the_gaining_teams_touch_and_ignores_the_other_team():
    touches = {400: 1, 420: 0, 470: 1}  # AWAY at 400 and 470, HOME at 420
    tracking = tracking_with_touches(touches)
    frames, teams = detect_touches(tracking)
    assert {400, 420, 470} <= set(map(int, frames))
    assert dict(zip(map(int, frames), map(int, teams), strict=True))[420] == 0
    labels = np.full(tracking.n_frames, X, dtype=np.int8)
    labels[:N] = H
    labels[430:N] = A  # the decoder put the turnover H -> A at frame 430
    window = 50
    gaining = snap_turnovers(labels, tracking.meta, frames, teams, window=window)
    assert int(np.flatnonzero(gaining[:N] == A)[0]) == 400  # the nearest AWAY touch (30 frames)
    assert (gaining[:400] == H).all() and (gaining[400:N] == A).all()
    anyone = snap_turnovers(labels, tracking.meta, frames, teams, window=window, any_team=True)
    assert int(np.flatnonzero(anyone[:N] == A)[0]) == 420  # the control picks the HOME touch (10)
    narrow = snap_turnovers(labels, tracking.meta, frames, teams, window=20)
    assert np.array_equal(narrow, labels)  # no AWAY touch within 20 frames: nothing moves


def test_stoppages_and_unplayed_frames_are_left_alone():
    tracking = tracking_with_touches({400: 1})
    frames, teams = detect_touches(tracking)
    labels = np.full(tracking.n_frames, X, dtype=np.int8)
    labels[:200] = H
    labels[200:300] = D
    labels[300:N] = A
    out = snap_turnovers(labels, tracking.meta, frames, teams, window=50)
    assert np.array_equal(out[:300], labels[:300])
    assert (out[N:] == X).all()
    assert (out[300:N] == A).all()  # the stoppage -> A edge is not a turnover


def test_no_touches_means_no_change():
    tracking = tracking_with_touches({})
    frames, teams = detect_touches(tracking)
    assert len(frames) == 0
    labels = np.full(tracking.n_frames, X, dtype=np.int8)
    labels[:500] = H
    labels[500:N] = A
    assert np.array_equal(snap_turnovers(labels, tracking.meta, frames, teams), labels)


def test_the_pipeline_applies_postprocess_steps_after_the_decoder():
    raw = {
        "name": "snap-test",
        "evidence": [{"name": "nearest_player", "params": {}}],
        "fusion": "first",
        "decoder": {"name": "majority_vote", "params": {"half_window": 10}},
        "postprocess": [{"name": "touch_snap", "params": {"window_s": 2.0}}],
    }
    config = PipelineConfig.from_dict(raw)
    assert [s.name for s in config.postprocess] == ["touch_snap"]
    assert config.to_dict()["postprocess"][0]["name"] == "touch_snap"
    plain = PipelineConfig.from_dict({k: v for k, v in raw.items() if k != "postprocess"})
    assert "postprocess" not in plain.to_dict()  # old configs keep their digests
    assert config.digest() != plain.digest()
    from aaaag.synthetic import SyntheticConfig, generate

    bundle: MatchBundle = generate(
        seed=3, config=SyntheticConfig(minutes_per_half=2.0, gap_minutes=0.5)
    )
    result = pipeline.run(config, bundle, use_cache=False)
    assert "postprocess:touch_snap" in result.timings
    assert result.labels.dtype == np.int8
