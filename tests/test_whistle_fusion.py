"""E156 whistle fusion in the dead gate (F1 dead bonus, F2 switch-cost multiplier), on toy posteriors."""

from __future__ import annotations

import numpy as np

from aaaag.contracts import D, Evidence, MatchMeta, Period
from aaaag.decoders import whistle_fusion as wf
from aaaag.decoders.dead_gated_hsmm import dead_gated_hsmm, gate_posterior
from aaaag.decoders.dead_gated_vote import _clean
from aaaag.fusion.factorised import factorised

FPS = 25


def make_meta(periods=((0, 3000), (3100, 6000))) -> MatchMeta:
    n = periods[-1][1] + 100
    return MatchMeta(
        match_id="synthetic",
        source="synthetic",
        n_frames=n,
        fps=FPS,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=tuple(Period(i + 1, a, b) for i, (a, b) in enumerate(periods)),
    )


def toy(meta: MatchMeta, seed: int = 0) -> np.ndarray:
    """Alive with P(D) 0.05, two stoppages per period (P(D) 0.9), team flips every 6 s."""
    rng = np.random.default_rng(seed)
    p_dead = np.full(meta.n_frames, 0.05)
    for start, _stop in meta.runs():
        p_dead[start + 500 : start + 700] = 0.9
        p_dead[start + 1500 : start + 1650] = 0.9
    p_home = np.where((np.arange(meta.n_frames) // 150) % 2 == 0, 0.85, 0.15)
    p_home = np.clip(p_home + rng.normal(0, 0.05, meta.n_frames), 0.01, 0.99)
    proba = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
    return proba.astype(np.float32)


def channel(n: int, onsets) -> np.ndarray:
    w = np.zeros(n, dtype=np.float32)
    w[list(onsets)] = 1.0
    return w


def test_onset_frames_reads_ones_only():
    w = np.array([np.nan, 0.0, 1.0, 0.0, np.nan, 1.0], dtype=np.float32)
    assert wf.onset_frames(w).tolist() == [2, 5]
    assert wf.onset_frames(None).size == 0


def test_window_mask_is_inclusive_and_clipped():
    m = wf.window_mask(20, np.array([1, 15]), 2, 3)
    assert np.flatnonzero(m).tolist() == [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18]


def test_dead_bonus_shifts_log_odds_in_the_window_only():
    proba = np.tile(np.array([[0.6, 0.2, 0.2]], dtype=np.float32), (200, 1))
    proba[50] = np.nan
    out = wf.dead_bonus(proba, np.array([100]), FPS, bonus=1.0, pre_s=0.4, post_s=2.0)
    inside = np.arange(90, 151)
    lo = np.log(out[inside, 2] / (1 - out[inside, 2])) - np.log(0.2 / 0.8)
    assert np.allclose(lo, 1.0, atol=1e-5)
    assert np.allclose(out[inside].sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(out[inside, 0] / out[inside, 1], 3.0, atol=1e-4)  # team odds kept
    outside = np.setdiff1d(np.arange(200), inside)
    assert np.array_equal(out[outside], proba[outside], equal_nan=True)


def test_clean_near_equals_the_gate_cleaning_when_off():
    rng = np.random.default_rng(3)
    dead = rng.random(3000) < 0.5
    dead = np.repeat(dead[::30], 30)[:3000] ^ (rng.random(3000) < 0.05)
    ref = _clean(dead, 50, 25)
    assert np.array_equal(
        wf.clean_near(dead, 50, 25, onsets=np.array([], int), half=12, mult=0.5), ref
    )
    assert np.array_equal(
        wf.clean_near(dead, 50, 25, onsets=np.array([400, 1700]), half=12, mult=1.0), ref
    )


def test_clean_near_keeps_a_short_stoppage_and_a_short_alive_gap_near_a_whistle():
    dead = np.zeros(600, dtype=bool)
    dead[100:130] = True  # 1.2 s stoppage: dropped by the 2 s rule
    dead[300:400] = True
    dead[410:500] = True  # 0.4 s alive gap between two stoppages: filled by the 1 s rule
    plain = _clean(dead, 50, 25)
    assert not plain[100:130].any() and plain[400:410].all()
    half = wf.clean_near(dead, 50, 25, onsets=np.array([95, 412]), half=12, mult=0.5)
    assert half[100:130].all()  # the stoppage needs 25 frames and has 30
    assert half[400:410].all()  # the gap needs 12 frames to stay alive and has 10
    quarter = wf.clean_near(dead, 50, 25, onsets=np.array([95, 412]), half=12, mult=0.25)
    assert quarter[100:130].all() and not quarter[400:410].any()  # 6 frames suffice now
    far = wf.clean_near(dead, 50, 25, onsets=np.array([200]), half=12, mult=0.25)
    assert np.array_equal(far, plain)


def test_decoder_is_byte_identical_when_off():
    meta = make_meta()
    proba = toy(meta)
    ref = dead_gated_hsmm(proba, meta, dead_threshold=0.3, switch_weight=0.0)
    w = channel(meta.n_frames, [800, 2400, 4000])
    for kwargs in (
        {},
        {"whistle": w},
        {"whistle": w, "whistle_bonus": 0.0, "whistle_switch_mult": 1.0},
        {"whistle": np.zeros(meta.n_frames, np.float32), "whistle_bonus": 2.0},
        {"whistle": None, "whistle_bonus": 2.0, "whistle_switch_mult": 0.25},
    ):
        out = dead_gated_hsmm(proba, meta, dead_threshold=0.3, switch_weight=0.0, **kwargs)
        assert out.tobytes() == ref.tobytes()
    gated = gate_posterior(proba, meta, dead_threshold=0.3)
    assert gate_posterior(proba, meta, dead_threshold=0.3, whistle=w).tobytes() == gated.tobytes()


def test_f1_bonus_opens_a_stoppage_at_a_whistle_where_p_dead_is_middling():
    meta = make_meta()
    proba = toy(meta)
    proba[1000:1100, 2] = 0.2  # a 4 s stretch the network calls 20% dead
    proba[1000:1100, :2] *= 0.8 / proba[1000:1100, :2].sum(axis=1, keepdims=True)
    plain = dead_gated_hsmm(proba, meta, dead_threshold=0.3)
    assert not (plain[1000:1100] == D).any()
    w = channel(meta.n_frames, [1010])
    gated = gate_posterior(proba, meta, dead_threshold=0.3, whistle=w, whistle_bonus=1.0)
    dead = np.flatnonzero(gated[900:1200, 2] == 1.0) + 900
    assert dead.min() == 1000 and dead.max() == 1060  # [t - 0.4 s, t + 2 s] within the stretch
    out = dead_gated_hsmm(proba, meta, dead_threshold=0.3, whistle=w, whistle_bonus=1.0)
    dead = np.flatnonzero(out[900:1200] == D) + 900
    assert abs(dead.min() - 1000) < 5 and abs(dead.max() - 1060) < 5  # the HSMM's 5-frame blocks
    assert np.array_equal(out[:900], plain[:900]) and np.array_equal(out[1300:], plain[1300:])


def test_f2_keeps_a_short_stoppage_near_a_whistle():
    meta = make_meta()
    proba = toy(meta)
    proba[1000:1035, 2] = 0.9  # 1.4 s at 0.9: shorter than the 2 s the gate asks
    plain = dead_gated_hsmm(proba, meta, dead_threshold=0.3)
    assert not (plain[1000:1035] == D).any()
    w = channel(meta.n_frames, [1005])
    half = dead_gated_hsmm(proba, meta, dead_threshold=0.3, whistle=w, whistle_switch_mult=0.5)
    assert (half[1000:1035] == D).all()
    far = channel(meta.n_frames, [1100])
    out = dead_gated_hsmm(proba, meta, dead_threshold=0.3, whistle=far, whistle_switch_mult=0.5)
    assert out.tobytes() == plain.tobytes()


def test_factorised_carries_the_whistle_channel_only_when_present():
    meta = make_meta()
    proba = toy(meta)
    learned = Evidence(
        "learned_team", proba.copy(), {"switch": np.zeros(meta.n_frames, np.float32)}
    )
    plain = factorised([learned], meta, dead_from=[0], team_from=[0])
    assert set(plain.channels) == {"switch"}
    n = meta.n_frames
    onset = np.full(n, np.nan, dtype=np.float32)
    onset[:3000] = 0.0
    onset[100] = 1.0
    whistle = Evidence(
        "ballstate_whistle", np.full((n, 3), np.nan, np.float32), {"whistle_onset": onset}
    )
    fused = factorised([learned, whistle], meta, dead_from=[0], team_from=[0])
    assert set(fused.channels) == {"switch", "whistle"}
    assert np.array_equal(np.asarray(fused), np.asarray(plain))
    assert fused.channels["whistle"][100] == 1.0 and fused.channels["whistle"][101] == 0.0
    assert np.isnan(fused.channels["whistle"][4000])
    no_switch = Evidence("learned_team", proba.copy())
    alone = factorised([no_switch, whistle], meta, dead_from=[0], team_from=[0])
    assert set(alone.channels) == {"whistle"}
