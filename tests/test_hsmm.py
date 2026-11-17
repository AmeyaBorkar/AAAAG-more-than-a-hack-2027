"""Explicit-duration HSMM decoder. Synthetic data only; no match data."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchMeta, Period, X, validate_labels
from aaaag.decoders.hsmm import (
    DEFAULT_MODEL,
    NEG,
    _block_sums,
    _expand,
    _hsmm_period,
    _log_pmf,
    _log_surv,
    _transition_logs,
    fit_priors,
    frame_emissions,
    hsmm,
    load_priors,
    prune_short_spells,
    refine_boundaries,
    save_priors,
)
from aaaag.official import pm
from aaaag.paths import REPO_ROOT

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


def truth_from(segments, meta) -> np.ndarray:
    """segments: per period a list of (code, n_frames)."""
    out = np.full(meta.n_frames, X, dtype=np.int8)
    for (lo, hi), segs in zip(meta.runs(), segments, strict=True):
        pos = lo
        for code, n in segs:
            out[pos : pos + n] = code
            pos += n
        assert pos == hi
    return out


def proba_from(truth: np.ndarray, p_dead_alive=0.03, p_dead_dead=0.97) -> np.ndarray:
    p_dead = np.where(truth == D, p_dead_dead, p_dead_alive)
    home = np.where(truth == H, 0.999, np.where(truth == A, 0.001, 0.5))
    proba = np.stack([(1 - p_dead) * home, (1 - p_dead) * (1 - home), p_dead], axis=1)
    proba[truth == X] = np.nan
    return proba.astype(np.float32)


SEGS = [
    [(D, 250), (H, 300), (A, 200), (H, 400), (D, 350), (A, 500), (H, 250), (A, 150), (D, 600)],
    [(D, 300), (A, 500), (H, 250), (D, 400), (H, 600), (A, 350), (H, 300), (D, 200)],
]


def test_fit_model_recovers_durations():
    rng = np.random.default_rng(0)
    labels = []
    for _ in range(4):
        segs, code = [], D
        for _ in range(400):
            code = int(rng.choice([c for c in (H, A, D) if c != code]))
            mu, sg = (1.3, 1.2) if code != D else (2.9, 0.8)
            segs.append((code, max(1, int(np.exp(rng.normal(mu, sg)) * FPS))))
        arr = np.concatenate([np.full(n, c, dtype=np.int8) for c, n in segs])
        labels.append(np.concatenate([arr, np.full(50, X, dtype=np.int8)]))
    model = fit_priors(labels)
    assert abs(model["alive"]["mu"] - 1.3) < 0.12 and abs(model["alive"]["sigma"] - 1.2) < 0.12
    assert abs(model["dead"]["mu"] - 2.9) < 0.15 and abs(model["dead"]["sigma"] - 0.8) < 0.12
    trans = np.asarray(model["trans"])
    assert np.allclose(trans.sum(axis=1), 1.0) and np.allclose(np.diag(trans), 0.0)


def test_hsmm_recovers_clean_segmentation():
    meta = make_meta()
    truth = truth_from(SEGS, meta)
    labels = hsmm(proba_from(truth), meta, lam_team=0.3, lam_dead=0.5)
    validate_labels(labels, meta.n_frames)
    assert (labels[~meta.played_mask()] == X).all()
    graded = meta.played_mask()
    assert (labels[graded] == truth[graded]).mean() > 0.995
    runs = pm.valid_runs(truth)
    assert len(pm.possession_spells(labels, runs)) == len(pm.possession_spells(truth, runs))


def test_hsmm_is_deterministic_and_registry_shaped():
    meta = make_meta()
    proba = proba_from(truth_from(SEGS, meta))
    a = hsmm(proba, meta)
    b = hsmm(proba, meta)
    assert a.dtype == np.int8 and a.shape == (meta.n_frames,) and np.array_equal(a, b)
    assert a.tobytes() == b.tobytes()


def test_labels_outside_periods_are_x():
    meta = make_meta()
    proba = proba_from(truth_from(SEGS, meta))
    labels = hsmm(proba, meta)
    validate_labels(labels, meta.n_frames)
    assert (labels[~meta.played_mask()] == X).all()
    assert np.isin(labels[meta.played_mask()], (H, A, D)).all()


def test_hsmm_survives_nan_rows_and_flat_evidence():
    meta = make_meta()
    proba = np.full((meta.n_frames, 3), np.nan, dtype=np.float32)
    proba[100:900] = 1 / 3.0
    labels = hsmm(proba, meta)
    validate_labels(labels, meta.n_frames)
    blank = np.full((meta.n_frames, 3), np.nan, dtype=np.float32)
    validate_labels(hsmm(blank, meta), meta.n_frames)


def test_period_boundary_does_not_carry_state():
    rng = np.random.default_rng(2)
    n = 800
    both_meta = MatchMeta(
        "synthetic", "synthetic", n, FPS, 105.0, 68.0, (Period(1, 0, 300), Period(2, 400, 700))
    ).validate()
    only1 = MatchMeta(
        "synthetic", "synthetic", n, FPS, 105.0, 68.0, (Period(1, 0, 300),)
    ).validate()
    only2 = MatchMeta(
        "synthetic", "synthetic", n, FPS, 105.0, 68.0, (Period(1, 400, 700),)
    ).validate()
    proba = rng.random((n, 3)).astype(np.float32)
    proba /= proba.sum(axis=1, keepdims=True)
    proba[::9] = np.nan
    both = hsmm(proba, both_meta, min_spell_s=0.4)
    assert np.array_equal(both[:300], hsmm(proba, only1, min_spell_s=0.4)[:300])
    assert np.array_equal(both[400:700], hsmm(proba, only2, min_spell_s=0.4)[400:700])
    assert (both[300:400] == X).all()


def test_hsmm_duration_prior_absorbs_a_blip():
    """A 6-frame (0.24 s) team flip inside a long possession is not worth a segment, a 3 s one is."""
    meta = make_meta(((0, 3000), (3100, 6000)))
    truth = truth_from([[(H, 3000)], [(A, 2900)]], meta)
    proba = proba_from(truth)
    blip = proba.copy()
    blip[1000:1006] = proba_from(truth_from([[(A, 3000)], [(A, 2900)]], meta))[1000:1006]
    out = hsmm(blip, meta, lam_team=0.1, lam_dead=0.3)
    assert (out[:3000] == H).all()
    real = proba.copy()
    real[1000:1075] = proba_from(truth_from([[(A, 3000)], [(A, 2900)]], meta))[1000:1075]
    out = hsmm(real, meta, lam_team=0.1, lam_dead=0.3)
    assert (out[1010:1065] == A).all()


def test_min_spell_s_removes_short_alive_spells():
    meta = make_meta(((0, 3000), (3100, 6000)))
    truth = truth_from([[(H, 3000)], [(A, 2900)]], meta)
    away = proba_from(truth_from([[(A, 3000)], [(A, 2900)]], meta))
    real = proba_from(truth)
    real[1000:1075] = away[1000:1075]
    kept = hsmm(real, meta, lam_team=0.1, lam_dead=0.3, min_spell_s=0.0)
    assert (kept[1010:1065] == A).all()
    removed = hsmm(real, meta, lam_team=0.1, lam_dead=0.3, min_spell_s=4.0)
    assert (removed[:3000] == H).all()
    # a spell that already clears the minimum is left alone
    long = hsmm(
        proba_from(truth_from(SEGS, meta)), meta, lam_team=0.3, lam_dead=0.5, min_spell_s=2.0
    )
    base = hsmm(
        proba_from(truth_from(SEGS, meta)), meta, lam_team=0.3, lam_dead=0.5, min_spell_s=0.0
    )
    assert np.array_equal(long, base)


def test_refine_moves_a_boundary_to_the_evidence_flip():
    meta = make_meta()
    truth = truth_from(SEGS, meta)
    proba = proba_from(truth)
    off = truth.copy()
    b = 250 + 300  # the H -> A boundary
    off[b - 6 : b] = A  # boundary 6 frames early
    fixed = refine_boundaries(off, proba, meta, max_shift_s=0.5)
    assert np.array_equal(fixed[:3000], truth[:3000])
    unchanged = refine_boundaries(off, proba, meta, max_shift_s=0.0)
    assert np.array_equal(unchanged, off)
    far = truth.copy()
    far[b - 20 : b] = A  # beyond +-0.5 s: moves by at most 12 frames
    part = refine_boundaries(far, proba, meta, max_shift_s=0.5)
    assert (part[b - 8 : b] == A).all() and (part[b - 20 : b - 12] == H).all()


def test_refine_modes_and_validity():
    meta = make_meta()
    truth = truth_from(SEGS, meta)
    proba = proba_from(truth)
    for mode in ("factorised", "joint"):
        out = refine_boundaries(truth, proba, meta, max_shift_s=1.0, mode=mode)
        assert np.array_equal(out, truth)
    try:
        refine_boundaries(truth, proba, meta, mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("bad mode accepted")


def test_prune_short_spells():
    seq = [H] * 100 + [A] * 10 + [H] * 100 + [D] * 90 + [A] * 90 + [H] * 5 + [D] * 5
    arr = np.array(seq, dtype=np.int8)
    out = prune_short_spells(arr, 0.5)  # 12.5 frames
    assert (out[100:110] == H).all()  # the 10-frame A between two H spells is absorbed
    assert (out[390:395] == A).all()  # the 5-frame H goes to the longer neighbour, the A spell
    assert np.array_equal(out[:100], arr[:100]) and np.array_equal(out[210:300], arr[210:300])
    assert np.array_equal(prune_short_spells(arr, 0.0), arr)


def test_priors_roundtrip_schema_and_relative_path(tmp_path):
    meta = make_meta()
    proba = proba_from(truth_from(SEGS, meta))
    fitted = fit_priors([truth_from(SEGS, meta)])
    path = tmp_path / "priors.json"
    save_priors(fitted, path)
    loaded = load_priors(path)
    base = hsmm(proba, meta, priors=None, refine_s=0.0)
    assert np.array_equal(base, hsmm(proba, meta, priors=DEFAULT_MODEL, refine_s=0.0))
    assert np.array_equal(
        base, hsmm(proba, meta, priors=load_priors(save_default(tmp_path)), refine_s=0.0)
    )
    rel = "_tmp_hsmm_priors.json"
    target = REPO_ROOT / rel
    try:
        save_priors(DEFAULT_MODEL, rel)
        assert target.is_file()
        again = hsmm(proba, meta, priors=rel, refine_s=0.0)
        assert np.array_equal(base, again)
    finally:
        target.unlink(missing_ok=True)
    bad = {
        "alive": {"mu": 0.0, "sigma": 1.0},
        "dead": {"mu": 0.0, "sigma": 1.0},
        "trans": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_priors(bad_path)
    short = {"alive": {"mu": 0.0, "sigma": 1.0}, "dead": {"mu": 0.0, "sigma": 1.0}}
    with pytest.raises(ValueError):
        save_priors(short, tmp_path / "short.json")
    uneven = {
        "alive": loaded["alive"],
        "dead": loaded["dead"],
        "trans": [[0.0, 0.2, 0.2], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    }
    with pytest.raises(ValueError):
        save_priors(uneven, tmp_path / "uneven.json")


def save_default(tmp_path):
    path = tmp_path / "default.json"
    save_priors(DEFAULT_MODEL, path)
    return path


def _duration_tables(nb, model, dt, dmax, flat):
    if flat:
        lp = [np.concatenate(([NEG], np.zeros(dmax[k]))) for k in range(3)]
        lsurv = [np.zeros(max(nb, dmax[k]) + 1) for k in range(3)]
    else:
        lp = [_log_pmf(model, k, dmax[k], dt) for k in range(3)]
        lsurv = [_log_surv(model, k, max(nb, dmax[k]), dt) for k in range(3)]
    return lp, lsurv


def _score_segs(segs, eb, lp, lsurv, log_t, log_pi, beta) -> float:
    total = 0.0
    prev = None
    last = len(segs) - 1
    for i, (s, e, k) in enumerate(segs):
        d = e - s
        total += float(eb[e, k] - eb[s, k] + beta)
        if i == 0:
            total += float(log_pi[k] + lsurv[k][d])
        elif i == last:
            total += float(log_t[prev, k] + lsurv[k][d])
        else:
            total += float(log_t[prev, k] + lp[k][d])
        prev = k
    return total


def _enumerate_best(nb, eb, lp, lsurv, log_t, log_pi, beta, dmax) -> float:
    """Every segmentation: 3 classes, caps on every spell except the opening one, beta per spell.
    The opening spell is left-censored in the DP and is not duration-capped; the last spell of a
    multi-spell path is capped and uses the survival, middle spells use the pmf."""
    best = -1e300

    def rec(start: int, prev: int, acc: float, first: bool) -> None:
        nonlocal best
        d_last = nb - start
        for k in range(3):
            if not first and d_last > dmax[k]:
                continue
            add = float(eb[nb, k] - eb[start, k] + beta)
            if first:
                add += float(log_pi[k] + lsurv[k][d_last])
            else:
                add += float(log_t[prev, k] + lsurv[k][d_last])
            val = acc + add
            if val > best:
                best = val
        for k in range(3):
            d_lim = nb - 1 - start
            if not first:
                d_lim = min(d_lim, dmax[k])
            if d_lim < 1:
                continue
            base = acc + beta + float(log_pi[k] if first else log_t[prev, k])
            dur = lsurv[k] if first else lp[k]
            origin = float(eb[start, k])
            column = eb[:, k]
            for d in range(1, d_lim + 1):
                rec(start + d, k, base + float(column[start + d] - origin + dur[d]), False)

    rec(0, 0, 0.0, True)
    return best


def test_dp_matches_exhaustive_segmentation_on_random_posteriors():
    rng = np.random.default_rng(0)
    fps, block = 5, 1
    dt = block / fps
    n_cases = 0
    for _ in range(100):
        nb = int(rng.integers(1, 9))  # 1..8 blocks, at most 10
        proba = rng.random((nb, 3)).astype(np.float32)
        proba /= proba.sum(axis=1, keepdims=True)
        if nb > 1 and rng.random() < 0.4:
            proba[int(rng.integers(0, nb))] = np.nan
        lam_team = float(rng.choice([0.0, 0.14, 0.5]))
        lam_dead = float(rng.choice([0.0, 0.1, 0.5]))
        beta = float(rng.choice([-6.0, -1.0, 0.0, 2.0]))
        self_prob = float(rng.choice([0.02, 0.05]))
        flat = bool(rng.random() < 0.25)
        max_alive_s = float(rng.choice([1.0, 3.0, 40.0]))
        max_dead_s = float(rng.choice([1.0, 5.0, 100.0]))
        dmax = (int(max_alive_s / dt), int(max_alive_s / dt), int(max_dead_s / dt))
        eb = _block_sums(frame_emissions(proba, lam_team, lam_dead), block)
        segs = _hsmm_period(eb, DEFAULT_MODEL, dt, beta, dmax, self_prob, flat=flat)
        assert segs[0][0] == 0 and segs[-1][1] == nb
        cursor = 0
        for s, e, k in segs:
            assert s == cursor and e > s and k in (0, 1, 2)
            cursor = e
        lp, lsurv = _duration_tables(nb, DEFAULT_MODEL, dt, dmax, flat)
        log_t = _transition_logs(DEFAULT_MODEL, self_prob)
        log_pi = np.log(np.full(3, 1.0 / 3.0))
        dp_score = _score_segs(segs, eb, lp, lsurv, log_t, log_pi, beta)
        best = _enumerate_best(nb, eb, lp, lsurv, log_t, log_pi, beta, dmax)
        assert dp_score == pytest.approx(best, rel=1e-6, abs=1e-5)
        meta = MatchMeta(
            "synthetic", "synthetic", nb, fps, 105.0, 68.0, (Period(1, 0, nb),)
        ).validate()
        labels = hsmm(
            proba,
            meta,
            lam_team=lam_team,
            lam_dead=lam_dead,
            beta=beta,
            block=block,
            duration_prior="flat" if flat else "lognormal",
            max_alive_s=max_alive_s,
            max_dead_s=max_dead_s,
            self_prob=self_prob,
            refine_s=0.0,
            min_spell_s=0.0,
        )
        assert np.array_equal(labels, _expand(segs, nb, block))
        n_cases += 1
    assert n_cases == 100


def test_rejects_bad_duration_prior_and_outside():
    meta = make_meta(((0, 20), (30, 50)))
    proba = np.full((meta.n_frames, 3), 1 / 3, dtype=np.float32)
    with pytest.raises(ValueError):
        hsmm(proba, meta, duration_prior="bogus")
    with pytest.raises(ValueError):
        hsmm(proba, meta, outside="nope")


@pytest.mark.slow
def test_hsmm_decodes_a_long_match_quickly():
    n = 160_000
    meta = MatchMeta(
        "synthetic",
        "synthetic",
        n,
        FPS,
        105.0,
        68.0,
        (Period(1, 0, 80_000), Period(2, 80_000, 160_000)),
    ).validate()
    rng = np.random.default_rng(0)
    proba = rng.random((n, 3)).astype(np.float32)
    proba /= proba.sum(axis=1, keepdims=True)
    proba[::100] = np.nan
    started = time.perf_counter()
    labels = hsmm(proba, meta)
    elapsed = time.perf_counter() - started
    validate_labels(labels, n)
    assert (labels[~meta.played_mask()] == X).all()
    assert elapsed <= 1.5, f"decode took {elapsed:.3f}s"
