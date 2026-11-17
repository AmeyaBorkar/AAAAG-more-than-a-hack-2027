"""Explicit-duration semi-Markov decoder: (T, 3) posteriors over (H, A, D) to (T,) int8 labels.

Signals used: the fused posterior, and, when `switch_weight` > 0, an optional per-frame switch logit.
Frame evidence is tempered log-odds (consecutive frames
of this posterior are not independent observations):

    D vs alive      lam_dead * logit P(D)
    H vs A | alive  lam_team * logit P(H) / (P(H) + P(A))

NaN rows contribute nothing. Each period is decoded on its own (a boundary never carries state):
block Viterbi with a lognormal duration and a class-transition prior, then a full-rate boundary
nudge of at most `refine_s`, then optional removal of alive spells shorter than `min_spell_s`.
Frames outside the periods are X.

Where it fails: it can only re-segment what the posterior already says, so a possession whose
team is wrong for its whole length stays wrong. Alive spells share one lognormal and stoppages
another, which is a poor fit to bimodal stoppage lengths. `beta` (default -6) charges every
spell, so a weakly supported short spell is dropped. The duration cap limits every spell except
the one that opens the period; that opening spell is left-censored and uncapped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from ..contracts import A, D, H, MatchMeta, X
from ..paths import REPO_ROOT
from ..registry import DECODERS

NEG = -1e18
CLIP = 1e-3
SWITCH_CLIP = 6.9  # a switch logit outside this is saturated before it enters the DP

# Fitted on the TRUE labels of the five IDSSE dev matches. mu, sigma: lognormal of the duration
# in seconds; trans[j][k]: P(next class k | current class j), j != k. Use on other matches.
DEFAULT_MODEL = {
    "alive": {"mu": 1.301769, "sigma": 1.238439},
    "dead": {"mu": 2.908683, "sigma": 0.828150},
    "trans": [
        [0.0, 0.756633, 0.243367],
        [0.774977, 0.0, 0.225023],
        [0.517034, 0.482966, 0.0],
    ],
}


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _segments(codes: np.ndarray) -> list[tuple[int, int, int]]:
    if len(codes) == 0:
        return []
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    b = np.concatenate(([0], br, [len(codes)]))
    return [(int(b[i]), int(b[i + 1]), int(codes[b[i]])) for i in range(len(b) - 1)]


def fit_priors(label_arrays: list[np.ndarray], fps: int = 25) -> dict:
    """Lognormal duration (seconds) of alive spells (H and A pooled) and of stoppages, and the
    class-to-class transition matrix, from true label arrays (X outside the periods). The first
    and last segment of every period are censored and skipped."""
    durs: dict[str, list[float]] = {"alive": [], "dead": []}
    counts = np.ones((3, 3))  # +1 smoothing
    for labels in label_arrays:
        for lo, hi in zip(*_runs(labels != X), strict=True):
            segs = _segments(labels[lo:hi])
            for i, (s, e, c) in enumerate(segs):
                if 0 < i < len(segs) - 1:
                    durs["dead" if c == D else "alive"].append((e - s) / fps)
                if i + 1 < len(segs):
                    counts[c, segs[i + 1][2]] += 1
    np.fill_diagonal(counts, 0.0)
    trans = counts / counts.sum(axis=1, keepdims=True)
    out: dict = {"trans": trans.tolist()}
    for name, values in durs.items():
        logs = np.log(np.maximum(np.asarray(values), 1.0 / fps))
        out[name] = {"mu": float(logs.mean()), "sigma": float(logs.std(ddof=1))}
    return out


def validate_priors(model: dict) -> dict:
    """Keys alive, dead, trans. trans is 3x3, zero diagonal, rows sum to 1."""
    if not isinstance(model, dict):
        raise ValueError("priors must be a dict with keys 'alive', 'dead' and 'trans'")
    missing = [key for key in ("alive", "dead", "trans") if key not in model]
    if missing:
        raise ValueError(f"priors missing keys: {', '.join(missing)}")
    for name in ("alive", "dead"):
        comp = model[name]
        if not isinstance(comp, dict) or "mu" not in comp or "sigma" not in comp:
            raise ValueError(f"priors[{name!r}] must contain mu and sigma")
        mu, sigma = float(comp["mu"]), float(comp["sigma"])
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"priors[{name!r}] needs finite mu and sigma > 0")
    trans = np.asarray(model["trans"], dtype=np.float64)
    if trans.shape != (3, 3) or not np.isfinite(trans).all():
        raise ValueError("priors['trans'] must be a finite 3x3 matrix")
    if np.any(trans < -1e-8):
        raise ValueError("priors['trans'] must be non-negative")
    if not np.allclose(np.diag(trans), 0.0, atol=1e-8):
        raise ValueError("priors['trans'] diagonal must be 0")
    if not np.allclose(trans.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("priors['trans'] rows must sum to 1")
    return model


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def save_priors(model: dict, path: str | Path) -> None:
    """Write a priors dict as JSON. Relative paths resolve from the repository root."""
    validate_priors(model)
    payload = {
        "alive": {"mu": float(model["alive"]["mu"]), "sigma": float(model["alive"]["sigma"])},
        "dead": {"mu": float(model["dead"]["mu"]), "sigma": float(model["dead"]["sigma"])},
        "trans": np.asarray(model["trans"], dtype=np.float64).tolist(),
    }
    _resolve_path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_priors(path: str | Path) -> dict:
    """Load a priors JSON file and check its schema. Relative paths resolve from the repository root."""
    resolved = _resolve_path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"priors file not found: {resolved}") from exc
    payload = json.loads(text)
    validate_priors(payload)
    return payload


def _resolve_priors(priors: dict | str | Path | None) -> dict:
    if priors is None:
        return DEFAULT_MODEL
    if isinstance(priors, str | Path):
        return load_priors(priors)
    if isinstance(priors, dict):
        validate_priors(priors)
        return priors
    raise TypeError(f"priors must be None, a dict, or a path, got {type(priors).__name__}")


def _class_of(state: int) -> str:
    return "dead" if state == D else "alive"


def _log_pmf(model: dict, state: int, dmax: int, dt: float) -> np.ndarray:
    """(dmax + 1,) log P(duration = d blocks), lognormal discretised on the block grid (index 0
    unused)."""
    p = model[_class_of(state)]
    edges = (np.arange(0, dmax + 2) - 0.5) * dt
    edges[0] = 0.0
    cdf = norm.cdf((np.log(np.maximum(edges, 1e-9)) - p["mu"]) / p["sigma"])
    cdf[0] = 0.0
    pmf = np.maximum(np.diff(cdf), 1e-300)
    out = np.full(dmax + 1, NEG)
    out[1:] = np.log(pmf[:dmax])
    return out


def _log_surv(model: dict, state: int, nmax: int, dt: float) -> np.ndarray:
    """(nmax + 1,) log P(duration >= d blocks) (index 0 unused)."""
    p = model[_class_of(state)]
    d = np.arange(0, nmax + 1)
    edge = np.maximum((d - 0.5) * dt, 1e-9)
    out = norm.logsf((np.log(edge) - p["mu"]) / p["sigma"])
    out[0] = 0.0
    return np.maximum(out, -700.0)


def posterior_logits(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ld, lt): per-frame log-odds of dead vs alive, logit P(D), and of home vs away given alive,
    logit P(H) / (P(H) + P(A)); both clipped to +/- 6.9 and 0 where the posterior has no opinion."""
    p = proba.astype(np.float64)
    has = ~np.isnan(p[:, 0])
    pd_ = np.where(has, np.clip(np.nan_to_num(p[:, 2], nan=0.5), CLIP, 1 - CLIP), 0.5)
    alive = np.nan_to_num(p[:, 0] + p[:, 1], nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        home = np.where(alive > CLIP, np.nan_to_num(p[:, 0] / alive, nan=0.5), 0.5)
    home = np.clip(home, CLIP, 1 - CLIP)
    return np.log(pd_) - np.log1p(-pd_), np.log(home) - np.log1p(-home)


def frame_emissions(proba: np.ndarray, lam_team: float, lam_dead: float) -> np.ndarray:
    """(n, 3) tempered per-frame log-evidence for (H, A, D) from a posterior span."""
    ld, lt = posterior_logits(proba)
    e = np.empty((len(ld), 3))
    e[:, H] = -0.5 * lam_dead * ld + 0.5 * lam_team * lt
    e[:, A] = -0.5 * lam_dead * ld - 0.5 * lam_team * lt
    e[:, D] = 0.5 * lam_dead * ld
    return e


def _block_sums(e: np.ndarray, block: int) -> np.ndarray:
    """(nb + 1, 3) cumulative block sums of the frame emissions (row 0 is zero)."""
    starts = np.arange(0, len(e), block)
    sums = np.add.reduceat(e, starts, axis=0)
    return np.vstack([np.zeros((1, 3)), np.cumsum(sums, axis=0)])


def _transition_logs(model: dict, self_prob: float) -> np.ndarray:
    t = np.asarray(model["trans"], dtype=np.float64) * (1.0 - self_prob)
    np.fill_diagonal(t, self_prob)
    return np.log(t)


def _period_switch_bonus(switch: np.ndarray, block: int, weight: float) -> np.ndarray:
    """(nb + 1,) bonus. Index t is `weight * max(clip(logit))` over the block that ends at t.

    NaN logits count as 0 before the clip. Index 0 is unused. The block that ends at step t is
    frames `[(t - 1) * block, t * block)`.
    """
    values = np.array(switch, dtype=np.float64, copy=True)
    values = np.nan_to_num(values, nan=0.0)
    np.clip(values, -SWITCH_CLIP, SWITCH_CLIP, out=values)
    starts = np.arange(0, len(values), block)
    block_max = np.maximum.reduceat(values, starts)
    bonus = np.zeros(len(block_max) + 1, dtype=np.float64)
    bonus[1:] = weight * block_max
    return bonus


def _hsmm_period(
    eb: np.ndarray,
    model: dict,
    dt: float,
    beta: float,
    dmax: tuple[int, int, int],
    self_prob: float,
    flat: bool = False,
    switch_bonus: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Viterbi over (state, duration) for one period. eb: (nb + 1, 3) cumulative block emissions.
    Score = sum of block emissions + sum over segments of [log P(duration) + log P(class | previous)
    + beta]; the first and the last segment use the survival function (the period cuts them)."""
    nb = eb.shape[0] - 1
    log_t = _transition_logs(model, self_prob)
    log_pi = np.log(np.full(3, 1.0 / 3.0))
    if flat:  # no duration information: every segment pays beta and the class transition only
        lp = [np.concatenate(([NEG], np.zeros(dmax[k]))) for k in range(3)]
        lsurv = [np.zeros(max(nb, dmax[k]) + 1) for k in range(3)]
    else:
        lp = [_log_pmf(model, k, dmax[k], dt) for k in range(3)]
        lsurv = [_log_surv(model, k, max(nb, dmax[k]), dt) for k in range(3)]
    lpr = [lp[k][1 : dmax[k] + 1][::-1].copy() for k in range(3)]
    lsr = [lsurv[k][1 : dmax[k] + 1][::-1].copy() for k in range(3)]
    g = np.full((3, nb + 1), NEG)
    bp_d = np.zeros((nb + 1, 3), dtype=np.int32)
    bp_j = np.zeros((nb + 1, 3), dtype=np.int8)
    delta = np.full(3, NEG)
    for t in range(1, nb + 1):
        for k in range(3):
            dm = min(t - 1, dmax[k])
            best, arg = log_pi[k] + lsurv[k][t], 0  # first segment: [0, t), censored at the start
            if dm >= 1:
                w = g[k, t - dm : t] + lpr[k][dmax[k] - dm :]
                j = int(w.argmax())
                if w[j] > best:
                    best, arg = float(w[j]), dm - j
            delta[k] = eb[t, k] + beta + best
            bp_d[t, k] = arg
        cand = delta[:, None] + log_t  # [j, k]
        if switch_bonus is not None:
            # Only a home/away change collects the bonus. Into or out of D is unchanged.
            bonus = switch_bonus[t]
            cand[H, A] += bonus
            cand[A, H] += bonus
        jbest = cand.argmax(axis=0)
        g[:, t] = cand[jbest, np.arange(3)] - eb[t]
        bp_j[t] = jbest
    # last segment: censored at the end of the period
    final, last_d = np.full(3, NEG), np.zeros(3, dtype=np.int64)
    for k in range(3):
        best, arg = log_pi[k] + lsurv[k][nb], 0
        dm = min(nb - 1, dmax[k])
        if dm >= 1:
            w = g[k, nb - dm : nb] + lsr[k][dmax[k] - dm :]
            j = int(w.argmax())
            if w[j] > best:
                best, arg = float(w[j]), dm - j
        final[k] = eb[nb, k] + beta + best
        last_d[k] = arg
    k, t, d = int(final.argmax()), nb, int(last_d[int(final.argmax())])
    segs = []
    while True:
        d = t if d == 0 else d
        s = t - d
        segs.append((s, t, k))
        if s == 0:
            break
        k = int(bp_j[s, k])
        t = s
        d = int(bp_d[t, k])
    return segs[::-1]


def _expand(segs: list[tuple[int, int, int]], n: int, block: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int8)
    for s, e, k in segs:
        out[s * block : min(e * block, n)] = k
    return out


def _finish(labels: np.ndarray, meta: MatchMeta, outside: str) -> np.ndarray:
    if outside == "X":
        labels[~meta.played_mask()] = X
    elif outside != "keep":
        raise ValueError(f"outside must be 'X' or 'keep', got {outside!r}")
    return labels


def refine_boundaries(
    labels: np.ndarray,
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    max_shift_s: float = 0.5,
    mode: str = "factorised",
    switch: np.ndarray | None = None,
) -> np.ndarray:
    """Move every boundary within +/- max_shift_s to the frame that maximises the local evidence.

    mode "factorised": a stoppage edge is judged by logit P(D) alone, a H/A edge by the team
    log-odds alone (so a team opinion can never drag a stoppage edge). mode "joint": the left label's
    log P plus the right label's log P (floored at 1e-3), the plain maximum-likelihood change point.
    Boundaries are visited left to right and stay at least one frame apart; ties keep the boundary
    where it is (or the nearest maximiser). NaN rows say nothing. mode "switch": a H/A edge goes to
    the frame with the highest learned switch logit (`switch`, NaN as 0, clipped like the DP bonus),
    a stoppage edge is judged as in "factorised" (E42)."""
    out = labels.copy()
    m = int(round(max_shift_s * meta.fps))
    if m <= 0:
        return out
    if mode not in ("factorised", "joint", "switch"):
        raise ValueError(f"mode must be 'factorised', 'joint' or 'switch', got {mode!r}")
    if mode == "switch":
        if switch is None or len(switch) != len(labels):
            raise ValueError("mode 'switch' needs a (T,) switch logit")
        sw = np.clip(
            np.nan_to_num(np.asarray(switch, dtype=np.float64), nan=0.0), -SWITCH_CLIP, SWITCH_CLIP
        )
    ld, lt = posterior_logits(proba)
    logp = np.log(np.clip(np.nan_to_num(proba.astype(np.float64), nan=1.0), CLIP, 1.0))
    for lo, hi in meta.runs():
        seg = _segments(out[lo:hi])
        if len(seg) < 2:
            continue
        cum_d = np.concatenate(([0.0], np.cumsum(ld[lo:hi])))
        cum_t = np.concatenate(([0.0], np.cumsum(lt[lo:hi])))
        cum_p = np.vstack([np.zeros((1, 3)), np.cumsum(logp[lo:hi], axis=0)])
        bounds = [s for s, _, _ in seg[1:]]
        codes = [c for _, _, c in seg]
        left_limit = 0
        for i, b in enumerate(bounds):
            u, v = codes[i], codes[i + 1]
            right_limit = bounds[i + 1] if i + 1 < len(bounds) else hi - lo
            cand = np.arange(max(b - m, left_limit + 1), min(b + m, right_limit - 1) + 1)
            if len(cand) > 1:
                if mode == "joint":
                    score = cum_p[cand, u] - cum_p[cand, v]
                elif D in (u, v):
                    score = (1.0 if u == D else -1.0) * cum_d[cand]
                elif mode == "switch":
                    score = sw[lo + cand]
                else:
                    score = (1.0 if u == H else -1.0) * cum_t[cand]
                best = score.max()
                near = cand[score >= best - 1e-9]
                bounds[i] = int(near[np.argmin(np.abs(near - b))])
            left_limit = bounds[i]
        edges = np.concatenate(([0], bounds, [hi - lo]))
        out[lo:hi] = np.repeat(np.array(codes, dtype=np.int8), np.diff(edges))
    return out


def prune_short_spells(labels: np.ndarray, min_s: float, fps: int = 25) -> np.ndarray:
    """Alive spells shorter than `min_s` are absorbed by their longer neighbour (shortest first,
    repeatedly), per period (runs of non-X labels)."""
    out = labels.copy()
    lim = int(round(min_s * fps))
    if lim <= 1:
        return out
    for lo, hi in zip(*_runs(labels != X), strict=True):
        segs = [list(s) for s in _segments(labels[lo:hi])]
        while len(segs) > 1:
            cand = [(e - s, i) for i, (s, e, c) in enumerate(segs) if c < D and e - s < lim]
            if not cand:
                break
            _, i = min(cand)
            prev = segs[i - 1] if i > 0 else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            pick = (
                prev
                if nxt is None
                else nxt
                if prev is None
                else (prev if prev[1] - prev[0] >= nxt[1] - nxt[0] else nxt)
            )
            segs[i][2] = pick[2]
            merged: list[list[int]] = []
            for s, e, c in segs:
                if merged and merged[-1][2] == c:
                    merged[-1][1] = e
                else:
                    merged.append([s, e, c])
            segs = merged
        for s, e, c in segs:
            out[lo + s : lo + e] = c
    return out


@DECODERS.register("hsmm", version="1", example={"refine_s": 0.5})
def hsmm(
    proba: np.ndarray,
    meta: MatchMeta,
    *,
    priors: dict | str | Path | None = None,
    lam_team: float = 0.14,
    lam_dead: float = 0.1,
    beta: float = -6.0,
    block: int = 5,
    duration_prior: str = "lognormal",
    max_alive_s: float = 40.0,
    max_dead_s: float = 100.0,
    self_prob: float = 0.02,
    refine_s: float = 0.5,
    refine_mode: str = "factorised",
    min_spell_s: float = 0.0,
    outside: str = "X",
    switch_weight: float = 0.0,
    switch: np.ndarray | None = None,
) -> np.ndarray:
    """MAP segmentation of the explicit-duration HSMM, per period, at `block` frames per step, then
    boundaries refined at full rate within +/- refine_s (0 = off). `min_spell_s` > 0 absorbs alive
    spells shorter than that afterwards (0 = off). `priors` is None (the built-in duration model),
    a dict of that shape, or a path to a JSON file (relative paths resolve from the repository root).
    `beta` is added once per segment: negative charges every spell. `duration_prior="flat"` drops
    the lognormal durations and keeps only beta and the class transitions.

    `switch_weight` > 0 and a `(T,)` `switch` logit add
    `switch_weight * max(clip(logit, -6.9, 6.9))` (NaN counts as 0) to the H↔A transition score
    of the block that ends at that boundary. Weight 0, or no logit, does not read the channel and
    the segmentation is the one without the argument."""
    if duration_prior not in ("lognormal", "flat"):
        raise ValueError(f"duration_prior must be 'lognormal' or 'flat', got {duration_prior!r}")
    model = _resolve_priors(priors)
    fps = meta.fps
    dt = block / fps
    dmax = (int(max_alive_s / dt), int(max_alive_s / dt), int(max_dead_s / dt))
    labels = np.full(len(proba), D, dtype=np.int8)
    use_switch = switch_weight > 0.0 and switch is not None
    if use_switch and len(switch) != len(proba):
        raise ValueError(f"switch length {len(switch)} != posterior length {len(proba)}")
    for lo, hi in meta.runs():
        e = frame_emissions(proba[lo:hi], lam_team, lam_dead)
        if use_switch:
            segs = _hsmm_period(
                _block_sums(e, block),
                model,
                dt,
                beta,
                dmax,
                self_prob,
                flat=duration_prior == "flat",
                switch_bonus=_period_switch_bonus(switch[lo:hi], block, float(switch_weight)),
            )
        else:
            segs = _hsmm_period(
                _block_sums(e, block),
                model,
                dt,
                beta,
                dmax,
                self_prob,
                flat=duration_prior == "flat",
            )
        labels[lo:hi] = _expand(segs, hi - lo, block)
    if refine_s > 0:
        labels = refine_boundaries(
            labels,
            proba,
            meta,
            max_shift_s=refine_s,
            mode=refine_mode,
            switch=switch if refine_mode == "switch" else None,
        )
    if min_spell_s > 0:
        for lo, hi in meta.runs():
            labels[lo:hi] = prune_short_spells(labels[lo:hi], min_spell_s, fps=fps)
    return _finish(labels, meta, outside)
