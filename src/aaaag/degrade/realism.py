"""Realism classifier for degrade (issue #44, arnav-10): can a classifier tell BAR-ATH windows from
degraded IDSSE windows, beyond how well it tells two IDSSE matches apart?

Windows are consecutive WINDOW_S stretches inside each period (a trailing partial window is
dropped). Each window gets two feature groups, from tracking only (no labels):
- GROUP_A, the tracking fingerprint (what degrade is meant to reproduce; the gate uses only these):
  ball missing share, gap starts, longest gap, seen-ball speed p50/p90 and acceleration p50, the
  share of seen ball frames off the pitch / within 2 m of a line / frozen; objects per team (mean,
  min, max), track-fragment starts, player acceleration p50/p90, the share of frames with a
  same-team pair closer than 0.5 m, and the share of slot-frames jumping faster than 12 m/s;
- GROUP_B, playing style (reported, never gated): collective player speed, team spread, and the
  ball's median distance from the halfway line and from the long axis.
Speeds are displacements over SPEED_LAG frames; accelerations are frame-to-frame second differences.

`cv_auc` cross-validates a classifier with folds made of whole time blocks (BLOCK_S), so
neighbouring windows never sit on both sides of a split, and returns out-of-fold scores, a
block-bootstrap interval and permutation importances.

Where it fails: one target match gives only a few hundred windows, so AUCs carry real noise (the interval
shows it); a feature that is undefined in a window (no seen ball) is imputed with the median.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import AWAY, HOME, Tracking
from .context import frozen_ball, line_distance
from .playerstats import _close_pair

WINDOW_S = 30.0
BLOCK_S = 300.0
SPEED_LAG = 5
JUMP_MPS = 12.0
GROUP_A = (
    "ball_missing",
    "ball_gap_starts",
    "ball_longest_gap_s",
    "ball_speed_p50",
    "ball_speed_p90",
    "ball_accel_p50",
    "ball_off_share",
    "ball_edge_share",
    "ball_frozen_share",
    "objects_mean",
    "objects_min",
    "objects_max",
    "fragment_starts",
    "player_accel_p50",
    "player_accel_p90",
    "close_pair_share",
    "jump_share",
)
GROUP_B = (
    "collective_speed",
    "team_spread",
    "ball_abs_x_p50",
    "ball_abs_y_p50",
)


def window_features(tracking: Tracking) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Features of every window, {name: (W,)}, and each window's time-block id (W,)."""
    meta, fps = tracking.meta, tracking.meta.fps
    size = int(round(WINDOW_S * fps))
    per_block = int(round(BLOCK_S / WINDOW_S))
    ball = tracking.ball_xy.astype(np.float64)
    xy = tracking.player_xy.astype(np.float64)
    ball_seen = ~np.isnan(ball[:, 0])
    seen = ~np.isnan(xy[..., 0])
    distance = line_distance(tracking)
    frozen = frozen_ball(tracking)

    ball_speed = _lagged_speed(ball, fps)
    ball_accel = _accel(ball[:, None, :], fps)[:, 0]
    player_speed = _lagged_speed(xy, fps)
    player_accel = _accel(xy, fps)
    step = np.full(seen.shape, np.nan)
    step[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=2) * fps
    starts = np.zeros(seen.shape, dtype=bool)
    starts[1:] = seen[1:] & ~seen[:-1]
    for period in meta.periods:
        starts[period.start] = False  # every track "starts" at a period start; not a fragment
    counts = np.stack([seen[:, tracking.slot_team == t].sum(axis=1) for t in (HOME, AWAY)], 1)
    close = np.zeros(len(ball), dtype=bool)
    spread = np.zeros(len(ball))
    for team in (HOME, AWAY):
        team_xy = xy[:, tracking.slot_team == team]
        close |= _close_pair(team_xy)
        with np.errstate(invalid="ignore"), _quiet():
            centre = np.nanmean(team_xy, axis=1, keepdims=True)
            spread += np.nanmean(np.linalg.norm(team_xy - centre, axis=2), axis=1) / 2

    columns: dict[str, list[np.ndarray]] = {name: [] for name in GROUP_A + GROUP_B}
    blocks = []
    for period in meta.periods:
        n = period.length // size
        if n == 0:
            continue
        lo, hi = period.start, period.start + n * size

        def win(a: np.ndarray, lo: int = lo, hi: int = hi, n: int = n) -> np.ndarray:
            return a[lo:hi].reshape(n, size, *a.shape[1:])

        b_seen = win(ball_seen)
        gaps = b_seen[:, :-1] & ~b_seen[:, 1:]
        with np.errstate(invalid="ignore"), _quiet():
            columns["ball_missing"].append(1 - b_seen.mean(axis=1))
            columns["ball_gap_starts"].append(gaps.sum(axis=1).astype(float))
            columns["ball_longest_gap_s"].append(_longest_run(~b_seen) / fps)
            columns["ball_speed_p50"].append(np.nanpercentile(win(ball_speed), 50, axis=1))
            columns["ball_speed_p90"].append(np.nanpercentile(win(ball_speed), 90, axis=1))
            columns["ball_accel_p50"].append(np.nanpercentile(win(ball_accel), 50, axis=1))
            seen_n = np.maximum(b_seen.sum(axis=1), 1)
            d = win(distance)
            columns["ball_off_share"].append((b_seen & (d < 0)).sum(axis=1) / seen_n)
            columns["ball_edge_share"].append((b_seen & (d >= 0) & (d < 2)).sum(axis=1) / seen_n)
            columns["ball_frozen_share"].append((b_seen & win(frozen)).sum(axis=1) / seen_n)
            c = win(counts)
            columns["objects_mean"].append(c.mean(axis=(1, 2)))
            columns["objects_min"].append(c.min(axis=(1, 2)).astype(float))
            columns["objects_max"].append(c.max(axis=(1, 2)).astype(float))
            columns["fragment_starts"].append(win(starts).sum(axis=(1, 2)).astype(float))
            pa = win(player_accel).reshape(n, -1)
            columns["player_accel_p50"].append(np.nanpercentile(pa, 50, axis=1))
            columns["player_accel_p90"].append(np.nanpercentile(pa, 90, axis=1))
            columns["close_pair_share"].append(win(close).mean(axis=1))
            st = win(step).reshape(n, -1)
            valid = np.maximum(np.isfinite(st).sum(axis=1), 1)
            columns["jump_share"].append((st > JUMP_MPS).sum(axis=1) / valid)
            columns["collective_speed"].append(np.nanmean(win(player_speed).reshape(n, -1), axis=1))
            columns["team_spread"].append(np.nanmean(win(spread), axis=1))
            bx = win(np.abs(ball[:, 0]))
            by = win(np.abs(ball[:, 1]))
            columns["ball_abs_x_p50"].append(np.nanpercentile(bx, 50, axis=1))
            columns["ball_abs_y_p50"].append(np.nanpercentile(by, 50, axis=1))
        blocks.append(period.index * 10_000 + np.arange(n) // per_block)
    features = {name: np.concatenate(parts) for name, parts in columns.items()}
    return features, np.concatenate(blocks)


def matrix(features: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.stack([features[name] for name in names], axis=1)


@dataclass
class AucResult:
    auc: float
    interval: tuple[float, float]
    importance: dict[str, float]  # mean drop in out-of-fold AUC when the feature is permuted


def cv_auc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    names: tuple[str, ...],
    *,
    kind: str = "logistic",
    n_splits: int = 5,
    n_boot: int = 1000,
    n_perm: int = 5,
    seed: int = 0,
) -> AucResult:
    """Out-of-fold ROC AUC of telling y=1 from y=0, folds made of whole groups (time blocks)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    rng = np.random.default_rng(seed)
    scores = np.full(len(y), np.nan)
    permuted = np.full((len(names), n_perm, len(y)), np.nan)
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in folds.split(x, y, groups):
        model = _model(kind, seed).fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
        for j in range(len(names)):
            for r in range(n_perm):
                shuffled = x[test].copy()
                shuffled[:, j] = rng.permutation(shuffled[:, j])
                permuted[j, r, test] = model.predict_proba(shuffled)[:, 1]
    auc = float(roc_auc_score(y, scores))
    importance = {
        name: auc - float(np.mean([roc_auc_score(y, permuted[j, r]) for r in range(n_perm)]))
        for j, name in enumerate(names)
    }
    boot = []
    by_class = [np.unique(groups[y == c]) for c in (0, 1)]
    index = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    for _ in range(n_boot):
        pick = np.concatenate(
            [index[g] for gs in by_class for g in rng.choice(gs, size=len(gs), replace=True)]
        )
        boot.append(roc_auc_score(y[pick], scores[pick]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return AucResult(auc, (float(lo), float(hi)), importance)


def _model(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
    if kind == "boosting":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    raise ValueError(f"unknown classifier {kind!r}")


def _lagged_speed(a: np.ndarray, fps: int) -> np.ndarray:
    out = np.full(a.shape[:-1], np.nan)
    out[SPEED_LAG:] = np.linalg.norm(a[SPEED_LAG:] - a[:-SPEED_LAG], axis=-1) * fps / SPEED_LAG
    return out


def _accel(a: np.ndarray, fps: int) -> np.ndarray:
    out = np.full(a.shape[:-1], np.nan)
    out[1:-1] = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1) * fps**2
    return out


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """(n,) longest run of True along axis 1 of an (n, size) mask."""
    longest = np.zeros(len(mask))
    current = np.zeros(len(mask))
    for t in range(mask.shape[1]):  # loops over the window length (750), not over frames
        current = np.where(mask[:, t], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


class _quiet:
    """Silence numpy's all-NaN-slice warnings (empty windows become NaN and are imputed)."""

    def __enter__(self):
        import warnings

        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._catch.__exit__(*exc)
