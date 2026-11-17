"""Logistic check that a short spell would have been a clean-ball pass.

`controlled_spell` keeps a spell shorter than `hold_s` when two teammates come within 1 m of the
ball. After an 8-frame Gaussian smooth that radius rarely contains the receiver, so the rule
collapses to a minimum length. These features are computed from the ball the pipeline actually
sees. A logistic regression trained on a smoothed copy, with labels from the clean ball, estimates
whether the clean-ball check would have fired.

The feature vector, in this order, is:

1. ``n_within_1m``, ``n_within_1_5m``, ``n_within_2m``, ``n_within_3m`` — distinct slots of the
   team that have the ball within 1, 1.5, 2 and 3 m on some frame of the span, each capped at 3.
2. ``path_length_m`` — sum of steps between consecutive frames where the ball is seen. A gap is
   not bridged.
3. ``displacement_m`` — straight-line distance from the first seen ball position in the span to
   the last. Zero when the ball is seen at most once.
4. ``max_first_contact_sep_m`` — maximum pairwise distance between the ball positions at the
   first frame each distinct team slot is within 2 m. Zero when fewer than two slots qualify.
5. ``median_min_dist_own_m``, ``median_min_dist_opp_m`` — median over the span of the per-frame
   minimum distance from the ball to a slot of this team, and to a slot of the other team.
   NaN when the span has no finite distance.
6. ``duration_s`` — ``(b - a) / fps``.
7. ``ball_seen_share`` — fraction of the span whose ball coordinates are both finite.

Signals: ``player_xy``, ``slot_team``, ``ball_xy``, and the frame rate on ``meta``. Nothing is
smoothed here; the caller passes the track the pipeline sees.

Where it fails: the count features cannot tell a fourth teammate from a third; a pass whose
receiver stays outside 3 m is invisible to the counts; a one-frame glimpse has path and
displacement zero; NaN medians have to be imputed before the linear score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Tracking
from ..ml import weights as weight_files

FEATURE_NAMES: tuple[str, ...] = (
    "n_within_1m",
    "n_within_1_5m",
    "n_within_2m",
    "n_within_3m",
    "path_length_m",
    "displacement_m",
    "max_first_contact_sep_m",
    "median_min_dist_own_m",
    "median_min_dist_opp_m",
    "duration_s",
    "ball_seen_share",
)
N_FEATURES = len(FEATURE_NAMES)
RADII_M = np.array([1.0, 1.5, 2.0, 3.0], dtype=np.float64)
CONTACT_M = 2.0
COUNT_CAP = 3

_COUNT = slice(0, 4)
_PATH = 4
_DISP = 5
_SEP = 6
_OWN = 7
_OPP = 8
_DUR = 9
_SEEN = 10

_MODEL_KEYS = ("mean", "scale", "coef", "intercept", "threshold")


def _expit(z: np.ndarray) -> np.ndarray:
    """Sigmoid, stable for large positive and negative logits."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def pass_probabilities(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """``(n,)`` probability that the clean-ball check would fire.

    Columns that are NaN are filled with the training mean, so after standardisation they
    contribute nothing. A stored scale of 0 is treated as 1.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mean = np.asarray(model["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(model["scale"], dtype=np.float64).reshape(-1)
    coef = np.asarray(model["coef"], dtype=np.float64).reshape(-1)
    if x.shape[1] != mean.shape[0] or scale.shape != mean.shape or coef.shape != mean.shape:
        raise ValueError(f"pass model has {mean.shape[0]} features, got {x.shape[1]}")
    safe = np.where(scale == 0.0, 1.0, scale)
    filled = np.where(np.isfinite(x), x, mean)
    return _expit((filled - mean) / safe @ coef + float(model["intercept"]))


def pass_probability(features: np.ndarray, model: dict[str, Any]) -> float:
    """Probability that the clean-ball check would fire on this one span."""
    return float(pass_probabilities(features, model)[0])


def load_pass_model(path: str | Path, sha256: str | None = None) -> dict[str, Any]:
    """Logistic weights. A relative ``path`` is read from the repository root.

    ``sha256``, when set, is a hex prefix of at least 8 characters. The file must contain
    ``mean``, ``scale``, ``coef``, ``intercept`` and ``threshold``.
    """
    resolved = _resolve_model(path, sha256)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"pass_model: {resolved.name} is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"pass_model: {resolved.name} must be a JSON object")
    missing = [key for key in _MODEL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"pass_model: {resolved.name} missing {', '.join(missing)}")
    model: dict[str, Any] = {
        "mean": np.asarray(payload["mean"], dtype=np.float64).reshape(-1),
        "scale": np.asarray(payload["scale"], dtype=np.float64).reshape(-1),
        "coef": np.asarray(payload["coef"], dtype=np.float64).reshape(-1),
        "intercept": float(payload["intercept"]),
        "threshold": float(payload["threshold"]),
    }
    width = model["mean"].shape[0]
    if model["scale"].shape != (width,) or model["coef"].shape != (width,) or width != N_FEATURES:
        raise ValueError(f"pass_model: expected {N_FEATURES} features, got {width}")
    finite = (
        np.isfinite(model["mean"]).all()
        and np.isfinite(model["scale"]).all()
        and np.isfinite(model["coef"]).all()
        and np.isfinite(model["intercept"])
        and np.isfinite(model["threshold"])
    )
    if not finite:
        raise ValueError("pass_model: mean, scale, coef, intercept and threshold must be finite")
    return model


def _resolve_model(path: str | Path, sha256: str | None) -> Path:
    resolved = weight_files.resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"pass_model weights not found: {resolved}")
    if sha256 is None:
        return resolved
    if not isinstance(sha256, str):
        raise TypeError(f"pass_model_sha256 must be a string, got {type(sha256).__name__}")
    pin = sha256.strip().lower()
    if len(pin) < weight_files.MIN_PIN:
        raise ValueError(
            f"pass_model sha256 pin {sha256!r} is shorter than {weight_files.MIN_PIN} characters"
        )
    actual = weight_files.sha256_file(resolved)
    if not actual.startswith(pin):
        raise ValueError(f"pass_model: {resolved.name} has sha256 {actual}, pinned {sha256}")
    return resolved


def _row_median(values: np.ndarray) -> np.ndarray:
    """Median of each row, ignoring non-finite entries. NaN when a row has none.

    An even count averages the two central values, matching ``np.median``.
    """
    finite = np.isfinite(values)
    counts = finite.sum(axis=1).astype(np.int64)
    ordered = np.sort(np.where(finite, values, np.inf), axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    odd = np.flatnonzero((counts % 2 == 1) & (counts > 0))
    if odd.size:
        mid = (counts[odd] - 1) // 2
        out[odd] = ordered[odd, mid]
    even = np.flatnonzero((counts % 2 == 0) & (counts > 0))
    if even.size:
        hi = counts[even] // 2
        lo = hi - 1
        out[even] = 0.5 * (ordered[even, lo] + ordered[even, hi])
    return out


def _window_medians(series: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Median of ``series[start:stop]`` for each window, ignoring non-finite samples."""
    out = np.full(starts.shape[0], np.nan, dtype=np.float64)
    lengths = stops - starts
    for length in np.unique(lengths):
        length = int(length)
        if length <= 0:
            continue
        rows = np.flatnonzero(lengths == length)
        offset = np.arange(length, dtype=np.int64)
        out[rows] = _row_median(series[starts[rows, None] + offset])
    return out


def _per_frame_min(dist: np.ndarray) -> np.ndarray:
    """``(T,)`` minimum over slots. NaN on a frame with no finite distance."""
    if dist.shape[1] == 0:
        return np.full(dist.shape[0], np.nan, dtype=np.float64)
    finite = np.isfinite(dist)
    mins = np.where(finite, dist, np.inf).min(axis=1)
    mins[~finite.any(axis=1)] = np.nan
    return mins


def _slot_distances(ball: np.ndarray, players: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(T, S_mask)`` metres. NaN where the ball or that slot is missing."""
    if not np.any(mask):
        return np.empty((ball.shape[0], 0), dtype=np.float64)
    delta = players[:, mask, :] - ball[:, None, :]
    return np.hypot(delta[..., 0], delta[..., 1])


def _radius_counts(dist_own: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Distinct own slots within each radius, capped at 3. Shape ``(n, 4)``."""
    n = starts.shape[0]
    counts = np.zeros((n, RADII_M.shape[0]), dtype=np.float64)
    if n == 0 or dist_own.shape[1] == 0:
        return counts
    t_len, n_slots = dist_own.shape
    within = np.isfinite(dist_own)[:, :, None] & (dist_own[:, :, None] <= RADII_M)
    prefix = np.zeros((t_len + 1, n_slots, RADII_M.shape[0]), dtype=np.int32)
    prefix[1:] = np.cumsum(within, axis=0, dtype=np.int32)
    hit = (prefix[stops] - prefix[starts]) > 0
    counts[:] = np.minimum(hit.sum(axis=1), COUNT_CAP)
    return counts


def _contact_separation(
    ball: np.ndarray, dist_own: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """Max distance between ball positions at each slot's first contact within 2 m.

    Zero when fewer than two slots of this team come within 2 m in the window.
    """
    n = starts.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or dist_own.shape[1] == 0:
        return out
    t_len, n_slots = dist_own.shape
    contact = np.isfinite(dist_own) & (dist_own <= CONTACT_M)
    frame = np.arange(t_len, dtype=np.int64)[:, None]
    contact_idx = np.where(contact, frame, np.int64(t_len))
    first_after = np.minimum.accumulate(contact_idx[::-1], axis=0)[::-1]
    live = (stops - starts) > 0
    if not np.any(live):
        return out
    first = np.full((n, n_slots), t_len, dtype=np.int64)
    first[live] = first_after[starts[live]]
    valid = first < stops[:, None]
    clipped = np.clip(first, 0, max(t_len - 1, 0))
    xs = np.where(valid, ball[clipped, 0], np.nan)
    ys = np.where(valid, ball[clipped, 1], np.nan)
    finite = valid & np.isfinite(xs) & np.isfinite(ys)
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            both = finite[:, i] & finite[:, j]
            if not np.any(both):
                continue
            dist = np.hypot(xs[both, i] - xs[both, j], ys[both, i] - ys[both, j])
            out[both] = np.maximum(out[both], dist)
    return out


def _kinematics(
    ball: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Path length, displacement and seen share for windows on this ball slice.

    Path length sums steps whose two ends are both seen. Displacement is the straight line
    between the first and last seen sample. Share is the fraction of frames with a ball.
    """
    n = starts.shape[0]
    t_len = ball.shape[0]
    lengths = stops - starts
    path = np.zeros(n, dtype=np.float64)
    disp = np.zeros(n, dtype=np.float64)
    share = np.zeros(n, dtype=np.float64)
    if t_len == 0:
        return path, disp, share
    seen = np.isfinite(ball).all(axis=1)
    seen_prefix = np.zeros(t_len + 1, dtype=np.int32)
    seen_prefix[1:] = np.cumsum(seen, dtype=np.int32)
    live = lengths > 0
    if np.any(live):
        share[live] = (seen_prefix[stops[live]] - seen_prefix[starts[live]]) / lengths[live]
    if t_len >= 2:
        delta = np.diff(ball, axis=0)
        both = seen[:-1] & seen[1:]
        step = np.where(both, np.hypot(delta[:, 0], delta[:, 1]), 0.0)
        step = np.where(np.isfinite(step), step, 0.0)
        prefix = np.zeros(t_len, dtype=np.float64)
        prefix[1:] = np.cumsum(step)
        long = lengths >= 2
        if np.any(long):
            path[long] = prefix[stops[long] - 1] - prefix[starts[long]]
    idx = np.arange(t_len, dtype=np.int64)
    last = np.where(seen, idx, np.int64(-1))
    np.maximum.accumulate(last, out=last)
    first = np.where(seen, idx, np.int64(t_len))
    first = np.minimum.accumulate(first[::-1])[::-1]
    if np.any(live):
        a = starts[live]
        b = stops[live]
        first_i = first[a]
        last_i = last[b - 1]
        ok = (first_i < b) & (last_i >= a) & (last_i >= 0)
        disp_live = np.zeros(int(live.sum()), dtype=np.float64)
        if np.any(ok):
            delta = ball[last_i[ok]] - ball[first_i[ok]]
            disp_live[ok] = np.hypot(delta[:, 0], delta[:, 1])
        disp[live] = disp_live
    return path, disp, share


def span_feature_matrix(tracking: Tracking, bounds: np.ndarray, teams: np.ndarray) -> np.ndarray:
    """``(n, 11)`` features for many spans. ``bounds`` is ``(n, 2)`` of ``[a, b)``.

    Same columns as ``span_features``. Spans are reduced together; each span still sees only
    its own frames. ``0 <= a <= b <= n_frames`` is required.
    """
    bounds = np.asarray(bounds, dtype=np.int64)
    teams = np.asarray(teams, dtype=np.int64).reshape(-1)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] != teams.shape[0]:
        raise ValueError("bounds must be (n, 2) and teams (n,)")
    n = bounds.shape[0]
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    ball_all = np.asarray(tracking.ball_xy)
    players_all = np.asarray(tracking.player_xy)
    slot_team = np.asarray(tracking.slot_team)
    n_frames = ball_all.shape[0]
    if players_all.shape[0] != n_frames:
        raise ValueError("ball_xy and player_xy differ in length")
    if slot_team.shape != (players_all.shape[1],):
        raise ValueError("slot_team does not match player_xy")
    starts = bounds[:, 0]
    stops = bounds[:, 1]
    if int(starts.min()) < 0 or int(stops.max()) > n_frames or np.any(stops < starts):
        raise ValueError("span bounds must satisfy 0 <= a <= b <= n_frames")
    fps = float(tracking.meta.fps)
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    out = np.full((n, N_FEATURES), np.nan, dtype=np.float64)
    out[:, _COUNT] = 0.0
    out[:, _PATH] = 0.0
    out[:, _DISP] = 0.0
    out[:, _SEP] = 0.0
    out[:, _SEEN] = 0.0
    lengths = stops - starts
    out[:, _DUR] = lengths / fps
    out[lengths <= 0, _DUR] = 0.0
    lo = int(starts.min())
    hi = int(stops.max())
    if hi == lo:
        return out
    ball = np.asarray(ball_all[lo:hi], dtype=np.float64)
    players = np.asarray(players_all[lo:hi], dtype=np.float64)
    local_starts = starts - lo
    local_stops = stops - lo
    path, disp, share = _kinematics(ball, local_starts, local_stops)
    out[:, _PATH] = path
    out[:, _DISP] = disp
    out[:, _SEEN] = share
    for team in np.unique(teams):
        rows = np.flatnonzero(teams == team)
        own = slot_team == team
        dist_own = _slot_distances(ball, players, own)
        dist_opp = _slot_distances(ball, players, ~own)
        out[rows, _COUNT] = _radius_counts(dist_own, local_starts[rows], local_stops[rows])
        out[rows, _SEP] = _contact_separation(ball, dist_own, local_starts[rows], local_stops[rows])
        out[rows, _OWN] = _window_medians(
            _per_frame_min(dist_own), local_starts[rows], local_stops[rows]
        )
        out[rows, _OPP] = _window_medians(
            _per_frame_min(dist_opp), local_starts[rows], local_stops[rows]
        )
    return out


def span_features(tracking: Tracking, a: int, b: int, team: int) -> np.ndarray:
    """``(11,)`` float64 features of frames ``[a, b)`` for ``team``. See the module docstring."""
    bounds = np.array([[int(a), int(b)]], dtype=np.int64)
    teams = np.array([int(team)], dtype=np.int64)
    return span_feature_matrix(tracking, bounds, teams)[0]
