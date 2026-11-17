"""Fill tactical ball gaps from precomputed video ball candidates (#86, off unless asked for).

The detector lives outside the submission build (`tools/ball_detector/`, torch). It writes a
candidates file: for every video frame, up to a few image-space ball detections projected to pitch
metres with the competition homography. This module only reads that file (numpy only) and uses it
where a candidate chain is trustworthy. Nothing here looks at video or labels.

File format (`.npz`, written by `tools/ball_detector/infer.py`, all arrays 1-D, same length N):
    frame  int32    video frame (tactical timeline)
    x_m    float32  pitch metres, centred (same convention as `Tracking.ball_xy`)
    y_m    float32
    conf   float32  detector confidence in [0, 1]
    flags  uint8    bit 0: cannot be on the ground plane (off pitch, behind the horizon, bad
                    homography), such rows are never used.
Rows of one frame are ranked by confidence and only the best `max_per_frame` are kept.

Fill rule (`fill_gaps`). A gap is a run of frames without a tactical ball, inside one period,
with a seen ball on both sides. It is filled only if a *chain* of candidates bridges it:
- every chain node has `conf >= min_conf`;
- consecutive nodes (and the two seen edge balls) are at most `max_skip_s` apart in time, and
  at most `max_speed_ms * dt + slack_m` apart in space (45 m/s: the champion's cleaning bound);
- among all such chains the one with the largest summed confidence is taken (deterministic ties);
- the gap frames between nodes are linearly interpolated.
Gaps longer than `max_gap_s` are never touched (E91: long gaps are occluded or airborne balls).
No chain, no fill: a gap stays a gap, so the missing-ball cues keep their meaning there.

Reproducibility: the build reads a file derived from the competition video, which a clean
checkout cannot regenerate in the 5-minute budget. `load` therefore requires the file's sha256
(a `sha256` parameter in the pipeline step; a hash of the array contents, not of the zip bytes) and fails loudly on a mismatch or when it is absent.
The file itself is never committed (`*.npz` is git-ignored; it is derived from the video).

Where it fails: a ball that is airborne projects onto the ground plane at the wrong place, so a
confident, continuous but wrong chain is possible; a ball in a crowd that the detector confuses
with a head; a restart where the ball is visible but the cue "no ball = dead" is relied on.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from aaaag.contracts import MatchBundle, Period

MAX_BALL_SPEED = 45.0  # m/s, as the champion's cues (aaaag.evidence.ballstate.cues)
FLAG_OFF_GROUND = 1


def file_sha256(path: str | Path) -> str:
    """Content hash of the arrays (names, dtypes, bytes), not of the zip: `np.savez` stamps the
    wall-clock time into the archive, so the file bytes differ between two writes of one table."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for name in sorted(z.files):
            digest.update(name.encode())
            digest.update(str(z[name].dtype).encode())
            digest.update(np.ascontiguousarray(z[name]).tobytes())
    return digest.hexdigest()


def load(path: str | Path, n_frames: int, *, sha256: str, max_per_frame: int = 3):
    """(xy (T, K, 2) float64, conf (T, K) float64), NaN / 0 where there is no candidate.

    Rows flagged off the ground plane are dropped. Raises ValueError if the file's sha256 differs
    from `sha256` (the reproducibility pin), a column is missing, or a frame is out of range."""
    actual = file_sha256(path)
    if actual != sha256:
        raise ValueError(f"{path}: sha256 {actual} differs from the pinned {sha256}")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ("frame", "x_m", "y_m", "conf", "flags") if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        frame, x, y, conf, flags = (z[k] for k in ("frame", "x_m", "y_m", "conf", "flags"))
    if not len({len(a) for a in (frame, x, y, conf, flags)}) == 1:
        raise ValueError(f"{path}: arrays differ in length")
    if len(frame) and (frame.min() < 0 or frame.max() >= n_frames):
        raise ValueError(f"{path}: frames outside 0..{n_frames - 1}")
    keep = ((flags & FLAG_OFF_GROUND) == 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(conf)
    frame, x, y, conf = frame[keep], x[keep], y[keep], conf[keep]
    order = np.lexsort((-conf, frame))  # frame ascending, confidence descending
    frame, x, y, conf = frame[order], x[order], y[order], conf[order]
    first = np.searchsorted(frame, frame, side="left")
    rank = np.arange(len(frame)) - first
    use = rank < max_per_frame
    xy = np.full((n_frames, max_per_frame, 2), np.nan)
    cf = np.zeros((n_frames, max_per_frame))
    xy[frame[use], rank[use], 0] = x[use]
    xy[frame[use], rank[use], 1] = y[use]
    cf[frame[use], rank[use]] = conf[use]
    return xy, cf


def _best_chain(left, right, a, z, cxy, ccf, fps, min_conf, vmax, slack_m, max_skip):
    """Frames `a-1` (seen, `left`) and `z` (seen, `right`) bracket the gap a..z-1.
    Returns [(frame, x, y), ...] of the best chain including both anchors, or None."""
    ff, kk = np.nonzero(ccf[a:z] >= min_conf)
    if not len(ff):
        return None
    nodes_f = ff + a  # sorted ascending by frame (nonzero is row-major)
    nodes_p = cxy[nodes_f, kk]
    nodes_c = ccf[nodes_f, kk]
    n = len(nodes_f)

    def ok(dt_frames, dist):
        return (dt_frames <= max_skip + 1) & (dist <= vmax * dt_frames / fps + slack_m)

    best = np.full(n, -np.inf)
    prev = np.full(n, -1, dtype=np.int64)  # -1: the left anchor
    for i in range(n):
        score = -np.inf
        arg = -1
        if ok(nodes_f[i] - (a - 1), np.hypot(*(nodes_p[i] - left))):
            score = nodes_c[i]
        lo = np.searchsorted(nodes_f, nodes_f[i] - max_skip - 1, side="left")
        hi = np.searchsorted(nodes_f, nodes_f[i], side="left")
        if hi > lo:
            d = np.hypot(*(nodes_p[lo:hi] - nodes_p[i]).T)
            good = ok(nodes_f[i] - nodes_f[lo:hi], d) & np.isfinite(best[lo:hi])
            if good.any():
                cand = np.where(good, best[lo:hi], -np.inf)
                j = int(np.argmax(cand))
                if cand[j] + nodes_c[i] > score:
                    score, arg = cand[j] + nodes_c[i], lo + j
        best[i], prev[i] = score, arg
    end = ok(z - nodes_f, np.hypot(*(nodes_p - right).T)) & np.isfinite(best)
    if not end.any():
        return None
    i = int(np.argmax(np.where(end, best, -np.inf)))
    chain = []
    while i >= 0:
        chain.append((int(nodes_f[i]), *nodes_p[i]))
        i = int(prev[i])
    chain.append((a - 1, *left))
    chain.reverse()
    chain.append((z, *right))
    return chain


def fill_gaps(
    ball: np.ndarray,
    cand_xy: np.ndarray,
    cand_conf: np.ndarray,
    periods: tuple[Period, ...],
    fps: float,
    *,
    min_conf: float = 0.5,
    max_speed_ms: float = MAX_BALL_SPEED,
    slack_m: float = 0.5,
    max_skip_s: float = 0.4,
    max_gap_s: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """`ball` (T, 2) with the gaps bridged by a candidate chain filled. Returns the filled ball
    (float64) and counts: gaps, gaps_filled, frames_filled, gap_frames."""
    out = ball.astype(np.float64).copy()
    seen = np.isfinite(ball).all(axis=1)
    max_skip = max(0, int(round(max_skip_s * fps)))
    stats = {"gaps": 0, "gaps_filled": 0, "frames_filled": 0, "gap_frames": 0}
    for period in periods:
        s = seen[period.start : period.stop]
        edges = np.diff(np.concatenate(([1], s.astype(np.int8), [1])))
        for a, z in zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1), strict=True):
            a, z = int(a + period.start), int(z + period.start)
            stats["gaps"] += 1
            stats["gap_frames"] += z - a
            if a == period.start or z == period.stop or z - a > max_gap_s * fps:
                continue  # an open edge or a long gap: never touched
            chain = _best_chain(
                ball[a - 1].astype(np.float64),
                ball[z].astype(np.float64),
                a,
                z,
                cand_xy,
                cand_conf,
                fps,
                min_conf,
                max_speed_ms,
                slack_m,
                max_skip,
            )
            if chain is None:
                continue
            frames = np.array([c[0] for c in chain], dtype=np.float64)
            pts = np.array([[c[1], c[2]] for c in chain])
            t = np.arange(a, z, dtype=np.float64)
            out[a:z, 0] = np.interp(t, frames, pts[:, 0])
            out[a:z, 1] = np.interp(t, frames, pts[:, 1])
            stats["gaps_filled"] += 1
            stats["frames_filled"] += z - a
    return out, stats


def filled_bundle(bundle: MatchBundle, *, path: str, sha256: str, **params) -> MatchBundle:
    """The bundle with its tracking ball gaps filled from the pinned candidates file.

    Params other than `path`, `sha256` and `max_per_frame` go to `fill_gaps`. The `ball_extra`
    channels are NaN on filled frames (they describe the delivered ball only)."""
    tracking = bundle.tracking
    n_frames = tracking.n_frames
    max_per_frame = int(params.pop("max_per_frame", 3))
    xy, conf = load(path, n_frames, sha256=sha256, max_per_frame=max_per_frame)
    filled, stats = fill_gaps(
        tracking.ball_xy, xy, conf, bundle.meta.periods, bundle.meta.fps, **params
    )
    if not stats["gaps_filled"]:
        return bundle
    changed = np.isfinite(filled).all(axis=1) & ~tracking.ball_seen()
    extra = {}
    for name, channel in tracking.ball_extra.items():
        channel = channel.copy()
        channel[changed] = np.nan
        extra[name] = channel
    new = dataclasses.replace(
        tracking, ball_xy=filled.astype(np.float32), ball_extra=extra, _fingerprint=None
    ).validate()
    return dataclasses.replace(bundle, tracking=new)
