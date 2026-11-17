"""The broadcast recording's ATD placed on the tactical timeline, and its ball used to fill the
tactical ball's gaps (E58, #86).

The organisers ship a second automatic tracking of BAR-ATH, from the TV feed, on its own timeline
with cuts, replays and close-ups and no mapping (DATA.md: "mapping it back is your problem"). The
broadcast camera follows the ball, so on live frames with most players in view it sees the ball
more often than the tactical one does.

Alignment, from the players alone (no ball, no labels, no tags):
- `frame_costs`: for a broadcast frame and candidate tactical frames, the median over the
  broadcast's seen players of the distance to the nearest same-team tactical player, capped at
  10 m.
- `align`: anchors every `anchor_step` broadcast frames with at least `anchor_players` players are
  placed by a local search around the current offset, else by a search in a window around the
  last placement, then over the whole match. A placement needs a cost of at most `anchor_cost_m`
  and, for a search, a margin of `anchor_margin_m` over every tactical frame more than 2 s away.
  Every broadcast frame with at least `min_players` players then takes the better of the offsets
  of the anchors either side (at most one cut between them), if that cost is at most `frame_cost_m`.
  A tactical frame claimed twice keeps the cheaper broadcast frame.

Ball fill (`fill_gaps`): inside each tactical gap (a run of frames with no tactical ball), the
mapped broadcast ball is used only when it agrees with the tactical ball over the `edge_frames`
seen frames on each side that have both (median distance at most `max_edge_m`, at least one side
present). The edge differences are then interpolated across the gap and added, which removes the
two cameras' systematic disagreement (a ball in the air projects to different pitch points).

Where it fails: frames with few players in view (close-ups) cannot be placed; a replay that
mimics live play could be placed at the wrong time if its players happen to fit (the margin rule
guards this); the broadcast ball shares the tactical ball's smoothing (both are Metrica's
pipeline). Signals: positions and team flags of both ATDs; never labels, tags or video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import MatchMeta, Period, Tracking


def load_broadcast_tracking() -> Tracking:
    """The broadcast ATD as one period over its own frames (pitch metres, as the tactical one)."""
    from .. import official
    from ..io import atd
    from ..paths import data_dir

    paths = official.games.asset_paths(official.GAME, data_dir=str(data_dir()), source="broadcast")
    n = official.games.source_frames("broadcast")
    meta = MatchMeta(
        match_id=f"{official.GAME}-broadcast",
        source="bar-ath",
        n_frames=n,
        fps=official.FPS,
        pitch_length=float(official.PITCH[0]),
        pitch_width=float(official.PITCH[1]),
        periods=(Period(1, 0, n),),
    ).validate()
    return atd.load_tracking(paths["atd_meta"], paths["atd_raw"], meta, cache_path=None)


def frame_costs(
    p_xy: np.ndarray, p_team: np.ndarray, q_xy: np.ndarray, q_team: np.ndarray, cap_m: float = 10.0
) -> np.ndarray:
    """(n,) cost of n candidate tactical frames `q_xy` (n, S, 2) for one broadcast frame's seen
    players `p_xy` (m, 2) with teams `p_team` (m,)."""
    parts = []
    for team in (0, 1):
        p = p_xy[p_team == team]
        q = q_xy[:, q_team == team]
        if not len(p) or not q.shape[1]:
            continue
        d = np.sqrt(((q[:, None, :, :] - p[None, :, None, :]) ** 2).sum(-1))
        parts.append(np.where(np.isnan(d), np.inf, d).min(axis=2))
    if not parts:
        return np.full(len(q_xy), cap_m)
    return np.median(np.minimum(np.concatenate(parts, axis=1), cap_m), axis=1)


def pair_costs(
    bro: Tracking, tac: Tracking, b_idx: np.ndarray, t_idx: np.ndarray, cap_m: float = 10.0
) -> np.ndarray:
    """(n,) cost of each (broadcast frame, tactical frame) pair; inf where a frame is out of range
    or the broadcast frame has no player."""
    out = np.full(len(b_idx), np.inf)
    ok = (t_idx >= 0) & (t_idx < tac.n_frames)
    same = bro.slot_team[:, None] == tac.slot_team[None, :]
    for lo in range(0, len(b_idx), 4096):
        sel = np.flatnonzero(ok[lo : lo + 4096]) + lo
        if not len(sel):
            continue
        p = bro.player_xy[b_idx[sel]].astype(np.float32)  # (n, Sb, 2)
        q = tac.player_xy[t_idx[sel]].astype(np.float32)  # (n, St, 2)
        d = np.sqrt(((p[:, :, None, :] - q[:, None, :, :]) ** 2).sum(-1))  # (n, Sb, St)
        d = np.where(same[None] & np.isfinite(d), d, np.inf).min(axis=2)
        d = np.where(np.isfinite(p[..., 0]), np.minimum(d, cap_m), np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(d, axis=1)
        out[sel] = np.where(np.isfinite(med), med, np.inf)
    return out


@dataclass(frozen=True)
class Alignment:
    t_of_b: np.ndarray  # (n_broadcast,) int64, -1 where the frame is not placed
    cost: np.ndarray  # (n_broadcast,) float64, inf where not placed
    anchors: int  # anchor frames tried
    placed_anchors: int
    full_searches: int


def _search(bro, tac, b, candidates, cap_m):
    seen = np.isfinite(bro.player_xy[b, :, 0])
    c = frame_costs(
        bro.player_xy[b, seen].astype(np.float64),
        bro.slot_team[seen],
        tac.player_xy[candidates].astype(np.float64),
        tac.slot_team,
        cap_m,
    )
    i = int(np.argmin(c))
    return int(candidates[i]), float(c[i]), c


def align(
    tac: Tracking,
    bro: Tracking,
    *,
    anchor_step: int = 25,
    anchor_players: int = 10,
    anchor_cost_m: float = 0.8,
    anchor_margin_m: float = 1.0,
    window_back_s: float = 180.0,
    window_ahead_s: float = 420.0,
    search_stride: int = 10,
    min_players: int = 6,
    frame_cost_m: float = 1.0,
) -> Alignment:
    """See the module docstring. Deterministic; no random numbers."""
    fps = tac.meta.fps
    n_b, n_t = bro.n_frames, tac.n_frames
    counts = np.isfinite(bro.player_xy[..., 0]).sum(axis=1)
    anchor_frames = [b for b in range(0, n_b, anchor_step) if counts[b] >= anchor_players]
    placed: list[tuple[int, int]] = []  # (b, offset)
    offset = None
    last_t = None
    full = 0
    for b in anchor_frames:
        if offset is not None:
            local = np.arange(b + offset - 3, b + offset + 4)
            local = local[(local >= 0) & (local < n_t)]
            if len(local):
                t, c, _ = _search(bro, tac, b, local, 10.0)
                if c <= anchor_cost_m:
                    offset, last_t = t - b, t
                    placed.append((b, offset))
                    continue
        windows = []
        if last_t is not None:
            lo = max(0, last_t - int(window_back_s * fps))
            hi = min(n_t, last_t + int(window_ahead_s * fps))
            windows.append(np.arange(lo, hi, search_stride))
        windows.append(np.arange(0, n_t, search_stride))
        for grid in windows:
            full += 1
            t, c, costs = _search(bro, tac, b, grid, 10.0)
            far = np.abs(grid - t) > 2 * fps
            margin = float(costs[far].min() - c) if far.any() else 0.0
            if c > anchor_cost_m + 0.5 or margin < anchor_margin_m:
                continue
            fine = np.arange(max(0, t - search_stride), min(n_t, t + search_stride + 1))
            t, c, _ = _search(bro, tac, b, fine, 10.0)
            if c <= anchor_cost_m:
                offset, last_t = t - b, t
                placed.append((b, offset))
                break
    t_of_b = np.full(n_b, -1, dtype=np.int64)
    cost = np.full(n_b, np.inf)
    if placed:
        pb = np.array([p[0] for p in placed])
        po = np.array([p[1] for p in placed])
        frames = np.flatnonzero(counts >= min_players)
        right = np.clip(np.searchsorted(pb, frames), 0, len(pb) - 1)
        left = np.clip(right - 1, 0, len(pb) - 1)
        best_c = np.full(len(frames), np.inf)
        best_t = np.full(len(frames), -1, dtype=np.int64)
        for o in (po[left], po[right]):
            t = frames + o
            c = pair_costs(bro, tac, frames, t)
            better = c < best_c
            best_c[better], best_t[better] = c[better], t[better]
        keep = best_c <= frame_cost_m
        t_of_b[frames[keep]] = best_t[keep]
        cost[frames[keep]] = best_c[keep]
        # a tactical frame claimed twice keeps the cheaper broadcast frame
        order = np.lexsort((cost, t_of_b))
        claimed = t_of_b[order]
        dup = np.zeros(len(order), dtype=bool)
        dup[1:] = (claimed[1:] == claimed[:-1]) & (claimed[1:] >= 0)
        t_of_b[order[dup]] = -1
        cost[order[dup]] = np.inf
    return Alignment(t_of_b, cost, len(anchor_frames), len(placed), full)


def mapped_ball(tac: Tracking, bro: Tracking, alignment: Alignment) -> np.ndarray:
    """(T_tactical, 2) float64: the broadcast ball on the tactical frames it was placed on."""
    out = np.full((tac.n_frames, 2), np.nan)
    b = np.flatnonzero(alignment.t_of_b >= 0)
    out[alignment.t_of_b[b]] = bro.ball_xy[b]
    return out


def fill_gaps(
    ball: np.ndarray,
    candidate: np.ndarray,
    periods: tuple[Period, ...],
    *,
    edge_frames: int = 5,
    max_edge_m: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """`ball` (T, 2) with its gaps filled from `candidate` (T, 2) as the module docstring says.
    Returns the filled ball and counts (gaps, filled gaps, filled frames, gap frames)."""
    out = ball.astype(np.float64).copy()
    seen = np.isfinite(ball).all(axis=1)
    have = np.isfinite(candidate).all(axis=1)
    stats = {"gaps": 0, "gaps_filled": 0, "frames_filled": 0, "gap_frames": 0, "gaps_rejected": 0}
    for period in periods:
        s = seen[period.start : period.stop]
        edges = np.diff(np.concatenate(([1], s.astype(np.int8), [1])))
        for a, z in zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1), strict=True):
            a, z = a + period.start, z + period.start
            stats["gaps"] += 1
            stats["gap_frames"] += z - a
            inside = have[a:z]
            if not inside.any():
                continue
            diffs = []
            for lo, hi in (
                (max(period.start, a - edge_frames), a),
                (z, min(period.stop, z + edge_frames)),
            ):
                both = seen[lo:hi] & have[lo:hi]
                if both.any():
                    d = ball[lo:hi][both] - candidate[lo:hi][both]
                    diffs.append(np.median(d, axis=0))
                else:
                    diffs.append(None)
            present = [d for d in diffs if d is not None]
            if not present or any(np.hypot(*d) > max_edge_m for d in present):
                stats["gaps_rejected"] += int(bool(present))
                continue
            before = diffs[0] if diffs[0] is not None else diffs[1]
            after = diffs[1] if diffs[1] is not None else diffs[0]
            w = (np.arange(z - a) + 0.5) / (z - a)
            corr = before[None, :] * (1 - w[:, None]) + after[None, :] * w[:, None]
            fill = candidate[a:z] + corr
            out[a:z][inside] = fill[inside]
            stats["gaps_filled"] += 1
            stats["frames_filled"] += int(inside.sum())
    return out, stats
