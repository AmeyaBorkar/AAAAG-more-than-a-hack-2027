"""Ball-mask augmentation that breaks the missing-ball shortcut.

On the provider of the competition tracking a missing ball mostly means a dead ball, and degrade-v5
reproduces that: it hides the ball through every stoppage. A network trained only on that learns
"ball missing => dead" and "ball hidden => no information", and it breaks if the competition
blanking differs (docs/ml/README.md, the robustness scout notes). So in half of the training
batches the ball visibility is drawn independently of the ball state: the match's complete
provider ball is hidden by a renewal process (run lengths from 0.5 s to 20 s, hidden share 5% to
75%, some blocks left fully visible) that knows nothing about the labels.

Signals: only the frame count and the period layout. Fails where a real blanking is neither: a
tracker that loses the ball for the same reason it loses the players.

`alive_gap_mask` (E158, `train.alive_gap_target`) uses the same renewal run lengths on the coupled
variant: label-free gaps on alive frames, until the alive missing share reaches a target that is
measured on the real provider's raw tracking (Metrica, about 0.12) instead of degrade-v5's 0.04.

Optional training augmentations live here too (`AugmentMix`): a formation warp that leaves the
players near the clean ball alone, a bias and lag on the ball, a Gaussian smooth of long seen
ball runs, and extra ghost or duplicate slots. Each is off unless its probability is positive,
and a probability of zero draws no random numbers, so a run that does not set `train.augment`
keeps today's tensors and today's random stream. They see the degraded players and the clean
ball; the token builder turns positions into velocities afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..contracts import AWAY, HOME, A, H, MatchMeta, Period, Tracking

RENEWAL_BLOCK_S = 120.0  # each block of a period draws its own mean hidden run
RENEWAL_RUN_S = (0.5, 20.0)  # log-uniform range of that mean, in seconds


def hidden_ball_mask(
    n_frames: int,
    periods: tuple[Period, ...],
    fps: int,
    rng: np.random.Generator,
    *,
    block_s: float = RENEWAL_BLOCK_S,
    clear_prob: float = 0.2,
    share_range: tuple[float, float] = (0.05, 0.75),
    run_s_range: tuple[float, float] = RENEWAL_RUN_S,
) -> np.ndarray:
    """(n_frames,) bool, True where the ball is hidden. Independent of every label.

    Each `block_s` block of a period gets its own draw: with probability `clear_prob` nothing is
    hidden; otherwise the hidden share is uniform in `share_range` and the mean hidden run is
    log-uniform in `run_s_range` (seconds), with exponential runs of hidden and of seen ball
    alternating.
    """
    hidden = np.zeros(n_frames, dtype=bool)
    block = max(1, int(round(block_s * fps)))
    log_low, log_high = np.log(run_s_range[0]), np.log(run_s_range[1])
    for period in periods:
        for lo in range(period.start, period.stop, block):
            hi = min(period.stop, lo + block)
            if rng.random() < clear_prob:
                continue
            share = float(rng.uniform(*share_range))
            mean_hidden = float(np.exp(rng.uniform(log_low, log_high))) * fps
            mean_seen = mean_hidden * (1.0 - share) / share
            hidden[lo:hi] = _alternating(hi - lo, mean_hidden, mean_seen, share, rng)
    return hidden


def _alternating(
    n: int, mean_hidden: float, mean_seen: float, share: float, rng: np.random.Generator
) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    hidden = bool(rng.random() < share)
    t = 0
    while t < n:
        length = max(1, int(round(rng.exponential(mean_hidden if hidden else mean_seen))))
        if hidden:
            out[t : t + length] = True
        t += length
        hidden = not hidden
    return out


def alive_gap_mask(
    missing: np.ndarray,
    labels: np.ndarray,
    periods: tuple[Period, ...],
    fps: int,
    rng: np.random.Generator,
    target: float,
    *,
    block_s: float = RENEWAL_BLOCK_S,
    run_s_range: tuple[float, float] = RENEWAL_RUN_S,
) -> np.ndarray:
    """(T,) bool: alive frames to hide so that the alive missing share reaches `target` (E158).

    `missing` is True where the ball is already unseen. Let s0 be the missing share over the alive
    frames (label H or A, inside a period). A renewal process that ignores the labels hides each
    frame with probability q = (target - s0) / (1 - s0). Its run lengths come from the decoupled
    renewal of `hidden_ball_mask`: each `block_s` block of a period draws a mean hidden run,
    log-uniform in `run_s_range` seconds. Exponential hidden and seen runs then alternate with a
    hidden share of q, and each block starts hidden with probability q, so every frame is hidden
    with probability q.
    The mask keeps that process on the alive frames only. Dead, X and unplayed frames are never
    set, so P(missing | alive) is s0 + (1 - s0) q = target in expectation and P(missing | dead)
    does not move. When s0 >= target, or there is no alive frame, nothing is hidden and nothing is
    drawn. Draws two numbers per block plus the run lengths, in chunks, with no loop over frames.
    Signals: the labels, only to find the alive frames and to measure s0, and the period layout.
    Fails where a real tracker's alive gaps cluster (near the goal, in crowds, high balls): these
    gaps are uniform in time.
    """
    if not 0.0 <= target < 1.0:
        raise ValueError(f"alive gap target must be in [0, 1), got {target!r}")
    labels = np.asarray(labels)
    n = len(labels)
    missing = np.asarray(missing, dtype=bool)
    if missing.shape != (n,):
        raise ValueError(f"missing {missing.shape} != labels {(n,)}")
    in_period = np.zeros(n, dtype=bool)
    for period in periods:
        in_period[period.start : period.stop] = True
    alive = in_period & np.isin(labels, (H, A))
    out = np.zeros(n, dtype=bool)
    if not alive.any():
        return out
    share = float(missing[alive].mean())
    if share >= target:
        return out
    q = (target - share) / (1.0 - share)
    block = max(1, int(round(block_s * fps)))
    log_low, log_high = np.log(run_s_range[0]), np.log(run_s_range[1])
    for period in periods:
        for lo in range(period.start, period.stop, block):
            hi = min(period.stop, lo + block)
            mean_hidden = float(np.exp(rng.uniform(log_low, log_high))) * fps
            mean_seen = mean_hidden * (1.0 - q) / q
            out[lo:hi] = _alternating_runs(hi - lo, mean_hidden, mean_seen, q, rng)
    return out & alive


def _alternating_runs(
    n: int, mean_hidden: float, mean_seen: float, share: float, rng: np.random.Generator
) -> np.ndarray:
    """(n,) bool: exponential hidden and seen runs (at least 1 frame) alternating, starting hidden
    with probability `share`. `_alternating` draws them one at a time; this draws them in chunks of
    an even length, so the alternation carries over from one chunk to the next."""
    hidden_first = bool(rng.random() < share)
    first, second = (mean_hidden, mean_seen) if hidden_first else (mean_seen, mean_hidden)
    scale = np.tile([first, second], int(np.ceil(n / (mean_hidden + mean_seen))) + 2)
    chunks: list[np.ndarray] = []
    total = 0
    while total < n:
        runs = np.maximum(1, np.rint(rng.standard_exponential(len(scale)) * scale))
        chunks.append(runs.astype(np.int64))
        total += int(chunks[-1].sum())
    runs = np.concatenate(chunks)
    ends = np.cumsum(runs)
    starts = ends - runs
    hidden = ((np.arange(len(runs)) % 2) == 0) == hidden_first
    take = hidden & (starts < n)
    delta = np.zeros(n + 1, dtype=np.int32)
    np.add.at(delta, starts[take], 1)
    np.add.at(delta, np.minimum(ends[take], n), -1)
    return np.cumsum(delta[:-1]) > 0


def with_ball(tracking: Tracking, ball_xy: np.ndarray) -> Tracking:
    """A copy of `tracking` with another ball track (players, teams and meta untouched)."""
    return replace(tracking, ball_xy=ball_xy.astype(np.float32), ball_extra={}, _fingerprint=None)


def hide_ball(tracking: Tracking, hidden: np.ndarray) -> Tracking:
    """`tracking` with the ball set to NaN where `hidden`."""
    ball = tracking.ball_xy.copy()
    ball[hidden] = np.nan
    return with_ball(tracking, ball)


# -- optional training augmentations --------------------------------------------------------------
#
# All four are applied to one degraded draw, before either tokenisation. Probabilities of zero
# draw nothing. The warp and the ghosts edit players only; the ball offset and the ball smooth
# edit both ball tracks the two variants read (the degraded ball, and the clean provider ball)
# with one shared draw each.

NEAR_BALL_M = 6.0  # inside this, team_style does not move a player
WARP_RAMP_M = 6.0  # smoothstep width; the warp is full at NEAR_BALL_M + WARP_RAMP_M (12 m)
GHOST_STEP_SD_M = 0.15  # random-walk step at GHOST_REF_FPS
GHOST_REF_FPS = 25.0
VISIBLE_RUN_S = (2.0, 20.0)  # log-uniform mean of an exponential visible run
HIDDEN_RUN_S = (5.0, 60.0)
DUPLICATE_DISTANCE_M = (0.2, 0.6)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, argument clipped to [0, 1]."""
    x = np.clip(t, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _check_prob(name: str, p: float) -> None:
    if not 0.0 <= p <= 1.0 or not np.isfinite(p):
        raise ValueError(f"{name} must be in [0, 1], got {p}")


@dataclass(frozen=True)
class TeamStyle:
    """Formation warp about each team's period centroid. `p` is the chance a draw applies it.

    `stretch` is the half-width of the axis scale U(1-stretch, 1+stretch); `shift_m` is the
    half-width of the translation, in metres, in the stadium frame.
    """

    p: float = 0.0
    stretch: float = 0.15
    shift_m: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", float(self.p))
        object.__setattr__(self, "stretch", float(self.stretch))
        object.__setattr__(self, "shift_m", float(self.shift_m))
        _check_prob("team_style.p", self.p)
        if self.stretch < 0.0 or self.shift_m < 0.0:
            raise ValueError(f"team_style stretch and shift_m must be >= 0, got {self}")


@dataclass(frozen=True)
class BallOffset:
    """One bias, one lag and optional white noise on the ball, shared by both variants."""

    p: float = 0.0
    bias_m: float = 2.0
    lag_frames: int = 4
    noise_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", float(self.p))
        object.__setattr__(self, "bias_m", float(self.bias_m))
        object.__setattr__(self, "lag_frames", int(self.lag_frames))
        object.__setattr__(self, "noise_m", float(self.noise_m))
        _check_prob("ball_offset.p", self.p)
        if self.bias_m < 0.0 or self.lag_frames < 0 or self.noise_m < 0.0:
            raise ValueError(f"ball_offset magnitudes must be >= 0, got {self}")


@dataclass(frozen=True)
class BallSmooth:
    """Gaussian smooth of each long seen ball run, shared by both variants.

    `p` is the chance a draw applies it. `sigma_frames` are the Gaussian sigmas, in frames;
    `sigma_p` is their probabilities, or None for a uniform draw. A seen run is smoothed only
    when it is longer than ``4 * sigma``, and only inside one period.
    """

    p: float = 0.0
    sigma_frames: tuple[float, ...] = (2.0, 4.0, 6.0)
    sigma_p: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", float(self.p))
        object.__setattr__(self, "sigma_frames", tuple(float(s) for s in self.sigma_frames))
        if self.sigma_p is not None:
            object.__setattr__(self, "sigma_p", tuple(float(s) for s in self.sigma_p))
        _check_prob("ball_smooth.p", self.p)
        if not self.sigma_frames or any(
            (not np.isfinite(s)) or s <= 0.0 for s in self.sigma_frames
        ):
            raise ValueError(
                f"ball_smooth.sigma_frames must be non-empty and all > 0, got {self.sigma_frames}"
            )
        if self.sigma_p is None:
            return
        if len(self.sigma_p) != len(self.sigma_frames):
            raise ValueError(
                "ball_smooth.sigma_p must have length "
                f"{len(self.sigma_frames)}, got {len(self.sigma_p)}"
            )
        if any((not np.isfinite(s)) or s < 0.0 for s in self.sigma_p):
            raise ValueError(f"ball_smooth.sigma_p must be non-negative, got {self.sigma_p}")
        total = float(sum(self.sigma_p))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ball_smooth.sigma_p must sum to 1, got {total}")


@dataclass(frozen=True)
class GhostSlots:
    """Extra slots: up to `max_ghost_slots` random-walk ghosts, plus per-frame duplicates."""

    p: float = 0.0
    max_ghost_slots: int = 4
    duplicate_p: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", float(self.p))
        object.__setattr__(self, "max_ghost_slots", int(self.max_ghost_slots))
        object.__setattr__(self, "duplicate_p", float(self.duplicate_p))
        _check_prob("ghosts.p", self.p)
        _check_prob("ghosts.duplicate_p", self.duplicate_p)
        if self.max_ghost_slots < 0:
            raise ValueError(f"max_ghost_slots must be >= 0, got {self.max_ghost_slots}")


def _block(cls: type[Any], raw: Any, key: str) -> Any:
    if not raw:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"augment.{key} must be a mapping")
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown augment.{key} keys: {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class AugmentMix:
    """The four optional warps. Missing blocks and a missing `augment` key are the defaults,
    every probability zero."""

    team_style: TeamStyle = field(default_factory=TeamStyle)
    ball_offset: BallOffset = field(default_factory=BallOffset)
    ball_smooth: BallSmooth = field(default_factory=BallSmooth)
    ghosts: GhostSlots = field(default_factory=GhostSlots)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AugmentMix:
        raw = dict(raw or {})
        unknown = set(raw) - {"team_style", "ball_offset", "ball_smooth", "ghosts"}
        if unknown:
            raise ValueError(f"unknown augment keys: {sorted(unknown)}")
        return cls(
            team_style=_block(TeamStyle, raw.get("team_style"), "team_style"),
            ball_offset=_block(BallOffset, raw.get("ball_offset"), "ball_offset"),
            ball_smooth=_block(BallSmooth, raw.get("ball_smooth"), "ball_smooth"),
            ghosts=_block(GhostSlots, raw.get("ghosts"), "ghosts"),
        )


def _ball_reference(ball_xy: np.ndarray, periods: tuple[Period, ...]) -> np.ndarray:
    """(T, 2) float64. Inside each period, NaN frames take the last finite ball in that period.

    The fill restarts every period: carrying the first half's last ball into the second would
    shield the wrong patch of grass after the teams change ends. Frames before the first finite
    ball of the period stay NaN (those players are not warped). Outside the periods the original
    samples are kept and are not read by the warp.
    """
    ref = np.array(ball_xy, dtype=np.float64, copy=True)
    for period in periods:
        seg = ref[period.start : period.stop]
        valid = np.isfinite(seg).all(axis=1)
        if not np.any(valid):
            continue
        index = np.arange(seg.shape[0])
        last = np.maximum.accumulate(np.where(valid, index, -1))
        known = last >= 0
        filled = seg[np.clip(last, 0, seg.shape[0] - 1)]
        filled[~known] = np.nan
        seg[:] = filled
    return ref


def warp_player_xy(
    player_xy: np.ndarray,
    slot_team: np.ndarray,
    clean_ball: np.ndarray,
    meta: MatchMeta,
    scales: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """(T, S, 2) float32. Per team, p -> p + w(d) * (c + S(p - c) + shift - p).

    `c` is the team's mean observed position in the period. `d` is the distance to the clean
    ball (forward-filled inside the period). `w` is 0 within 6 m and 1 beyond 12 m, so the
    geometry next to the ball is bitwise unchanged. `scales` and `shifts` are (2, 2), team then
    axis, in the stadium frame. Unplayed frames are left as they were.
    """
    out = player_xy.astype(np.float32, copy=True)
    ref = _ball_reference(clean_ball, meta.periods)
    scales = np.asarray(scales, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    for period in meta.periods:
        lo, hi = period.start, period.stop
        ball = ref[lo:hi]
        for team in (HOME, AWAY):
            slots = np.flatnonzero(slot_team == team)
            if slots.size == 0:
                continue
            block = out[lo:hi, slots].astype(np.float64)
            observed = np.isfinite(block).all(axis=-1)
            if not np.any(observed):
                continue
            centre = np.array(
                [block[..., 0][observed].mean(), block[..., 1][observed].mean()],
                dtype=np.float64,
            )
            distance = np.hypot(block[..., 0] - ball[:, None, 0], block[..., 1] - ball[:, None, 1])
            weight = np.where(
                np.isfinite(distance),
                _smoothstep((distance - NEAR_BALL_M) / WARP_RAMP_M),
                0.0,
            )
            move = observed & (weight > 0.0)
            if not np.any(move):
                continue
            target = (block - centre) * scales[team] + centre + shifts[team]
            updated = block + weight[..., None] * (target - block)
            piece = out[lo:hi, slots].copy()
            piece[move] = updated[move].astype(np.float32)
            out[lo:hi, slots] = piece
    return out


def apply_team_style(
    tracking: Tracking, clean_ball: np.ndarray, cfg: TeamStyle, rng: np.random.Generator
) -> Tracking:
    """`tracking` with the formation warp, or `tracking` itself when the draw skips it.

    Signals: degraded player positions, slot teams, the clean ball, the period layout. The ball,
    the teams and the delivered `player_speed` are not touched (tokens recompute velocity from
    positions; the delivered speed is then stale). Fails open on a period with no finite clean
    ball: nobody in that period moves, which keeps an unknown carrier in place.
    """
    if cfg.p <= 0.0 or rng.random() >= cfg.p:
        return tracking
    scales = rng.uniform(1.0 - cfg.stretch, 1.0 + cfg.stretch, size=(2, 2))
    shifts = rng.uniform(-cfg.shift_m, cfg.shift_m, size=(2, 2))
    xy = warp_player_xy(
        tracking.player_xy, tracking.slot_team, clean_ball, tracking.meta, scales, shifts
    )
    return replace(tracking, player_xy=xy, _fingerprint=None)


def offset_ball_track(
    ball_xy: np.ndarray,
    bias: np.ndarray,
    lag: int,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """(T, 2) float32. Shift the track by `lag` frames (positive: the ball arrives late), add
    `bias`, then `noise` where the shifted sample is finite.

    A missing sample stays missing: it is never filled, it only moves with the lag, and the
    `lag` frames that slide off the ends become NaN. `noise` is indexed by the output frame.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    out = np.full(ball.shape, np.nan, dtype=np.float64)
    lag = int(lag)
    if lag == 0:
        out[:] = ball
    elif lag > 0:
        out[lag:] = ball[:-lag]
    else:
        out[:lag] = ball[-lag:]
    seen = np.isfinite(out).all(axis=1)
    if np.any(seen):
        out[seen] += np.asarray(bias, dtype=np.float64)
        if noise is not None:
            out[seen] += np.asarray(noise, dtype=np.float64)[seen]
    return out.astype(np.float32)


def apply_ball_offset(
    tracking: Tracking,
    clean_ball: np.ndarray,
    cfg: BallOffset,
    rng: np.random.Generator,
) -> tuple[Tracking, np.ndarray]:
    """`(tracking, decoupled ball)`. One bias, lag and noise for the degraded ball and the clean one.

    The bias has magnitude U(0, bias_m) and a uniform direction. The lag is an integer in
    [-lag_frames, +lag_frames]. White noise of sd `noise_m` is drawn only when that sd is
    positive. Players are untouched. When the draw skips, the clean ball array is returned as
    given and `rng` is not asked for the parameters (the coin itself is drawn only if `p > 0`).
    """
    if cfg.p <= 0.0 or rng.random() >= cfg.p:
        return tracking, clean_ball
    magnitude = float(rng.uniform(0.0, cfg.bias_m))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    bias = np.array([magnitude * np.cos(theta), magnitude * np.sin(theta)], dtype=np.float64)
    lag = int(rng.integers(-cfg.lag_frames, cfg.lag_frames + 1))
    noise = None
    if cfg.noise_m > 0.0:
        noise = rng.normal(0.0, cfg.noise_m, size=(tracking.n_frames, 2))
    coupled = offset_ball_track(tracking.ball_xy, bias, lag, noise)
    decoupled = offset_ball_track(clean_ball, bias, lag, noise)
    return replace(tracking, ball_xy=coupled, _fingerprint=None), decoupled


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs where the 1-D mask is true."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def smooth_ball_track(ball_xy: np.ndarray, periods: tuple[Period, ...], sigma: float) -> np.ndarray:
    """(T, 2) float32. Gaussian-smooth each seen run longer than ``4 * sigma``, per period.

    Unseen frames, and frames outside the periods, stay as they are and are never filled. A run
    of length at most ``4 * sigma`` is copied unchanged: that is the support of the default
    Gaussian kernel, so a short zigzag is not pulled toward its ends. Runs stop at a NaN and at
    a period boundary. The filter is `gaussian_filter1d(..., mode="nearest")` along time.
    """
    out = np.array(ball_xy, dtype=np.float64, copy=True)
    limit = 4.0 * float(sigma)
    for period in periods:
        seg = out[period.start : period.stop]
        seen = np.isfinite(seg).all(axis=1)
        for start, stop in _mask_runs(seen):
            if stop - start <= limit:
                continue
            seg[start:stop] = gaussian_filter1d(seg[start:stop], sigma, axis=0, mode="nearest")
    return out.astype(np.float32)


def apply_ball_smooth(
    tracking: Tracking,
    clean_ball: np.ndarray,
    cfg: BallSmooth,
    rng: np.random.Generator,
) -> tuple[Tracking, np.ndarray]:
    """`(tracking, decoupled ball)`. One coin and one sigma for the degraded ball and the clean one.

    Each track is smoothed on its own seen runs, so the two masks can differ. Players are
    untouched. When `p <= 0` nothing is drawn and both objects are returned as given. The coin
    is drawn only if `p > 0`, and the sigma only if the coin applies. Fails to round a corner
    that sits in a run of length at most ``4 * sigma``, or that is split by a gap or a period.
    """
    if cfg.p <= 0.0 or rng.random() >= cfg.p:
        return tracking, clean_ball
    n_sigma = len(cfg.sigma_frames)
    if cfg.sigma_p is None:
        sigma = float(cfg.sigma_frames[int(rng.integers(0, n_sigma))])
    else:
        picked = int(rng.choice(n_sigma, p=np.asarray(cfg.sigma_p, dtype=np.float64)))
        sigma = float(cfg.sigma_frames[picked])
    periods = tracking.meta.periods
    coupled = smooth_ball_track(tracking.ball_xy, periods, sigma)
    decoupled = smooth_ball_track(clean_ball, periods, sigma)
    return replace(tracking, ball_xy=coupled, _fingerprint=None), decoupled


def _burst_mask(
    n: int, mean_on: float, mean_off: float, start_on: bool, rng: np.random.Generator
) -> np.ndarray:
    """(n,) bool. Exponential on/off runs. Vectorised over each batch of runs, not over frames
    one at a time: the loop counts runs, and each run writes a slice."""
    out = np.zeros(n, dtype=bool)
    if n <= 0:
        return out
    on = bool(start_on)
    t = 0
    while t < n:
        means = np.empty(512, dtype=np.float64)
        if on:
            means[0::2] = mean_on
            means[1::2] = mean_off
        else:
            means[0::2] = mean_off
            means[1::2] = mean_on
        lengths = np.maximum(1, np.minimum(np.rint(rng.exponential(means)), n).astype(np.int64))
        for length in lengths:
            end = t + int(length)
            if on:
                out[t : min(end, n)] = True
            on = not on
            t = end
            if t >= n:
                break
    return out


def _ghost_visible(n: int, fps: int, rng: np.random.Generator) -> np.ndarray:
    """One ghost's visibility. Means are log-uniform in the run ranges; the phase is stationary."""
    mean_on = float(np.exp(rng.uniform(np.log(VISIBLE_RUN_S[0]), np.log(VISIBLE_RUN_S[1])))) * fps
    mean_off = float(np.exp(rng.uniform(np.log(HIDDEN_RUN_S[0]), np.log(HIDDEN_RUN_S[1])))) * fps
    start_on = bool(rng.random() < mean_on / (mean_on + mean_off))
    return _burst_mask(n, mean_on, mean_off, start_on, rng)


def _duplicate_track(
    player_xy: np.ndarray,
    slot_team: np.ndarray,
    team: int,
    duplicate_p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(T, 2) float32. On each frame, with probability `duplicate_p`, one observed slot of `team`
    copied 0.2 to 0.6 m away. Frames that are not chosen, and frames with nobody observed, are NaN.
    """
    slots = np.flatnonzero(slot_team == team)
    frames = player_xy.shape[0]
    blank = np.full((frames, 2), np.nan, dtype=np.float32)
    if slots.size == 0:
        return blank
    xy = player_xy[:, slots]
    observed = np.isfinite(xy).all(axis=-1)
    keys = np.where(observed, rng.random((frames, slots.size)), -1.0)
    pick = np.argmax(keys, axis=1)
    chosen = xy[np.arange(frames), pick]
    do = (rng.random(frames) < duplicate_p) & observed.any(axis=1)
    low, high = DUPLICATE_DISTANCE_M
    magnitude = rng.uniform(low, high, size=frames)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=frames)
    offset = np.stack((magnitude * np.cos(theta), magnitude * np.sin(theta)), axis=1)
    dup = chosen.astype(np.float64) + offset
    dup[~do] = np.nan
    return dup.astype(np.float32)


def apply_ghosts(tracking: Tracking, cfg: GhostSlots, rng: np.random.Generator) -> Tracking:
    """`tracking` plus extra slots, or `tracking` itself when the draw adds none.

    Ghosts: `n ~ U{0..max_ghost_slots}` slots, a random team each, a random walk of step sd
    0.15 m per frame at 25 Hz (scaled with sqrt(25/fps)), started uniformly on the pitch. The
    integrated walk is clipped into the pitch, so a path that would leave sits on the boundary
    until the latent sum comes back. Visibility alternates exponential runs (visible mean
    log-uniform in 2..20 s, hidden in 5..60 s); hidden frames are NaN and the walk keeps going
    underneath. Duplicates, when `duplicate_p > 0`: two further columns (home then away), each
    finite only on the frames that coin lands, 0.2..0.6 m from one observed teammate. Original
    columns are copied bitwise. `player_speed` becomes None once the slot count changes, because
    the delivered channel would no longer line up; other fields keep their meaning.

    The duplicate column picks a slot independently per frame, so its velocity is not a run.
    """
    if cfg.p <= 0.0 or rng.random() >= cfg.p:
        return tracking
    meta = tracking.meta
    n = int(rng.integers(0, cfg.max_ghost_slots + 1))
    ghost_xy = None
    ghost_team = None
    if n:
        ghost_team = rng.integers(0, 2, size=n).astype(np.int8)
        half_l = float(meta.pitch_length) / 2.0
        half_w = float(meta.pitch_width) / 2.0
        start = np.column_stack(
            (rng.uniform(-half_l, half_l, size=n), rng.uniform(-half_w, half_w, size=n))
        )
        step_sd = GHOST_STEP_SD_M * np.sqrt(GHOST_REF_FPS / float(meta.fps))
        steps = rng.normal(0.0, step_sd, size=(meta.n_frames, n, 2))
        steps[0] = 0.0
        pos = start[None, :, :] + np.cumsum(steps, axis=0)
        pos[..., 0] = np.clip(pos[..., 0], -half_l, half_l)
        pos[..., 1] = np.clip(pos[..., 1], -half_w, half_w)
        visible = np.column_stack([_ghost_visible(meta.n_frames, meta.fps, rng) for _ in range(n)])
        pos[~visible] = np.nan
        ghost_xy = pos.astype(np.float32)
    dups: list[np.ndarray] = []
    dup_teams: list[int] = []
    if cfg.duplicate_p > 0.0:
        for team in (HOME, AWAY):
            dups.append(
                _duplicate_track(tracking.player_xy, tracking.slot_team, team, cfg.duplicate_p, rng)
            )
            dup_teams.append(team)
    if ghost_xy is None and not dups:
        return tracking
    parts = [tracking.player_xy]
    teams: list[np.ndarray] = [tracking.slot_team]
    if ghost_xy is not None and ghost_team is not None:
        parts.append(ghost_xy)
        teams.append(ghost_team)
    if dups:
        parts.append(np.stack(dups, axis=1))
        teams.append(np.asarray(dup_teams, dtype=np.int8))
    return replace(
        tracking,
        player_xy=np.concatenate(parts, axis=1),
        slot_team=np.concatenate(teams).astype(np.int8, copy=False),
        player_speed=None,
        _fingerprint=None,
    )


def _tracking_with_ball(tracking: Tracking, ball_xy: np.ndarray) -> Tracking:
    """`tracking` with `ball_xy` when that array is not already its ball. Players stay as they are."""
    if tracking.ball_xy is ball_xy:
        return tracking
    return replace(tracking, ball_xy=ball_xy, _fingerprint=None)


def _augment(
    tracking: Tracking,
    clean_ball: np.ndarray,
    mix: AugmentMix,
    rng: np.random.Generator,
) -> tuple[Tracking, np.ndarray, Tracking, np.ndarray]:
    """The augmentation draws, plus the pre-smooth tracking and decoupled ball.

    The third value is the returned tracking's players (ghosts included) with the ball from just
    before `apply_ball_smooth`. The fourth is that same moment's decoupled ball. Ghosts are drawn
    once, after the smooth, and shared. A block with `p == 0` is skipped before any draw.
    """
    tracking = apply_team_style(tracking, clean_ball, mix.team_style, rng)
    tracking, ball = apply_ball_offset(tracking, clean_ball, mix.ball_offset, rng)
    before_tracking, before_ball = tracking, ball
    tracking, ball = apply_ball_smooth(tracking, ball, mix.ball_smooth, rng)
    tracking = apply_ghosts(tracking, mix.ghosts, rng)
    teacher_tracking = _tracking_with_ball(tracking, before_tracking.ball_xy)
    return tracking, ball, teacher_tracking, before_ball


def apply_augmentations(
    tracking: Tracking,
    clean_ball: np.ndarray,
    mix: AugmentMix,
    rng: np.random.Generator,
) -> tuple[Tracking, np.ndarray]:
    """`(tracking, ball for the decoupled variant)`.

    Order, and the order random numbers are drawn: team style, then ball offset, then ball
    smooth, then ghosts. A block with `p == 0` is skipped before any draw. The returned ball is
    `clean_ball` itself when neither the offset nor the smooth is applied, so the caller can
    keep today's decoupled mask bit for bit. `apply_augmentations_presmooth` is this function
    plus the tracking and decoupled ball from just before the smooth.
    """
    tracking, ball, _teacher_tracking, _teacher_ball = _augment(tracking, clean_ball, mix, rng)
    return tracking, ball


def apply_augmentations_presmooth(
    tracking: Tracking,
    clean_ball: np.ndarray,
    mix: AugmentMix,
    rng: np.random.Generator,
) -> tuple[Tracking, np.ndarray, Tracking, np.ndarray]:
    """`apply_augmentations`, and the same draw with the ball-smoothing step skipped.

    Returns `(tracking, decoupled ball, pre-smooth tracking, pre-smooth decoupled ball)`. The
    pre-smooth tracking has the ghosts of the returned tracking and the ball from just before
    `apply_ball_smooth`. The random draws are the ones `apply_augmentations` makes, in the same
    order, including the smooth coin and sigma when `ball_smooth.p > 0`.
    """
    return _augment(tracking, clean_ball, mix, rng)
