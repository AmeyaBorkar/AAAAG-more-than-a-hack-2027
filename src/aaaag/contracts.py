"""Stage contracts: the only shapes that cross a module boundary.

Every lane codes against these, which is what lets four people (and their agents) build stages in
parallel: a stage that honours the contract can be developed on synthetic data today and dropped
into the real pipeline tomorrow without anyone waiting for anyone.

Conventions, enforced by the validate() methods and by tests/test_contracts.py:

- Time axis: the video frame index of the match video, 0 .. n_frames - 1. For BAR-ATH that is the
  submission row. Never the ATD frame_id, which counts from 1.
- Space: metres, pitch-centred. x runs along the length in [-L/2, L/2], y across in [-W/2, W/2];
  anything outside is off the pitch. Orientation is as delivered (no flipping at half-time), so a
  feature that needs the attacking direction has to infer it.
- Missing: NaN. Never 0, never a sentinel value.
- Label codes: the official ones, H=0 A=1 D=2 X=3 (asserted against possession_metrics in tests).
- Probabilities: float32 (T, 3) over (H, A, D). A row of NaN means "no opinion on this frame".
- Channels: optional (T,) float32 arrays on Evidence. Empty is the default and adds nothing.
- Changing anything here is a contract change: additive changes are fine, breaking ones need an
  RFC issue first (docs/WORKFLOW.md#contract-changes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
CLASSES = ("H", "A", "D")
HOME, AWAY = H, A  # team indices share the label codes on purpose: slot_team == H means home

PROBA_ATOL = 1e-3  # tolerance for "rows sum to one" after float32 arithmetic


@dataclass(frozen=True)
class Period:
    """A played period on the frame axis, half-open: frames start .. stop - 1."""

    index: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.stop:
            raise ValueError(f"bad period {self}")

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class MatchMeta:
    """Match-level constants. `periods` are the frames that are played (and, for BAR-ATH, graded)."""

    match_id: str
    source: str
    n_frames: int
    fps: int
    pitch_length: float
    pitch_width: float
    periods: tuple[Period, ...]
    home: str = "home"
    away: str = "away"

    def validate(self) -> MatchMeta:
        if self.n_frames <= 0 or self.fps <= 0:
            raise ValueError(f"{self.match_id}: n_frames and fps must be positive")
        if not self.periods:
            raise ValueError(f"{self.match_id}: no periods")
        previous_stop = 0
        for period in self.periods:
            if period.start < previous_stop or period.stop > self.n_frames:
                raise ValueError(f"{self.match_id}: periods overlap or exceed n_frames: {period}")
            previous_stop = period.stop
        return self

    def played_mask(self) -> np.ndarray:
        """(T,) bool, True inside a period."""
        mask = np.zeros(self.n_frames, dtype=bool)
        for period in self.periods:
            mask[period.start : period.stop] = True
        return mask

    def runs(self) -> list[tuple[int, int]]:
        """[(start, stop)] of the periods, in the format possession_metrics uses for runs."""
        return [(period.start, period.stop) for period in self.periods]


@dataclass
class Tracking:
    """Team-level tracking on the video-frame axis. Arrays only, never per-frame Python objects."""

    meta: MatchMeta
    player_xy: np.ndarray  # (T, S, 2) float32 metres, NaN where the slot is not observed
    slot_team: np.ndarray  # (S,) int8, HOME or AWAY
    ball_xy: np.ndarray  # (T, 2) float32 metres, NaN where the ball is not observed
    player_speed: np.ndarray | None = None  # (T, S) float32 m/s as delivered, NaN where missing
    ball_extra: dict[str, np.ndarray] = field(default_factory=dict)  # channel -> (T,) float32
    _fingerprint: str | None = field(default=None, repr=False, compare=False)

    @property
    def n_frames(self) -> int:
        return int(self.player_xy.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.player_xy.shape[1])

    def ball_seen(self) -> np.ndarray:
        """(T,) bool, True where the ball has a position."""
        return ~np.isnan(self.ball_xy[:, 0])

    def team_slots(self, team: int) -> np.ndarray:
        """Indices of the track slots belonging to HOME or AWAY."""
        return np.flatnonzero(self.slot_team == team)

    def validate(self) -> Tracking:
        meta = self.meta.validate()
        T, S = self.player_xy.shape[:2]
        _check(T == meta.n_frames, f"player_xy has {T} frames, meta says {meta.n_frames}")
        _check(self.player_xy.shape == (T, S, 2), "player_xy must be (T, S, 2)")
        _check(self.player_xy.dtype == np.float32, "player_xy must be float32")
        _check(self.slot_team.shape == (S,), "slot_team must be (S,)")
        _check(self.slot_team.dtype == np.int8, "slot_team must be int8")
        _check(np.isin(self.slot_team, (HOME, AWAY)).all(), "slot_team must be HOME or AWAY")
        _check(self.ball_xy.shape == (T, 2), "ball_xy must be (T, 2)")
        _check(self.ball_xy.dtype == np.float32, "ball_xy must be float32")
        if self.player_speed is not None:
            _check(self.player_speed.shape == (T, S), "player_speed must be (T, S)")
            _check(self.player_speed.dtype == np.float32, "player_speed must be float32")
        for name, channel in self.ball_extra.items():
            _check(channel.shape == (T,), f"ball_extra[{name!r}] must be (T,)")
        return self

    def fingerprint(self) -> str:
        """Content hash of the arrays and meta: the cache key for anything derived from them."""
        if self._fingerprint is None:
            digest = hashlib.sha256(repr(self.meta).encode())
            for array in (self.player_xy, self.slot_team, self.ball_xy, self.player_speed):
                if array is not None:
                    digest.update(np.ascontiguousarray(array).tobytes())
            for name in sorted(self.ball_extra):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(self.ball_extra[name]).tobytes())
            self._fingerprint = digest.hexdigest()
        return self._fingerprint


@dataclass(frozen=True)
class TagEvent:
    """One Smart Tagging instance on the frame axis, half-open [start, stop)."""

    code: str  # upper case, e.g. "SET PIECES"
    start: int
    stop: int
    team: int | None  # HOME, AWAY, or None when the tag names no team ("N/A", every throw-in)
    half: int | None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class MatchBundle:
    """Everything a stage may read about one match.

    `labels` is ground truth and exists ONLY for synthetic and external (other-match) bundles. It is
    always None for BAR-ATH: the public five minutes are never loaded into a bundle, so no stage
    can train or tune on them even by accident (docs/COMPLIANCE.md).
    """

    tracking: Tracking
    tags: list[TagEvent] | None = None
    labels: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> MatchMeta:
        return self.tracking.meta

    def validate(self) -> MatchBundle:
        self.tracking.validate()
        if self.labels is not None:
            _check(self.meta.source != "bar-ath", "BAR-ATH bundles must never carry labels")
            validate_labels(self.labels, self.meta.n_frames)
        return self

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.tracking.fingerprint().encode())
        digest.update(repr(self.tags).encode())
        return digest.hexdigest()


@dataclass
class Evidence:
    """One producer's per-frame opinion over (H, A, D).

    `channels` holds optional (T,) float32 side signals, for example a switch logit. Empty means
    the producer has none, which is every producer that does not set one.
    """

    name: str
    proba: np.ndarray  # (T, 3) float32; rows sum to 1, or are all NaN for "no opinion"
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self, n_frames: int) -> Evidence:
        p = self.proba
        _check(
            p.shape == (n_frames, 3), f"{self.name}: proba must be ({n_frames}, 3), got {p.shape}"
        )
        _check(p.dtype == np.float32, f"{self.name}: proba must be float32")
        nan = np.isnan(p)
        _check((nan.all(axis=1) | ~nan.any(axis=1)).all(), f"{self.name}: partial-NaN rows")
        known = p[~nan.any(axis=1)]
        _check(
            ((known >= -PROBA_ATOL) & (known <= 1 + PROBA_ATOL)).all(), f"{self.name}: out of [0,1]"
        )
        _check(
            np.allclose(known.sum(axis=1), 1.0, atol=PROBA_ATOL), f"{self.name}: rows must sum to 1"
        )
        for key, channel in self.channels.items():
            _check(
                isinstance(key, str) and key != "",
                f"{self.name}: channel name must be a non-empty string",
            )
            _check(
                isinstance(channel, np.ndarray), f"{self.name}: channel {key!r} must be an array"
            )
            _check(
                channel.shape == (n_frames,),
                f"{self.name}: channel {key!r} must be ({n_frames},), got {channel.shape}",
            )
            _check(channel.dtype == np.float32, f"{self.name}: channel {key!r} must be float32")
        return self

    def coverage(self) -> float:
        """Fraction of frames with an opinion."""
        return float((~np.isnan(self.proba[:, 0])).mean())

    @classmethod
    def from_codes(cls, name: str, codes: np.ndarray) -> Evidence:
        """One-hot evidence from per-frame codes in {H, A, D}; any other code becomes a NaN row."""
        proba = np.full((len(codes), 3), np.nan, dtype=np.float32)
        known = np.isin(codes, (H, A, D))
        proba[known] = 0.0
        proba[np.flatnonzero(known), codes[known].astype(np.int64)] = 1.0
        return cls(name, proba)

    @classmethod
    def from_factors(cls, name: str, p_dead: np.ndarray, p_home_if_alive: np.ndarray) -> Evidence:
        """Joint (H, A, D) from P(dead) and P(home | alive). NaN in either factor -> NaN row."""
        p_dead = p_dead.astype(np.float32)
        p_home = p_home_if_alive.astype(np.float32)
        alive = 1.0 - p_dead
        proba = np.stack([alive * p_home, alive * (1.0 - p_home), p_dead], axis=1)
        proba[np.isnan(proba).any(axis=1)] = np.nan
        return cls(name, proba.astype(np.float32))


def validate_labels(labels: np.ndarray, n_frames: int) -> np.ndarray:
    """Labels are (T,) int8 codes in {H, A, D, X}."""
    _check(labels.shape == (n_frames,), f"labels must be ({n_frames},), got {labels.shape}")
    _check(labels.dtype == np.int8, "labels must be int8")
    _check(((labels >= H) & (labels <= X)).all(), "labels must be in {H, A, D, X}")
    return labels


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
