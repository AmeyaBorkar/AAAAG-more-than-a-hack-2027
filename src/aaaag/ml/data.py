"""Training data: clean labelled matches, degraded on the fly into token tensors.

Per variant set (a few hundred steps) every training match gets one fresh draw of
`aaaag.degrade` (a new seed, a player-ladder rung and a stoppage-edge jitter, all drawn from a
generator seeded by the run seed, the set and the match), and is built into two token tensors:

- **coupled**: the ball as degrade-v5 leaves it, hidden through the stoppages and in live gaps;
- **decoupled**: the same players with the match's complete provider ball hidden by a renewal
  process that ignores the labels (augment.py), which breaks the "missing ball means dead" shortcut.

Batches draw from one or the other, so half of them break it. Optional `AugmentMix` warps are
applied to that degraded tracking before either tokenisation (both variants share the player
change and the ball offset); with every probability at zero they draw no random numbers. Tokens
are stored as float16 (values are O(1)): a 146k-frame match is 94 MB per variant. A set
keeps every match's tensors, and the next set is built while they are still resident, so
48 matches and both variants are about 9 GB on the host before a device copy.
`train.matches_per_set`, when it is a positive int smaller than the roster, caps how many
matches a set builds. Null, the default, leaves the set as it is. With `teacher_input` `clean_ball` each variant also stores a second array
from that same draw with the ball-smoothing step skipped (one extra array per variant).
`same` stores none. With `train.aux_displaced_m` above 0 each variant also stores a (T,) bool
mask of the frames whose fed ball is seen but farther than that from the clean provider ball;
0 stores none.

Labels, the auxiliary ball target and the class weights come from the clean bundle. The IDSSE
exam matches never get here: the source refuses them unless `exam=True`, and nothing passes it.
`load_extra_matches` is the other door: `source/match_id` from a registered source other than
`idsse` or `bar-ath`, always labelled. X stays in the label array and the loss ignores it, as for
every IDSSE match.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..contracts import A, D, H, MatchBundle, X
from ..degrade import degrade
from ..postprocess.touch_snap import detect_touches
from ..registry import SOURCES
from ..sources import idsse, idsse_events
from .augment import (
    AugmentMix,
    alive_gap_mask,
    apply_augmentations,
    apply_augmentations_presmooth,
    hidden_ball_mask,
    with_ball,
)
from .geometry import attack_sign
from .label_views import INTERIOR_S, interior_view
from .possessor import possessor_targets
from .spec import TokenSpec
from .track_stress import (
    SWAP_P,
    dead_ball_visible,
    drop_and_duplicate,
    fragment_tracks,
    long_ball_gaps,
)
from .windows import build_match_tokens

SEED_LOW, SEED_HIGH = 1000, 2**30  # degrade seeds of training draws; the suites use 0-9

# The sealed exam lives in the IDSSE source. BAR-ATH is the submission match and has no labels.
FORBIDDEN_EXTRA_SOURCES = frozenset({"idsse", "bar-ath"})


@dataclass(frozen=True)
class DegradeMix:
    """Which degradations a training draw picks from (all fitted on or set by physics, never on
    BAR-ATH). `version=None` trains on the bundle as given (synthetic tests)."""

    version: str | None = "5"
    ladders: tuple[str | None, ...] = (
        None,
        "v51-mild",
        "v51-medium",
        "v51-severe",
        "v51-medium-nearball",
    )
    ladder_p: tuple[float, ...] = (0.30, 0.15, 0.20, 0.20, 0.15)
    jitter_s: tuple[float, ...] = (0.0, 0.5, 2.0)
    jitter_p: tuple[float, ...] = (0.50, 0.25, 0.25)

    def __post_init__(self) -> None:
        if len(self.ladders) != len(self.ladder_p) or len(self.jitter_s) != len(self.jitter_p):
            raise ValueError("DegradeMix: values and probabilities differ in length")

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """`degrade` keyword arguments for one draw, seed included."""
        if self.version is None:
            return {}
        ladder = self.ladders[int(rng.choice(len(self.ladders), p=_norm(self.ladder_p)))]
        jitter = self.jitter_s[int(rng.choice(len(self.jitter_s), p=_norm(self.jitter_p)))]
        draw: dict[str, Any] = {
            "version": self.version,
            "seed": int(rng.integers(SEED_LOW, SEED_HIGH)),
        }
        if ladder is not None:
            draw["ladder"] = ladder
        if jitter:
            draw["edge_jitter_s"] = float(jitter)
        return draw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DegradeMix:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def _norm(p: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    return arr / arr.sum()


@dataclass
class TrainMatch:
    """A clean labelled match and the per-frame arrays training reads from it."""

    match_id: str
    bundle: MatchBundle
    labels: np.ndarray  # (T,) int8 H/A/D/X
    sign: np.ndarray  # (T,) float32 attack-direction multiplier of the clean match
    ball_target: (
        np.ndarray
    )  # (T, 2) float32: clean ball, attack-normalised, in units of ref_scale_m
    target_ok: np.ndarray  # (T,) bool: the target exists and the ball is alive
    played: np.ndarray  # (T,) bool
    extra: bool = False  # True: not an IDSSE dev match; DegradeMix is optional
    label_free: bool = False  # True: extra_labels "none"; aux and team-agnostic possessor only
    touch: np.ndarray | None = None  # (T,) float32 touch target; None unless training asks for it
    event_switch: np.ndarray | None = None  # (T,) float32 event-switch target; None unless asked
    boundary: np.ndarray | None = None  # (T,) float32 E155 frame weight; None unless training asks

    @classmethod
    def from_bundle(
        cls, bundle: MatchBundle, spec: TokenSpec, *, extra: bool = False
    ) -> TrainMatch:
        if bundle.labels is None:
            raise ValueError(f"{bundle.meta.match_id}: training needs a labelled match")
        tracking = bundle.tracking
        sign = attack_sign(tracking)
        ball = tracking.ball_xy * sign[:, None] / spec.ref_scale_m
        played = tracking.meta.played_mask()
        ok = ~np.isnan(ball[:, 0]) & np.isin(bundle.labels, (H, A)) & played
        return cls(
            match_id=bundle.meta.match_id,
            bundle=bundle,
            labels=bundle.labels,
            sign=sign,
            ball_target=np.nan_to_num(ball).astype(np.float32),
            target_ok=ok,
            played=played,
            extra=extra,
        )


TOUCH_SOURCES = ("tracking", "events")


def _idsse_touch_events(bundle: MatchBundle) -> np.ndarray:
    return idsse_events.touch_events(bundle.meta.match_id, bundle=bundle)


# Sources whose matches have provider event data: source name -> (clean bundle -> TOUCH_DTYPE
# events, `aaaag.sources.idsse_events`). A source not listed here cannot use event targets.
EVENT_SOURCES: dict[str, Callable[[MatchBundle], np.ndarray]] = {"idsse": _idsse_touch_events}


def source_touch_events(bundle: MatchBundle) -> np.ndarray:
    """The provider events of a clean training match (TOUCH_DTYPE), or a clear error."""
    source = bundle.meta.source
    load = EVENT_SOURCES.get(source)
    if load is None:
        raise ValueError(
            f"{bundle.meta.match_id}: source {source!r} has no touch_events, so it cannot train on "
            f"event targets (train.touch_source 'events' or train.event_switch_weight > 0); "
            f"sources with events: {sorted(EVENT_SOURCES)}"
        )
    return load(bundle)


def _check_tol(tol: int, name: str) -> None:
    if isinstance(tol, bool) or not isinstance(tol, int) or tol < 0:
        raise ValueError(f"{name} must be an int >= 0, got {tol!r}")


def touch_target(match: TrainMatch, tol: int, source: str = "tracking") -> np.ndarray:
    """(T,) float32 target for a ball touch on the clean tracking. 1 positive, 0 negative, NaN ignored.

    `source` "tracking" (the default): touches are `detect_touches(clean_tracking)` with its default
    theta, reach and w. "events": they are the frames of the match's provider on-ball events
    (`source_touch_events`, `idsse_events.TOUCH_KINDS`), and a source without events raises.
    Every team counts either way. A frame within `tol` of one of those frames is positive when it
    is also supervised: the label is H or A, the frame is inside a period, and the clean ball is
    seen. Every other supervised frame is 0. An unseen clean ball, a dead label, X and an unplayed
    frame are NaN, including when they sit inside the tolerance of a touch. Built once per match,
    before degradation. A team swap and a y flip do not change it: a touch has no team.
    """
    _check_tol(tol, "touch_tol_frames")
    if source not in TOUCH_SOURCES:
        raise ValueError(f"train.touch_source must be one of {TOUCH_SOURCES}, got {source!r}")
    tracking = match.bundle.tracking
    n = tracking.n_frames
    labels = np.asarray(match.labels)
    if labels.shape != (n,):
        raise ValueError(f"{match.match_id}: labels {labels.shape} != {(n,)}")
    if source == "tracking":
        frames, _teams = detect_touches(tracking)
    else:
        frames = idsse_events.touch_frames(source_touch_events(match.bundle))
    near = _touch_neighbourhood(n, frames, tol)
    supervised = tracking.ball_seen() & np.isin(labels, (H, A)) & match.played
    target = np.full(n, np.nan, dtype=np.float32)
    target[supervised] = 0.0
    target[supervised & near] = 1.0
    return target


def with_touch_target(match: TrainMatch, tol: int, source: str = "tracking") -> TrainMatch:
    """`match` with `touch` filled from the clean ball (or its events, `source` "events"). Does
    not draw random numbers.

    The original match is left without a target. Call this only when the touch loss is on:
    weight 0 never builds the array.
    """
    return replace(match, touch=touch_target(match, tol, source))


def event_switch_target(match: TrainMatch, tol: int) -> np.ndarray:
    """(T,) float32 switch-head target from the provider events. 1 positive, 0 negative, NaN ignored.

    The changes are `idsse_events.possession_changes`: consecutive on-ball events of different
    teams with no foul, out or set piece between them, placed on the new team's first event. A
    frame labelled H or A inside a period is 1 within `tol` frames of a change and 0 elsewhere;
    D, X and unplayed frames are NaN. Built once per match from the clean bundle, before
    degradation; a source without events raises. A team swap and a y flip leave it unchanged,
    because a switch has no direction.
    """
    _check_tol(tol, "switch_tol_frames")
    n = match.bundle.meta.n_frames
    labels = np.asarray(match.labels)
    if labels.shape != (n,):
        raise ValueError(f"{match.match_id}: labels {labels.shape} != {(n,)}")
    events = source_touch_events(match.bundle)
    changes = idsse_events.possession_changes(events, match.bundle.meta.periods)
    near = _touch_neighbourhood(n, changes, tol)
    supervised = np.isin(labels, (H, A)) & match.played
    target = np.full(n, np.nan, dtype=np.float32)
    target[supervised] = 0.0
    target[supervised & near] = 1.0
    return target


def with_event_switch_target(match: TrainMatch, tol: int) -> TrainMatch:
    """`match` with `event_switch` filled. Does not draw random numbers; build it only when
    `train.event_switch_weight` is above 0."""
    return replace(match, event_switch=event_switch_target(match, tol))


def _touch_neighbourhood(n: int, frames: np.ndarray, tol: int) -> np.ndarray:
    """(T,) bool, True within `tol` frames of a touch index. No wrap at the ends of the match."""
    if len(frames) == 0:
        return np.zeros(n, dtype=bool)
    index = np.asarray(frames, dtype=np.int64)
    lo = np.clip(index - tol, 0, n)
    hi = np.clip(index + tol + 1, 0, n)
    delta = np.zeros(n + 1, dtype=np.int32)
    np.add.at(delta, lo, 1)
    np.add.at(delta, hi, -1)
    return np.cumsum(delta[:-1]) > 0


@dataclass
class SetEntry:
    """One match's two variants for one variant set."""

    match_id: str
    coupled: np.ndarray  # (T, n_tokens, N_FEATURES) float16
    decoupled: np.ndarray
    visible_coupled: np.ndarray  # (T,) bool: the ball is seen in the input
    visible_decoupled: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    possessor_coupled: np.ndarray | None = None  # (T,) int16, or None when the head is off
    possessor_decoupled: np.ndarray | None = None
    # Clean-ball teacher tokens, float16, or None when the teacher sees the student tokens.
    # One extra array per variant: the same shape as `coupled` / `decoupled`.
    teacher_coupled: np.ndarray | None = None
    teacher_decoupled: np.ndarray | None = None
    # (T,) bool: the fed ball is seen and farther than `aux_displaced_m` from the clean provider
    # ball. None unless `aux_displaced_m` is positive.
    displaced_coupled: np.ndarray | None = None
    displaced_decoupled: np.ndarray | None = None


def displaced_ball_mask(
    fed_xy: np.ndarray, target_xy: np.ndarray, threshold_m: float
) -> np.ndarray:
    """(T,) bool, True where both balls are seen and lie more than `threshold_m` metres apart.

    Both are (T, 2) pitch positions in metres, in the same frame, before attack normalisation and
    before `ref_scale_m`. A NaN on either side is False. Draws no random numbers.
    """
    fed = np.asarray(fed_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if fed.shape != target.shape:
        raise ValueError(f"fed ball {fed.shape} != target ball {target.shape}")
    distance = np.hypot(fed[:, 0] - target[:, 0], fed[:, 1] - target[:, 1])
    with np.errstate(invalid="ignore"):
        return np.isfinite(distance) & (distance > float(threshold_m))


def restart_blank_mask(
    labels: np.ndarray,
    periods,
    fps: int,
    rng: np.random.Generator,
    p: float,
    before_s: float,
    after_s: float,
    *,
    min_dead_s: float = 2.0,
) -> np.ndarray:
    """(T,) bool, True where the ball is hidden around true restarts (E56).

    A restart is the end of a D run of at least `min_dead_s` that is followed by H or A inside one
    period. Each is drawn with probability `p`; a drawn restart at frame r hides
    [r - U(0, before_s), r + U(0, after_s)) seconds, clipped to its period. On BAR-ATH most
    restarts have no ball in sight, while degrade-v5 shows the ball again exactly at the restart.
    Draws from `rng` once per restart (three numbers), so the stream depends only on the labels.
    """
    mask = np.zeros(len(labels), dtype=bool)
    for period, _start, r in true_stoppages(labels, periods, fps, min_dead_s):
        take, before, after = rng.random(3)
        if take >= p:
            continue
        lo = max(period.start, r - int(round(before * before_s * fps)))
        hi = min(period.stop, r + int(round(after * after_s * fps)))
        mask[lo:hi] = True
    return mask


def true_stoppages(labels: np.ndarray, periods, fps: int, min_dead_s: float = 2.0):
    """[(period, start, restart)]: D runs of at least `min_dead_s` followed by H or A in one period.

    `start` is the first dead frame and `restart` the first alive one, both absolute. A dead run
    that ends the period, or runs into X, is not a stoppage. Periods in order, runs in order.
    """
    out = []
    min_dead = int(round(min_dead_s * fps))
    for period in periods:
        x = np.asarray(labels[period.start : period.stop])
        breaks = np.flatnonzero(x[1:] != x[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(x)]))
        for i in range(len(bounds) - 2):
            a, z = bounds[i], bounds[i + 1]
            if x[a] != D or z - a < min_dead or x[z] not in (H, A):
                continue
            out.append((period, period.start + int(a), period.start + int(z)))
    return out


def stoppage_carry_ball(
    player_xy: np.ndarray,
    ball_xy: np.ndarray,
    clean_ball: np.ndarray,
    labels: np.ndarray,
    periods,
    fps: int,
    rng: np.random.Generator,
    p: float,
    max_s: float,
    *,
    min_dead_s: float = 2.0,
    offset_m: tuple[float, float] = (0.2, 1.0),
) -> tuple[np.ndarray, int]:
    """The ball carried by the restart taker through true stoppages (E59), and the frames changed.

    degrade-v5 hides the ball through every stoppage and the clean IDSSE ball freezes at the
    out-of-play spot, so in training a stoppage never shows a ball moving with a player. On a
    broadcast tracker the ball is often kept while a player carries it back to the restart.
    Each stoppage (`true_stoppages`) is drawn with probability `p`. Its taker is the slot nearest
    the clean ball at the restart frame r. From r - U(0, max_s) s, but not before the stoppage
    starts, up to r, the ball is the taker's position plus one offset of U(offset_m) metres in a
    uniform direction, on the frames where the taker is seen; other frames keep `ball_xy`. Draws
    four numbers per stoppage whatever happens, so the stream depends only on the labels.
    Fails when the taker is wrong: a slot swap in the tracking moves the carried ball with it.
    """
    out = np.array(ball_xy, dtype=np.float32, copy=True)
    changed = 0
    lo_m, hi_m = offset_m
    for _period, start, r in true_stoppages(labels, periods, fps, min_dead_s):
        take, length, angle, radius = rng.random(4)
        if take >= p:
            continue
        ref = clean_ball[r]
        if not np.isfinite(ref).all():
            continue
        dist = np.hypot(*(player_xy[r] - ref).T)
        if not np.isfinite(dist).any():
            continue
        slot = int(np.nanargmin(dist))
        lo = max(start, r - int(round(length * max_s * fps)))
        track = player_xy[lo:r, slot]
        seen = np.isfinite(track).all(axis=1)
        size = lo_m + radius * (hi_m - lo_m)
        offset = size * np.array([np.cos(2 * np.pi * angle), np.sin(2 * np.pi * angle)])
        out[lo:r][seen] = (track[seen] + offset).astype(np.float32)
        changed += int(seen.sum())
    return out, changed


def build_set(
    matches: list[TrainMatch],
    set_index: int,
    seed: int,
    spec: TokenSpec,
    mix: DegradeMix,
    *,
    workers: int = 4,
    augment: AugmentMix | None = None,
    possessor_weight: float = 0.0,
    possessor_reach_m: float = 2.0,
    extra_degrade: bool = False,
    teacher_input: str = "same",
    restart_blank: tuple[float, float, float] | None = None,
    stoppage_carry: tuple[float, float] | None = None,
    dead_visible: float | None = None,
    track_stress: tuple[float, float, float] | None = None,
    seed_index: Sequence[int] | None = None,
    airborne: tuple[float, float, float] | None = None,
    aux_displaced_m: float = 0.0,
    airborne_extra: bool = False,
    alive_gap: float | None = None,
) -> list[SetEntry]:
    """One fresh degraded draw per match; deterministic in (seed, set_index, match order).

    The matches are built on `workers` threads (numpy releases the GIL in the heavy steps); each
    draws from its own generator, so the result does not depend on the thread timing. `augment`
    is the optional training warp; every probability at zero leaves the draw unchanged.
    `possessor_weight` above 0 stores an int16 slot target per variant. Zero leaves those arrays
    None and does not draw from `rng`. An extra match skips `DegradeMix` unless `extra_degrade`;
    the augmentations still run. Each match keeps the generator seeded by its index, so matches
    already on the list do not move when extra matches are appended.
    `teacher_input` `clean_ball` stores the pre-smooth tokens; `same` stores none and draws no
    extra random numbers.
    `seed_index`, when given, is each match's index in the full roster and seeds that match's
    generator. None uses the position in `matches`, which is the same draw.
    `aux_displaced_m` above 0 stores each variant's displaced-ball mask (`build_entry`); 0 stores
    none. `airborne_extra` also projects extra matches that have a ball height (`build_entry`);
    False leaves them as they are.
    `alive_gap` above 0 adds label-free gaps to the coupled ball on alive frames (`build_entry`);
    None or 0 draws nothing.
    """
    augment = AugmentMix() if augment is None else augment
    if seed_index is not None and len(seed_index) != len(matches):
        raise ValueError(f"seed_index length {len(seed_index)} != {len(matches)} matches")

    def one(i: int) -> SetEntry:
        at = i if seed_index is None else seed_index[i]
        return build_entry(
            matches[i],
            np.random.default_rng([seed, set_index, at]),
            spec,
            mix,
            augment,
            possessor_weight=possessor_weight,
            possessor_reach_m=possessor_reach_m,
            skip_degrade=matches[i].extra and not extra_degrade,
            teacher_input=teacher_input,
            restart_blank=restart_blank,
            stoppage_carry=stoppage_carry,
            dead_visible=dead_visible,
            track_stress=track_stress,
            airborne=airborne,
            aux_displaced_m=aux_displaced_m,
            airborne_extra=airborne_extra,
            alive_gap=alive_gap,
        )

    if workers <= 1 or len(matches) <= 1:
        return [one(i) for i in range(len(matches))]
    with ThreadPoolExecutor(max_workers=min(workers, len(matches))) as pool:
        return list(pool.map(one, range(len(matches))))


def build_entry(
    match: TrainMatch,
    rng: np.random.Generator,
    spec: TokenSpec,
    mix: DegradeMix,
    augment: AugmentMix | None = None,
    *,
    possessor_weight: float = 0.0,
    possessor_reach_m: float = 2.0,
    skip_degrade: bool = False,
    teacher_input: str = "same",
    restart_blank: tuple[float, float, float] | None = None,
    stoppage_carry: tuple[float, float] | None = None,
    dead_visible: float | None = None,
    track_stress: tuple[float, float, float] | None = None,
    airborne: tuple[float, float, float] | None = None,
    aux_displaced_m: float = 0.0,
    airborne_extra: bool = False,
    alive_gap: float | None = None,
) -> SetEntry:
    """One match, one draw. `match.sign` stays the clean match's attack direction.

    Augmentations run on the degraded tracking before both tokenisations. The ball offset is
    applied to the degraded ball (coupled) and to the clean provider ball (decoupled) with the
    same draw. A mix whose probabilities are all zero does not touch `rng`.
    When `possessor_weight` is positive, each variant gets an int16 possessor target from its own
    tokens (possessor.py) and the clean ball. Zero weight leaves those arrays None and draws
    nothing more from `rng`. `skip_degrade` does not draw from `DegradeMix` and does not call
    `degrade`, so the tracking is the bundle as delivered; augmentations still run and still use
    `rng`. Leaving it false is the previous draw.
    `teacher_input` `clean_ball` also stores tokens from this draw with ball smoothing skipped.
    The smooth coin and sigma are still drawn, and every other draw matches `same`. `same` does
    not build those arrays and does not draw any extra random numbers.
    `restart_blank` (p, before_s, after_s), E56: on a degraded match, the coupled variant's ball
    is also hidden around each true restart with probability p (`restart_blank_mask`). None draws
    nothing.
    `stoppage_carry` (p, max_s), E59: on a degraded match, before `restart_blank`, the coupled
    variant's ball follows the restart taker through each true stoppage with probability p
    (`stoppage_carry_ball`). None draws nothing.
    `dead_visible` p, E71: on a degraded match, after `stoppage_carry` and before `restart_blank`,
    the coupled variant's ball is the visible uncarried dead ball on each true stoppage of at
    least 4 s with probability p (`dead_ball_visible`). None or 0 draws nothing.
    `track_stress` (frag_swap_p, drop_dup_p, long_gap_p), E66: on a degraded match, after
    `restart_blank`, each tracker error of track_stress.py is applied with its probability: wrong-team
    fragments and dropouts with duplicates change the players of both variants, and long ball gaps
    hide the coupled ball. A zero probability draws nothing; None draws nothing.
    `airborne` (p, cz_min, cz_max), E119: on an IDSSE match, after `dead_visible` and before
    `restart_blank`, the coupled variant's ball is projected through one random tactical camera
    with probability p (`airborne.airborne_ball`, using the ball's Z). None or 0 draws nothing.
    An extra match is skipped and draws nothing unless `airborne_extra`; then it is projected the
    same way, at the same point of the draw, when its source has a ball height
    (`airborne.ball_height`: PFF's raw-ball z), and a source without one draws nothing.
    `aux_displaced_m` above 0 stores, per variant, where the ball that variant's tokens are built
    from is seen but more than that many metres from the clean provider ball, the ball behind
    `TrainMatch.ball_target` (`displaced_ball_mask`). The fed ball is taken after every step
    above (degrade, stoppage carry, dead visible, airborne, restart blank, track stress,
    augmentations, and the renewal hiding of the decoupled variant), and both are compared in
    pitch metres before attack normalisation. 0 computes and stores nothing; neither value draws
    random numbers.
    `alive_gap` target, E158: on a degraded match, after the augmentations and after the decoupled
    variant's renewal mask is drawn, the coupled variant's ball is also hidden on alive frames by
    a label-free renewal process (`augment.alive_gap_mask`, run lengths from that decoupled
    renewal) until its missing share over the alive frames reaches the target in expectation.
    Dead frames, the decoupled variant and every earlier draw are unchanged; a clean-ball teacher
    gets the same gaps. `info["alive_gap_frames"]` counts the frames it hid. None or 0 draws
    nothing.
    """
    if not aux_displaced_m >= 0.0:
        raise ValueError(f"aux_displaced_m must be >= 0, got {aux_displaced_m!r}")
    if teacher_input not in ("same", "clean_ball"):
        raise ValueError(f"teacher_input must be 'same' or 'clean_ball', got {teacher_input!r}")
    augment = AugmentMix() if augment is None else augment
    draw = {} if skip_degrade else mix.draw(rng)
    clean = match.bundle
    degraded = degrade(clean, **draw) if draw else clean
    tracking = degraded.tracking
    meta = tracking.meta
    carried = 0
    if stoppage_carry is not None and not skip_degrade and stoppage_carry[0] > 0:
        ball, carried = stoppage_carry_ball(
            tracking.player_xy,
            tracking.ball_xy,
            clean.tracking.ball_xy,
            match.labels,
            meta.periods,
            meta.fps,
            rng,
            *stoppage_carry,
        )
        tracking = with_ball(tracking, ball)
    changed = 0
    if dead_visible is not None and not skip_degrade and dead_visible > 0:
        ball, changed = dead_ball_visible(
            tracking.ball_xy,
            clean.tracking.ball_xy,
            match.labels,
            meta.periods,
            meta.fps,
            rng,
            dead_visible,
        )
        tracking = with_ball(tracking, ball)
    if airborne is not None and (airborne_extra or not match.extra) and airborne[0] > 0:
        from .airborne import airborne_ball, ball_height

        height = ball_height(match.match_id, clean, extra=match.extra)
        if height is not None:
            ball, _ = airborne_ball(tracking.ball_xy, height, meta.pitch_width, rng, *airborne)
            tracking = with_ball(tracking, ball)
    if restart_blank is not None and not skip_degrade and restart_blank[0] > 0:
        blank = restart_blank_mask(match.labels, meta.periods, meta.fps, rng, *restart_blank)
        tracking = with_ball(tracking, np.where(blank[:, None], np.nan, tracking.ball_xy))
    stressed: list[str] = []
    if track_stress is not None and not skip_degrade:
        frag_p, dup_p, gap_p = track_stress
        if frag_p > 0 and rng.random() < frag_p:
            tracking = fragment_tracks(tracking, rng, swap_p=SWAP_P)
            stressed.append("frag_swap")
        if dup_p > 0 and rng.random() < dup_p:
            tracking = drop_and_duplicate(tracking, rng)
            stressed.append("drop_dup")
        if gap_p > 0 and rng.random() < gap_p:
            gaps = long_ball_gaps(tracking.ball_xy, match.labels, meta.periods, meta.fps, rng)
            tracking = with_ball(tracking, np.where(gaps[:, None], np.nan, tracking.ball_xy))
            stressed.append("long_gaps")
    teacher_tracking = teacher_ball = None
    if teacher_input == "clean_ball":
        tracking, variant_ball, teacher_tracking, teacher_ball = apply_augmentations_presmooth(
            tracking, clean.tracking.ball_xy, augment, rng
        )
    else:
        tracking, variant_ball = apply_augmentations(tracking, clean.tracking.ball_xy, augment, rng)

    # The token builder draws nothing, so the decoupled mask can be drawn before the coupled tokens.
    hidden = hidden_ball_mask(meta.n_frames, meta.periods, meta.fps, rng)
    gap_frames = None
    if alive_gap is not None and not skip_degrade and alive_gap > 0:
        unseen = ~tracking.ball_seen()
        gaps = alive_gap_mask(unseen, match.labels, meta.periods, meta.fps, rng, alive_gap)
        gap_frames = int((gaps & ~unseen).sum())
        shared = teacher_tracking is tracking
        tracking = with_ball(tracking, np.where(gaps[:, None], np.nan, tracking.ball_xy))
        if teacher_tracking is not None:
            teacher_tracking = (
                tracking
                if shared
                else with_ball(
                    teacher_tracking, np.where(gaps[:, None], np.nan, teacher_tracking.ball_xy)
                )
            )
    coupled = build_match_tokens(tracking, spec, sign=match.sign).astype(np.float16)
    independent = with_ball(tracking, np.where(hidden[:, None], np.nan, variant_ball))
    decoupled = build_match_tokens(independent, spec, sign=match.sign).astype(np.float16)
    teacher_coupled = teacher_decoupled = None
    if teacher_input == "clean_ball":
        if teacher_tracking is None or teacher_ball is None:
            raise RuntimeError("clean_ball did not keep the pre-smooth tracking")
        # No smooth this draw: the pre-smooth tracking is the student tracking. Share the arrays.
        if teacher_tracking is tracking and teacher_ball is variant_ball:
            teacher_coupled = coupled
            teacher_decoupled = decoupled
        else:
            teacher_coupled = build_match_tokens(teacher_tracking, spec, sign=match.sign).astype(
                np.float16
            )
            teacher_hidden = with_ball(
                teacher_tracking, np.where(hidden[:, None], np.nan, teacher_ball)
            )
            teacher_decoupled = build_match_tokens(teacher_hidden, spec, sign=match.sign).astype(
                np.float16
            )

    played = match.played
    info = {
        "draw": draw,
        "ball_hidden_coupled": float((~tracking.ball_seen())[played].mean()),
        "ball_hidden_decoupled": float((~independent.ball_seen())[played].mean()),
    }
    if stoppage_carry is not None:
        info["ball_carried_frames"] = carried
    if dead_visible is not None and dead_visible > 0:
        info["dead_visible_stoppages"] = changed
    if track_stress is not None:
        info["track_stress"] = stressed
    if gap_frames is not None:
        info["alive_gap_frames"] = gap_frames
    possessor_coupled = possessor_decoupled = None
    if possessor_weight > 0:
        clean_ball = clean.tracking.ball_xy
        possessor_coupled = possessor_targets(
            coupled,
            match.labels,
            clean_ball,
            match.sign,
            clean.meta,
            spec,
            possessor_reach_m,
            team_agnostic=match.label_free,
        )
        possessor_decoupled = possessor_targets(
            decoupled,
            match.labels,
            clean_ball,
            match.sign,
            clean.meta,
            spec,
            possessor_reach_m,
            team_agnostic=match.label_free,
        )
    visible_coupled = tracking.ball_seen() & played
    visible_decoupled = independent.ball_seen() & played
    displaced_coupled = displaced_decoupled = None
    if aux_displaced_m > 0:
        # The fed balls the tokens were built from, against the clean provider ball the aux loss
        # regresses to (`ball_target` is that ball times the attack sign over ref_scale_m), both
        # in pitch metres. The sign is +-1 on both axes, so it does not change a distance.
        target = clean.tracking.ball_xy
        displaced_coupled = visible_coupled & displaced_ball_mask(
            tracking.ball_xy, target, aux_displaced_m
        )
        displaced_decoupled = visible_decoupled & displaced_ball_mask(
            independent.ball_xy, target, aux_displaced_m
        )
        info["ball_displaced_coupled"] = float(displaced_coupled[played].mean())
        info["ball_displaced_decoupled"] = float(displaced_decoupled[played].mean())
    return SetEntry(
        match_id=match.match_id,
        coupled=coupled,
        decoupled=decoupled,
        visible_coupled=visible_coupled,
        visible_decoupled=visible_decoupled,
        info=info,
        possessor_coupled=possessor_coupled,
        possessor_decoupled=possessor_decoupled,
        teacher_coupled=teacher_coupled,
        teacher_decoupled=teacher_decoupled,
        displaced_coupled=displaced_coupled,
        displaced_decoupled=displaced_decoupled,
    )


def class_weights(matches: list[TrainMatch]) -> np.ndarray:
    """(3,) float64 weights for H, A, D: inverse frequency, with H and A sharing one weight (they
    exchange under the team swap), scaled so the expected weight of a played frame is 1.

    A roster with no H, A or D frame (every label is X) returns ones. A pretraining run does not
    call this; the ones keep a direct call from dividing a zero count into the share."""
    counts = np.array(
        [sum(int(((m.labels == c) & m.played).sum()) for m in matches) for c in range(3)],
        dtype=np.float64,
    )
    if counts.sum() == 0:
        return np.ones(3, dtype=np.float64)
    share = counts / counts.sum()
    share[:2] = share[:2].mean()
    weight = 1.0 / np.maximum(share, 1e-6)
    return weight / float((share * weight).sum())


def load_train_matches(match_ids: Sequence[str], spec: TokenSpec) -> dict[str, TrainMatch]:
    """The clean provider tracking of IDSSE dev matches, ready for training. Refuses anything that
    is not a dev match, so the sealed exam cannot get in by a typo."""
    bad = [m for m in match_ids if m not in idsse.DEV_MATCHES]
    if bad:
        raise ValueError(f"only the IDSSE dev matches {idsse.DEV_MATCHES}; got {bad}")
    return {m: TrainMatch.from_bundle(idsse.load_idsse(match_id=m), spec) for m in match_ids}


def split_extra_entry(entry: str) -> tuple[str, str]:
    """`'source/match_id'`. Refuses `idsse` and `bar-ath` before any source is called."""
    if not isinstance(entry, str):
        raise ValueError(f"extra match must be 'source/match_id', got {entry!r}")
    source, sep, match_id = entry.strip().partition("/")
    if not sep or not source or not match_id:
        raise ValueError(f"extra match must be 'source/match_id', got {entry!r}")
    if source in FORBIDDEN_EXTRA_SOURCES:
        if source == "idsse":
            raise ValueError(
                f"extra matches cannot use source 'idsse' ({entry.strip()}): "
                "that would let the sealed exam into training"
            )
        raise ValueError(
            f"extra matches cannot use source 'bar-ath' ({entry.strip()}): "
            "that is the submission match"
        )
    return source, match_id


EXTRA_LABEL_MODES = ("all", "team", "core", "none", "interior")
CORE_MIN_S = 2.0  # "core": a dead run shorter than this is X
CORE_EDGE_S = 0.5  # "core": this much at each edge of a longer dead run is X


def extra_label_view(labels: np.ndarray, bundle: MatchBundle, mode: str) -> np.ndarray:
    """The labels an extra match trains on (E35 arms; the SkillCorner research note).

    - "all": as the source delivers them.
    - "team": D becomes X, so only H and A enter the loss. Use this when a source's dead-ball
      signal is indirect.
    - "core": inside each period, a D run shorter than `CORE_MIN_S` becomes X, and so do
      `CORE_EDGE_S` at each edge of a longer run. Only the core of each stoppage stays D.
    - "interior" (E80): D becomes X, and so does every frame within `label_views.INTERIOR_S`
      seconds of a label change or without an observed ball. Only possession interiors stay, so
      an event-timed source teaches who has the ball and not when it changes hands.

    X is ignored by the loss. The input array is not modified.
    """
    if mode not in EXTRA_LABEL_MODES:
        raise ValueError(f"train.extra_labels must be one of {EXTRA_LABEL_MODES}, got {mode!r}")
    if mode == "all":
        return labels
    if mode == "none":  # label-free (E50): only label-free targets train on this match
        return np.full_like(labels, X)
    if mode == "interior":
        seen = ~np.isnan(bundle.tracking.ball_xy[:, 0])
        return interior_view(labels, bundle.meta, INTERIOR_S, seen=seen)
    out = labels.copy()
    if mode == "team":
        out[out == D] = X
        return out
    fps = bundle.meta.fps
    min_len = int(round(CORE_MIN_S * fps))
    edge = int(round(CORE_EDGE_S * fps))
    for period in bundle.meta.periods:
        seg = out[period.start : period.stop]
        hit = np.concatenate(([False], seg == D, [False]))
        bounds = np.flatnonzero(hit[1:] != hit[:-1])
        for start, stop in zip(bounds[::2], bounds[1::2], strict=True):
            if stop - start < min_len:
                seg[start:stop] = X
            else:
                seg[start : start + edge] = X
                seg[stop - edge : stop] = X
    return out


def _loader_keywords(load) -> set[str] | None:
    """Names the loader accepts, or None when it takes ``**kwargs``."""
    params = inspect.signature(load).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None
    return {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _extra_loader_kwargs(source: str, load, extra_source_kwargs: dict | None) -> dict:
    """Keyword arguments for one source. A key the loader does not accept raises."""
    if not extra_source_kwargs or source not in extra_source_kwargs:
        return {}
    raw = extra_source_kwargs[source]
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"train.extra_source_kwargs[{source!r}] must be a mapping, got {type(raw).__name__}"
        )
    accepted = _loader_keywords(load)
    if accepted is not None:
        unknown = sorted(key for key in raw if key not in accepted)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                f"extra_source_kwargs for {source!r} has keys the loader does not accept: {names}"
            )
    return dict(raw)


def _refuse_pff_test_split(entry: str, match_id: str) -> None:
    """Refuse a PFF id that `game_ids('test')` holds out. The message names `SPLIT_RULE`."""
    from ..sources.pff import SPLIT_RULE, game_ids

    if str(match_id) in game_ids("test"):
        raise ValueError(
            f"{entry} is in the PFF test split and cannot be used for training. "
            f"SPLIT_RULE: {SPLIT_RULE}"
        )


def load_extra_matches(
    entries: Sequence[str],
    spec: TokenSpec,
    *,
    labels: str = "all",
    extra_source_kwargs: dict | None = None,
) -> list[TrainMatch]:
    """Clean labelled matches from other sources, in the order given.

    Each entry is `source/match_id` and is loaded with `SOURCES.get(source).fn(match_id=...)`.
    `idsse` and `bar-ath` are refused by name. A `pff/<id>` whose id is in the PFF test
    split (`pff.game_ids('test')` on that id) is refused, and the message names `SPLIT_RULE`.
    `extra_source_kwargs` maps a source name to keyword arguments for its loader, for example
    `{pff: {variant: raw, labels: control_touch, quality: strict}}`. A key the loader does
    not accept raises. A bundle without labels is refused. The label array is the one
    `TrainMatch.from_bundle` stores, after `extra_label_view(labels)`, X included; the loss
    ignores X. `build_set(..., extra_degrade=False)` leaves these matches' tracking as the
    loader returned it.
    """
    if labels not in EXTRA_LABEL_MODES:
        raise ValueError(f"train.extra_labels must be one of {EXTRA_LABEL_MODES}, got {labels!r}")
    if extra_source_kwargs is not None and not isinstance(extra_source_kwargs, dict):
        raise ValueError(
            "train.extra_source_kwargs must be a mapping of source name to keyword arguments, "
            f"got {type(extra_source_kwargs).__name__}"
        )
    loaded: list[TrainMatch] = []
    for entry in entries:
        source, match_id = split_extra_entry(entry)
        try:
            load = SOURCES.get(source).fn
        except KeyError as exc:
            raise ValueError(f"unknown extra-match source {source!r} in {entry!r}") from exc
        # Most sources take `match_id`; the Metrica sample games take `game` (E46).
        key = "match_id" if "match_id" in inspect.signature(load).parameters else "game"
        kwargs = _extra_loader_kwargs(source, load, extra_source_kwargs)
        if key in kwargs:
            raise ValueError(
                f"extra_source_kwargs for {source!r} cannot set {key!r}; the entry supplies it"
            )
        if source == "pff":
            _refuse_pff_test_split(entry.strip(), match_id)
        bundle = load(**{**kwargs, key: match_id})
        if bundle.labels is None:
            raise ValueError(f"{entry}: training needs a labelled match")
        if labels != "all":
            bundle = replace(bundle, labels=extra_label_view(bundle.labels, bundle, labels))
        match = TrainMatch.from_bundle(bundle, spec, extra=True)
        if labels == "none":
            seen = ~np.isnan(match.ball_target[:, 0]) & ~np.isnan(bundle.tracking.ball_xy[:, 0])
            match = replace(match, target_ok=seen & match.played, label_free=True)
        loaded.append(match)
    return loaded


def include_extra_matches(
    matches: Sequence[TrainMatch],
    entries: Sequence[str],
    spec: TokenSpec,
    *,
    labels: str = "all",
    extra_source_kwargs: dict | None = None,
) -> list[TrainMatch]:
    """Append `entries` once. An empty list returns `matches` itself when it is already a list.

    A list that already carries the same number of `TrainMatch.extra` rows is returned as given,
    so a caller that appended them (every LOMO fold) does not load or append them again.
    `extra_source_kwargs` is forwarded to `load_extra_matches` on that first load.
    """
    roster = matches if isinstance(matches, list) else list(matches)
    if not entries:
        return roster
    flagged = sum(match.extra for match in roster)
    if flagged == 0:
        return [
            *roster,
            *load_extra_matches(
                entries, spec, labels=labels, extra_source_kwargs=extra_source_kwargs
            ),
        ]
    if flagged != len(entries):
        raise ValueError(
            f"{flagged} extra matches were passed in but train.extra_matches lists {len(entries)}"
        )
    return roster
