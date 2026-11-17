"""Synthetic matches with exact labels: for tests, CI, and building stages without the real data.

This is not a football simulator. It reproduces the *structure* the pipeline must survive --
possession spells with a realistic spread of lengths (including sub-second ones), dead-ball
stretches with the ball leaving the pitch, a defender pressing the carrier, and ATD-like damage:
bursty ball dropout (worse during stoppages), anonymous fragmented track slots, positional jitter.
Everything is seeded, so a seed is a fixed match.

Do not select models on synthetic scores: realism is deliberately low. Model selection uses other
real matches, degraded to look like the ATD (docs/VALIDATION.md). Synthetic data answers "does it
run, is it deterministic, does it honour the contracts, does an obvious signal get picked up".
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from .contracts import AWAY, HOME, D, MatchBundle, MatchMeta, Period, Tracking, X

# 4-4-2 shape for a team attacking towards +x, metres from the team centre; index 0 is the keeper.
FORMATION = np.array(
    [
        [-40.0, 0.0],
        [-22.0, -20.0], [-24.0, -7.0], [-24.0, 7.0], [-22.0, 20.0],
        [-6.0, -22.0], [-8.0, -7.0], [-8.0, 7.0], [-6.0, 22.0],
        [8.0, -8.0], [8.0, 8.0],
    ]
)  # fmt: skip
N_PER_TEAM = len(FORMATION)


@dataclass(frozen=True)
class SyntheticConfig:
    minutes_per_half: float = 10.0
    gap_minutes: float = 1.0  # ungraded padding before, between and after the halves
    fps: int = 25
    pitch: tuple[float, float] = (105.0, 68.0)
    mean_possession_s: float = 12.0
    min_possession_s: float = 0.6
    mean_dead_s: float = 14.0
    p_stoppage: float = 0.3  # chance a possession ends in a dead ball rather than a turnover
    ball_missing_rate: float = 0.4
    ball_mean_gap_s: float = 1.0
    dead_extra_missing_rate: float = 0.5
    n_slots: int = 46
    mean_track_life_s: float = 40.0
    player_dropout: float = 0.02
    pos_noise_m: float = 0.15
    ball_noise_m: float = 0.3


def generate(seed: int = 0, config: SyntheticConfig | None = None) -> MatchBundle:
    """One synthetic match with labels. Same seed and config -> identical arrays."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed)
    meta = _layout(cfg, seed)
    segments = _segments(meta, cfg, rng)
    labels = np.full(meta.n_frames, X, dtype=np.int8)
    for start, stop, code in segments:
        labels[start:stop] = code
    ball, dribbles = _ball_path(meta, segments, rng)
    players = _players(meta, labels, ball, dribbles, rng)
    tracking = _observe(meta, cfg, labels, players, ball, rng)
    carrier = np.full(meta.n_frames, -1, dtype=np.int8)
    for start, stop, team in dribbles:
        carrier[start:stop] = team
    return MatchBundle(
        tracking=tracking,
        labels=labels,
        extras={"carrier_team": carrier, "true_ball_xy": ball.astype(np.float32)},
    ).validate()


# -- structure ------------------------------------------------------------------------------------


def _layout(cfg: SyntheticConfig, seed: int) -> MatchMeta:
    half = int(round(cfg.minutes_per_half * 60 * cfg.fps))
    gap = int(round(cfg.gap_minutes * 60 * cfg.fps))
    first = Period(1, gap, gap + half)
    second = Period(2, first.stop + gap, first.stop + gap + half)
    return MatchMeta(
        match_id=f"SYN-{seed}",
        source="synthetic",
        n_frames=second.stop + gap,
        fps=cfg.fps,
        pitch_length=cfg.pitch[0],
        pitch_width=cfg.pitch[1],
        periods=(first, second),
        home="Synthetic Home",
        away="Synthetic Away",
    ).validate()


def _segments(
    meta: MatchMeta, cfg: SyntheticConfig, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """[(start, stop, code)] covering every period: a kickoff stoppage, then spells and stoppages."""
    fps = meta.fps
    first_team = int(rng.integers(2))
    out = []
    for period in meta.periods:
        t = period.start
        stop = t + int(rng.integers(fps, 4 * fps))
        out.append((t, stop, D))
        t = stop
        team = first_team if period.index == 1 else 1 - first_team
        while t < period.stop:
            seconds = max(cfg.min_possession_s, rng.gamma(1.3, cfg.mean_possession_s / 1.3))
            stop = min(period.stop, t + max(1, int(round(seconds * fps))))
            out.append((t, stop, team))
            t = stop
            if t >= period.stop:
                break
            if rng.random() < cfg.p_stoppage:
                seconds = max(3.0, rng.gamma(3.0, cfg.mean_dead_s / 3.0))
                stop = min(period.stop, t + int(round(seconds * fps)))
                out.append((t, stop, D))
                t = stop
                team = int(rng.integers(2))
            else:
                team = 1 - team
    return out


def _attack_sign(meta: MatchMeta, frame: int, team: int) -> float:
    """+1 if `team` attacks towards +x at `frame`. Home attacks +x in the first period."""
    period = next(p for p in meta.periods if p.start <= frame < p.stop)
    home_sign = 1.0 if period.index % 2 == 1 else -1.0
    return home_sign if team == HOME else -home_sign


def _ball_path(meta, segments, rng) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """True ball positions (T, 2) (NaN outside periods) and the dribble intervals [(start, stop, team)]."""
    fps = meta.fps
    half_l, half_w = meta.pitch_length / 2, meta.pitch_width / 2
    inner = np.array([half_l - 1.0, half_w - 1.0])
    keys: dict[int, np.ndarray] = {}
    dribbles = []
    pos = np.zeros(2)
    period_starts = {p.start for p in meta.periods}

    for start, stop, code in segments:
        if code == D:
            if start in period_starts:  # kickoff: ball on the centre spot
                pos = np.zeros(2)
                keys[start], keys[stop - 1] = pos, pos
                continue
            if stop - start < 2 * fps:  # stoppage cut short by the end of the period
                keys[start], keys[stop - 1] = pos, pos
                continue
            if abs(pos[1]) / half_w > abs(pos[0]) / half_l:
                out = np.array([pos[0], np.sign(pos[1] or 1.0) * (half_w + 2.0)])
            else:
                out = np.array([np.sign(pos[0] or 1.0) * (half_l + 2.0), pos[1]])
            restart = np.clip(out, [-half_l, -half_w], [half_l, half_w])
            t_out = min(stop - 3, start + fps // 2)
            t_place = max(t_out + 2, stop - fps)
            keys[start], keys[t_out], keys[t_place - 1] = pos, out, out
            keys[t_place], keys[stop - 1] = restart, restart
            pos = restart
            continue

        sign = _attack_sign(meta, start, code)
        t = start
        keys[t] = pos
        while t < stop:
            end = min(stop, t + max(1, int(rng.uniform(0.6, 2.5) * fps)))
            heading = np.array([sign * rng.uniform(0.3, 1.0), rng.uniform(-0.7, 0.7)])
            heading /= np.linalg.norm(heading)
            pos = np.clip(pos + heading * rng.uniform(2.0, 5.0) * (end - t) / fps, -inner, inner)
            dribbles.append((t, end, code))
            keys[end] = pos
            t = end
            if t >= stop:
                break
            heading = np.array([sign * rng.uniform(-0.3, 1.0), rng.uniform(-1.0, 1.0)])
            heading /= np.linalg.norm(heading)
            target = np.clip(pos + heading * rng.uniform(8.0, 30.0), -inner, inner)
            flight = max(2, int(np.linalg.norm(target - pos) / rng.uniform(12.0, 22.0) * fps))
            end = min(stop, t + flight)
            pos = pos + (target - pos) * (end - t) / flight
            keys[end] = pos
            t = end

    frames = np.array(sorted(keys))
    values = np.array([keys[f] for f in frames])
    every = np.arange(meta.n_frames)
    ball = np.stack(
        [np.interp(every, frames, values[:, 0]), np.interp(every, frames, values[:, 1])], axis=1
    )
    ball[~meta.played_mask()] = np.nan
    return ball, dribbles


def _players(meta, labels, ball, dribbles, rng) -> np.ndarray:
    """True positions (T, 22, 2): home 0..10, away 11..21. NaN outside periods."""
    T, fps = meta.n_frames, meta.fps
    played = meta.played_mask()
    home_sign = np.zeros(T)
    for period in meta.periods:
        home_sign[period.start : period.stop] = 1.0 if period.index % 2 == 1 else -1.0

    alpha = 1.0 / fps  # the team block follows the ball with a ~1 s lag
    follow = lfilter([alpha], [1.0, alpha - 1.0], np.nan_to_num(ball), axis=0)
    pos = np.zeros((T, 2 * N_PER_TEAM, 2))
    for team in (HOME, AWAY):
        sign = home_sign if team == HOME else -home_sign
        push = lfilter([alpha], [1.0, alpha - 1.0], np.where(labels == team, 6.0, -4.0))
        cols = slice(team * N_PER_TEAM, (team + 1) * N_PER_TEAM)
        pos[:, cols, 0] = (0.55 * follow[:, 0] + sign * push)[:, None] + sign[:, None] * FORMATION[
            None, :, 0
        ]
        pos[:, cols, 1] = 0.35 * follow[:, 1][:, None] + FORMATION[None, :, 1]
        keeper = team * N_PER_TEAM
        pos[:, keeper, 0] = -sign * (meta.pitch_length / 2 - 6.0) + 0.08 * follow[:, 0]
        pos[:, keeper, 1] = 0.15 * follow[:, 1]

    wander = lfilter([0.05], [1.0, -0.95], rng.normal(0.0, 1.0, pos.shape), axis=0)
    pos += 2.5 * wander / max(1e-9, float(wander.std()))

    for start, stop, team in dribbles:
        mates = np.arange(team * N_PER_TEAM + 1, (team + 1) * N_PER_TEAM)
        rivals = np.arange((1 - team) * N_PER_TEAM + 1, (2 - team) * N_PER_TEAM)
        sign = _attack_sign(meta, start, team)
        carrier = mates[np.argmin(np.linalg.norm(pos[start, mates] - ball[start], axis=1))]
        presser = rivals[np.argmin(np.linalg.norm(pos[start, rivals] - ball[start], axis=1))]
        pos[start:stop, carrier] = ball[start:stop] + np.array([-0.7 * sign, 0.0])
        pos[start:stop, presser] = ball[start:stop] + np.array([2.5 * sign, rng.uniform(-1.0, 1.0)])

    pos[~played] = np.nan
    return pos


# -- observation damage -----------------------------------------------------------------------------


def _bursty_mask(rng: np.random.Generator, n: int, rate: float, mean_run: float) -> np.ndarray:
    """(n,) bool, True on about `rate` of frames, in runs of mean length `mean_run` frames."""
    out = np.zeros(n, dtype=bool)
    if rate <= 0.0:
        return out
    mean_on = max(1.0, mean_run)
    mean_off = max(1.0, mean_on * (1.0 - rate) / rate)
    t = int(rng.integers(0, int(mean_off) + 1))
    while t < n:
        on = int(rng.geometric(1.0 / mean_on))
        out[t : t + on] = True
        t += on + int(rng.geometric(1.0 / mean_off))
    return out


def _observe(meta, cfg, labels, players, ball, rng) -> Tracking:
    T, fps, S = meta.n_frames, meta.fps, cfg.n_slots
    per_team = S // 2
    slot_team = np.array([HOME] * per_team + [AWAY] * (S - per_team), dtype=np.int8)

    speed = np.zeros(players.shape[:2])
    speed[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    speed = lfilter([0.2], [1.0, -0.8], np.nan_to_num(speed), axis=0)
    noisy = players + rng.normal(0.0, cfg.pos_noise_m, players.shape)

    fragments = []
    for period in meta.periods:
        for player in range(2 * N_PER_TEAM):
            t = period.start
            while t < period.stop:
                life = max(fps, int(rng.exponential(cfg.mean_track_life_s * fps)))
                fragments.append((t, min(period.stop, t + life), player))
                t += life
    fragments.sort()

    slot_xy = np.full((T, S, 2), np.nan, dtype=np.float32)
    slot_speed = np.full((T, S), np.nan, dtype=np.float32)
    busy_until = np.zeros(S, dtype=np.int64)
    present = np.stack(
        [~_bursty_mask(rng, T, cfg.player_dropout, fps / 2) for _ in range(2 * N_PER_TEAM)], axis=1
    )
    for start, stop, player in fragments:
        team = HOME if player < N_PER_TEAM else AWAY
        pool = np.flatnonzero((slot_team == team) & (busy_until <= start))
        slot = int(pool[rng.integers(len(pool))])
        busy_until[slot] = stop
        keep = present[start:stop, player]
        rows = np.arange(start, stop)[keep]
        slot_xy[rows, slot] = noisy[rows, player]
        slot_speed[rows, slot] = speed[rows, player]

    ball_obs = ball + rng.normal(0.0, cfg.ball_noise_m, ball.shape)
    missing = _bursty_mask(rng, T, cfg.ball_missing_rate, cfg.ball_mean_gap_s * fps)
    missing |= _bursty_mask(rng, T, cfg.dead_extra_missing_rate, 2 * fps) & (labels == D)
    ball_obs[missing] = np.nan

    return Tracking(
        meta=meta,
        player_xy=slot_xy,
        slot_team=slot_team,
        ball_xy=ball_obs.astype(np.float32),
        player_speed=slot_speed,
    ).validate()


# -- EPTS writer (parser tests) ---------------------------------------------------------------------


def write_epts(tracking: Tracking, directory: str | Path, stem: str = "SYN") -> tuple[Path, Path]:
    """Write `tracking` as a Metrica-style FIFA/EPTS pair (<stem>_atd.xml, <stem>_atd.txt).

    Mirrors the layout the real export is documented to have (one 'x,y,speed' group per slot,
    'NaN' for missing, frame ids from 1), so the parser can be tested end to end without the data.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = tracking.meta
    L, W = meta.pitch_length, meta.pitch_width
    xn = tracking.player_xy[..., 0] / L + 0.5
    yn = tracking.player_xy[..., 1] / W + 0.5
    speed = (
        tracking.player_speed if tracking.player_speed is not None else np.full(xn.shape, np.nan)
    )
    bx = tracking.ball_xy[:, 0] / L + 0.5
    by = tracking.ball_xy[:, 1] / W + 0.5
    bz = tracking.ball_extra.get("z", np.full(meta.n_frames, np.nan))

    def group(*values: float) -> str:
        if np.isnan(values[0]):
            return ",".join("NaN" for _ in values)
        return ",".join("NaN" if np.isnan(v) else f"{v:.5f}" for v in values)

    raw_path = directory / f"{stem}_atd.txt"
    with open(raw_path, "w", encoding="ascii", newline="\n") as handle:
        for t in range(meta.n_frames):
            slots = ";".join(
                group(xn[t, s], yn[t, s], speed[t, s]) for s in range(tracking.n_slots)
            )
            handle.write(f"{t + 1}:{slots}:{group(bx[t], by[t], bz[t])}\n")

    root = ET.Element("main")
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(ET.SubElement(metadata, "GlobalConfig"), "FrameRate").text = str(meta.fps)
    teams = ET.SubElement(metadata, "Teams")
    for team_id, name in (("T-HOME", meta.home), ("T-AWAY", meta.away)):
        ET.SubElement(ET.SubElement(teams, "Team", id=team_id), "Name").text = name
    players = ET.SubElement(metadata, "Players")
    for slot, team in enumerate(tracking.slot_team, start=1):
        player = ET.SubElement(
            players, "Player", id=f"P{slot}", teamId="T-HOME" if team == HOME else "T-AWAY"
        )
        ET.SubElement(player, "Name").text = f"Track {slot}"
        ET.SubElement(player, "ShirtNumber").text = str(slot)
    specs = ET.SubElement(
        ET.SubElement(root, "DataFormatSpecifications"), "DataFormatSpecification"
    )
    ET.SubElement(specs, "StringRegister", name="frameCount")
    all_slots = ET.SubElement(specs, "SplitRegister", separator=";")
    channels = ET.SubElement(root, "PlayerChannels")
    for slot in range(1, tracking.n_slots + 1):
        one = ET.SubElement(all_slots, "SplitRegister", separator=",")
        for channel in ("x", "y", "speed"):
            ET.SubElement(one, "PlayerChannelRef", playerChannelId=f"player{slot}_{channel}")
            ET.SubElement(
                channels,
                "PlayerChannel",
                id=f"player{slot}_{channel}",
                playerId=f"P{slot}",
                channelId=channel,
            )
    ball_register = ET.SubElement(specs, "SplitRegister", separator=",")
    for channel in ("x", "y", "z"):
        ET.SubElement(ball_register, "BallChannelRef", channelId=channel)
    meta_path = directory / f"{stem}_atd.xml"
    ET.ElementTree(root).write(meta_path, encoding="utf-8", xml_declaration=True)
    return meta_path, raw_path
