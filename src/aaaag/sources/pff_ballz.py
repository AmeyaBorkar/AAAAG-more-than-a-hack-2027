"""The PFF raw ball's height (z), which `pff.load_pff` does not keep, on a bundle's frame axis.

External data only (PFF FC World Cup 2022, PFF terms). This module mirrors `idsse_ballz` for
PFF. `ball_z(match_id, bundle)` reads `pff.ball_z_axis`: the z of the single VISIBLE raw
ball, resampled to 25 fps like its x and y. That axis is checked against the bundle's frames
and periods. The result is cached as one small .npz per game under
`cache_dir()/pff_ballz`, keyed by the tracking file's size, `pff.PARSER_VERSION` and
`pff.BALL_Z_VERSION`. The bundle caches are not touched. The function is thread-safe:
training builds matches on threads, and each game parses at most once per process.

Train split only. The 16 test-split games (`pff.SPLIT_RULE`) are refused, as
`load_extra_matches` refuses them. `train.airborne_extra` (E119's projection on extra
matches) and the stage-0a probe below are the only readers.

Stage-0a probe (`python -m aaaag.sources.pff_ballz --probe [--games N]`) prints aggregates
only, on the train split's visible-ball frames (raw ball seen, inside a period):
- the share of z above 1.5, 3 and 6 m, and z quantiles;
- lofted flights: runs of z above 3 m lasting at least 0.2 s (a ballistic ball with a 3.05 m
  apex spends 0.2 s above 3 m), with unseen stretches of at most 0.4 s inside a flight
  bridged. Rates are per played minute and per minute of seen ball;
- E114's ghost rate (E115's jump): steps between seen ball frames at most 1 s apart, faster
  than 40 m/s, per minute of seen ball, counted inside each period. It is printed for the raw
  ball, and for the raw ball projected through one camera at the halfway line, 20 m behind
  the near touchline, at 15, 18 and 25 m (E118's geometry and the physics range of
  `train.airborne_cz_*`).
- the decision, declared before the run (#86, B3 stage 0a): KILL if lofted flights are under
  0.5 per played minute, pooled over the games read; otherwise GO.

Where it fails: PFF's z is estimated from broadcast video, and its accuracy is not
documented. A game whose height axis does not match the bundle raises; nothing is guessed.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

from ..contracts import Period
from ..paths import cache_dir
from . import pff

FLIGHT_Z_M = 3.0  # a lofted ball: well above any player's head
FLIGHT_MIN_S = 0.2  # a ballistic ball with a 3.05 m apex spends 0.2 s above 3 m
BRIDGE_S = 0.4  # an unseen stretch this short inside a flight does not split it
LOFT_M = (1.5, 3.0, 6.0)
PROBE_CZ = (15.0, 18.0, 25.0)  # the physics camera range of train.airborne_cz_* and E118's 18 m
KILL_PER_MIN = 0.5  # declared stage-0a rule: fewer lofted flights per played minute is a KILL
_Z_BINS = np.arange(-2.0, 40.0 + 1e-9, 0.05)  # pooled histogram for the quantiles

_LOCK = threading.Lock()
_GAME_LOCKS: dict[tuple[str, str, str], threading.Lock] = {}
_MEMO: dict[tuple[str, str, str], tuple[np.ndarray, tuple]] = {}


def _guard(match_id, allow_test: bool) -> str:
    gid = pff._require_catalogue_id(str(match_id))
    if not allow_test and gid in set(pff.game_ids("test")):
        raise ValueError(
            f"PFF game {gid} is in the test split; its ball height is never read for "
            f"training. SPLIT_RULE: {pff.SPLIT_RULE}"
        )
    return gid


def _cache_file(gid: str) -> Path:
    size = pff.file_paths(gid)["tracking"].stat().st_size  # type: ignore[union-attr]
    name = f"{gid}_{size}_v{pff.PARSER_VERSION}z{pff.BALL_Z_VERSION}.npz"
    return cache_dir() / "pff_ballz" / name


def _read(path: Path) -> tuple[np.ndarray, tuple]:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=np.float32)
        periods = tuple(
            Period(int(index), int(start), int(stop)) for index, start, stop in data["periods"]
        )
    return z, periods


def _write(path: Path, z: np.ndarray, periods: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as handle:
        np.savez_compressed(
            handle,
            z=z,
            periods=np.array([(p.index, p.start, p.stop) for p in periods], dtype=np.int64),
        )
    part.replace(path)


def heights(match_id, *, allow_test: bool = False) -> tuple[np.ndarray, tuple]:
    """(z, periods): `pff.ball_z_axis`, cached on disk and in memory. Refuses the test split."""
    gid = _guard(match_id, allow_test)
    # The data and cache roots are part of the key, so a process that switches them (the tests)
    # does not reuse another tree's heights. Computing it scans no directory.
    key = (gid, str(pff.data_root()), str(cache_dir()))
    with _LOCK:
        lock = _GAME_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key in _MEMO:
            return _MEMO[key]
        path = _cache_file(gid)
        if path.is_file():
            got = _read(path)
        else:
            got = pff.ball_z_axis(gid)
            _write(path, *got)
        _MEMO[key] = got
        return got


def ball_z(match_id, bundle, *, allow_test: bool = False) -> np.ndarray:
    """(T,) float32 ball height in metres on `bundle`'s frame axis, NaN where there is none.

    The bundle must be a raw-variant PFF load of the same game: its frame count and periods
    must equal the height axis, or this raises. The 16 test-split games raise unless
    `allow_test` (label-free evaluation tooling only, never training).
    """
    provenance = (getattr(bundle, "extras", None) or {}).get("provenance") or {}
    variant = provenance.get("variant", "raw")
    if variant != "raw":
        raise ValueError(
            f"PFF game {match_id}: the height is the raw ball's, and this bundle is variant "
            f"{variant!r}"
        )
    z, periods = heights(match_id, allow_test=allow_test)
    got = [(p.index, p.start, p.stop) for p in periods]
    want = [(p.index, p.start, p.stop) for p in bundle.meta.periods]
    if len(z) != bundle.meta.n_frames or got != want:
        raise ValueError(
            f"PFF game {match_id}: the height axis ({len(z)} frames, periods {got}) does not "
            f"match the bundle ({bundle.meta.n_frames} frames, periods {want})"
        )
    return z


# -- stage-0a probe (aggregates only) ---------------------------------------------------------


def flight_durations(z: np.ndarray, periods, fps: int) -> np.ndarray:
    """Durations (s) of lofted flights: runs of z above `FLIGHT_Z_M` inside one period.

    Two runs are one flight when every frame between them is unseen (NaN) and the gap is at
    most `BRIDGE_S`. A flight counts when it has at least `FLIGHT_MIN_S` of frames above the
    height. Its duration runs from its first to its last frame above the height.
    """
    min_frames = int(round(FLIGHT_MIN_S * fps))
    bridge = int(round(BRIDGE_S * fps))
    out: list[float] = []
    for period in periods:
        seg = np.asarray(z[period.start : period.stop], dtype=np.float64)
        known = np.isfinite(seg)
        with np.errstate(invalid="ignore"):
            above = known & (seg > FLIGHT_Z_M)
        edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if not len(starts):
            continue
        first, last, count = int(starts[0]), int(stops[0]), int(stops[0] - starts[0])
        for start, stop in zip(starts[1:].tolist(), stops[1:].tolist(), strict=True):
            if start - last <= bridge and not known[last:start].any():
                last, count = stop, count + stop - start
                continue
            if count >= min_frames:
                out.append((last - first) / fps)
            first, last, count = start, stop, stop - start
        if count >= min_frames:
            out.append((last - first) / fps)
    return np.asarray(out, dtype=np.float64)


def jump_pairs(ball_xy: np.ndarray, periods, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """(from, to) frames of E114's ghost jumps: seen frames at most 1 s apart, over 40 m/s."""
    from ..features.ball_flight import FAST_MS, GAP_S

    ball = np.asarray(ball_xy, dtype=np.float64)
    lo: list[np.ndarray] = []
    hi: list[np.ndarray] = []
    for period in periods:
        seen = np.flatnonzero(np.isfinite(ball[period.start : period.stop, 0])) + period.start
        if len(seen) < 2:
            continue
        gap = np.diff(seen)
        step = np.linalg.norm(ball[seen[1:]] - ball[seen[:-1]], axis=1)
        fast = (gap <= GAP_S * fps) & (step / (gap / fps) > FAST_MS)
        lo.append(seen[:-1][fast])
        hi.append(seen[1:][fast])
    if not lo:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(lo), np.concatenate(hi)


def probe_bundle(bundle, z: np.ndarray) -> dict:
    """Aggregates of one game. Counts and minutes, never a frame-level value."""
    from ..ml.airborne import BEHIND_M, project

    meta = bundle.meta
    fps = meta.fps
    ball = bundle.tracking.ball_xy
    played = meta.played_mask()
    visible = np.isfinite(ball[:, 0]) & played
    zv = np.asarray(z, dtype=np.float64)[visible]
    with_z = np.isfinite(zv)
    zz = zv[with_z]
    hist, _ = np.histogram(np.clip(zz, _Z_BINS[0], _Z_BINS[-1]), bins=_Z_BINS)
    durations = flight_durations(np.where(visible, z, np.nan), meta.periods, fps)
    lo, hi = jump_pairs(np.where(visible[:, None], ball, np.nan), meta.periods, fps)
    with np.errstate(invalid="ignore"):
        lofted = (np.asarray(z)[lo] > FLIGHT_Z_M) | (np.asarray(z)[hi] > FLIGHT_Z_M)
    projected = {}
    for cz in PROBE_CZ:
        camera = np.array([0.0, -(meta.pitch_width / 2.0 + BEHIND_M), cz])
        moved = project(np.where(visible[:, None], ball, np.nan), z, camera)
        projected[cz] = len(jump_pairs(moved, meta.periods, fps)[0])
    return {
        "played_min": float(played.sum()) / fps / 60.0,
        "visible_min": float(visible.sum()) / fps / 60.0,
        "visible": int(visible.sum()),
        "with_z": int(with_z.sum()),
        "above": {t: int((zz > t).sum()) for t in LOFT_M},
        "below_0": int((zz < 0.0).sum()),
        "above_30": int((zz > 30.0).sum()),
        "hist": hist.astype(np.int64),
        "flights": durations,
        "jumps_raw": int(len(lo)),
        "jumps_lofted": int(lofted.sum()),
        "jumps_projected": projected,
    }


def _hist_quantile(hist: np.ndarray, q: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return float("nan")
    k = int(np.searchsorted(np.cumsum(hist), q * total, side="left"))
    k = min(k, len(hist) - 1)
    return float((_Z_BINS[k] + _Z_BINS[k + 1]) / 2.0)


def _rate(n: float, minutes: float) -> float:
    return float(n / minutes) if minutes > 0 else float("nan")


def format_probe(rows: list[dict], errors: list[str]) -> str:
    """The stage-0a report. Pooled aggregates plus the spread across games."""
    lines = [
        f"pff_ballz probe, train split, aggregates only: games={len(rows)} errors={len(errors)}"
    ]
    lines.extend(f"  error {text}" for text in errors)
    if not rows:
        lines.append("decision (KILL if lofted flights < 0.5 per played minute): no game read")
        return "\n".join(lines) + "\n"
    played = sum(r["played_min"] for r in rows)
    seen_min = sum(r["visible_min"] for r in rows)
    visible = sum(r["visible"] for r in rows)
    with_z = sum(r["with_z"] for r in rows)
    hist = np.sum([r["hist"] for r in rows], axis=0)
    flights = np.concatenate([r["flights"] for r in rows])
    per_game = np.array([_rate(len(r["flights"]), r["played_min"]) for r in rows])
    jumps = sum(r["jumps_raw"] for r in rows)
    lofted = sum(r["jumps_lofted"] for r in rows)
    shares = " ".join(
        f"above_{t:g}m={sum(r['above'][t] for r in rows) / max(with_z, 1):.4f}" for t in LOFT_M
    )
    quantiles = " ".join(
        f"p{q * 100:g}={_hist_quantile(hist, q):.2f}" for q in (0.5, 0.9, 0.99, 0.999)
    )
    rate = _rate(len(flights), played)
    lines += [
        f"  minutes played={played:.1f} seen_ball={seen_min:.1f} "
        f"visible_frames={visible} z_coverage={with_z / max(visible, 1):.4f}",
        f"  z_share {shares}",
        f"  z_m {quantiles} below_0={sum(r['below_0'] for r in rows) / max(with_z, 1):.4f} "
        f"above_30={sum(r['above_30'] for r in rows) / max(with_z, 1):.4f}",
        f"  flights_z>{FLIGHT_Z_M:g}m n={len(flights)} per_played_min={rate:.3f} "
        f"per_seen_min={_rate(len(flights), seen_min):.3f}",
    ]
    if len(flights):
        q = np.percentile(flights, [25, 50, 75, 90])
        lines.append(
            "  flight_s "
            + " ".join(f"p{p}={v:.2f}" for p, v in zip((25, 50, 75, 90), q, strict=True))
        )
    spread = np.percentile(per_game, [0, 25, 50, 75, 100])
    lines.append(
        "  flights_per_played_min_by_game "
        + " ".join(
            f"{name}={v:.3f}"
            for name, v in zip(("min", "p25", "p50", "p75", "max"), spread, strict=True)
        )
        + f" games_under_{KILL_PER_MIN:g}={int((per_game < KILL_PER_MIN).sum())}"
    )
    lines.append(
        f"  jumps_per_seen_min raw={_rate(jumps, seen_min):.3f} "
        f"lofted_share={lofted / jumps if jumps else float('nan'):.3f} "
        + " ".join(
            f"projected_cz{cz:g}={_rate(sum(r['jumps_projected'][cz] for r in rows), seen_min):.3f}"
            for cz in PROBE_CZ
        )
    )
    verdict = "GO" if rate >= KILL_PER_MIN else "KILL"
    lines.append(
        f"decision (declared: KILL if lofted flights < {KILL_PER_MIN:g} per played minute): {verdict}"
    )
    return "\n".join(lines) + "\n"


def probe(ids: list[str], *, labels: str = "control") -> str:
    """Load each train-split id (raw variant) with its height and return the report."""
    rows: list[dict] = []
    errors: list[str] = []
    for gid in ids:
        gid = _guard(gid, allow_test=False)
        try:
            bundle = pff.load_pff(match_id=gid, variant="raw", labels=labels)
            rows.append(probe_bundle(bundle, ball_z(gid, bundle)))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{gid} {type(exc).__name__}")
        print(f"pff_ballz {len(rows) + len(errors)}/{len(ids)} read", file=sys.stderr, flush=True)
    return format_probe(rows, errors)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PFF ball height: the stage-0a probe on the train split (aggregates only)."
    )
    parser.add_argument("--probe", action="store_true", help="print the stage-0a aggregates")
    parser.add_argument("--games", default=None, help="'all' or the first N train ids")
    parser.add_argument(
        "--labels",
        default="control",
        choices=("sequence", "control", "control_touch"),
        help="labels of the bundle load (it only sets which bundle cache is read)",
    )
    args = parser.parse_args(argv)
    if not args.probe:
        parser.print_help()
        raise SystemExit(2)
    ids = pff._limit_ids(pff.game_ids("train", installed_only=True), args.games)
    sys.stdout.write(probe(ids, labels=args.labels))


if __name__ == "__main__":
    main()
