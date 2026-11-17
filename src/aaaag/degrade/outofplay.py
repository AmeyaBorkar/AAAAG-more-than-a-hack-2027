"""degrade-v4: the ATD does not show a dead ball parked beyond a line (issue #21).

Automatic tracking tends to show the ball beyond a line only in short, deep
flashes, and the band 0-0.5 m beyond a line -- where a ball waits for a throw-in or a corner --
almost never holds a sighting; the clean IDSSE ball sits there through every such stoppage (about
10% of frames). The camera is not the reason: it usually shows several metres beyond the nearest
line, and the ball the ATD does show is always inside the frame (measured label-free; the figures
stay private).

Rule: inside each dead segment (labels D), the ball is hidden from its first to its last frame
beyond a line (the hull, so jitter at the line cannot flicker it on and off), except for an exit
flash at the start of the hull whose length is drawn from the calibration match's off-pitch runs
that end in a gap (stats.exit_flash_lengths). The ball comes back when the hull ends, i.e. at or just before the
restart.

Signals: the (restaged) ball position and the IDSSE dead-ball labels. Where it fails: a ball that
goes out while alive and is played back in before the whistle is not hidden (rare: 1% of IDSSE
off-pitch frames are alive); the ATD's behaviour for balls beyond a line in open play (airborne
projections: most off-pitch runs on the target match come back on the pitch while seen) is left to
the on-pitch gap process.
"""

from __future__ import annotations

import numpy as np

from ..contracts import D, MatchBundle


def out_of_play_mask(
    bundle: MatchBundle, ball_xy: np.ndarray, flash_s: tuple[float, ...], rng
) -> np.ndarray:
    """(T,) bool, True where the dead ball beyond a line is hidden. `ball_xy` is the ball to use
    (v4 passes the restaged one); `flash_s` are the exit-flash lengths, sampled uniformly."""
    if bundle.labels is None:
        raise ValueError("degrade-v4 needs labels: the dead segments come from them")
    meta = bundle.meta
    with np.errstate(invalid="ignore"):
        beyond = (
            np.minimum(
                meta.pitch_length / 2 - np.abs(ball_xy[:, 0]),
                meta.pitch_width / 2 - np.abs(ball_xy[:, 1]),
            )
            < 0
        )
    flashes = np.maximum(0, np.rint(np.asarray(flash_s) * meta.fps)).astype(np.int64)
    mask = np.zeros(meta.n_frames, dtype=bool)
    for period in meta.periods:
        dead = bundle.labels[period.start : period.stop] == D
        edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(np.int8), [0]))))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            out = np.flatnonzero(beyond[period.start + start : period.start + stop])
            if len(out) == 0:
                continue
            first = period.start + start + int(out[0])
            last = period.start + start + int(out[-1])
            hidden_from = first + int(flashes[rng.integers(len(flashes))])
            mask[hidden_from : last + 1] = True
    return mask
