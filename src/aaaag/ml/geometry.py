"""Label-free orientation, and the symmetries of the pitch that keep a label valid.

The tracking is delivered with the teams switching ends at half-time and nothing saying so. The
per-period orientation is read from the slot geometry (`home_defends_sign`: each team's centroid
sits towards its own goal), never from labels, so it works on the competition tracking too.

Symmetries on an attack-normalised token tensor (home attacks +x):
- `flip_y`: y -> -y. The label is unchanged.
- `swap`: the teams exchange roles. The pitch is rotated by 180 degrees (so the new home team
  attacks +x again), the home and away blocks trade places and the labels H and A trade places.
  When control memory is on, the ball token's controller column changes sign with the teams; its
  age does not, and a y-flip changes neither column. Contact columns, when on, sit after those:
  TOUCH_TEAM and CTRL_NOW change sign under a team swap (like CTRL_TEAM); BALL_DV and ACCEL do
  not, and a y-flip changes none of the four. TOUCH_DV, when the touch detector is g2l, sits
  after those four and before the path columns: a team swap does not negate it, and a y-flip
  changes neither it nor TOUCH_TEAM. Path columns, when on, sit after that: a team swap negates
  PATH_SIGN; PATH_OUT, PATH_T and PPCF do not change (player tokens move with their blocks), and
  a y-flip changes none of the four. Gap-bracket columns, when on, come last: a team swap negates
  all six (both teams, both positions), and a y-flip negates the two Y columns.
  Together the two symmetries generate the four views used for augmentation and test-time averaging.
`symmetric` works on numpy arrays and on torch tensors, and always returns a copy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import Tracking
from ..evidence.ballstate.cues import home_defends_sign
from .spec import (
    FLIP_Y_COLUMNS,
    ROTATE_COLUMNS,
    Col,
    TokenSpec,
    bracket_columns,
    contact_columns,
    path_columns,
)


def attack_sign(tracking: Tracking) -> np.ndarray:
    """(T,) float32 multiplier s such that s * (x, y) has the home team attacking towards +x.

    +1 or -1 inside each period; 1 outside the periods and wherever the orientation cannot be read
    (a period without a frame where both teams are seen).
    """
    defends = home_defends_sign(tracking)
    sign = np.ones(tracking.n_frames, dtype=np.float32)
    for period in tracking.meta.periods:
        home_defends = defends.get(period.index, 0.0)
        if home_defends:
            sign[period.start : period.stop] = -home_defends
    return sign


def team_swap_permutation(spec: TokenSpec) -> list[int]:
    """Token order after a team swap: the away block first, then the home block, then the ball."""
    k = spec.tokens_per_team
    return [*range(k, 2 * k), *range(k), 2 * k]


def _copy(array: Any) -> Any:
    return array.copy() if isinstance(array, np.ndarray) else array.clone()


def symmetric(tokens: Any, spec: TokenSpec, *, flip_y: bool = False, swap: bool = False) -> Any:
    """The tokens of the mirrored match: `tokens` is (..., n_tokens, N_FEATURES), a copy is returned.

    With `swap` the caller must also exchange the H and A labels (`swap_labels`) or the H and A
    logits (`SWAP_CLASSES`).
    """
    k = spec.tokens_per_team
    out = tokens[..., team_swap_permutation(spec), :] if swap else _copy(tokens)
    if swap:
        for column in ROTATE_COLUMNS:
            out[..., column] = -out[..., column]
        out[..., :k, Col.IS_AWAY] = 0.0
        out[..., k : 2 * k, Col.IS_AWAY] = 1.0
        if spec.control_memory:
            out[..., Col.CTRL_TEAM] = -out[..., Col.CTRL_TEAM]
        if spec.contact_inputs:
            columns = contact_columns(spec)
            # TOUCH_TEAM and CTRL_NOW. TOUCH_DV is a speed: it stays, and a y-flip leaves it too.
            for column in (columns[1], columns[2]):
                out[..., column] = -out[..., column]
        if spec.ball_path_control:
            path_sign = path_columns(spec)[0]
            out[..., path_sign] = -out[..., path_sign]
        if spec.gap_bracket:
            for column in bracket_columns(spec):  # the teams trade signs, the pitch turns 180
                out[..., column] = -out[..., column]
    if flip_y:
        for column in FLIP_Y_COLUMNS:
            out[..., column] = -out[..., column]
        if spec.gap_bracket:
            bracket = bracket_columns(spec)
            for column in (bracket[3], bracket[5]):
                out[..., column] = -out[..., column]
    return out


SWAP_CLASSES = [1, 0, 2]  # index with it to exchange the H and A entries of a (..., 3) array


def swap_labels(labels: np.ndarray) -> np.ndarray:
    """H <-> A, D and X unchanged."""
    return np.where(labels == 0, 1, np.where(labels == 1, 0, labels)).astype(labels.dtype)
