"""The token layout every part of the harness agrees on.

One frame becomes `2 * tokens_per_team + 1` tokens: the home block, the away block, then the ball.
Each token has `N_FEATURES` columns (`Col`). A model is free to ignore the layout (the team and
ball flags are columns too), but the shipped one reads it.

Conventions:
- Positions are rotated per period so that the home team attacks towards +x (label-free, from the
  slot geometry: geometry.py), then divided by the half pitch (x, y in about [-1, 1]).
- Velocities, offsets and speeds are in physical scales (`velocity_scale` m/s, `ref_scale_m` m).
- A token slot with nobody in it is all zeros with VALID = 0, and a team's tokens are sorted by
  distance to the reference point (the ball where seen, else the team centroid). The order carries
  no slot identity: slots are anonymous fragments on the competition tracking.
- `control_memory` appends two ball-token columns, who last controlled the ball and how long ago.
  Off, the tensor is the 14 columns below. Checkpoints written before those spec keys existed still
  load, and `FEATURE_VERSION` stays "1".
- `contact_inputs` appends four columns after the control-memory pair (or after the base 14 when
  that flag is off): ball speed-change, a decayed touch-team sign, the live control flag, and
  player acceleration. Off, the width and the values stay exactly the layout without the flag.
  `contact_touch_w` (default 2) is the touch-detector window and `contact_dv_w` (default 5) is the
  speed-change window. A spec without those keys keeps the defaults, so the tensor is unchanged.
  `touch_detector` (default "w2") picks that windowed detector. "g2l" is the guarded two-line
  impulse, and only then, with contact on, one more column follows the four: the decayed impulse
  `TOUCH_DV`. A spec without the key stays "w2", `FEATURE_VERSION` stays "1", and the tensor is
  unchanged.
- `ball_path_control` appends four columns after those contact columns (and after `TOUCH_DV` when
  that column is present, after the control-memory pair when contact is off, after the base 14
  when both are off): signed control along the ball's path, the probability the ball leaves or is
  still free at the horizon, the expected time of control, and each player's share. Off, the width
  and the values stay exactly the layout without the flag.
- `gap_bracket` (#33) appends six ball-token columns after everything above (after the path
  columns when those are on, else after whichever block is last): while the ball is unseen, the
  controlling team (+1 home, -1 away, 0 unknown) at the last sighting and from the next sighting
  on, and the attack-normalised ball position at the last and at the next sighting. All six are
  zero where the ball is seen, outside the periods and on a side with no sighting in the period.
  Off, the width and the values stay exactly the layout without the flag.
- `slot_index` appends two player-token columns after everything above, for
  `model.slot_temporal` only: `SLOT_ID` is 1 + the token's tracker slot (its column in
  `Tracking.player_xy`, the slot the distance argsort picked) and `SLOT_RUN` is 1 + the index of
  that slot's current observed run modulo `SLOT_RUN_MOD`. A run starts where the slot is observed
  after a frame where it was not, and at every period start. Both are 0 on invalid tokens and on
  the ball. They are identifiers, not features: the network never embeds them, and the symmetries
  leave them alone. Integers up to 2048 are exact in the float16 training tensors. Off, the width
  and the values stay exactly the layout without the flag, and `to_dict` leaves the key out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FEATURE_VERSION = "1"  # bump when a column changes meaning: checkpoints record it


class Col:
    """Column indices of a token."""

    X = 0  # x / half pitch length, attack-normalised
    Y = 1  # y / half pitch width
    VX = 2  # smoothed velocity / velocity_scale
    VY = 3
    SPEED = 4  # |v| / velocity_scale
    IS_AWAY = 5  # 0 home block, 1 away block, 0 for the ball
    IS_BALL = 6
    VALID = 7  # 1 when the slot (or the ball) is observed at this frame
    DX = 8  # (x - reference) / ref_scale_m, players only
    DY = 9
    DIST = 10  # distance to the reference / ref_scale_m, capped at 3
    REF_BALL = 11  # 1 when the reference is the seen ball, 0 when it is the team centroid
    GAP_BEFORE = 12  # ball token: log time since the ball was last seen, in [0, 1]
    GAP_AFTER = 13  # ball token: log time until it is next seen, in [0, 1]
    CTRL_TEAM = 14  # ball token, only with control_memory: +1 home, -1 away, 0 unknown
    CTRL_AGE = 15  # ball token, only with control_memory: log age of that run, in [0, 1]
    # BALL_DV, TOUCH_TEAM, CTRL_NOW, ACCEL are not fixed indices: contact_columns() places them
    # after this pair when control_memory is on, and after GAP_AFTER when it is off.
    # TOUCH_DV follows ACCEL when touch_detector is "g2l": touch_dv_column() places it.
    # PATH_SIGN, PATH_OUT, PATH_T, PPCF follow that column when it is present, else the contact
    # columns, this pair, or GAP_AFTER, depending on which of those flags are on.
    # BR_TEAM_BEFORE, BR_TEAM_AFTER, BR_X_BEFORE, BR_Y_BEFORE, BR_X_AFTER, BR_Y_AFTER (gap_bracket)
    # come after those: bracket_columns() places them.
    # SLOT_ID, SLOT_RUN (slot_index) come last: slot_index_columns() places them.


N_FEATURES = 14  # base; control_memory +2, contact_inputs +4, g2l +1, ball_path_control +4,
# gap_bracket +6
CONTACT_FEATURES = 4  # BALL_DV, TOUCH_TEAM, CTRL_NOW, ACCEL
PATH_FEATURES = 4  # PATH_SIGN, PATH_OUT, PATH_T, PPCF
BRACKET_FEATURES = 6  # BR_TEAM_BEFORE, BR_TEAM_AFTER, BR_X/Y_BEFORE, BR_X/Y_AFTER
SLOT_INDEX_FEATURES = 2  # SLOT_ID, SLOT_RUN: identifiers for model.slot_temporal, never embedded
SLOT_RUN_MOD = 256  # SLOT_RUN wraps here; runs of one slot sharing an id are >= 512 frames apart
MAX_SLOT_ID = 2047  # float16 holds every integer up to 2048 exactly
_TOUCH_DETECTORS = ("w2", "g2l")
FEATURE_NAMES = (
    "x", "y", "vx", "vy", "speed", "is_away", "is_ball", "valid",
    "dx", "dy", "dist", "ref_ball", "gap_before", "gap_after",
)  # fmt: skip

# columns that flip sign under y -> -y, and under the 180 degree rotation that goes with a team swap
FLIP_Y_COLUMNS = (Col.Y, Col.VY, Col.DY)
ROTATE_COLUMNS = (Col.X, Col.Y, Col.VX, Col.VY, Col.DX, Col.DY)


@dataclass(frozen=True)
class TokenSpec:
    """How a frame becomes tokens. Everything is in metres and seconds."""

    tokens_per_team: int = 11
    velocity_half_s: float = 0.2  # centred difference over +-0.2 s
    max_player_speed: float = 12.0  # faster is a slot swap or a jump: the velocity is zeroed
    max_ball_speed: float = 45.0  # the data lane's cleaning bound for ball jumps
    velocity_scale: float = 8.0  # m/s
    ref_scale_m: float = 20.0
    gap_cap_s: float = 30.0
    control_memory: bool = False  # append last-controller team and age on the ball token
    control_reach_m: float = 1.0  # control_possession reach
    control_hold_frames: int = 3  # control_possession qualifying-run length
    contact_inputs: bool = False  # append BALL_DV, TOUCH_TEAM, CTRL_NOW and ACCEL
    contact_touch_w: int = 2  # frames; detect_touches window for TOUCH_TEAM when detector is w2
    contact_dv_w: int = 5  # frames; ball_velocity_change window for BALL_DV
    touch_detector: str = "w2"  # "w2" (detect_touches) or "g2l"; g2l also appends TOUCH_DV
    ball_path_control: bool = False  # append PATH_SIGN, PATH_OUT, PATH_T and PPCF
    gap_bracket: bool = False  # append the ball gap's two ends: controlling team and position (#33)
    slot_index: bool = False  # append SLOT_ID and SLOT_RUN for model.slot_temporal

    def __post_init__(self) -> None:
        for name in ("contact_touch_w", "contact_dv_w"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"token.{name} must be an integer >= 1, got {value!r}")
        detector = self.touch_detector
        if not isinstance(detector, str) or detector not in _TOUCH_DETECTORS:
            raise ValueError(f"token.touch_detector must be 'w2' or 'g2l', got {detector!r}")

    @property
    def n_tokens(self) -> int:
        return 2 * self.tokens_per_team + 1

    @property
    def ball_index(self) -> int:
        return 2 * self.tokens_per_team

    @property
    def n_features(self) -> int:
        """Columns in one token. Control memory adds two, contact four, g2l one, path four,
        gap bracket six, slot index two."""
        return self.model_features + (SLOT_INDEX_FEATURES if self.slot_index else 0)

    @property
    def model_features(self) -> int:
        """Columns a network embeds: every column except the slot-index pair."""
        extra = (2 if self.control_memory else 0) + (CONTACT_FEATURES if self.contact_inputs else 0)
        extra += _g2l_features(self)
        extra += PATH_FEATURES if self.ball_path_control else 0
        extra += BRACKET_FEATURES if self.gap_bracket else 0
        return N_FEATURES + extra

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dict. `slot_index` is left out while it is off, so a spec without it
        writes the dict it wrote before the field existed."""
        raw = asdict(self)
        if not self.slot_index:
            raw.pop("slot_index")
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TokenSpec:
        """Missing control-memory, contact, ball-path, gap-bracket or slot-index keys mean those
        columns are off.

        Missing `contact_touch_w` and `contact_dv_w` keep 2 and 5, the windows the contact columns
        were built with before those keys existed. Missing `touch_detector` keeps "w2".
        """
        return cls(**raw)


def _g2l_features(spec: TokenSpec) -> int:
    """1 when TOUCH_DV is in the layout (contact on and touch_detector "g2l"), else 0."""
    if spec.contact_inputs and spec.touch_detector == "g2l":
        return 1
    return 0


def contact_columns(spec: TokenSpec) -> tuple[int, int, int, int]:
    """Absolute indices `(BALL_DV, TOUCH_TEAM, CTRL_NOW, ACCEL)`.

    Only present when `contact_inputs` is set. They follow CTRL_TEAM and CTRL_AGE when
    `control_memory` is on, and the base 14 columns otherwise. BALL_DV, TOUCH_TEAM and CTRL_NOW
    are ball-token columns (zero on players); ACCEL is the player tokens (zero on the ball).
    """
    origin = N_FEATURES + (2 if spec.control_memory else 0)
    return origin, origin + 1, origin + 2, origin + 3


def touch_dv_column(spec: TokenSpec) -> int:
    """Absolute index of `TOUCH_DV`.

    Present only when `contact_inputs` is on and `touch_detector` is "g2l". It follows ACCEL,
    the last contact column, and precedes the ball-path columns. The value is the G2L event's
    impulse over `velocity_scale`, clipped to [0, 4], with the same decay as `TOUCH_TEAM`. It is
    a ball-token column (zero on players). A team swap does not negate it.
    """
    if not spec.contact_inputs or spec.touch_detector != "g2l":
        raise ValueError("TOUCH_DV exists only with contact_inputs and touch_detector 'g2l'")
    return N_FEATURES + (2 if spec.control_memory else 0) + CONTACT_FEATURES


def path_columns(spec: TokenSpec) -> tuple[int, int, int, int]:
    """Absolute indices `(PATH_SIGN, PATH_OUT, PATH_T, PPCF)`.

    Only present when `ball_path_control` is set. They follow TOUCH_DV when that column is
    present, else the contact columns when `contact_inputs` is on, else CTRL_TEAM and CTRL_AGE
    when `control_memory` is on, else the base 14. PATH_SIGN, PATH_OUT and PATH_T are ball-token
    columns (zero on players); PPCF is the player tokens (zero on the ball and on a slot that is
    unseen at the frame).
    """
    origin = N_FEATURES + (2 if spec.control_memory else 0)
    origin += CONTACT_FEATURES if spec.contact_inputs else 0
    origin += _g2l_features(spec)
    return origin, origin + 1, origin + 2, origin + 3


def bracket_columns(spec: TokenSpec) -> tuple[int, int, int, int, int, int]:
    """Absolute indices `(BR_TEAM_BEFORE, BR_TEAM_AFTER, BR_X_BEFORE, BR_Y_BEFORE, BR_X_AFTER,
    BR_Y_AFTER)`.

    Only present when `gap_bracket` is set. They are the last six embedded columns: after the
    path columns when `ball_path_control` is on, else after whatever block comes last, and before
    the slot-index pair when `slot_index` is on. All six are ball-token columns (zero on players).
    A team swap negates all six (the teams trade signs and the pitch turns by 180 degrees); a
    y-flip negates the two Y columns.
    """
    if not spec.gap_bracket:
        raise ValueError("the bracket columns exist only with gap_bracket")
    origin = spec.model_features - BRACKET_FEATURES
    return tuple(range(origin, origin + BRACKET_FEATURES))  # type: ignore[return-value]


def slot_index_columns(spec: TokenSpec) -> tuple[int, int]:
    """Absolute indices `(SLOT_ID, SLOT_RUN)`, the last two columns.

    Only present when `slot_index` is set. Player tokens only (zero on the ball and on invalid
    tokens). `model.slot_temporal` reads them to follow each tracker slot through time; no layer
    embeds them.
    """
    if not spec.slot_index:
        raise ValueError("the slot-index columns exist only with slot_index")
    origin = spec.model_features
    return origin, origin + 1


# `model.slot_temporal.inputs` may name columns; RunConfig resolves them against the run's spec.
_SLOT_INPUT_NAMES = {
    "x": Col.X,
    "y": Col.Y,
    "vx": Col.VX,
    "vy": Col.VY,
    "speed": Col.SPEED,
    "dx": Col.DX,
    "dy": Col.DY,
    "dist": Col.DIST,
}


def slot_input_columns(entries: Any, spec: TokenSpec) -> tuple[int, ...]:
    """Column indices for `model.slot_temporal.inputs`: integers pass through, names resolve.

    Names: x, y, vx, vy, speed, dx, dy, dist, and accel (the contact ACCEL column, so only with
    `contact_inputs`). An unknown name raises ValueError.
    """
    out = []
    for entry in entries:
        if isinstance(entry, bool):
            raise ValueError(f"model.slot_temporal.inputs entry {entry!r} is not a column")
        if isinstance(entry, int):
            out.append(entry)
        elif entry == "accel":
            if not spec.contact_inputs:
                raise ValueError("model.slot_temporal input 'accel' needs token.contact_inputs")
            out.append(contact_columns(spec)[3])
        elif isinstance(entry, str) and entry in _SLOT_INPUT_NAMES:
            out.append(_SLOT_INPUT_NAMES[entry])
        else:
            known = ", ".join([*_SLOT_INPUT_NAMES, "accel"])
            raise ValueError(f"model.slot_temporal input {entry!r} is not one of {known}")
    return tuple(out)
