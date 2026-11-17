"""Synthetic matches (aaaag.synthetic) as a registered source, so suites can list them."""

from __future__ import annotations

from ..contracts import MatchBundle
from ..registry import SOURCES
from ..synthetic import SyntheticConfig, generate


@SOURCES.register("synthetic", version="1", example={"seed": 0, "minutes_per_half": 2.0})
def load_synthetic(*, seed: int = 0, match_id: str | int | None = None, **overrides) -> MatchBundle:
    """`overrides` are SyntheticConfig fields, e.g. minutes_per_half=5, ball_missing_rate=0.5.

    `match_id` is the integer seed, so an extra-match entry `synthetic/1` is `generate(seed=1)`.
    Suites keep passing `seed`. When both are set, `match_id` is the one that is used.
    """
    if match_id is not None:
        try:
            seed = int(match_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"synthetic match_id must be an integer seed, got {match_id!r}"
            ) from exc
    return generate(seed=seed, config=SyntheticConfig(**overrides) if overrides else None)
