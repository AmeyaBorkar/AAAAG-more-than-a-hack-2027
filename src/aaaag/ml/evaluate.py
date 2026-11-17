"""Scoring a learned team factor against the champion on held-out matches.

Two comparisons, on identical bundles, both aggregates only:

1. **Frame level**, on the frames the truth says are alive: the share where the team call is right,
   for the champion's `control_possession` rule (last controller) and for `learned_team`, plus the
   same split by whether the ball is seen. The rule abstains before its first control, so it is
   scored twice: with abstentions as errors, and only on the frames where it has an opinion (where
   the learned model is also scored, so the two are compared on the same frames).
2. **Pipeline level**: the unmodified champion against the champion with `control_possession`
   replaced by `learned_team` (ball-state producer, fusion and decoder untouched), by the
   organisers' metric. spellF1 per held-out match is the mean over its degradation seeds.

The paired possession bootstrap is the repository's own (analysis/scripts/ballstate_idsse_dev.py),
loaded by path so there is one implementation; without it the interval is left out.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import pipeline
from ..contracts import A, H, MatchBundle
from ..evaluation.score import evaluate, summarize
from ..paths import CONFIGS_DIR, REPO_ROOT
from ..pipeline import PipelineConfig, Step
from . import weights as weight_files

CHAMPION = "possession_snap_05s"
RULE_STEP = "control_possession"  # the champion producer that `learned_team` replaces


def champion_config(name: str = CHAMPION) -> PipelineConfig:
    return PipelineConfig.load(CONFIGS_DIR / "pipelines" / f"{name}.yaml")


def learned_config(
    base: PipelineConfig,
    weights: Sequence[str],
    *,
    device: str = "cpu",
    views: int = 1,
    chunk: int = 2048,
    threads: int = 1,
    step: str = RULE_STEP,
) -> PipelineConfig:
    """`base` with its `step` producer swapped for `learned_team` on `weights`, pinned by SHA-256."""
    pins = [weight_files.sha256_file(weight_files.resolve(w)) for w in weights]
    params: dict[str, Any] = {
        "weights": list(weights),
        "sha256": pins,
        "device": device,
        "views": views,
        "chunk": chunk,
        "threads": threads,
    }
    raw = base.to_dict()
    swapped = [
        Step("learned_team", params).to_dict() if s["name"] == step else s for s in raw["evidence"]
    ]
    if swapped == raw["evidence"]:
        raise ValueError(f"{base.name}: no {step!r} evidence step to replace")
    raw["evidence"] = swapped
    raw["name"] = f"{base.name}+learned_team"
    return PipelineConfig.from_dict(raw, source_path=base.source_path)


def joint_config(
    base: PipelineConfig,
    weights: Sequence[str],
    *,
    device: str = "cpu",
    views: int = 1,
    chunk: int = 2048,
    threads: int = 1,
) -> PipelineConfig:
    """The network's own heads for BOTH factors and no rule-based ball state (#86, E9 and E11):
    `learned_team` with `output: joint`, `factorised` fusion reading P(dead) and P(home | alive) from
    it, and `base`'s decoder. The ball-state producer, its missing-ball cue and its restart-side lean
    are not in the pipeline."""
    pins = [weight_files.sha256_file(weight_files.resolve(w)) for w in weights]
    params: dict[str, Any] = {
        "weights": list(weights),
        "sha256": pins,
        "device": device,
        "views": views,
        "chunk": chunk,
        "threads": threads,
        "output": "joint",
    }
    raw = base.to_dict()
    raw["evidence"] = [Step("learned_team", params).to_dict()]
    raw["fusion"] = {
        "name": "factorised",
        "params": {"dead_from": ["learned_team"], "team_from": ["learned_team"]},
    }
    raw["name"] = f"{base.name}+learned_joint"
    return PipelineConfig.from_dict(raw, source_path=base.source_path)


# the pipelines `lomo` can score: the team factor only (the rule keeps P(dead)) or both factors
LEARNED_PIPELINES = {"team": learned_config, "joint": joint_config}


def team_accuracy(
    labels: np.ndarray, rule: np.ndarray, learned: np.ndarray, ball_seen: np.ndarray
) -> dict[str, float]:
    """Team-call accuracy on truth-alive frames. `rule` and `learned` are (T, 3) evidence; the
    call is home when P(H) > P(A), a NaN row is an abstention."""
    alive = np.isin(labels, (H, A))
    truth_home = labels == H

    def call(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        has = ~np.isnan(proba[:, 0])
        return has, proba[:, 0] > proba[:, 1]

    def share(correct: np.ndarray, mask: np.ndarray) -> float:
        return float(correct[mask].mean()) if mask.any() else float("nan")

    out: dict[str, float] = {"n_alive": float(alive.sum())}
    has_r, home_r = call(rule)
    has_l, home_l = call(learned)
    right_r = has_r & (home_r == truth_home)  # an abstention is wrong
    right_l = has_l & (home_l == truth_home)
    for name, right, has in (("learned", right_l, has_l), ("rule", right_r, has_r)):
        out[f"{name}_acc"] = share(right, alive)
        out[f"{name}_acc_ball_seen"] = share(right, alive & ball_seen)
        out[f"{name}_acc_ball_missing"] = share(right, alive & ~ball_seen)
        out[f"{name}_coverage"] = float(has[alive].mean()) if alive.any() else float("nan")
    voted = alive & has_r
    out["learned_acc_where_rule_votes"] = share(right_l, voted)
    out["rule_acc_where_rule_votes"] = share(right_r, voted)
    return out


def score_bundle(
    bundle: MatchBundle,
    champion: PipelineConfig,
    learned: PipelineConfig,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run both pipelines on `bundle` (labelled) and score them. Returns aggregates plus the
    `flags` the paired bootstrap needs for each pipeline."""
    if bundle.labels is None:
        raise ValueError("score_bundle needs a labelled bundle")
    base = pipeline.run(champion, bundle, use_cache=use_cache)
    new = pipeline.run(learned, bundle, use_cache=use_cache)
    rule_index = next(i for i, s in enumerate(champion.evidence) if s.name == RULE_STEP)
    learned_index = next(i for i, s in enumerate(learned.evidence) if s.name == "learned_team")
    metrics = {
        "champion": summarize(evaluate(bundle.labels, base.labels)),
        "learned": summarize(evaluate(bundle.labels, new.labels)),
    }
    team = team_accuracy(
        bundle.labels,
        base.evidence[rule_index].proba,
        new.evidence[learned_index].proba,
        bundle.tracking.ball_seen(),
    )
    tools = _bootstrap_tools()
    flags = (
        {
            "champion": tools.spell_flags(bundle.labels, base.labels),
            "learned": tools.spell_flags(bundle.labels, new.labels),
        }
        if tools is not None
        else None
    )
    return {"metrics": metrics, "team": team, "flags": flags}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean over seeds of a match's `score_bundle` results (flags dropped)."""

    def mean(path: Sequence[str]) -> float:
        values = []
        for row in rows:
            node: Any = row
            for key in path:
                node = node[key]
            values.append(node)
        return float(np.nanmean(values))

    def common(path: Sequence[str]) -> list[str]:
        keys = []
        for row in rows:
            node: Any = row
            for key in path:
                node = node[key]
            keys.append(set(node))
        return sorted(set.intersection(*keys))

    out: dict[str, Any] = {"n_bundles": len(rows)}
    for who in ("champion", "learned"):
        out[who] = {k: mean(("metrics", who, k)) for k in common(("metrics", who))}
    out["team"] = {k: mean(("team", k)) for k in common(("team",))}
    return out


def paired_interval(
    champion_flags: dict[str, list], learned_flags: dict[str, list], *, draws: int = 1000
) -> tuple[float, float] | None:
    """5-95% interval of the mean spellF1 gain (learned - champion) of the paired possession
    bootstrap, or None if the repository's implementation cannot be loaded."""
    tools = _bootstrap_tools()
    if tools is None:
        return None
    return tools.paired_interval(champion_flags, learned_flags, draws=draws)


_TOOLS: Any = False  # False: not tried; None: unavailable


def _bootstrap_tools() -> Any:
    global _TOOLS
    if _TOOLS is False:
        path = REPO_ROOT / "analysis" / "scripts" / "ballstate_idsse_dev.py"
        _TOOLS = None
        if path.is_file():
            spec = importlib.util.spec_from_file_location("ballstate_idsse_dev_tools", path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _TOOLS = module
    return _TOOLS
