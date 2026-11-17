"""Leave-one-match-out over the five IDSSE dev matches: train a fold, score it on its held-out match.

    python -m aaaag.ml lomo --run configs/ml/team_baseline.yaml --folds all --out <dir>

Per fold: train on the other four matches (`train.train`), then score the fold's model on the
held-out match degraded the way the primary suite is (degrade-v5, seeds 0-4 = `idsse-dev-v5`, so
the champion's numbers here are the team's 0.516-level numbers): the frame-level team call against
the champion's last-controller rule, and the spellF1 of the champion pipeline with the learned team
factor against the unmodified champion (evaluate.py). The held-out match is scored once, after
training, and nothing is selected on it. The sealed exam matches are not in `DEV_MATCHES`.

`--metrica` also scores every fold model, and the mean-logit ensemble of all of them, on the three
Metrica games (a different label source and the competition's tracking provider, never trained on).
Everything printed and written is an aggregate.

`--exclude M1,M2` drops those dev matches from every fold: they are not scored and not trained on.
`--folds all` then means the remaining dev matches. Listing an excluded match in `--folds` is an
error. With no exclusions, each fold still trains on the other four dev matches.

`train.extra_matches` are appended to the training matches of every fold that is actually
trained, and they are never a held-out match. The fold list and the held-out scoring stay the
IDSSE dev matches, so a score can still be compared with a run that did not set them.

`train.pretrain` is refused: a pretraining run has no folds and loads no IDSSE match.

`train.init_from` may contain `{fold}`. Each fold replaces it with that fold's held-out
match id before the weights are loaded, so the fold does not start from a model that
trained on the match it holds out. A path without `{fold}` is loaded unchanged.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ..contracts import A, H
from ..paths import cache_dir
from ..registry import SOURCES
from ..sources import idsse
from . import evaluate as ev
from .data import load_extra_matches, load_train_matches
from .windows import build_match_tokens

if TYPE_CHECKING:
    from .train import RunConfig

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
METRICA_GAMES = ("1", "2", "3")


def default_device(device: str | None = None) -> str:
    """`device`, else cuda when torch sees a GPU, else cpu."""
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def make_monitor(
    match_id: str, run: RunConfig, device: str | None
) -> Callable[[Any], dict[str, float]]:
    """A learning curve on the held-out match: after each variant set, the team-call accuracy on
    its truth-alive frames (degrade-v5 seed 0). It is logged and kept in the checkpoint history for
    the reader; it never selects a checkpoint, stops a run or changes the schedule."""
    import torch

    from .infer import logits_from_tokens

    bundle = SOURCES.get("idsse").fn(match_id=match_id, degrade={"version": "5", "seed": 0})
    tokens = build_match_tokens(bundle.tracking, run.token)
    labels = bundle.labels
    alive = np.isin(labels, (H, A))
    where = torch.device(default_device(device))

    def monitor(model: Any) -> dict[str, float]:
        logits = logits_from_tokens(
            model, run.token, tokens, bundle.meta.periods, chunk=8192, device=where
        )
        home = logits[:, 0] > logits[:, 1]
        return {"holdout_team_acc": float((home == (labels == H))[alive].mean())}

    return monitor


def default_out_dir(name: str) -> Path:
    return cache_dir() / "ml" / name


def evaluate_match(
    match_id: str,
    weights: Sequence[str],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    ladder: str | None = None,
    device: str | None = None,
    views: int = 1,
    pipeline: str = "team",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Score `weights` on the held-out `match_id`, degraded by v5 (and a player rung if given), once
    per seed. Returns the per-seed rows (with bootstrap flags) and their aggregate. `pipeline`:
    `team` = the champion with the learned team factor (the rule keeps P(dead)); `joint` = the
    network's own heads for both factors and no rule-based ball state (evaluate.joint_config)."""
    champion = ev.champion_config()
    learned = ev.LEARNED_PIPELINES[pipeline](
        champion, weights, device=default_device(device), views=views, chunk=8192
    )
    rows = []
    for seed in seeds:
        degrade = {"version": "5", "seed": int(seed), **({"ladder": ladder} if ladder else {})}
        bundle = SOURCES.get("idsse").fn(match_id=match_id, degrade=degrade)
        began = time.perf_counter()
        row = ev.score_bundle(bundle, champion, learned)
        rows.append(row)
        log(
            f"  {match_id} seed {seed}: spellF1 champion {row['metrics']['champion']['spell_f1_mean']:.4f}"
            f" learned {row['metrics']['learned']['spell_f1_mean']:.4f}; team-acc rule "
            f"{row['team']['rule_acc']:.3f} learned {row['team']['learned_acc']:.3f} "
            f"({time.perf_counter() - began:.0f}s)"
        )
    return {"rows": rows, "aggregate": ev.aggregate(rows)}


def evaluate_metrica(
    fold_weights: dict[str, str],
    *,
    device: str | None = None,
    pipeline: str = "team",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every fold model alone and the ensemble of all of them, on the Metrica games."""
    champion = ev.champion_config()
    groups: dict[str, list[str]] = {f"fold-{m}": [w] for m, w in fold_weights.items()}
    groups["ensemble"] = list(fold_weights.values())
    out: dict[str, Any] = {}
    for name, weights in groups.items():
        learned = ev.LEARNED_PIPELINES[pipeline](
            champion, weights, device=default_device(device), chunk=8192
        )
        rows = []
        for game in METRICA_GAMES:
            bundle = SOURCES.get("metrica").fn(game=game)
            rows.append(ev.score_bundle(bundle, champion, learned))
        out[name] = {"rows": rows, "aggregate": ev.aggregate(rows)}
        agg = out[name]["aggregate"]
        log(
            f"  metrica {name}: spellF1 champion {agg['champion']['spell_f1_mean']:.4f} "
            f"learned {agg['learned']['spell_f1_mean']:.4f}; team-acc rule "
            f"{agg['team']['rule_acc']:.3f} learned {agg['team']['learned_acc']:.3f}"
        )
    return out


class LomoSplit(NamedTuple):
    """Which dev matches are scored, which train each fold, and which were excluded.

    `train_ids[holdout]` is the dev roster minus that holdout minus `exclude`, in roster order.
    """

    folds: tuple[str, ...]
    train_ids: dict[str, tuple[str, ...]]
    exclude: tuple[str, ...]


def _match_list(raw: Sequence[str] | str | None) -> list[str]:
    """Comma-separated match ids, or a sequence of them, de-duplicated in given order."""
    if raw is None:
        return []
    parts = raw.split(",") if isinstance(raw, str) else [str(item) for item in raw]
    seen: list[str] = []
    for part in parts:
        for piece in part.split(","):
            match = piece.strip()
            if match and match not in seen:
                seen.append(match)
    return seen


def resolve_lomo(
    folds: Sequence[str] | str,
    exclude: Sequence[str] | str = (),
    *,
    dev_matches: Sequence[str] = idsse.DEV_MATCHES,
) -> LomoSplit:
    """Fold ids and training sets for one leave-one-match-out run.

    `folds="all"` scores every dev match that is not excluded. An excluded match named in
    `folds` is an error: it is neither trained on nor scored. Each fold trains on the dev
    matches except the held-out one and the excluded ones, so a shorter `--folds` list does
    not drop the unnamed dev matches from training. No `exclude` reproduces the original
    split: five folds, each trained on the other four.
    """
    dev = tuple(dev_matches)
    known = set(dev)
    excluded = _match_list(exclude)
    unknown_excluded = [match for match in excluded if match not in known]
    if unknown_excluded:
        raise ValueError(f"unknown excluded match {unknown_excluded}; dev matches are {list(dev)}")
    excluded_set = set(excluded)
    if isinstance(folds, str) and folds.strip() == "all":
        fold_ids = [match for match in dev if match not in excluded_set]
    else:
        fold_ids = _match_list(folds)
        unknown = [match for match in fold_ids if match not in known]
        if unknown:
            raise ValueError(f"unknown fold {unknown}; dev matches are {list(dev)}")
        overlap = [match for match in fold_ids if match in excluded_set]
        if overlap:
            raise ValueError(
                f"excluded match listed in --folds: {', '.join(overlap)}; "
                "an excluded match is neither trained on nor scored"
            )
    if not fold_ids:
        raise ValueError("no folds left to score")
    train_ids = {
        holdout: tuple(match for match in dev if match != holdout and match not in excluded_set)
        for holdout in fold_ids
    }
    empty = [holdout for holdout, ids in train_ids.items() if not ids]
    if empty:
        raise ValueError(f"fold {empty[0]} has no training match after exclusions")
    return LomoSplit(tuple(fold_ids), train_ids, tuple(excluded))


def run_lomo(
    run: RunConfig,
    folds: Sequence[str] | str,
    out_dir: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    ladder: str | None = None,
    metrica: bool = False,
    device: str | None = None,
    threads: int = 6,
    views: int = 1,
    reuse: bool = False,
    pipeline: str = "team",
    exclude: Sequence[str] | str = (),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train and score every fold in `folds`; write `report.json` (aggregates) to `out_dir`
    (`report-<rung>-views<k>.json` when scored under a player rung or with mirrored views).

    With `reuse`, a fold whose `final.pt` is already in `out_dir` is scored as it is instead of
    retrained: a finished run can be re-scored under another player rung, `views` or seeds.
    `exclude` removes dev matches from scoring and from every training set. The split is
    `resolve_lomo`; with nothing excluded it is the dev matches other than the holdout.
    `train.extra_matches` are loaded once and appended to each fold that trains. They are not
    scored and they do not add a fold. A reused `final.pt` is left as it is, and extras are not
    loaded when every fold is reused. `train.pretrain` is refused before any match is loaded:
    a pretraining run has no folds. `train.init_from` with `{fold}` is resolved by passing
    this fold's held-out match id; a path without `{fold}` is passed through unchanged."""
    if run.train.pretrain:
        raise ValueError(
            "a pretraining run has no folds; use python -m aaaag.ml train --holdout none"
        )
    split = resolve_lomo(folds, exclude)
    from .train import teacher_paths, train

    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve every fold that will actually train before loading matches, so a missing teacher
    # fails before the first step. A reused final.pt is scored as it is and needs no teacher.
    teacher_for: dict[str, list[Path]] = {}
    if run.train.distill is not None:
        for holdout in split.folds:
            if reuse and (out_dir / f"fold-{holdout}" / "final.pt").is_file():
                continue
            teacher_for[holdout] = teacher_paths(run.train.distill.teacher_runs, holdout)
    if split.exclude:
        needed = [
            match
            for match in idsse.DEV_MATCHES
            if any(match in ids for ids in split.train_ids.values())
        ]
        matches = load_train_matches(needed, run.token) if needed else {}
    else:
        matches = load_train_matches(idsse.DEV_MATCHES, run.token)
    will_train = any(
        not (reuse and (out_dir / f"fold-{holdout}" / "final.pt").is_file())
        for holdout in split.folds
    )
    extra = []
    if will_train and run.train.extra_matches:
        extra = load_extra_matches(
            run.train.extra_matches,
            run.token,
            labels=run.train.extra_labels,
            extra_source_kwargs=run.train.extra_source_kwargs,
        )
    began = time.perf_counter()
    if pipeline not in ev.LEARNED_PIPELINES:
        raise ValueError(
            f"pipeline must be one of {sorted(ev.LEARNED_PIPELINES)}, got {pipeline!r}"
        )
    report: dict[str, Any] = {
        "run": run.to_dict(),
        "folds": {},
        "seeds": list(seeds),
        "pipeline": pipeline,
    }
    if split.exclude:
        report["exclude"] = list(split.exclude)
    weights: dict[str, str] = {}
    flags: dict[str, dict[str, list]] = {"champion": {}, "learned": {}}
    for holdout in split.folds:
        log(f"== fold: hold out {holdout}")
        fold_dir = out_dir / f"fold-{holdout}"
        train_on = [matches[m] for m in split.train_ids[holdout]]
        final = fold_dir / "final.pt"
        if not (reuse and final.is_file()):
            if extra:
                train_on.extend(extra)
            final = train(
                run,
                train_on,
                fold_dir,
                device=device,
                log=log,
                meta={"holdout": holdout, "train_matches": [m.match_id for m in train_on]},
                threads=threads,
                monitor=make_monitor(holdout, run, device),
                teachers=teacher_for.get(holdout),
                fold=holdout,
            )
        weights[holdout] = str(final)
        result = evaluate_match(
            holdout,
            [str(final)],
            seeds=seeds,
            ladder=ladder,
            device=device,
            views=views,
            pipeline=pipeline,
            log=log,
        )
        report["folds"][holdout] = {"weights": str(final), "aggregate": result["aggregate"]}
        for who in flags:
            flags[who][holdout] = [row["flags"][who] for row in result["rows"] if row["flags"]]
        log(f"   fold wall-clock so far {time.perf_counter() - began:.0f}s")
    report["summary"] = summarise(report["folds"], flags)
    if metrica:
        report["metrica"] = {
            name: res["aggregate"]
            for name, res in evaluate_metrica(
                weights, device=device, pipeline=pipeline, log=log
            ).items()
        }
    report["wall_clock_s"] = time.perf_counter() - began
    name = (
        "report.json"
        if ladder is None and views == 1
        else f"report-{ladder or 'v5'}-views{views}.json"
    )
    (out_dir / name).write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    log(format_summary(report))
    return report


def summarise(folds: dict[str, Any], flags: dict[str, dict[str, list]]) -> dict[str, Any]:
    """Mean, spread and worst fold of the pipeline and frame-level numbers, and the paired
    bootstrap interval of the spellF1 gain when the repository's implementation is available."""
    champ = np.array([f["aggregate"]["champion"]["spell_f1_mean"] for f in folds.values()])
    new = np.array([f["aggregate"]["learned"]["spell_f1_mean"] for f in folds.values()])
    delta = new - champ
    out: dict[str, Any] = {
        "n_folds": len(folds),
        "spell_f1_champion_mean": float(champ.mean()),
        "spell_f1_learned_mean": float(new.mean()),
        "spell_f1_delta_mean": float(delta.mean()),
        "spell_f1_delta_worst_fold": float(delta.min()),
        "spell_f1_delta_best_fold": float(delta.max()),
        "folds_improved": int((delta > 0).sum()),
    }
    for key in (
        "learned_acc",
        "rule_acc",
        "learned_acc_where_rule_votes",
        "rule_acc_where_rule_votes",
    ):
        out[f"team_{key}_mean"] = float(
            np.mean([f["aggregate"]["team"][key] for f in folds.values()])
        )
    interval = (
        ev.paired_interval(flags["champion"], flags["learned"])
        if all(flags["champion"].values())
        else None
    )
    if interval is not None:
        out["spell_f1_delta_bootstrap_5_95"] = [float(interval[0]), float(interval[1])]
    return out


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        "",
        "held-out matches (mean over degradation seeds), champion -> champion + learned_team",
    ]
    for match, fold in report["folds"].items():
        agg = fold["aggregate"]
        lines.append(
            f"  {match}: spellF1 {agg['champion']['spell_f1_mean']:.4f} -> "
            f"{agg['learned']['spell_f1_mean']:.4f} "
            f"({agg['learned']['spell_f1_mean'] - agg['champion']['spell_f1_mean']:+.4f}); "
            f"team accuracy on truth-alive frames: rule {agg['team']['rule_acc']:.3f}, "
            f"learned {agg['team']['learned_acc']:.3f}"
        )
    s = report["summary"]
    lines.append(
        f"  mean spellF1 {s['spell_f1_champion_mean']:.4f} -> {s['spell_f1_learned_mean']:.4f} "
        f"({s['spell_f1_delta_mean']:+.4f}); worst fold {s['spell_f1_delta_worst_fold']:+.4f}; "
        f"improved {s['folds_improved']}/{s['n_folds']}"
        + (
            f"; paired bootstrap 5-95% [{s['spell_f1_delta_bootstrap_5_95'][0]:+.4f}, "
            f"{s['spell_f1_delta_bootstrap_5_95'][1]:+.4f}]"
            if "spell_f1_delta_bootstrap_5_95" in s
            else ""
        )
    )
    return "\n".join(lines)


def write_experiment_record(
    report: dict[str, Any], *, issue: int | None, notes: str = "", member: str | None = None
) -> Path:
    """One experiments/<member>/ record for the learned pipeline on the primary suite: the learned
    spellF1 per held-out match as the headline, the champion's and the team accuracies beside it.
    The config shows the swapped producer; the weights are the fold models, which are not shipped,
    so their paths are replaced by a note."""
    from ..experiments import Suite, write_record
    from ..paths import CONFIGS_DIR
    from ..pipeline import PipelineConfig, Step
    from ..team import current_member

    base = ev.champion_config()
    raw = base.to_dict()
    placeholder = Step("learned_team", {"weights": "leave-one-match-out fold models, not shipped"})
    raw["evidence"] = [
        placeholder.to_dict() if s["name"] == ev.RULE_STEP else s for s in raw["evidence"]
    ]
    raw["name"] = f"{base.name}+learned_team-lomo"
    config = PipelineConfig.from_dict(raw, source_path=base.source_path)
    folds = report["folds"]
    learned = {m: f["aggregate"]["learned"] for m, f in folds.items()}
    keys = sorted(set.intersection(*(set(v) for v in learned.values())))
    values = {k: np.array([learned[m][k] for m in learned], dtype=np.float64) for k in keys}
    results = {
        "n_matches": len(folds),
        "mean": {k: float(v.mean()) for k, v in values.items()},
        "std": {k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in values.items()},
        "min": {k: float(v.min()) for k, v in values.items()},
        "per_match": learned,
        "champion_per_match": {m: f["aggregate"]["champion"] for m, f in folds.items()},
        "team_accuracy_per_match": {m: f["aggregate"]["team"] for m, f in folds.items()},
        "summary": report["summary"],
        "metrica": report.get("metrica"),
        "seeds": report["seeds"],
    }
    suite = Suite.load(CONFIGS_DIR / "suites" / "idsse-dev-v5.yaml")
    return write_record(
        config, suite, results, member=member or current_member(), issue=issue, notes=notes
    )
