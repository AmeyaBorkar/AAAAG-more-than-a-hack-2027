"""python -m aaaag.ml {train, lomo, infer, determinism}: the harness from the command line.

    train        one fold (--holdout MATCH) or the model on all five dev matches (--holdout none)
    lomo         every fold: train, then score on the held-out match against the champion
    infer        run saved weights on one degraded dev match; frame-level team accuracy
    determinism  the same inference in fresh processes, at other thread counts: bit-exact or not

Every command prints aggregates only. Training uses the IDSSE dev matches plus any
`train.extra_matches` (other sources, always in training, never held out). `--extra-only`
trains on those extras alone, with the same loss and checkpoint as any other run; a
`--holdout MATCH` dev match is monitored and is not trained on. `train.matches_per_set`
caps how many matches one variant set tokenises. `train.pretrain` trains only on label-free
extras and does not load an IDSSE match; leave-one-match-out refuses it.
Scoring uses the held-out dev match (and the Metrica games with --metrica). Never the exam
matches and never BAR-ATH. Set AAAAG_EXTERNAL_DIR, and AAAAG_CACHE_DIR to keep the cache
outside the repository (docs/ml/README.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import A, H
from ..paths import CONFIGS_DIR
from ..sources import idsse
from . import weights as weight_files

DEFAULT_RUN = CONFIGS_DIR / "ml" / "team_baseline.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m aaaag.ml", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train one fold, or all five matches with --holdout none")
    _run_arguments(train)
    train.add_argument("--holdout", default="none", help="IDSSE dev match to leave out, or none")
    train.add_argument(
        "--extra-only",
        action="store_true",
        help="train on train.extra_matches only; --holdout MATCH is monitored, not trained on",
    )
    train.add_argument("--no-resume", action="store_true")

    lomo = sub.add_parser("lomo", help="leave-one-match-out: train every fold and score it")
    _run_arguments(lomo)
    lomo.add_argument("--folds", default="all", help="all, or comma-separated dev match ids")
    lomo.add_argument(
        "--exclude",
        default="",
        help="comma-separated dev matches left out of both training and scoring",
    )
    lomo.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    lomo.add_argument("--ladder", default=None, help="v5.1 player rung for the held-out matches")
    lomo.add_argument("--views", type=int, default=1, choices=(1, 2, 4), help="mirrored views")
    lomo.add_argument(
        "--pipeline",
        choices=("team", "joint"),
        default="team",
        help="team: the rule keeps P(dead); joint: the network's heads for both factors (#86 E9)",
    )
    lomo.add_argument(
        "--reuse", action="store_true", help="score final.pt files in --out as they are"
    )
    lomo.add_argument("--metrica", action="store_true", help="also score on the Metrica games")
    lomo.add_argument("--record", action="store_true", help="write an experiments/ record")
    lomo.add_argument("--issue", type=int, default=None)

    infer = sub.add_parser("infer", help="run saved weights on one degraded dev match")
    _weights_arguments(infer)
    infer.add_argument("--dump", type=Path, default=None, help="save team probability and labels")

    det = sub.add_parser("determinism", help="repeat inference in fresh processes and compare")
    _weights_arguments(det)
    det.add_argument("--threads-list", type=int, nargs="+", default=[1, 1, 4])
    det.add_argument("--gpu", action="store_true", help="also compare against a CUDA run")
    det.add_argument("--work", type=Path, default=None, help="where the runs leave their dumps")

    args = parser.parse_args(argv)
    {"train": _train, "lomo": _lomo, "infer": _infer, "determinism": _determinism}[args.command](
        args
    )


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="the run config YAML")
    parser.add_argument("--out", type=Path, default=None, help="output directory (under the cache)")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps")
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if present)")
    parser.add_argument("--threads", type=int, default=6, help="CPU threads for torch")


def _weights_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", nargs="+", required=True, help="checkpoint(s), mean logits")
    parser.add_argument("--match", default="J03WMX", help="an IDSSE dev match")
    parser.add_argument("--seed", type=int, default=0, help="degrade-v5 seed")
    parser.add_argument("--ladder", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--views", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--chunk", type=int, default=2048)


def _load_run(args: argparse.Namespace) -> Any:
    from .train import RunConfig, replace_train

    run = RunConfig.from_yaml(args.run)
    changes = {k: v for k in ("steps", "seed") if (v := getattr(args, k)) is not None}
    return replace_train(run, **changes) if changes else run


def _train(args: argparse.Namespace) -> None:
    from .data import load_train_matches
    from .lomo import default_out_dir
    from .train import teacher_paths, train

    run = _load_run(args)
    if getattr(args, "extra_only", False):
        _train_extra_only(args, run)
        return
    if run.train.pretrain:
        if args.holdout != "none":
            raise ValueError(
                "a pretraining run has no holdout and loads no IDSSE match; pass --holdout none"
            )
        out = args.out or default_out_dir("train-pretrain")
        # The extras are loaded inside train. No IDSSE match, and distillation stays off.
        train(
            run,
            [],
            out,
            device=args.device,
            resume=not args.no_resume,
            meta={"holdout": None, "train_matches": []},
            threads=args.threads,
        )
        return

    holdout = None if args.holdout == "none" else args.holdout
    teachers = None
    if run.train.distill is not None:
        teachers = teacher_paths(run.train.distill.teacher_runs, holdout)
    ids = [m for m in idsse.DEV_MATCHES if m != holdout]
    matches = load_train_matches(ids, run.token)
    out = args.out or default_out_dir(f"train-{holdout or 'all'}")
    train(
        run,
        [matches[m] for m in ids],
        out,
        device=args.device,
        resume=not args.no_resume,
        meta={"holdout": holdout, "train_matches": ids},
        threads=args.threads,
        teachers=teachers,
    )


def _train_extra_only(args: argparse.Namespace, run: Any) -> None:
    """Train on `train.extra_matches` only. No IDSSE match enters the training roster.

    `--holdout none` does not monitor. `--holdout MATCH` monitors that dev match and still
    does not train on it. An empty extra list, a holdout that is not a dev match, and
    `train.pretrain` (label-free, class loss off) raise before any match is loaded.
    `train.extra_source_kwargs` is passed to the extra-match loader and `train.extra_degrade`
    (default false) is passed to `build_set`, both through `train()`.
    """
    from .lomo import default_out_dir, make_monitor
    from .train import teacher_paths, train

    if run.train.pretrain:
        raise ValueError(
            "--extra-only trains on labelled train.extra_matches with the normal loss; "
            "train.pretrain is label-free and turns that loss off. Use one of them."
        )
    if not run.train.extra_matches:
        raise ValueError("--extra-only trains only on train.extra_matches, and that list is empty")
    holdout = None if args.holdout == "none" else args.holdout
    if holdout is not None and holdout not in idsse.DEV_MATCHES:
        raise ValueError(
            "--extra-only --holdout must be 'none' or an IDSSE dev match to monitor, "
            f"got {args.holdout!r}"
        )
    teachers = None
    if run.train.distill is not None:
        teachers = teacher_paths(run.train.distill.teacher_runs, holdout)
    out = args.out or default_out_dir(f"train-extra-{holdout or 'all'}")
    train(
        run,
        [],
        out,
        device=args.device,
        resume=not args.no_resume,
        meta={"holdout": holdout, "train_matches": [], "extra_only": True},
        threads=args.threads,
        teachers=teachers,
        monitor=None if holdout is None else make_monitor(holdout, run, args.device),
    )


def _lomo(args: argparse.Namespace) -> None:
    from .lomo import default_out_dir, run_lomo, write_experiment_record

    run = _load_run(args)
    out = args.out or default_out_dir("lomo")
    report = run_lomo(
        run,
        args.folds,
        out,
        seeds=args.seeds,
        ladder=args.ladder,
        metrica=args.metrica,
        device=args.device,
        threads=args.threads,
        views=args.views,
        reuse=args.reuse,
        pipeline=args.pipeline,
        exclude=args.exclude,
    )
    if args.record:
        notes = f"leave-one-match-out over {len(report['folds'])} folds, seeds {args.seeds}" + (
            f", excluding {args.exclude}" if args.exclude else ""
        )
        print(f"recorded {write_experiment_record(report, issue=args.issue, notes=notes)}")


def _bundle(args: argparse.Namespace) -> Any:
    from ..registry import SOURCES

    degrade = {
        "version": "5",
        "seed": args.seed,
        **({"ladder": args.ladder} if args.ladder else {}),
    }
    return SOURCES.get("idsse").fn(match_id=args.match, degrade=degrade)


def _infer(args: argparse.Namespace) -> None:
    """Learned evidence through the champion pipeline on a degraded dev match (labels only score)."""
    from .. import pipeline
    from . import evaluate as ev

    bundle = _bundle(args)
    champion = ev.champion_config()
    config = ev.learned_config(
        champion,
        args.weights,
        device=args.device,
        views=args.views,
        chunk=args.chunk,
        threads=args.threads,
    )
    began = time.perf_counter()
    result = pipeline.run(config, bundle, use_cache=False)
    seconds = time.perf_counter() - began
    index = next(i for i, s in enumerate(config.evidence) if s.name == "learned_team")
    team = result.evidence[index].proba
    labels = bundle.labels
    alive = np.isin(labels, (H, A))
    has = ~np.isnan(team[:, 0])
    call_home = team[:, 0] > team[:, 1]
    print(
        f"{bundle.meta.match_id}: pipeline {seconds:.1f}s on {args.device} with {args.threads} "
        f"thread(s); learned_team coverage {has[alive].mean():.3f}, team accuracy on truth-alive "
        f"frames {((call_home == (labels == H)) & has)[alive].mean():.4f}"
    )
    print(
        f"sha256 team {_digest(team)} labels {_digest(result.labels)} "
        f"weights {[weight_files.sha256_file(w)[:12] for w in args.weights]}"
    )
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.dump, team=team, labels=result.labels)


def _digest(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _determinism(args: argparse.Namespace) -> None:
    """Run `infer --dump` in a fresh process per entry of --threads-list (and once on CUDA), then
    compare every run with the first: bit-exact or not, largest probability difference, label
    flips of the decoded pipeline output."""
    from ..paths import cache_dir

    work = args.work or cache_dir() / "ml" / "determinism"
    work.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, Path]] = []
    plan = [("cpu", t) for t in args.threads_list] + ([("cuda", 1)] if args.gpu else [])
    for i, (device, threads) in enumerate(plan):
        dump = work / f"run{i}-{device}-t{threads}.npz"
        command = [
            sys.executable,
            "-m",
            "aaaag.ml",
            "infer",
            "--weights",
            *args.weights,
            "--match",
            args.match,
            "--seed",
            str(args.seed),
            "--device",
            device,
            "--threads",
            str(threads),
            "--views",
            str(args.views),
            "--chunk",
            str(args.chunk),
            "--dump",
            str(dump),
        ]
        if args.ladder:
            command += ["--ladder", args.ladder]
        done = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"run {i} ({device}, {threads} thread(s)): " + done.stdout.strip().splitlines()[0])
        runs.append((f"{device}/t{threads}", dump))
    first_name, first_path = runs[0]
    with np.load(first_path) as first:
        base_team, base_labels = first["team"], first["labels"]
    report = []
    for name, path in runs[1:]:
        with np.load(path) as other:
            team, labels = other["team"], other["labels"]
        both = ~np.isnan(base_team[:, 0]) & ~np.isnan(team[:, 0])
        row = {
            "against": first_name,
            "run": name,
            "bit_exact_team": bool(np.array_equal(base_team, team, equal_nan=True)),
            "max_abs_diff": float(np.abs(base_team[both] - team[both]).max())
            if both.any()
            else 0.0,
            "label_flips": int((base_labels != labels).sum()),
            "n_frames": int(len(labels)),
        }
        report.append(row)
        print(json.dumps(row))
    exact = [r for r in report if r["bit_exact_team"] and r["label_flips"] == 0]
    print(f"{len(exact)}/{len(report)} runs identical to {first_name} (team evidence and labels)")
