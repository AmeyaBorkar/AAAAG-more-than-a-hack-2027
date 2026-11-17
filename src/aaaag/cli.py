"""`python -m aaaag <command>` -- the team's single entry point. Every command prints aggregates only.

doctor                       environment, pins, vendored-code integrity, data presence
ingest [--force]             parse + cache BAR-ATH tracking; print a layout/coverage summary
run CONFIG [--synthetic N]   run a pipeline on BAR-ATH (or a synthetic match), write + validate
validate CSV                 the organisers' validate_submission.py, unchanged
smoke CSV                    public-window CORRECTNESS check (never for choosing versions)
diagnose CSV                 label-free diagnostics of a BAR-ATH labels file
evaluate CONFIG --suite S    score a pipeline on a validation suite; record it in experiments/
leaderboard [--suite S]      ranked experiment records
submit                       build AAAAG.csv from configs/submission.yaml; validate; fingerprint
orient                       session start: your issues, requests to your lane, reviews, merges
new handover|finding|adr T   create a session handover, analysis finding or decision record
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from .paths import (
    CONFIGS_DIR,
    DECISIONS_DIR,
    FINDINGS_DIR,
    HANDOVER_DIR,
    REPO_ROOT,
    cache_dir,
    outputs_dir,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aaaag", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the environment")

    p = sub.add_parser("ingest", help="parse and cache the BAR-ATH tracking")
    p.add_argument("--force", action="store_true", help="ignore the cache and re-parse")

    p = sub.add_parser("run", help="run a pipeline config")
    p.add_argument("config", type=Path)
    p.add_argument("--synthetic", type=int, metavar="SEED", help="run on a synthetic match instead")
    p.add_argument("--out", type=Path, help="CSV path (default outputs/submissions/<name>.csv)")
    p.add_argument("--no-cache", action="store_true")

    for name in ("validate", "smoke", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("csv", type=Path)

    p = sub.add_parser("evaluate", help="score a pipeline on a validation suite")
    p.add_argument("config", type=Path)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--no-record", action="store_true", help="do not write an experiment record")
    p.add_argument("--issue", type=int, help="GitHub issue this experiment belongs to")
    p.add_argument("--notes", default="")
    p.add_argument("--member", help="roster key (default: detected from gh / git)")
    p.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("leaderboard", help="ranked experiment records")
    p.add_argument("--suite", help="suite name (rows from different suites are not comparable)")
    p.add_argument("--metric", default="spell_f1_mean")
    p.add_argument("--top", type=int, default=20)

    p = sub.add_parser("submit", help="build the final AAAAG.csv")
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "submission.yaml")

    p = sub.add_parser("orient", help="session-start overview from GitHub")
    p.add_argument("--member")

    p = sub.add_parser("new", help="create a finding or decision record from its template")
    p.add_argument("kind", choices=("handover", "finding", "adr"))
    p.add_argument("title")
    p.add_argument("--member")

    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


# -- commands -------------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    return run_doctor()


def cmd_ingest(args: argparse.Namespace) -> int:
    from .io import atd
    from .sources.bar_ath import asset_paths, bar_ath_meta, load_bar_ath

    paths = asset_paths()
    print("layout:", atd.describe_layout(paths["atd_meta"]))
    if args.force:
        for stale in (cache_dir() / "BAR-ATH").glob("tracking-*"):
            stale.unlink()
    started = time.perf_counter()
    bundle = load_bar_ath()
    print(f"loaded in {time.perf_counter() - started:.1f} s")
    trk, meta = bundle.tracking, bar_ath_meta()
    seen = trk.ball_seen()
    played = meta.played_mask()
    objects = (~np.isnan(trk.player_xy[:, :, 0])).sum(axis=1)
    print(
        f"frames {trk.n_frames}, slots {trk.n_slots} (home {(trk.slot_team == 0).sum()}, away {(trk.slot_team == 1).sum()})"
    )
    print(f"ball seen: {seen.mean():.1%} of all frames, {seen[played].mean():.1%} of graded frames")
    for period in meta.periods:
        window = slice(period.start, period.stop)
        print(
            f"  period {period.index}: ball seen {seen[window].mean():.1%}, mean tracked objects {objects[window].mean():.1f}"
        )
    print(
        f"ball channels beyond x,y: {sorted(trk.ball_extra) or 'none'}; player speed: {trk.player_speed is not None}"
    )
    if bundle.tags is not None:
        print(
            f"smart tagging: {len(bundle.tags)} instances, {sum(t.team is None for t in bundle.tags)} without a team"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from . import pipeline
    from .evaluation.diagnostics import describe
    from .pipeline import PipelineConfig
    from .registry import SOURCES

    config = PipelineConfig.load(args.config)
    if args.synthetic is not None:
        bundle = SOURCES.get("synthetic").fn(seed=args.synthetic)
    else:
        bundle = SOURCES.get("bar-ath").fn()
    started = time.perf_counter()
    result = pipeline.run(config, bundle, use_cache=not args.no_cache)
    print(
        f"{config.name}: ran in {time.perf_counter() - started:.1f} s  "
        + "  ".join(f"{k} {v:.2f}s" for k, v in result.timings.items())
    )
    _print_dict("label-free diagnostics", describe(result.labels, bundle.meta))

    if args.synthetic is not None:
        from .evaluation.score import evaluate, summarize

        _print_dict(
            "synthetic score (plumbing check only)",
            summarize(evaluate(bundle.labels, result.labels)),
        )
        return 0

    from .submission import validate_official, write_submission

    out = args.out or outputs_dir() / "submissions" / f"{config.name}.csv"
    digest = write_submission(result.labels, out)
    ok, report = validate_official(out)
    print(f"\nwrote {out}  sha256 {digest[:16]}  official validation: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(report)
    return 0 if ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from .submission import validate_official

    ok, report = validate_official(args.csv)
    print(report)
    return 0 if ok else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    from .evaluation.smoke import public_smoke

    report = public_smoke(_read_labels(args.csv))
    print("\n".join(report.lines()))
    return 0 if report.passed else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .evaluation.diagnostics import describe
    from .sources.bar_ath import bar_ath_meta

    _print_dict("label-free diagnostics", describe(_read_labels(args.csv), bar_ath_meta()))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .experiments import Suite, evaluate_suite, write_record
    from .pipeline import PipelineConfig
    from .team import current_member

    config = PipelineConfig.load(args.config)
    suite = Suite.load(args.suite)
    results = evaluate_suite(config, suite, use_cache=not args.no_cache)
    print(
        f"{config.name} on {suite.name}: {results['n_matches']} matches in {results['runtime_s']:.1f} s"
    )
    for key in (
        "spell_f1_mean",
        "macro_f1",
        "n_pred_spells",
        "n_gt_spells",
        "alive_dead_f1",
        "team_acc_both_alive",
    ):
        print(
            f"  {key:<22} mean {results['mean'][key]:8.4f}  std {results['std'][key]:7.4f}  min {results['min'][key]:8.4f}"
        )
    if not args.no_record:
        path = write_record(
            config,
            suite,
            results,
            member=current_member(args.member),
            issue=args.issue,
            notes=args.notes,
        )
        print(f"recorded {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .experiments import leaderboard

    rows = leaderboard(args.suite, args.metric)[: args.top]
    if not rows:
        print("no experiment records yet")
        return 0
    print(
        f"{'rank':>4}  {'mean':>7} {'std':>6} {'min':>7}  {'macroF1':>7}  {'n':>2}  {'suite':<16} {'pipeline':<28} member"
    )
    for rank, row in enumerate(rows, start=1):
        dirty = "*" if row["dirty"] else " "
        print(
            f"{rank:>4}  {row['mean']:7.4f} {row['std']:6.4f} {row['min']:7.4f}  {row['macro_f1']:7.4f}  "
            f"{row['n']:>2}  {row['suite'][:16]:<16} {row['pipeline'][:28]:<28} {row['member']}{dirty}"
        )
    print(
        f"\nmetric: {args.metric}. Compare rows of the same suite only. * = run from a dirty tree."
    )
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    from . import pipeline
    from .evaluation.diagnostics import describe
    from .evaluation.smoke import public_smoke
    from .pipeline import PipelineConfig
    from .registry import SOURCES
    from .submission import TEAM, validate_official, write_submission

    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = PipelineConfig.load(REPO_ROOT / spec["pipeline"])
    # Cold, like the organisers' fresh clone: a tracking cache from an older parser that
    # kept its PARSER_VERSION would otherwise change the file we upload (#131).
    bundle = SOURCES.get("bar-ath").fn(use_cache=False)
    result = pipeline.run(config, bundle, use_cache=False)
    out = outputs_dir() / "submission" / f"{TEAM}.csv"
    digest = write_submission(result.labels, out)
    ok, report = validate_official(out)
    print(report)
    print("\n".join(public_smoke(result.labels).lines()))
    _print_dict("label-free diagnostics", describe(result.labels, bundle.meta))
    print(f"\n{out}\nsha256 {digest}")
    expected = spec.get("expected_sha256")
    if expected:
        print(
            "reproduction check:",
            "MATCH" if expected == digest else f"MISMATCH (expected {expected})",
        )
        ok = ok and expected == digest
    return 0 if ok else 1


def cmd_orient(args: argparse.Namespace) -> int:
    from .team import orient

    return orient(args.member)


def cmd_new(args: argparse.Namespace) -> int:
    from .team import current_member

    now = datetime.now(UTC)
    today = now.date().isoformat()
    slug = re.sub(r"[^a-z0-9]+", "-", args.title.lower()).strip("-")[:60]
    member = current_member(args.member)
    if args.kind == "finding":
        template = FINDINGS_DIR / "_TEMPLATE.md"
        target = FINDINGS_DIR / f"{today}-{member}-{slug}.md"
    elif args.kind == "handover":
        template = HANDOVER_DIR / "_TEMPLATE.md"
        target = HANDOVER_DIR / f"{now:%Y-%m-%d_%H%MZ}_{member}_{slug}.md"
    else:
        template, target = DECISIONS_DIR / "_TEMPLATE.md", DECISIONS_DIR / f"{today}-{slug}.md"
    if target.exists():
        print(f"{target} already exists")
        return 1
    text = template.read_text(encoding="utf-8")
    stamp = f"{now:%Y-%m-%d %H:%M} UTC"
    replacements = {
        "{{title}}": args.title,
        "{{date}}": today,
        "{{time}}": stamp,
        "{{member}}": member,
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    target.write_text(text, encoding="utf-8")
    print(target.relative_to(REPO_ROOT).as_posix())
    return 0


COMMANDS = {
    "doctor": cmd_doctor,
    "ingest": cmd_ingest,
    "run": cmd_run,
    "validate": cmd_validate,
    "smoke": cmd_smoke,
    "diagnose": cmd_diagnose,
    "evaluate": cmd_evaluate,
    "leaderboard": cmd_leaderboard,
    "submit": cmd_submit,
    "orient": cmd_orient,
    "new": cmd_new,
}


def _read_labels(path: Path) -> np.ndarray:
    from .official import pm

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, len(table))
    if reason:
        raise SystemExit(f"{path}: {reason}")
    return pm.label_codes(table)


def _print_dict(title: str, values: dict) -> None:
    print(f"\n{title}")
    for key, value in values.items():
        print(f"  {key:<26} {value:10.4f}" if isinstance(value, float) else f"  {key:<26} {value}")


if __name__ == "__main__":
    sys.exit(main())
