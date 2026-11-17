"""Validation suites and the experiment registry.

A suite is a YAML list of LABELLED matches -- synthetic today, external real matches (degraded to
look like the ATD) as their adapters land:

    name: synthetic-v1
    description: plumbing and determinism only; never select models on it
    matches:
      - {name: synthetic, params: {seed: 0}}

A record is one JSON file per run in experiments/<member>/, named <utc>-<member>-<pipeline>: one
writer per folder, so parallel runs by four people never collide in git. Records hold aggregate metrics only (no
per-frame data), the exact config, its digest and the git state, so any number can be traced to
the code that produced it.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from . import pipeline
from .evaluation.score import evaluate, summarize
from .paths import EXPERIMENTS_DIR, REPO_ROOT
from .pipeline import PipelineConfig, Step
from .registry import SOURCES


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    matches: tuple[Step, ...]
    source_path: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        matches = tuple(Step.parse(entry) for entry in raw.get("matches") or [])
        if not matches:
            raise ValueError(f"{path}: a suite needs at least one match")
        return cls(raw["name"], raw.get("description", ""), matches, str(path))


def evaluate_suite(config: PipelineConfig, suite: Suite, *, use_cache: bool = True) -> dict:
    """Run the pipeline on every match of the suite and score it with the official metrics."""
    per_match: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for entry in suite.matches:
        bundle = SOURCES.get(entry.name).fn(**entry.params)
        if bundle.labels is None:
            raise ValueError(
                f"source {entry.name!r} returned no labels; suites need labelled matches"
            )
        began = time.perf_counter()
        result = pipeline.run(config, bundle, use_cache=use_cache)
        metrics = summarize(evaluate(bundle.labels, result.labels))
        metrics["runtime_s"] = time.perf_counter() - began
        per_match[bundle.meta.match_id] = metrics

    keys = sorted(set.intersection(*(set(m) for m in per_match.values())))
    values = {key: np.array([m[key] for m in per_match.values()], dtype=np.float64) for key in keys}
    return {
        "n_matches": len(per_match),
        "mean": {k: float(v.mean()) for k, v in values.items()},
        "std": {k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in values.items()},
        "min": {k: float(v.min()) for k, v in values.items()},
        "per_match": per_match,
        "runtime_s": time.perf_counter() - started,
    }


def write_record(
    config: PipelineConfig,
    suite: Suite,
    results: dict,
    *,
    member: str,
    issue: int | None = None,
    notes: str = "",
    directory: Path = EXPERIMENTS_DIR,
) -> Path:
    now = datetime.now(UTC)
    record_id = f"{now:%Y%m%dT%H%M%SZ}-{_slug(member)}-{_slug(config.name)}"
    record = {
        "id": record_id,
        "created_utc": now.isoformat(timespec="seconds"),
        "member": member,
        "issue": issue,
        "notes": notes,
        "pipeline": {
            "name": config.name,
            "path": _relative(config.source_path),
            "digest": config.digest(),
            "config": config.to_dict(),
        },
        "suite": {
            "name": suite.name,
            "path": _relative(suite.source_path),
            "matches": [entry.to_dict() for entry in suite.matches],
        },
        "git": git_state(),
        "results": results,
    }
    folder = directory / _slug(member)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record_id}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_records(directory: Path = EXPERIMENTS_DIR) -> list[dict]:
    records = []
    for path in sorted(directory.rglob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue  # a half-written file must not take the board down
    return records


def leaderboard(
    suite_name: str | None = None,
    metric: str = "spell_f1_mean",
    directory: Path = EXPERIMENTS_DIR,
) -> list[dict]:
    """One row per record, best first. Only rows from the same suite are comparable."""
    rows = []
    for record in load_records(directory):
        if suite_name and record["suite"]["name"] != suite_name:
            continue
        results = record["results"]
        if metric not in results["mean"]:
            continue
        rows.append(
            {
                "id": record["id"],
                "member": record["member"],
                "pipeline": record["pipeline"]["name"],
                "suite": record["suite"]["name"],
                "n": results["n_matches"],
                "mean": results["mean"][metric],
                "std": results["std"][metric],
                "min": results["min"][metric],
                "macro_f1": results["mean"].get("macro_f1", float("nan")),
                "dirty": record["git"].get("dirty", True),
            }
        )
    return sorted(rows, key=lambda row: (-row["mean"], row["id"]))


def git_state() -> dict:
    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return done.stdout.strip() if done.returncode == 0 else ""

    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "x"


def _relative(path: str | None) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
