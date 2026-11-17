"""Compose registered stages from a YAML config and run them on one match.

    name: my_pipeline
    description: one line on what this tries and why (link the issue / finding)
    evidence:
      - name: nearest_player
        params: {when_unseen: abstain}
    fusion: first                      # or {name: log_pool, params: {weights: [1.0, 0.5]}}
    decoder:
      name: majority_vote
      params: {half_window: 50}

Evidence is cached per (component, version, params, match fingerprint) under cache/, so iterating
on fusion or decoding never recomputes features. Synthetic matches are never cached (cheap).

An optional `preprocess:` list edits the bundle before every other stage (evidence, decoder and
post-process all read the edited bundle). The steps are `ball_flight_gate` (E127, #86:
`aaaag.features.ball_flight.gated_bundle`) and `ball_candidates_fill` (#86, video ball candidates
from a sha256-pinned file: `aaaag.features.ball_candidates`). A config without the key keeps its digest and output.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contracts import Evidence, MatchBundle, validate_labels
from .paths import cache_dir
from .registry import DECODERS, EVIDENCE, FUSION, POSTPROCESS


@dataclass(frozen=True)
class Step:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: str | dict) -> Step:
        if isinstance(raw, str):
            return cls(raw)
        unknown = set(raw) - {"name", "params"}
        if unknown:
            raise ValueError(f"unknown keys in step {raw}: {sorted(unknown)}")
        return cls(raw["name"], dict(raw.get("params") or {}))

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params}


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    description: str
    evidence: tuple[Step, ...]
    fusion: Step
    decoder: Step
    source_path: str | None = None
    postprocess: tuple[
        Step, ...
    ] = ()  # label -> label steps after the decoder (they may read the bundle)
    preprocess: tuple[Step, ...] = ()  # bundle -> bundle steps before the evidence (E127)

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw, source_path=str(path))

    @classmethod
    def from_dict(cls, raw: dict, source_path: str | None = None) -> PipelineConfig:
        unknown = set(raw) - {
            "name",
            "description",
            "evidence",
            "fusion",
            "decoder",
            "postprocess",
            "preprocess",
        }
        if unknown:
            raise ValueError(f"unknown pipeline keys: {sorted(unknown)}")
        evidence = tuple(Step.parse(step) for step in raw.get("evidence") or [])
        if not evidence:
            raise ValueError("a pipeline needs at least one evidence step")
        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            evidence=evidence,
            fusion=Step.parse(raw.get("fusion", "first")),
            decoder=Step.parse(raw["decoder"]),
            source_path=source_path,
            postprocess=tuple(Step.parse(step) for step in raw.get("postprocess") or []),
            preprocess=tuple(_check_preprocess(Step.parse(s)) for s in raw.get("preprocess") or []),
        )

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "description": self.description,
            "evidence": [step.to_dict() for step in self.evidence],
            "fusion": self.fusion.to_dict(),
            "decoder": self.decoder.to_dict(),
        }
        if self.postprocess:  # absent for old configs, so their digests do not change
            out["postprocess"] = [step.to_dict() for step in self.postprocess]
        if self.preprocess:  # likewise
            out["preprocess"] = [step.to_dict() for step in self.preprocess]
        return out

    def digest(self) -> str:
        """Hash of everything that affects the output (not the name or description)."""
        payload = self.to_dict()
        payload.pop("name")
        payload.pop("description")
        payload["versions"] = {
            "evidence": [EVIDENCE.get(s.name).version for s in self.evidence],
            "fusion": FUSION.get(self.fusion.name).version,
            "decoder": DECODERS.get(self.decoder.name).version,
        }
        if self.postprocess:
            payload["versions"]["postprocess"] = [
                POSTPROCESS.get(s.name).version for s in self.postprocess
            ]
        if self.preprocess:
            payload["versions"]["preprocess"] = [
                PREPROCESS_VERSIONS[s.name] for s in self.preprocess
            ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class PipelineResult:
    labels: np.ndarray
    proba: np.ndarray
    evidence: list[Evidence]
    timings: dict[str, float]


PREPROCESS_VERSIONS = {"ball_flight_gate": "1", "ball_candidates_fill": "1"}


def _check_preprocess(step: Step) -> Step:
    if step.name not in PREPROCESS_VERSIONS:
        raise ValueError(
            f"unknown preprocess step {step.name!r}; known: {sorted(PREPROCESS_VERSIONS)}"
        )
    return step


def preprocess(config: PipelineConfig, bundle: MatchBundle) -> MatchBundle:
    """The bundle after the config's `preprocess` steps (the same object when there are none)."""
    for step in config.preprocess:
        if step.name == "ball_flight_gate":
            from .features import ball_flight

            bundle = ball_flight.gated_bundle(bundle, **step.params)
        elif step.name == "ball_candidates_fill":  # needs path and sha256 params (#86)
            from .features import ball_candidates

            bundle = ball_candidates.filled_bundle(bundle, **step.params)
    return bundle


def run(config: PipelineConfig, bundle: MatchBundle, *, use_cache: bool = True) -> PipelineResult:
    timings: dict[str, float] = {}
    if config.preprocess:
        started = time.perf_counter()
        bundle = preprocess(config, bundle)
        timings["preprocess"] = time.perf_counter() - started
    n_frames = bundle.meta.n_frames
    evidence = []
    for step in config.evidence:
        started = time.perf_counter()
        evidence.append(_evidence(step, bundle, use_cache))
        timings[f"evidence:{step.name}"] = time.perf_counter() - started

    started = time.perf_counter()
    proba = FUSION.get(config.fusion.name).fn(evidence, bundle.meta, **config.fusion.params)
    timings["fusion"] = time.perf_counter() - started
    if proba.shape != (n_frames, 3) or proba.dtype != np.float32:
        raise ValueError(f"fusion {config.fusion.name!r} returned {proba.shape} {proba.dtype}")

    started = time.perf_counter()
    decoder = DECODERS.get(config.decoder.name)
    params, proba = _prepare_decoder(decoder.fn, config.decoder.params, proba)
    labels = decoder.fn(proba, bundle.meta, **params)
    timings["decoder"] = time.perf_counter() - started
    validate_labels(labels, n_frames)
    for step in config.postprocess:
        started = time.perf_counter()
        fn = POSTPROCESS.get(step.name).fn
        extra = {"proba": proba} if "proba" in inspect.signature(fn).parameters else {}
        labels = fn(labels, bundle, **step.params, **extra)
        timings[f"postprocess:{step.name}"] = time.perf_counter() - started
        validate_labels(labels, n_frames)
    return PipelineResult(labels=labels, proba=proba, evidence=evidence, timings=timings)


def _evidence(step: Step, bundle: MatchBundle, use_cache: bool) -> Evidence:
    component = EVIDENCE.get(step.name)
    n_frames = bundle.meta.n_frames
    cacheable = use_cache and bundle.meta.source != "synthetic"
    path = None
    if cacheable:
        key = hashlib.sha256(
            json.dumps(
                {
                    "name": step.name,
                    "version": component.version,
                    "params": step.params,
                    "bundle": bundle.fingerprint(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
        path = cache_dir() / bundle.meta.match_id / "evidence" / f"{step.name}-{key}.npz"
        if path.is_file():
            return _load_cached_evidence(path, step.name, n_frames)

    result = component.fn(bundle, **step.params)
    if not isinstance(result, Evidence):
        raise TypeError(f"evidence {step.name!r} returned {type(result).__name__}, not Evidence")
    result.validate(n_frames)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_cached_evidence(path, result)
    return result


def _save_cached_evidence(path: Path, result: Evidence) -> None:
    """Write one evidence file. No channels uses the historical `proba`-only archive.

    The archive is written to a temporary file in the same directory and then renamed over `path`,
    so a concurrent reader (several training or scoring jobs evaluating the same match) never
    opens a half-written file. A rename within one directory is atomic on POSIX and on Windows.
    """
    arrays: dict[str, np.ndarray] = {"proba": result.proba}
    for name in sorted(result.channels):
        arrays[f"channel_{name}"] = result.channels[name]
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_cached_evidence(path: Path, name: str, n_frames: int) -> Evidence:
    """Load an evidence file. An archive with only `proba` comes back with empty channels."""
    with np.load(path, allow_pickle=False) as stored:
        extra = [key for key in stored.files if key != "proba"]
        if not extra:
            return Evidence(name, stored["proba"]).validate(n_frames)
        channels: dict[str, np.ndarray] = {}
        for key in extra:
            if not key.startswith("channel_"):
                raise ValueError(f"unexpected array {key!r} in {path}")
            channels[key.removeprefix("channel_")] = np.array(stored[key], copy=True)
        proba = np.array(stored["proba"], copy=True)
    return Evidence(name, proba, channels).validate(n_frames)


def _prepare_decoder(
    fn: Any, params: dict[str, Any], proba: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Pass fusion channels whose names are decoder parameters. No channels leaves both as given."""
    channels = getattr(proba, "channels", None)
    if not channels:
        return params, proba
    accepted = inspect.signature(fn).parameters
    extra = {key: value for key, value in channels.items() if key in accepted and key not in params}
    plain = np.array(proba, dtype=np.float32, copy=True)
    if not extra:
        return params, plain
    return {**params, **extra}, plain
