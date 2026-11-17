"""Checkpoint files: weights plus everything needed to rebuild the model and its inputs.

A checkpoint is one `.pt` file holding only tensors and plain Python values, so it loads with
`torch.load(..., weights_only=True)` (no pickled code). It records the token spec, the
architecture name and its config, so an inference run needs nothing but the file (and the module
that registers the architecture); the feature version guards against a token layout that changed
after training. A file without an architecture name is the baseline `team_net`. `sha256` pins the
exact bytes in a pipeline config, which is what keeps the evidence cache honest when a file is
overwritten.

`save_checkpoint` writes atomically (a temporary file, then a rename), so an interrupted run never
leaves a truncated checkpoint behind. A file under `models/` or similar is shipped by a human
decision; nothing here writes into the repository tree (the default output is under the cache).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import get_architecture
from .spec import FEATURE_VERSION, TokenSpec
from .weights import sha256_file

FORMAT = 1


@dataclass
class Loaded:
    """A checkpoint read back: the model in eval mode on the CPU, its spec, and the raw payload."""

    model: nn.Module
    spec: TokenSpec
    meta: dict[str, Any]
    sha256: str
    payload: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    spec: TokenSpec,
    meta: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `model` and its configs to `path` atomically. `extra` (optimizer state, the step)
    makes it a training checkpoint; without it the file is an inference checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "feature_version": FEATURE_VERSION,
        "token_spec": spec.to_dict(),
        "arch": model.arch,
        "model_config": model.cfg.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": meta,
        **(extra or {}),
    }
    partial = path.with_name(path.name + ".part")
    torch.save(payload, partial)
    os.replace(partial, path)
    return path


def load_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: checkpoint format {payload.get('format')!r}, expected {FORMAT}")
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"{path}: trained on feature version {payload.get('feature_version')!r}, "
            f"this code builds {FEATURE_VERSION!r}"
        )
    return payload


def load_checkpoint(path: str | Path) -> Loaded:
    """The model (eval mode, CPU, float32), its token spec and metadata, and the file's SHA-256."""
    payload = load_payload(path)
    config_cls, model_cls = get_architecture(payload.get("arch", "team_net"))
    model = model_cls(config_cls.from_dict(payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return Loaded(
        model=model,
        spec=TokenSpec.from_dict(payload["token_spec"]),
        meta=payload.get("meta", {}),
        sha256=sha256_file(path),
        payload=payload,
    )
