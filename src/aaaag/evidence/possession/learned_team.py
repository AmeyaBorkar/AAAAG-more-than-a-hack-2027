"""learned_team (#82): P(home | alive) from a network trained on OTHER matches.

A per-frame set encoder over the nearest players of each team and the ball, then a non-causal
temporal convolution stack (src/aaaag/ml/model.py). It is trained leave-one-match-out on the IDSSE
dev matches degraded by `aaaag.degrade`, with random ball masking, so it does not need the ball and
does not read a missing ball as a dead ball (docs/ml/README.md). Its weights are a small file
named in the pipeline config and pinned by `sha256`.

Output: rows (p, 1 - p, 0) with p = P(home | alive), like `control_possession`, so it speaks only to
the team factor of `factorised` fusion and the ball-state producer keeps P(dead). `output: joint`
returns the network's own (H, A, D) softmax instead. NaN rows outside the periods and where no
player is observed.

When the checkpoint was trained with `switch_head`, the Evidence also carries
`channels["switch"]`: a (T,) float32 logit, the mean over the models and the views, NaN outside
the periods. A network without the head leaves that channel unset and the probabilities unchanged.
When it was trained with `touch_head`, the Evidence also carries `channels["touch"]`, the same
kind of mean. A touch has no team, so a team swap does not reorder it. Nothing reads that channel
yet. A network without the head does not set it.

Uses: slot positions and teams, the ball where seen and the length of its gaps; never slot
identities, tags or labels. Fails where the training matches do not look like the target: another
tracker's noise, label definitions that differ from IDSSE's (docs/technical_report/REPORT.md,
section 6), or a team that is mostly unobserved.

With `weights` unset it has nothing to say and returns NaN for every frame (this is what the
contract tests run). A set `weights` that does not exist, or whose SHA-256 differs from `sha256`,
raises. `weights` may be a list: the logits of the models are averaged (a seed ensemble).
`backend: torch` (the default) runs the `.pt` checkpoint in torch, float32, one CPU thread
(`device: cuda` is for evaluation sweeps only). `backend: numpy` loads the JSON weight file
written by `python -m aaaag.ml.export` and runs the float64 numpy forward pass, with torch not
installed. `device` and `threads` apply to the torch backend only. `workers` applies to the
numpy backend only: 1 is the serial loop, and 0 or a negative value uses up to four cores.

`ball_deblur` is an optional JSON file from `python -m aaaag.ml.deblur train`. When it is set,
team inference reads a copy of the bundle whose `tracking.ball_xy` has been through `deblur_ball`;
the caller's bundle is not modified. Unset, the ball and the probabilities are unchanged.
`ball_deblur_sha256` pins that file the same way `sha256` pins the team weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...contracts import Evidence, MatchBundle, MatchMeta
from ...ml import weights as weight_files
from ...registry import EVIDENCE

NAME = "learned_team"


@EVIDENCE.register(NAME, version="1", example={})
def learned_team(
    bundle: MatchBundle,
    *,
    weights: str | Sequence[str] | None = None,
    sha256: str | Sequence[str] | None = None,
    chunk: int = 2048,
    views: int = 1,
    device: str = "cpu",
    threads: int = 1,
    workers: int = 1,
    output: str = "team",
    backend: str = "torch",
    ball_deblur: str | None = None,
    ball_deblur_sha256: str | None = None,
) -> Evidence:
    """P(home | alive) per frame from the checkpoint(s) in `weights` (paths from the repository
    root). `views` 1, 2 or 4 averages the logits over mirrored copies of the input.
    `backend` is `torch` (a `.pt` file) or `numpy` (a JSON weight file; torch is not imported).
    `workers` is the numpy process pool (1 serial; 0 or negative for up to four cores). The torch
    backend ignores it. `ball_deblur` replaces the ball on a copy of the bundle before that."""
    if output not in ("team", "joint"):
        raise ValueError(f"output must be 'team' or 'joint', got {output!r}")
    if backend not in ("torch", "numpy"):
        raise ValueError(f"backend must be 'torch' or 'numpy', got {backend!r}")
    n_frames = bundle.meta.n_frames
    paths = [weight_files.resolve(p) for p in weight_files.as_list(weights)]
    if paths:
        weight_files.check_files(paths, weight_files.as_list(sha256))
    if ball_deblur is not None:
        from ...ml.np_deblur import deblurred_bundle

        bundle = deblurred_bundle(bundle, ball_deblur, ball_deblur_sha256)
    if not paths:
        return Evidence(NAME, np.full((n_frames, 3), np.nan, dtype=np.float32))
    if backend == "numpy":
        from ...ml.np_infer import predict_logits_np

        prediction = predict_logits_np(
            paths, bundle.tracking, chunk=chunk, views=views, workers=workers
        )
    else:
        try:
            from ...ml.checkpoint import load_checkpoint
            from ...ml.infer import predict_logits
        except ImportError as error:  # torch missing: the normal environment has none
            raise ImportError(
                "learned_team runs a torch model; install the DL environment "
                "(python -m pip install -r requirements-dl.txt, docs/ml/README.md)"
            ) from error
        loaded = [load_checkpoint(path) for path in paths]
        prediction = predict_logits(
            loaded, bundle.tracking, chunk=chunk, views=views, device=device, threads=threads
        )
    if output == "joint":
        evidence = Evidence(NAME, prediction.proba())
    else:
        evidence = Evidence.from_factors(
            NAME, np.zeros(n_frames, dtype=np.float32), prediction.team_probability()
        )
    return _attach_touch(_attach_switch(evidence, prediction, bundle.meta), prediction, bundle.meta)


def _attach_switch(evidence: Evidence, prediction: object, meta: MatchMeta) -> Evidence:
    """Copy the ensemble switch logit onto `evidence`, NaN outside the periods.

    No logit (the head is off) returns `evidence` with its channels untouched, which is empty.
    """
    switch = getattr(prediction, "switch", None)
    if switch is None:
        return evidence
    channel = np.array(switch, dtype=np.float32, copy=True)
    channel[~meta.played_mask()] = np.nan
    evidence.channels = {"switch": channel}
    return evidence


def _attach_touch(evidence: Evidence, prediction: object, meta: MatchMeta) -> Evidence:
    """Copy the ensemble touch logit onto `evidence`, NaN outside the periods.

    No logit (the head is off) returns `evidence` unchanged, switch channel included. A touch
    has no team. Nothing in the decoder reads this channel yet.
    """
    touch = getattr(prediction, "touch", None)
    if touch is None:
        return evidence
    channel = np.array(touch, dtype=np.float32, copy=True)
    channel[~meta.played_mask()] = np.nan
    evidence.channels = {**evidence.channels, "touch": channel}
    return evidence
