"""Inference from saved weights: chunked over a whole match, deterministic on one CPU thread.

    loaded = load_checkpoint("models/team.pt")
    prediction = predict_logits([loaded], tracking)   # .logits (T, 3), NaN outside the periods
    p_home = prediction.team_probability()            # P(home | alive), NaN where no opinion

Determinism (AGENTS.md rule 4): on the CPU with `threads=1` and float32 the output is bit-exact
from run to run on one machine; more threads change the last bits of the matrix products, which
can flip a label that sits on a tie, so the submit step pins one thread (`cpu_determinism`, which
also turns on `torch.use_deterministic_algorithms` for the block and restores both afterwards).
Nothing here draws random numbers or reads the clock. Across different CPUs the instruction sets
used by the BLAS kernels may differ, so bit-exactness across machines is not promised
(docs/ml/README.md, "Known gaps").

A match is cut into chunks of `chunk` kept frames; each is computed with `receptive_half` frames of
context on both sides (clipped at the period edges), so the stitched result equals one pass over
the whole period up to float rounding. `views` > 1 averages the logits over mirrored copies of the
input (y flip, team swap), and several checkpoints average their logits too (a seed ensemble).
A model with `switch_head` also averages its switch logit over those views and checkpoints; a team
swap does not reorder that scalar. Without the head, `Prediction.switch` stays unset and the class
logits are unchanged.
A model with `touch_head` averages its touch logit the same way. A y flip leaves that scalar as
the forward wrote it, and a team swap does not reorder it, because a touch has no team. Without
the head, `Prediction.touch` stays unset.

Signals: the tracking only. Fails with an unobserved team (no valid token): the frame gets no
opinion (NaN), as do frames outside the periods.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit

from ..contracts import Tracking
from .checkpoint import Loaded
from .geometry import SWAP_CLASSES, symmetric
from .spec import Col, TokenSpec
from .windows import build_match_tokens, plan_chunks

# (flip_y, swap) per view: 1 = as given, 2 = + y mirror, 4 = + team swap and both
VIEWS = {
    1: ((False, False),),
    2: ((False, False), (True, False)),
    4: ((False, False), (True, False), (False, True), (True, True)),
}


def configure_cpu(threads: int = 1) -> None:
    """Pin torch to `threads` CPU threads and to deterministic kernels, for the rest of the
    process (best effort for the inter-op pool, which torch only lets you set before it first runs
    parallel work). Prefer `cpu_determinism` where the process does other torch work."""
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


@contextmanager
def cpu_determinism(threads: int = 1) -> Iterator[None]:
    """`configure_cpu` for the duration of a block, then the previous thread count and mode."""
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    configure_cpu(threads)
    try:
        yield
    finally:
        torch.set_num_threads(threads_before)
        torch.use_deterministic_algorithms(deterministic_before)


@dataclass
class Prediction:
    logits: np.ndarray  # (T, 3) float32 over (H, A, D); NaN where there is no opinion
    opinion: np.ndarray  # (T,) bool
    switch: np.ndarray | None = None  # (T,) float32 switch logit; None without the head
    touch: np.ndarray | None = None  # (T,) float32 touch logit; None without the head

    def proba(self) -> np.ndarray:
        """(T, 3) float32 softmax of the logits, NaN rows where there is no opinion."""
        z = self.logits.astype(np.float64)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    def team_probability(self) -> np.ndarray:
        """(T,) float32 P(home | alive) = sigmoid(z_H - z_A), NaN where there is no opinion."""
        z = self.logits.astype(np.float64)
        return expit(z[:, 0] - z[:, 1]).astype(np.float32)


def logits_from_tokens(
    model: torch.nn.Module,
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int = 2048,
    views: int = 1,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """(T, 3) float32 logits of `model` over the played frames of a match's token tensor, NaN
    elsewhere. See the module docstring for the chunking and the views."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    out = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    margin = model.receptive_half
    model.eval()
    with torch.inference_mode():
        for part in plan_chunks(tuple(periods), chunk, margin):
            x = torch.from_numpy(tokens[part.lo : part.hi]).to(device).unsqueeze(0)
            total = None
            for flip_y, swap in VIEWS[views]:
                z = model(symmetric(x, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else x)
                z = z["logits"][0].float()
                z = z[:, SWAP_CLASSES] if swap else z
                total = z if total is None else total + z
            keep = slice(part.keep_lo - part.lo, part.keep_hi - part.lo)
            out[part.keep_lo : part.keep_hi] = (total[keep] / len(VIEWS[views])).cpu().numpy()
    return out


def logits_and_switch_from_tokens(
    model: torch.nn.Module,
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int = 2048,
    views: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Class logits and the switch logit. The class half matches `logits_from_tokens`.

    The switch logit is averaged over the views and is not reordered by a team swap.
    """
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    logits = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    switch = np.full(len(tokens), np.nan, dtype=np.float32)
    margin = model.receptive_half
    model.eval()
    n_views = len(VIEWS[views])
    with torch.inference_mode():
        for part in plan_chunks(tuple(periods), chunk, margin):
            x = torch.from_numpy(tokens[part.lo : part.hi]).to(device).unsqueeze(0)
            total_z = None
            total_s = None
            for flip_y, swap in VIEWS[views]:
                out = model(symmetric(x, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else x)
                z = out["logits"][0].float()
                z = z[:, SWAP_CLASSES] if swap else z
                s = out["switch"][0].float()
                total_z = z if total_z is None else total_z + z
                total_s = s if total_s is None else total_s + s
            keep = slice(part.keep_lo - part.lo, part.keep_hi - part.lo)
            logits[part.keep_lo : part.keep_hi] = (total_z[keep] / n_views).cpu().numpy()
            switch[part.keep_lo : part.keep_hi] = (total_s[keep] / n_views).cpu().numpy()
    return logits, switch


def logits_and_scalars_from_tokens(
    model: torch.nn.Module,
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int = 2048,
    views: int = 1,
    device: str | torch.device = "cpu",
    names: tuple[str, ...] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Class logits and one scalar logit per name. The class half matches `logits_from_tokens`.

    Each scalar is averaged over the views and is not reordered by a team swap or a y flip.
    """
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    if not names:
        raise ValueError("logits_and_scalars_from_tokens needs at least one head")
    logits = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    scalars = {name: np.full(len(tokens), np.nan, dtype=np.float32) for name in names}
    margin = model.receptive_half
    model.eval()
    n_views = len(VIEWS[views])
    with torch.inference_mode():
        for part in plan_chunks(tuple(periods), chunk, margin):
            x = torch.from_numpy(tokens[part.lo : part.hi]).to(device).unsqueeze(0)
            total_z = None
            total_s: dict[str, torch.Tensor | None] = {name: None for name in names}
            for flip_y, swap in VIEWS[views]:
                out = model(symmetric(x, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else x)
                z = out["logits"][0].float()
                z = z[:, SWAP_CLASSES] if swap else z
                total_z = z if total_z is None else total_z + z
                for name in names:
                    value = out[name][0].float()
                    total_s[name] = value if total_s[name] is None else total_s[name] + value
            keep = slice(part.keep_lo - part.lo, part.keep_hi - part.lo)
            logits[part.keep_lo : part.keep_hi] = (total_z[keep] / n_views).cpu().numpy()
            for name in names:
                accumulated = total_s[name]
                assert accumulated is not None
                scalars[name][part.keep_lo : part.keep_hi] = (
                    (accumulated[keep] / n_views).cpu().numpy()
                )
    return logits, scalars


def predict_logits(
    checkpoints: Sequence[Loaded],
    tracking: Tracking,
    *,
    chunk: int = 2048,
    views: int = 1,
    device: str | torch.device = "cpu",
    threads: int | None = 1,
) -> Prediction:
    """Mean logits of `checkpoints` (all with one token spec) over a match's tracking.

    `threads` pins the CPU path (None leaves torch alone). Frames with no observed player, and
    frames outside the periods, have no opinion.
    """
    if not checkpoints:
        raise ValueError("predict_logits needs at least one checkpoint")
    spec = checkpoints[0].spec
    if any(c.spec != spec for c in checkpoints):
        raise ValueError("checkpoints in one ensemble must share a token spec")
    pinned = threads is not None and torch.device(device).type == "cpu"
    tokens = build_match_tokens(tracking, spec)
    total = np.zeros((tracking.n_frames, 3), dtype=np.float64)
    wants_switch = any(getattr(item.model.cfg, "switch_head", False) for item in checkpoints)
    wants_touch = any(getattr(item.model.cfg, "touch_head", False) for item in checkpoints)
    switch_total = np.zeros(tracking.n_frames, dtype=np.float64) if wants_switch else None
    touch_total = np.zeros(tracking.n_frames, dtype=np.float64) if wants_touch else None
    n_switch = 0
    n_touch = 0
    for loaded in checkpoints:
        with cpu_determinism(threads) if pinned else nullcontext():
            model = loaded.model.to(device)
            has_switch = bool(getattr(model.cfg, "switch_head", False))
            has_touch = bool(getattr(model.cfg, "touch_head", False))
            # A touch head shares the forward with switch when both are on, so the class logits
            # are added once. Switch alone stays on the path that existed before the touch head.
            if has_touch:
                names = tuple(
                    name for name, flag in (("switch", has_switch), ("touch", True)) if flag
                )
                logits_i, scalars = logits_and_scalars_from_tokens(
                    model,
                    spec,
                    tokens,
                    tracking.meta.periods,
                    chunk=chunk,
                    views=views,
                    device=device,
                    names=names,
                )
                total += logits_i
                if has_switch:
                    assert switch_total is not None
                    switch_total += scalars["switch"]
                    n_switch += 1
                assert touch_total is not None
                touch_total += scalars["touch"]
                n_touch += 1
            elif has_switch:
                logits_i, sw = logits_and_switch_from_tokens(
                    model,
                    spec,
                    tokens,
                    tracking.meta.periods,
                    chunk=chunk,
                    views=views,
                    device=device,
                )
                total += logits_i
                assert switch_total is not None
                switch_total += sw
                n_switch += 1
            else:
                total += logits_from_tokens(
                    model,
                    spec,
                    tokens,
                    tracking.meta.periods,
                    chunk=chunk,
                    views=views,
                    device=device,
                )
        loaded.model.to("cpu")
    logits = (total / len(checkpoints)).astype(np.float32)
    k = spec.tokens_per_team
    seen_players = tokens[:, : 2 * k, Col.VALID].sum(axis=1) > 0
    opinion = tracking.meta.played_mask() & seen_players
    logits[~opinion] = np.nan
    played = tracking.meta.played_mask()
    switch = None
    touch = None
    if switch_total is not None:
        switch = (switch_total / n_switch).astype(np.float32)
        switch[~played] = np.nan
    if touch_total is not None:
        touch = (touch_total / n_touch).astype(np.float32)
        touch[~played] = np.nan
    if switch is None and touch is None:
        return Prediction(logits=logits, opinion=opinion)
    return Prediction(logits=logits, opinion=opinion, switch=switch, touch=touch)
