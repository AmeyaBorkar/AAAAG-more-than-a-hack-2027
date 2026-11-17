"""Train a residual CNN that undoes Gaussian smoothing of a ball track.

The match ball is smoothed, roughly an 8-frame Gaussian at 25 fps, which blunts the kinks the
touch features read. This module learns the inverse on other matches: smooth a clean ball with
``smooth_ball_track``, sometimes punch gaps in it, and ask the network for the correction that
puts the clean position back. Inference itself is ``np_deblur`` and does not need torch.

    python -m aaaag.ml.deblur train --matches metrica/1 metrica/2 metrica/3 --seed 0 --out w.json
    python -m aaaag.ml.deblur eval --weights w.json --matches idsse/J03WMX

Signals: the ball's ``(x, y)`` in metres and whether the frame is finite. Players, tags and
labels are not read. Where it fails: the network has no sigma input, so it learns one correction
for a blur drawn from ``{0, 2, 4, 6, 8, 10}`` frames; a tracker blur outside that set, or a run
shorter than 8 frames (left unchanged at inference), is not what it was trained to undo.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import MatchBundle, Period
from .np_deblur import (
    CHANNELS,
    DILATIONS,
    KERNEL,
    MIN_RUN,
    X_SCALE,
    Y_SCALE,
    as_bundles,
    ball_features,
    dump_deblur_weights,
    evaluate_deblur,
    format_eval,
    smooth_for_deblur,
)

LR = 2e-3
DEFAULT_STEPS = 3000
DEFAULT_BATCH = 64
DEFAULT_WINDOW = 512
PROBE_BATCHES = 4
TRAIN_SIGMAS = (0, 2, 4, 6, 8, 10)
DROPOUT_P = 0.2
DV_WEIGHT = 0.5
DV_WINDOW = 2
DEFAULT_MATCHES = ("metrica/1", "metrica/2", "metrica/3")


class DeblurNet(nn.Module):
    """1-D residual CNN. Features ``(B, T, 3)`` -> ``(dx, dy)`` metres ``(B, T, 2)``.

    Five dilated convolutions, kernel 5, dilations 1, 2, 4, 8, 16, 32 channels, GELU.
    The first maps 3 channels to 32, so it has no identity skip. Each later layer is
    ``h + GELU(conv(h))``. A pointwise head, without a GELU, emits the signed correction.
    """

    def __init__(
        self,
        channels: int = CHANNELS,
        kernel: int = KERNEL,
        dilations: tuple[int, ...] = DILATIONS,
    ) -> None:
        super().__init__()
        if channels < 1 or not dilations:
            raise ValueError("deblur net needs channels and at least one dilation")
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so same-padding is symmetric")
        self.channels = int(channels)
        self.kernel = int(kernel)
        self.dilations = tuple(int(d) for d in dilations)
        convs = []
        cin = 3
        for dilation in self.dilations:
            padding = dilation * (kernel - 1) // 2
            convs.append(
                nn.Conv1d(cin, self.channels, kernel, dilation=dilation, padding=padding, bias=True)
            )
            cin = self.channels
        self.convs = nn.ModuleList(convs)
        self.head = nn.Conv1d(self.channels, 2, kernel_size=1, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, T, 3)`` -> ``(B, T, 2)`` metres, added to the input position by the caller."""
        h = features.transpose(1, 2)
        for index, conv in enumerate(self.convs):
            y = F.gelu(conv(h))
            h = y if index == 0 else h + y
        return self.head(h).transpose(1, 2)


def config_from_module(model: DeblurNet) -> dict[str, object]:
    """The JSON config that describes ``model``. Scales and the short-run rule are fixed."""
    return {
        "channels": model.channels,
        "kernel": model.kernel,
        "dilations": list(model.dilations),
        "x_scale": X_SCALE,
        "y_scale": Y_SCALE,
        "min_run": MIN_RUN,
    }


def export_weights(model: nn.Module, path: str | Path, config: dict | None = None) -> Path:
    """Write ``model`` as a deblur JSON file. Float64 weights are stored as float32."""
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    cfg = config_from_module(model) if config is None and isinstance(model, DeblurNet) else config
    return dump_deblur_weights(path, arrays, cfg)


@contextmanager
def _deterministic_cpu():
    """One CPU thread and deterministic kernels for the body of ``train_deblur``."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def _eligible(
    bundles: Sequence[MatchBundle], window: int
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """``(bundle index, first legal start, number of starts)`` for every period that fits."""
    spans: list[tuple[int, int, int]] = []
    counts: list[int] = []
    for index, bundle in enumerate(bundles):
        for period in bundle.meta.periods:
            n_starts = period.length - window + 1
            if n_starts > 0:
                spans.append((index, period.start, n_starts))
                counts.append(n_starts)
    if not spans:
        raise ValueError(f"no period is {window} frames or longer")
    return spans, np.asarray(counts, dtype=np.float64)


def _pick_start(
    spans: Sequence[tuple[int, int, int]], counts: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    """One window start, uniform over legal ``(match, start)`` pairs. Draws random, then integers."""
    total = float(counts.sum())
    pick = int(np.searchsorted(np.cumsum(counts), float(rng.random()) * total, side="right"))
    pick = min(pick, len(spans) - 1)
    index, lo, n_starts = spans[pick]
    return index, lo + int(rng.integers(0, n_starts))


def _sample_batch(
    bundles: Sequence[MatchBundle],
    spans: Sequence[tuple[int, int, int]],
    counts: np.ndarray,
    window: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(clean, blurred, fps)``.

    Per row the generator draws a window, a sigma from ``{0, 2, 4, 6, 8, 10}``, then a dropout
    coin. On a hit (probability 0.2) it draws 1 to 3 gaps of 5 to 50 frames and writes NaNs into
    the smoothed track only. The clean row is the unsmoothed window.
    """
    clean = np.empty((batch_size, window, 2), dtype=np.float64)
    blurred = np.empty((batch_size, window, 2), dtype=np.float64)
    fps = np.empty(batch_size, dtype=np.float64)
    span = (Period(0, 0, window),)
    for row in range(batch_size):
        index, start = _pick_start(spans, counts, rng)
        bundle = bundles[index]
        raw = np.array(bundle.tracking.ball_xy[start : start + window], dtype=np.float64, copy=True)
        sigma = float(TRAIN_SIGMAS[int(rng.integers(0, len(TRAIN_SIGMAS)))])
        smooth = np.asarray(smooth_for_deblur(raw, span, sigma), dtype=np.float64)
        if float(rng.random()) < DROPOUT_P:
            n_gaps = int(rng.integers(1, 4))
            for _gap in range(n_gaps):
                length = min(int(rng.integers(5, 51)), window)
                gap_at = int(rng.integers(0, window - length + 1))
                smooth[gap_at : gap_at + length] = np.nan
        clean[row] = raw
        blurred[row] = smooth
        fps[row] = bundle.meta.fps
    return clean, blurred, fps


def _velocity_change(pos: torch.Tensor, fps: torch.Tensor) -> torch.Tensor:
    """``(B, T-4, 2)`` of ``v_out - v_in`` in m/s, ``w=2``, aligned with frames ``[2, T-2)``."""
    w = DV_WINDOW
    scale = (fps / w).view(-1, 1, 1)
    prev = pos[:, : -2 * w]
    mid = pos[:, w:-w]
    nxt = pos[:, 2 * w :]
    return (nxt - mid) * scale - (mid - prev) * scale


def _deblur_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, fps: torch.Tensor
) -> torch.Tensor:
    """L1 on position (metres) plus 0.5 L1 on the 2-frame velocity change, on frames in ``mask``.

    ``mask`` is true where the smoothed input and the clean target are both finite. The velocity
    term also needs the frames ``t-2`` and ``t+2`` in that mask, which is where the touch
    detector's ``v_out - v_in`` is defined. The reduction is the mean over valid coordinates.
    """
    err = (pred - target).abs()
    n = mask.sum()
    pos_loss = err[mask].sum() / (n * err.shape[-1]).to(dtype=err.dtype).clamp(min=1)
    w = DV_WINDOW
    stencil = mask[:, w:-w] & mask[:, : -2 * w] & mask[:, 2 * w :]
    dv = (_velocity_change(pred, fps) - _velocity_change(target, fps)).abs()
    n_dv = stencil.sum()
    dv_loss = dv[stencil].sum() / (n_dv * dv.shape[-1]).to(dtype=dv.dtype).clamp(min=1)
    # ``pred.sum() * 0`` keeps the graph connected when a batch has no valid frame.
    return pos_loss + DV_WEIGHT * dv_loss + pred.sum() * 0.0


def _batch_loss(
    model: nn.Module, clean: np.ndarray, blurred: np.ndarray, fps: np.ndarray
) -> torch.Tensor:
    """Position is the smoothed input plus the network's ``(dx, dy)``, on finite input frames."""
    param = next(model.parameters())
    features = torch.as_tensor(
        np.ascontiguousarray(ball_features(blurred)), dtype=param.dtype, device=param.device
    )
    delta = model(features)
    base = torch.as_tensor(
        np.ascontiguousarray(np.nan_to_num(blurred, nan=0.0)),
        dtype=param.dtype,
        device=param.device,
    )
    target = torch.as_tensor(
        np.ascontiguousarray(np.nan_to_num(clean, nan=0.0)),
        dtype=param.dtype,
        device=param.device,
    )
    valid = np.isfinite(blurred).all(axis=-1) & np.isfinite(clean).all(axis=-1)
    mask = torch.as_tensor(np.ascontiguousarray(valid), device=param.device)
    fps_t = torch.as_tensor(np.ascontiguousarray(fps), dtype=param.dtype, device=param.device)
    return _deblur_loss(base + delta, target, mask, fps_t)


def train_deblur(
    matches: Sequence[str | MatchBundle],
    *,
    seed: int = 0,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH,
    window: int = DEFAULT_WINDOW,
    out: str | Path | None = None,
    probe_batches: int = PROBE_BATCHES,
    log: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """AdamW at ``2e-3`` for ``steps`` batches. Same seed, same loss.

    ``torch.manual_seed`` fixes the initial weights. A numpy Generator fixes the windows, the
    sigmas and the gaps. The returned probe losses are the mean over one fixed set of batches
    drawn before the optimiser steps. ``out``, when set, is the JSON weight file.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 1 or window < MIN_RUN:
        raise ValueError(f"batch must be positive and window at least {MIN_RUN}")
    if probe_batches < 1:
        raise ValueError("probe_batches must be positive")
    torch.manual_seed(seed)
    with _deterministic_cpu():
        model = DeblurNet().to(device="cpu", dtype=torch.float32)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, foreach=False)
        bundles = as_bundles(matches)
        spans, counts = _eligible(bundles, window)
        probe_rng = np.random.default_rng(seed)
        train_rng = np.random.default_rng(seed)
        probe = [
            _sample_batch(bundles, spans, counts, window, batch_size, probe_rng)
            for _batch in range(probe_batches)
        ]

        def probe_mean() -> float:
            total = 0.0
            with torch.no_grad():
                for clean, blurred, fps in probe:
                    total += float(_batch_loss(model, clean, blurred, fps).detach())
            return total / len(probe)

        before = probe_mean()
        losses: list[float] = []
        n_params = sum(int(p.numel()) for p in model.parameters())
        for step in range(steps):
            clean, blurred, fps = _sample_batch(
                bundles, spans, counts, window, batch_size, train_rng
            )
            opt.zero_grad(set_to_none=True)
            loss = _batch_loss(model, clean, blurred, fps)
            losses.append(float(loss.detach()))
            loss.backward()
            opt.step()
            if log is not None and (step % 100 == 0 or step == steps - 1):
                log(f"step={step} loss={losses[-1]:.6f}")
        after = probe_mean()
        if out is not None:
            export_weights(model, out)
    return {
        "parameters": float(n_params),
        "steps": float(steps),
        "seed": float(seed),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "probe_before": before,
        "probe_after": after,
    }


def build_parser() -> argparse.ArgumentParser:
    """``train`` writes JSON weights. ``eval`` prints one aggregate line per sigma."""
    parser = argparse.ArgumentParser(
        prog="python -m aaaag.ml.deblur",
        description="Train or evaluate a residual CNN that undoes Gaussian ball smoothing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train on match ball tracks and write JSON weights")
    train.add_argument("--matches", nargs="+", default=DEFAULT_MATCHES)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    train.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    train.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    evaluate = sub.add_parser("eval", help="RMSE and touch rates at sigma 0, 4 and 8")
    evaluate.add_argument("--weights", required=True, type=Path)
    evaluate.add_argument("--matches", nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Train to a JSON file, or print aggregate RMSE and touch shares. No frame rows."""
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = train_deblur(
            args.matches,
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch,
            window=args.window,
            out=args.out,
            log=print,
        )
        print(
            f"parameters={int(result['parameters'])} steps={int(result['steps'])} "
            f"loss_first={result['loss_first']:.6f} loss_last={result['loss_last']:.6f} "
            f"probe_before={result['probe_before']:.6f} probe_after={result['probe_after']:.6f}"
        )
        print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
        return
    text = format_eval(evaluate_deblur(args.weights, args.matches))
    if text:
        print(text)


if __name__ == "__main__":
    main()
