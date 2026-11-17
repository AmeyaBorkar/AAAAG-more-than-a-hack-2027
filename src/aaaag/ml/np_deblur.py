"""Float64 numpy inference for the ball de-blurrer. This module does not import torch.

A JSON weight file stores each parameter as little-endian float32 bytes, hex-encoded, in the
same flat layout as ``np_infer``. Values are promoted to float64 before any arithmetic.

``deblur_ball`` walks each period, then each contiguous run of finite ball frames. A run
shorter than 8 frames is copied. A NaN stays a NaN: the network is never asked to fill a gap,
and a gap splits the context so a touch on one side of it cannot leak across. Frames outside
the periods are copied as well.

The network sees ``[x / 52.5, y / 34.0, valid]`` and returns a correction in metres that is
added to the input position. Invalid input frames are the zero vector with valid 0; at
inference those frames are simply not in the run.

Where it fails: a run of 8 to a few dozen frames is mostly padding (the stack looks about
±62 frames ahead), and a blur wider than that receptive field is only partly undone.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf

from ..contracts import MatchBundle, Tracking
from ..postprocess.touch_snap import detect_touches
from ..registry import SOURCES
from . import weights as weight_files
from .augment import smooth_ball_track

FORMAT_VERSION = 1
ARCH = "deblur_net"
CHANNELS = 32
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 16)
X_SCALE = 52.5
Y_SCALE = 34.0
MIN_RUN = 8
EVAL_SIGMAS = (0, 4, 8)
_SQRT2 = math.sqrt(2.0)
_BLOCKED_SOURCES = frozenset({"bar-ath", "bar_ath"})


def default_config() -> dict[str, Any]:
    """The architecture this repository trains. Written into every weight file."""
    return {
        "channels": CHANNELS,
        "kernel": KERNEL,
        "dilations": list(DILATIONS),
        "x_scale": X_SCALE,
        "y_scale": Y_SCALE,
        "min_run": MIN_RUN,
    }


def parameter_layout(config: Mapping[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    """``(name, shape)`` of a deblur state dict, in torch's parameter order.

    Five dilated convolutions, then a pointwise head. The first convolution reads 3 channels
    and every later one reads ``channels``.
    """
    channels = int(config["channels"])
    kernel = int(config["kernel"])
    dilations = [int(d) for d in config["dilations"]]
    if channels < 1 or kernel < 1 or not dilations:
        raise ValueError("deblur config needs channels, an odd kernel and at least one dilation")
    if kernel % 2 == 0:
        raise ValueError("kernel must be odd so same-padding is symmetric")
    names: list[tuple[str, tuple[int, ...]]] = []
    cin = 3
    for index, _dilation in enumerate(dilations):
        names.append((f"convs.{index}.weight", (channels, cin, kernel)))
        names.append((f"convs.{index}.bias", (channels,)))
        cin = channels
    names.append(("head.weight", (2, channels, 1)))
    names.append(("head.bias", (2,)))
    return names


def count_parameters(config: Mapping[str, Any] | None = None) -> int:
    """Number of scalars in ``parameter_layout``."""
    cfg = default_config() if config is None else config
    return sum(math.prod(shape) for _name, shape in parameter_layout(cfg))


def ball_features(
    ball_xy: np.ndarray, x_scale: float = X_SCALE, y_scale: float = Y_SCALE
) -> np.ndarray:
    """``(..., 2)`` metres -> ``(..., 3)`` ``[x / x_scale, y / y_scale, valid]``.

    A frame with a non-finite coordinate is ``(0, 0, 0)``. Finite frames keep their position.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    if ball.shape[-1] != 2:
        raise ValueError(f"ball_xy must end with 2 coordinates, got {ball.shape}")
    valid = np.isfinite(ball).all(axis=-1)
    feat = np.zeros(ball.shape[:-1] + (3,), dtype=np.float64)
    feat[..., 0][valid] = ball[..., 0][valid] / float(x_scale)
    feat[..., 1][valid] = ball[..., 1][valid] / float(y_scale)
    feat[..., 2] = valid.astype(np.float64)
    return feat


def smooth_for_deblur(ball_xy: np.ndarray, periods: Sequence[Any], sigma: float) -> np.ndarray:
    """``(T, 2)`` float32. Sigma 0 is a copy.

    ``gaussian_filter1d`` divides by ``sigma**2`` when it builds the kernel, so a literal call
    at sigma 0 raises. No smoothing is the identity, which is what sigma 0 means.
    """
    if float(sigma) == 0.0:
        return np.array(ball_xy, dtype=np.float32, copy=True)
    return smooth_ball_track(ball_xy, tuple(periods), float(sigma))


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` runs where the 1-D mask is true."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _gelu(x: np.ndarray) -> np.ndarray:
    """Torch's exact-erf GELU (``approximate='none'``)."""
    return 0.5 * x * (1.0 + erf(x / _SQRT2))


class _Conv:
    """One ``nn.Conv1d``: taps are ``(C_in, C_out)`` float64, bias is ``(C_out,)``."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray, dilation: int) -> None:
        array = np.ascontiguousarray(weight, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(f"conv weight must be (C_out, C_in, K), got {array.shape}")
        self.taps = [np.ascontiguousarray(array[:, :, tap].T) for tap in range(array.shape[-1])]
        self.bias = np.ascontiguousarray(bias, dtype=np.float64).reshape(-1)
        self.dilation = int(dilation)
        if self.bias.shape != (array.shape[0],):
            raise ValueError(f"bias {self.bias.shape} does not match {array.shape[0]} channels")


def _conv_same(x: np.ndarray, conv: _Conv) -> np.ndarray:
    """Channels-last conv1d with the same zero padding as ``nn.Conv1d`` (odd kernel).

    ``x`` is ``(B, T, C_in)``. Output is ``(B, T, C_out)``.
    """
    if x.ndim != 3:
        raise ValueError(f"conv input must be (B, T, C), got {x.shape}")
    length = x.shape[1]
    cin = x.shape[2]
    if conv.taps[0].shape[0] != cin:
        raise ValueError(f"conv channels {cin} != weight {conv.taps[0].shape[0]}")
    k = len(conv.taps)
    pad = conv.dilation * (k - 1) // 2
    out = np.empty((x.shape[0], length, conv.bias.shape[0]), dtype=np.float64)
    out[...] = conv.bias
    values = np.ascontiguousarray(x, dtype=np.float64)
    for tap, weight_t in enumerate(conv.taps):
        shift = tap * conv.dilation - pad
        if shift == 0:
            out += values @ weight_t
        elif 0 < shift < length:
            out[:, : length - shift] += values[:, shift:] @ weight_t
        elif -length < shift < 0:
            skip = -shift
            out[:, skip:] += values[:, : length - skip] @ weight_t
    return out


class NumpyDeblur:
    """The de-blurrer in float64. ``forward`` maps features ``(B, T, 3)`` to deltas ``(B, T, 2)``."""

    def __init__(self, config: Mapping[str, Any], weights: Mapping[str, np.ndarray]) -> None:
        self.config = _checked_config(config)
        layout = parameter_layout(self.config)
        expected = {name: shape for name, shape in layout}
        missing = [name for name in expected if name not in weights]
        extra = [name for name in weights if name not in expected]
        if missing or extra:
            raise ValueError(f"deblur weights missing {missing} extra {extra}")
        stored: dict[str, np.ndarray] = {}
        for name, shape in layout:
            array = np.array(weights[name], dtype=np.float64, copy=True)
            if tuple(array.shape) != shape:
                raise ValueError(f"{name}: shape {tuple(array.shape)} != {shape}")
            stored[name] = array
        self.weights = stored
        self.x_scale = float(self.config["x_scale"])
        self.y_scale = float(self.config["y_scale"])
        self.min_run = int(self.config["min_run"])
        self.channels = int(self.config["channels"])
        self.kernel = int(self.config["kernel"])
        self.dilations = tuple(int(d) for d in self.config["dilations"])
        self.convs = [
            _Conv(stored[f"convs.{i}.weight"], stored[f"convs.{i}.bias"], dilation)
            for i, dilation in enumerate(self.dilations)
        ]
        self.head = _Conv(stored["head.weight"], stored["head.bias"], 1)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """``(B, T, 3)`` -> ``(B, T, 2)`` metres. The first layer has no skip; later layers do."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"features must be (B, T, 3), got {x.shape}")
        h = _gelu(_conv_same(x, self.convs[0]))
        for layer in self.convs[1:]:
            h = h + _gelu(_conv_same(h, layer))
        return _conv_same(h, self.head)


def _checked_config(config: Mapping[str, Any]) -> dict[str, Any]:
    need = ("channels", "kernel", "dilations", "x_scale", "y_scale", "min_run")
    missing = [key for key in need if key not in config]
    if missing:
        raise ValueError(f"deblur config missing {missing}")
    parsed = {
        "channels": int(config["channels"]),
        "kernel": int(config["kernel"]),
        "dilations": [int(d) for d in config["dilations"]],
        "x_scale": float(config["x_scale"]),
        "y_scale": float(config["y_scale"]),
        "min_run": int(config["min_run"]),
    }
    if parsed["min_run"] != MIN_RUN:
        raise ValueError(f"min_run must be {MIN_RUN}, got {parsed['min_run']}")
    if parsed["x_scale"] != X_SCALE or parsed["y_scale"] != Y_SCALE:
        raise ValueError("weight file pitch scales do not match this deblurrer")
    parameter_layout(parsed)
    return parsed


def _f32_hex(array: np.ndarray) -> str:
    return np.ascontiguousarray(array, dtype="<f4").tobytes().hex()


def dump_deblur_weights(
    path: str | Path, arrays: Mapping[str, np.ndarray], config: Mapping[str, Any] | None = None
) -> Path:
    """Write the JSON weight file. Each parameter is little-endian float32, hex-encoded."""
    cfg = _checked_config(default_config() if config is None else config)
    layout = parameter_layout(cfg)
    expect = {name for name, _shape in layout}
    if set(arrays) != expect:
        raise ValueError("weight names do not match the deblur layout for this config")
    parameters = []
    for name, shape in layout:
        array = np.ascontiguousarray(arrays[name])
        if tuple(array.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(array.shape)} != {shape}")
        parameters.append({"name": name, "shape": [int(s) for s in shape], "data": _f32_hex(array)})
    doc = {
        "format_version": FORMAT_VERSION,
        "arch": ARCH,
        "config": cfg,
        "parameters": parameters,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return path


def load_deblur_weights(path: str | Path) -> NumpyDeblur:
    """The model stored by ``dump_deblur_weights`` / ``python -m aaaag.ml.deblur train``."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not a JSON deblur weight file") from error
    if doc.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: weight format {doc.get('format_version')!r}, expected {FORMAT_VERSION}"
        )
    if doc.get("arch") != ARCH:
        raise ValueError(f"{path}: expected arch {ARCH!r}, got {doc.get('arch')!r}")
    weights: dict[str, np.ndarray] = {}
    for entry in doc["parameters"]:
        shape = tuple(int(s) for s in entry["shape"])
        raw = np.frombuffer(bytes.fromhex(entry["data"]), dtype="<f4")
        if raw.size != math.prod(shape):
            raise ValueError(f"{entry['name']}: {raw.size} values for shape {shape}")
        weights[entry["name"]] = np.array(raw, dtype=np.float64).reshape(shape)
    return NumpyDeblur(doc["config"], weights)


def correct_ball(
    ball_xy: np.ndarray,
    periods: Sequence[Any],
    delta_fn: Callable[[np.ndarray], np.ndarray],
    *,
    min_run: int = MIN_RUN,
) -> np.ndarray:
    """``(T, 2)`` float64. ``delta_fn(run) -> (L, 2)`` metres is added on runs of length ``min_run``.

    Shorter runs, non-finite frames and frames outside ``periods`` are copies of ``ball_xy``.
    ``delta_fn`` is not shown a gap and is not asked to invent coordinates inside one.
    """
    ball = np.asarray(ball_xy)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError(f"ball_xy must be (T, 2), got {ball.shape}")
    out = np.array(ball, dtype=np.float64, copy=True)
    n = out.shape[0]
    for period in periods:
        start_p, stop_p = int(period.start), int(period.stop)
        if start_p < 0 or stop_p > n or start_p >= stop_p:
            raise ValueError(f"period {(start_p, stop_p)} outside the track of length {n}")
        seg = out[start_p:stop_p]
        seen = np.isfinite(seg).all(axis=1)
        for start, stop in _mask_runs(seen):
            if stop - start < min_run:
                continue
            run = np.array(seg[start:stop], dtype=np.float64, copy=True)
            delta = np.asarray(delta_fn(run), dtype=np.float64)
            if delta.shape != run.shape:
                raise ValueError(f"delta shape {delta.shape} != run {run.shape}")
            seg[start:stop] = run + delta
    return out


def apply_deblur(ball_xy: np.ndarray, periods: Sequence[Any], model: NumpyDeblur) -> np.ndarray:
    """``(T, 2)`` float64 track. Same run rules as ``deblur_ball``."""

    def delta_fn(run: np.ndarray) -> np.ndarray:
        features = ball_features(run, model.x_scale, model.y_scale)
        return model.forward(features[None])[0]

    return correct_ball(ball_xy, periods, delta_fn, min_run=model.min_run)


def deblur_ball(
    ball_xy: np.ndarray, periods: Sequence[Any], weights_path: str | Path
) -> np.ndarray:
    """``(T, 2)`` float32. Float64 compute, per period, per contiguous seen run.

    A run shorter than 8 frames is copied. NaN stays NaN. The input array is not modified.
    """
    model = load_deblur_weights(weights_path)
    return np.ascontiguousarray(apply_deblur(ball_xy, periods, model), dtype=np.float32)


def resolve_deblur_weights(path: str | Path, sha256: str | None = None) -> Path:
    """Absolute path of a deblur file that exists and, when ``sha256`` is set, matches the pin.

    The pin is a hex prefix of at least 8 characters, compared case-insensitively, the same
    rule as the team-weight pin.
    """
    resolved = weight_files.resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"ball_deblur weights not found: {resolved}")
    if sha256 is None:
        return resolved
    if not isinstance(sha256, str):
        raise TypeError(f"ball_deblur_sha256 must be a string, got {type(sha256).__name__}")
    if len(sha256) < weight_files.MIN_PIN:
        raise ValueError(
            f"ball_deblur sha256 pin {sha256!r} is shorter than {weight_files.MIN_PIN} characters"
        )
    actual = weight_files.sha256_file(resolved)
    if not actual.startswith(sha256.lower()):
        raise ValueError(f"ball_deblur: {resolved.name} has sha256 {actual}, pinned {sha256}")
    return resolved


def deblurred_tracking(
    tracking: Tracking, weights_path: str | Path, sha256: str | None = None
) -> Tracking:
    """A new ``Tracking`` whose ball is ``deblur_ball``. ``tracking`` itself is not modified."""
    path = resolve_deblur_weights(weights_path, sha256)
    ball = deblur_ball(tracking.ball_xy, tracking.meta.periods, path)
    return replace(tracking, ball_xy=ball, _fingerprint=None)


def deblurred_bundle(
    bundle: MatchBundle, weights_path: str | Path, sha256: str | None = None
) -> MatchBundle:
    """A new bundle whose tracking ball is deblurred. ``bundle`` itself is not modified."""
    return replace(bundle, tracking=deblurred_tracking(bundle.tracking, weights_path, sha256))


def load_named_match(spec: str) -> MatchBundle:
    """``source/id`` via ``SOURCES``.

    ``metrica/<game>`` uses ``game=``. ``idsse/<id>`` uses ``match_id=`` and refuses anything
    that is not a dev match, so the sealed matches are not opened. ``synthetic/<seed>`` uses
    ``match_id=``. Any other source is called with ``match_id=``.
    """
    text = spec.strip()
    source, sep, ident = text.partition("/")
    if not sep or not source or not ident:
        raise ValueError(f"match must be 'source/id', got {spec!r}")
    if source.casefold() in _BLOCKED_SOURCES:
        raise ValueError(f"source {source!r} is not available to this tool")
    if source == "metrica":
        return SOURCES.get("metrica").fn(game=ident)
    if source == "idsse":
        from ..sources.idsse import DEV_MATCHES

        if ident not in DEV_MATCHES:
            raise ValueError(f"idsse match {ident!r} is not a dev match")
        return SOURCES.get("idsse").fn(match_id=ident)
    if source == "synthetic":
        return SOURCES.get("synthetic").fn(match_id=ident)
    try:
        load = SOURCES.get(source).fn
    except KeyError as exc:
        raise ValueError(f"unknown match source {source!r}") from exc
    return load(match_id=ident)


def as_bundles(matches: Sequence[str | MatchBundle]) -> list[MatchBundle]:
    """Strings are ``load_named_match`` entries. Bundles are kept in order."""
    if not matches:
        raise ValueError("need at least one match")
    loaded: list[MatchBundle] = []
    for item in matches:
        if isinstance(item, MatchBundle):
            loaded.append(item)
        else:
            loaded.append(load_named_match(str(item)))
    return loaded


def _touch_count(tracking: Tracking, ball_xy: np.ndarray) -> int:
    """How many frames ``detect_touches`` marks. The detector's default window is 2 frames."""
    swapped = replace(
        tracking,
        ball_xy=np.ascontiguousarray(ball_xy, dtype=np.float32),
        _fingerprint=None,
    )
    frames, _teams = detect_touches(swapped)
    return int(frames.shape[0])


def _squared_position_error(
    estimate: np.ndarray, clean: np.ndarray, played: np.ndarray
) -> tuple[float, int]:
    """Sum of squared Euclidean errors, and the frame count, on played frames with a finite ball."""
    mask = played & np.isfinite(clean).all(axis=-1) & np.isfinite(estimate).all(axis=-1)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0, 0
    diff = estimate[mask] - clean[mask]
    return float(np.sum(diff * diff)), count


def evaluate_deblur(
    weights: str | Path, matches: Sequence[str | MatchBundle]
) -> list[dict[str, float]]:
    """One aggregate row per sigma in ``{0, 4, 8}``.

    Position RMSE is the root mean square of the Euclidean error in metres, over played frames
    whose clean ball is finite, pooled across matches. Touch shares are the fraction of played
    frames on which the 2-frame touch detector fires, for the clean, smoothed and deblurred
    tracks. Nothing frame-level is returned.
    """
    bundles = as_bundles(matches)
    path = resolve_deblur_weights(weights, None)
    rows: list[dict[str, float]] = []
    for sigma in EVAL_SIGMAS:
        sq_smooth = 0.0
        sq_deblur = 0.0
        n_frames = 0
        touch_clean = 0
        touch_smooth = 0
        touch_deblur = 0
        played_frames = 0
        for bundle in bundles:
            tracking = bundle.tracking
            periods = tracking.meta.periods
            clean = np.asarray(tracking.ball_xy, dtype=np.float64)
            played = tracking.meta.played_mask()
            smooth = np.asarray(smooth_for_deblur(clean, periods, sigma), dtype=np.float64)
            deblurred = np.asarray(deblur_ball(smooth, periods, path), dtype=np.float64)
            err_s, count_s = _squared_position_error(smooth, clean, played)
            err_d, count_d = _squared_position_error(deblurred, clean, played)
            if count_d != count_s:
                raise RuntimeError("deblur changed which frames are finite")
            sq_smooth += err_s
            sq_deblur += err_d
            n_frames += count_s
            frames, _teams = detect_touches(tracking)
            touch_clean += int(frames.shape[0])
            touch_smooth += _touch_count(tracking, smooth)
            touch_deblur += _touch_count(tracking, deblurred)
            played_frames += int(np.count_nonzero(played))
        rows.append(
            {
                "sigma": float(sigma),
                "matches": float(len(bundles)),
                "frames": float(n_frames),
                "rmse_smooth_m": math.sqrt(sq_smooth / n_frames) if n_frames else float("nan"),
                "rmse_deblur_m": math.sqrt(sq_deblur / n_frames) if n_frames else float("nan"),
                "touch_clean": (touch_clean / played_frames) if played_frames else float("nan"),
                "touch_smooth": (touch_smooth / played_frames) if played_frames else float("nan"),
                "touch_deblur": (touch_deblur / played_frames) if played_frames else float("nan"),
            }
        )
    return rows


def format_eval(rows: Sequence[Mapping[str, float]]) -> str:
    """The aggregate lines ``python -m aaaag.ml.deblur eval`` prints. No frame values."""
    lines = []
    for row in rows:
        lines.append(
            f"sigma={int(row['sigma'])} matches={int(row['matches'])} frames={int(row['frames'])} "
            f"rmse_smooth_m={row['rmse_smooth_m']:.6f} rmse_deblur_m={row['rmse_deblur_m']:.6f} "
            f"touch_clean={row['touch_clean']:.6f} touch_smooth={row['touch_smooth']:.6f} "
            f"touch_deblur={row['touch_deblur']:.6f}"
        )
    return "\n".join(lines)
