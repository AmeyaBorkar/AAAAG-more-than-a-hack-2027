"""Float64 numpy inference for `team_net`, with no torch import.

The arithmetic follows `model.py`: embed MLP, pre-LN relation-biased attention with invalid
tokens masked as keys, per-team attention pooling, dilated residual convolutions, then the
class and ball heads. Weights are stored as float32 (see `export.py`) and computed in float64.
With `possessor_head` in the config, player and nobody logits are softmaxed and the home, away
and nobody sums are concatenated onto the frame vector. A config without that key leaves the
arithmetic as it was. The possessor logits themselves are not returned.
With `switch_head`, the temporal output also feeds a linear layer and the ensemble averages that
logit over models and views (a team swap does not reorder it). Without the key the class-logit
path is unchanged and no switch channel is returned.
With `touch_head`, the same temporal output feeds one more linear layer, created after the switch
layer. The ensemble averages that logit over models and views. A y flip does not change how it is
reduced, and a team swap does not reorder it, because a touch has no team. Without the key the
class-logit path is unchanged and no touch channel is returned.
With `slot_temporal` enabled in the config, the slot branch of `model.py` runs first on the whole
window: the same slot links (SLOT_ID and SLOT_RUN equal in the same team block, both tokens valid),
the same residual dilated taps, and the output added to the embedded player tokens. The tokens
then carry `n_features + 2` columns and only the first `n_features` are embedded. A config without
the key, or with it disabled, leaves the arithmetic and the parameter list as they were.
GELU is the exact erf form and LayerNorm uses eps 1e-5, matching torch's defaults.

Matmuls run over the whole chunk in one call. A row's product on this BLAS changes by about
1e-14 when the chunk length changes, so a chunked match agrees with one pass inside the
1e-10 band (infer.py's chunking, with the same context margin).

`predict_logits_np(..., workers=n)` with n above 1 runs those chunks in a spawn process pool.
Each worker loads the JSON weights once, pins BLAS to one thread and returns its kept frames;
the parent writes them back in chunk order. Those bits match the serial loop when that loop is
also on one BLAS thread: the pool forces one thread per worker, and `workers=1` leaves the
caller's thread count alone. `workers=1` is the serial loop in this process.

Signals: the token tensor only. Fails closed where torch does: frames outside the periods and
frames with no observed player are NaN. Only architecture `team_net` is implemented. Dropout is
off, as in `eval`. `device` and torch's thread pin do not apply here.
"""

from __future__ import annotations

import base64
import json
import math
import multiprocessing as mp
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf, expit
from threadpoolctl import threadpool_limits

from ..contracts import Tracking
from .geometry import SWAP_CLASSES, symmetric
from .spec import FEATURE_VERSION, Col, TokenSpec
from .windows import build_match_tokens, plan_chunks

FORMAT_VERSION = 1  # parameter data hex-encoded (the default)
# Weight file versions and how each one encodes the float32 bytes. Version 2 stores the same bytes
# base64-encoded, about two thirds the size, so a d_model 128 network stays under the repository's
# 5 MB file limit. Both load to identical arrays.
ENCODINGS = {1: "hex", 2: "base64"}
MASKED = -1.0e9
LN_EPS = 1e-5
N_RELATIONS = 5
_SQRT2 = math.sqrt(2.0)

# (flip_y, swap). Same table as infer.VIEWS.
VIEWS = {
    1: ((False, False),),
    2: ((False, False), (True, False)),
    4: ((False, False), (True, False), (False, True), (True, True)),
}

# Row products of a float64 GEMM on this BLAS match across batch sizes down to ~1e-14
# (except a pure matrix-vector). Chunking therefore stays inside the 1e-10 band without
# padding every product up to a fixed row count, which dominated the runtime.


def parameter_layout(config: dict[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    """`(name, shape)` of a `team_net` state dict, in torch's parameter order."""
    d = int(config["d_model"])
    heads = int(config["n_heads"])
    blocks = int(config["n_blocks"])
    mult = int(config["ffn_mult"])
    features = int(config["n_features"])
    kernel = int(config["kernel"])
    dilations = tuple(int(x) for x in config["dilations"])
    if d % heads:
        raise ValueError(f"d_model {d} is not a multiple of n_heads {heads}")
    if kernel % 2 == 0:
        raise ValueError("kernel must be odd so that 'same' padding is symmetric")
    possessor_head = bool(config.get("possessor_head", False))
    frame_in = 3 * d + (3 if possessor_head else 0)
    names: list[tuple[str, tuple[int, ...]]] = [("pool_query", (2, d))]
    names += [
        ("embed.0.weight", (d, features)),
        ("embed.0.bias", (d,)),
        ("embed.2.weight", (d, d)),
        ("embed.2.bias", (d,)),
    ]
    for i in range(blocks):
        p = f"blocks.{i}"
        names += [
            (f"{p}.relation_bias", (heads, N_RELATIONS)),
            (f"{p}.norm1.weight", (d,)),
            (f"{p}.norm1.bias", (d,)),
            (f"{p}.qkv.weight", (3 * d, d)),
            (f"{p}.qkv.bias", (3 * d,)),
            (f"{p}.proj.weight", (d, d)),
            (f"{p}.proj.bias", (d,)),
            (f"{p}.norm2.weight", (d,)),
            (f"{p}.norm2.bias", (d,)),
            (f"{p}.ffn.0.weight", (mult * d, d)),
            (f"{p}.ffn.0.bias", (mult * d,)),
            (f"{p}.ffn.3.weight", (d, mult * d)),
            (f"{p}.ffn.3.bias", (d,)),
        ]
    names += [
        ("norm.weight", (d,)),
        ("norm.bias", (d,)),
        ("pool_key.weight", (d, d)),
        ("frame.0.weight", (frame_in,)),
        ("frame.0.bias", (frame_in,)),
        ("frame.1.weight", (d, frame_in)),
        ("frame.1.bias", (d,)),
    ]
    for i in range(len(dilations)):
        p = f"temporal.{i}"
        names += [
            (f"{p}.norm.weight", (d,)),
            (f"{p}.norm.bias", (d,)),
            (f"{p}.conv.weight", (d, d, kernel)),
            (f"{p}.conv.bias", (d,)),
            (f"{p}.pointwise.weight", (d, d, 1)),
            (f"{p}.pointwise.bias", (d,)),
        ]
    names += [
        ("out_norm.weight", (d,)),
        ("out_norm.bias", (d,)),
        ("head.weight", (3, d)),
        ("head.bias", (3,)),
        ("ball_head.weight", (2, d)),
        ("ball_head.bias", (2,)),
    ]
    if possessor_head:
        names += [
            ("possessor.weight", (1, d)),
            ("possessor.bias", (1,)),
            ("possessor_none.weight", (1, d)),
            ("possessor_none.bias", (1,)),
        ]
    if bool(config.get("switch_head", False)):
        names += [
            ("switch.weight", (1, d)),
            ("switch.bias", (1,)),
        ]
    if bool(config.get("touch_head", False)):
        names += [
            ("touch.weight", (1, d)),
            ("touch.bias", (1,)),
        ]
    slot = _slot_config(config)
    if slot is not None:
        ds, taps = int(slot["d"]), int(slot["kernel"])
        names += [
            ("slot_temporal.in_proj.weight", (ds, len(slot["inputs"]))),
            ("slot_temporal.in_proj.bias", (ds,)),
        ]
        for i in range(len(slot["dilations"])):
            p = f"slot_temporal.layers.{i}"
            names += [
                (f"{p}.norm.weight", (ds,)),
                (f"{p}.norm.bias", (ds,)),
                (f"{p}.conv.weight", (ds, taps * ds)),
                (f"{p}.conv.bias", (ds,)),
                (f"{p}.pointwise.weight", (ds, ds)),
                (f"{p}.pointwise.bias", (ds,)),
            ]
        names += [
            ("slot_temporal.out_norm.weight", (ds,)),
            ("slot_temporal.out_norm.bias", (ds,)),
            ("slot_temporal.out_proj.weight", (d, ds)),
            ("slot_temporal.out_proj.bias", (d,)),
        ]
    return names


def _slot_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """The `slot_temporal` block when it is enabled, else None."""
    slot = config.get("slot_temporal")
    if isinstance(slot, dict) and bool(slot.get("enabled", False)):
        return slot
    return None


def receptive_half(config: dict[str, Any]) -> int:
    kernel = int(config["kernel"])
    reach = sum(int(d) * (kernel - 1) // 2 for d in config["dilations"])
    slot = _slot_config(config)
    if slot is not None:
        reach += sum(int(d) * (int(slot["kernel"]) - 1) // 2 for d in slot["dilations"])
    return reach


def _f64(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_weight_T(array: np.ndarray) -> np.ndarray:
    """Torch `(out, in)` linear weight as a contiguous `(in, out)` matrix."""
    return np.ascontiguousarray(np.ascontiguousarray(array, dtype=np.float64).T)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a @ b` with `a` shaped `(..., K)` and `b` shaped `(K, N)`.

    One GEMM over the leading rows. Splitting a match into chunks changes those dimensions,
    and on this BLAS that moves a product by about 1e-14, which the chunk check allows.
    """
    k = a.shape[-1]
    flat = a.reshape(-1, k) if a.flags.c_contiguous else np.ascontiguousarray(a).reshape(-1, k)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b)
    return (flat @ b).reshape(*a.shape[:-1], b.shape[-1])


def _linear(x: np.ndarray, weight_t: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    y = matmul(x, weight_t)
    if bias is not None:
        y = y + bias
    return y


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * (1.0 / np.sqrt(var + LN_EPS)) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / _SQRT2))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def relation_matrix(tokens_per_team: int) -> np.ndarray:
    """`(n, n)` relation id of token i attending to token j. Same cases as `model.relation_matrix`."""
    k = tokens_per_team
    n = 2 * k + 1
    team = np.array([0] * k + [1] * k + [2])
    rel = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if team[i] == 2:
                rel[i, j] = 4 if team[j] == 2 else 3
            elif team[j] == 2:
                rel[i, j] = 2
            else:
                rel[i, j] = 0 if team[i] == team[j] else 1
    return rel


@dataclass
class _Attn:
    n1_w: np.ndarray
    n1_b: np.ndarray
    qkv_wT: np.ndarray
    qkv_b: np.ndarray
    proj_wT: np.ndarray
    proj_b: np.ndarray
    rel: np.ndarray
    n2_w: np.ndarray
    n2_b: np.ndarray
    ffn0_wT: np.ndarray
    ffn0_b: np.ndarray
    ffn3_wT: np.ndarray
    ffn3_b: np.ndarray


@dataclass
class _Temp:
    ln_w: np.ndarray
    ln_b: np.ndarray
    taps: list[np.ndarray]
    conv_b: np.ndarray
    dilation: int
    pw_wT: np.ndarray
    pw_b: np.ndarray


@dataclass
class _SlotLayer:
    ln_w: np.ndarray
    ln_b: np.ndarray
    conv_wT: np.ndarray  # (kernel * d, d): the taps side by side, earliest frame first
    conv_b: np.ndarray
    pw_wT: np.ndarray
    pw_b: np.ndarray
    offsets: tuple[int, ...]


class NumpyTeamNet:
    """`team_net` evaluated in float64. `forward` takes tokens `(B, W, n_tokens, F)`."""

    def __init__(
        self,
        config: dict[str, Any],
        spec: TokenSpec | dict[str, Any],
        weights: dict[str, np.ndarray],
    ):
        self.config = config
        self.spec = spec if isinstance(spec, TokenSpec) else TokenSpec.from_dict(spec)
        self.d = int(config["d_model"])
        self.heads = int(config["n_heads"])
        self.head_dim = self.d // self.heads
        self.k = int(config["tokens_per_team"])
        self.n_tokens = 2 * self.k + 1
        self.n_features = int(config["n_features"])
        self.dilations = tuple(int(x) for x in config["dilations"])
        self.kernel = int(config["kernel"])
        self.receptive_half = receptive_half(config)
        self.possessor_head = bool(config.get("possessor_head", False))
        self.switch_head = bool(config.get("switch_head", False))
        self.touch_head = bool(config.get("touch_head", False))
        self.attn_scale = self.head_dim**-0.5
        self.pool_scale = float(self.d**-0.5)
        expected = dict(parameter_layout(config))
        missing = [name for name in expected if name not in weights]
        extra = [name for name in weights if name not in expected]
        if missing or extra:
            raise ValueError(f"team_net weights missing {missing} extra {extra}")
        for name, shape in expected.items():
            if tuple(weights[name].shape) != shape:
                raise ValueError(f"{name}: shape {tuple(weights[name].shape)} != {shape}")
        w = weights
        rel = relation_matrix(self.k)
        self.blocks: list[_Attn] = []
        for i in range(int(config["n_blocks"])):
            p = f"blocks.{i}"
            bias = _f64(w[f"{p}.relation_bias"])
            self.blocks.append(
                _Attn(
                    n1_w=_f64(w[f"{p}.norm1.weight"]),
                    n1_b=_f64(w[f"{p}.norm1.bias"]),
                    qkv_wT=_as_weight_T(w[f"{p}.qkv.weight"]),
                    qkv_b=_f64(w[f"{p}.qkv.bias"]),
                    proj_wT=_as_weight_T(w[f"{p}.proj.weight"]),
                    proj_b=_f64(w[f"{p}.proj.bias"]),
                    rel=bias[:, rel],
                    n2_w=_f64(w[f"{p}.norm2.weight"]),
                    n2_b=_f64(w[f"{p}.norm2.bias"]),
                    ffn0_wT=_as_weight_T(w[f"{p}.ffn.0.weight"]),
                    ffn0_b=_f64(w[f"{p}.ffn.0.bias"]),
                    ffn3_wT=_as_weight_T(w[f"{p}.ffn.3.weight"]),
                    ffn3_b=_f64(w[f"{p}.ffn.3.bias"]),
                )
            )
        self.embed0_wT = _as_weight_T(w["embed.0.weight"])
        self.embed0_b = _f64(w["embed.0.bias"])
        self.embed2_wT = _as_weight_T(w["embed.2.weight"])
        self.embed2_b = _f64(w["embed.2.bias"])
        self.norm_w = _f64(w["norm.weight"])
        self.norm_b = _f64(w["norm.bias"])
        self.pool_wT = _as_weight_T(w["pool_key.weight"])
        self.pool_query = _f64(w["pool_query"])
        self.frame_ln_w = _f64(w["frame.0.weight"])
        self.frame_ln_b = _f64(w["frame.0.bias"])
        self.frame_wT = _as_weight_T(w["frame.1.weight"])
        self.frame_b = _f64(w["frame.1.bias"])
        self.temporal: list[_Temp] = []
        for i, dilation in enumerate(self.dilations):
            p = f"temporal.{i}"
            conv_w = _f64(w[f"{p}.conv.weight"])
            pw = _f64(w[f"{p}.pointwise.weight"])
            self.temporal.append(
                _Temp(
                    ln_w=_f64(w[f"{p}.norm.weight"]),
                    ln_b=_f64(w[f"{p}.norm.bias"]),
                    taps=[np.ascontiguousarray(conv_w[:, :, tap].T) for tap in range(self.kernel)],
                    conv_b=_f64(w[f"{p}.conv.bias"]),
                    dilation=dilation,
                    pw_wT=np.ascontiguousarray(pw[:, :, 0].T),
                    pw_b=_f64(w[f"{p}.pointwise.bias"]),
                )
            )
        self.out_w = _f64(w["out_norm.weight"])
        self.out_b = _f64(w["out_norm.bias"])
        self.head_wT = _as_weight_T(w["head.weight"])
        self.head_b = _f64(w["head.bias"])
        self.ball_wT = _as_weight_T(w["ball_head.weight"])
        self.ball_b = _f64(w["ball_head.bias"])
        if self.possessor_head:
            self.poss_wT = _as_weight_T(w["possessor.weight"])
            self.poss_b = _f64(w["possessor.bias"])
            self.none_wT = _as_weight_T(w["possessor_none.weight"])
            self.none_b = _f64(w["possessor_none.bias"])
        if self.switch_head:
            self.switch_wT = _as_weight_T(w["switch.weight"])
            self.switch_b = _f64(w["switch.bias"])
        if self.touch_head:
            self.touch_wT = _as_weight_T(w["touch.weight"])
            self.touch_b = _f64(w["touch.bias"])
        slot = _slot_config(config)
        self.slot_temporal = slot is not None
        self.token_width = self.n_features + (2 if self.slot_temporal else 0)
        if slot is not None:
            if not self.spec.slot_index:
                raise ValueError("slot_temporal needs a token spec with slot_index")
            half = (int(slot["kernel"]) - 1) // 2
            self.slot_inputs = np.array([int(c) for c in slot["inputs"]], dtype=np.int64)
            self.slot_in_wT = _as_weight_T(w["slot_temporal.in_proj.weight"])
            self.slot_in_b = _f64(w["slot_temporal.in_proj.bias"])
            self.slot_layers: list[_SlotLayer] = []
            for i, dilation in enumerate(int(x) for x in slot["dilations"]):
                p = f"slot_temporal.layers.{i}"
                self.slot_layers.append(
                    _SlotLayer(
                        ln_w=_f64(w[f"{p}.norm.weight"]),
                        ln_b=_f64(w[f"{p}.norm.bias"]),
                        conv_wT=_as_weight_T(w[f"{p}.conv.weight"]),
                        conv_b=_f64(w[f"{p}.conv.bias"]),
                        pw_wT=_as_weight_T(w[f"{p}.pointwise.weight"]),
                        pw_b=_f64(w[f"{p}.pointwise.bias"]),
                        offsets=tuple(j * dilation for j in range(-half, half + 1)),
                    )
                )
            self.slot_offsets = tuple(sorted({o for la in self.slot_layers for o in la.offsets}))
            self.slot_out_ln_w = _f64(w["slot_temporal.out_norm.weight"])
            self.slot_out_ln_b = _f64(w["slot_temporal.out_norm.bias"])
            self.slot_out_wT = _as_weight_T(w["slot_temporal.out_proj.weight"])
            self.slot_out_b = _f64(w["slot_temporal.out_proj.bias"])

    def forward(self, tokens: np.ndarray) -> dict[str, np.ndarray]:
        """tokens `(B, W, n_tokens, F)` -> `{"logits": (B, W, 3), "ball": (B, W, 2)}` float64.

        With the switch head the dict also has `"switch": (B, W)`. With the touch head it also
        has `"touch": (B, W)`. Without a flag that key is absent and the other arrays are the
        ones a model without the flag returns. With `slot_temporal`, F is `n_features + 2`.
        """
        x = np.asarray(tokens, dtype=np.float64)
        if x.ndim != 4:
            raise ValueError(f"tokens must be (B, W, n, F), got {x.shape}")
        b, w, n, f = x.shape
        if n != self.n_tokens or f != self.token_width:
            raise ValueError(f"tokens {(n, f)} != model {(self.n_tokens, self.token_width)}")
        if self.slot_temporal:
            slot = self._slot_branch(x).reshape(b * w, 2 * self.k, self.d)
            x = np.ascontiguousarray(x[..., : self.n_features])
            emb = self._encode_all(x.reshape(b * w, n, self.n_features), slot)
        else:
            emb = self.encode_frames(x.reshape(b * w, n, f))
        emb = emb.reshape(b, w, self.d)
        h = emb
        for layer in self.temporal:
            h = _temporal(h, layer)
        h = _layer_norm(h, self.out_w, self.out_b)
        out = {
            "logits": _linear(h, self.head_wT, self.head_b),
            "ball": _linear(h, self.ball_wT, self.ball_b),
        }
        if self.switch_head:
            out["switch"] = _linear(h, self.switch_wT, self.switch_b)[..., 0]
        if self.touch_head:
            out["touch"] = _linear(h, self.touch_wT, self.touch_b)[..., 0]
        return out

    def encode_frames(self, tokens: np.ndarray) -> np.ndarray:
        """`(M, n_tokens, F)` -> `(M, d)` frame embeddings, before the temporal stack. With
        `slot_temporal` the M frames are read as one window in time order."""
        if self.slot_temporal:
            x = np.asarray(tokens, dtype=np.float64)
            slot = self._slot_branch(x[None])[0]
            return self._encode_all(np.ascontiguousarray(x[..., : self.n_features]), slot)
        return self._encode_all(tokens, None)

    def _encode_all(self, tokens: np.ndarray, slot: np.ndarray | None) -> np.ndarray:
        m = tokens.shape[0]
        out = np.empty((m, self.d), dtype=np.float64)
        step = 8192
        for start in range(0, m, step):
            stop = min(m, start + step)
            part = None if slot is None else slot[start:stop]
            out[start:stop] = self._encode_frames(tokens[start:stop], part)
        return out

    def _slot_branch(self, x: np.ndarray) -> np.ndarray:
        """`(B, W, n_tokens, n_features + 2)` -> `(B, W, 2K, d)`: `model.SlotTemporal`."""
        players = x[:, :, : 2 * self.k]
        valid = players[..., Col.VALID] > 0.5
        links = slot_links_np(
            players[..., self.n_features],
            players[..., self.n_features + 1],
            valid,
            self.k,
            self.slot_offsets,
        )
        # A tap with no partner reads row n, a zero row appended to the layer's input (torch reads
        # the token's own row times 0, the same value).
        n = valid.size
        rows = [
            np.stack([np.where(links[o][1], links[o][0], n) for o in layer.offsets], axis=-1)
            for layer in self.slot_layers
        ]
        h = _linear(players[..., self.slot_inputs], self.slot_in_wT, self.slot_in_b)
        for layer, taps in zip(self.slot_layers, rows, strict=True):
            u = _layer_norm(h, layer.ln_w, layer.ln_b)
            flat = np.concatenate([u.reshape(n, u.shape[-1]), np.zeros((1, u.shape[-1]))])
            gathered = np.take(flat, taps, axis=0)  # (B, W, 2K, kernel, d)
            c = _linear(gathered.reshape(*taps.shape[:-1], -1), layer.conv_wT, layer.conv_b)
            h = h + _linear(_gelu(c), layer.pw_wT, layer.pw_b)
        out = _layer_norm(h, self.slot_out_ln_w, self.slot_out_ln_b)
        return np.where(valid[..., None], _linear(out, self.slot_out_wT, self.slot_out_b), 0.0)

    def _encode_frames(self, tokens: np.ndarray, slot: np.ndarray | None = None) -> np.ndarray:
        valid = tokens[..., Col.VALID] > 0.5
        key_bias = np.where(valid, 0.0, MASKED)
        h = _gelu(_linear(tokens, self.embed0_wT, self.embed0_b))
        h = _linear(h, self.embed2_wT, self.embed2_b)
        if slot is not None:
            h[:, : 2 * self.k] += slot
        for block in self.blocks:
            h = _set_block(h, key_bias, block, self.heads, self.head_dim, self.attn_scale)
        h = _layer_norm(h, self.norm_w, self.norm_b)
        pooled = []
        for team in range(2):
            sl = slice(team * self.k, (team + 1) * self.k)
            part = h[:, sl]
            keys = _linear(part, self.pool_wT, None)
            score = matmul(keys, self.pool_query[team][:, None])[..., 0] * self.pool_scale
            seen = valid[:, sl]
            score = np.where(seen, score, MASKED)
            weight = _softmax(score) * seen
            pooled.append(np.sum(weight[..., None] * part, axis=1))
        cat = np.concatenate([pooled[0], pooled[1], h[:, -1]], axis=-1)
        if self.possessor_head:
            cat = np.concatenate([cat, self._possessor_sums(h, valid)], axis=-1)
        return _gelu(
            _linear(_layer_norm(cat, self.frame_ln_w, self.frame_ln_b), self.frame_wT, self.frame_b)
        )

    def _possessor_sums(self, h: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """(M, 3) softmax mass on home, away and nobody. Invalid player logits are MASKED."""
        k = self.k
        players = h[:, : 2 * k]
        player_logit = _linear(players, self.poss_wT, self.poss_b)[..., 0]
        nobody = _linear(h[:, -1], self.none_wT, self.none_b)
        player_logit = np.where(valid[:, : 2 * k], player_logit, MASKED)
        logits = np.concatenate([player_logit, nobody], axis=-1)
        probs = _softmax(logits)
        return np.stack(
            [
                probs[:, :k].sum(axis=-1),
                probs[:, k : 2 * k].sum(axis=-1),
                probs[:, 2 * k],
            ],
            axis=-1,
        )


def _set_block(
    h: np.ndarray,
    key_bias: np.ndarray,
    block: _Attn,
    heads: int,
    head_dim: int,
    scale: float,
) -> np.ndarray:
    m, n, d = h.shape
    qkv = _linear(_layer_norm(h, block.n1_w, block.n1_b), block.qkv_wT, block.qkv_b)
    qkv = np.transpose(qkv.reshape(m, n, 3, heads, head_dim), (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    scores = scores + block.rel[None, :, :, :] + key_bias[:, None, None, :]
    attended = np.matmul(_softmax(scores), v)
    attended = np.transpose(attended, (0, 2, 1, 3)).reshape(m, n, d)
    h = h + _linear(attended, block.proj_wT, block.proj_b)
    hidden = _gelu(_linear(_layer_norm(h, block.n2_w, block.n2_b), block.ffn0_wT, block.ffn0_b))
    return h + _linear(hidden, block.ffn3_wT, block.ffn3_b)


def _conv_same(
    x: np.ndarray, taps: list[np.ndarray], bias: np.ndarray, dilation: int
) -> np.ndarray:
    """Channels-last conv1d with the same zero padding as `nn.Conv1d` (odd kernel)."""
    b, length, _cin = x.shape
    cout = bias.shape[0]
    pad = dilation * (len(taps) - 1) // 2
    out = np.zeros((b, length, cout), dtype=np.float64)
    out += bias
    for tap, weight_t in enumerate(taps):
        shift = tap * dilation - pad
        if shift == 0:
            out += matmul(x, weight_t)
        elif 0 < shift < length:
            out[:, : length - shift] += matmul(x[:, shift:], weight_t)
        elif -length < shift < 0:
            s = -shift
            out[:, s:] += matmul(x[:, : length - s], weight_t)
    return out


def _shift_frames(x: np.ndarray, offset: int, fill: Any) -> np.ndarray:
    """`y[:, t] = x[:, t + offset]` along axis 1, `fill` where t + offset is outside the window."""
    w = x.shape[1]
    out = np.full_like(x, fill)
    if 0 <= offset < w:
        out[:, : w - offset] = x[:, offset:]
    elif -w < offset < 0:
        out[:, -offset:] = x[:, : w + offset]
    return out


def slot_links_np(
    slot: np.ndarray,
    run: np.ndarray,
    valid: np.ndarray,
    tokens_per_team: int,
    offsets: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Same as `model.slot_links`: for each offset, `(row, found)`, both `(B, W, 2K)`. `row` is
    the flattened `(B * W * 2K)` row of the partner at t + offset (same SLOT_ID and SLOT_RUN in
    the same team block, both valid), and the token's own row where `found` is false."""
    b, w, n2 = slot.shape
    k = tokens_per_team
    rows = np.arange(b * w * n2, dtype=np.int64).reshape(b, w, 2, k)
    here_slot, here_run, here_valid = (
        slot.reshape(b, w, 2, k),
        run.reshape(b, w, 2, k),
        valid.reshape(b, w, 2, k),
    )
    links: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for offset in offsets:
        if offset == 0:
            links[0] = (rows.reshape(b, w, n2), here_valid.reshape(b, w, n2))
            continue
        same = (
            (here_slot[..., :, None] == _shift_frames(here_slot, offset, 0.0)[..., None, :])
            & (here_run[..., :, None] == _shift_frames(here_run, offset, 0.0)[..., None, :])
            & here_valid[..., :, None]
            & _shift_frames(here_valid, offset, False)[..., None, :]
        )
        found = same.any(axis=-1)
        # token i's partner is token argmax_j at t + offset (SLOT_ID is unique in a block)
        partner = np.take_along_axis(_shift_frames(rows, offset, 0), same.argmax(-1), axis=-1)
        links[offset] = (np.where(found, partner, rows).reshape(b, w, n2), found.reshape(b, w, n2))
    return links


def _temporal(x: np.ndarray, layer: _Temp) -> np.ndarray:
    h = _layer_norm(x, layer.ln_w, layer.ln_b)
    h = _gelu(_conv_same(h, layer.taps, layer.conv_b, layer.dilation))
    return x + _linear(h, layer.pw_wT, layer.pw_b)


def _f32_encode(array: np.ndarray, encoding: str) -> str:
    raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
    return raw.hex() if encoding == "hex" else base64.b64encode(raw).decode("ascii")


def _f32_decode(data: str, encoding: str) -> np.ndarray:
    raw = bytes.fromhex(data) if encoding == "hex" else base64.b64decode(data, validate=True)
    return np.frombuffer(raw, dtype="<f4")


def dump_numpy_weights(
    path: str | Path,
    *,
    arch: str,
    config: dict[str, Any],
    token_spec: dict[str, Any],
    arrays: dict[str, np.ndarray],
    feature_version: str = FEATURE_VERSION,
    encoding: str = "hex",
) -> Path:
    """Write the JSON weight file. Each parameter is little-endian float32 bytes, hex-encoded
    (format 1) or, with `encoding="base64"`, base64-encoded (format 2, about two thirds the size).
    The stored bytes are the same either way."""
    if arch != "team_net":
        raise ValueError(f"numpy export supports team_net, got {arch!r}")
    versions = {name: version for version, name in ENCODINGS.items()}
    if encoding not in versions:
        raise ValueError(f"encoding must be one of {sorted(versions)}, got {encoding!r}")
    layout = parameter_layout(config)
    expect = {name for name, _shape in layout}
    if set(arrays) != expect:
        raise ValueError("weight names do not match the team_net layout for this config")
    parameters = []
    for name, shape in layout:
        array = np.ascontiguousarray(arrays[name])
        if tuple(array.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(array.shape)} != {shape}")
        data = _f32_encode(array, encoding)
        parameters.append({"name": name, "shape": [int(s) for s in shape], "data": data})
    doc = {
        "format_version": versions[encoding],
        "arch": arch,
        "feature_version": feature_version,
        "config": config,
        "token_spec": token_spec,
        "parameters": parameters,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return path


def load_numpy_weights(path: str | Path) -> NumpyTeamNet:
    """The model stored by `dump_numpy_weights` / `python -m aaaag.ml.export`."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not a JSON team_net weight file") from error
    encoding = ENCODINGS.get(doc.get("format_version"))
    if encoding is None:
        raise ValueError(
            f"{path}: weight format {doc.get('format_version')!r}, expected one of {sorted(ENCODINGS)}"
        )
    if doc.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"{path}: trained on feature version {doc.get('feature_version')!r}, "
            f"this code builds {FEATURE_VERSION!r}"
        )
    if doc.get("arch") != "team_net":
        raise ValueError(f"{path}: numpy inference supports team_net, got {doc.get('arch')!r}")
    weights: dict[str, np.ndarray] = {}
    for entry in doc["parameters"]:
        shape = tuple(int(s) for s in entry["shape"])
        raw = _f32_decode(entry["data"], encoding)
        if raw.size != math.prod(shape):
            raise ValueError(f"{entry['name']}: {raw.size} values for shape {shape}")
        weights[entry["name"]] = np.array(raw.reshape(shape), dtype=np.float64, copy=True)
    return NumpyTeamNet(doc["config"], doc["token_spec"], weights)


# One thread per worker. Set in the parent before spawn (the child imports numpy, which reads
# these, before the initializer runs) and again in the initializer before any GEMM.
_BLAS_ENV = (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("GOTO_NUM_THREADS", "1"),
    ("OMP_DYNAMIC", "FALSE"),
    ("MKL_DYNAMIC", "FALSE"),
)
# Tests set this so a worker task raises. Unset in normal runs.
_FAIL_ENV = "AAAAG_NP_INFER_FAIL"
_WORKER: dict[str, Any] = {}


def _resolve_workers(workers: int) -> int:
    """`1` is the serial loop. `0` or negative is min(4, cpu_count), and at least 1."""
    if workers > 0:
        return workers
    return max(1, min(4, os.cpu_count() or 1))


def _model_paths(models: Sequence[NumpyTeamNet | str | Path]) -> tuple[str, ...] | None:
    """Filesystem paths the workers can load, or None when an in-memory net has to stay serial."""
    paths: list[str] = []
    for item in models:
        if isinstance(item, NumpyTeamNet):
            return None
        paths.append(os.fspath(item))
    return tuple(paths)


def _push_blas_env() -> list[tuple[str, str | None]]:
    saved = [(key, os.environ.get(key)) for key, _value in _BLAS_ENV]
    for key, value in _BLAS_ENV:
        os.environ[key] = value
    return saved


def _pop_blas_env(saved: list[tuple[str, str | None]]) -> None:
    for key, value in saved:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _kept_logits(
    model: NumpyTeamNet,
    spec: TokenSpec,
    window: np.ndarray,
    lo: int,
    keep_lo: int,
    keep_hi: int,
    views: int,
) -> np.ndarray:
    """Float32 logits for frames [keep_lo, keep_hi). Same arithmetic as one serial chunk."""
    total = None
    n_views = len(VIEWS[views])
    for flip_y, swap in VIEWS[views]:
        x = symmetric(window, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else window
        z = model.forward(x[None])["logits"][0]
        if swap:
            z = z[:, SWAP_CLASSES]
        total = z if total is None else total + z
    keep = slice(keep_lo - lo, keep_hi - lo)
    assert total is not None
    return (total[keep] / n_views).astype(np.float32)


def _kept_switch(
    model: NumpyTeamNet,
    spec: TokenSpec,
    window: np.ndarray,
    lo: int,
    keep_lo: int,
    keep_hi: int,
    views: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Class logits and the switch logit for one chunk. The class half matches `_kept_logits`.

    A team swap exchanges the H and A class logits and leaves the switch logit alone: a switch
    has no direction. Views are averaged, then cast to float32, the same reduction as the classes.
    """
    total = None
    total_s = None
    n_views = len(VIEWS[views])
    for flip_y, swap in VIEWS[views]:
        x = symmetric(window, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else window
        forward = model.forward(x[None])
        z = forward["logits"][0]
        if swap:
            z = z[:, SWAP_CLASSES]
        total = z if total is None else total + z
        s = forward["switch"][0]
        total_s = s if total_s is None else total_s + s
    keep = slice(keep_lo - lo, keep_hi - lo)
    assert total is not None and total_s is not None
    logits = (total[keep] / n_views).astype(np.float32)
    switch = (total_s[keep] / n_views).astype(np.float32)
    return logits, switch


def _init_worker(weight_paths: tuple[str, ...]) -> None:
    """Pin BLAS to one thread, then load each JSON weight file once for this process."""
    for key, value in _BLAS_ENV:
        os.environ[key] = value
    # Kept alive for the process: dropping the limiter restores the old thread count.
    _WORKER["limiter"] = threadpool_limits(limits=1)
    _WORKER["models"] = [load_numpy_weights(path) for path in weight_paths]


def _worker_chunk(
    job: tuple[int, int, int, int, int, np.ndarray],
) -> tuple[int, int, np.ndarray]:
    """One chunk of one model. Returns kept-frame logits; the parent places them by index."""
    reason = os.environ.get(_FAIL_ENV)
    if reason:
        raise RuntimeError(f"numpy inference worker failed: {reason}")
    model_index, lo, keep_lo, keep_hi, views, window = job
    model = _WORKER["models"][model_index]
    with threadpool_limits(limits=1):
        values = _kept_logits(model, model.spec, window, lo, keep_lo, keep_hi, views)
    if values.shape != (keep_hi - keep_lo, 3):
        raise RuntimeError(f"chunk logits shape {values.shape}, expected {(keep_hi - keep_lo, 3)}")
    return keep_lo, keep_hi, values


def _parallel_total(
    paths: tuple[str, ...],
    loaded: Sequence[NumpyTeamNet],
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
    workers: int,
) -> np.ndarray:
    """Sum of each model's float32 chunk logits, in model order, as float64. Same reduction as
    the serial loop in `predict_logits_np`."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    plans = [plan_chunks(tuple(periods), chunk, model.receptive_half) for model in loaded]
    n_pool = min(workers, max(len(plan) for plan in plans))
    total = np.zeros((len(tokens), 3), dtype=np.float64)
    # The child imports numpy before `_init_worker`, so the variables have to be in the
    # environment at spawn, not only inside the initializer.
    saved = _push_blas_env()
    try:
        with ProcessPoolExecutor(
            max_workers=n_pool,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(paths,),
        ) as pool:
            for model_index, plan in enumerate(plans):
                jobs = [
                    (
                        model_index,
                        part.lo,
                        part.keep_lo,
                        part.keep_hi,
                        views,
                        np.array(tokens[part.lo : part.hi], copy=True, order="C"),
                    )
                    for part in plan
                ]
                out = np.full((len(tokens), 3), np.nan, dtype=np.float32)
                for keep_lo, keep_hi, values in pool.map(_worker_chunk, jobs, chunksize=1):
                    out[keep_lo:keep_hi] = values
                total += out
    finally:
        _pop_blas_env(saved)
    return total


def _logits_and_switch_np(
    model: NumpyTeamNet,
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`(T, 3)` class logits and `(T,)` switch logits, both float32, NaN outside the chunks."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    logits = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    switch = np.full(len(tokens), np.nan, dtype=np.float32)
    for part in plan_chunks(tuple(periods), chunk, model.receptive_half):
        window = tokens[part.lo : part.hi]
        z, s = _kept_switch(model, spec, window, part.lo, part.keep_lo, part.keep_hi, views)
        logits[part.keep_lo : part.keep_hi] = z
        switch[part.keep_lo : part.keep_hi] = s
    return logits, switch


def _serial_switch(
    loaded: Sequence[NumpyTeamNet],
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of class logits and the mean switch logit. Models without the head add classes only."""
    total = np.zeros((len(tokens), 3), dtype=np.float64)
    switch_total = np.zeros(len(tokens), dtype=np.float64)
    n_switch = 0
    for model in loaded:
        if model.switch_head:
            logits, sw = _logits_and_switch_np(
                model, spec, tokens, periods, chunk=chunk, views=views
            )
            total += logits
            switch_total += sw
            n_switch += 1
        else:
            total += logits_from_tokens_np(model, spec, tokens, periods, chunk=chunk, views=views)
    if n_switch == 0:
        raise RuntimeError("switch ensemble called with no switch head")
    return total, (switch_total / n_switch).astype(np.float32)


def _worker_chunk_switch(
    job: tuple[int, int, int, int, int, np.ndarray],
) -> tuple[int, int, np.ndarray, np.ndarray | None]:
    """One chunk. The fourth value is the switch logit when that model has the head."""
    reason = os.environ.get(_FAIL_ENV)
    if reason:
        raise RuntimeError(f"numpy inference worker failed: {reason}")
    model_index, lo, keep_lo, keep_hi, views, window = job
    model = _WORKER["models"][model_index]
    with threadpool_limits(limits=1):
        if model.switch_head:
            values, switch = _kept_switch(model, model.spec, window, lo, keep_lo, keep_hi, views)
        else:
            values = _kept_logits(model, model.spec, window, lo, keep_lo, keep_hi, views)
            switch = None
    if values.shape != (keep_hi - keep_lo, 3):
        raise RuntimeError(f"chunk logits shape {values.shape}, expected {(keep_hi - keep_lo, 3)}")
    return keep_lo, keep_hi, values, switch


def _parallel_switch(
    paths: tuple[str, ...],
    loaded: Sequence[NumpyTeamNet],
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`_parallel_total` plus the mean switch logit of the models that have the head."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    plans = [plan_chunks(tuple(periods), chunk, model.receptive_half) for model in loaded]
    n_pool = min(workers, max(len(plan) for plan in plans))
    total = np.zeros((len(tokens), 3), dtype=np.float64)
    switch_total = np.zeros(len(tokens), dtype=np.float64)
    n_switch = 0
    saved = _push_blas_env()
    try:
        with ProcessPoolExecutor(
            max_workers=n_pool,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(paths,),
        ) as pool:
            for model_index, plan in enumerate(plans):
                jobs = [
                    (
                        model_index,
                        part.lo,
                        part.keep_lo,
                        part.keep_hi,
                        views,
                        np.array(tokens[part.lo : part.hi], copy=True, order="C"),
                    )
                    for part in plan
                ]
                out = np.full((len(tokens), 3), np.nan, dtype=np.float32)
                sw = np.full(len(tokens), np.nan, dtype=np.float32)
                has = loaded[model_index].switch_head
                for keep_lo, keep_hi, values, switch in pool.map(
                    _worker_chunk_switch, jobs, chunksize=1
                ):
                    out[keep_lo:keep_hi] = values
                    if switch is not None:
                        sw[keep_lo:keep_hi] = switch
                total += out
                if has:
                    switch_total += sw
                    n_switch += 1
    finally:
        _pop_blas_env(saved)
    if n_switch == 0:
        raise RuntimeError("switch ensemble called with no switch head")
    return total, (switch_total / n_switch).astype(np.float32)


def _kept_scalars(
    model: NumpyTeamNet,
    spec: TokenSpec,
    window: np.ndarray,
    lo: int,
    keep_lo: int,
    keep_hi: int,
    views: int,
    names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Class logits and the scalar heads in `names` for one chunk.

    The class half matches `_kept_switch`: a team swap exchanges H and A. Each scalar is averaged
    over the views and is not reordered. A y flip and a team swap both leave a touch logit as the
    forward wrote it, because a touch has no team.
    """
    total = None
    totals: dict[str, np.ndarray | None] = {name: None for name in names}
    n_views = len(VIEWS[views])
    for flip_y, swap in VIEWS[views]:
        x = symmetric(window, spec, flip_y=flip_y, swap=swap) if (flip_y or swap) else window
        forward = model.forward(x[None])
        z = forward["logits"][0]
        if swap:
            z = z[:, SWAP_CLASSES]
        total = z if total is None else total + z
        for name in names:
            value = forward[name][0]
            totals[name] = value if totals[name] is None else totals[name] + value
    keep = slice(keep_lo - lo, keep_hi - lo)
    assert total is not None
    logits = (total[keep] / n_views).astype(np.float32)
    scalars = {}
    for name in names:
        accumulated = totals[name]
        assert accumulated is not None
        scalars[name] = (accumulated[keep] / n_views).astype(np.float32)
    return logits, scalars


def _logits_and_scalars_np(
    model: NumpyTeamNet,
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
    names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`(T, 3)` class logits and one `(T,)` array per name, NaN outside the chunks."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    logits = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    scalars = {name: np.full(len(tokens), np.nan, dtype=np.float32) for name in names}
    for part in plan_chunks(tuple(periods), chunk, model.receptive_half):
        window = tokens[part.lo : part.hi]
        z, got = _kept_scalars(
            model, spec, window, part.lo, part.keep_lo, part.keep_hi, views, names
        )
        logits[part.keep_lo : part.keep_hi] = z
        for name in names:
            scalars[name][part.keep_lo : part.keep_hi] = got[name]
    return logits, scalars


def _serial_scalars(
    loaded: Sequence[NumpyTeamNet],
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
    names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sum of class logits and the mean of each scalar head. A model without that head adds classes
    only, the same split `_serial_switch` uses."""
    total = np.zeros((len(tokens), 3), dtype=np.float64)
    sums = {name: np.zeros(len(tokens), dtype=np.float64) for name in names}
    counts = {name: 0 for name in names}
    for model in loaded:
        present = tuple(name for name in names if getattr(model, f"{name}_head"))
        if present:
            logits, scalars = _logits_and_scalars_np(
                model, spec, tokens, periods, chunk=chunk, views=views, names=present
            )
            total += logits
            for name in present:
                sums[name] += scalars[name]
                counts[name] += 1
        else:
            total += logits_from_tokens_np(model, spec, tokens, periods, chunk=chunk, views=views)
    out: dict[str, np.ndarray] = {}
    for name in names:
        if counts[name] == 0:
            raise RuntimeError(f"{name} ensemble called with no {name} head")
        out[name] = (sums[name] / counts[name]).astype(np.float32)
    return total, out


def _worker_chunk_scalars(
    job: tuple[int, int, int, int, int, np.ndarray],
) -> tuple[int, int, np.ndarray, dict[str, np.ndarray]]:
    """One chunk. The dict holds each scalar head that model has (switch, touch, or both)."""
    reason = os.environ.get(_FAIL_ENV)
    if reason:
        raise RuntimeError(f"numpy inference worker failed: {reason}")
    model_index, lo, keep_lo, keep_hi, views, window = job
    model = _WORKER["models"][model_index]
    names = tuple(name for name in ("switch", "touch") if getattr(model, f"{name}_head"))
    with threadpool_limits(limits=1):
        if names:
            values, scalars = _kept_scalars(
                model, model.spec, window, lo, keep_lo, keep_hi, views, names
            )
        else:
            values = _kept_logits(model, model.spec, window, lo, keep_lo, keep_hi, views)
            scalars = {}
    if values.shape != (keep_hi - keep_lo, 3):
        raise RuntimeError(f"chunk logits shape {values.shape}, expected {(keep_hi - keep_lo, 3)}")
    return keep_lo, keep_hi, values, scalars


def _parallel_scalars(
    paths: tuple[str, ...],
    loaded: Sequence[NumpyTeamNet],
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int,
    views: int,
    workers: int,
    names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`_parallel_switch` for every scalar head in `names`. Models without a head add classes only."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    plans = [plan_chunks(tuple(periods), chunk, model.receptive_half) for model in loaded]
    n_pool = min(workers, max(len(plan) for plan in plans))
    total = np.zeros((len(tokens), 3), dtype=np.float64)
    sums = {name: np.zeros(len(tokens), dtype=np.float64) for name in names}
    counts = {name: 0 for name in names}
    saved = _push_blas_env()
    try:
        with ProcessPoolExecutor(
            max_workers=n_pool,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(paths,),
        ) as pool:
            for model_index, plan in enumerate(plans):
                jobs = [
                    (
                        model_index,
                        part.lo,
                        part.keep_lo,
                        part.keep_hi,
                        views,
                        np.array(tokens[part.lo : part.hi], copy=True, order="C"),
                    )
                    for part in plan
                ]
                out = np.full((len(tokens), 3), np.nan, dtype=np.float32)
                present = tuple(
                    name for name in names if getattr(loaded[model_index], f"{name}_head")
                )
                buffers = {name: np.full(len(tokens), np.nan, dtype=np.float32) for name in present}
                for keep_lo, keep_hi, values, scalars in pool.map(
                    _worker_chunk_scalars, jobs, chunksize=1
                ):
                    out[keep_lo:keep_hi] = values
                    for name in present:
                        buffers[name][keep_lo:keep_hi] = scalars[name]
                total += out
                for name in present:
                    sums[name] += buffers[name]
                    counts[name] += 1
    finally:
        _pop_blas_env(saved)
    averaged: dict[str, np.ndarray] = {}
    for name in names:
        if counts[name] == 0:
            raise RuntimeError(f"{name} ensemble called with no {name} head")
        averaged[name] = (sums[name] / counts[name]).astype(np.float32)
    return total, averaged


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


def _as_model(item: NumpyTeamNet | str | Path) -> NumpyTeamNet:
    if isinstance(item, NumpyTeamNet):
        return item
    return load_numpy_weights(item)


def logits_from_tokens_np(
    model: NumpyTeamNet,
    spec: TokenSpec,
    tokens: np.ndarray,
    periods: Sequence,
    *,
    chunk: int = 2048,
    views: int = 1,
) -> np.ndarray:
    """`(T, 3)` float32 logits over the played frames, NaN elsewhere. Chunking and views match
    `infer.logits_from_tokens`: `chunk` kept frames, `receptive_half` of context, views 1, 2 or 4.
    Serial; `predict_logits_np(..., workers=n)` runs these same chunks in a process pool."""
    if views not in VIEWS:
        raise ValueError(f"views must be one of {sorted(VIEWS)}, got {views}")
    out = np.full((len(tokens), 3), np.nan, dtype=np.float32)
    margin = model.receptive_half
    for part in plan_chunks(tuple(periods), chunk, margin):
        window = tokens[part.lo : part.hi]
        out[part.keep_lo : part.keep_hi] = _kept_logits(
            model, spec, window, part.lo, part.keep_lo, part.keep_hi, views
        )
    return out


def predict_logits_np(
    models: Sequence[NumpyTeamNet | str | Path],
    tracking: Tracking,
    *,
    chunk: int = 2048,
    views: int = 1,
    workers: int = 1,
) -> Prediction:
    """Mean logits of JSON weight files or `NumpyTeamNet`s. Same `Prediction` contract as
    `infer.predict_logits`: NaN outside the periods and where no player is observed.

    `workers=1` is the serial loop and does not change the caller's BLAS thread count.
    `workers>1` runs the chunks in a spawn process pool, one BLAS thread per worker, and
    stitches them in plan order; that matches the serial loop when the serial loop is also on
    one BLAS thread.     `workers<=0` uses min(4, os.cpu_count()). The pool is capped at the number
    of chunks. An in-memory `NumpyTeamNet` stays on the serial loop, because a worker loads
    weights from the JSON path rather than from a pickled array.

    When a model has `switch_head`, `Prediction.switch` is the mean of that logit over the
    models that have it and over the views, NaN outside the periods. No such model leaves
    `switch` unset and the class-logit reduction is the one above.
    When a model has `touch_head`, `Prediction.touch` is that same mean of the touch logit.
    A y flip and a team swap do not reorder it. No such model leaves `touch` unset.
    """
    loaded = [_as_model(item) for item in models]
    if not loaded:
        raise ValueError("predict_logits needs at least one checkpoint")
    spec = loaded[0].spec
    if any(model.spec != spec for model in loaded):
        raise ValueError("checkpoints in one ensemble must share a token spec")
    tokens = build_match_tokens(tracking, spec)
    periods = tracking.meta.periods
    n_workers = _resolve_workers(workers)
    paths = _model_paths(models) if n_workers > 1 else None
    total = np.zeros((tracking.n_frames, 3), dtype=np.float64)
    switch = None
    touch = None
    wants_switch = any(model.switch_head for model in loaded)
    wants_touch = any(model.touch_head for model in loaded)
    # Touch uses one forward that can also carry switch, so a model with both heads is not added
    # twice. Switch alone stays on the path that existed before the touch head.
    if wants_touch:
        names = ("switch", "touch") if wants_switch else ("touch",)
        if paths is not None and plan_chunks(tuple(periods), chunk, 0):
            total, scalars = _parallel_scalars(
                paths,
                loaded,
                tokens,
                periods,
                chunk=chunk,
                views=views,
                workers=n_workers,
                names=names,
            )
        else:
            total, scalars = _serial_scalars(
                loaded, spec, tokens, periods, chunk=chunk, views=views, names=names
            )
        switch = scalars.get("switch")
        touch = scalars["touch"]
    elif wants_switch:
        if paths is not None and plan_chunks(tuple(periods), chunk, 0):
            total, switch = _parallel_switch(
                paths, loaded, tokens, periods, chunk=chunk, views=views, workers=n_workers
            )
        else:
            total, switch = _serial_switch(loaded, spec, tokens, periods, chunk=chunk, views=views)
    elif paths is not None and plan_chunks(tuple(periods), chunk, 0):
        total = _parallel_total(
            paths, loaded, tokens, periods, chunk=chunk, views=views, workers=n_workers
        )
    else:
        for model in loaded:
            total += logits_from_tokens_np(model, spec, tokens, periods, chunk=chunk, views=views)
    logits = (total / len(loaded)).astype(np.float32)
    seen_players = tokens[:, : 2 * spec.tokens_per_team, Col.VALID].sum(axis=1) > 0
    opinion = tracking.meta.played_mask() & seen_players
    logits[~opinion] = np.nan
    played = tracking.meta.played_mask()
    if switch is not None:
        switch = np.array(switch, dtype=np.float32, copy=True)
        switch[~played] = np.nan
    if touch is not None:
        touch = np.array(touch, dtype=np.float32, copy=True)
        touch[~played] = np.nan
    if switch is None and touch is None:
        return Prediction(logits=logits, opinion=opinion)
    return Prediction(logits=logits, opinion=opinion, switch=switch, touch=touch)
