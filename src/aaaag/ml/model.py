"""TeamNet: a per-frame set encoder over the token layout, then a small non-causal temporal stack.

    tokens (B, W, 2K+1, F)
      -> per frame: pre-LN self-attention over the 2K+1 tokens, with a learned bias per head that
         depends only on the relation of two tokens (same team, other team, to the ball, from the
         ball, ball itself) and invalid tokens masked out as keys
      -> per frame: attention pooling inside each team (a learned query per team) and the ball
         token, concatenated and projected to one embedding
      -> dilated residual 1-D convolutions over the frames (non-causal, zero padded)
      -> 3-class logits (H, A, D) and an auxiliary ball position (x, y in units of `ref_scale_m`,
         attack-normalised) read from the same embedding

With `ModelConfig.possessor_head` (off by default) each player token and the ball token also get a
logit after the set norm. Invalid player tokens are masked, a softmax over the 2K players plus a
"nobody" slot is summed into home, away and nobody, and those three numbers are concatenated onto
the frame vector (`frame` then sees 3d+3). `forward` returns the logits under "possessor". The
flag off builds the same modules, the same initial weights and the same outputs as without it.

With `ModelConfig.switch_head` (off by default) a linear layer reads the temporal output, after
`out_norm`, and `forward` returns its logit under "switch" with shape (B, W). The layer is created
after every other parameter, so the flag off draws the same initial weights as a model without it.
The head does not change the frame width or the class and ball outputs.

With `ModelConfig.touch_head` (off by default) the same temporal output feeds one more linear
layer, created after the switch layer, and `forward` returns its logit under "touch" with shape
(B, W). The flag off draws the same initial weights and returns the same dict as a model without
it. A touch has no team, so the logit is not reordered by a team swap.

With `ModelConfig.slot_temporal.enabled` (off by default; it needs `TokenSpec.slot_index`, whose two
identifier columns follow the `n_features` embedded ones) a `SlotTemporal` branch runs before the
set blocks. Each player token's motion columns (`inputs`) are projected to `d` channels, then a
stack of residual dilated convolutions (kernel `kernel`, one layer per entry of `dilations`) runs
along that token's own tracker slot: the tap at offset o reads the token of frame t + o that has the
same SLOT_ID and SLOT_RUN in the same team block, and reads zero where there is none (the slot is
unobserved or outside the 2K nearest there, a new fragment run started, or t + o is outside the
window). The result goes back into the token through a projection whose weight and bias start at
zero, so a freshly built branch adds exactly 0 and the network computes what the parent computed,
bit for bit. The branch is created after every other layer, so the flag off draws the same initial
weights and builds the same state dict as before. Its receptive field adds
`sum(dilations) * (kernel - 1) / 2` frames to the temporal stack's.

The encoder is permutation-equivariant inside each team block and has no positional or identity
embedding; teams enter through the block layout and the team flag. SLOT_ID only says which tokens
of two frames are the same tracker slot. It is never embedded, so relabelling the slots changes
nothing. The temporal stack has a finite receptive field of +-`receptive_half` frames, so a long
match can be processed in chunks with that margin and give the same answer as a single pass
(infer.py). Every normalisation is per frame, so nothing depends on the window length.

Signals: only the token tensor. Fails where a sequence model of 3 000 frames would not: possession
that depends on events more than ~5 s away (a long ball gap that the ball-state producer should
have turned into a stoppage anyway). Size: about 0.2M parameters (`count_parameters`).
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .spec import N_FEATURES, SLOT_INDEX_FEATURES, SLOT_RUN_MOD, Col

N_RELATIONS = 5  # same team, other team, player -> ball, ball -> player, ball -> ball
MASKED = -1e9  # an additive score for an invalid key: all-invalid rows stay finite (uniform)
SLOT_INPUTS = (Col.X, Col.Y, Col.VX, Col.VY, Col.SPEED)  # default slot_temporal.inputs

# -- architectures: drop in another design by adding a file under aaaag/ml/archs/ ---------------------
#
# An architecture is a pair: a config dataclass (`to_dict`, `from_dict`, a `receptive_half`
# property in frames, and `tokens_per_team` if it reads the token layout) and an `nn.Module` built
# from it (`model.cfg` is the config) whose forward takes tokens (B, W, 2K+1, N_FEATURES) and
# returns {"logits": (B, W, 3) over (H, A, D), "ball": (B, W, 2) optional} and, when
# `possessor_head` is on, "possessor": (B, W, 2K+1) logits. When `switch_head` is on it also
# returns "switch": (B, W) logits. When `touch_head` is on it also returns "touch": (B, W)
# logits. It must be
# non-causal-safe: the output at a frame may depend on frames within +-receptive_half only, because
# inference cuts a match into chunks with that much context (infer.py) and the loss skips that
# margin at the edges of a training window. Training, checkpoints, inference, the producer and the
# fold scoring need nothing else.
ARCHITECTURES: dict[str, tuple[type, type]] = {}
ARCH_PACKAGE = "aaaag.ml.archs"
_arch_modules_loaded = False


def register_architecture(name: str, config_cls: type) -> Callable[[type], type]:
    """Class decorator: register an `nn.Module` and its config class under `name`."""

    def decorator(model_cls: type) -> type:
        known = ARCHITECTURES.get(name)
        if known is not None and known[1].__module__ != model_cls.__module__:
            raise ValueError(
                f"architecture {name!r} is already registered by {known[1].__module__}"
            )
        model_cls.arch = name
        ARCHITECTURES[name] = (config_cls, model_cls)
        return model_cls

    return decorator


def get_architecture(name: str) -> tuple[type, type]:
    """(config class, model class) of a registered architecture; imports aaaag.ml.archs.* once."""
    global _arch_modules_loaded
    if not _arch_modules_loaded:
        package = importlib.import_module(ARCH_PACKAGE)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{ARCH_PACKAGE}."):
            importlib.import_module(info.name)
        _arch_modules_loaded = True
    if name not in ARCHITECTURES:
        raise KeyError(f"no architecture {name!r}; available: {', '.join(sorted(ARCHITECTURES))}")
    return ARCHITECTURES[name]


def build_model(name: str, cfg: Any) -> nn.Module:
    """A fresh model of architecture `name` from its config object."""
    return get_architecture(name)[1](cfg)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True)
class SlotTemporalConfig:
    """`model.slot_temporal`: a dilated conv stack along each tracker slot's own series.

    `inputs` are token columns (integers here; a run config may name them, see
    `spec.slot_input_columns`). `d` channels, an odd `kernel`, one residual layer per dilation.
    The largest tap offset must stay under `2 * SLOT_RUN_MOD` frames, the distance inside which a
    run id is unique on its slot.
    """

    enabled: bool = False
    d: int = 32
    kernel: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    inputs: tuple[int, ...] = SLOT_INPUTS

    def __post_init__(self) -> None:
        for name in ("dilations", "inputs"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        if not isinstance(self.enabled, bool):
            raise ValueError(f"model.slot_temporal.enabled must be a bool, got {self.enabled!r}")
        if not _positive_int(self.d):
            raise ValueError(f"model.slot_temporal.d must be an integer >= 1, got {self.d!r}")
        if not _positive_int(self.kernel) or self.kernel % 2 == 0:
            raise ValueError(
                f"model.slot_temporal.kernel must be odd and >= 1, got {self.kernel!r}"
            )
        if not isinstance(self.dilations, tuple) or not self.dilations:
            raise ValueError("model.slot_temporal.dilations must be a non-empty list")
        if not all(_positive_int(d) for d in self.dilations):
            raise ValueError(
                f"model.slot_temporal.dilations must be integers >= 1, got {self.dilations!r}"
            )
        inputs = self.inputs
        if not isinstance(inputs, tuple) or not inputs:
            raise ValueError("model.slot_temporal.inputs must be a non-empty list of columns")
        if not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in inputs):
            raise ValueError(
                f"model.slot_temporal.inputs must be column indices, got {inputs!r} (names such as "
                "'x' or 'accel' are resolved only by a run config, against its token spec)"
            )
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"model.slot_temporal.inputs repeats a column: {inputs!r}")
        if self.max_offset >= 2 * SLOT_RUN_MOD:
            raise ValueError(
                f"model.slot_temporal reaches {self.max_offset} frames in one tap; a run id is "
                f"unique on its slot only within {2 * SLOT_RUN_MOD - 1}"
            )

    @property
    def max_offset(self) -> int:
        """The largest distance in frames between a token and one of its taps."""
        return max(self.dilations) * (self.kernel - 1) // 2

    @property
    def receptive_half(self) -> int:
        """Frames each side that one token's output reads (0 when off)."""
        if not self.enabled:
            return 0
        return sum(d * (self.kernel - 1) // 2 for d in self.dilations)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["dilations"] = list(self.dilations)
        raw["inputs"] = list(self.inputs)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SlotTemporalConfig:
        return cls(**raw)


@dataclass(frozen=True)
class ModelConfig:
    tokens_per_team: int = 11
    n_features: int = N_FEATURES  # columns the network embeds (TokenSpec.model_features)
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    ffn_mult: int = 2
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    kernel: int = 3
    dropout: float = 0.1
    possessor_head: bool = False  # per-token possessor logits; off keeps today's network
    switch_head: bool = False  # one logit per frame for an H↔A switch; off keeps today's network
    touch_head: bool = False  # one logit per frame for a ball touch; off keeps today's network
    # A per-slot temporal branch before the set blocks; disabled keeps today's network.
    slot_temporal: SlotTemporalConfig = field(default_factory=SlotTemporalConfig)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} is not a multiple of n_heads {self.n_heads}")
        if self.kernel % 2 == 0:
            raise ValueError("kernel must be odd so that 'same' padding is symmetric")
        slot = self.slot_temporal
        if isinstance(slot, dict):
            slot = SlotTemporalConfig.from_dict(slot)
            object.__setattr__(self, "slot_temporal", slot)
        if not isinstance(slot, SlotTemporalConfig):
            raise ValueError(f"model.slot_temporal must be a mapping, got {slot!r}")
        if slot.enabled and max(slot.inputs) >= self.n_features:
            raise ValueError(
                f"model.slot_temporal.inputs {slot.inputs} must be embedded columns "
                f"(< n_features {self.n_features})"
            )

    @property
    def n_tokens(self) -> int:
        return 2 * self.tokens_per_team + 1

    @property
    def token_width(self) -> int:
        """Columns of the token tensor the model takes: `n_features`, plus SLOT_ID and SLOT_RUN
        when the slot branch is on."""
        return self.n_features + (SLOT_INDEX_FEATURES if self.slot_temporal.enabled else 0)

    @property
    def receptive_half(self) -> int:
        """Frames of context the network reads on each side: the temporal stack's plus the slot
        branch's (0 when it is off)."""
        temporal = sum(d * (self.kernel - 1) // 2 for d in self.dilations)
        return temporal + self.slot_temporal.receptive_half

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["dilations"] = list(self.dilations)
        if not self.possessor_head:
            raw.pop("possessor_head", None)
        if not self.switch_head:
            raw.pop("switch_head", None)
        if not self.touch_head:
            raw.pop("touch_head", None)
        if self.slot_temporal.enabled:
            raw["slot_temporal"] = self.slot_temporal.to_dict()
        else:
            raw.pop("slot_temporal", None)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        return cls(**{k: tuple(v) if k == "dilations" else v for k, v in raw.items()})


def relation_matrix(tokens_per_team: int) -> torch.Tensor:
    """(n, n) long: the relation id of token i attending to token j (see N_RELATIONS)."""
    k = tokens_per_team
    n = 2 * k + 1
    team = torch.tensor([0] * k + [1] * k + [2])
    rel = torch.zeros(n, n, dtype=torch.long)
    for i in range(n):
        for j in range(n):
            if team[i] == 2:
                rel[i, j] = 4 if team[j] == 2 else 3
            elif team[j] == 2:
                rel[i, j] = 2
            else:
                rel[i, j] = 0 if team[i] == team[j] else 1
    return rel


class SetBlock(nn.Module):
    """Pre-LN transformer block whose attention logits carry a per-head relation bias."""

    def __init__(self, d: int, heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.relation_bias = nn.Parameter(torch.zeros(heads, N_RELATIONS))
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d)
        )

    def forward(
        self, h: torch.Tensor, key_bias: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        m, n, d = h.shape
        qkv = self.qkv(self.norm1(h)).view(m, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (m, heads, n, head_dim)
        scores = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        scores = scores + self.relation_bias[:, relation] + key_bias  # (heads, n, n), (m, 1, 1, n)
        attended = (scores.softmax(dim=-1).to(v.dtype) @ v).transpose(1, 2).reshape(m, n, d)
        h = h + self.proj(attended)
        return h + self.ffn(self.norm2(h))


class TemporalBlock(nn.Module):
    """x + pointwise(gelu(dilated conv(layer norm(x)))), over the frame axis, channels last."""

    def __init__(self, d: int, dilation: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.pointwise = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, d)
        h = self.norm(x).transpose(1, 2)
        h = self.pointwise(F.gelu(self.conv(h))).transpose(1, 2)
        return x + self.drop(h)


def _shift_frames(x: torch.Tensor, offset: int, fill: float) -> torch.Tensor:
    """`y[:, t] = x[:, t + offset]` along axis 1, `fill` where t + offset is outside the window."""
    if offset == 0:
        return x
    w = x.shape[1]
    pad = x.new_full((x.shape[0], min(abs(offset), w), *x.shape[2:]), fill)
    if offset > 0:
        return torch.cat([x[:, offset:], pad], dim=1)[:, :w]
    return torch.cat([pad, x[:, : max(w + offset, 0)]], dim=1)[:, :w]


def slot_links(
    slot: torch.Tensor,
    run: torch.Tensor,
    valid: torch.Tensor,
    tokens_per_team: int,
    offsets: tuple[int, ...],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """For each offset o, `(row, found)`: where token (b, t, i) reads its tap at t + o.

    `found` (B, W, 2K) bool is true when frame t + o of the same window holds a token in the same
    team block with token i's SLOT_ID and SLOT_RUN, and both tokens are valid. `row` (B, W, 2K)
    long is that token's row in the flattened (B * W * 2K) tokens, and token i's own row where
    `found` is false (a missing slot, a new fragment run, the window edge, token i invalid), so a
    caller that multiplies by `found` reads zero there. Offset 0 is the token itself, where valid.
    """
    b, w, n2 = slot.shape
    k = tokens_per_team
    rows = torch.arange(b * w * n2, device=slot.device).view(b, w, 2, k)
    blocks = (slot.view(b, w, 2, k), run.view(b, w, 2, k), valid.view(b, w, 2, k))
    links: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for offset in offsets:
        if offset == 0:
            links[0] = (rows.view(b, w, n2), blocks[2].reshape(b, w, n2))
            continue
        other_slot, other_run, other_valid, other_rows = (
            _shift_frames(blocks[0], offset, 0.0),
            _shift_frames(blocks[1], offset, 0.0),
            _shift_frames(blocks[2], offset, False),
            _shift_frames(rows, offset, 0),
        )
        same = (
            (blocks[0][..., :, None] == other_slot[..., None, :])
            & (blocks[1][..., :, None] == other_run[..., None, :])
            & blocks[2][..., :, None]
            & other_valid[..., None, :]
        )  # (B, W, 2, K, K): token i here, token j at t + offset
        found = same.any(dim=-1)
        j = same.to(torch.uint8).argmax(dim=-1)  # SLOT_ID is unique in a block
        partner = torch.gather(other_rows, -1, j)  # token i's partner: token j at t + offset
        links[offset] = (
            torch.where(found, partner, rows).view(b, w, n2),
            found.view(b, w, n2),
        )
    return links


class SlotTemporalBlock(nn.Module):
    """h + pointwise(gelu(conv(layer norm(h)))), where the conv's taps follow the token's slot."""

    def __init__(self, d: int, kernel: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Linear(kernel * d, d)  # the taps side by side, earliest frame first
        self.pointwise = nn.Linear(d, d)


class SlotTemporal(nn.Module):
    """Dilated residual convolutions along each player token's own tracker slot (see the module
    docstring). Returns (B, W, 2K, d_model), zero on invalid tokens and, at initialisation,
    everywhere.

    A valid token reads only valid taps of its own slot and run, so an invalid token's hidden
    state is never read and is not cleared between layers; the output is zeroed there at the end.
    Under autocast the normalised taps are cast to the autocast dtype before they are gathered.
    The tap layer would make that cast anyway, so the values are the same and the gathered copies
    are half the size.
    """

    def __init__(self, cfg: SlotTemporalConfig, d_model: int, tokens_per_team: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokens_per_team = tokens_per_team
        half = (cfg.kernel - 1) // 2
        self.taps = tuple(tuple(j * d for j in range(-half, half + 1)) for d in cfg.dilations)
        self.offsets = tuple(sorted({o for taps in self.taps for o in taps}))
        self.in_proj = nn.Linear(len(cfg.inputs), cfg.d)
        self.layers = nn.ModuleList(SlotTemporalBlock(cfg.d, cfg.kernel) for _ in cfg.dilations)
        self.out_norm = nn.LayerNorm(cfg.d)
        self.out_proj = nn.Linear(cfg.d, d_model)
        nn.init.zeros_(self.out_proj.weight)  # step 0 adds exactly zero: the parent's network
        nn.init.zeros_(self.out_proj.bias)
        self.register_buffer("inputs", torch.tensor(cfg.inputs, dtype=torch.long), persistent=False)

    def forward(self, tokens: torch.Tensor, n_features: int) -> torch.Tensor:
        """tokens (B, W, n_tokens, n_features + 2): SLOT_ID and SLOT_RUN are the last two."""
        k = self.tokens_per_team
        players = tokens[:, :, : 2 * k]
        valid = players[..., Col.VALID] > 0.5
        with torch.no_grad():
            links = slot_links(
                players[..., n_features], players[..., n_features + 1], valid, k, self.offsets
            )
            # per layer: (B, W, 2K, kernel) rows of its taps, earliest first, and their masks
            gathers = [
                (
                    torch.stack([links[o][0] for o in taps], dim=-1),
                    torch.stack([links[o][1] for o in taps], dim=-1).unsqueeze(-1),
                )
                for taps in self.taps
            ]
        device = tokens.device.type
        cast = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else None
        h = self.in_proj(players.index_select(-1, self.inputs))
        for layer, (rows, found) in zip(self.layers, gathers, strict=True):
            u = layer.norm(h)
            flat = (u if cast is None else u.to(cast)).reshape(-1, u.shape[-1])
            c = layer.conv((flat[rows] * found).flatten(-2))  # (B, W, 2K, kernel * d)
            h = h + layer.pointwise(F.gelu(c))
        out = self.out_proj(self.out_norm(h))
        return torch.where(valid.unsqueeze(-1), out, 0.0)


@register_architecture("team_net", ModelConfig)
class TeamNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.d_model
        self.embed = nn.Sequential(nn.Linear(cfg.n_features, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            SetBlock(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.n_blocks)
        )
        self.norm = nn.LayerNorm(d)
        self.pool_key = nn.Linear(d, d, bias=False)
        self.pool_query = nn.Parameter(torch.randn(2, d) * d**-0.5)
        # The three possessor sums join this vector only when the head is on. Off, the width is
        # 3d, so this layer draws the same initial weights as a model without the flag.
        frame_in = 3 * d + (3 if cfg.possessor_head else 0)
        self.frame = nn.Sequential(nn.LayerNorm(frame_in), nn.Linear(frame_in, d), nn.GELU())
        self.temporal = nn.ModuleList(
            TemporalBlock(d, dilation, cfg.kernel, cfg.dropout) for dilation in cfg.dilations
        )
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3)
        self.ball_head = nn.Linear(d, 2)
        self.register_buffer("relation", relation_matrix(cfg.tokens_per_team), persistent=False)
        # After every other layer, so a flag that is off does not consume initialisation randomness.
        # The switch layer follows the possessor layers. The touch layer is last, so it does not
        # move the switch layer's initial weights.
        if cfg.possessor_head:
            self.possessor = nn.Linear(d, 1)
            self.possessor_none = nn.Linear(d, 1)
        if cfg.switch_head:
            self.switch = nn.Linear(d, 1)
        if cfg.touch_head:
            self.touch = nn.Linear(d, 1)
        # Last of all: off, nothing is drawn; on, every layer above keeps its initial weights.
        if cfg.slot_temporal.enabled:
            self.slot_temporal = SlotTemporal(cfg.slot_temporal, d, cfg.tokens_per_team)

    @property
    def receptive_half(self) -> int:
        return self.cfg.receptive_half

    def encode_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        """(M, n_tokens, F) -> (M, d): the per-frame embedding, before the temporal stack. With
        the slot branch on, the M frames are read as one window in time order."""
        slot = None
        if self.cfg.slot_temporal.enabled:
            slot = self._slot_branch(tokens[None])[0]
            tokens = tokens[..., : self.cfg.n_features].contiguous()
        return self._encode_frames(tokens, slot)[0]

    def _slot_branch(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, W, n_tokens, n_features + 2) -> (B, W, 2K, d) slot-branch addend."""
        width = self.cfg.token_width
        if tokens.shape[-1] != width:
            raise ValueError(
                f"slot_temporal needs tokens with {width} columns (n_features "
                f"{self.cfg.n_features} plus SLOT_ID and SLOT_RUN: token.slot_index), "
                f"got {tokens.shape[-1]}"
            )
        return self.slot_temporal(tokens, self.cfg.n_features)

    def _possessor_logits(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(M, 2K+1) logits from the post-norm tokens. Invalid players are MASKED; slot 2K is nobody."""
        k = self.cfg.tokens_per_team
        player = self.possessor(h[:, : 2 * k]).squeeze(-1)
        player = player.masked_fill(~valid[:, : 2 * k], MASKED)
        nobody = self.possessor_none(h[:, -1])
        return torch.cat([player, nobody], dim=-1)

    def _encode_frames(
        self, tokens: torch.Tensor, slot: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Frame embedding and, when the head is on, possessor logits of shape (M, 2K+1).

        `slot` (M, 2K, d), the slot branch's output, is added to the embedded player tokens.
        """
        k = self.cfg.tokens_per_team
        valid = tokens[..., Col.VALID] > 0.5  # (M, n)
        key_bias = torch.where(valid, 0.0, MASKED)[:, None, None, :]
        h = self.embed(tokens)
        if slot is not None:
            h = torch.cat([h[:, : 2 * k] + slot.to(h.dtype), h[:, 2 * k :]], dim=1)
        for block in self.blocks:
            h = block(h, key_bias, self.relation)
        h = self.norm(h)
        pooled = []
        for team in range(2):
            part = h[:, team * k : (team + 1) * k]
            score = (self.pool_key(part) @ self.pool_query[team]) * part.shape[-1] ** -0.5
            weight = score.masked_fill(~valid[:, team * k : (team + 1) * k], MASKED).softmax(-1)
            weight = weight * valid[:, team * k : (team + 1) * k]  # an empty team pools to zero
            pooled.append((weight.unsqueeze(-1).to(part.dtype) * part).sum(dim=1))
        frame_in = torch.cat([pooled[0], pooled[1], h[:, -1]], dim=-1)
        possessor = None
        if self.cfg.possessor_head:
            possessor = self._possessor_logits(h, valid)
            probs = possessor.softmax(dim=-1)
            sums = torch.stack(
                (
                    probs[:, :k].sum(dim=-1),
                    probs[:, k : 2 * k].sum(dim=-1),
                    probs[:, 2 * k],
                ),
                dim=-1,
            )
            frame_in = torch.cat([frame_in, sums], dim=-1)
        return self.frame(frame_in), possessor

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """tokens (B, W, n_tokens, F) -> {"logits": (B, W, 3), "ball": (B, W, 2)}.

        With the possessor head, the dict also has "possessor": (B, W, 2K+1) logits. Without it
        that key is absent. With the switch head, "switch" is (B, W) logits read from the same
        normalised temporal output as the class head. Without it that key is absent. With the
        touch head, "touch" is (B, W) logits from that same output. Without it that key is absent.
        With the slot branch, F is n_features + 2 (SLOT_ID, SLOT_RUN last); only the first
        n_features columns are embedded.
        """
        b, w = tokens.shape[:2]
        slot = None
        if self.cfg.slot_temporal.enabled:
            slot = self._slot_branch(tokens).reshape(b * w, 2 * self.cfg.tokens_per_team, -1)
            tokens = tokens[..., : self.cfg.n_features].contiguous()
        embedding, possessor = self._encode_frames(tokens.reshape(b * w, *tokens.shape[2:]), slot)
        x = embedding.view(b, w, -1)
        for block in self.temporal:
            x = block(x)
        x = self.out_norm(x)
        out = {"logits": self.head(x), "ball": self.ball_head(x)}
        if possessor is not None:
            out["possessor"] = possessor.view(b, w, -1)
        if self.cfg.switch_head:
            out["switch"] = self.switch(x).squeeze(-1)
        if self.cfg.touch_head:
            out["touch"] = self.touch(x).squeeze(-1)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
