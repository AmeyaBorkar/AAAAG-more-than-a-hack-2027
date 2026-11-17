"""TeamNetTx (E81, #86): TeamNet with attention over time in place of the dilated convolutions.

    tokens (B, W, 2K+1, F)
      -> TeamNet's per-frame set encoder, unchanged (self-attention over the players and the ball,
         team pooling, optional possessor head)
      -> `t_blocks` pre-LN transformer blocks over the frames. Each frame attends only to frames
         within +-`t_radius` of itself (a band mask), with a learned bias per head and per relative
         offset, so the receptive field is exactly +-(t_blocks * t_radius) frames
      -> the same heads as TeamNet (class logits, ball, switch, touch)

Why: the question "does a temporal transformer beat our temporal convolutions?" (#86, E81). The
defaults give +-128 frames, the same reach as v11b's dilated stack (+-127), so the comparison is
attention against convolution and not context length (E31, E33 and E70 already measured context).

Signals: only the token tensor, like TeamNet. The band mask keeps the output at a frame a function
of frames within +-receptive_half only, so chunked inference (infer.py) gives the same answer as a
single pass. Where it fails: no built-in locality inside the band (a convolution's kernel has it),
which the published comparisons on small tracking datasets found costly (ballradar, KDD 2023;
PathCRF, KDD 2026); attention cost grows with W * (2 * t_radius + 1) per block, so CPU inference is
slower than the convolutions. No numpy engine (np_infer) exists for it; it runs through torch only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ..model import ModelConfig, TeamNet, register_architecture

TX_MASKED = -1e9


@dataclass(frozen=True)
class TxConfig(ModelConfig):
    """ModelConfig plus the temporal attention settings. `dilations` is unused (empty)."""

    dilations: tuple[int, ...] = ()
    t_blocks: int = 2
    t_radius: int = 64
    t_heads: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.d_model % self.t_heads:
            raise ValueError(f"d_model {self.d_model} is not a multiple of t_heads {self.t_heads}")
        if self.t_blocks < 1 or self.t_radius < 1:
            raise ValueError("t_blocks and t_radius must be >= 1")
        if self.dilations:
            raise ValueError("team_net_tx has no dilated convolutions; leave dilations empty")

    @property
    def receptive_half(self) -> int:
        """Frames of context the temporal stack reads on each side: t_blocks * t_radius."""
        return self.t_blocks * self.t_radius

    def to_dict(self) -> dict[str, Any]:
        raw = super().to_dict()
        raw.update({k: v for k, v in asdict(self).items() if k.startswith("t_")})
        return raw


class TemporalAttentionBlock(nn.Module):
    """Pre-LN transformer block over the frame axis with a band mask and a relative-offset bias."""

    def __init__(self, d: int, heads: int, radius: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim, self.radius = heads, d // heads, radius
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.offset_bias = nn.Parameter(torch.zeros(heads, 2 * radius + 1))
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def bias(self, w: int, device: torch.device) -> torch.Tensor:
        """(heads, W, W) additive scores: the offset bias inside the band, TX_MASKED outside."""
        pos = torch.arange(w, device=device)
        offset = pos[None, :] - pos[:, None]  # key minus query
        inside = offset.abs() <= self.radius
        index = (offset.clamp(-self.radius, self.radius) + self.radius).long()
        table = self.offset_bias[:, index]  # (heads, W, W)
        return torch.where(inside, table, torch.full_like(table, TX_MASKED))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, d)
        b, w, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, w, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, W, head_dim)
        scores = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        scores = scores + self.bias(w, x.device).to(scores.dtype)
        attended = (scores.softmax(dim=-1).to(v.dtype) @ v).transpose(1, 2).reshape(b, w, d)
        x = x + self.drop(self.proj(attended))
        return x + self.drop(self.ffn(self.norm2(x)))


@register_architecture("team_net_tx", TxConfig)
class TeamNetTx(TeamNet):
    """TeamNet's frame encoder and heads; attention over time instead of dilated convolutions."""

    def __init__(self, cfg: TxConfig | None = None) -> None:
        cfg = cfg or TxConfig()
        super().__init__(cfg)  # builds no TemporalBlock: cfg.dilations is empty
        self.temporal_tx = nn.ModuleList(
            TemporalAttentionBlock(
                cfg.d_model, cfg.t_heads, cfg.t_radius, cfg.ffn_mult, cfg.dropout
            )
            for _ in range(cfg.t_blocks)
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same inputs and outputs as `TeamNet.forward`; see its docstring."""
        b, w = tokens.shape[:2]
        embedding, possessor = self._encode_frames(tokens.reshape(b * w, *tokens.shape[2:]))
        x = embedding.view(b, w, -1)
        for block in self.temporal_tx:
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
