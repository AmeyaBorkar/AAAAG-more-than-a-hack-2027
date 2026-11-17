"""Semi-Markov CRF loss through the shipped dead-gated HSMM (E69).

Tables come from ``decoders.hsmm`` (``_log_pmf``, ``_log_surv``, ``_transition_logs``,
``DEFAULT_MODEL``), with block 5, dt 0.2 s, dmax (200, 200, 500), self-transition 0.02 and beta 0.
The opening segment is left-censored (uncapped survival plus a uniform start). The closing segment
is right-censored and capped at dmax. The H↔A bonus is ``switch_weight`` times the clamped switch
logit of the block that ends at that boundary.

Team evidence is ``0.5 * lam_team * clamp(z_H - z_A)`` on alive frames and 0 on dead frames. The
clamp is the decoder's: ±log(999) for the team log-odds (``CLIP`` 1e-3) and ±``SWITCH_CLIP`` for
the switch logit, with a straight-through backward. A block is dead when at least 3 of its 5
frames are dead. In the constrained term a 2-2 alive tie, and a block sitting mostly inside a true
H/A spell shorter than 38 frames, may be either team.

The partition recursion is float64, outside autocast, and on the CPU: a GPU launch per block
costs seconds, while the tensors are small enough to move. Forbidden segments are one additive mask of
about −1e5, chosen from an integer count of forbidden blocks, never folded into the emission
prefix. ``map_labels`` is the max-semiring decode of the decoder's own frame emissions (including
``lam_dead``), so mixed blocks match ``dead_gated_hsmm``. The training NLL does not feed
``lam_dead``: it is a constant on the frames the mask already fixes, and it would give z_D a
gradient.

Where it fails: blocks start at the first interior frame, not at the period start; the first and
last blocks are censored as if they were the period edges; a 1-4 frame tail is outside the energy
but still in the divisor. Off unless ``train.scrf_weight`` is positive.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from ..contracts import A, D, H, X
from ..decoders.dead_gated_hsmm import GATED_LAM_DEAD, gate_posterior
from ..decoders.hsmm import (
    CLIP,
    DEFAULT_MODEL,
    SWITCH_CLIP,
    _log_pmf,
    _log_surv,
    _period_switch_bonus,
    _transition_logs,
    frame_emissions,
)

# One forbidden segment. Far below any legal score, and identical for −1e4 and −1e5 in float64
# (both underflow in logsumexp). Never hsmm.NEG, and never added into a cumulative sum.
MASK = -1e5
BLOCK = 5
DT = 0.2
DMAX = (200, 200, 500)
SELF_PROB = 0.02
# posterior_logits clips a probability at CLIP, so the team log-odds stop at logit(1 - CLIP).
TEAM_CLIP = float(math.log(1.0 - CLIP) - math.log(CLIP))
SHORT_SPELL = 38  # frames; 1.5 s at 25 fps, shorter than this is free in Z_c


def ste_clamp(values: torch.Tensor, limit: float) -> torch.Tensor:
    """Decoder clamp in the forward, identity in the backward."""
    limit = float(limit)
    return values + (values.clamp(-limit, limit) - values).detach()


def short_spell_mask(labels: np.ndarray, shorter_than: int = SHORT_SPELL) -> np.ndarray:
    """(T,) bool, True on frames of an H or A run shorter than ``shorter_than``.

    The run length is the one in ``labels``, so a window that cuts a spell does not shorten it.
    Swapping H and A leaves the mask unchanged: a run boundary is a change of label, and both
    teams count. A y flip does not read the labels.
    """
    lab = np.asarray(labels)
    n = int(lab.shape[0])
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = lab[1:] != lab[:-1]
    run_id = np.cumsum(change) - 1
    length = np.bincount(run_id)[run_id]
    team = (lab == H) | (lab == A)
    out[team & (length < int(shorter_than))] = True
    return out


@lru_cache(maxsize=4)
def _cached_tables(
    nb: int, dt: float, dmax: tuple[int, int, int], self_prob: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Duration, survival and transition tables as ``_hsmm_period`` builds them.

    Invalid durations are ``MASK``, not ``hsmm.NEG``. Valid entries are copied out of
    ``_log_pmf`` and ``_log_surv``.
    """
    lp = np.full((3, nb + 1), MASK)
    ls = np.zeros((3, nb + 1))
    for k in range(3):
        pmf = _log_pmf(DEFAULT_MODEL, k, dmax[k], dt)
        m = min(dmax[k], nb)
        lp[k, 1 : m + 1] = pmf[1 : m + 1]
        ls[k] = _log_surv(DEFAULT_MODEL, k, max(nb, dmax[k]), dt)[: nb + 1]
    capped = np.full((3, nb + 1), MASK)
    for k in range(3):
        m = min(dmax[k], nb)
        capped[k, 1 : m + 1] = ls[k, 1 : m + 1]
    return lp, ls, capped, _transition_logs(DEFAULT_MODEL, self_prob)


def decoder_tables(
    nb: int,
    dt: float = DT,
    dmax: tuple[int, int, int] = DMAX,
    self_prob: float = SELF_PROB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copies of the cached tables. ``nb`` is the number of blocks."""
    cached = _cached_tables(
        int(nb), float(dt), (int(dmax[0]), int(dmax[1]), int(dmax[2])), float(self_prob)
    )
    return tuple(array.copy() for array in cached)


def semi_markov(
    e: torch.Tensor,
    bonus: torch.Tensor,
    lp: torch.Tensor,
    ls: torch.Tensor,
    ls_cap: torch.Tensor,
    log_t: torch.Tensor,
    beta: float = 0.0,
    semiring: str = "log",
    allow: torch.Tensor | None = None,
    mask: float = MASK,
    return_path: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """(B,) log-partition or max score of the HSMM energy.

    ``e`` is (B, nb, 3) block emissions. ``bonus`` is (B, nb + 1): index ``t`` is added to an
    H↔A transition at the boundary where a new segment starts, i.e. the block that ends there.
    ``allow`` is (B, nb, 3); a segment of class k is impossible unless every block it covers
    allows k. ``None`` is the decoder's unconstrained recursion. With ``return_path`` (max
    semiring only) the second return is (B, nb) block labels of one best path.
    """
    if return_path and semiring != "max":
        raise ValueError("return_path needs the max semiring")
    if semiring == "log":

        def red(values: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.logsumexp(values, dim)

    elif semiring == "max":

        def red(values: torch.Tensor, dim: int) -> torch.Tensor:
            return values.max(dim).values

    else:
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")
    e = e.double()
    bonus = bonus.to(device=e.device, dtype=torch.float64)
    b, nb, n_cls = e.shape
    if n_cls != 3 or bonus.shape != (b, nb + 1):
        raise ValueError(
            f"emissions (B, nb, 3) and bonus (B, nb + 1) required, "
            f"got {tuple(e.shape)} and {tuple(bonus.shape)}"
        )
    if nb == 0:
        empty = e.new_zeros(b)
        return (empty, e.new_zeros(b, 0, dtype=torch.long)) if return_path else empty
    device = e.device
    lp = lp.to(device=device, dtype=torch.float64)
    ls = ls.to(device=device, dtype=torch.float64)
    ls_cap = ls_cap.to(device=device, dtype=torch.float64)
    log_t = log_t.to(device=device, dtype=torch.float64)
    cum = torch.cat((e.new_zeros(b, 1, 3), e.cumsum(1)), 1)
    log_pi = float(math.log(1.0 / 3.0))
    ha = e.new_zeros(3, 3)
    ha[H, A] = ha[A, H] = 1.0
    prefix = None
    if allow is not None:
        if tuple(allow.shape) != (b, nb, 3):
            raise ValueError(f"allow must be {(b, nb, 3)}, got {tuple(allow.shape)}")
        forbid = (~allow.to(device=device)).to(dtype=torch.int32)
        prefix = torch.cat((forbid.new_zeros(b, 1, 3), forbid.cumsum(1)), 1)
    zero = e.new_zeros(())
    neg = e.new_full((), float(mask))
    idx = torch.arange(nb + 1, device=device)
    seg_start = e.new_zeros(b, nb + 1, 3, dtype=torch.long) if return_path else None
    prev_class = e.new_zeros(b, nb + 1, 3, dtype=torch.long) if return_path else None
    final_alpha = None

    def penalty(starts: torch.Tensor, end: int) -> torch.Tensor:
        count = prefix[:, end].unsqueeze(1) - prefix[:, starts]
        return torch.where(count != 0, neg, zero)

    prev: list[torch.Tensor] = []
    out = e.new_zeros(b)
    for t in range(1, nb + 1):
        base = log_pi + ls[:, t]
        first = base.expand(b, 3) if prefix is None else base + penalty(idx[:1], t).squeeze(1)
        cands = [first.unsqueeze(1)]
        if t > 1:
            durs = t - idx[1:t]
            dur = (ls_cap if t == nb else lp)[:, durs].T
            seg = torch.stack(prev, 1) + dur.unsqueeze(0)
            if prefix is not None:
                seg = seg + penalty(idx[1:t], t)
            cands.append(seg)
        packed = torch.cat(cands, 1)
        alpha = cum[:, t] + beta + red(packed, 1)
        if return_path:
            # Index 0 is the opening segment (start 0). Later rows are starts 1, 2, ... so the
            # argmax index is the start block. Ties keep the earlier start, and the opening
            # segment wins a tie with a split, as in ``_hsmm_period`` (strict ``>``).
            seg_start[:, t] = packed.argmax(1)
        if t == nb:
            out = red(alpha, 1)
            final_alpha = alpha
            break
        step = alpha.unsqueeze(2) + log_t.unsqueeze(0) + bonus[:, t, None, None] * ha
        if return_path:
            prev_class[:, t] = step.argmax(1)
        prev.append(red(step, 1) - cum[:, t])
    if not return_path:
        return out
    return out, _recover_blocks(seg_start, prev_class, final_alpha)


def _recover_blocks(
    seg_start: torch.Tensor, prev_class: torch.Tensor, final_alpha: torch.Tensor
) -> torch.Tensor:
    """(B, nb) labels of one best path. ``seg_start[t, k]`` starts the class-k segment ending at t."""
    b = final_alpha.shape[0]
    nb = seg_start.shape[1] - 1
    labels = seg_start.new_full((b, nb), D)
    chosen = final_alpha.argmax(-1)
    for i in range(b):
        k = int(chosen[i])
        t = nb
        guard = 0
        while t > 0 and guard <= nb:
            s = int(seg_start[i, t, k])
            if s < 0 or s >= t:
                break
            labels[i, s:t] = k
            if s == 0:
                break
            k = int(prev_class[i, s, k])
            t = s
            guard += 1
    return labels


def constraint_masks(
    labels: torch.Tensor, short: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(allow_z, allow_zc)``, each (B, nb, 3), for the interior labels.

    A block is D when at least 3 frames are D. An alive block's team is the majority of its H/A
    frames; a tie allows either team in Z_c. A block with at least 3 frames inside a short H/A
    spell also allows either team in Z_c. Z always allows both teams on an alive block.
    """
    frames = labels.shape[1]
    nb = frames // BLOCK
    chunk = labels[:, : nb * BLOCK].reshape(labels.shape[0], nb, BLOCK)
    n_dead = chunk.eq(D).sum(-1)
    n_h = chunk.eq(H).sum(-1)
    n_a = chunk.eq(A).sum(-1)
    is_d = n_dead >= 3
    if short is None:
        n_short = torch.zeros_like(n_dead)
    else:
        bits = short[:, : nb * BLOCK].to(dtype=torch.bool).reshape(short.shape[0], nb, BLOCK)
        n_short = bits.sum(-1)
    free = (~is_d) & ((n_h == n_a) | (n_short >= 3))
    allow_z = torch.stack((~is_d, ~is_d, is_d), dim=-1)
    only_h = (~is_d) & (~free) & (n_h > n_a)
    only_a = (~is_d) & (~free) & (n_a > n_h)
    allow_c = torch.stack((~is_d & ~only_a, ~is_d & ~only_h, is_d), dim=-1)
    return allow_z, allow_c


def _zero(logits: torch.Tensor, switch: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return ((logits.sum() + switch.sum()) * 0.0).to(dtype=dtype)


def scrf_nll(
    logits: torch.Tensor,
    switch: torch.Tensor,
    labels: torch.Tensor,
    *,
    lam_team: float = 0.6,
    switch_weight: float = 2.0,
    short: torch.Tensor | None = None,
    log_t: torch.Tensor | np.ndarray | None = None,
    mask: float = MASK,
) -> torch.Tensor:
    """Mean NLL per interior frame. The divisor is ``B * T``, tail and all-D rows included.

    ``logits`` is (B, T, 3) and ``switch`` / ``labels`` are (B, T): the window interior. ``short``
    is the full-match short-spell mask sliced the same way; None is "no short spell". A row that
    contains X contributes 0 (the decoder does not cross a period) and stays in the divisor.
    An all-D row is 0 with zero gradient. The recursion runs in float64 with autocast off.
    """
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError(f"logits must be (B, T, 3), got {tuple(logits.shape)}")
    if switch.shape != logits.shape[:2] or labels.shape != logits.shape[:2]:
        raise ValueError(
            f"switch {tuple(switch.shape)} and labels {tuple(labels.shape)} "
            f"must match logits {tuple(logits.shape[:2])}"
        )
    if short is not None and short.shape != labels.shape:
        raise ValueError(f"short {tuple(short.shape)} must match labels {tuple(labels.shape)}")
    out_dtype = logits.dtype if logits.dtype in (torch.float32, torch.float64) else torch.float32
    b, frames, _ = logits.shape
    nb = frames // BLOCK
    if b == 0 or frames == 0 or nb == 0:
        return _zero(logits, switch, out_dtype)
    device_type = logits.device.type if logits.device.type in ("cpu", "cuda") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        loss = _scrf_forward(
            logits.double(),
            switch.double(),
            labels.to(dtype=torch.long),
            None if short is None else short.to(dtype=torch.bool),
            lam_team=float(lam_team),
            switch_weight=float(switch_weight),
            log_t=log_t,
            mask=float(mask),
            nb=nb,
        )
    return loss.to(dtype=out_dtype)


def _scrf_forward(
    logits: torch.Tensor,
    switch: torch.Tensor,
    labels: torch.Tensor,
    short: torch.Tensor | None,
    *,
    lam_team: float,
    switch_weight: float,
    log_t: torch.Tensor | np.ndarray | None,
    mask: float,
    nb: int,
) -> torch.Tensor:
    # 149 sequential steps. On a GPU each step is its own launch and the step takes seconds;
    # the tensors are a few hundred kilobytes, so the loop runs on the CPU and the scalar
    # (and its gradient) goes back to the caller's device.
    home = logits.device
    if home.type != "cpu":
        logits = logits.to("cpu")
        switch = switch.to("cpu")
        labels = labels.to("cpu")
        if short is not None:
            short = short.to("cpu")
    b, frames, _ = logits.shape
    used = nb * BLOCK
    keep = ~labels.eq(X).any(dim=1)
    dead = labels[:, :used].eq(D)
    odds = ste_clamp(logits[:, :used, H] - logits[:, :used, A], TEAM_CLIP)
    team = (0.5 * lam_team) * odds * (~dead).to(dtype=odds.dtype)
    emissions = torch.stack((team, -team, torch.zeros_like(team)), dim=-1)
    emissions = emissions.reshape(b, nb, BLOCK, 3).sum(dim=2)
    clipped = ste_clamp(switch[:, :used], SWITCH_CLIP)
    block_max = clipped.reshape(b, nb, BLOCK).amax(dim=-1)
    bonus = torch.cat((clipped.new_zeros(b, 1), switch_weight * block_max), 1)
    allow_z, allow_c = constraint_masks(labels, short)
    lp, ls, ls_cap, shipped = decoder_tables(nb)
    device = logits.device
    tables = [torch.tensor(array, dtype=torch.float64, device=device) for array in (lp, ls, ls_cap)]
    trans = shipped if log_t is None else log_t
    if torch.is_tensor(trans):
        tables.append(trans.to(device=device, dtype=torch.float64))
    else:
        tables.append(torch.tensor(np.asarray(trans), dtype=torch.float64, device=device))
    # Z and Z_c are one batch. The two halves share the emissions, so the backward is one pass.
    both = semi_markov(
        torch.cat((emissions, emissions), 0),
        torch.cat((bonus, bonus), 0),
        *tables,
        allow=torch.cat((allow_z, allow_c), 0),
        mask=mask,
    )
    nll = torch.where(keep, both[:b] - both[b:], torch.zeros_like(both[:b]))
    loss = nll.sum() / float(b * frames)
    return loss if home.type == "cpu" else loss.to(home)


def map_labels(
    logits: torch.Tensor | np.ndarray,
    switch: torch.Tensor | np.ndarray,
    meta,
    *,
    lam_team: float = 0.6,
    switch_weight: float = 2.0,
    dead_threshold: float = 0.4,
) -> np.ndarray:
    """Frame labels of the max-semiring path under the decoder's gated frame emissions.

    One row, or a batch of rows that share ``meta`` (the same periods). Emissions are
    ``frame_emissions(gate_posterior(softmax(logits)))`` with ``lam_dead`` 50, and the bonus is
    ``_period_switch_bonus``. The path is this module's max semiring, not ``_hsmm_period``.
    """
    single = logits.ndim == 2
    if torch.is_tensor(logits):
        logits_np = logits.detach().cpu().double().numpy()
        switch_np = switch.detach().cpu().double().numpy()
    else:
        logits_np = np.asarray(logits, dtype=np.float64)
        switch_np = np.asarray(switch, dtype=np.float64)
    if single:
        logits_np = logits_np[None]
        switch_np = switch_np[None]
    rows = [
        _map_one(logits_np[i], switch_np[i], meta, lam_team, switch_weight, dead_threshold)
        for i in range(len(logits_np))
    ]
    out = np.stack(rows)
    return out[0] if single else out


def _map_one(
    logits: np.ndarray,
    switch: np.ndarray,
    meta,
    lam_team: float,
    switch_weight: float,
    dead_threshold: float,
) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    proba = np.exp(shifted)
    proba /= proba.sum(axis=-1, keepdims=True)
    gated = gate_posterior(proba, meta, dead_threshold=float(dead_threshold))
    labels = np.full(len(logits), D, dtype=np.int8)
    for lo, hi in meta.runs():
        evidence = frame_emissions(gated[lo:hi], float(lam_team), GATED_LAM_DEAD)
        if len(evidence) == 0:
            continue
        starts = np.arange(0, len(evidence), BLOCK)
        blocks = np.add.reduceat(evidence, starts, axis=0)
        bonus = _period_switch_bonus(switch[lo:hi], BLOCK, float(switch_weight))
        lp, ls, ls_cap, log_t = decoder_tables(len(blocks))
        _score, block_y = semi_markov(
            torch.tensor(blocks[None], dtype=torch.float64),
            torch.tensor(bonus[None], dtype=torch.float64),
            torch.tensor(lp),
            torch.tensor(ls),
            torch.tensor(ls_cap),
            torch.tensor(log_t),
            semiring="max",
            return_path=True,
        )
        n = hi - lo
        seg = np.empty(n, dtype=np.int8)
        chosen = block_y[0].detach().cpu().numpy()
        for i, kind in enumerate(chosen):
            seg[i * BLOCK : min((i + 1) * BLOCK, n)] = np.int8(kind)
        labels[lo:hi] = seg
    outside = ~meta.played_mask()
    labels[outside] = X
    return labels
