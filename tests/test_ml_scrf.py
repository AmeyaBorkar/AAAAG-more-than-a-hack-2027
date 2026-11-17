"""Semi-Markov CRF loss through the shipped dead-gated HSMM (E69).

Synthetic tensors only. Skipped where torch is not installed.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aaaag.contracts import A, D, H, MatchMeta, Period  # noqa: E402
from aaaag.decoders.dead_gated_hsmm import dead_gated_hsmm  # noqa: E402
from aaaag.decoders.hsmm import (  # noqa: E402
    DEFAULT_MODEL,
    _log_pmf,
    _log_surv,
    _transition_logs,
)
from aaaag.ml.scrf import (  # noqa: E402
    BLOCK,
    DMAX,
    DT,
    MASK,
    SELF_PROB,
    SHORT_SPELL,
    TEAM_CLIP,
    constraint_masks,
    decoder_tables,
    map_labels,
    scrf_nll,
    semi_markov,
    short_spell_mask,
    ste_clamp,
)
from aaaag.ml.train import RunConfig, TrainConfig, loss_terms  # noqa: E402
from aaaag.paths import CONFIGS_DIR  # noqa: E402


def _cuts(nb):
    def rec(start):
        if start == nb:
            yield []
            return
        for end in range(start + 1, nb + 1):
            for rest in rec(end):
                yield [(start, end), *rest]

    yield from rec(0)


def _score(segs, e, bonus, tables):
    lp, ls, cap, log_t = tables
    nb = e.shape[0]
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(e, 0)])
    total = 0.0
    for i, (s, t, k) in enumerate(segs):
        total += cum[t, k] - cum[s, k]
        if s == 0:
            total += math.log(1.0 / 3.0) + ls[k, t]
        elif t == nb:
            total += cap[k, t - s]
        else:
            total += lp[k, t - s]
        if i:
            j = segs[i - 1][2]
            total += log_t[j, k]
            if {j, k} == {H, A}:
                total += bonus[s]
    return total


def _enumerate(nb):
    for cuts in _cuts(nb):
        for classes in itertools.product((H, A, D), repeat=len(cuts)):
            yield [(s, t, k) for (s, t), k in zip(cuts, classes, strict=True)]


def _sym_log_t():
    trans = np.asarray(DEFAULT_MODEL["trans"], dtype=np.float64)
    swapped = trans[np.ix_([1, 0, 2], [1, 0, 2])]
    sym = 0.5 * (trans + swapped)
    np.fill_diagonal(sym, 0.0)
    sym /= sym.sum(axis=1, keepdims=True)
    return _transition_logs({"trans": sym}, SELF_PROB)


def _meta(n):
    return MatchMeta("synthetic", "synthetic", n, 25, 105.0, 68.0, (Period(1, 0, n),))


def test_t1_log_partition_and_block_marginals():
    rng = np.random.default_rng(0)
    for nb in (1, 4, 7):
        tables = decoder_tables(nb)
        e_np = rng.normal(0.0, 0.7, size=(nb, 3))
        bonus_np = np.concatenate(([0.0], rng.normal(0.0, 0.8, size=nb)))
        scores = []
        cover = []
        for segs in _enumerate(nb):
            scores.append(_score(segs, e_np, bonus_np, tables))
            occ = np.zeros((nb, 3))
            for s, t, k in segs:
                occ[s:t, k] = 1.0
            cover.append(occ)
        scores = np.asarray(scores)
        log_z = np.logaddexp.reduce(scores)
        weight = np.exp(scores - log_z)
        marginal = sum(w * occ for w, occ in zip(weight, cover, strict=True))
        e = torch.tensor(e_np[None], dtype=torch.float64, requires_grad=True)
        bonus = torch.tensor(bonus_np[None], dtype=torch.float64)
        z = semi_markov(e, bonus, *[torch.tensor(x) for x in tables])
        assert abs(float(z[0].detach()) - log_z) <= 1e-10
        z.sum().backward()
        assert np.max(np.abs(e.grad[0].numpy() - marginal)) <= 1e-10


def test_t2_max_path_matches_dead_gated_hsmm():
    spec = [(D, 52), (H, 18), (A, 12), (D, 55), (A, 27), (D, 50)]
    labels = np.concatenate([np.full(n, k, dtype=np.int64) for k, n in spec])
    n = len(labels)
    logits = np.full((2, n, 3), -4.0)
    for row, scale in enumerate((12.0, 15.0)):
        for t, k in enumerate(labels):
            logits[row, t, k] = scale if k == D else scale - 1.0
        # A second alive opinion so a mixed block is not a one-hot tie.
        logits[row, labels == H, A] = -8.0
        logits[row, labels == A, H] = -8.0
    switch = np.zeros((2, n))
    switch[:, 48:56] = np.array([0.0, -12.0, -3.0, 1.0, 4.0, -1.0, 2.0, 0.5])
    switch[1, 120] = -12.0
    meta = _meta(n)
    got = map_labels(logits, switch, meta, dead_threshold=0.4)
    for row in range(2):
        shifted = logits[row] - logits[row].max(axis=-1, keepdims=True)
        proba = np.exp(shifted)
        proba /= proba.sum(axis=-1, keepdims=True)
        ref = dead_gated_hsmm(
            proba,
            meta,
            lam_team=0.6,
            switch_weight=2.0,
            switch=switch[row],
            refine_s=0.0,
            dead_threshold=0.4,
        )
        assert np.array_equal(got[row], ref)


def test_t3_mask_constants_agree_and_autocast_stays_float64():
    torch.manual_seed(0)
    logits = torch.randn(2, 20, 3)
    switch = torch.randn(2, 20)
    labels = torch.randint(0, 3, (2, 20))
    labels[:, 5:10] = D
    wide = scrf_nll(logits, switch, labels, mask=-1e4)
    tight = scrf_nll(logits, switch, labels, mask=-1e5)
    assert abs(float(wide - tight)) <= 1e-8
    tables = decoder_tables(4)
    e = torch.randn(2, 4, 3)
    bonus = torch.zeros(2, 5)
    with torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16):
        out = semi_markov(e, bonus, *[torch.tensor(x) for x in tables])
        loss = scrf_nll(logits, switch, labels)
    assert out.dtype == torch.float64
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_t4_loss_is_nonnegative_finite_and_small_on_a_clear_long_spell():
    torch.manual_seed(1)
    logits = torch.randn(3, 30, 3) * 1e3
    switch = torch.randn(3, 30) * 1e3
    labels = torch.randint(0, 3, (3, 30))
    logits = logits.requires_grad_(True)
    switch = switch.requires_grad_(True)
    loss = scrf_nll(logits, switch, labels)
    assert float(loss.detach()) >= -1e-9
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert torch.isfinite(switch.grad).all()

    frames = 80 + 80 + 50
    labels = torch.cat(
        (
            torch.full((80,), H),
            torch.full((80,), A),
            torch.full((50,), D),
        )
    )[None]
    logits = torch.full((1, frames, 3), -20.0)
    logits[0, :80, H] = 20.0
    logits[0, 80:160, A] = 20.0
    logits[0, 160:, D] = 20.0
    clear = scrf_nll(logits, torch.zeros(1, frames), labels)
    assert float(clear.detach()) < 1e-3
    assert float(clear.detach()) >= -1e-9


def test_t5_zero_gradients_and_straight_through_clamp():
    raw = torch.tensor([100.0], requires_grad=True)
    raw.clamp(-TEAM_CLIP, TEAM_CLIP).sum().backward()
    assert raw.grad.item() == 0.0
    raw2 = torch.tensor([100.0], requires_grad=True)
    ste_clamp(raw2, TEAM_CLIP).sum().backward()
    assert raw2.grad.item() == 1.0

    frames = 23
    logits = torch.zeros(1, frames, 3)
    logits[0, :, H] = 4.0
    logits[0, 5:10, :] = 0.0
    logits[0, 5:10, D] = 4.0
    logits[0, 0, H] = 100.0
    logits = logits.requires_grad_(True)
    switch = torch.zeros(1, frames, requires_grad=True)
    labels = torch.full((1, frames), H)
    labels[0, 5:10] = D
    scrf_nll(logits, switch, labels).backward()
    assert torch.equal(logits.grad[..., D], torch.zeros_like(logits.grad[..., D]))
    assert torch.equal(logits.grad[0, 5:10], torch.zeros_like(logits.grad[0, 5:10]))
    assert torch.equal(logits.grad[0, 20:], torch.zeros_like(logits.grad[0, 20:]))
    assert logits.grad[0, 0, H].abs() > 0.0

    full = torch.zeros(1, 20, 3)
    full[0, :, A] = 3.0
    full = full.requires_grad_(True)
    sw = torch.zeros(1, 20, requires_grad=True)
    lab = torch.full((1, 20), A)
    margin = 4
    scrf_nll(full[:, margin:-margin], sw[:, margin:-margin], lab[:, margin:-margin]).backward()
    assert torch.equal(full.grad[:, :margin], torch.zeros_like(full.grad[:, :margin]))
    assert torch.equal(full.grad[:, -margin:], torch.zeros_like(full.grad[:, -margin:]))


def test_t6_switch_gradient_hits_only_the_block_argmax():
    frames = 80
    labels = torch.cat((torch.full((40,), H), torch.full((40,), A)))[None]
    logits = torch.zeros(1, frames, 3)
    logits[0, :40, H] = 6.0
    logits[0, 40:, A] = 6.0
    switch = torch.zeros(1, frames)
    for block in range(frames // BLOCK):
        switch[0, block * BLOCK + (block % BLOCK)] = 0.4 + 0.01 * block
    switch[0, 35:40] = torch.tensor([0.0, -1.0, 3.0, 0.5, -0.2])
    switch = switch.requires_grad_(True)
    scrf_nll(logits, switch, labels).backward()
    grad = switch.grad[0]
    nb = frames // BLOCK
    for block in range(nb):
        sl = grad[block * BLOCK : (block + 1) * BLOCK]
        values = switch.detach()[0, block * BLOCK : (block + 1) * BLOCK]
        winner = int(values.argmax())
        others = [i for i in range(BLOCK) if i != winner]
        assert torch.equal(sl[others], torch.zeros(len(others)))
        if block == nb - 1:
            assert torch.equal(sl, torch.zeros(BLOCK))
    assert grad[37].abs() > 0.0


def test_t7_swap_invariance():
    torch.manual_seed(1)
    frames = 30
    logits = torch.randn(1, frames, 3)
    switch = torch.randn(1, frames)
    labels = torch.randint(0, 3, (1, frames))
    labels[:, :10] = H
    labels[:, 10:20] = A
    labels[:, 20:] = D
    flipped = logits.clone()
    flipped[..., H], flipped[..., A] = logits[..., A].clone(), logits[..., H].clone()
    swapped = torch.where(labels == H, A, torch.where(labels == A, H, labels))
    sym = _sym_log_t()
    same = scrf_nll(logits, switch, labels, log_t=sym)
    same_s = scrf_nll(flipped, switch, swapped, log_t=sym)
    assert abs(float(same.detach() - same_s.detach())) <= 1e-10
    shipped = scrf_nll(logits, switch, labels)
    shipped_s = scrf_nll(flipped, switch, swapped)
    assert abs(float(shipped.detach() - shipped_s.detach())) * frames <= 0.1


def test_t8_weight_zero_is_byte_identical():
    bare = RunConfig()
    assert "scrf_weight" not in bare.to_dict()["train"]
    explicit = RunConfig(
        train=TrainConfig(scrf_weight=0.0, scrf_lam_team=0.6, scrf_switch_weight=2.0)
    )
    assert explicit.to_dict() == bare.to_dict()
    on = RunConfig(model=bare.model.__class__(switch_head=True), train=TrainConfig(scrf_weight=1))
    assert RunConfig.from_dict(on.to_dict()) == on
    e69 = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e69.yaml")
    e56 = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e56a.yaml")
    got, base = e69.to_dict(), e56.to_dict()
    assert got["train"].pop("scrf_weight") == 1.0
    assert got["train"].pop("scrf_lam_team") == 0.6
    assert got["train"].pop("scrf_switch_weight") == 2.0
    assert got == base

    torch.manual_seed(4)
    n, w, margin = 2, 30, 5
    labels = torch.full((n, w), H)
    labels[:, 12:18] = A
    labels[:, 18:24] = D
    out = {"logits": torch.randn(n, w, 3), "ball": torch.randn(n, w, 2)}
    batch = {
        "tokens": torch.zeros(n, w, 1),
        "labels": labels,
        "ball_ok": torch.zeros(n, w, dtype=torch.bool),
        "ball": torch.zeros(n, w, 2),
    }
    weights = torch.tensor([1.0, 1.2, 0.8])
    plain = loss_terms(out, batch, weights, margin, 0.3)
    state = torch.get_rng_state()
    off = loss_terms(out, batch, weights, margin, 0.3, scrf_weight=0.0)
    assert torch.equal(torch.get_rng_state(), state)
    assert set(plain) == set(off)
    for key in plain:
        assert torch.equal(plain[key], off[key])
    headed = {**out, "switch": torch.randn(n, w)}
    still = loss_terms(headed, batch, weights, margin, 0.3, scrf_weight=0.0)
    assert torch.equal(still["loss"], plain["loss"]) and "scrf" not in still


def test_t9_divisor_is_every_interior_frame_and_all_dead_is_zero():
    torch.manual_seed(2)
    logits = torch.randn(1, 20, 3)
    switch = torch.randn(1, 20)
    labels = torch.randint(0, 3, (1, 20))
    one = scrf_nll(logits, switch, labels)
    dead_logits = torch.randn(1, 20, 3, requires_grad=True)
    dead_switch = torch.randn(1, 20, requires_grad=True)
    dead_labels = torch.full((1, 20), D)
    zero = scrf_nll(dead_logits, dead_switch, dead_labels)
    assert float(zero.detach()) == 0.0
    zero.backward()
    assert torch.equal(dead_logits.grad, torch.zeros_like(dead_logits))
    assert torch.equal(dead_switch.grad, torch.zeros_like(dead_switch))
    both = scrf_nll(
        torch.cat((logits, dead_logits.detach()), 0),
        torch.cat((switch, dead_switch.detach()), 0),
        torch.cat((labels, dead_labels), 0),
    )
    assert torch.allclose(both, one / 2, rtol=0.0, atol=1e-8)


def test_t10_short_spells_are_free_only_from_the_full_match():
    frames = 45
    labels = torch.cat((torch.full((10,), D), torch.full((25,), A), torch.full((10,), D)))[None]
    short = torch.tensor(short_spell_mask(labels[0].numpy()))[None]
    logits = torch.randn(1, frames, 3)
    switch = torch.randn(1, frames)
    assert float(scrf_nll(logits, switch, labels, short=short).detach()) == 0.0
    plain = scrf_nll(logits, switch, labels, short=None)
    zeros = scrf_nll(logits, switch, labels, short=torch.zeros(1, frames, dtype=torch.bool))
    long = torch.cat((torch.full((10,), D), torch.full((40,), H), torch.full((10,), D)))
    assert not short_spell_mask(long.numpy()).any()
    assert torch.equal(plain, zeros)

    full = np.full(80, D, dtype=np.int64)
    full[:40] = A
    assert not short_spell_mask(full)[:40].any()
    window = full[20:40]
    assert short_spell_mask(window).all()
    sliced = torch.tensor(short_spell_mask(full)[20:40])[None]
    _allow_z, allow_c = constraint_masks(torch.tensor(window)[None], sliced)
    free = allow_c[0, :, H] & allow_c[0, :, A]
    assert not bool(free.any())
    local = torch.tensor(short_spell_mask(window))[None]
    _z, allow_local = constraint_masks(torch.tensor(window)[None], local)
    assert bool((allow_local[0, :, H] & allow_local[0, :, A]).any())

    mixed = np.array([H, H, A, A, D, D, H, A, A], dtype=np.int64)
    mask = short_spell_mask(mixed)
    swapped = np.where(mixed == H, A, np.where(mixed == A, H, mixed))
    assert np.array_equal(mask, short_spell_mask(swapped))
    base = torch.tensor(mixed)[None]
    free_base = constraint_masks(base, torch.tensor(mask)[None])[1]
    free_swap = constraint_masks(torch.tensor(swapped)[None], torch.tensor(mask)[None])[1]
    both = free_base[0, :, H] & free_base[0, :, A]
    both_s = free_swap[0, :, H] & free_swap[0, :, A]
    assert torch.equal(both, both_s)
    assert SHORT_SPELL == 38


def test_t11_tables_come_from_the_decoder_and_beta_is_zero():
    nb = 10
    lp, ls, cap, log_t = decoder_tables(nb, DT, DMAX, SELF_PROB)
    for k, dmax in enumerate(DMAX):
        pmf = _log_pmf(DEFAULT_MODEL, k, dmax, DT)
        surv = _log_surv(DEFAULT_MODEL, k, max(nb, dmax), DT)
        assert np.allclose(lp[k, 1 : nb + 1], pmf[1 : nb + 1])
        assert np.allclose(ls[k], surv[: nb + 1])
        assert np.allclose(cap[k, 1 : nb + 1], surv[1 : nb + 1])
    assert np.allclose(log_t, _transition_logs(DEFAULT_MODEL, SELF_PROB))
    assert (BLOCK, DT, SELF_PROB) == (5, 0.2, 0.02)

    tables = decoder_tables(2)
    e = torch.tensor([[[0.3, -0.2, 0.1], [-0.4, 0.5, 0.0]]], dtype=torch.float64)
    bonus = torch.zeros(1, 3, dtype=torch.float64)
    only_h = torch.zeros(1, 2, 3, dtype=torch.bool)
    only_h[..., H] = True
    z = semi_markov(e, bonus, *[torch.tensor(x) for x in tables], allow=only_h)
    lp, ls, cap, log_t = tables
    # The opening segment, including one that covers the whole period, uses uncapped survival.
    # A segment that ends the period and does not open it uses survival capped at dmax. Beta is 0.
    one = e[0, :, H].sum() + math.log(1.0 / 3.0) + ls[H, 2]
    two = e[0, 0, H] + math.log(1.0 / 3.0) + ls[H, 1] + log_t[H, H] + e[0, 1, H] + cap[H, 1]
    manual = torch.logsumexp(torch.tensor([one, two], dtype=torch.float64), 0)
    assert abs(float(z[0]) - float(manual)) <= 1e-10
    again = semi_markov(e, bonus, *[torch.tensor(x) for x in tables], beta=0.0, allow=only_h)
    assert float(again[0]) == float(z[0])
    assert MASK == -1e5


def test_t12_z_and_zc_share_one_batch():
    torch.manual_seed(3)
    logits = torch.randn(4, 20, 3, dtype=torch.float64, requires_grad=True)
    switch = torch.randn(4, 20, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 3, (4, 20))
    labels[:, :10] = H
    labels[:, 10:] = A
    loss = scrf_nll(logits, switch, labels)
    allow_z, allow_c = constraint_masks(labels)
    # Rebuild the emissions the loss uses and compare one call against two.
    dead = labels.eq(D)
    odds = ste_clamp(logits[..., H] - logits[..., A], TEAM_CLIP)
    team = 0.3 * odds * (~dead).to(odds.dtype)
    emissions = torch.stack((team, -team, torch.zeros_like(team)), -1).reshape(4, 4, 5, 3).sum(2)
    clipped = ste_clamp(switch, 6.9)
    bonus = torch.cat((clipped.new_zeros(4, 1), 2.0 * clipped.reshape(4, 4, 5).amax(-1)), 1)
    tables = [torch.tensor(x, dtype=torch.float64) for x in decoder_tables(4)]
    one = semi_markov(
        torch.cat((emissions, emissions), 0),
        torch.cat((bonus, bonus), 0),
        *tables,
        allow=torch.cat((allow_z, allow_c), 0),
    )
    z = semi_markov(emissions, bonus, *tables, allow=allow_z)
    zc = semi_markov(emissions, bonus, *tables, allow=allow_c)
    assert torch.allclose(one[:4], z, rtol=0.0, atol=0.0)
    assert torch.allclose(one[4:], zc, rtol=0.0, atol=0.0)
    assert torch.isfinite(loss)
