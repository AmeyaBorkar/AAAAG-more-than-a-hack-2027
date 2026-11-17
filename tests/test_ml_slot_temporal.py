"""The per-slot temporal branch (#86, roadmap B): the slot-index columns, `SlotTemporal`, its
numpy port and the partial init.

The token tests are numpy only and run in the repository environment. The rest skip without
torch. Everything is synthetic: hand-made tracking or `aaaag.synthetic` matches.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period, Tracking
from aaaag.ml.geometry import attack_sign, symmetric
from aaaag.ml.np_infer import NumpyTeamNet, parameter_layout, receptive_half, slot_links_np
from aaaag.ml.spec import (
    SLOT_RUN_MOD,
    Col,
    TokenSpec,
    bracket_columns,
    contact_columns,
    slot_index_columns,
    slot_input_columns,
)
from aaaag.ml.windows import build_match_tokens

SPEC = TokenSpec()
SLOT_SPEC = TokenSpec(slot_index=True)
RICH = TokenSpec(control_memory=True, contact_inputs=True, gap_bracket=True)
RICH_SLOT = replace(RICH, slot_index=True)
FPS = 25
HALF_L = 52.5
TINY = {"d_model": 16, "n_heads": 2, "n_blocks": 1, "dilations": (1, 2, 4), "dropout": 0.0}
SLOT = {"enabled": True, "d": 8, "kernel": 5, "dilations": (1, 2, 4, 8)}


def _tracking(xy: np.ndarray, teams: list[int], periods: tuple[Period, ...]) -> Tracking:
    n = xy.shape[0]
    meta = MatchMeta("t", "test", n, FPS, 105.0, 68.0, periods)
    ball = np.zeros((n, 2), dtype=np.float32)
    return Tracking(meta, xy.astype(np.float32), np.array(teams, dtype=np.int8), ball)


# -- tokens ---------------------------------------------------------------------------------------


def test_the_flag_off_writes_the_old_spec_dict_and_the_same_tokens(small_bundle):
    assert "slot_index" not in SPEC.to_dict() and TokenSpec.from_dict(SPEC.to_dict()) == SPEC
    assert SLOT_SPEC.to_dict()["slot_index"] is True
    assert TokenSpec.from_dict(SLOT_SPEC.to_dict()) == SLOT_SPEC
    for off, on in ((SPEC, SLOT_SPEC), (RICH, RICH_SLOT)):
        assert on.n_features == off.n_features + 2 and on.model_features == off.n_features
        assert slot_index_columns(on) == (off.n_features, off.n_features + 1)
        plain = build_match_tokens(small_bundle.tracking, off)
        tagged = build_match_tokens(small_bundle.tracking, on)
        assert tagged.shape == (*plain.shape[:2], on.n_features)
        assert np.array_equal(tagged[..., : off.n_features], plain)  # every old column, bitwise
    assert bracket_columns(RICH_SLOT) == bracket_columns(RICH)
    with pytest.raises(ValueError, match="slot_index"):
        slot_index_columns(SPEC)


def test_slot_ids_name_the_argsort_choice_and_runs_split_at_gaps_and_periods():
    n = 40
    xs = np.array([-10.0, -20.0, -30.0, 12.0, 25.0], dtype=np.float32)  # three home, two away
    xy = np.zeros((n, len(xs), 2), dtype=np.float32)
    xy[..., 0] = xs
    xy[5:8, 1] = np.nan  # home slot 1 unobserved for three frames: a new fragment after
    xy[33:, 4] = np.nan  # away slot 4 leaves for good
    tracking = _tracking(xy, [HOME, HOME, HOME, AWAY, AWAY], (Period(1, 0, 20), Period(2, 20, n)))
    spec = TokenSpec(tokens_per_team=2, slot_index=True)
    tokens = build_match_tokens(tracking, spec)
    slot_col, run_col = slot_index_columns(spec)
    sign = attack_sign(tracking)
    valid = tokens[..., Col.VALID] > 0.5
    slot = tokens[..., slot_col].astype(int) - 1
    for t, i in zip(*np.nonzero(valid[:, :4]), strict=True):  # the token holds that slot's x
        assert tokens[t, i, Col.X] == np.float32(sign[t] * xy[t, slot[t, i], 0] / HALF_L)
    # the two nearest per team: home picks slots 0 and 1, or 0 and 2 while 1 is missing
    assert (slot[:5, :2] == [0, 1]).all() and (slot[5:8, :2] == [0, 2]).all()
    assert (slot[8:, :2] == [0, 1]).all() and (slot[:33, 2:4] == [3, 4]).all()
    run = tokens[..., run_col]
    assert (run[:20, 0] == 1).all() and (run[20:, 0] == 2).all()  # a new period is a new run
    assert (run[:5, 1] == 1).all() and (run[8:20, 1] == 2).all() and (run[20:, 1] == 3).all()
    assert (run[5:8, 1] == 1).all()  # slot 2 stands in for those frames: its own first run
    # invalid tokens and the ball carry no identity
    assert not valid[33:, 3].any() and (tokens[33:, 3, [slot_col, run_col]] == 0).all()
    assert (tokens[:, spec.ball_index, [slot_col, run_col]] == 0).all()


def test_run_ids_wrap_and_stay_exact_in_float16():
    n = 2 * (SLOT_RUN_MOD + 20)
    xy = np.zeros((n, 2, 2), dtype=np.float32)
    xy[:, 0, 0] = -5.0
    xy[:, 1, 0] = 5.0
    xy[1::2, 0] = np.nan  # seen every other frame: a new run each time
    tokens = build_match_tokens(
        _tracking(xy, [HOME, AWAY], (Period(1, 0, n),)),
        TokenSpec(tokens_per_team=1, slot_index=True),
    )
    run = tokens[0::2, 0, -1]
    assert np.array_equal(run, (np.arange(len(run)) % SLOT_RUN_MOD + 1).astype(np.float32))
    assert run.max() == SLOT_RUN_MOD and np.array_equal(run.astype(np.float16), run)
    assert (tokens[1::2, 0, -2:] == 0).all() and (tokens[:, 1, -1] == 1).all()


def test_the_symmetries_carry_the_identifiers_with_their_tokens(small_bundle):
    tokens = build_match_tokens(small_bundle.tracking, SLOT_SPEC)[500:700]
    k = SLOT_SPEC.tokens_per_team
    cols = list(slot_index_columns(SLOT_SPEC))
    swapped = symmetric(tokens, SLOT_SPEC, swap=True)
    assert np.array_equal(swapped[:, :k][..., cols], tokens[:, k : 2 * k][..., cols])
    assert np.array_equal(swapped[:, k : 2 * k][..., cols], tokens[:, :k][..., cols])
    flipped = symmetric(tokens, SLOT_SPEC, flip_y=True)
    assert np.array_equal(flipped[..., cols], tokens[..., cols])


def test_input_names_resolve_against_the_spec():
    assert slot_input_columns(["x", "y", "vx", "vy", "speed"], SPEC) == (0, 1, 2, 3, 4)
    assert slot_input_columns(["dist", 7, "accel"], RICH) == (Col.DIST, 7, contact_columns(RICH)[3])
    with pytest.raises(ValueError, match="contact_inputs"):
        slot_input_columns(["accel"], SPEC)
    with pytest.raises(ValueError, match="not one of"):
        slot_input_columns(["jerk"], SPEC)


def test_numpy_links_follow_slot_and_run_inside_the_window():
    # one window, four frames, two tokens per team; the away block is empty
    slot = np.zeros((1, 4, 4))
    run = np.zeros((1, 4, 4))
    valid = np.zeros((1, 4, 4), dtype=bool)
    slot[0, :, :2] = [[1, 2], [2, 1], [1, 0], [1, 2]]
    run[0, :, :2] = [[1, 1], [1, 1], [1, 0], [2, 1]]
    valid[0, :, :2] = [[True, True], [True, True], [True, False], [True, True]]
    links = slot_links_np(slot, run, valid, 2, (-1, 0, 1, 2))
    own = np.arange(16).reshape(1, 4, 4)
    for row, found in links.values():  # no partner: the token's own row, read as zero
        assert np.array_equal(row[~found], own[~found])
    partner = {o: np.where(found, row, -1)[0] for o, (row, found) in links.items()}
    assert partner[1][0].tolist() == [5, 4, -1, -1]  # frame 1 holds slot 1 in token 1
    assert partner[1][1].tolist() == [-1, 8, -1, -1]  # slot 2 is not seen at frame 2
    assert partner[1][2].tolist() == [-1] * 4  # frame 3: slot 1 began a new run
    assert partner[1][3].tolist() == [-1] * 4  # past the window
    assert partner[-1][0].tolist() == [-1] * 4 and partner[-1][1].tolist() == [1, 0, -1, -1]
    assert partner[2][0].tolist() == [8, -1, -1, -1]  # slot 1, two frames on, same run
    assert partner[2][1].tolist() == [13, -1, -1, -1]  # slot 2 again; slot 1 there is run 2
    assert partner[0][2].tolist() == [8, -1, -1, -1]
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import slot_links

    got = slot_links(
        torch.from_numpy(slot), torch.from_numpy(run), torch.from_numpy(valid), 2, (-1, 0, 1, 2)
    )
    for offset, (row, found) in links.items():
        assert np.array_equal(got[offset][0].numpy(), row)
        assert np.array_equal(got[offset][1].numpy(), found)


# -- torch ----------------------------------------------------------------------------------------


def _configs(**changes):
    from aaaag.ml.model import ModelConfig, SlotTemporalConfig

    parent = ModelConfig(**TINY, **changes)
    return parent, replace(parent, slot_temporal=SlotTemporalConfig(**SLOT))


def _pair(seed: int = 0, **changes):
    """A parent TeamNet and the same network with the slot branch, the parent's weights loaded."""
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import TeamNet

    parent_cfg, child_cfg = _configs(**changes)
    torch.manual_seed(seed)
    parent = TeamNet(parent_cfg).eval()
    child = TeamNet(child_cfg).eval()
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("slot_temporal.") for k in missing)
    return parent, child


def _wake(model, seed: int = 9, scale: float = 0.3):
    """Random slot-branch weights, the output projection included, so the branch matters."""
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in model.slot_temporal.named_parameters():
            if name.endswith("norm.weight"):
                value.copy_(1.0 + 0.1 * torch.randn(value.shape, generator=gen))
            else:
                value.copy_(scale * torch.randn(value.shape, generator=gen))
    return model


def _slot_window(bundle, lo: int = 1000, n: int = 200) -> np.ndarray:
    return build_match_tokens(bundle.tracking, SLOT_SPEC)[lo : lo + n].copy()


def _arrays(model) -> dict[str, np.ndarray]:
    return {name: t.detach().double().numpy() for name, t in model.state_dict().items()}


def test_step_zero_logits_equal_the_parent_bit_for_bit(small_bundle):
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import TeamNet

    parent, child = _pair(possessor_head=True, switch_head=True, touch_head=True)
    off = torch.from_numpy(build_match_tokens(small_bundle.tracking, SPEC)[900:1150])[None]
    on = torch.from_numpy(_slot_window(small_bundle, 900, 250))[None]
    with torch.no_grad():
        want, got = parent(off), child(on)
        branch = child.slot_temporal(on, child.cfg.n_features)
    assert set(want) == set(got) and all(torch.equal(want[key], got[key]) for key in want)
    assert torch.count_nonzero(branch) == 0
    assert child.receptive_half == parent.receptive_half + 30 == child.cfg.receptive_half
    # Built from one seed, every layer the parent has starts from the same weights: the branch
    # is created last and does not move the draws of the others.
    torch.manual_seed(4)
    a = TeamNet(parent.cfg).state_dict()
    torch.manual_seed(4)
    b = TeamNet(child.cfg).state_dict()
    assert all(torch.equal(a[name], b[name]) for name in a)
    with pytest.raises(ValueError, match="slot_index"):
        child(off)


def test_the_flag_off_builds_todays_network():
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import ModelConfig, SlotTemporalConfig, TeamNet

    cfg = ModelConfig(**TINY)
    off = replace(cfg, slot_temporal=SlotTemporalConfig(enabled=False, d=48))
    assert "slot_temporal" not in cfg.to_dict() and off.to_dict() == cfg.to_dict()
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg and off.receptive_half == cfg.receptive_half
    torch.manual_seed(0)
    a = TeamNet(cfg).state_dict()
    torch.manual_seed(0)
    b = TeamNet(off).state_dict()
    assert list(a) == list(b) and all(torch.equal(a[name], b[name]) for name in a)
    _, on = _configs()
    assert ModelConfig.from_dict(on.to_dict()) == on
    assert on.to_dict()["slot_temporal"]["dilations"] == [1, 2, 4, 8]
    with pytest.raises(ValueError, match="kernel"):
        SlotTemporalConfig(enabled=True, kernel=4)
    with pytest.raises(ValueError, match="unique"):
        SlotTemporalConfig(enabled=True, dilations=(1, 300))
    with pytest.raises(ValueError, match="n_features"):
        ModelConfig(slot_temporal={"enabled": True, "inputs": [0, 14]})
    with pytest.raises(ValueError, match="column indices"):
        SlotTemporalConfig(enabled=True, inputs=("x",))


def test_a_permutation_inside_a_team_and_relabelled_slots_change_nothing(small_bundle):
    torch = pytest.importorskip("torch")
    _, child = _pair(1)
    model = _wake(child).double()  # float64, so a reordered sum stays inside 1e-10
    tokens = _slot_window(small_bundle).astype(np.float64)
    rng = np.random.default_rng(0)
    k = SLOT_SPEC.tokens_per_team
    order = np.stack(
        [np.concatenate([rng.permutation(k), k + rng.permutation(k), [2 * k]]) for _ in tokens]
    )
    shuffled = np.take_along_axis(tokens, order[..., None], axis=1)
    relabelled = tokens.copy()
    slot_col = slot_index_columns(SLOT_SPEC)[0]
    seen = relabelled[..., slot_col] > 0
    relabelled[..., slot_col][seen] = 1000.0 - relabelled[..., slot_col][seen]  # still distinct
    with torch.no_grad():
        x = torch.from_numpy(tokens)[None]
        base, branch = model(x)["logits"], model.slot_temporal(x, model.cfg.n_features)[0]
        y = torch.from_numpy(shuffled)[None]
        moved = model.slot_temporal(y, model.cfg.n_features)[0]
        assert torch.allclose(model(y)["logits"], base, rtol=0.0, atol=1e-10)
        z = torch.from_numpy(relabelled)[None]
        assert torch.equal(model.slot_temporal(z, model.cfg.n_features)[0], branch)
    assert torch.count_nonzero(branch) > 0
    back = np.take_along_axis(branch.numpy(), order[:, : 2 * k, None], axis=1)
    assert np.allclose(moved.numpy(), back, rtol=0.0, atol=1e-10)


def test_a_chunked_pass_equals_one_pass_and_the_reach_is_the_config(small_bundle):
    torch = pytest.importorskip("torch")
    from aaaag.ml.infer import logits_from_tokens

    _, child = _pair(2)
    model = _wake(child, seed=3)
    tokens = build_match_tokens(small_bundle.tracking, SLOT_SPEC)
    periods = small_bundle.meta.periods
    whole = logits_from_tokens(model, SLOT_SPEC, tokens, periods, chunk=100_000)
    chunked = logits_from_tokens(model, SLOT_SPEC, tokens, periods, chunk=700)
    played = small_bundle.meta.played_mask()
    assert not np.isnan(whole[played]).any()
    assert np.allclose(whole[played], chunked[played], atol=1e-5)
    window = tokens[1000:1300].copy()
    changed = window.copy()
    changed[0, : 2 * SLOT_SPEC.tokens_per_team, : SLOT_SPEC.model_features] += 0.5  # frame 0
    with torch.no_grad():
        a = model(torch.from_numpy(window)[None])["logits"][0]
        b = model(torch.from_numpy(changed)[None])["logits"][0]
    reach = model.receptive_half
    assert (a[: reach + 1] != b[: reach + 1]).any() and torch.equal(a[reach + 1 :], b[reach + 1 :])


def test_a_fragment_edge_and_an_unseen_frame_stop_the_branch(small_bundle):
    torch = pytest.importorskip("torch")
    _, child = _pair(3)
    model = _wake(child, seed=5)
    tokens = _slot_window(small_bundle, 1500, 120)
    n2 = 2 * SLOT_SPEC.tokens_per_team
    slot_col, run_col = slot_index_columns(SLOT_SPEC)
    t0, i0 = 60, 0
    target = tokens[t0, i0, slot_col]
    assert target > 0
    mine = (tokens[:, :n2, slot_col] == target) & (tokens[:, :n2, Col.VALID] > 0.5)
    later = mine.copy()
    later[: t0 + 1] = False
    before = mine & ~later
    assert later[t0 + 1 : t0 + 20].sum() > 10 and before[t0 - 10 :].sum() > 5

    def branch(array):
        with torch.no_grad():
            x = torch.from_numpy(array)[None]
            return model.slot_temporal(x, model.cfg.n_features)[0].numpy()

    def nudge(array):
        out = array.copy()
        players = out[:, :n2]  # a view: the edits land in `out`
        players[..., Col.X][later] += 0.3
        players[..., Col.VX][later] -= 0.2
        return out

    split = tokens.copy()
    split[:, :n2, run_col][later] += 1.0  # the slot's later frames are another fragment run
    a, b = branch(split), branch(nudge(split))
    assert np.array_equal(a[before], b[before])  # nothing crosses the fragment edge
    assert not np.array_equal(a[later], b[later])
    assert np.array_equal(a[~mine], b[~mine])  # nor reaches another slot
    # the same nudge without the edge does reach back
    assert not np.array_equal(branch(tokens)[t0, i0], branch(nudge(tokens))[t0, i0])
    # an unseen token is read by nobody, whatever it holds
    hidden = tokens.copy()
    hidden[t0 - 1, :n2, Col.VALID] = 0.0
    junk = hidden.copy()
    junk[t0 - 1, :n2, : SLOT_SPEC.model_features] = 7.0
    junk[t0 - 1, :n2, Col.VALID] = 0.0
    c, d = branch(hidden), branch(junk)
    assert np.array_equal(c, d) and (c[t0 - 1] == 0).all()


def test_numpy_matches_torch_in_float64_and_step_zero_is_exact(small_bundle):
    torch = pytest.importorskip("torch")
    parent, child = _pair(4, possessor_head=True, switch_head=True)
    layout = [name for name, _shape in parameter_layout(child.cfg.to_dict())]
    assert layout == list(child.state_dict())
    assert receptive_half(child.cfg.to_dict()) == child.receptive_half
    window = _slot_window(small_bundle, 2000, 160)
    k = SLOT_SPEC.tokens_per_team
    window[5, k : 2 * k] = 0  # an empty team
    window[8, 2, : SLOT_SPEC.model_features] = 3.0  # junk on an invalid token
    window[8, 2, Col.VALID] = 0.0
    batch = np.stack([window, window[::-1].copy()])
    plain = np.ascontiguousarray(batch[..., : SLOT_SPEC.model_features])

    def numpy_net(model, spec):
        return NumpyTeamNet(model.cfg.to_dict(), spec, _arrays(model))

    zero_child, zero_parent = numpy_net(child, SLOT_SPEC), numpy_net(parent, SPEC)
    got, want = zero_child.forward(batch), zero_parent.forward(plain)
    assert set(got) == set(want) and all(np.array_equal(got[key], want[key]) for key in want)
    awake = _wake(child, seed=6).double()
    net = numpy_net(awake, SLOT_SPEC)
    with torch.no_grad():
        ref = awake(torch.from_numpy(batch).double())
    out = net.forward(batch)
    for key in ("logits", "ball", "switch"):
        assert np.max(np.abs(out[key] - ref[key].numpy())) <= 1e-9
    assert not np.allclose(out["logits"], zero_parent.forward(plain)["logits"], atol=1e-3)
    with pytest.raises(ValueError, match="slot_index"):
        NumpyTeamNet(awake.cfg.to_dict(), SPEC, _arrays(awake))


def test_export_carries_the_branch_and_numpy_inference_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint, save_checkpoint
    from aaaag.ml.export import export_file
    from aaaag.ml.infer import logits_from_tokens, predict_logits
    from aaaag.ml.np_infer import load_numpy_weights, logits_from_tokens_np, predict_logits_np
    from aaaag.synthetic import SyntheticConfig, generate

    short = generate(seed=11, config=SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.1))
    _, child = _pair(5, switch_head=True)
    model = _wake(child, seed=8)
    pt, js = tmp_path / "slot.pt", tmp_path / "slot.json"
    save_checkpoint(pt, model, SLOT_SPEC, {})
    export_file(pt, js, encoding="base64")
    loaded = load_checkpoint(pt)
    assert loaded.spec == SLOT_SPEC and loaded.model.cfg == model.cfg
    net = load_numpy_weights(js)
    assert net.slot_temporal and net.receptive_half == model.receptive_half
    tokens = build_match_tokens(short.tracking, SLOT_SPEC)
    periods = short.meta.periods
    with torch.inference_mode():
        ref = logits_from_tokens(model, SLOT_SPEC, tokens, periods, chunk=1500, views=4)
    got = logits_from_tokens_np(net, SLOT_SPEC, tokens, periods, chunk=1500, views=4)
    mask = ~np.isnan(ref[:, 0])
    assert np.array_equal(np.isnan(got[:, 0]), np.isnan(ref[:, 0]))
    assert np.max(np.abs(got[mask] - ref[mask])) <= 1e-4
    a = predict_logits([loaded], short.tracking, chunk=1500)
    b = predict_logits_np([js], short.tracking, chunk=1500)
    assert np.array_equal(a.opinion, b.opinion) and b.switch is not None
    assert np.max(np.abs(a.logits[a.opinion] - b.logits[b.opinion])) <= 1e-4
    whole = predict_logits_np([js], short.tracking, chunk=100_000)
    assert np.max(np.abs(whole.logits[b.opinion] - b.logits[b.opinion])) <= 1e-10


# -- partial init and the run config ---------------------------------------------------------------


def test_init_from_partial_loads_a_parent_without_the_branch_and_refuses_anything_else(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import save_checkpoint
    from aaaag.ml.model import ModelConfig, TeamNet
    from aaaag.ml.train import RunConfig, TrainConfig, _load_init_weights

    parent_cfg, child_cfg = _configs(switch_head=True)
    torch.manual_seed(0)
    parent = TeamNet(parent_cfg)
    path = tmp_path / "parent.pt"
    save_checkpoint(path, parent, SPEC, {})
    run = RunConfig(
        token=SLOT_SPEC,
        model=child_cfg,
        train=TrainConfig(init_from=str(path), init_from_partial=True),
    )
    torch.manual_seed(1)
    child = TeamNet(child_cfg)
    fresh = {k: v.clone() for k, v in child.state_dict().items()}
    kept = _load_init_weights(child, str(path), run, partial=True)
    state = child.state_dict()
    assert kept and all(name.startswith("slot_temporal.") for name in kept)
    assert all(torch.equal(state[name], fresh[name]) for name in kept)
    assert all(torch.equal(state[name], value) for name, value in parent.state_dict().items())
    with pytest.raises(ValueError, match="does not match"):  # the strict load still refuses
        _load_init_weights(TeamNet(child_cfg), str(path), run)

    def refuses(checkpoint_model, spec, match, state_edit=None):
        bad = tmp_path / "bad.pt"
        save_checkpoint(bad, checkpoint_model, spec, {})
        if state_edit is not None:
            payload = torch.load(bad, weights_only=True)
            state_edit(payload["state_dict"])
            torch.save(payload, bad)
        with pytest.raises(ValueError, match=match):
            _load_init_weights(TeamNet(child_cfg), str(bad), run, partial=True)

    refuses(TeamNet(replace(parent_cfg, d_model=32)), SPEC, "does not match")
    refuses(TeamNet(replace(parent_cfg, switch_head=False)), SPEC, "does not match")
    refuses(parent, replace(SPEC, velocity_half_s=0.4), "token spec")
    refuses(parent, SPEC, "does not have", lambda s: s.update(extra=torch.zeros(1)))
    refuses(parent, SPEC, "missing tensors outside", lambda s: s.pop("head.bias"))
    refuses(parent, SPEC, "shape or dtype", lambda s: s.update({"head.bias": torch.zeros(4)}))
    # a parent that already has the branch loads whole
    twin = TeamNet(child_cfg)
    save_checkpoint(path, twin, SLOT_SPEC, {})
    assert _load_init_weights(TeamNet(child_cfg), str(path), run, partial=True) == []
    assert ModelConfig.from_dict(child_cfg.to_dict()) == child_cfg


def test_the_run_config_pairs_the_flags_resolves_names_and_round_trips(tmp_path):
    pytest.importorskip("torch")
    import yaml

    from aaaag.ml.train import RunConfig, TrainConfig

    parent_cfg, child_cfg = _configs()
    with pytest.raises(ValueError, match="go together"):
        RunConfig(token=SPEC, model=child_cfg)
    with pytest.raises(ValueError, match="go together"):
        RunConfig(token=SLOT_SPEC, model=parent_cfg)
    with pytest.raises(ValueError, match="init_from"):
        TrainConfig(init_from_partial=True)
    assert "init_from_partial" not in RunConfig().to_dict()["train"]
    raw = {
        "token": {"control_memory": True, "contact_inputs": True, "slot_index": True},
        "model": {
            "n_features": 20,
            "slot_temporal": {"enabled": True, "inputs": ["x", "y", "vx", "vy", "speed", "accel"]},
        },
        "train": {"init_from": "parent/fold-{fold}/final.pt", "init_from_partial": True},
    }
    run = RunConfig.from_dict(raw)
    assert run.model.slot_temporal.inputs == (0, 1, 2, 3, 4, 19)
    assert run.model.token_width == run.token.n_features == 22
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(run.to_dict()), encoding="utf-8")
    assert RunConfig.from_yaml(path) == run and run.to_dict()["train"]["init_from_partial"] is True
    with pytest.raises(ValueError, match="model.n_features"):
        RunConfig.from_dict({**raw, "model": {**raw["model"], "n_features": 22}})


def test_a_fine_tune_from_a_partial_parent_trains_the_branch(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import load_checkpoint
    from aaaag.ml.data import DegradeMix, TrainMatch
    from aaaag.ml.train import RunConfig, TrainConfig, train
    from aaaag.synthetic import SyntheticConfig, generate

    bundle = generate(seed=7, config=SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5))
    matches = [TrainMatch.from_bundle(bundle, SPEC)]
    parent_cfg, child_cfg = _configs()
    common = {
        "seed": 3,
        "steps": 3,
        "refresh_steps": 3,
        "batch_size": 2,
        "window": 96,
        "warmup_steps": 1,
        "lr": 3e-3,
        "degrade": DegradeMix(version=None),
        "amp": False,
    }
    quiet = {"device": "cpu", "resume": False, "threads": 1}
    parent = train(
        RunConfig(token=SPEC, model=parent_cfg, train=TrainConfig(**common)),
        matches,
        tmp_path / "parent",
        log=lambda _line: None,
        **quiet,
    )
    lines: list[str] = []
    child_run = RunConfig(
        token=SLOT_SPEC,
        model=child_cfg,
        train=TrainConfig(**common, init_from=str(parent), init_from_partial=True),
    )
    final = train(child_run, matches, tmp_path / "child", log=lines.append, **quiet)
    assert any("partial: 30 new tensors" in line for line in lines)  # 2 + 4 * 6 + 2 + 2
    loaded = load_checkpoint(final)
    assert loaded.spec == SLOT_SPEC and loaded.model.cfg == child_cfg
    out_proj = loaded.model.slot_temporal.out_proj.weight
    assert torch.count_nonzero(out_proj) > 0  # the zero projection learned
