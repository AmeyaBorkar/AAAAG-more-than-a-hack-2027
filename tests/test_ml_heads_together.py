"""The possessor head and the switch head on one network, as the stacked candidate (E20) uses them.

Each head was built and tested alone (test_ml_possessor.py, test_ml_switch_head.py). Here both are on
at once: the numpy engine must match torch on every output, both loss terms must add up, and the
stacked run config must load with both heads and the contact columns. Torch tests skip without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.paths import CONFIGS_DIR


def _both_heads_config():
    from aaaag.ml.model import ModelConfig

    return ModelConfig(
        tokens_per_team=2,
        d_model=8,
        n_heads=2,
        n_blocks=1,
        ffn_mult=2,
        dilations=(1, 2),
        kernel=3,
        dropout=0.0,
        possessor_head=True,
        switch_head=True,
    )


def test_numpy_engine_matches_torch_with_both_heads():
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import TeamNet
    from aaaag.ml.np_infer import NumpyTeamNet, parameter_layout
    from aaaag.ml.spec import Col, TokenSpec

    torch.manual_seed(0)
    cfg = _both_heads_config()
    module = TeamNet(cfg).eval().double()
    assert [name for name, _shape in parameter_layout(cfg.to_dict())] == list(module.state_dict())
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in module.state_dict().items()}
    net = NumpyTeamNet(cfg.to_dict(), TokenSpec(tokens_per_team=2), arrays)
    batch = torch.randn(2, 9, cfg.n_tokens, cfg.n_features)
    batch[..., Col.VALID] = 1.0
    batch[0, :, 1, Col.VALID] = 0.0  # one invalid home token, so the possessor mask is exercised
    with torch.inference_mode():
        ref = module(batch.double())
    got = net.forward(batch.numpy())
    assert set(ref) == {"logits", "ball", "possessor", "switch"}
    for key in ("logits", "ball", "switch"):
        assert np.max(np.abs(got[key] - ref[key].numpy())) <= 1e-6


def test_both_loss_terms_add_up():
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import TeamNet
    from aaaag.ml.train import loss_terms

    torch.manual_seed(1)
    cfg = _both_heads_config()
    module = TeamNet(cfg)
    b, w, margin = 2, 40, 3
    tokens = torch.randn(b, w, cfg.n_tokens, cfg.n_features)
    tokens[..., 7] = 1.0  # Col.VALID
    labels = torch.zeros(b, w, dtype=torch.long)
    labels[:, 20:] = 1  # one direct H -> A change per window
    batch = {
        "tokens": tokens,
        "labels": labels,
        "ball": torch.zeros(b, w, 2),
        "ball_ok": torch.zeros(b, w, dtype=torch.bool),
        "possessor": torch.where(labels == 0, 0, 2).long(),  # a home slot, then an away slot
    }
    out = module(tokens)
    weights = torch.ones(3)
    plain = loss_terms(out, batch, weights, margin, 0.2)
    both = loss_terms(out, batch, weights, margin, 0.2, possessor_weight=0.2, switch_weight=0.2)
    assert {"pos", "sw"} <= set(both)
    expected = plain["loss"] + 0.2 * both["pos"] + 0.2 * both["sw"]
    assert torch.isfinite(both["loss"]) and torch.allclose(both["loss"], expected, atol=1e-6)


def test_stacked_run_config_loads_with_both_heads_and_the_contact_columns():
    pytest.importorskip("torch")
    from aaaag.ml.train import RunConfig

    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_possessor_switch.yaml")
    assert run.token.control_memory and run.token.contact_inputs
    assert run.model.n_features == run.token.n_features == 20
    assert run.model.possessor_head and run.model.switch_head
    assert run.train.possessor_weight == 0.2 and run.train.switch_weight == 0.2
