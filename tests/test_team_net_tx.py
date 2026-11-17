"""TeamNetTx (E81, #86): attention over time in place of TeamNet's dilated convolutions."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aaaag.contracts import Period  # noqa: E402
from aaaag.ml.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from aaaag.ml.infer import logits_from_tokens  # noqa: E402
from aaaag.ml.model import ModelConfig, build_model, get_architecture  # noqa: E402
from aaaag.ml.spec import N_FEATURES, Col, TokenSpec  # noqa: E402
from aaaag.paths import CONFIGS_DIR  # noqa: E402

HEADS = {"possessor_head": True, "switch_head": True, "touch_head": True}


def _tokens(b: int, w: int, k: int = 11, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, w, 2 * k + 1, N_FEATURES, generator=g)
    x[..., Col.VALID] = 1.0
    return x


def _tx(radius: int = 8, blocks: int = 2, **heads: bool):
    config_cls, _ = get_architecture("team_net_tx")
    torch.manual_seed(0)
    return build_model("team_net_tx", config_cls(t_radius=radius, t_blocks=blocks, **heads)).eval()


def test_the_receptive_field_is_exactly_blocks_times_radius():
    model = _tx(radius=8, blocks=2)
    assert model.receptive_half == 16
    x = _tokens(1, 101)
    t = 50
    with torch.no_grad():
        base = model(x)["logits"][0, t]
        far = x.clone()
        far[0, : t - 16] += 5.0  # every frame more than 16 before t
        far[0, t + 17 :] += 5.0  # every frame more than 16 after t
        torch.testing.assert_close(model(far)["logits"][0, t], base)
        near = x.clone()
        near[0, t + 16] += 5.0  # the edge of the field
        assert not torch.allclose(model(near)["logits"][0, t], base)


def test_outputs_match_teamnet_keys_and_shapes():
    tx = _tx(**HEADS)
    torch.manual_seed(0)
    ref = build_model("team_net", ModelConfig(**HEADS)).eval()
    x = _tokens(2, 40)
    with torch.no_grad():
        a, b = tx(x), ref(x)
    assert set(a) == set(b) == {"logits", "ball", "possessor", "switch", "touch"}
    for key in a:
        assert a[key].shape == b[key].shape


def test_the_e81_run_config_loads_with_the_teamnet_heads():
    from aaaag.ml.train import RunConfig

    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_tx_e81.yaml")
    assert run.arch == "team_net_tx"
    assert run.model.receptive_half == 128
    assert run.model.dilations == ()
    assert run.model.switch_head and run.model.possessor_head
    assert run.train.switch_weight > 0 and run.train.possessor_weight > 0
    assert type(run.model).from_dict(run.model.to_dict()) == run.model


def test_dilations_are_refused():
    config_cls, _ = get_architecture("team_net_tx")
    with pytest.raises(ValueError):
        config_cls(dilations=(1, 2))


def test_a_checkpoint_round_trips(tmp_path):
    model = _tx(**HEADS)
    path = save_checkpoint(tmp_path / "m.pt", model, TokenSpec(), {"note": "test"})
    loaded = load_checkpoint(path)
    assert loaded.model.arch == "team_net_tx"
    x = _tokens(1, 30)
    with torch.no_grad():
        torch.testing.assert_close(loaded.model.eval()(x)["logits"], model(x)["logits"])


def test_chunked_inference_equals_one_pass():
    model = _tx(radius=8, blocks=2)
    tokens = _tokens(1, 300, seed=3)[0].numpy().astype(np.float32)
    periods = (Period(1, 0, 300),)
    chunked = logits_from_tokens(model, TokenSpec(), tokens, periods, chunk=64)
    whole = logits_from_tokens(model, TokenSpec(), tokens, periods, chunk=4096)
    np.testing.assert_allclose(chunked, whole, atol=1e-5)
