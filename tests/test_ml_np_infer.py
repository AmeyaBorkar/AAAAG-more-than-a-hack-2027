"""Numpy `team_net` inference: parity with torch, export, chunking, determinism, the producer.

Torch is optional. The parity and `backend="torch"` tests skip without it; the numpy path,
including the producer with torch blocked, runs in the repository environment.
"""

from __future__ import annotations

import base64
import json
import sys
import time

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from aaaag.contracts import Evidence, MatchMeta, Period, Tracking
from aaaag.evidence.possession.learned_team import learned_team
from aaaag.ml.np_infer import (
    dump_numpy_weights,
    load_numpy_weights,
    logits_from_tokens_np,
    parameter_layout,
    predict_logits_np,
)
from aaaag.ml.spec import FEATURE_VERSION, Col, TokenSpec
from aaaag.ml.weights import sha256_file
from aaaag.ml.windows import build_match_tokens

SPEC = TokenSpec()
TINY = {
    "tokens_per_team": 11,
    "n_features": 14,
    "d_model": 16,
    "n_heads": 2,
    "n_blocks": 1,
    "ffn_mult": 2,
    "dilations": [1, 2, 4],
    "kernel": 3,
    "dropout": 0.0,
}
DEFAULT = {
    "tokens_per_team": 11,
    "n_features": 14,
    "d_model": 64,
    "n_heads": 4,
    "n_blocks": 2,
    "ffn_mult": 2,
    "dilations": [1, 2, 4, 8, 16, 32, 64],
    "kernel": 3,
    "dropout": 0.1,
}


def _write(path, config, seed, spec=SPEC):
    rng = np.random.default_rng(seed)
    arrays = {
        name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in parameter_layout(config)
    }
    dump_numpy_weights(
        path,
        arch="team_net",
        config=config,
        token_spec=spec.to_dict(),
        arrays=arrays,
        feature_version=FEATURE_VERSION,
    )
    return arrays


def _window(bundle, n=160):
    """A played stretch of real tokens, plus an empty team, junk on an invalid slot, and a frame
    with no player token."""
    tokens = build_match_tokens(bundle.tracking, SPEC)
    k = SPEC.tokens_per_team
    valid = tokens[:, :, Col.VALID] > 0.5
    played = bundle.meta.played_mask()
    both = played & valid[:, :k].any(axis=1) & valid[:, k : 2 * k].any(axis=1)
    start = int(np.flatnonzero(both)[len(np.flatnonzero(both)) // 2])
    if start + n > len(tokens):
        start = len(tokens) - n
    window = tokens[start : start + n].copy()
    window[5, k : 2 * k] = 0
    window[2, : 2 * k] = 0
    window[8, 2, :] = 0
    window[8, 2, Col.X] = 4.0
    return window


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_weight_round_trip_is_exact_and_the_default_file_is_small(tmp_path):
    path = tmp_path / "model.json"
    arrays = _write(path, DEFAULT, seed=0)
    assert path.stat().st_size < 3_000_000
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format_version"] == 1 and doc["arch"] == "team_net"
    assert doc["feature_version"] == FEATURE_VERSION
    assert doc["config"]["d_model"] == 64 and doc["token_spec"]["tokens_per_team"] == 11
    for entry, (name, shape) in zip(doc["parameters"], parameter_layout(DEFAULT), strict=True):
        raw = np.frombuffer(bytes.fromhex(entry["data"]), dtype="<f4").reshape(shape)
        assert entry["name"] == name
        assert raw.tobytes() == np.ascontiguousarray(arrays[name], dtype="<f4").tobytes()
    model = load_numpy_weights(path)
    assert model.receptive_half == 127 and model.spec == SPEC


def test_base64_weights_hold_the_same_bytes_in_a_smaller_file(small_bundle, tmp_path):
    rng = np.random.default_rng(3)
    arrays = {
        name: (rng.standard_normal(shape) * 0.02).astype(np.float32)
        for name, shape in parameter_layout(TINY)
    }
    paths = {}
    for encoding in ("hex", "base64"):
        paths[encoding] = tmp_path / f"{encoding}.json"
        dump_numpy_weights(
            paths[encoding],
            arch="team_net",
            config=TINY,
            token_spec=SPEC.to_dict(),
            arrays=arrays,
            feature_version=FEATURE_VERSION,
            encoding=encoding,
        )
    doc = json.loads(paths["base64"].read_text(encoding="utf-8"))
    assert doc["format_version"] == 2
    assert paths["base64"].stat().st_size < 0.75 * paths["hex"].stat().st_size
    hex_doc = json.loads(paths["hex"].read_text(encoding="utf-8"))
    for h, b in zip(hex_doc["parameters"], doc["parameters"], strict=True):
        assert h["name"] == b["name"] and h["shape"] == b["shape"]
        assert bytes.fromhex(h["data"]) == base64.b64decode(b["data"])
    a = predict_logits_np([paths["hex"]], small_bundle.tracking, chunk=1500, views=1)
    b = predict_logits_np([paths["base64"]], small_bundle.tracking, chunk=1500, views=1)
    assert np.array_equal(a.opinion, b.opinion)
    assert np.array_equal(a.logits[a.opinion], b.logits[b.opinion])
    with pytest.raises(ValueError, match="encoding"):
        dump_numpy_weights(
            tmp_path / "x.json",
            arch="team_net",
            config=TINY,
            token_spec=SPEC.to_dict(),
            arrays=arrays,
            encoding="raw",
        )
    doc["format_version"] = 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="weight format"):
        load_numpy_weights(bad)


def test_chunked_logits_match_one_pass(small_bundle, tmp_path):
    path = tmp_path / "tiny.json"
    _write(path, TINY, seed=1)
    whole = predict_logits_np([path], small_bundle.tracking, chunk=100_000, views=1)
    chunked = predict_logits_np([path], small_bundle.tracking, chunk=700, views=1)
    assert np.array_equal(whole.opinion, chunked.opinion)
    assert _maxabs(whole.logits[whole.opinion], chunked.logits[chunked.opinion]) <= 1e-10


def test_an_ensemble_of_two_files_is_the_mean(small_bundle, tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, TINY, seed=1)
    _write(b, TINY, seed=2)
    alone_a = predict_logits_np([a], small_bundle.tracking, chunk=1500)
    alone_b = predict_logits_np([b], small_bundle.tracking, chunk=1500)
    mean = predict_logits_np([a, b], small_bundle.tracking, chunk=1500)
    expect = ((alone_a.logits.astype(np.float64) + alone_b.logits.astype(np.float64)) / 2).astype(
        np.float32
    )
    expect[~mean.opinion] = np.nan
    assert np.array_equal(mean.logits, expect, equal_nan=True)
    assert np.array_equal(mean.opinion, alone_a.opinion)


def test_results_do_not_depend_on_the_thread_pool(small_bundle, tmp_path):
    path = tmp_path / "tiny.json"
    _write(path, TINY, seed=3)
    with threadpool_limits(1):
        first = predict_logits_np([path], small_bundle.tracking, chunk=1500)
        second = predict_logits_np([path], small_bundle.tracking, chunk=1500)
    with threadpool_limits(4):
        third = predict_logits_np([path], small_bundle.tracking, chunk=1500)
        fourth = predict_logits_np([path], small_bundle.tracking, chunk=1500)
    assert np.array_equal(first.logits, second.logits, equal_nan=True)
    assert np.array_equal(third.logits, fourth.logits, equal_nan=True)
    assert _maxabs(first.logits[first.opinion], third.logits[third.opinion]) <= 1e-12


def test_numpy_producer_abstains_checks_the_pin_and_returns_evidence(small_bundle, tmp_path):
    n = small_bundle.meta.n_frames
    abstain = learned_team(small_bundle, backend="numpy")
    assert isinstance(abstain, Evidence) and np.isnan(abstain.proba).all()
    path = tmp_path / "tiny.json"
    _write(path, TINY, seed=4)
    with pytest.raises(ValueError, match="pinned"):
        learned_team(small_bundle, weights=str(path), sha256="f" * 16, backend="numpy")
    evidence = learned_team(
        small_bundle, weights=str(path), sha256=sha256_file(path), backend="numpy"
    ).validate(n)
    played = small_bundle.meta.played_mask()
    assert evidence.proba.shape == (n, 3) and evidence.proba.dtype == np.float32
    assert np.isnan(evidence.proba[~played]).all()
    assert evidence.coverage() > 0
    assert (evidence.proba[played][:, 2][~np.isnan(evidence.proba[played, 0])] == 0).all()


def test_numpy_producer_runs_when_torch_is_blocked(small_bundle, tmp_path, monkeypatch):
    path = tmp_path / "tiny.json"
    _write(path, TINY, seed=5)
    monkeypatch.setitem(sys.modules, "torch", None)
    evidence = learned_team(small_bundle, weights=str(path), backend="numpy")
    assert evidence.proba.shape == (small_bundle.meta.n_frames, 3)
    assert not np.isnan(evidence.proba[evidence.proba[:, 0] == evidence.proba[:, 0]]).any()


@pytest.mark.slow
def test_one_match_one_view_one_thread(tmp_path):
    n = 199_000
    path = tmp_path / "full.json"
    _write(path, DEFAULT, seed=6)
    rng = np.random.default_rng(0)
    meta = MatchMeta("T", "synthetic", n, 25, 105.0, 68.0, (Period(1, 0, n),)).validate()
    player = rng.normal(0, 15, size=(n, 22, 2)).astype(np.float32)
    player[rng.random((n, 22)) < 0.05] = np.nan
    teams = np.zeros(22, dtype=np.int8)
    teams[11:] = 1
    ball = rng.normal(0, 10, size=(n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    tracking = Tracking(meta, player, teams, ball).validate()
    with threadpool_limits(1):
        started = time.perf_counter()
        prediction = predict_logits_np([path], tracking, chunk=2048, views=1)
        elapsed = time.perf_counter() - started
    print(f"SPEED numpy team_net {n} frames one view one thread: {elapsed:.2f} s")
    assert prediction.logits.shape == (n, 3) and prediction.logits.dtype == np.float32
    assert prediction.opinion.sum() > n // 2
    assert np.isfinite(prediction.logits[prediction.opinion]).all()


def test_float64_forward_matches_torch_including_empty_teams(small_bundle):
    torch = pytest.importorskip("torch")
    from aaaag.ml.model import ModelConfig, TeamNet

    torch.manual_seed(0)
    cfg = ModelConfig()
    module = TeamNet(cfg).eval()
    assert [name for name, _shape in parameter_layout(cfg.to_dict())] == list(module.state_dict())
    module = module.double()
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in module.state_dict().items()}
    net = __import__("aaaag.ml.np_infer", fromlist=["NumpyTeamNet"]).NumpyTeamNet(
        cfg.to_dict(), SPEC, arrays
    )
    window = _window(small_bundle)
    k = SPEC.tokens_per_team
    assert window[5, k : 2 * k, Col.VALID].sum() == 0
    assert (window[..., Col.VALID] < 0.5).any()
    batch = np.stack([window, window[::-1].copy()])
    with torch.inference_mode():
        ref = module(torch.from_numpy(np.ascontiguousarray(batch)).double())
    got = net.forward(batch)
    assert _maxabs(got["logits"], ref["logits"].numpy()) <= 1e-9
    assert _maxabs(got["ball"], ref["ball"].numpy()) <= 1e-9
    assert np.isfinite(got["logits"]).all()


def test_export_round_trip_matches_the_float32_values(small_bundle, tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import save_checkpoint
    from aaaag.ml.export import export_file, main
    from aaaag.ml.model import ModelConfig, TeamNet

    torch.manual_seed(1)
    cfg = ModelConfig(d_model=32, n_heads=4, n_blocks=2, dilations=(1, 2, 4, 8), dropout=0.1)
    module = TeamNet(cfg).eval()
    pt = tmp_path / "final.pt"
    js = tmp_path / "model.json"
    save_checkpoint(pt, module, SPEC, {})
    main(["--weights", str(pt), "--out", str(js)])
    assert js.is_file() and export_file(pt, tmp_path / "again.json").is_file()
    doc = json.loads(js.read_text(encoding="utf-8"))
    state = {}
    original = module.state_dict()
    for entry in doc["parameters"]:
        shape = tuple(entry["shape"])
        raw = np.frombuffer(bytes.fromhex(entry["data"]), dtype="<f4").reshape(shape).copy()
        assert raw.tobytes() == original[entry["name"]].detach().cpu().numpy().tobytes()
        state[entry["name"]] = torch.from_numpy(raw)
    fresh = TeamNet(cfg).eval()
    fresh.load_state_dict(state)
    fresh = fresh.double()
    net = load_numpy_weights(js)
    window = _window(small_bundle, n=96)
    with torch.inference_mode():
        ref = fresh(torch.from_numpy(np.ascontiguousarray(window)).double()[None])["logits"][0]
    got = net.forward(window[None])["logits"][0]
    assert _maxabs(got, ref.numpy()) <= 1e-9


def test_four_views_match_torch_and_the_torch_backend_is_unchanged(small_bundle, tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.checkpoint import save_checkpoint
    from aaaag.ml.export import export_file
    from aaaag.ml.infer import logits_from_tokens
    from aaaag.ml.model import ModelConfig, TeamNet

    torch.manual_seed(2)
    cfg = ModelConfig(**{**TINY, "dilations": (1, 2, 4)})
    module = TeamNet(cfg).eval()
    pt = tmp_path / "tiny.pt"
    js = tmp_path / "tiny.json"
    save_checkpoint(pt, module, SPEC, {})
    export_file(pt, js)
    tokens = build_match_tokens(small_bundle.tracking, SPEC)
    periods = small_bundle.meta.periods
    with torch.inference_mode():
        ref = logits_from_tokens(module, SPEC, tokens, periods, chunk=2000, views=4)
    got = logits_from_tokens_np(load_numpy_weights(js), SPEC, tokens, periods, chunk=2000, views=4)
    mask = ~np.isnan(ref[:, 0])
    assert np.array_equal(np.isnan(got[:, 0]), np.isnan(ref[:, 0]))
    assert _maxabs(got[mask], ref[mask]) <= 1e-4
    torch_evidence = learned_team(small_bundle, weights=str(pt), chunk=2000)
    again = learned_team(small_bundle, weights=str(pt), chunk=2000, backend="torch")
    assert np.array_equal(torch_evidence.proba, again.proba, equal_nan=True)
    numpy_evidence = learned_team(small_bundle, weights=str(js), chunk=2000, backend="numpy")
    played = small_bundle.meta.played_mask()
    finite = played & ~np.isnan(torch_evidence.proba[:, 0]) & ~np.isnan(numpy_evidence.proba[:, 0])
    assert _maxabs(numpy_evidence.proba[finite], torch_evidence.proba[finite]) <= 1e-4
