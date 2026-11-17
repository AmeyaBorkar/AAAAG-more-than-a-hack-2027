"""Ball de-blurrer: numpy inference, gaps, a tiny train, and the producers left unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, H, MatchBundle, MatchMeta, Period, Tracking
from aaaag.evidence.possession.learned_team import learned_team
from aaaag.ml.np_deblur import (
    MIN_RUN,
    NumpyDeblur,
    apply_deblur,
    ball_features,
    correct_ball,
    count_parameters,
    deblur_ball,
    default_config,
    dump_deblur_weights,
    evaluate_deblur,
    format_eval,
    load_deblur_weights,
    load_named_match,
    parameter_layout,
    smooth_for_deblur,
)
from aaaag.ml.weights import sha256_file
from aaaag.postprocess.controlled import controlled_possession, controlled_spell
from aaaag.registry import EVIDENCE, SOURCES


def _bias_weights(path, bias=(8.0, 0.0)) -> None:
    config = default_config()
    arrays = {name: np.zeros(shape, np.float64) for name, shape in parameter_layout(config)}
    arrays["head.bias"] = np.asarray(bias, dtype=np.float64)
    dump_deblur_weights(path, arrays, config)


def _random_weights(seed: int, scale: float = 0.15) -> NumpyDeblur:
    rng = np.random.default_rng(seed)
    config = default_config()
    arrays = {name: rng.normal(scale=scale, size=shape) for name, shape in parameter_layout(config)}
    return NumpyDeblur(config, arrays)


def test_default_layout_is_five_dilated_layers():
    config = default_config()
    assert config["dilations"] == [1, 2, 4, 8, 16]
    assert config["channels"] == 32 and config["kernel"] == 5 and config["min_run"] == 8
    # first conv 3->32 kernel 5, four residual 32->32 kernel 5, pointwise head 32->2
    assert count_parameters() == 32 * 3 * 5 + 32 + 4 * (32 * 32 * 5 + 32) + 2 * 32 + 2


def test_features_scale_positions_and_zero_invalid_frames():
    ball = np.array([[52.5, -34.0], [np.nan, 0.0], [0.0, 0.0]], dtype=np.float64)
    feat = ball_features(ball)
    assert feat.shape == (3, 3)
    assert feat[0].tolist() == pytest.approx([1.0, -1.0, 1.0])
    assert feat[1].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert feat[2].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_sigma_zero_is_a_copy_not_a_gaussian():
    ball = np.zeros((30, 2), dtype=np.float32)
    ball[10] = (5.0, -3.0)
    out = smooth_for_deblur(ball, (Period(1, 0, 30),), 0)
    assert out.dtype == np.float32
    assert np.array_equal(out, ball)
    assert not np.shares_memory(out, ball)


def test_json_roundtrip_keeps_float32_values(tmp_path):
    rng = np.random.default_rng(0)
    config = default_config()
    arrays = {name: rng.normal(size=shape) for name, shape in parameter_layout(config)}
    path = tmp_path / "w.json"
    dump_deblur_weights(path, arrays, config)
    loaded = load_deblur_weights(path)
    quantised = {name: np.array(value, dtype=np.float32) for name, value in arrays.items()}
    direct = NumpyDeblur(config, quantised)
    feat = rng.normal(size=(2, 24, 3))
    assert np.array_equal(loaded.forward(feat), direct.forward(feat))


def test_nan_gaps_are_preserved_and_short_runs_are_unchanged(tmp_path):
    path = tmp_path / "bias.json"
    _bias_weights(path, bias=(8.0, 0.0))
    ball = np.full((100, 2), np.nan, dtype=np.float32)
    ball[0:15] = (1.0, 2.0)  # outside the period, long enough to be corrected if we ignored it
    ball[90:100] = (3.0, 4.0)
    ball[20:27] = (5.0, 6.0)  # 7 frames, copied
    ball[30:60] = (10.0, 20.0)  # 30 frames, corrected
    ball[70:78] = (7.0, 8.0)  # 8 frames, the shortest run that is corrected
    original = ball.copy()
    out = deblur_ball(ball, (Period(1, 20, 90),), path)
    assert out.dtype == np.float32 and out.shape == ball.shape
    assert np.array_equal(ball, original, equal_nan=True)
    assert np.array_equal(out[0:15], original[0:15])
    assert np.array_equal(out[90:100], original[90:100])
    assert np.array_equal(out[20:27], original[20:27])
    assert np.isnan(out[27:30]).all() and np.isnan(out[60:70]).all() and np.isnan(out[78:90]).all()
    assert np.allclose(out[30:60], original[30:60] + np.array([8.0, 0.0]))
    assert np.allclose(out[70:78], original[70:78] + np.array([8.0, 0.0]))
    # A finite stretch that crosses a period boundary is two runs. Five and five stay put.
    crossed = np.full((20, 2), 1.0, dtype=np.float32)
    crossed_out = deblur_ball(crossed, (Period(1, 0, 5), Period(2, 5, 10)), path)
    assert np.array_equal(crossed_out[:10], crossed[:10])


def test_a_gap_stops_the_receptive_field():
    model = _random_weights(0)
    track = np.linspace(-10.0, 20.0, 40)
    ball = np.stack((track, np.zeros(40)), axis=1)
    periods = (Period(1, 0, 40),)
    intact = apply_deblur(ball, periods, model)
    split_ball = ball.copy()
    split_ball[20] = np.nan
    split = apply_deblur(split_ball, periods, model)
    assert np.isnan(split[20]).all()
    assert np.max(np.abs(intact[16:20] - split[16:20])) > 1e-3
    missing = ~np.isfinite(split_ball).all(axis=1)
    assert np.array_equal(np.isnan(split).all(axis=1), missing)


def test_numpy_matches_torch_on_random_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from aaaag.ml.deblur import DeblurNet, export_weights

    torch.manual_seed(0)
    model = DeblurNet().double().eval()
    layout = [name for name, _shape in parameter_layout(default_config())]
    assert list(model.state_dict()) == layout
    assert sum(p.numel() for p in model.parameters()) == count_parameters()
    features = torch.randn(2, 37, 3, dtype=torch.float64)
    with torch.no_grad():
        reference = model(features).numpy()
    arrays = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    got = NumpyDeblur(default_config(), arrays).forward(features.numpy())
    assert float(np.max(np.abs(got - reference))) < 1e-4

    path = tmp_path / "deblur.json"
    export_weights(model, path)
    loaded = load_deblur_weights(path)
    rebuilt = DeblurNet().double().eval()
    rebuilt.load_state_dict(
        {name: torch.from_numpy(value) for name, value in loaded.weights.items()}
    )
    with torch.no_grad():
        quantised = rebuilt(features).numpy()
    again = loaded.forward(features.numpy())
    assert float(np.max(np.abs(again - quantised))) < 1e-4

    rng = np.random.default_rng(1)
    ball = rng.normal(scale=8.0, size=(60, 2))
    ball[15:22] = np.nan
    ball[40:44, 0] = np.nan
    periods = (Period(1, 0, 50),)

    def torch_delta(run: np.ndarray) -> np.ndarray:
        feat = torch.from_numpy(np.ascontiguousarray(ball_features(run)))
        with torch.no_grad():
            return rebuilt(feat[None])[0].detach().cpu().numpy()

    reference_ball = correct_ball(ball, periods, torch_delta, min_run=MIN_RUN)
    numpy_ball = apply_deblur(ball, periods, loaded)
    finite = np.isfinite(reference_ball) & np.isfinite(numpy_ball)
    assert np.array_equal(np.isnan(reference_ball), np.isnan(numpy_ball))
    assert float(np.max(np.abs(reference_ball[finite] - numpy_ball[finite]))) < 1e-4
    public = deblur_ball(ball, periods, path)
    assert public.dtype == np.float32
    assert np.array_equal(public, numpy_ball.astype(np.float32), equal_nan=True)


def _clean_synthetic(seed: int, minutes: float):
    return SOURCES.get("synthetic").fn(
        seed=seed,
        minutes_per_half=minutes,
        gap_minutes=0.0,
        ball_noise_m=0.0,
        ball_missing_rate=0.0,
        dead_extra_missing_rate=0.0,
    )


def test_training_reduces_the_loss_on_synthetic_tracks():
    pytest.importorskip("torch")
    from aaaag.ml.deblur import train_deblur

    bundles = [_clean_synthetic(0, 1.0), _clean_synthetic(1, 1.0)]
    result = train_deblur(
        bundles, seed=0, steps=50, batch_size=8, window=256, probe_batches=2, out=None
    )
    assert result["parameters"] == count_parameters()
    assert np.isfinite([result["probe_before"], result["probe_after"]]).all()
    assert result["probe_after"] < result["probe_before"], result


def test_training_is_deterministic_for_a_seed():
    pytest.importorskip("torch")
    from aaaag.ml.deblur import train_deblur

    bundle = _clean_synthetic(2, 0.5)
    kwargs = {"seed": 0, "steps": 3, "batch_size": 2, "window": 64, "probe_batches": 1}
    first = train_deblur([bundle], **kwargs)
    second = train_deblur([bundle], **kwargs)
    for key in ("loss_first", "loss_last", "probe_before", "probe_after"):
        assert first[key] == second[key]


def test_eval_prints_aggregates_and_identity_weights_do_not_move_the_ball(tmp_path):
    path = tmp_path / "zero.json"
    _bias_weights(path, bias=(0.0, 0.0))
    bundle = _clean_synthetic(3, 0.5)
    rows = evaluate_deblur(path, [bundle])
    assert [int(row["sigma"]) for row in rows] == [0, 4, 8]
    assert rows[0]["rmse_smooth_m"] == 0.0
    assert rows[0]["rmse_deblur_m"] == 0.0
    assert rows[0]["frames"] > 0
    assert rows[0]["touch_clean"] == rows[0]["touch_smooth"] == rows[0]["touch_deblur"]
    assert rows[1]["rmse_smooth_m"] > 1e-3
    assert rows[1]["rmse_deblur_m"] == pytest.approx(rows[1]["rmse_smooth_m"])
    assert rows[2]["rmse_deblur_m"] == pytest.approx(rows[2]["rmse_smooth_m"])
    assert rows[1]["touch_smooth"] == rows[1]["touch_deblur"]
    text = format_eval(rows)
    assert text.count("\n") == 2
    assert "[" not in text


def test_named_matches_refuse_the_submission_source_and_non_dev_idsse():
    from aaaag.sources.idsse import EXAM_MATCHES

    with pytest.raises(ValueError, match="not available"):
        load_named_match("bar-ath/H")
    with pytest.raises(ValueError, match="dev match"):
        load_named_match(f"idsse/{EXAM_MATCHES[0]}")
    with pytest.raises(ValueError, match="source/id"):
        load_named_match("metrica")


def test_cli_train_defaults_match_the_three_metrica_games():
    pytest.importorskip("torch")
    from aaaag.ml.deblur import build_parser

    train = build_parser().parse_args(["train", "--out", "w.json"])
    assert tuple(train.matches) == ("metrica/1", "metrica/2", "metrica/3")
    assert train.seed == 0 and train.steps == 3000 and train.batch == 64 and train.window == 512
    evaluate = build_parser().parse_args(
        ["eval", "--weights", "w.json", "--matches", "idsse/J03WMX"]
    )
    assert evaluate.matches == ["idsse/J03WMX"]


def test_learned_team_without_deblur_is_byte_identical(small_bundle):
    ball = small_bundle.tracking.ball_xy
    before = ball.tobytes()
    plain = learned_team(small_bundle)
    explicit = learned_team(small_bundle, ball_deblur=None, ball_deblur_sha256=None)
    component = EVIDENCE.get("learned_team")
    via_example = component.fn(small_bundle, **component.example)
    assert plain.proba.tobytes() == explicit.proba.tobytes() == via_example.proba.tobytes()
    assert plain.channels == explicit.channels == via_example.channels == {}
    assert np.isnan(plain.proba).all()
    assert small_bundle.tracking.ball_xy is ball
    assert ball.tobytes() == before


def test_a_set_deblur_file_is_checked_and_does_not_mutate_the_bundle(small_bundle, tmp_path):
    path = tmp_path / "bias.json"
    _bias_weights(path)
    ball = small_bundle.tracking.ball_xy
    before = ball.tobytes()
    with pytest.raises(FileNotFoundError, match="ball_deblur"):
        learned_team(small_bundle, ball_deblur=str(tmp_path / "missing.json"))
    evidence = learned_team(
        small_bundle, ball_deblur=str(path), ball_deblur_sha256=sha256_file(path)[:12]
    )
    assert np.isnan(evidence.proba).all()
    assert small_bundle.tracking.ball_xy is ball and ball.tobytes() == before
    with pytest.raises(ValueError, match="pinned"):
        learned_team(small_bundle, ball_deblur=str(path), ball_deblur_sha256="0123456789abcdef")


def _pass_bundle():
    """A short away spell kept only because two teammates are within a metre of the ball."""
    n = 92
    labels = np.array([H] * 40 + [A] * 12 + [H] * 40, dtype=np.int8)
    ball = np.zeros((n, 2), dtype=np.float32)
    players = np.zeros((n, 3, 2), dtype=np.float32)
    players[:, 0] = (0.2, 0.0)
    players[:, 1] = (0.0, 0.2)
    players[:, 2] = (40.0, 40.0)
    teams = np.array([A, A, H], dtype=np.int8)
    meta = MatchMeta(
        match_id="deblur-pass",
        source="synthetic",
        n_frames=n,
        fps=25,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(Period(1, 0, n),),
    )
    tracking = Tracking(meta=meta, player_xy=players, slot_team=teams, ball_xy=ball).validate()
    return labels, MatchBundle(tracking=tracking, labels=labels)


def test_controlled_spell_reads_a_deblurred_copy(tmp_path):
    labels, bundle = _pass_bundle()
    ball = bundle.tracking.ball_xy
    before = ball.tobytes()
    plain = controlled_spell(labels, bundle)
    same = controlled_spell(labels, bundle, ball_deblur=None, ball_deblur_sha256=None)
    assert np.array_equal(plain, same)
    assert np.array_equal(plain, controlled_possession(labels, bundle.tracking))
    assert (plain[40:52] == A).all()
    path = tmp_path / "shift.json"
    _bias_weights(path, bias=(8.0, 0.0))
    shifted = controlled_spell(labels, bundle, ball_deblur=str(path))
    assert (shifted == H).all()
    assert bundle.tracking.ball_xy is ball and ball.tobytes() == before
    with pytest.raises(ValueError, match="pinned"):
        controlled_spell(
            labels, bundle, ball_deblur=str(path), ball_deblur_sha256="fedcba9876543210"
        )
