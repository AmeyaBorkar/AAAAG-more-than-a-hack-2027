"""Clean-ball teacher distillation (E51).

Torch only, on synthetic matches. `teacher_input` "same" is today's distillation, including the
weights. `clean_ball` stores a second token array per variant from the same draw with ball
smoothing skipped. The E51 config's teacher paths are placeholders and are not opened.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F  # noqa: E402

from aaaag.contracts import AWAY, HOME, MatchBundle, MatchMeta, Period, Tracking  # noqa: E402
from aaaag.ml.augment import (  # noqa: E402
    AugmentMix,
    BallOffset,
    BallSmooth,
    GhostSlots,
    TeamStyle,
)
from aaaag.ml.checkpoint import load_payload, save_checkpoint  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch, build_entry  # noqa: E402
from aaaag.ml.model import ModelConfig, TeamNet  # noqa: E402
from aaaag.ml.spec import Col, TokenSpec  # noqa: E402
from aaaag.ml.train import (  # noqa: E402
    DeviceSet,
    DistillConfig,
    RunConfig,
    TrainConfig,
    _teacher_batch_tokens,
    augment,
    loss_terms,
    train,
)
from aaaag.paths import CONFIGS_DIR  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
HEADED = replace(TINY, switch_head=True)
SHORT = SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.1)
SMOOTH = BallSmooth(p=1.0, sigma_frames=(4.0, 8.0))


def _run(steps: int = 4, model: ModelConfig = TINY, **train_changes) -> RunConfig:
    cfg = dict(
        seed=3,
        steps=steps,
        refresh_steps=steps,
        batch_size=2,
        window=64,
        warmup_steps=1,
        lr=3e-3,
        degrade=DegradeMix(version=None),
        amp=False,
    )
    cfg.update(train_changes)
    return RunConfig(token=SPEC, model=model, train=TrainConfig(**cfg))


def _fit(run: RunConfig, matches, out, **kwargs):
    return train(
        run,
        matches,
        out,
        device="cpu",
        log=lambda _: None,
        resume=False,
        threads=1,
        **kwargs,
    )


def _weights(path):
    return load_payload(path)["state_dict"]


def _equal(one, two) -> None:
    assert one.keys() == two.keys()
    assert all(torch.equal(one[key], two[key]) for key in one)


def _checkpoint(path, model_cfg: ModelConfig, seed: int):
    torch.manual_seed(seed)
    model = TeamNet(model_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_checkpoint(path, model, SPEC, {})


def _rng_blob(rng: np.random.Generator) -> bytes:
    return pickle.dumps(rng.bit_generator.state)


def _moving_match(n: int = 64) -> tuple[TrainMatch, TokenSpec]:
    """A fully seen square-wave ball and four fixed players, so smoothing cannot reorder slots."""
    spec = TokenSpec(tokens_per_team=2)
    step = (np.arange(n) // 4) % 2
    ball = np.stack(
        (np.where(step == 0, -8.0, 8.0), np.where(step == 0, -3.0, 3.0)), axis=1
    ).astype(np.float32)
    players = np.array([[-45.0, 0.0], [-25.0, 0.0], [25.0, 0.0], [45.0, 0.0]], np.float32)
    xy = np.broadcast_to(players, (n, 4, 2)).copy()
    teams = np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8)
    meta = MatchMeta("move", "synthetic", n, 25, 105.0, 68.0, (Period(1, 0, n),))
    labels = np.zeros(n, dtype=np.int8)
    labels[n // 2 :] = 1
    bundle = MatchBundle(Tracking(meta, xy, teams, ball), labels=labels).validate()
    return TrainMatch.from_bundle(bundle, spec), spec


def _only_ball_columns(student: np.ndarray, teacher: np.ndarray, spec: TokenSpec) -> None:
    """Differences sit on the ball token's kinematics and on player offsets to that ball."""
    assert student.shape == teacher.shape
    diff = student != teacher
    ball = spec.ball_index
    player_cols = np.zeros(spec.n_features, dtype=bool)
    player_cols[[Col.DX, Col.DY, Col.DIST]] = True
    ball_cols = np.zeros(spec.n_features, dtype=bool)
    ball_cols[[Col.X, Col.Y, Col.VX, Col.VY, Col.SPEED]] = True
    assert not diff[:, :ball, ~player_cols].any()
    assert not diff[:, ball, ~ball_cols].any()
    assert diff[:, ball, ball_cols].any()


def _keep_ball(ball_xy, periods, sigma):
    """The smooth draw already happened; return the track unchanged."""
    del periods, sigma
    return np.array(ball_xy, dtype=np.float32, copy=True)


@pytest.fixture(scope="module")
def matches():
    return [TrainMatch.from_bundle(generate(seed=s, config=SHORT), SPEC) for s in (7, 8)]


def test_same_teacher_input_matches_a_run_without_the_key(matches, tmp_path):
    """Explicit `same` is the distillation run that does not set the key, weights included."""
    model = replace(TINY, dropout=0.1)
    teacher = _checkpoint(tmp_path / "teacher" / "final.pt", TINY, seed=11)
    runs = str(teacher.parent)
    plain = _fit(
        _run(
            model=model,
            refresh_steps=2,
            distill=DistillConfig(teacher_runs=(runs,), alpha=0.5, temperature=2.0),
        ),
        matches,
        tmp_path / "plain",
        teachers=[teacher],
    )
    same = _fit(
        _run(
            model=model,
            refresh_steps=2,
            distill=DistillConfig(
                teacher_runs=(runs,),
                alpha=0.5,
                temperature=2.0,
                teacher_input="same",
                switch_alpha=0.0,
            ),
        ),
        matches,
        tmp_path / "same",
        teachers=[teacher],
    )
    _equal(_weights(plain), _weights(same))


def test_same_draws_no_extra_random_numbers_and_stores_no_teacher_tokens():
    match, spec = _moving_match()
    mix = AugmentMix(ball_smooth=SMOOTH)
    degrade = DegradeMix(version=None)
    rng_same, rng_clean = np.random.default_rng(3), np.random.default_rng(3)
    same = build_entry(match, rng_same, spec, degrade, mix, teacher_input="same")
    clean = build_entry(match, rng_clean, spec, degrade, mix, teacher_input="clean_ball")
    assert same.teacher_coupled is None and same.teacher_decoupled is None
    assert np.array_equal(same.coupled, clean.coupled)
    assert np.array_equal(same.decoupled, clean.decoupled)
    assert _rng_blob(rng_same) == _rng_blob(rng_clean)


def test_clean_ball_tokens_differ_only_on_ball_columns_and_match_smoothing_off(monkeypatch):
    match, spec = _moving_match()
    degrade = DegradeMix(version=None)
    simple = AugmentMix(ball_smooth=SMOOTH)
    rich = AugmentMix(
        team_style=TeamStyle(p=1.0, stretch=0.1, shift_m=2.0),
        ball_offset=BallOffset(p=1.0, bias_m=1.0, lag_frames=2, noise_m=0.0),
        ball_smooth=SMOOTH,
        ghosts=GhostSlots(p=1.0, max_ghost_slots=2, duplicate_p=0.0),
    )
    simple_clean = build_entry(
        match, np.random.default_rng(5), spec, degrade, simple, teacher_input="clean_ball"
    )
    rich_clean = build_entry(
        match, np.random.default_rng(6), spec, degrade, rich, teacher_input="clean_ball"
    )
    for variant in ("coupled", "decoupled"):
        _only_ball_columns(
            getattr(simple_clean, variant),
            getattr(simple_clean, f"teacher_{variant}"),
            spec,
        )
    # The smooth coin and sigma are still drawn; the filter itself is the identity, so every
    # other draw matches the clean-ball teacher.
    monkeypatch.setattr("aaaag.ml.augment.smooth_ball_track", _keep_ball)
    simple_off = build_entry(
        match, np.random.default_rng(5), spec, degrade, simple, teacher_input="same"
    )
    rich_off = build_entry(
        match, np.random.default_rng(6), spec, degrade, rich, teacher_input="same"
    )
    assert np.array_equal(simple_clean.teacher_coupled, simple_off.coupled)
    assert np.array_equal(simple_clean.teacher_decoupled, simple_off.decoupled)
    assert np.array_equal(rich_clean.teacher_coupled, rich_off.coupled)
    assert np.array_equal(rich_clean.teacher_decoupled, rich_off.decoupled)


def test_the_batch_windows_swaps_and_flips_teacher_tokens_with_the_student():
    match, spec = _moving_match()
    mix = AugmentMix(ball_smooth=SMOOTH)
    entry = build_entry(
        match,
        np.random.default_rng(4),
        spec,
        DegradeMix(version=None),
        mix,
        teacher_input="clean_ball",
    )
    data = DeviceSet([entry], [match], torch.device("cpu"))
    cfg = TrainConfig()
    for decoupled, kind in ((False, "coupled"), (True, "decoupled")):
        batch = data.batch(np.random.default_rng(1), 2, 24, decoupled=decoupled)
        assert batch["teacher_tokens"].shape == batch["tokens"].shape
        stored_s = torch.from_numpy(getattr(entry, kind)).float()
        stored_t = torch.from_numpy(getattr(entry, f"teacher_{kind}")).float()
        window = 24
        for row in range(batch["tokens"].shape[0]):
            found = False
            for start in range(stored_s.shape[0] - window + 1):
                same_student = torch.equal(batch["tokens"][row], stored_s[start : start + window])
                same_teacher = torch.equal(
                    batch["teacher_tokens"][row], stored_t[start : start + window]
                )
                if same_student and same_teacher:
                    found = True
                    break
            assert found
        raw = {key: value.clone() for key, value in batch.items()}
        augment(batch, spec, cfg, np.random.default_rng(2))
        replay = {key: value.clone() for key, value in raw.items() if key != "teacher_tokens"}
        replay["tokens"] = raw["teacher_tokens"].clone()
        augment(replay, spec, cfg, np.random.default_rng(2))
        assert torch.equal(batch["teacher_tokens"], replay["tokens"])
        assert torch.equal(batch["labels"], replay["labels"])
        plain = {key: value.clone() for key, value in raw.items() if key != "teacher_tokens"}
        augment(plain, spec, cfg, np.random.default_rng(2))
        assert torch.equal(plain["tokens"], batch["tokens"])


def test_same_selects_the_student_tokens_and_clean_ball_selects_the_teacher_tokens():
    student = torch.zeros(1, 2, 1, 1)
    teacher = torch.ones(1, 2, 1, 1)
    batch = {"tokens": student, "teacher_tokens": teacher}
    same = DistillConfig(teacher_runs=("r",), teacher_input="same")
    clean = replace(same, teacher_input="clean_ball")
    assert _teacher_batch_tokens(batch, same) is student
    assert _teacher_batch_tokens(batch, clean) is teacher
    with pytest.raises(ValueError, match="teacher tokens"):
        _teacher_batch_tokens({"tokens": student}, clean)


def test_clean_ball_run_trains_and_the_kl_is_finite(matches, tmp_path, monkeypatch):
    import aaaag.ml.train as train_mod

    teacher = _checkpoint(tmp_path / "teacher" / "final.pt", HEADED, seed=4)
    calls: list[bool] = []
    real = train_mod._teacher_batch_tokens

    def spy(batch, distill):
        got = real(batch, distill)
        calls.append(got is batch["teacher_tokens"])
        return got

    monkeypatch.setattr(train_mod, "_teacher_batch_tokens", spy)
    final = _fit(
        _run(
            steps=2,
            model=HEADED,
            distill=DistillConfig(
                teacher_runs=(str(teacher.parent),),
                alpha=0.5,
                temperature=2.0,
                teacher_input="clean_ball",
                switch_alpha=0.2,
            ),
            augment=AugmentMix(ball_smooth=BallSmooth(p=1.0, sigma_frames=(2.0, 4.0))),
        ),
        matches,
        tmp_path / "student",
        teachers=[teacher],
    )
    history = load_payload(final)["meta"]["history"]
    assert calls and all(calls)
    assert math.isfinite(history[-1]["loss"])
    assert math.isfinite(history[-1]["kl"])
    assert math.isfinite(history[-1]["swd"])


def test_switch_distill_is_bce_toward_the_sigmoid_and_weight_zero_is_a_no_op():
    b, w, margin = 2, 20, 4
    switch = torch.randn(b, w)
    teacher_switch = torch.randn(b, w)
    labels = torch.randint(0, 3, (b, w))
    batch = {
        "tokens": torch.zeros(b, w, 1, 1),
        "labels": labels,
        "ball": torch.zeros(b, w, 2),
        "ball_ok": torch.zeros(b, w, dtype=torch.bool),
    }
    out = {
        "logits": torch.randn(b, w, 3),
        "ball": torch.zeros(b, w, 2),
        "switch": switch,
    }
    weights = torch.ones(3)
    base = loss_terms(out, batch, weights, margin, 0.0)
    off = loss_terms(
        out,
        batch,
        weights,
        margin,
        0.0,
        teacher_logits=torch.randn(b, w, 3),
        teacher_switch=teacher_switch,
        distill=DistillConfig(teacher_runs=("r",), alpha=0.0, switch_alpha=0.0),
    )
    assert torch.equal(off["loss"], base["loss"])
    assert "swd" not in off and "kl" not in off

    on = loss_terms(
        out,
        batch,
        weights,
        margin,
        0.0,
        teacher_logits=torch.randn(b, w, 3),
        teacher_switch=teacher_switch,
        distill=DistillConfig(teacher_runs=("r",), alpha=0.0, temperature=2.0, switch_alpha=0.2),
    )
    inner = slice(margin, w - margin)
    expected = F.binary_cross_entropy_with_logits(
        switch[:, inner].reshape(-1), torch.sigmoid(teacher_switch[:, inner].reshape(-1))
    )
    assert float(on["swd"]) == pytest.approx(float(expected))
    assert torch.equal(on["loss"], base["loss"] + 0.2 * on["swd"])
    assert "kl" not in on

    edge = teacher_switch.clone()
    edge[:, :margin] += 4.0
    ignored = loss_terms(
        out,
        batch,
        weights,
        margin,
        0.0,
        teacher_logits=torch.zeros(b, w, 3),
        teacher_switch=edge,
        distill=DistillConfig(teacher_runs=("r",), alpha=0.0, switch_alpha=0.2),
    )
    assert torch.equal(ignored["swd"], on["swd"])
    moved = teacher_switch.clone()
    moved[:, margin] += 1.5
    changed = loss_terms(
        out,
        batch,
        weights,
        margin,
        0.0,
        teacher_logits=torch.zeros(b, w, 3),
        teacher_switch=moved,
        distill=DistillConfig(teacher_runs=("r",), alpha=0.0, switch_alpha=0.2),
    )
    assert not torch.equal(changed["swd"], on["swd"])


def test_clean_ball_without_ball_smooth_raises():
    with pytest.raises(ValueError, match="ball_smooth"):
        TrainConfig(distill=DistillConfig(teacher_runs=("r",), teacher_input="clean_ball"))
    with pytest.raises(ValueError, match="ball_smooth"):
        TrainConfig(
            augment=AugmentMix(ball_smooth=BallSmooth(p=0.0, sigma_frames=(4.0,))),
            distill=DistillConfig(teacher_runs=("r",), teacher_input="clean_ball"),
        )
    with pytest.raises(ValueError, match="teacher_runs"):
        DistillConfig(teacher_runs=(), teacher_input="clean_ball")
    with pytest.raises(ValueError, match="teacher_input"):
        DistillConfig(teacher_runs=("r",), teacher_input="blurred")
    kept = TrainConfig(
        augment=AugmentMix(ball_smooth=BallSmooth(p=1.0, sigma_frames=(4.0,))),
        distill=DistillConfig(
            teacher_runs=("r",), teacher_input="clean_ball", switch_alpha=0.2, alpha=0.5
        ),
    )
    assert kept.distill is not None and kept.distill.teacher_input == "clean_ball"


def test_to_dict_drops_the_new_defaults_and_e51_accepts_placeholders():
    """Placeholders are directory names. This round-trip does not resolve them."""
    previous = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_distill.yaml")
    assert set(previous.to_dict()["train"]["distill"]) == {"teacher_runs", "alpha", "temperature"}
    explicit = replace(
        previous.train.distill,
        teacher_input="same",
        switch_alpha=0.0,
    )
    assert explicit is not None
    dumped = replace(previous, train=replace(previous.train, distill=explicit)).to_dict()
    assert dumped["train"]["distill"].keys() == previous.to_dict()["train"]["distill"].keys()

    e51 = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_both_smooth_wide_e51.yaml")
    base = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_cm_contact_both_smooth_wide.yaml")
    assert e51.train.distill == DistillConfig(
        teacher_runs=("TEACHER_RUN_0", "TEACHER_RUN_1"),
        alpha=0.5,
        temperature=2.0,
        teacher_input="clean_ball",
        switch_alpha=0.2,
    )
    assert replace(e51, train=replace(e51.train, distill=None)) == base
    assert RunConfig.from_dict(e51.to_dict()) == e51
    block = e51.to_dict()["train"]["distill"]
    assert block["teacher_input"] == "clean_ball" and block["switch_alpha"] == 0.2
