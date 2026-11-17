"""The torch side of the learned team harness: model, training, checkpoints, inference, producer.

Skipped cleanly where torch is not installed (the repository's normal environment); run in the DL
environment (requirements-dl.txt, docs/ml/README.md). Synthetic matches and a tiny model, on the
CPU, so the whole file takes seconds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from aaaag import pipeline  # noqa: E402
from aaaag.contracts import AWAY, HOME, Evidence, validate_labels  # noqa: E402
from aaaag.evidence.possession.learned_team import learned_team  # noqa: E402
from aaaag.ml.checkpoint import load_checkpoint, load_payload, save_checkpoint  # noqa: E402
from aaaag.ml.data import DegradeMix, TrainMatch  # noqa: E402
from aaaag.ml.geometry import SWAP_CLASSES  # noqa: E402
from aaaag.ml.infer import logits_from_tokens, predict_logits  # noqa: E402
from aaaag.ml.model import (  # noqa: E402
    ModelConfig,
    TeamNet,
    count_parameters,
    get_architecture,
    register_architecture,
)
from aaaag.ml.spec import Col, TokenSpec  # noqa: E402
from aaaag.ml.train import RunConfig, TrainConfig, augment, loss_terms, lr_at, train  # noqa: E402
from aaaag.ml.weights import sha256_file  # noqa: E402
from aaaag.ml.windows import build_match_tokens  # noqa: E402
from aaaag.pipeline import PipelineConfig  # noqa: E402
from aaaag.submission import to_csv_bytes  # noqa: E402
from aaaag.synthetic import SyntheticConfig, generate  # noqa: E402

SPEC = TokenSpec()
TINY = ModelConfig(d_model=16, n_heads=2, n_blocks=1, dilations=(1, 2, 4), dropout=0.0)
SMALL = SyntheticConfig(minutes_per_half=3.0, gap_minutes=0.5)


def _tiny_model(seed: int = 0) -> TeamNet:
    torch.manual_seed(seed)
    return TeamNet(TINY).eval()


def _tokens(bundle, lo: int = 1000, n: int = 120) -> np.ndarray:
    return build_match_tokens(bundle.tracking, SPEC)[lo : lo + n]


def test_the_default_model_is_small_and_shapes_follow_the_layout(small_bundle):
    model = TeamNet().eval()
    assert 100_000 < count_parameters(model) < 1_500_000
    x = torch.from_numpy(_tokens(small_bundle, n=40))[None].repeat(2, 1, 1, 1)
    out = model(x)
    assert out["logits"].shape == (2, 40, 3) and out["ball"].shape == (2, 40, 2)
    assert torch.isfinite(out["logits"]).all()
    assert model.receptive_half == 127 == model.cfg.receptive_half


def test_invalid_player_tokens_and_the_order_inside_a_team_do_not_matter(small_bundle):
    model = _tiny_model()
    tokens = _tokens(small_bundle, n=60).copy()
    with torch.no_grad():
        base = model(torch.from_numpy(tokens)[None])["logits"]
        noisy = tokens.copy()
        junk = np.random.default_rng(0).normal(size=noisy.shape).astype(np.float32)
        invalid = noisy[..., Col.VALID] < 0.5
        invalid[:, 22] = False  # the ball token is read even when the ball is unseen: its gap
        for column in range(noisy.shape[-1]):  # features are the point; players are masked
            if column != Col.VALID:
                noisy[..., column] = np.where(invalid, junk[..., column], noisy[..., column])
        assert invalid.any() and not np.array_equal(noisy, tokens)
        assert torch.allclose(model(torch.from_numpy(noisy)[None])["logits"], base, atol=1e-5)
        rng = np.random.default_rng(1)
        order = np.concatenate([rng.permutation(11), 11 + rng.permutation(11), [22]])
        shuffled = tokens[:, order]
        assert torch.allclose(model(torch.from_numpy(shuffled)[None])["logits"], base, atol=1e-4)


def test_a_frame_with_nobody_observed_stays_finite():
    out = _tiny_model()(torch.zeros(1, 30, SPEC.n_tokens, 14))
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["ball"]).all()


def test_the_temporal_receptive_field_is_what_the_config_says(small_bundle):
    model = _tiny_model()
    tokens = _tokens(small_bundle, n=80).copy()
    changed = tokens.copy()
    changed[0] += 1.0  # frame 0 only
    with torch.no_grad():
        a = model(torch.from_numpy(tokens)[None])["logits"][0]
        b = model(torch.from_numpy(changed)[None])["logits"][0]
    reach = model.receptive_half
    assert (a[: reach + 1] != b[: reach + 1]).any()
    assert torch.equal(a[reach + 1 :], b[reach + 1 :])


def test_chunked_inference_equals_one_pass(small_bundle):
    model = _tiny_model(3)
    tokens = build_match_tokens(small_bundle.tracking, SPEC)
    periods = small_bundle.meta.periods
    whole = logits_from_tokens(model, SPEC, tokens, periods, chunk=100_000)
    chunked = logits_from_tokens(model, SPEC, tokens, periods, chunk=700)
    played = small_bundle.meta.played_mask()
    assert np.isnan(whole[~played]).all() and not np.isnan(whole[played]).any()
    assert np.allclose(whole[played], chunked[played], atol=1e-4)


def test_four_views_make_the_prediction_symmetric(small_bundle):
    model = _tiny_model(5)
    loaded = _loaded(model)
    tracking = small_bundle.tracking
    swapped = replace(
        tracking, slot_team=(1 - tracking.slot_team).astype(np.int8), _fingerprint=None
    )
    mirror = np.array([1.0, -1.0], dtype=np.float32)
    flipped = replace(
        tracking,
        player_xy=tracking.player_xy * mirror,
        ball_xy=tracking.ball_xy * mirror,
        _fingerprint=None,
    )
    base = predict_logits([loaded], tracking, views=4, chunk=2000).logits
    played = small_bundle.meta.played_mask()
    swapped_logits = predict_logits([loaded], swapped, views=4, chunk=2000).logits
    assert np.allclose(swapped_logits[played][:, SWAP_CLASSES], base[played], atol=1e-4)
    mirrored = predict_logits([loaded], flipped, views=2, chunk=2000).logits
    plain = predict_logits([loaded], tracking, views=2, chunk=2000).logits
    assert np.allclose(mirrored[played], plain[played], atol=1e-4)


def _loaded(model: TeamNet):
    from aaaag.ml.checkpoint import Loaded

    return Loaded(model=model, spec=SPEC, meta={}, sha256="", payload={})


def test_inference_is_bit_exact_from_run_to_run_on_one_thread(small_bundle):
    loaded = _loaded(_tiny_model(7))
    first = predict_logits([loaded], small_bundle.tracking, chunk=1500, threads=1)
    second = predict_logits([loaded], small_bundle.tracking, chunk=1500, threads=1)
    assert np.array_equal(first.logits, second.logits, equal_nan=True)
    assert np.array_equal(first.opinion, second.opinion)
    proba = first.proba()
    assert np.allclose(proba[first.opinion].sum(axis=1), 1.0, atol=1e-6)
    assert np.isnan(proba[~first.opinion]).all()
    team = first.team_probability()
    assert np.allclose(
        team[first.opinion],
        proba[first.opinion, 0] / (proba[first.opinion, :2].sum(axis=1)),
        atol=1e-5,
    )


def test_an_ensemble_averages_logits_and_needs_one_spec(small_bundle):
    a, b = _loaded(_tiny_model(1)), _loaded(_tiny_model(2))
    tracking = small_bundle.tracking
    mean = predict_logits([a, b], tracking, chunk=1500).logits
    alone = [predict_logits([m], tracking, chunk=1500).logits for m in (a, b)]
    played = small_bundle.meta.played_mask()
    assert np.allclose(mean[played], (alone[0][played] + alone[1][played]) / 2, atol=1e-6)
    odd = _loaded(_tiny_model(3))
    odd.spec = TokenSpec(tokens_per_team=5)
    with pytest.raises(ValueError, match="share a token spec"):
        predict_logits([a, odd], tracking)
    with pytest.raises(ValueError):
        predict_logits([], tracking)


# -- training -------------------------------------------------------------------------------------


def _run(steps: int = 6, **train_changes) -> RunConfig:
    train_cfg = TrainConfig(
        seed=3,
        steps=steps,
        refresh_steps=3,
        batch_size=2,
        window=64,
        warmup_steps=1,
        lr=3e-3,
        degrade=DegradeMix(version=None),
        amp=False,
        **train_changes,
    )
    return RunConfig(token=SPEC, model=TINY, train=train_cfg)


@pytest.fixture(scope="module")
def matches():
    return [TrainMatch.from_bundle(generate(seed=s, config=SMALL), SPEC) for s in (7, 8)]


@pytest.fixture(scope="module")
def trained(matches, tmp_path_factory):
    """One tiny training run on the CPU: its output directory and the monitor calls it made."""
    out = tmp_path_factory.mktemp("fold")
    calls = []

    def monitor(model):
        calls.append(count_parameters(model))
        return {"probe": 0.5}

    lines = []
    final = train(
        _run(),
        matches,
        out,
        device="cpu",
        log=lines.append,
        resume=False,
        threads=1,
        meta={"holdout": "none"},
        monitor=monitor,
    )
    return {"out": out, "final": final, "calls": calls, "log": lines}


def test_training_writes_a_loadable_checkpoint_and_a_learning_log(trained):
    out, final = trained["out"], trained["final"]
    assert final == out / "final.pt" and (out / "last.pt").is_file()
    assert not list(out.glob("*.part"))
    payload = load_payload(final)
    assert payload["meta"]["holdout"] == "none" and payload["meta"]["torch"]
    history = payload["meta"]["history"]
    assert [row["set"] for row in history] == [0, 1] and history[-1]["step"] == 6
    assert all(np.isfinite(row["loss"]) and row["monitor_probe"] == 0.5 for row in history)
    assert len(trained["calls"]) == 2 and any("monitor_probe" in line for line in trained["log"])
    loaded = load_checkpoint(final)
    assert loaded.spec == SPEC and loaded.model.cfg == TINY and not loaded.model.training
    assert loaded.sha256 == sha256_file(final)


def test_training_with_a_seed_is_reproducible_and_resumable(matches, trained, tmp_path):
    again = train(
        _run(),
        matches,
        tmp_path / "again",
        device="cpu",
        log=lambda _: None,
        resume=False,
        threads=1,
    )
    one, two = load_payload(trained["final"])["state_dict"], load_payload(again)["state_dict"]
    assert all(torch.equal(one[k], two[k]) for k in one)
    # a run that finds last.pt at the end of its schedule resumes there and only rewrites final.pt
    lines = []
    resumed = train(_run(), matches, trained["out"], device="cpu", log=lines.append, threads=1)
    assert any("resumed" in line for line in lines)
    three = load_payload(resumed)["state_dict"]
    assert all(torch.equal(one[k], three[k]) for k in one)
    # interrupted after the first of two sets, then resumed: the same weights as one go
    seen = []

    def interrupt_in_the_second_set(model):
        seen.append(1)
        if len(seen) == 2:
            raise RuntimeError("interrupted")
        return {}

    split = tmp_path / "split"
    with pytest.raises(RuntimeError, match="interrupted"):
        train(
            _run(),
            matches,
            split,
            device="cpu",
            log=lambda _: None,
            threads=1,
            resume=False,
            monitor=interrupt_in_the_second_set,
        )
    assert torch.load(split / "last.pt", weights_only=True)["set_index"] == 0
    assert not (split / "final.pt").exists()
    final = train(_run(), matches, split, device="cpu", log=lambda _: None, threads=1)
    four = load_payload(final)["state_dict"]
    assert all(torch.equal(one[k], four[k]) for k in one)


def test_a_training_window_must_leave_an_interior(matches, tmp_path):
    bad = replace(_run(), train=replace(_run().train, window=14))
    with pytest.raises(ValueError, match="no interior"):
        train(bad, matches, tmp_path, device="cpu", log=lambda _: None, threads=1)


def test_the_run_config_round_trips_through_yaml(tmp_path):
    import yaml

    from aaaag.paths import CONFIGS_DIR

    run = RunConfig.from_yaml(CONFIGS_DIR / "ml" / "team_baseline.yaml")
    assert run.model.receptive_half == 127 and run.train.degrade.ladders[0] is None
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(run.to_dict()), encoding="utf-8")
    assert RunConfig.from_yaml(path) == run
    assert count_parameters(TeamNet(run.model)) < 1_500_000
    with pytest.raises(ValueError, match="unknown run config keys"):
        RunConfig.from_dict({"models": {}})


def test_the_schedule_warms_up_and_decays():
    cfg = TrainConfig(steps=100, warmup_steps=10, lr=1.0, final_lr_frac=0.1)
    rates = [lr_at(s, cfg) for s in range(100)]
    assert rates[0] < rates[9] <= 1.0 and rates[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(rates[10:], rates[11:], strict=False))
    assert rates[-1] == pytest.approx(0.1, abs=0.01)


def test_augmentation_swaps_labels_with_the_teams_and_mirrors_the_ball(small_bundle):
    tokens = torch.from_numpy(
        np.stack([_tokens(small_bundle, 1000 + 50 * i, 30) for i in range(8)])
    )
    labels = torch.from_numpy(np.random.default_rng(0).integers(0, 3, (8, 30)))
    ball = torch.from_numpy(np.random.default_rng(1).normal(size=(8, 30, 2)).astype(np.float32))
    batch = {"tokens": tokens.clone(), "labels": labels.clone(), "ball": ball.clone()}
    cfg = TrainConfig(flip_y=False, swap_teams=True)
    out = augment(batch, SPEC, cfg, np.random.default_rng(11))
    rows = np.flatnonzero(np.random.default_rng(11).random(8) < 0.5)
    assert 0 < len(rows) < 8
    for i in range(8):
        if i in rows:
            assert torch.equal(out["ball"][i], -ball[i])
            assert torch.equal(
                out["labels"][i],
                torch.where(labels[i] == 0, 1, torch.where(labels[i] == 1, 0, labels[i])),
            )
            assert out["tokens"][i, :, :11, Col.IS_AWAY].sum() == 0
            assert torch.equal(out["tokens"][i, :, 22, Col.X], -tokens[i, :, 22, Col.X])
        else:
            assert torch.equal(out["tokens"][i], tokens[i]) and torch.equal(
                out["labels"][i], labels[i]
            )


def test_the_loss_ignores_edges_unseen_ball_targets_and_the_x_label():
    b, w, margin = 2, 40, 5
    logits = torch.zeros(b, w, 3)
    logits[..., 0] = 4.0  # confident home
    out = {"logits": logits, "ball": torch.zeros(b, w, 2)}
    labels = torch.zeros(b, w, dtype=torch.long)
    labels[:, :margin] = 1  # wrong, but outside the interior
    labels[0, 20] = 3  # X: ignored
    batch = {
        "tokens": torch.zeros(b, w, 1, 1),
        "labels": labels,
        "ball": torch.ones(b, w, 2),
        "ball_ok": torch.zeros(b, w, dtype=torch.bool),
    }
    terms = loss_terms(out, batch, torch.ones(3), margin, aux_weight=1.0)
    assert terms["acc"] == pytest.approx(1.0) and terms["aux"] == 0.0  # no hidden-ball frames
    batch["ball_ok"][:, 10:12] = True
    assert loss_terms(out, batch, torch.ones(3), margin, aux_weight=1.0)["aux"] > 0


# -- the producer ---------------------------------------------------------------------------------


def test_the_producer_gives_team_evidence_that_honours_the_contract(small_bundle, trained):
    weights = str(trained["final"])
    n = small_bundle.meta.n_frames
    evidence = learned_team(small_bundle, weights=weights, sha256=sha256_file(weights)).validate(n)
    played = small_bundle.meta.played_mask()
    assert isinstance(evidence, Evidence) and np.isnan(evidence.proba[~played]).all()
    assert not np.isnan(evidence.proba[played]).any()
    assert (evidence.proba[played, 2] == 0).all()  # speaks to the team factor only
    again = learned_team(small_bundle, weights=weights)
    assert np.array_equal(evidence.proba, again.proba, equal_nan=True)  # bit-exact, one thread
    joint = learned_team(small_bundle, weights=weights, output="joint").validate(n)
    assert (joint.proba[played, 2] > 0).all()
    two = learned_team(small_bundle, weights=[weights, weights], sha256=[sha256_file(weights)] * 2)
    assert np.allclose(two.proba[played], evidence.proba[played], atol=1e-6)
    with pytest.raises(ValueError, match="pinned"):
        learned_team(small_bundle, weights=weights, sha256="f" * 16)


def test_a_checkpoint_from_other_features_is_refused(trained, tmp_path):
    payload = load_payload(trained["final"])
    payload["feature_version"] = "0"
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="feature version"):
        load_checkpoint(path)
    bad = load_payload(trained["final"])
    bad["format"] = 99
    torch.save(bad, path)
    with pytest.raises(ValueError, match="checkpoint format"):
        load_checkpoint(path)


def test_the_champion_pipeline_runs_with_the_learned_team_factor(small_bundle, trained):
    from aaaag.ml.evaluate import champion_config, learned_config

    config = learned_config(champion_config(), [str(trained["final"])])
    assert isinstance(config, PipelineConfig) and config.evidence[0].name == "learned_team"
    first = pipeline.run(config, small_bundle, use_cache=False)
    second = pipeline.run(config, small_bundle, use_cache=False)
    validate_labels(first.labels, small_bundle.meta.n_frames)
    assert to_csv_bytes(first.labels) == to_csv_bytes(second.labels)
    team = first.evidence[0].proba
    assert np.isin(first.labels[small_bundle.meta.played_mask()], (HOME, AWAY, 2)).all()
    assert team.shape == (small_bundle.meta.n_frames, 3)


def test_save_checkpoint_is_atomic_and_keeps_extra_state(trained, tmp_path):
    loaded = load_checkpoint(trained["final"])
    path = save_checkpoint(
        tmp_path / "sub" / "m.pt", loaded.model, SPEC, {"a": 1}, extra={"step": 4}
    )
    assert load_payload(path)["step"] == 4 and load_payload(path)["meta"] == {"a": 1}
    assert not list((tmp_path / "sub").glob("*.part"))


# -- another architecture drops in ----------------------------------------------------------------


@dataclass(frozen=True)
class ToyConfig:
    tokens_per_team: int = 11
    width: int = 4

    @property
    def receptive_half(self) -> int:
        return 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> ToyConfig:
        return cls(**raw)


@register_architecture("toy_mean", ToyConfig)
class ToyMean(nn.Module):
    """The mean of the valid tokens of a frame, one linear layer: the smallest possible design."""

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(nn.Linear(14, cfg.width), nn.GELU(), nn.Linear(cfg.width, 3))

    @property
    def receptive_half(self) -> int:
        return self.cfg.receptive_half

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = tokens[..., Col.VALID : Col.VALID + 1]
        mean = (tokens * valid).sum(-2) / valid.sum(-2).clamp(min=1)
        return {"logits": self.net(mean)}  # no ball head: the auxiliary loss is skipped


def test_a_registered_architecture_trains_saves_loads_and_serves_the_producer(
    matches, small_bundle, tmp_path
):
    run = RunConfig.from_dict({"arch": "toy_mean", "model": {"width": 6}, "train": {}})
    assert run.arch == "toy_mean" and run.model == ToyConfig(tokens_per_team=11, width=6)
    run = replace(run, train=replace(_run().train, window=32))
    final = train(run, matches, tmp_path, device="cpu", log=lambda _: None, threads=1)
    loaded = load_checkpoint(final)
    assert isinstance(loaded.model, ToyMean) and loaded.payload["arch"] == "toy_mean"
    assert loaded.model.cfg.width == 6
    evidence = learned_team(small_bundle, weights=str(final)).validate(small_bundle.meta.n_frames)
    assert not np.isnan(evidence.proba[small_bundle.meta.played_mask()]).any()
    assert RunConfig.from_dict(run.to_dict()) == run
    assert get_architecture("team_net")[1] is TeamNet
    with pytest.raises(KeyError, match="no architecture 'nope'"):
        get_architecture("nope")


def test_a_checkpoint_without_an_architecture_name_is_the_baseline(trained, tmp_path):
    payload = load_payload(trained["final"])
    payload.pop("arch")  # the format of the first baseline run
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    assert isinstance(load_checkpoint(path).model, TeamNet)


# -- the driver's reporting -----------------------------------------------------------------------


def test_fold_summary_reports_gains_the_worst_fold_and_the_paired_interval(small_bundle):
    from aaaag.ml import evaluate as ev
    from aaaag.ml.lomo import format_summary, summarise

    tools = ev._bootstrap_tools()
    if tools is None:
        # The public release (#76) does not ship analysis/scripts/, where the bootstrap lives.
        pytest.skip("analysis/scripts/ballstate_idsse_dev.py holds the bootstrap")
    labels = small_bundle.labels
    worse = np.where(np.arange(len(labels)) % 7 == 0, 2, labels).astype(np.int8)
    team = {
        "learned_acc": 0.9,
        "rule_acc": 0.8,
        "learned_acc_where_rule_votes": 0.9,
        "rule_acc_where_rule_votes": 0.85,
    }
    folds, flags = {}, {"champion": {}, "learned": {}}
    for name, learned_labels, champ, new in (("A", labels, 0.5, 0.6), ("B", worse, 0.5, 0.4)):
        folds[name] = {
            "aggregate": {
                "champion": {"spell_f1_mean": champ},
                "learned": {"spell_f1_mean": new},
                "team": team,
            }
        }
        flags["champion"][name] = [tools.spell_flags(labels, worse)]
        flags["learned"][name] = [tools.spell_flags(labels, learned_labels)]
    summary = summarise(folds, flags)
    assert summary["n_folds"] == 2 and summary["folds_improved"] == 1
    assert summary["spell_f1_delta_mean"] == pytest.approx(0.0)
    assert summary["spell_f1_delta_worst_fold"] == pytest.approx(-0.1)
    low, high = summary["spell_f1_delta_bootstrap_5_95"]
    assert low <= high
    text = format_summary({"folds": folds, "summary": summary})
    assert "A: spellF1 0.5000 -> 0.6000" in text and "improved 1/2" in text


def test_the_command_line_names_its_commands(capsys):
    from aaaag.ml.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert all(command in out for command in ("train", "lomo", "infer", "determinism"))
