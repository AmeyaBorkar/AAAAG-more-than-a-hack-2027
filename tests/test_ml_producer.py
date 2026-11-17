"""The registered `learned_team` producer, its config and the fold scoring helpers, without torch.

The repository's normal environment has no torch, so everything here must hold there: the producer
loads, abstains without weights, checks files before it needs the framework, and the pipeline
config swaps it in for the champion's rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import aaaag
from aaaag.contracts import Evidence
from aaaag.ml import evaluate as ev
from aaaag.ml import weights as weight_files
from aaaag.paths import CONFIGS_DIR
from aaaag.pipeline import PipelineConfig
from aaaag.registry import EVIDENCE


def test_learned_team_is_registered_and_abstains_without_weights(small_bundle):
    component = EVIDENCE.get("learned_team")
    assert component.module.endswith("evidence.possession.learned_team")
    evidence = component.fn(small_bundle, **component.example).validate(small_bundle.meta.n_frames)
    assert np.isnan(evidence.proba).all() and evidence.coverage() == 0.0


def test_missing_or_mispinned_weights_fail_loudly_before_torch(small_bundle, tmp_path):
    component = EVIDENCE.get("learned_team")
    with pytest.raises(FileNotFoundError, match="weights not found"):
        component.fn(small_bundle, weights=str(tmp_path / "nope.pt"))
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="pinned"):
        component.fn(small_bundle, weights=str(weights), sha256="0" * 16)
    with pytest.raises(ValueError, match="shorter"):
        component.fn(small_bundle, weights=str(weights), sha256="abc")
    with pytest.raises(ValueError, match="2 sha256 pins for 1"):
        component.fn(small_bundle, weights=str(weights), sha256=["a" * 16, "b" * 16])
    with pytest.raises(ValueError, match="output must be"):
        component.fn(small_bundle, weights=str(weights), output="odds")


def test_a_pin_is_a_prefix_of_the_files_sha256(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"some bytes")
    digest = weight_files.sha256_file(weights)
    weight_files.check_files([weights], [digest])
    weight_files.check_files([weights], [digest[:10].upper()])
    weight_files.check_files([weights], [])  # unpinned is allowed
    assert weight_files.as_list("a") == ["a"] and weight_files.as_list(None) == []
    assert (
        weight_files.resolve("configs/ml").is_dir() and weight_files.resolve(tmp_path) == tmp_path
    )


def test_the_baseline_pipeline_config_swaps_only_the_team_producer():
    champion = ev.champion_config()
    config = PipelineConfig.load(
        CONFIGS_DIR / "pipelines" / "possession_learned_team_baseline.yaml"
    )
    assert [s.name for s in champion.evidence] == ["control_possession", "ballstate"]
    assert [s.name for s in config.evidence] == ["learned_team", "ballstate"]
    assert config.evidence[1] == champion.evidence[1]  # the ball-state producer is untouched
    assert config.fusion == champion.fusion and config.decoder == champion.decoder
    assert config.digest() != champion.digest()


def test_learned_config_pins_the_weights_and_keeps_the_rest(tmp_path):
    weights = tmp_path / "fold.pt"
    weights.write_bytes(b"fold model")
    champion = ev.champion_config()
    config = ev.learned_config(champion, [str(weights)], device="cuda", views=2)
    step = config.evidence[0]
    assert step.name == "learned_team"
    assert step.params["weights"] == [str(weights)]
    assert step.params["sha256"] == [weight_files.sha256_file(weights)]
    assert step.params["device"] == "cuda" and step.params["views"] == 2
    assert config.evidence[1:] == champion.evidence[1:]
    assert config.name == f"{champion.name}+learned_team"
    with pytest.raises(ValueError, match="no 'nearest_player'"):
        ev.learned_config(champion, [str(weights)], step="nearest_player")


def _evidence_from(p_home: list[float]) -> np.ndarray:
    p = np.array(p_home, dtype=np.float32)
    return Evidence.from_factors("x", np.zeros(len(p), np.float32), p).proba


def test_team_accuracy_counts_abstentions_and_splits_by_ball():
    labels = np.array([0, 0, 1, 1, 2, 3, 0, 1], dtype=np.int8)  # H H A A D X H A
    nan = np.nan
    rule = _evidence_from([nan, 1.0, 0.0, 1.0, nan, nan, nan, 0.0])  # abstains twice; one wrong
    learned = _evidence_from([0.9, 0.8, 0.3, 0.6, 0.5, 0.5, 0.2, 0.1])  # wrong on frames 3 and 6
    seen = np.array([1, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
    out = ev.team_accuracy(labels, rule, learned, seen)
    assert out["n_alive"] == 6
    assert out["rule_acc"] == pytest.approx(3 / 6) and out["rule_coverage"] == pytest.approx(4 / 6)
    assert out["learned_acc"] == pytest.approx(4 / 6) and out["learned_coverage"] == 1.0
    assert out["rule_acc_where_rule_votes"] == pytest.approx(3 / 4)
    assert out["learned_acc_where_rule_votes"] == pytest.approx(3 / 4)
    assert out["learned_acc_ball_seen"] == pytest.approx(1.0)  # frames 0, 1, 7
    assert out["learned_acc_ball_missing"] == pytest.approx(1 / 3)  # frames 2, 3, 6
    assert out["rule_acc_ball_seen"] == pytest.approx(2 / 3)
    assert out["rule_acc_ball_missing"] == pytest.approx(1 / 3)


def _code_names(path: Path) -> set[str]:
    """Every name, attribute, keyword and import in a module's code (docstrings and comments are
    not code, so prose may mention the things this test forbids)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.keyword) and node.arg:
            found.add(f"{node.arg}=")
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
    return found


def test_the_harness_code_never_touches_the_exam_flag_or_the_competition_sources():
    """AGENTS.md rules 1-3 as a tripwire: the learned-model code opens no competition data, passes
    no `exam=` flag, and reads no organiser ground truth."""
    root = Path(aaaag.__file__).parent
    files = [*sorted((root / "ml").glob("*.py")), root / "evidence/possession/learned_team.py"]
    forbidden = {"exam=", "data_dir", "outputs_dir", "PUBLIC_GT", "asset_paths", "EXAM_MATCHES"}
    for path in files:
        names = _code_names(path)
        assert not names & forbidden, f"{path.name} references {sorted(names & forbidden)}"
        assert not any("bar_ath" in name for name in names), path.name
