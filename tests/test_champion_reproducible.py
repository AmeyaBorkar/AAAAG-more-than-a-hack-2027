"""The champion and the fallback give pinned label bytes on a fixed synthetic match (#131).

The organisers' rules, section 3, say a rebuild must reproduce the uploaded file exactly. The
champion runs float64 numpy and BLAS products, and the repro-probe workflow
(`analysis/scripts/repro_probe.py`) found the same bytes on four platforms:
- Linux x86 and Linux ARM;
- macOS ARM, which uses Apple's BLAS;
- Windows.

These hashes were produced on a Windows laptop, and CI checks them on Linux under Python 3.12 and
3.13. A failure means one of two things:
- the platforms disagree, which would break the rebuild;
- a pinned pipeline changed its output.

If the change is deliberate, update its hash in the same PR.

The synthetic match starts at the kick-off. A step that acts only before the opening kick-off,
such as `pre_kickoff_dead` in v7 and v8, therefore leaves this hash unchanged: v5, v7 and v8 share
it. What this test pins is the network, the decoder and the touch and pass rules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from aaaag.pipeline import PipelineConfig, run
from aaaag.registry import SOURCES

REPO = Path(__file__).resolve().parents[1]
MATCH = {"seed": 0, "minutes_per_half": 2.0}  # 10,500 frames
PINNED = {
    "possession_learned_joint_v8": "4f3cb2ea4aa2e9feca68a07c87f8c9992e055bd06e580aa245bc3b36328b2661",
    "possession_learned_joint_v9": "4f3cb2ea4aa2e9feca68a07c87f8c9992e055bd06e580aa245bc3b36328b2661",
    "possession_learned_joint_v10": "99734156925f67c20486706ca7ef296e8237c9178496d79ecc43b4ecd1afe907",
    "possession_learned_joint_v11": "54c367ea4b9ed0da381a5b7268ec9830993ff3f3b5c4be0e2621a59b200e0d4c",
    "possession_learned_joint_v11b": "54c367ea4b9ed0da381a5b7268ec9830993ff3f3b5c4be0e2621a59b200e0d4c",
    "possession_learned_joint_v12": "2313fb2b29c3e7b74be1e1e65e9cc57cab6fc4a4f2f50ac3c0702d2316df2f52",
    "possession_learned_joint_v14": "2a0fc72b6da6382d927f24043ef35aa50f3c64dfee03c34fcb0e5d28cdbf834b",
    "possession_learned_joint_v15": "f8b52143b726d24f6d90b9e874cf3bb1c5a8a108a10d52b54313397f64a69d3d",
    "possession_control_v1_hw12": "d2cd6e291d27a3b20f337867974c88ef90b286f6b2df9c796c0fc44ea555b2be",
}


def _labels_sha256(name: str) -> str:
    config = PipelineConfig.load(REPO / "configs" / "pipelines" / f"{name}.yaml")
    bundle = SOURCES.get("synthetic").fn(**MATCH)
    labels = run(config, bundle, use_cache=False).labels
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(PINNED))
def test_pinned_pipeline_labels(name: str) -> None:
    assert _labels_sha256(name) == PINNED[name], (
        f"{name} changed its labels on the pinned synthetic match. If the change is deliberate, "
        "update PINNED in this file; otherwise two platforms or two builds disagree (#131)."
    )


def test_champion_is_pinned() -> None:
    submission = yaml.safe_load((REPO / "configs" / "submission.yaml").read_text("utf-8"))
    name = Path(submission["pipeline"]).stem
    assert name in PINNED, f"the champion {name} has no pinned hash; add it here (#131)"
