"""Contracts, and every registered component checked against them automatically.

A component added by any lane is picked up here with its registry `example` parameters, so a PR
that breaks a contract fails CI without anyone having to write a new test for it.
"""

import numpy as np
import pytest

from aaaag import contracts
from aaaag.contracts import A, D, Evidence, H, Tracking, X
from aaaag.official import pm
from aaaag.registry import DECODERS, EVIDENCE, FUSION, SOURCES


def test_label_codes_match_the_official_ones():
    assert (contracts.H, contracts.A, contracts.D, contracts.X) == (pm.H, pm.A, pm.D, pm.X)
    assert contracts.LABELS == pm.LABELS


def test_tracking_validates(small_bundle):
    tracking = small_bundle.tracking
    assert isinstance(tracking, Tracking)
    tracking.validate()
    assert tracking.player_xy.dtype == np.float32
    assert len(tracking.fingerprint()) == 64


def test_bundle_with_labels_cannot_be_bar_ath(small_bundle):
    from dataclasses import replace

    meta = replace(small_bundle.meta, source="bar-ath")
    tracking = replace(small_bundle.tracking, meta=meta, _fingerprint=None)
    with pytest.raises(ValueError, match="BAR-ATH"):
        contracts.MatchBundle(tracking=tracking, labels=small_bundle.labels).validate()


def test_evidence_constructors():
    codes = np.array([H, A, D, -1, H], dtype=np.int8)
    ev = Evidence.from_codes("x", codes).validate(5)
    assert np.isnan(ev.proba[3]).all()
    assert ev.proba[1].tolist() == [0.0, 1.0, 0.0]
    joint = Evidence.from_factors("y", np.array([0.2, np.nan]), np.array([0.75, 0.5])).validate(2)
    assert np.allclose(joint.proba[0], [0.6, 0.2, 0.2])
    assert np.isnan(joint.proba[1]).all()


@pytest.mark.parametrize("name", SOURCES.names())
def test_every_source_with_example_params(name):
    component = SOURCES.get(name)
    if not component.example:
        pytest.skip(f"{name} has no offline example (needs data or a download)")
    bundle = component.fn(**component.example).validate()
    assert bundle.labels is not None, "non-competition sources must carry labels"


@pytest.mark.parametrize("name", EVIDENCE.names())
def test_every_evidence_producer_honours_the_contract(name, small_bundle):
    component = EVIDENCE.get(name)
    evidence = component.fn(small_bundle, **component.example)
    assert isinstance(evidence, Evidence)
    evidence.validate(small_bundle.meta.n_frames)


@pytest.mark.parametrize("name", FUSION.names())
def test_every_fusion_rule_honours_the_contract(name, small_bundle):
    component = FUSION.get(name)
    n = small_bundle.meta.n_frames
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    p[::7] = np.nan
    evidence = [Evidence("a", p), Evidence("b", np.roll(p, 3, axis=0))]
    fused = component.fn(evidence, small_bundle.meta, **component.example)
    assert fused.shape == (n, 3) and fused.dtype == np.float32
    Evidence("fused", fused).validate(n)


@pytest.mark.parametrize("name", DECODERS.names())
def test_every_decoder_honours_the_contract(name, small_bundle):
    component = DECODERS.get(name)
    meta = small_bundle.meta
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=meta.n_frames).astype(np.float32)
    p[::11] = np.nan
    labels = component.fn(p, meta, **component.example)
    contracts.validate_labels(labels, meta.n_frames)
    played = meta.played_mask()
    assert (labels[~played] == X).all(), "decoders write X outside the periods by default"
    assert np.isin(labels[played], (H, A, D)).all(), "decoders give an answer on every played frame"
