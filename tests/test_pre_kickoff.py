"""Dead before the opening kick-off (E55). Synthetic labels only; the anchors are the organisers'."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import A, D, H, MatchBundle, MatchMeta, Period, Tracking, X
from aaaag.official import GAME, games
from aaaag.postprocess import pre_kickoff
from aaaag.postprocess.pre_kickoff import anchor_kicks, dead_before, pre_kickoff_dead
from aaaag.registry import POSTPROCESS


def _meta(n: int, match_id: str = "syn") -> MatchMeta:
    return MatchMeta(
        match_id=match_id,
        source="synthetic",
        n_frames=n,
        fps=10,
        pitch_length=105.0,
        pitch_width=68.0,
        periods=(Period(1, 10, 100), Period(2, 120, 200)),
    )


def _bundle(meta: MatchMeta) -> MatchBundle:
    s = 4
    tracking = Tracking(
        meta=meta,
        player_xy=np.full((meta.n_frames, s, 2), np.nan, dtype=np.float32),
        slot_team=np.array([H, H, A, A], dtype=np.int8),
        ball_xy=np.full((meta.n_frames, 2), np.nan, dtype=np.float32),
    )
    return MatchBundle(tracking=tracking)


def _labels(n: int) -> np.ndarray:
    out = np.full(n, X, dtype=np.int8)
    out[10:100] = H
    out[120:200] = A
    return out


def test_dead_from_the_period_start_up_to_the_kick():
    meta = _meta(210)
    out = dead_before(_labels(210), meta.periods, {1: 10, 2: 150})
    assert (out[120:150] == D).all()
    assert (out[150:200] == A).all()
    assert (out[10:100] == H).all()  # a kick on the period's first frame changes nothing
    assert (out[:10] == X).all() and (out[200:] == X).all()


def test_a_kick_outside_its_period_is_ignored():
    meta = _meta(210)
    labels = _labels(210)
    assert np.array_equal(dead_before(labels, meta.periods, {1: 5, 2: 205}), labels)


def test_a_match_without_anchors_is_unchanged():
    meta = _meta(210)
    labels = _labels(210)
    out = pre_kickoff_dead(labels, _bundle(meta))
    assert out.dtype == np.int8 and np.array_equal(out, labels)


def test_the_anchors_are_the_organisers_kickoffs():
    kicks = anchor_kicks(GAME)
    assert kicks == {
        1: games.anchor_video_frame(games.GAMES[GAME]["kickoff"]),
        2: games.anchor_video_frame(games.GAMES[GAME]["kickoff_2nd"]),
    }
    assert anchor_kicks("no-such-match") == {}


def test_anchors_or_detect_falls_back_to_the_detector(monkeypatch):
    meta = _meta(210)
    monkeypatch.setattr(pre_kickoff, "detected_kicks", lambda _bundle: {2: 140})
    out = pre_kickoff_dead(_labels(210), _bundle(meta), source="anchors_or_detect")
    assert (out[120:140] == D).all() and (out[140:200] == A).all()


def test_an_unknown_source_raises():
    with pytest.raises(ValueError):
        pre_kickoff_dead(_labels(210), _bundle(_meta(210)), source="video")


def test_registered():
    assert POSTPROCESS.get("pre_kickoff_dead").fn is pre_kickoff_dead


def test_cut_first_leaves_no_short_tail_past_the_convention_step():
    """E55b: a possession that straddles the kick leaves a tail; cut first, the tail is a short spell
    the convention step absorbs. Cut last (v7), it would survive."""
    from aaaag.postprocess.min_spell import absorb_short

    meta = _meta(210)
    labels = _labels(210)
    labels[120:155] = H  # a possession from the walk-in, past the kick at 150
    cut_first = absorb_short(dead_before(labels, meta.periods, {2: 150}), meta, 1.5)
    assert (cut_first[120:150] == D).all() and (cut_first[150:200] == A).all()
    cut_last = dead_before(absorb_short(labels, meta, 1.5), meta.periods, {2: 150})
    assert (cut_last[150:155] == H).all()  # the 0.5 s tail the v7 order kept


def test_the_champion_cuts_before_the_convention_step():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs" / "submission.yaml").read_text(encoding="utf-8"))
    steps = yaml.safe_load((root / spec["pipeline"]).read_text(encoding="utf-8"))["postprocess"]
    names = [s["name"] for s in steps]
    if "pre_kickoff_dead" in names:
        assert names[0] == "pre_kickoff_dead", names
