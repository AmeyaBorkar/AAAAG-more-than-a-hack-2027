"""The PFF raw ball's height (#86, B3): parser plumbing, the cached reader, the split guard and the
stage-0a probe. Synthetic fixtures only; no real match file is read."""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import Period
from tests.test_pff_source import _ball, _frame, _player, _write

TRAIN_ID = "3812"  # index 0 of GAME_IDS: train
OTHER_TRAIN_ID = "3813"
TEST_ID = "3815"  # index 3: test


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "external"
    root.mkdir()
    monkeypatch.setenv("AAAAG_EXTERNAL_DIR", str(root))
    monkeypatch.setenv("AAAAG_CACHE_DIR", str(tmp_path / "cache"))
    import aaaag.sources.pff as pff
    import aaaag.sources.pff_ballz as pff_ballz

    return pff, pff_ballz, root, tmp_path / "cache"


def _linear_frames(n: int, *, period: int = 1, start: int = 0, missing_z=(), doubled=()):
    """x = 10 * i and z = 2 * i on every native frame, so a resampled z is x / 5."""
    frames = []
    for i in range(n):
        k = start + i
        x, z = 10.0 * i, 2.0 * i
        if i in doubled:
            balls = [_ball(x, 0.0, z=z), _ball(x + 1.0, 0.0, z=z)]
        elif i in missing_z:
            balls = [{"visibility": "VISIBLE", "x": x, "y": 0.0, "z": None}]
        else:
            balls = [_ball(x, 0.0, z=z)]
        frames.append(_frame(k, i / 29.97, period=period, home=[_player(9, x, 1.0)], balls=balls))
    return frames


def _game(root, gid, **kwargs):
    frames = _linear_frames(40, **kwargs) + _linear_frames(30, period=2, start=40)
    _write(root, gid, frames)


# -- the parser and the axis ----------------------------------------------------------------------


def test_height_is_resampled_like_the_ball_on_the_bundle_axis(env):
    pff, _ballz, root, _cache = env
    _game(root, TRAIN_ID, missing_z=(10,), doubled=(20,))
    bundle = pff.load_pff(match_id=TRAIN_ID, use_cache=False)
    z, periods = pff.ball_z_axis(TRAIN_ID)
    assert z.dtype == np.float32 and z.shape == (bundle.meta.n_frames,)
    assert periods == bundle.meta.periods
    x = bundle.tracking.ball_xy[:, 0]
    both = np.isfinite(z)
    # Wherever the height exists the ball does, and it is the same interpolation: z = x / 5.
    assert not (both & ~np.isfinite(x)).any()
    assert np.allclose(z[both], x[both] / 5.0, atol=1e-4)
    # The two VISIBLE balls are a gap in both; the missing z is a gap in z only.
    assert (np.isfinite(x) & ~both).any()
    assert both.sum() > 0.8 * len(z)


def test_raw_ball_z_takes_only_the_single_visible_entry():
    from aaaag.sources import pff

    assert pff._raw_ball_z([_ball(1.0, 2.0, z=3.5)]) == 3.5
    assert pff._raw_ball_z([_ball(1.0, 2.0, z=3.5), _ball(1.0, 2.0, z=1.0)]) is None
    assert pff._raw_ball_z([_ball(1.0, 2.0, visibility="ESTIMATED", z=3.5)]) is None
    assert pff._raw_ball_z([{"visibility": "VISIBLE", "x": 1, "y": 2, "z": "nan"}]) is None
    assert pff._raw_ball_z([{"visibility": "VISIBLE", "x": 1, "y": 2, "z": True}]) is None
    assert pff._raw_ball_z(None) is None


def test_the_flag_off_parse_and_the_bundle_cache_are_unchanged(env):
    pff, pff_ballz, root, cache = env
    _game(root, TRAIN_ID)
    path = pff.file_paths(TRAIN_ID)["tracking"]
    off = pff._parse_tracking(path, [], variant="raw", setpiece_of={}, apply_setpiece=False)
    on = pff._parse_tracking(
        path, [], variant="raw", setpiece_of={}, apply_setpiece=False, keep_ball_z=True
    )
    assert off["ball_raw_z"] is None and on["ball_raw_z"] is not None
    for key, value in off.items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == on[key].tobytes(), key
    # The bundle cache name and the parser version do not move; the height has its own folder.
    assert pff.PARSER_VERSION == "2"
    before = pff.load_pff(match_id=TRAIN_ID)
    names_before = sorted(p.name for p in (cache / f"pff-{TRAIN_ID}").iterdir())
    pff_ballz.ball_z(TRAIN_ID, before)
    after = pff.load_pff(match_id=TRAIN_ID)
    assert sorted(p.name for p in (cache / f"pff-{TRAIN_ID}").iterdir()) == names_before
    assert names_before[0].startswith("pff-v2-raw-sequence-none")
    assert before.tracking.ball_xy.tobytes() == after.tracking.ball_xy.tobytes()
    assert before.labels.tobytes() == after.labels.tobytes()
    assert list((cache / "pff_ballz").iterdir())


# -- the cached reader ----------------------------------------------------------------------------


def test_ball_z_caches_on_disk_and_checks_the_axis(env, monkeypatch):
    pff, pff_ballz, root, cache = env
    _game(root, TRAIN_ID)
    _write(root, OTHER_TRAIN_ID, _linear_frames(12))
    bundle = pff.load_pff(match_id=TRAIN_ID, use_cache=False)
    first = pff_ballz.ball_z(TRAIN_ID, bundle)
    files = list((cache / "pff_ballz").glob(f"{TRAIN_ID}_*_v2z1.npz"))
    assert len(files) == 1
    # A fresh process reads the file, not the tracking.
    pff_ballz._MEMO.clear()
    monkeypatch.setattr(pff, "ball_z_axis", lambda *_: pytest.fail("parsed again"))
    again = pff_ballz.ball_z(TRAIN_ID, bundle)
    assert again.tobytes() == first.tobytes()
    other = pff.load_pff(match_id=OTHER_TRAIN_ID, use_cache=False)
    with pytest.raises(ValueError, match="does not match the bundle"):
        pff_ballz.ball_z(TRAIN_ID, other)


def test_a_smoothed_bundle_is_refused(env):
    pff, pff_ballz, root, _cache = env
    frames = _linear_frames(10)
    for row in frames:
        row["ballsSmoothed"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    _write(root, TRAIN_ID, frames)
    smoothed = pff.load_pff(match_id=TRAIN_ID, variant="smoothed", use_cache=False)
    with pytest.raises(ValueError, match="variant 'smoothed'"):
        pff_ballz.ball_z(TRAIN_ID, smoothed)


def test_the_test_split_is_refused_before_any_file_is_read(env, monkeypatch):
    pff, pff_ballz, root, _cache = env
    _game(root, TEST_ID)
    bundle = pff.load_pff(match_id=TEST_ID, use_cache=False)
    monkeypatch.setattr(pff, "file_paths", lambda *_: pytest.fail("a test game was opened"))
    with pytest.raises(ValueError, match="test split") as err:
        pff_ballz.ball_z(TEST_ID, bundle)
    assert pff.SPLIT_RULE in str(err.value)
    with pytest.raises(ValueError, match="test split"):
        pff_ballz.heights(TEST_ID)
    with pytest.raises(ValueError, match="test split"):
        pff_ballz.probe([TEST_ID])
    with pytest.raises(ValueError, match="GAME_IDS"):
        pff_ballz.heights("1")
    assert TEST_ID in pff.game_ids("test") and TRAIN_ID in pff.game_ids("train")


# -- the probe ------------------------------------------------------------------------------------


def test_flight_durations_bridge_only_short_unseen_gaps():
    from aaaag.sources.pff_ballz import flight_durations

    fps = 25
    z = np.full(400, 1.0)
    z[10:20] = 5.0  # 0.4 s flight
    z[40:42] = 5.0  # 0.08 s spike: not a flight
    z[60:66] = 5.0  # 6 + 6 frames above, 5 unseen between: one flight
    z[66:71] = np.nan
    z[71:77] = 5.0
    z[100:106] = 5.0  # 15 unseen frames between: two flights
    z[106:121] = np.nan
    z[121:127] = 5.0
    z[150:156] = 5.0  # a seen dip below 3 m: two flights
    z[156:158] = 2.0
    z[158:164] = 5.0
    z[195:205] = 5.0  # crosses the period boundary at 200: two flights
    periods = (Period(1, 0, 200), Period(2, 200, 400))
    got = flight_durations(z, periods, fps)
    assert len(got) == 8
    assert got[0] == pytest.approx(0.4)
    assert got[1] == pytest.approx((77 - 60) / fps)


def test_jump_pairs_follow_e114():
    from aaaag.sources.pff_ballz import jump_pairs

    fps = 25
    ball = np.zeros((100, 2))
    ball[:, 0] = np.arange(100) * 0.5  # 12.5 m/s: no jump
    ball[30:, 0] += 3.0  # one 3.5 m step: 87.5 m/s
    ball[50:80] = np.nan  # a 1.2 s gap, then a big step: not counted
    ball[80:, 0] += 100.0
    lo, hi = jump_pairs(ball, (Period(1, 0, 100),), fps)
    assert lo.tolist() == [29] and hi.tolist() == [30]
    split = jump_pairs(ball, (Period(1, 0, 30), Period(2, 30, 100)), fps)
    assert len(split[0]) == 0


def _lofted_game(root, gid, *, seconds: float, lofted: bool):
    frames = []
    n = int(seconds * 29.97)
    for i in range(n):
        t = i / 29.97
        z = 8.0 if lofted and 5.0 <= t % 10.0 < 6.0 else 0.1
        frames.append(_frame(i, t, home=[_player(9, 0.0, 0.0)], balls=[_ball(10.0 + t, 0.0, z=z)]))
    _write(root, gid, frames)


def test_probe_prints_aggregates_and_the_declared_decision(env):
    pff, pff_ballz, root, _cache = env
    _lofted_game(root, TRAIN_ID, seconds=30.0, lofted=True)  # 3 one-second flights in 0.5 min
    report = pff_ballz.probe([TRAIN_ID])
    lines = report.strip().splitlines()
    assert lines[0].startswith("pff_ballz probe, train split, aggregates only: games=1 errors=0")
    assert lines[-1].endswith(": GO")
    flights = next(line for line in lines if "flights_z>3m" in line)
    assert "n=3 " in flights
    assert len(lines) < 15

    _lofted_game(root, OTHER_TRAIN_ID, seconds=30.0, lofted=False)
    flat = pff_ballz.probe([OTHER_TRAIN_ID]).strip().splitlines()
    assert flat[-1].endswith(": KILL")


def test_probe_cli_reads_only_the_installed_train_split(env, capsys):
    pff, pff_ballz, root, _cache = env
    _lofted_game(root, TRAIN_ID, seconds=12.0, lofted=True)
    _lofted_game(root, TEST_ID, seconds=12.0, lofted=True)
    pff_ballz.main(["--probe"])
    out = capsys.readouterr().out
    assert "games=1 errors=0" in out
    assert not list((root.parent / "cache" / "pff_ballz").glob(f"{TEST_ID}_*"))
