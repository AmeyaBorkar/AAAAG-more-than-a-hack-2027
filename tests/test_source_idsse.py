"""The IDSSE adapter on a tiny hand-written DFL fixture: frame axis, labels, slots, cache, download."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, A, D, H, X
from aaaag.sources import idsse

HOME_ID, AWAY_ID = "DFL-CLU-HOME", "DFL-CLU-AWAY"

MATCH_INFO = f"""<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-TEST" HomeTeamId="{HOME_ID}" GuestTeamId="{AWAY_ID}"
  HomeTeamName="Home FC" GuestTeamName="Away FC" />
<Environment PitchX="105.00" PitchY="68.00" />
</MatchInformation></PutDataRequest>
"""


def _frames(first: int, xs, extra=None) -> str:
    rows = []
    for i, x in enumerate(xs):
        tail = extra(i) if extra else ""
        rows.append(
            f'<Frame N="{first + i}" T="t" X="{x:.2f}" Y="{-x:.2f}" D="0.00" S="36.00" '
            f'A="0.00" M="1"{tail}/>'
        )
    return "\n".join(rows)


def _frameset(section: str, team: str, person: str, body: str) -> str:
    return (
        f'<FrameSet GameSection="{section}" MatchId="DFL-MAT-TEST" TeamId="{team}" '
        f'PersonId="{person}">\n{body}\n</FrameSet>'
    )


# First half: 4 frames (N 10000..10003), ball alive H, H, dead, alive A.
# Second half: 3 frames (N 100000..100002), ball alive with no owner (0), A, H.
# The away substitute P-A2 appears only in the second half's last two frames.
BALL_1 = [(1, 1), (1, 1), (1, 0), (2, 1)]
BALL_2 = [(0, 1), (2, 1), (1, 1)]


def _ball(states):
    return lambda i: f' BallPossession="{states[i][0]}" BallStatus="{states[i][1]}"'


POSITIONS = "\n".join(
    [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<PutDataRequest><Positions>",
        '<MetaData MatchId="DFL-MAT-TEST" Type="pitch-size"><PitchSize X="105.00" Y="68.00" />'
        "</MetaData>",
        _frameset("firstHalf", "referee", "REF", _frames(10000, [9, 9, 9, 9])),
        _frameset("firstHalf", HOME_ID, "P-H1", _frames(10000, [1, 2, 3, 4])),
        _frameset("firstHalf", AWAY_ID, "P-A1", _frames(10000, [-1, -2, -3, -4])),
        _frameset("firstHalf", "BALL", "B", _frames(10000, [0, 1, 2, 3], _ball(BALL_1))),
        _frameset("secondHalf", HOME_ID, "P-H1", _frames(100000, [5, 6, 7])),
        _frameset("secondHalf", AWAY_ID, "P-A1", _frames(100000, [-5, -6, -7])),
        _frameset("secondHalf", AWAY_ID, "P-A2", _frames(100001, [-8, -9])),
        _frameset("secondHalf", "BALL", "B", _frames(100000, [4, 5, 6], _ball(BALL_2))),
        "</Positions></PutDataRequest>",
    ]
)


@pytest.fixture
def files(tmp_path):
    meta, positions = tmp_path / "meta.xml", tmp_path / "positions.xml"
    meta.write_text(MATCH_INFO, encoding="utf-8")
    positions.write_text(POSITIONS, encoding="utf-8")
    return meta, positions


def test_frame_axis_labels_and_slots(files):
    bundle = idsse.load_files("TEST", *files, gap_frames=2).validate()
    meta = bundle.meta
    assert meta.source == "idsse" and meta.fps == 25
    assert (meta.pitch_length, meta.pitch_width) == (105.0, 68.0)
    assert (meta.home, meta.away) == ("Home FC", "Away FC")
    assert [(p.start, p.stop) for p in meta.periods] == [(0, 4), (6, 9)]
    assert meta.n_frames == 9
    # dead -> D; alive with no owner -> D and counted; X in the half-time gap
    assert bundle.labels.tolist() == [H, H, D, A, X, X, D, A, H]
    assert bundle.extras["provenance"]["alive_without_owner"] == 1
    assert bundle.extras["provenance"]["revision"] == idsse.REVISION

    tracking = bundle.tracking
    # referee dropped; slots home first, then away, sorted by PersonId; the substitute has a slot
    assert tracking.slot_team.tolist() == [HOME, AWAY, AWAY]
    np.testing.assert_allclose(tracking.player_xy[:4, 0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(tracking.player_xy[6:9, 1, 1], [5, 6, 7])  # Y as delivered
    assert np.isnan(tracking.player_xy[:7, 2]).all()  # substitute absent until N=100001
    np.testing.assert_allclose(tracking.player_xy[7:9, 2, 0], [-8, -9])
    assert np.isnan(tracking.player_xy[4:6]).all() and np.isnan(tracking.ball_xy[4:6]).all()
    np.testing.assert_allclose(tracking.ball_xy[meta.played_mask(), 0], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(tracking.player_speed[0, 0], 10.0, rtol=1e-6)  # 36 km/h


def test_parse_is_deterministic_and_the_cache_round_trips(files, tmp_path):
    cache = tmp_path / "cache" / "idsse.npz"
    fresh = idsse.load_files("TEST", *files, gap_frames=2)
    written = idsse.load_files("TEST", *files, gap_frames=2, cache_path=cache)
    assert cache.is_file() and cache.with_suffix(".json").is_file()
    cached = idsse.load_files("TEST", *files, gap_frames=2, cache_path=cache)
    assert fresh.fingerprint() == written.fingerprint() == cached.fingerprint()
    assert cached.meta == fresh.meta
    np.testing.assert_array_equal(cached.labels, fresh.labels)
    assert cached.extras == fresh.extras
    # a different gap is a different stamp: re-parsed, not served stale
    regapped = idsse.load_files("TEST", *files, gap_frames=5, cache_path=cache)
    assert regapped.meta.n_frames == 12


def test_count_mismatch_in_a_frameset_is_an_error(files):
    _, positions = files
    text = positions.read_text(encoding="utf-8").replace(' BallStatus="0"', "", 1)
    positions.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="BallStatus"):
        idsse.load_files("TEST", *files)


def test_download_checks_the_hash(tmp_path):
    source = tmp_path / "remote.xml"
    source.write_bytes(b"<x/>")
    good = hashlib.sha256(b"<x/>").hexdigest()
    dest = tmp_path / "local" / "file.xml"
    idsse.download(source.as_uri(), dest, good)
    assert dest.read_bytes() == b"<x/>"

    bad = tmp_path / "local" / "bad.xml"
    with pytest.raises(ValueError, match="sha256"):
        idsse.download(source.as_uri(), bad, "0" * 64)
    assert not bad.exists() and not bad.with_name(bad.name + ".part").exists()


def test_fetch_rehashes_files_already_on_disk(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.setenv("AAAAG_EXTERNAL_DIR", str(tmp_path / "external"))
    monkeypatch.setattr(idsse, "MIRROR_URL", remote.as_uri() + "/{name}")
    content = {"meta": b"<meta/>", "positions": b"<positions/>"}
    shas = {kind: hashlib.sha256(data).hexdigest() for kind, data in content.items()}
    monkeypatch.setitem(idsse.MATCHES, "FAKE", ("000001", shas["meta"], shas["positions"]))
    for kind, name in idsse.file_names("FAKE").items():
        (remote / name).write_bytes(content[kind])

    paths = idsse.fetch("FAKE")
    assert {kind: path.read_bytes() for kind, path in paths.items()} == content
    paths["positions"].write_bytes(b"<stale/>")  # a corrupted or stale local copy
    idsse.fetch("FAKE")
    assert paths["positions"].read_bytes() == content["positions"]
    for name in idsse.file_names("FAKE").values():
        (remote / name).unlink()
    idsse.fetch("FAKE")  # files that match the pin are not downloaded again


def test_manifest_and_missing_files(tmp_path, monkeypatch):
    assert len(idsse.MATCHES) == 7
    for competition, meta_sha, positions_sha in idsse.MATCHES.values():
        assert competition in ("000001", "000002")
        assert all(len(sha) == 64 and int(sha, 16) >= 0 for sha in (meta_sha, positions_sha))
    with pytest.raises(KeyError, match="unknown IDSSE match"):
        idsse.file_names("NOPE")
    monkeypatch.setenv("AAAAG_EXTERNAL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="python -m aaaag.sources.idsse J03WMX"):
        idsse.load_idsse(match_id="J03WMX")
