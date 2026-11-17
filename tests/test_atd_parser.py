"""The ATD reader, end to end on files written in the EPTS layout, including the awkward cases."""

import numpy as np
import pytest

from aaaag.contracts import AWAY, HOME, MatchMeta, Period
from aaaag.io import atd
from aaaag.synthetic import SyntheticConfig, generate, write_epts

TINY = SyntheticConfig(minutes_per_half=0.5, gap_minutes=0.1)


@pytest.fixture(scope="module")
def tiny_files(tmp_path_factory):
    bundle = generate(seed=11, config=TINY)
    meta_path, raw_path = write_epts(bundle.tracking, tmp_path_factory.mktemp("epts"))
    return bundle, meta_path, raw_path


def test_round_trip_matches_the_source(tiny_files, tmp_path):
    bundle, meta_path, raw_path = tiny_files
    parsed = atd.load_tracking(meta_path, raw_path, bundle.meta, cache_path=tmp_path / "t.npz")
    source = bundle.tracking
    assert np.array_equal(np.isnan(parsed.player_xy), np.isnan(source.player_xy))
    assert np.nanmax(np.abs(parsed.player_xy - source.player_xy)) < 1e-3
    assert np.nanmax(np.abs(parsed.ball_xy - source.ball_xy)) < 1e-3
    assert np.array_equal(parsed.slot_team, source.slot_team)
    assert parsed.player_speed is not None and "z" in parsed.ball_extra


def test_cache_round_trip_is_identical(tiny_files, tmp_path):
    bundle, meta_path, raw_path = tiny_files
    first = atd.load_tracking(meta_path, raw_path, bundle.meta, cache_path=tmp_path / "c.npz")
    assert (tmp_path / "c.npz").is_file()
    second = atd.load_tracking(meta_path, raw_path, bundle.meta, cache_path=tmp_path / "c.npz")
    assert first.fingerprint() == second.fingerprint()


def test_fast_and_slow_paths_agree(tiny_files):
    _, meta_path, raw_path = tiny_files
    spec = atd.read_metadata(meta_path).specs[0]
    raw = raw_path.read_bytes()
    fast = atd._parse_fast(raw, spec.n_columns)
    slow = atd._parse_slow(raw, spec.n_columns)
    assert fast is not None
    assert np.array_equal(np.isnan(fast), np.isnan(slow))
    assert np.allclose(np.nan_to_num(fast), np.nan_to_num(slow))


def test_layout_is_read_from_the_xml(tiny_files):
    _, meta_path, _ = tiny_files
    facts = atd.describe_layout(meta_path)
    assert facts["n_slots"] == 46 and facts["n_blocks"] == 1
    assert facts["player_channels"] == ["speed", "x", "y"]
    assert facts["slot_order_matches_player_order"]


def test_malformed_line_falls_back_and_then_fails_loudly(tmp_path):
    raw = b"1:0.5,0.5,1.0;0.4,0.4,1.0:0.5,0.5,0.0\n2:0.5,0.5,1.0:0.5,0.5,0.0\n"
    assert atd._parse_fast(raw, 10) is None
    with pytest.raises(ValueError, match="fields"):
        atd._parse_slow(raw, 10)


MULTI_BLOCK_XML = """<?xml version="1.0"?>
<main xmlns:tns="FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS">
 <Metadata>
  <Teams><Team id="TA"><Name>Home</Name></Team><Team id="TB"><Name>Away</Name></Team></Teams>
  <Players>
   <Player id="P1" teamId="TA"/><Player id="P2" teamId="TB"/><Player id="P3" teamId="TA"/>
  </Players>
  <PlayerChannels>
   <PlayerChannel id="player1_x" playerId="P1" channelId="x"/><PlayerChannel id="player1_y" playerId="P1" channelId="y"/>
   <PlayerChannel id="player2_x" playerId="P2" channelId="x"/><PlayerChannel id="player2_y" playerId="P2" channelId="y"/>
   <PlayerChannel id="player3_x" playerId="P3" channelId="x"/><PlayerChannel id="player3_y" playerId="P3" channelId="y"/>
  </PlayerChannels>
 </Metadata>
 <DataFormatSpecifications>
  <DataFormatSpecification startFrame="1" endFrame="2" separator=":">
   <StringRegister name="frameCount"/>
   <SplitRegister separator=";">
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="player1_x"/><PlayerChannelRef playerChannelId="player1_y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="player2_x"/><PlayerChannelRef playerChannelId="player2_y"/></SplitRegister>
   </SplitRegister>
   <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
  </DataFormatSpecification>
  <DataFormatSpecification startFrame="3" endFrame="4" separator=":">
   <StringRegister name="frameCount"/>
   <SplitRegister separator=";">
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="player3_x"/><PlayerChannelRef playerChannelId="player3_y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="player2_x"/><PlayerChannelRef playerChannelId="player2_y"/></SplitRegister>
   </SplitRegister>
   <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
  </DataFormatSpecification>
 </DataFormatSpecifications>
</main>
"""


def test_blocks_with_different_players_land_on_the_right_slots(tmp_path):
    """Metrica sample game 3 changes the player set between blocks; slots must follow players."""
    meta_path = tmp_path / "m_atd.xml"
    raw_path = tmp_path / "m_atd.txt"
    meta_path.write_text(MULTI_BLOCK_XML, encoding="utf-8")
    raw_path.write_bytes(
        b"1:0.10,0.50;0.90,0.50:0.50,0.50\n"
        b"2:0.11,0.50;0.89,0.50:NaN,NaN\n"
        b"3:0.20,0.40;0.80,0.60:0.55,0.50\n"
        b"4:0.21,0.40;0.79,0.60:0.56,0.50\n"
    )
    meta = MatchMeta("M", "test", 4, 25, 100.0, 50.0, (Period(1, 0, 4),))
    tracking = atd.load_tracking(meta_path, raw_path, meta)
    assert tracking.slot_team.tolist() == [HOME, AWAY, HOME]
    assert np.isclose(tracking.player_xy[0, 0, 0], (0.10 - 0.5) * 100)  # P1 in block 1
    assert np.isnan(tracking.player_xy[2, 0, 0])  # P1 absent in block 2
    assert np.isclose(tracking.player_xy[2, 2, 0], (0.20 - 0.5) * 100)  # P3 in block 2, slot 2
    assert np.isclose(tracking.player_xy[3, 1, 1], (0.60 - 0.5) * 50)  # P2 present throughout
    assert np.isnan(tracking.ball_xy[1]).all()
