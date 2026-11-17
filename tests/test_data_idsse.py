"""The IDSSE adapter on the real files (local only: `pytest -m data`, after fetching).

Assertions are aggregates of OTHER matches (CC BY 4.0), plus exact parity with kloppy's independent
Sportec reader on every frame of every dev match: labels, ball, every player, slot order and teams.
The sealed exam matches are never loaded here. The full parity run takes about 10 minutes.
"""

import numpy as np
import pytest

from aaaag.contracts import A, D, H, X
from aaaag.sources import idsse

pytestmark = pytest.mark.data

MATCH = "J03WMX"


def _require(match_id: str) -> None:
    if not all(path.is_file() for path in idsse.file_paths(match_id).values()):
        pytest.skip(f"IDSSE {match_id} not fetched: python -m aaaag.sources.idsse {match_id}")


@pytest.fixture(scope="module")
def bundle():
    _require(MATCH)
    return idsse.load_idsse(match_id=MATCH).validate()


def test_halves_labels_and_slots(bundle):
    meta = bundle.meta
    assert [p.length for p in meta.periods] == [70_708, 75_259]
    played = meta.played_mask()
    assert int((bundle.labels[played] == D).sum()) == 23_483 + 37_115
    assert bundle.extras["provenance"]["alive_without_owner"] == 0
    assert bundle.tracking.ball_seen()[played].all()
    assert np.bincount(bundle.tracking.slot_team).tolist() == [16, 16]


def _kloppy_arrays(match_id: str, meta) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """kloppy's reading on our frame axis: labels, ball, players (home then away, each by
    PersonId, as the adapter orders slots) and the team of each player column."""
    sportec = pytest.importorskip("kloppy.sportec")
    from kloppy.domain import BallState

    paths = idsse.file_paths(match_id)
    dataset = sportec.load_tracking(
        meta_data=str(paths["meta"]),
        raw_data=str(paths["positions"]),
        only_alive=False,
        coordinates="sportec",
    )
    home, away = dataset.metadata.teams
    by_period: dict[int, list] = {}
    seen_ids: dict[str, set] = {home.team_id: set(), away.team_id: set()}
    for record in dataset.records:
        by_period.setdefault(record.period.id, []).append(record)
        for player, data in record.players_data.items():
            if data.coordinates is not None:
                seen_ids[player.team.team_id].add(player.player_id)
    assert {p: len(r) for p, r in by_period.items()} == {p.index: p.length for p in meta.periods}
    order = sorted(seen_ids[home.team_id]) + sorted(seen_ids[away.team_id])
    column = {player_id: i for i, player_id in enumerate(order)}
    teams = np.array([H] * len(seen_ids[home.team_id]) + [A] * len(seen_ids[away.team_id]))

    labels = np.full(meta.n_frames, X, dtype=np.int8)
    ball = np.full((meta.n_frames, 2), np.nan)
    players = np.full((meta.n_frames, len(order), 2), np.nan)
    for period in meta.periods:
        for k, record in enumerate(by_period[period.index]):
            frame = period.start + k
            if record.ball_coordinates is not None:
                ball[frame] = (record.ball_coordinates.x, record.ball_coordinates.y)
            if record.ball_state != BallState.ALIVE or record.ball_owning_team is None:
                labels[frame] = D
            else:
                labels[frame] = H if record.ball_owning_team == home else A
            for player, data in record.players_data.items():
                if data.coordinates is not None:
                    players[frame, column[player.player_id]] = (
                        data.coordinates.x,
                        data.coordinates.y,
                    )
    return labels, ball, players, teams


@pytest.mark.parametrize("match_id", idsse.DEV_MATCHES)
def test_full_parity_with_kloppy(match_id):
    _require(match_id)
    ours = idsse.load_idsse(match_id=match_id, use_cache=False).validate()
    labels, ball, players, teams = _kloppy_arrays(match_id, ours.meta)
    tracking = ours.tracking
    np.testing.assert_array_equal(ours.labels, labels)
    np.testing.assert_array_equal(tracking.slot_team, teams)
    np.testing.assert_array_equal(np.isnan(tracking.ball_xy), np.isnan(ball))
    np.testing.assert_allclose(tracking.ball_xy, ball, atol=1e-5)  # float32 storage
    assert tracking.player_xy.shape == players.shape
    np.testing.assert_array_equal(np.isnan(tracking.player_xy), np.isnan(players))
    np.testing.assert_allclose(tracking.player_xy, players, atol=1e-5)
