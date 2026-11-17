"""Metrica event labels on hand-made events: dead before restarts, owner from the latest action."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aaaag.contracts import AWAY, HOME, A, D, H, Period, X
from aaaag.sources.metrica import event_labels


def _events(rows):
    return pd.DataFrame(rows, columns=["type", "period", "start", "end", "team", "subtypes"])


def test_restart_dead_stretch_and_owner():
    events = _events(
        [
            ("SET PIECE", 1, 10, 10, HOME, "KICK OFF"),
            ("PASS", 1, 10, 30, HOME, ""),
            ("BALL OUT", 1, 40, 50, HOME, ""),
            ("CARD", 1, 55, 55, AWAY, "YELLOW"),  # bookkeeping: never the event before a restart
            ("SET PIECE", 1, 70, 70, AWAY, "THROW IN"),
            ("RECOVERY", 1, 80, 80, HOME, ""),
        ]
    )
    labels = event_labels(events, (Period(1, 5, 100),), 105)
    assert (labels[:5] == X).all() and (labels[100:] == X).all()
    assert (labels[5:10] == D).all()  # before the period's first event
    assert (labels[10:50] == H).all()
    assert (labels[50:70] == D).all()  # end of BALL OUT to the throw-in
    assert (labels[70:80] == A).all() and (labels[80:100] == H).all()


def test_no_owner_yet_is_dead():
    events = _events([("PASS", 1, 20, 25, AWAY, "")])
    labels = event_labels(events, (Period(1, 0, 40),), 40)
    assert (labels[:20] == D).all() and (labels[20:] == A).all()
    assert np.isin(labels, (H, A, D)).all()
