"""Fold and training-set rules for `lomo --exclude`, without torch."""

from __future__ import annotations

import pytest

from aaaag.ml.lomo import resolve_lomo
from aaaag.sources.idsse import DEV_MATCHES

DEV = tuple(DEV_MATCHES)


def test_no_exclude_is_the_full_roster_trained_on_the_other_four():
    split = resolve_lomo("all")
    assert split.folds == DEV
    assert split.exclude == ()
    for holdout in DEV:
        assert split.train_ids[holdout] == tuple(match for match in DEV if match != holdout)


def test_a_list_of_folds_without_exclude_matches_the_string_all():
    assert resolve_lomo(list(DEV), ()) == resolve_lomo("all", "")


def test_exclude_drops_the_match_from_scoring_and_from_every_training_set():
    held = "J03WPY"
    split = resolve_lomo("all", [held])
    assert held not in split.folds
    assert split.folds == tuple(match for match in DEV if match != held)
    for fold in split.folds:
        assert held not in split.train_ids[fold]
        assert fold not in split.train_ids[fold]
        assert split.train_ids[fold] == tuple(match for match in DEV if match not in {fold, held})


def test_explicit_folds_still_train_on_the_other_dev_matches():
    split = resolve_lomo(["J03WMX", "J03WOH"], ())
    assert split.folds == ("J03WMX", "J03WOH")
    assert split.train_ids["J03WMX"] == tuple(match for match in DEV if match != "J03WMX")
    assert "J03WOH" in split.train_ids["J03WMX"]


def test_inner_and_outer_splits_used_by_the_nested_sweep():
    inner = resolve_lomo(["J03WOH", "J03WOY"], ["J03WMX"])
    assert inner.folds == ("J03WOH", "J03WOY")
    assert inner.train_ids["J03WOH"] == tuple(
        match for match in DEV if match not in {"J03WOH", "J03WMX"}
    )
    assert len(inner.train_ids["J03WOH"]) == 3
    outer = resolve_lomo("J03WMX", "")
    assert outer.folds == ("J03WMX",)
    assert outer.train_ids["J03WMX"] == tuple(match for match in DEV if match != "J03WMX")
    assert len(outer.train_ids["J03WMX"]) == 4


def test_excluded_match_listed_in_folds_is_an_error():
    with pytest.raises(ValueError, match="excluded"):
        resolve_lomo(["J03WMX", "J03WOH"], "J03WOH")
    with pytest.raises(ValueError, match="excluded"):
        resolve_lomo("J03WMX,J03WOH", ["J03WMX"])


def test_unknown_matches_and_an_empty_training_set_are_errors():
    with pytest.raises(ValueError, match="unknown fold"):
        resolve_lomo(["NOPE"], ())
    with pytest.raises(ValueError, match="unknown excluded"):
        resolve_lomo("all", ["NOPE"])
    with pytest.raises(ValueError, match="no training match"):
        resolve_lomo("all", list(DEV[1:]))


def test_comma_separated_exclude_and_a_second_resolve_are_stable():
    split = resolve_lomo(" J03WMX, J03WOH ", " J03WR9, J03WR9 ")
    assert split.folds == ("J03WMX", "J03WOH")
    assert split.exclude == ("J03WR9",)
    assert "J03WR9" not in split.train_ids["J03WMX"]
    again = resolve_lomo(split.folds, split.exclude)
    assert again == split


def test_lomo_help_lists_exclude(capsys):
    from aaaag.ml.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["lomo", "--help"])
    assert stopped.value.code == 0
    assert "--exclude" in capsys.readouterr().out
