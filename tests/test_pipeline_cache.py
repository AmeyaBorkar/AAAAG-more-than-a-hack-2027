"""The evidence cache is written atomically, so concurrent jobs never read a half-written file.

On 7 Nov four leave-one-match-out jobs on one rented GPU evaluated the same held-out match; one of
them opened the champion's evidence file while another was still writing it (`EOFError: No data
left in file`) and the run failed. The writer now renames a finished temporary file over the target.
"""

from __future__ import annotations

import numpy as np
import pytest

from aaaag.contracts import Evidence
from aaaag.pipeline import _load_cached_evidence, _save_cached_evidence


def _evidence(n: int = 5, value: float = 1.0 / 3.0) -> Evidence:
    return Evidence("probe", np.full((n, 3), value, np.float32))


def test_save_leaves_only_the_target_and_it_loads(tmp_path):
    path = tmp_path / "probe-key.npz"
    _save_cached_evidence(path, _evidence())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["probe-key.npz"]
    loaded = _load_cached_evidence(path, "probe", 5)
    assert np.array_equal(loaded.proba, _evidence().proba) and loaded.channels == {}


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(tmp_path, monkeypatch):
    path = tmp_path / "probe-key.npz"
    _save_cached_evidence(path, _evidence())
    before = path.read_bytes()

    def broken(file, **arrays):
        with open(file, "wb") as handle:  # a partial archive, then the failure
            handle.write(b"PK\x03\x04")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError, match="disk full"):
        _save_cached_evidence(path, _evidence(value=0.5))
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["probe-key.npz"]


def test_channels_survive_the_atomic_write(tmp_path):
    path = tmp_path / "probe-key.npz"
    channel = np.linspace(-1, 1, 5).astype(np.float32)
    evidence = Evidence("probe", _evidence().proba, {"switch": channel})
    _save_cached_evidence(path, evidence)
    loaded = _load_cached_evidence(path, "probe", 5)
    assert np.array_equal(loaded.channels["switch"], channel)
