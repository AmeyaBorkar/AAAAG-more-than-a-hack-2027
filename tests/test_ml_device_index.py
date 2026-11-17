"""Row indices for the batch augmentation: the same values on every device, copied without a sync."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aaaag.ml.train import device_index  # noqa: E402


def test_device_index_values_on_cpu() -> None:
    out = device_index(np.flatnonzero(np.array([0, 1, 0, 1, 1]) > 0), torch.device("cpu"))
    assert out.tolist() == [1, 3, 4]
    assert out.dtype == torch.int64


def test_device_index_empty() -> None:
    out = device_index(np.flatnonzero(np.zeros(4) > 0), torch.device("cpu"))
    assert len(out) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_index_cuda_matches_a_blocking_copy() -> None:
    indices = np.flatnonzero(np.random.default_rng(0).random(64) < 0.5)
    fast = device_index(indices, torch.device("cuda"))
    slow = torch.from_numpy(indices).to("cuda")
    assert torch.equal(fast, slow)
