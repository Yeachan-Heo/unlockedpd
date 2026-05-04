"""Regression coverage for broad axis=0 cumulative fast paths."""

import numpy as np
import pandas as pd
import pandas.testing as tm

import unlockedpd
from unlockedpd._resources import get_last_selected_path
from unlockedpd.ops import cumulative as cumulative_ops


def _finite_frame(rows=32, cols=32):
    rng = np.random.default_rng(97531)
    return pd.DataFrame(rng.normal(size=(rows, cols)))


def test_axis0_cumulative_row_block_numba_fast_path_matches_pandas(monkeypatch):
    df = _finite_frame()

    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_THRESHOLD", 1)
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_BYTES_PER_THREAD", 1)

    for method in ("cumsum", "cummin", "cummax"):
        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = getattr(df, method)(axis=0)
        result = getattr(df, method)(axis=0)

        tm.assert_frame_equal(result, expected)
        assert get_last_selected_path() == "parallel_numba"


def test_axis0_cumprod_row_block_numba_fast_path_matches_pandas(monkeypatch):
    rng = np.random.default_rng(86420)
    df = pd.DataFrame(rng.uniform(0.99, 1.01, size=(32, 32)))

    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_THRESHOLD", 1)
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_BYTES_PER_THREAD", 1)

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.cumprod(axis=0)
    result = df.cumprod(axis=0)

    tm.assert_frame_equal(result, expected)
    assert get_last_selected_path() == "parallel_numba"


def test_axis0_cumulative_thread_cap_scales_with_frame_size(monkeypatch):
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_BYTES_PER_THREAD", 1)
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_MEDIUM_FRAME_BYTES", 4096)
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_SMALL_THREAD_CAP", 4)
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_LARGE_THREAD_CAP", 12)
    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_THREAD_CAP", 16)

    small = np.empty((8, 32), dtype=np.float64)
    large = np.empty((32, 512), dtype=np.float64)

    assert cumulative_ops._axis0_cumulative_thread_cap(small) == 4
    assert cumulative_ops._axis0_cumulative_thread_cap(large) == 12


def test_axis0_cumulative_nonfinite_uses_existing_safe_path(monkeypatch):
    arr = np.ones((8, 8), dtype=np.float64)
    arr[0, 0] = np.inf

    monkeypatch.setattr(cumulative_ops, "AXIS0_CUMULATIVE_THRESHOLD", 1)

    assert cumulative_ops._axis0_numba_cumulative(arr, "cumsum") is None
