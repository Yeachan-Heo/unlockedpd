"""Regression coverage for broad axis=1 aggregation fast paths."""

import numpy as np
import pandas as pd
import pandas.testing as tm

import unlockedpd
from unlockedpd._resources import get_last_selected_path


def _frame(rows=1024, cols=512):
    rng = np.random.default_rng(12345)
    return pd.DataFrame(rng.normal(size=(rows, cols)))


def test_axis1_sum_uses_numpy_vectorized_dense_fast_path():
    df = _frame()

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.sum(axis=1)
    result = df.sum(axis=1)

    tm.assert_series_equal(result, expected)
    assert get_last_selected_path() == "numpy_vectorized"


def test_axis1_mean_nan_path_matches_pandas():
    df = _frame()
    df.iloc[::7, ::11] = np.nan

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.mean(axis=1)
    result = df.mean(axis=1)

    tm.assert_series_equal(result, expected)
    assert get_last_selected_path() == "parallel_numba"


def test_axis1_std_uses_bounded_parallel_numba_path():
    df = _frame()

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.std(axis=1)
    result = df.std(axis=1)

    tm.assert_series_equal(result, expected)
    assert get_last_selected_path() == "parallel_numba"
