"""Regression coverage for broad axis=1 DataFrame fast paths."""

import numpy as np
import pandas as pd
import pandas.testing as tm

import unlockedpd
from unlockedpd._resources import get_last_selected_path
from unlockedpd.ops import transform as transform_ops


def _wide_frame(rows=512, cols=512):
    rng = np.random.default_rng(24680)
    return pd.DataFrame(rng.normal(size=(rows, cols)))


def test_axis1_cumulative_uses_numpy_vectorized_dense_fast_path():
    df = _wide_frame()

    for method in ("cumsum", "cummin", "cummax"):
        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = getattr(df, method)(axis=1)
        result = getattr(df, method)(axis=1)

        tm.assert_frame_equal(result, expected)
        assert get_last_selected_path() == "numpy_vectorized"


def test_axis1_cumprod_uses_numpy_vectorized_dense_fast_path():
    rng = np.random.default_rng(13579)
    df = pd.DataFrame(rng.uniform(0.99, 1.01, size=(512, 512)))

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.cumprod(axis=1)
    result = df.cumprod(axis=1)

    tm.assert_frame_equal(result, expected)
    assert get_last_selected_path() == "numpy_vectorized"


def test_axis1_cumulative_skipna_matches_pandas():
    df = _wide_frame(128, 128)
    df.iloc[::7, ::11] = np.nan

    for method in ("cumsum", "cumprod", "cummin", "cummax"):
        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = getattr(df, method)(axis=1)
        result = getattr(df, method)(axis=1)

        tm.assert_frame_equal(result, expected)
        assert get_last_selected_path() == "numpy_vectorized"


def test_axis1_diff_shift_pct_change_match_pandas_native_fallbacks():
    df = _wide_frame()
    df.iloc[::17, ::19] = np.nan

    operations = [
        ("diff", {"axis": 1}),
        ("diff", {"axis": 1, "periods": -2}),
        ("shift", {"axis": 1, "fill_value": -5.5}),
        ("shift", {"axis": 1, "periods": -3}),
        ("pct_change", {"axis": 1, "fill_method": None}),
        ("pct_change", {"axis": "columns", "periods": -2, "fill_method": None}),
    ]

    for method, kwargs in operations:
        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = getattr(df, method)(**kwargs)
        result = getattr(df, method)(**kwargs)

        tm.assert_frame_equal(result, expected)


def test_axis1_pct_change_default_fill_is_columnwise():
    df = pd.DataFrame(
        [
            [1.0, np.nan, 4.0, 8.0],
            [2.0, 0.0, np.nan, 6.0],
            [np.inf, 4.0, 8.0, np.inf],
        ]
    )

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.pct_change(axis=1)
    result = df.pct_change(axis=1)

    tm.assert_frame_equal(result, expected)


def test_axis1_pct_change_fill_none_uses_pandas_primitives_path():
    df = _wide_frame()
    df.iloc[::17, ::19] = np.nan

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.pct_change(axis=1, fill_method=None)
    result = df.pct_change(axis=1, fill_method=None)

    tm.assert_frame_equal(result, expected)
    assert get_last_selected_path() == "pandas_primitives"


def test_axis1_pct_change_parallel_numba_fastpath_matches_pandas(monkeypatch):
    df = _wide_frame(8, 6)
    df.iloc[0, 0] = 0.0
    df.iloc[1, 1] = np.nan
    df.iloc[2, 2] = np.inf

    monkeypatch.setenv("UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS", "1")
    monkeypatch.setattr(transform_ops, "PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr(transform_ops, "MIN_ROWS_FOR_PARALLEL", 1)

    operations = [
        {"axis": 1, "fill_method": None},
        {"axis": 1, "periods": -2, "fill_method": None},
        {"axis": 1, "periods": 8, "fill_method": None},
    ]

    for kwargs in operations:
        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = df.pct_change(**kwargs)
        result = df.pct_change(**kwargs)

        tm.assert_frame_equal(result, expected)
        assert get_last_selected_path() == "parallel_numba"


def test_axis1_native_transform_fastpath_matches_pandas_when_available(monkeypatch):
    df = _wide_frame(8, 6)
    df.iloc[0, 0] = 0.0
    df.iloc[1, 1] = np.nan
    df.iloc[2, 2] = np.inf

    monkeypatch.delenv("UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS", raising=False)
    monkeypatch.setattr(transform_ops, "PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr(transform_ops, "MIN_ROWS_FOR_PARALLEL", 1)

    operations = [
        ("diff", {"axis": 1}),
        ("diff", {"axis": 1, "periods": -2}),
        ("pct_change", {"axis": 1, "fill_method": None}),
        ("pct_change", {"axis": 1, "periods": -2, "fill_method": None}),
    ]

    for method, kwargs in operations:
        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = getattr(df, method)(**kwargs)
        result = getattr(df, method)(**kwargs)

        tm.assert_frame_equal(result, expected)
        assert get_last_selected_path() in {
            "native_c",
            "pandas_primitives",
            "parallel_numba",
        }


def test_axis1_shift_fill_none_uses_no_fill_value_native_path():
    df = _wide_frame()

    with unlockedpd._PatchRegistry.temporarily_unpatched():
        expected = df.shift(axis=1, fill_value=None)
    result = df.shift(axis=1, fill_value=None)

    tm.assert_frame_equal(result, expected)
    assert get_last_selected_path() == "pandas_native"
