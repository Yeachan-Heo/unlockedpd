"""Parallel transform operations using Numba.

This module provides Numba-accelerated transform operations (diff, pct_change, shift)
with SHAPE-ADAPTIVE parallelization - automatically chooses row vs column parallel
based on array dimensions for maximum CPU utilization.
"""
import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Any

from .._compat import (
    get_numeric_columns, get_numeric_columns_fast, is_all_numeric,
    wrap_result, wrap_result_fast, ensure_float64,
    get_optimal_parallel_axis, prepare_array_for_parallel
)

# Threshold for parallel vs serial execution (elements)
# Parallel overhead is ~1-2ms, so we need enough work to amortize it.
# Testing shows crossover around 1-8M elements for narrow arrays.
# Use 500K as a reasonable default - serial is still very fast.
PARALLEL_THRESHOLD = 500_000

# Minimum rows for row-parallel to be effective
# With fewer rows, parallel overhead dominates. Testing shows:
# - 1000 rows with 64 CPUs = ~16 rows per CPU (borderline)
# - 10000 rows with 64 CPUs = ~156 rows per CPU (good)
MIN_ROWS_FOR_PARALLEL = 2000


# ============================================================================
# ROW-PARALLEL versions (for tall arrays: rows >> cols)
# Memory access: C-contiguous optimal
# ============================================================================

@njit(parallel=True, cache=True)
def _diff_row_parallel(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute diff parallelized across ROWS."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        for row in prange(periods):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in prange(periods, n_rows):
            for col in range(n_cols):
                result[row, col] = arr[row, col] - arr[row - periods, col]
    else:
        abs_periods = -periods
        for row in prange(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in prange(n_rows - abs_periods):
            for col in range(n_cols):
                result[row, col] = arr[row, col] - arr[row + abs_periods, col]
    return result


@njit(parallel=True, cache=True)
def _pct_change_row_parallel(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute pct_change parallelized across ROWS."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        for row in prange(periods):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in prange(periods, n_rows):
            for col in range(n_cols):
                old_val = arr[row - periods, col]
                new_val = arr[row, col]
                if old_val != 0.0 and not np.isnan(old_val) and not np.isnan(new_val):
                    result[row, col] = (new_val - old_val) / old_val
                else:
                    result[row, col] = np.nan
    else:
        abs_periods = -periods
        for row in prange(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in prange(n_rows - abs_periods):
            for col in range(n_cols):
                old_val = arr[row + abs_periods, col]
                new_val = arr[row, col]
                if old_val != 0.0 and not np.isnan(old_val) and not np.isnan(new_val):
                    result[row, col] = (new_val - old_val) / old_val
                else:
                    result[row, col] = np.nan
    return result


@njit(parallel=True, cache=True)
def _shift_row_parallel(arr: np.ndarray, periods: int = 1, fill_value: float = np.nan) -> np.ndarray:
    """Compute shift parallelized across ROWS."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        for row in prange(periods):
            for col in range(n_cols):
                result[row, col] = fill_value
        for row in prange(periods, n_rows):
            for col in range(n_cols):
                result[row, col] = arr[row - periods, col]
    elif periods < 0:
        abs_periods = -periods
        for row in prange(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = fill_value
        for row in prange(n_rows - abs_periods):
            for col in range(n_cols):
                result[row, col] = arr[row + abs_periods, col]
    else:
        for row in prange(n_rows):
            for col in range(n_cols):
                result[row, col] = arr[row, col]
    return result


# ============================================================================
# COLUMN-PARALLEL versions (for wide arrays: cols >> rows)
# Memory access: F-contiguous optimal
# ============================================================================

@njit(parallel=True, cache=True)
def _diff_col_parallel(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute diff parallelized across COLUMNS."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        if periods > 0:
            for row in range(periods):
                result[row, col] = np.nan
            for row in range(periods, n_rows):
                result[row, col] = arr[row, col] - arr[row - periods, col]
        else:
            abs_periods = -periods
            for row in range(n_rows - abs_periods, n_rows):
                result[row, col] = np.nan
            for row in range(n_rows - abs_periods):
                result[row, col] = arr[row, col] - arr[row + abs_periods, col]
    return result


@njit(parallel=True, cache=True)
def _pct_change_col_parallel(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute pct_change parallelized across COLUMNS."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        if periods > 0:
            for row in range(periods):
                result[row, col] = np.nan
            for row in range(periods, n_rows):
                old_val = arr[row - periods, col]
                new_val = arr[row, col]
                if old_val != 0.0 and not np.isnan(old_val) and not np.isnan(new_val):
                    result[row, col] = (new_val - old_val) / old_val
                else:
                    result[row, col] = np.nan
        else:
            abs_periods = -periods
            for row in range(n_rows - abs_periods, n_rows):
                result[row, col] = np.nan
            for row in range(n_rows - abs_periods):
                old_val = arr[row + abs_periods, col]
                new_val = arr[row, col]
                if old_val != 0.0 and not np.isnan(old_val) and not np.isnan(new_val):
                    result[row, col] = (new_val - old_val) / old_val
                else:
                    result[row, col] = np.nan
    return result


@njit(parallel=True, cache=True)
def _shift_col_parallel(arr: np.ndarray, periods: int = 1, fill_value: float = np.nan) -> np.ndarray:
    """Compute shift parallelized across COLUMNS."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        if periods > 0:
            for row in range(periods):
                result[row, col] = fill_value
            for row in range(periods, n_rows):
                result[row, col] = arr[row - periods, col]
        elif periods < 0:
            abs_periods = -periods
            for row in range(n_rows - abs_periods, n_rows):
                result[row, col] = fill_value
            for row in range(n_rows - abs_periods):
                result[row, col] = arr[row + abs_periods, col]
        else:
            for row in range(n_rows):
                result[row, col] = arr[row, col]
    return result


# ============================================================================
# SERIAL versions (for small arrays below PARALLEL_THRESHOLD)
# ============================================================================

@njit(cache=True)
def _diff_serial(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Serial diff for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        for row in range(periods):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in range(periods, n_rows):
            for col in range(n_cols):
                result[row, col] = arr[row, col] - arr[row - periods, col]
    else:
        abs_periods = -periods
        for row in range(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in range(n_rows - abs_periods):
            for col in range(n_cols):
                result[row, col] = arr[row, col] - arr[row + abs_periods, col]
    return result


@njit(cache=True)
def _pct_change_serial(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Serial pct_change for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        for row in range(periods):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in range(periods, n_rows):
            for col in range(n_cols):
                old_val = arr[row - periods, col]
                new_val = arr[row, col]
                if old_val != 0.0 and not np.isnan(old_val) and not np.isnan(new_val):
                    result[row, col] = (new_val - old_val) / old_val
                else:
                    result[row, col] = np.nan
    else:
        abs_periods = -periods
        for row in range(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in range(n_rows - abs_periods):
            for col in range(n_cols):
                old_val = arr[row + abs_periods, col]
                new_val = arr[row, col]
                if old_val != 0.0 and not np.isnan(old_val) and not np.isnan(new_val):
                    result[row, col] = (new_val - old_val) / old_val
                else:
                    result[row, col] = np.nan
    return result


@njit(cache=True)
def _shift_serial(arr: np.ndarray, periods: int = 1, fill_value: float = np.nan) -> np.ndarray:
    """Serial shift for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        for row in range(periods):
            for col in range(n_cols):
                result[row, col] = fill_value
        for row in range(periods, n_rows):
            for col in range(n_cols):
                result[row, col] = arr[row - periods, col]
    elif periods < 0:
        abs_periods = -periods
        for row in range(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = fill_value
        for row in range(n_rows - abs_periods):
            for col in range(n_cols):
                result[row, col] = arr[row + abs_periods, col]
    else:
        for row in range(n_rows):
            for col in range(n_cols):
                result[row, col] = arr[row, col]
    return result


# ============================================================================
# Shape-adaptive dispatch functions
# ============================================================================

def _diff_dispatch(arr: np.ndarray, periods: int) -> np.ndarray:
    """Dispatch to optimal diff implementation based on array shape.

    For C-contiguous arrays, ALWAYS use row-parallel because:
    1. Row elements are contiguous in memory → excellent cache utilization
    2. No cache line contention between threads
    3. Achieves ~32 GB/s vs ~7 GB/s for column-parallel
    """
    n_rows = arr.shape[0]

    # Use serial for small arrays or insufficient rows for parallelization
    if arr.size < PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
        return _diff_serial(arr, periods)

    # For C-contiguous (row-major) arrays, row-parallel is always faster
    # due to memory access patterns. Column-parallel only makes sense
    # for F-contiguous (column-major) arrays.
    if arr.flags['C_CONTIGUOUS'] or not arr.flags['F_CONTIGUOUS']:
        return _diff_row_parallel(arr, periods)
    else:
        return _diff_col_parallel(arr, periods)


def _pct_change_dispatch(arr: np.ndarray, periods: int) -> np.ndarray:
    """Dispatch to optimal pct_change implementation based on array shape.

    For C-contiguous arrays, ALWAYS use row-parallel for cache efficiency.
    """
    n_rows = arr.shape[0]

    if arr.size < PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
        return _pct_change_serial(arr, periods)

    if arr.flags['C_CONTIGUOUS'] or not arr.flags['F_CONTIGUOUS']:
        return _pct_change_row_parallel(arr, periods)
    else:
        return _pct_change_col_parallel(arr, periods)


def _shift_dispatch(arr: np.ndarray, periods: int, fill_value: float) -> np.ndarray:
    """Dispatch to optimal shift implementation based on array shape.

    For C-contiguous arrays, ALWAYS use row-parallel for cache efficiency.
    """
    n_rows = arr.shape[0]

    if arr.size < PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
        return _shift_serial(arr, periods, fill_value)

    if arr.flags['C_CONTIGUOUS'] or not arr.flags['F_CONTIGUOUS']:
        return _shift_row_parallel(arr, periods, fill_value)
    else:
        return _shift_col_parallel(arr, periods, fill_value)


# ============================================================================
# Wrapper functions for pandas DataFrame methods
# ============================================================================

def optimized_diff(df, periods=1, axis=0):
    """Optimized diff implementation for DataFrames.

    Uses shape-adaptive parallelization for maximum CPU utilization.
    """
    if axis not in (0, 'index'):
        raise TypeError("Optimization only for axis=0")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Fast path: all-numeric DataFrame (common case)
    if is_all_numeric(df):
        arr = ensure_float64(df.values)
        result = _diff_dispatch(arr, periods)
        return wrap_result_fast(result, df)

    # Slow path: mixed-dtype DataFrame
    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    result = _diff_dispatch(arr, periods)

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def optimized_pct_change(df, periods=1, fill_method='pad', limit=None, freq=None, **kwargs):
    """Optimized pct_change implementation for DataFrames.

    Uses shape-adaptive parallelization for maximum CPU utilization.

    Args:
        df: Input DataFrame
        periods: Periods to shift for forming percent change (default 1)
        fill_method: How to handle NAs before computing percent changes.
            - 'pad'/'ffill': Forward fill NaN values (pandas default, matches pandas behavior)
            - 'bfill'/'backfill': Backward fill NaN values
            - None: Don't fill NaN values (NaN in input = NaN in output)
        limit: Not supported (raises ValueError)
        freq: Not supported (raises ValueError)

    Returns:
        DataFrame with percentage changes

    Note:
        pandas is deprecating fill_method='pad' as default in future versions.
        We maintain 'pad' as default to match current pandas behavior.
    """
    if limit is not None:
        raise ValueError("limit is not supported in optimized pct_change")
    if freq is not None:
        raise ValueError("freq is not supported in optimized pct_change")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle fill_method - match pandas behavior
    # pandas default is 'pad' (forward fill) before computing pct_change
    if fill_method in ('pad', 'ffill'):
        df = df.ffill()
    elif fill_method in ('bfill', 'backfill'):
        df = df.bfill()
    elif fill_method is not None:
        raise ValueError(f"fill_method must be 'pad', 'ffill', 'bfill', 'backfill', or None, got {fill_method!r}")
    # fill_method=None: don't fill, compute pct_change with NaNs as-is

    # Fast path: all-numeric DataFrame (common case)
    if is_all_numeric(df):
        arr = ensure_float64(df.values)
        result = _pct_change_dispatch(arr, periods)
        return wrap_result_fast(result, df)

    # Slow path: mixed-dtype DataFrame
    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    result = _pct_change_dispatch(arr, periods)

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def optimized_shift(df, periods=1, freq=None, axis=0, fill_value=None):
    """Optimized shift implementation for DataFrames.

    Uses shape-adaptive parallelization for maximum CPU utilization.
    """
    if freq is not None:
        raise ValueError("freq is not supported in optimized shift")

    if axis not in (0, 'index'):
        raise TypeError("Optimization only for axis=0")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    fv = float(fill_value) if fill_value is not None else np.nan

    # Fast path: all-numeric DataFrame (common case)
    if is_all_numeric(df):
        arr = ensure_float64(df.values)
        result = _shift_dispatch(arr, periods, fv)
        return wrap_result_fast(result, df)

    # Slow path: mixed-dtype DataFrame
    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    result = _shift_dispatch(arr, periods, fv)

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def apply_transform_patches():
    """Apply all transform operation patches to pandas."""
    from .._patch import patch

    patch(pd.DataFrame, 'diff', optimized_diff)
    patch(pd.DataFrame, 'pct_change', optimized_pct_change)
    patch(pd.DataFrame, 'shift', optimized_shift)


# Backwards compatibility aliases
_diff_2d = _diff_row_parallel
_diff_2d_serial = _diff_serial
_pct_change_2d = _pct_change_row_parallel
_pct_change_2d_serial = _pct_change_serial
_shift_2d = _shift_row_parallel
_shift_2d_serial = _shift_serial
