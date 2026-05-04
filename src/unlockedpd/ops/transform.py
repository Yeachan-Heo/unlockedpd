"""Parallel transform operations using Numba.

This module provides Numba-accelerated transform operations (diff, pct_change, shift)
with SHAPE-ADAPTIVE parallelization - automatically chooses row vs column parallel
based on array dimensions for maximum CPU utilization.
"""
import numpy as np
from numba import njit, prange
import pandas as pd

from .._compat import (
    get_numeric_columns,
    is_all_numeric,
    wrap_result,
    wrap_result_fast,
    ensure_float64,
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


def _normalize_axis(axis) -> int:
    if axis in (0, "index", None):
        return 0
    if axis in (1, "columns"):
        return 1
    raise ValueError(f"No axis named {axis!r}")


def _call_original_dataframe_method(df: pd.DataFrame, method_name: str, *args, **kwargs):
    from .._patch import _PatchRegistry

    original = _PatchRegistry.get_original(pd.DataFrame, method_name)
    if original is None:
        raise TypeError(f"Original pandas DataFrame.{method_name} is unavailable")
    return original(df, *args, **kwargs)



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
                if np.isnan(old_val) or np.isnan(new_val):
                    result[row, col] = np.nan
                elif np.isinf(old_val):
                    # pandas special handling for inf values
                    if np.isinf(new_val):
                        result[row, col] = np.nan  # inf/inf or -inf/-inf
                    else:
                        result[row, col] = -1.0  # inf -> finite
                elif old_val == 0.0:
                    if new_val == 0.0:
                        result[row, col] = np.nan  # 0/0 undefined
                    else:
                        result[row, col] = np.inf if new_val > 0 else -np.inf
                else:
                    result[row, col] = (new_val - old_val) / old_val
    else:
        abs_periods = -periods
        for row in prange(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in prange(n_rows - abs_periods):
            for col in range(n_cols):
                old_val = arr[row + abs_periods, col]
                new_val = arr[row, col]
                if np.isnan(old_val) or np.isnan(new_val):
                    result[row, col] = np.nan
                elif np.isinf(old_val):
                    # pandas special handling for inf values
                    if np.isinf(new_val):
                        result[row, col] = np.nan  # inf/inf or -inf/-inf
                    else:
                        result[row, col] = -1.0  # inf -> finite
                elif old_val == 0.0:
                    if new_val == 0.0:
                        result[row, col] = np.nan  # 0/0 undefined
                    else:
                        result[row, col] = np.inf if new_val > 0 else -np.inf
                else:
                    result[row, col] = (new_val - old_val) / old_val
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
                if np.isnan(old_val) or np.isnan(new_val):
                    result[row, col] = np.nan
                elif np.isinf(old_val):
                    # pandas special handling for inf values
                    if np.isinf(new_val):
                        result[row, col] = np.nan  # inf/inf or -inf/-inf
                    else:
                        result[row, col] = -1.0  # inf -> finite
                elif old_val == 0.0:
                    if new_val == 0.0:
                        result[row, col] = np.nan  # 0/0 undefined
                    else:
                        result[row, col] = np.inf if new_val > 0 else -np.inf
                else:
                    result[row, col] = (new_val - old_val) / old_val
        else:
            abs_periods = -periods
            for row in range(n_rows - abs_periods, n_rows):
                result[row, col] = np.nan
            for row in range(n_rows - abs_periods):
                old_val = arr[row + abs_periods, col]
                new_val = arr[row, col]
                if np.isnan(old_val) or np.isnan(new_val):
                    result[row, col] = np.nan
                elif np.isinf(old_val):
                    # pandas special handling for inf values
                    if np.isinf(new_val):
                        result[row, col] = np.nan  # inf/inf or -inf/-inf
                    else:
                        result[row, col] = -1.0  # inf -> finite
                elif old_val == 0.0:
                    if new_val == 0.0:
                        result[row, col] = np.nan  # 0/0 undefined
                    else:
                        result[row, col] = np.inf if new_val > 0 else -np.inf
                else:
                    result[row, col] = (new_val - old_val) / old_val
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
                if np.isnan(old_val) or np.isnan(new_val):
                    result[row, col] = np.nan
                elif np.isinf(old_val):
                    # pandas special handling for inf values
                    if np.isinf(new_val):
                        result[row, col] = np.nan  # inf/inf or -inf/-inf
                    else:
                        result[row, col] = -1.0  # inf -> finite
                elif old_val == 0.0:
                    if new_val == 0.0:
                        result[row, col] = np.nan  # 0/0 undefined
                    else:
                        result[row, col] = np.inf if new_val > 0 else -np.inf
                else:
                    result[row, col] = (new_val - old_val) / old_val
    else:
        abs_periods = -periods
        for row in range(n_rows - abs_periods, n_rows):
            for col in range(n_cols):
                result[row, col] = np.nan
        for row in range(n_rows - abs_periods):
            for col in range(n_cols):
                old_val = arr[row + abs_periods, col]
                new_val = arr[row, col]
                if np.isnan(old_val) or np.isnan(new_val):
                    result[row, col] = np.nan
                elif np.isinf(old_val):
                    # pandas special handling for inf values
                    if np.isinf(new_val):
                        result[row, col] = np.nan  # inf/inf or -inf/-inf
                    else:
                        result[row, col] = -1.0  # inf -> finite
                elif old_val == 0.0:
                    if new_val == 0.0:
                        result[row, col] = np.nan  # 0/0 undefined
                    else:
                        result[row, col] = np.inf if new_val > 0 else -np.inf
                else:
                    result[row, col] = (new_val - old_val) / old_val
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
    axis = _normalize_axis(axis)
    if axis == 1:
        return _call_original_dataframe_method(
            df, "diff", periods=periods, axis=axis
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

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
    axis_arg = kwargs.pop("axis", 0)
    axis = _normalize_axis(axis_arg)
    if axis == 1:
        return _call_original_dataframe_method(
            df,
            "pct_change",
            periods=periods,
            fill_method=fill_method,
            limit=limit,
            freq=freq,
            axis=axis_arg,
            **kwargs,
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

    # Handle fill_method - match pandas behavior
    # pandas default is 'pad' (forward fill) before computing pct_change
    if fill_method in ('pad', 'ffill'):
        df = df.ffill(axis=axis)
    elif fill_method in ('bfill', 'backfill'):
        df = df.bfill(axis=axis)
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

    axis = _normalize_axis(axis)
    if axis == 1:
        return _call_original_dataframe_method(
            df,
            "shift",
            periods=periods,
            freq=freq,
            axis=axis,
            fill_value=fill_value,
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

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
    from .._config import config

    original_diff = pd.DataFrame.diff
    original_pct_change = pd.DataFrame.pct_change
    original_shift = pd.DataFrame.shift

    def _warn_and_call_original(method_name, original, self, *args, error=None, **kwargs):
        if error is not None and config.warn_on_fallback:
            import warnings

            warnings.warn(
                f"unlockedpd: Falling back to pandas for {method_name}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        return original(self, *args, **kwargs)

    def _patched_diff(self, periods=1, axis=0):
        if not config.enabled or _normalize_axis(axis) == 1:
            return original_diff(self, periods=periods, axis=axis)
        try:
            return optimized_diff(self, periods=periods, axis=axis)
        except Exception as exc:
            return _warn_and_call_original(
                "diff", original_diff, self, periods=periods, axis=axis, error=exc
            )

    def _patched_pct_change(
        self, periods=1, fill_method='pad', limit=None, freq=None, **kwargs
    ):
        axis = kwargs.get("axis", 0)
        if not config.enabled or _normalize_axis(axis) == 1:
            return original_pct_change(
                self,
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
                **kwargs,
            )
        try:
            return optimized_pct_change(
                self,
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
                **kwargs,
            )
        except Exception as exc:
            return _warn_and_call_original(
                "pct_change",
                original_pct_change,
                self,
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
                error=exc,
                **kwargs,
            )

    def _patched_shift(self, periods=1, freq=None, axis=0, fill_value=None, **kwargs):
        if not config.enabled or _normalize_axis(axis) == 1:
            return original_shift(
                self,
                periods=periods,
                freq=freq,
                axis=axis,
                fill_value=fill_value,
                **kwargs,
            )
        try:
            return optimized_shift(
                self, periods=periods, freq=freq, axis=axis, fill_value=fill_value
            )
        except Exception as exc:
            return _warn_and_call_original(
                "shift",
                original_shift,
                self,
                periods=periods,
                freq=freq,
                axis=axis,
                fill_value=fill_value,
                error=exc,
                **kwargs,
            )

    patch(pd.DataFrame, 'diff', _patched_diff, fallback=False)
    patch(pd.DataFrame, 'pct_change', _patched_pct_change, fallback=False)
    patch(pd.DataFrame, 'shift', _patched_shift, fallback=False)


# Backwards compatibility aliases
_diff_2d = _diff_row_parallel
_diff_2d_serial = _diff_serial
_pct_change_2d = _pct_change_row_parallel
_pct_change_2d_serial = _pct_change_serial
_shift_2d = _shift_row_parallel
_shift_2d_serial = _shift_serial
