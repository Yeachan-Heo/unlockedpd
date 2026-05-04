"""Parallel transform operations using Numba.

This module provides Numba-accelerated transform operations (diff, pct_change, shift)
with SHAPE-ADAPTIVE parallelization - automatically chooses row vs column parallel
based on array dimensions for maximum CPU utilization.
"""
import os
from typing import Optional

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
import pandas as pd

from .._compat import (
    get_numeric_columns,
    is_all_numeric,
    wrap_result,
    wrap_result_fast,
    ensure_float64,
)
from .._resources import logical_cpu_count, record_dispatch_path, resolve_threadpool_workers
from ._axis1_native import native_axis1_transform

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
AXIS1_TRANSFORM_THREAD_CAP = 8
AXIS1_NATIVE_BYTES_PER_THREAD = 4 * 1024 * 1024
AXIS1_NATIVE_DIFF_SMALL_CAP = 8
AXIS1_NATIVE_DIFF_MEDIUM_CAP = 32
AXIS1_NATIVE_DIFF_LARGE_CAP = 32
AXIS1_NATIVE_PCT_SMALL_CAP = 32
AXIS1_NATIVE_PCT_MEDIUM_CAP = 16
AXIS1_NATIVE_PCT_LARGE_CAP = 32
AXIS1_NATIVE_SMALL_FRAME_BYTES = 64 * 1024 * 1024
AXIS1_NATIVE_MEDIUM_FRAME_BYTES = 512 * 1024 * 1024
AXIS1_NATIVE_AUTO_MIN_BYTES = 256 * 1024 * 1024


def _native_transforms_disabled() -> bool:
    """Return whether the optional native transform path is explicitly off."""

    return os.environ.get("UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_transforms_explicitly_enabled() -> bool:
    """Return whether the optional native transform path is explicitly on."""

    return os.environ.get("UNLOCKEDPD_ENABLE_NATIVE_TRANSFORMS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_transforms_enabled(arr: Optional[np.ndarray] = None) -> bool:
    """Return whether native axis=1 transforms should run for this frame.

    The native pthread kernels create and join their workers inside the call, so
    they do not leave an unbounded thread leak behind.  They are still promoted
    only for large frames by default because smaller 32MB transform cases are
    noisy and can be faster on the existing Numba/pandas paths.  The environment
    variables remain an explicit override surface for experiments and support:
    ``UNLOCKEDPD_ENABLE_NATIVE_TRANSFORMS=1`` forces the native attempt, while
    ``UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS=1`` always disables it.
    """

    if _native_transforms_disabled():
        return False
    if _native_transforms_explicitly_enabled():
        return True
    return (
        arr is not None
        and int(getattr(arr, "nbytes", 0)) >= AXIS1_NATIVE_AUTO_MIN_BYTES
    )


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


def _pct_change_axis1_primitives(
    df: pd.DataFrame,
    periods: int,
    **shift_kwargs,
) -> pd.DataFrame:
    """Fast all-numeric pct_change(axis=1, fill_method=None) via pandas kernels."""

    shifted = _call_original_dataframe_method(
        df,
        "shift",
        periods=periods,
        axis=1,
        **shift_kwargs,
    )
    record_dispatch_path("pandas_primitives")
    return df / shifted - 1.0


def _real_numeric_values(df: pd.DataFrame):
    """Return real numeric values suitable for float64 transform kernels."""

    if not is_all_numeric(df):
        return None
    arr = df.values
    if arr.dtype.kind not in "biuf":
        return None
    return ensure_float64(arr)


def _bounded_axis1_transform(kernel, arr: np.ndarray, periods: int) -> np.ndarray:
    """Run row-parallel axis=1 transform kernels with bounded CPU fan-out."""

    from .._config import config

    configured = config.num_threads
    target_threads = (
        min(configured, AXIS1_TRANSFORM_THREAD_CAP)
        if configured > 0
        else resolve_threadpool_workers(
            arr.shape[0],
            operation="transform",
            operation_cap=AXIS1_TRANSFORM_THREAD_CAP,
            memory_bandwidth_cap=AXIS1_TRANSFORM_THREAD_CAP,
            cap=AXIS1_TRANSFORM_THREAD_CAP,
        )
    )
    target_threads = max(1, min(int(target_threads), arr.shape[0]))
    current_threads = get_num_threads()

    record_dispatch_path("parallel_numba")
    if target_threads != current_threads:
        set_num_threads(target_threads)
    return kernel(arr, periods)


def _axis1_native_operation_cap(arr: np.ndarray, op: str) -> int:
    """Return a machine- and size-aware cap for ephemeral native workers.

    Native axis=1 transforms are memory-bandwidth-bound.  More cores help only
    until row chunks become too small or the memory subsystem saturates, so the
    cap scales with frame bytes but remains below the machine's logical CPU
    count.  The per-operation ceilings reflect measured saturation points on
    the 384-logical-CPU benchmark host: large ``diff`` benefits from a wider
    32-worker burst under machine load, while ``pct_change`` clears the large
    target with a smaller but still doubled 16-worker cap at 256MB.
    """

    nbytes = int(getattr(arr, "nbytes", 0))
    if nbytes <= 0:
        return 1

    size_cap = max(
        1,
        (nbytes + AXIS1_NATIVE_BYTES_PER_THREAD - 1)
        // AXIS1_NATIVE_BYTES_PER_THREAD,
    )
    if op == "diff":
        operation_cap = (
            AXIS1_NATIVE_DIFF_SMALL_CAP
            if nbytes < AXIS1_NATIVE_SMALL_FRAME_BYTES
            else AXIS1_NATIVE_DIFF_MEDIUM_CAP
            if nbytes < AXIS1_NATIVE_MEDIUM_FRAME_BYTES
            else AXIS1_NATIVE_DIFF_LARGE_CAP
        )
    else:
        operation_cap = (
            AXIS1_NATIVE_PCT_SMALL_CAP
            if nbytes < AXIS1_NATIVE_MEDIUM_FRAME_BYTES
            else AXIS1_NATIVE_PCT_MEDIUM_CAP
            if nbytes < 2 * AXIS1_NATIVE_MEDIUM_FRAME_BYTES
            else AXIS1_NATIVE_PCT_LARGE_CAP
        )

    return max(
        1,
        min(int(size_cap), int(operation_cap), int(logical_cpu_count()), arr.shape[0]),
    )


def _axis1_native_transform(arr: np.ndarray, periods: int, op: str):
    """Run optional native axis=1 kernels within an adaptive thread cap."""

    thread_cap = _axis1_native_operation_cap(arr, op)
    threads = resolve_threadpool_workers(
        arr.shape[0],
        operation="transform",
        operation_cap=thread_cap,
        memory_bandwidth_cap=thread_cap,
        cap=thread_cap,
    )
    result = native_axis1_transform(arr, periods, op=op, threads=threads)
    if result is not None:
        record_dispatch_path("native_c")
    return result


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


@njit(parallel=True, fastmath=True, cache=True)
def _diff_axis1_parallel(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute diff(axis=1) across contiguous rows."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        if periods >= n_cols:
            for row in prange(n_rows):
                for col in range(n_cols):
                    result[row, col] = np.nan
        else:
            for row in prange(n_rows):
                for col in range(periods):
                    result[row, col] = np.nan
                for col in range(periods, n_cols):
                    result[row, col] = arr[row, col] - arr[row, col - periods]
    else:
        abs_periods = -periods
        if abs_periods >= n_cols:
            for row in prange(n_rows):
                for col in range(n_cols):
                    result[row, col] = np.nan
        else:
            for row in prange(n_rows):
                for col in range(n_cols - abs_periods):
                    result[row, col] = arr[row, col] - arr[row, col + abs_periods]
                for col in range(n_cols - abs_periods, n_cols):
                    result[row, col] = np.nan
    return result


@njit(parallel=True, fastmath=True, cache=True)
def _pct_change_axis1_no_fill_parallel(
    arr: np.ndarray, periods: int = 1
) -> np.ndarray:
    """Compute pct_change(axis=1, fill_method=None) across contiguous rows."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    if periods > 0:
        if periods >= n_cols:
            for row in prange(n_rows):
                for col in range(n_cols):
                    result[row, col] = np.nan
        else:
            for row in prange(n_rows):
                for col in range(periods):
                    result[row, col] = np.nan
                for col in range(periods, n_cols):
                    result[row, col] = arr[row, col] / arr[row, col - periods] - 1.0
    else:
        abs_periods = -periods
        if abs_periods >= n_cols:
            for row in prange(n_rows):
                for col in range(n_cols):
                    result[row, col] = np.nan
        else:
            for row in prange(n_rows):
                for col in range(n_cols - abs_periods):
                    result[row, col] = arr[row, col] / arr[row, col + abs_periods] - 1.0
                for col in range(n_cols - abs_periods, n_cols):
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
        if isinstance(periods, int) and periods != 0:
            arr = _real_numeric_values(df)
            if (
                arr is not None
                and arr.size >= PARALLEL_THRESHOLD
                and arr.shape[0] >= MIN_ROWS_FOR_PARALLEL
            ):
                if _native_transforms_enabled(arr):
                    result = _axis1_native_transform(arr, periods, "diff")
                    if result is not None:
                        return wrap_result_fast(result, df)
                result = _bounded_axis1_transform(_diff_axis1_parallel, arr, periods)
                return wrap_result_fast(result, df)
            shifted = _call_original_dataframe_method(
                df, "shift", periods=periods, axis=axis
            )
            record_dispatch_path("pandas_primitives")
            return df - shifted
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
        if fill_method is None and isinstance(periods, int):
            arr = _real_numeric_values(df)
            if (
                arr is not None
                and periods != 0
                and not kwargs
                and arr.size >= PARALLEL_THRESHOLD
                and arr.shape[0] >= MIN_ROWS_FOR_PARALLEL
            ):
                if _native_transforms_enabled(arr):
                    result = _axis1_native_transform(arr, periods, "pct")
                    if result is not None:
                        return wrap_result_fast(result, df)
                result = _bounded_axis1_transform(
                    _pct_change_axis1_no_fill_parallel, arr, periods
                )
                return wrap_result_fast(result, df)
            if arr is not None:
                return _pct_change_axis1_primitives(df, periods, **kwargs)
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
        if not config.enabled:
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
        if not config.enabled:
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
        axis_number = _normalize_axis(axis)
        if not config.enabled:
            return original_shift(
                self,
                periods=periods,
                freq=freq,
                axis=axis,
                fill_value=fill_value,
                **kwargs,
            )
        if axis_number == 1:
            record_dispatch_path("pandas_native")
            if fill_value is None:
                return original_shift(
                    self,
                    periods=periods,
                    freq=freq,
                    axis=axis,
                    **kwargs,
                )
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
