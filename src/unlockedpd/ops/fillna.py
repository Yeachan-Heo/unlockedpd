"""3-tier dispatched missing value fill operations using Numba.

This module provides fill operations (ffill, bfill, fillna) with automatic
3-tier dispatch based on array size:
  - Serial: < 500K elements (simple loop)
  - Parallel: 500K - 10M elements (prange parallelization)
  - ThreadPool: >= 10M elements (nogil kernels with thread pool)

The dispatch automatically selects the optimal execution strategy without
manual configuration.
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Union
from numba import njit, prange
import os

from .._compat import get_numeric_columns_fast, wrap_result, ensure_float64, ensure_optimal_layout

# Element count thresholds for 3-tier dispatch
PARALLEL_THRESHOLD = 500_000      # Min elements for parallel prange
THREADPOOL_THRESHOLD = 10_000_000  # Min elements for ThreadPool + nogil

# Optimal worker count (memory bandwidth limits benefit of more threads)
_CPU_COUNT = os.cpu_count() or 8
THREADPOOL_WORKERS = min(_CPU_COUNT, 32)


# ============================================================================
# Serial kernels for small arrays
# ============================================================================

@njit(cache=True)
def _ffill_serial(arr):
    """Forward fill for small arrays - serial execution."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for c in range(n_cols):
        last_valid = np.nan
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = last_valid
            else:
                last_valid = val
                result[row, c] = val
    return result


@njit(cache=True)
def _bfill_serial(arr):
    """Backward fill for small arrays - serial execution."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for c in range(n_cols):
        last_valid = np.nan
        for row in range(n_rows - 1, -1, -1):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = last_valid
            else:
                last_valid = val
                result[row, c] = val
    return result


@njit(cache=True)
def _fillna_scalar_serial(arr, fill_value):
    """Fill NaN with scalar for small arrays - serial execution."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for c in range(n_cols):
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = fill_value
            else:
                result[row, c] = val
    return result


# ============================================================================
# Parallel prange kernels for medium arrays
# ============================================================================

@njit(parallel=True, cache=True)
def _ffill_parallel(arr):
    """Forward fill using parallel prange over columns."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for c in prange(n_cols):
        last_valid = np.nan
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = last_valid
            else:
                last_valid = val
                result[row, c] = val
    return result


@njit(parallel=True, cache=True)
def _bfill_parallel(arr):
    """Backward fill using parallel prange over columns."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for c in prange(n_cols):
        last_valid = np.nan
        for row in range(n_rows - 1, -1, -1):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = last_valid
            else:
                last_valid = val
                result[row, c] = val
    return result


@njit(parallel=True, cache=True)
def _fillna_scalar_parallel(arr, fill_value):
    """Fill NaN with scalar using parallel prange over columns."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for c in prange(n_cols):
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = fill_value
            else:
                result[row, c] = val
    return result


# ============================================================================
# Nogil kernels for ThreadPool (GIL-released for true parallelism)
# ============================================================================

@njit(nogil=True, cache=True)
def _ffill_nogil_chunk(arr, result, start_col, end_col):
    """Forward fill - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        last_valid = np.nan
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = last_valid
            else:
                last_valid = val
                result[row, c] = val


@njit(nogil=True, cache=True)
def _bfill_nogil_chunk(arr, result, start_col, end_col):
    """Backward fill - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        last_valid = np.nan
        for row in range(n_rows - 1, -1, -1):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = last_valid
            else:
                last_valid = val
                result[row, c] = val


@njit(nogil=True, cache=True)
def _fillna_scalar_nogil_chunk(arr, result, start_col, end_col, fill_value):
    """Fill NaN with scalar - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = fill_value
            else:
                result[row, c] = val


# ============================================================================
# ThreadPool wrapper functions
# ============================================================================

def _ffill_threadpool(arr):
    """ThreadPool-based forward fill using nogil kernels."""
    n_rows, n_cols = arr.shape

    # For wide DataFrames (column-parallel), F-contiguous is optimal
    if n_cols > n_rows:
        arr = ensure_optimal_layout(arr, axis=0)

    result = np.empty_like(arr)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _ffill_nogil_chunk(arr, result, start_col, end_col)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _bfill_threadpool(arr):
    """ThreadPool-based backward fill using nogil kernels."""
    n_rows, n_cols = arr.shape

    # For wide DataFrames (column-parallel), F-contiguous is optimal
    if n_cols > n_rows:
        arr = ensure_optimal_layout(arr, axis=0)

    result = np.empty_like(arr)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _bfill_nogil_chunk(arr, result, start_col, end_col)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _fillna_scalar_threadpool(arr, fill_value):
    """ThreadPool-based fillna with scalar using nogil kernels."""
    n_rows, n_cols = arr.shape

    # For wide DataFrames (column-parallel), F-contiguous is optimal
    if n_cols > n_rows:
        arr = ensure_optimal_layout(arr, axis=0)

    result = np.empty_like(arr)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _fillna_scalar_nogil_chunk(arr, result, start_col, end_col, fill_value)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


# ============================================================================
# Dispatch functions (choose serial/parallel/threadpool based on array size)
# ============================================================================

def _get_dispatch_tier(arr):
    """Determine execution tier based on array size.

    Returns:
        'threadpool': For very large arrays (>= 10M elements)
        'parallel': For medium arrays (>= 500K elements)
        'serial': For small arrays (< 500K elements)
    """
    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        return 'threadpool'
    elif n_elements >= PARALLEL_THRESHOLD:
        return 'parallel'
    else:
        return 'serial'


def _ffill_dispatch(arr):
    """Dispatch ffill to appropriate implementation based on array size."""
    tier = _get_dispatch_tier(arr)
    if tier == 'threadpool':
        return _ffill_threadpool(arr)
    elif tier == 'parallel':
        # For wide DataFrames (column-parallel), F-contiguous is optimal
        n_rows, n_cols = arr.shape
        if n_cols > n_rows:
            arr = ensure_optimal_layout(arr, axis=0)
        return _ffill_parallel(arr)
    else:
        return _ffill_serial(arr)


def _bfill_dispatch(arr):
    """Dispatch bfill to appropriate implementation based on array size."""
    tier = _get_dispatch_tier(arr)
    if tier == 'threadpool':
        return _bfill_threadpool(arr)
    elif tier == 'parallel':
        # For wide DataFrames (column-parallel), F-contiguous is optimal
        n_rows, n_cols = arr.shape
        if n_cols > n_rows:
            arr = ensure_optimal_layout(arr, axis=0)
        return _bfill_parallel(arr)
    else:
        return _bfill_serial(arr)


def _fillna_scalar_dispatch(arr, fill_value):
    """Dispatch fillna to appropriate implementation based on array size."""
    tier = _get_dispatch_tier(arr)
    if tier == 'threadpool':
        return _fillna_scalar_threadpool(arr, fill_value)
    elif tier == 'parallel':
        # For wide DataFrames (column-parallel), F-contiguous is optimal
        n_rows, n_cols = arr.shape
        if n_cols > n_rows:
            arr = ensure_optimal_layout(arr, axis=0)
        return _fillna_scalar_parallel(arr, fill_value)
    else:
        return _fillna_scalar_serial(arr, fill_value)


# ============================================================================
# Wrapper functions for pandas DataFrame methods
# ============================================================================

def optimized_ffill(self, axis=0, inplace=False, limit=None, limit_area=None, downcast=None):
    """Optimized forward fill with 3-tier dispatch (serial/parallel/threadpool)."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Only support axis=0, no inplace, no limit (fall back to pandas for unsupported params)
    if axis != 0 or inplace or limit is not None:
        raise TypeError("Unsupported parameters")

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    tier = _get_dispatch_tier(arr)
    if tier == 'threadpool':
        result = _ffill_threadpool(arr)
    elif tier == 'parallel':
        result = _ffill_parallel(arr)
    else:
        result = _ffill_serial(arr)

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=self
    )


def optimized_bfill(self, axis=0, inplace=False, limit=None, limit_area=None, downcast=None):
    """Optimized backward fill with 3-tier dispatch (serial/parallel/threadpool)."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Only support axis=0, no inplace, no limit (fall back to pandas for unsupported params)
    if axis != 0 or inplace or limit is not None:
        raise TypeError("Unsupported parameters")

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    tier = _get_dispatch_tier(arr)
    if tier == 'threadpool':
        result = _bfill_threadpool(arr)
    elif tier == 'parallel':
        result = _bfill_parallel(arr)
    else:
        result = _bfill_serial(arr)

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=self
    )


def optimized_fillna(self, value=None, method=None, axis=0, inplace=False, limit=None, downcast=None):
    """Optimized fillna with scalar using 3-tier dispatch (serial/parallel/threadpool)."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Only support scalar value, axis=0, no inplace, no limit
    # Fall back to pandas for dict/Series value, method parameter, etc.
    if method is not None or axis != 0 or inplace or limit is not None:
        raise TypeError("Unsupported parameters")

    # Only support scalar fill values
    if value is None or not np.isscalar(value):
        raise TypeError("Only scalar fill values supported")

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    fill_value = float(value)

    tier = _get_dispatch_tier(arr)
    if tier == 'threadpool':
        result = _fillna_scalar_threadpool(arr, fill_value)
    elif tier == 'parallel':
        result = _fillna_scalar_parallel(arr, fill_value)
    else:
        result = _fillna_scalar_serial(arr, fill_value)

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=self
    )


# ============================================================================
# Patch application
# ============================================================================

def apply_fillna_patches():
    """Apply fill operation patches to pandas with 3-tier dispatch.

    Automatically selects execution strategy based on array size:
    - Serial: < 500K elements
    - Parallel (prange): 500K - 10M elements
    - ThreadPool (nogil): >= 10M elements
    """
    from .._patch import patch

    patch(pd.DataFrame, 'ffill', optimized_ffill)
    patch(pd.DataFrame, 'bfill', optimized_bfill)
    patch(pd.DataFrame, 'fillna', optimized_fillna)
