"""Parallel element-wise operations using Numba with 3-tier dispatch.

This module provides parallelized element-wise operations (clip, abs, round)
with intelligent dispatch between serial, parallel (prange), and threadpool+nogil
execution based on array size.

Key insight: Row-parallel processing with C-contiguous arrays provides optimal
cache efficiency for element-wise operations.
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import os

from .._compat import get_numeric_columns_fast, wrap_result, ensure_float64

# Thresholds for parallel execution strategy
# Based on benchmarking: 3-tier dispatch for optimal performance
PARALLEL_THRESHOLD = 500000  # Switch from serial to prange
THREADPOOL_THRESHOLD = 10000000  # Switch from prange to threadpool+nogil

# Optimal worker count for threadpool (memory bandwidth limits benefit)
_CPU_COUNT = os.cpu_count() or 8
THREADPOOL_WORKERS = min(_CPU_COUNT, 32)


# ============================================================================
# Serial implementations (for small arrays)
# ============================================================================

@njit(cache=True)
def _clip_serial(arr, lower, upper):
    """Clip values - serial version."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for row in range(n_rows):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            elif val < lower:
                result[row, col] = lower
            elif val > upper:
                result[row, col] = upper
            else:
                result[row, col] = val
    return result


@njit(cache=True)
def _abs_serial(arr):
    """Absolute value - serial version."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for row in range(n_rows):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            else:
                result[row, col] = abs(val)
    return result


@njit(cache=True)
def _round_serial(arr, decimals):
    """Round values - serial version."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    multiplier = 10.0 ** decimals
    for row in range(n_rows):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            else:
                # Round to nearest even (banker's rounding)
                result[row, col] = np.round(val * multiplier) / multiplier
    return result


# ============================================================================
# Parallel prange implementations (for medium arrays)
# ============================================================================

@njit(parallel=True, cache=True)
def _clip_parallel(arr, lower, upper):
    """Clip values - parallel prange version (row-parallel)."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for row in prange(n_rows):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            elif val < lower:
                result[row, col] = lower
            elif val > upper:
                result[row, col] = upper
            else:
                result[row, col] = val
    return result


@njit(parallel=True, cache=True)
def _abs_parallel(arr):
    """Absolute value - parallel prange version (row-parallel)."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    for row in prange(n_rows):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            else:
                result[row, col] = abs(val)
    return result


@njit(parallel=True, cache=True)
def _round_parallel(arr, decimals):
    """Round values - parallel prange version (row-parallel)."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    multiplier = 10.0 ** decimals
    for row in prange(n_rows):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            else:
                # Round to nearest even (banker's rounding)
                result[row, col] = np.round(val * multiplier) / multiplier
    return result


# ============================================================================
# Nogil kernels for ThreadPool (GIL-released for true parallelism)
# ============================================================================

@njit(nogil=True, cache=True)
def _clip_nogil_chunk(arr, result, start_row, end_row, lower, upper):
    """Clip values - GIL released, row-parallel for cache efficiency."""
    n_cols = arr.shape[1]
    for row in range(start_row, end_row):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            elif val < lower:
                result[row, col] = lower
            elif val > upper:
                result[row, col] = upper
            else:
                result[row, col] = val


@njit(nogil=True, cache=True)
def _abs_nogil_chunk(arr, result, start_row, end_row):
    """Absolute value - GIL released."""
    n_cols = arr.shape[1]
    for row in range(start_row, end_row):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            else:
                result[row, col] = abs(val)


@njit(nogil=True, cache=True)
def _round_nogil_chunk(arr, result, start_row, end_row, decimals):
    """Round values - GIL released."""
    n_cols = arr.shape[1]
    multiplier = 10.0 ** decimals
    for row in range(start_row, end_row):
        for col in range(n_cols):
            val = arr[row, col]
            if np.isnan(val):
                result[row, col] = np.nan
            else:
                # Round to nearest even (banker's rounding)
                result[row, col] = np.round(val * multiplier) / multiplier


# ============================================================================
# ThreadPool implementations (for large arrays)
# ============================================================================

def _clip_threadpool(arr, lower, upper):
    """Parallel clip using nogil kernels with thread pool."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_rows + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_row, end_row = args
        _clip_nogil_chunk(arr, result, start_row, end_row, lower, upper)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_rows))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_rows]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _abs_threadpool(arr):
    """Parallel abs using nogil kernels with thread pool."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_rows + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_row, end_row = args
        _abs_nogil_chunk(arr, result, start_row, end_row)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_rows))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_rows]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _round_threadpool(arr, decimals):
    """Parallel round using nogil kernels with thread pool."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_rows + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_row, end_row = args
        _round_nogil_chunk(arr, result, start_row, end_row, decimals)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_rows))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_rows]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


# ============================================================================
# Dispatch functions (3-tier: serial -> prange -> threadpool+nogil)
# ============================================================================

def _clip_dispatch(arr, lower, upper):
    """Dispatch clip to appropriate implementation based on array size."""
    total_elements = arr.size

    if total_elements < PARALLEL_THRESHOLD:
        # Small arrays: serial execution
        return _clip_serial(arr, lower, upper)
    elif total_elements < THREADPOOL_THRESHOLD:
        # Medium arrays: parallel prange
        return _clip_parallel(arr, lower, upper)
    else:
        # Large arrays: threadpool with nogil
        return _clip_threadpool(arr, lower, upper)


def _abs_dispatch(arr):
    """Dispatch abs to appropriate implementation based on array size."""
    total_elements = arr.size

    if total_elements < PARALLEL_THRESHOLD:
        return _abs_serial(arr)
    elif total_elements < THREADPOOL_THRESHOLD:
        return _abs_parallel(arr)
    else:
        return _abs_threadpool(arr)


def _round_dispatch(arr, decimals):
    """Dispatch round to appropriate implementation based on array size."""
    total_elements = arr.size

    if total_elements < PARALLEL_THRESHOLD:
        return _round_serial(arr, decimals)
    elif total_elements < THREADPOOL_THRESHOLD:
        return _round_parallel(arr, decimals)
    else:
        return _round_threadpool(arr, decimals)


# ============================================================================
# Wrapper functions for pandas DataFrame methods
# ============================================================================

def optimized_clip(self, lower=None, upper=None, axis=None, inplace=False, **kwargs):
    """Optimized clip with 3-tier parallel dispatch."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")
    if inplace or axis is not None:
        raise TypeError("Unsupported parameters")

    # Handle None bounds
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    arr = ensure_float64(numeric_df.values)
    result = _clip_dispatch(arr, lower, upper)

    return wrap_result(result, numeric_df, columns=numeric_cols,
                      merge_non_numeric=True, original_df=self)


def optimized_abs(self):
    """Optimized abs with 3-tier parallel dispatch."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    arr = ensure_float64(numeric_df.values)
    result = _abs_dispatch(arr)

    return wrap_result(result, numeric_df, columns=numeric_cols,
                      merge_non_numeric=True, original_df=self)


def optimized_round(self, decimals=0, *args, **kwargs):
    """Optimized round with 3-tier parallel dispatch."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    arr = ensure_float64(numeric_df.values)
    result = _round_dispatch(arr, decimals)

    return wrap_result(result, numeric_df, columns=numeric_cols,
                      merge_non_numeric=True, original_df=self)


def apply_element_wise_patches():
    """Apply element-wise operation patches to pandas.

    These patches use 3-tier dispatch:
    - Serial for small arrays (< 500K elements)
    - Parallel prange for medium arrays (500K - 10M elements)
    - ThreadPool+nogil for large arrays (> 10M elements)
    """
    from .._patch import patch

    patch(pd.DataFrame, 'clip', optimized_clip)
    patch(pd.DataFrame, 'abs', optimized_abs)
    patch(pd.DataFrame, 'round', optimized_round)
