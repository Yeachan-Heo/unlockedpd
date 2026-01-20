"""Parallel cumulative operations using NumPy with thread pooling.

This module provides parallelized cumulative operations by distributing
columns across threads. Since NumPy releases the GIL during computation,
multiple threads can execute NumPy cumsum in parallel.

Key insight: While we can't beat NumPy's SIMD-optimized single-column cumsum,
we CAN parallelize across columns when there are many columns (500+).
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Union

from .._compat import get_numeric_columns, wrap_result, ensure_float64

# Thresholds for parallel execution
# Based on benchmarking: parallel helps when n_cols >= 200 and n_rows >= 5000
MIN_COLS_FOR_PARALLEL = 200
MIN_ROWS_FOR_PARALLEL = 5000

# Optimal worker count (memory bandwidth limits benefit of more threads)
OPTIMAL_WORKERS = 8


# ============================================================================
# Core parallel implementations using ThreadPoolExecutor + NumPy
# ============================================================================

def _cumsum_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cumsum across columns using NumPy (GIL-releasing).

    Keeps C-contiguous layout (pandas default) and processes column chunks.
    Each thread uses NumPy's vectorized cumsum on a slice of columns.
    """
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_cols + OPTIMAL_WORKERS - 1) // OPTIMAL_WORKERS)

    def process_chunk(worker_id):
        start_col = worker_id * chunk_size
        end_col = min(start_col + chunk_size, n_cols)
        # NumPy cumsum on column slice - vectorized and GIL-releasing
        result[:, start_col:end_col] = np.cumsum(arr[:, start_col:end_col], axis=0)

    with ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
        list(executor.map(process_chunk, range(OPTIMAL_WORKERS)))

    return result


def _cumprod_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cumprod across columns using NumPy."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_cols + OPTIMAL_WORKERS - 1) // OPTIMAL_WORKERS)

    def process_chunk(worker_id):
        start_col = worker_id * chunk_size
        end_col = min(start_col + chunk_size, n_cols)
        result[:, start_col:end_col] = np.cumprod(arr[:, start_col:end_col], axis=0)

    with ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
        list(executor.map(process_chunk, range(OPTIMAL_WORKERS)))

    return result


def _cummin_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cummin across columns using NumPy."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_cols + OPTIMAL_WORKERS - 1) // OPTIMAL_WORKERS)

    def process_chunk(worker_id):
        start_col = worker_id * chunk_size
        end_col = min(start_col + chunk_size, n_cols)
        result[:, start_col:end_col] = np.minimum.accumulate(arr[:, start_col:end_col], axis=0)

    with ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
        list(executor.map(process_chunk, range(OPTIMAL_WORKERS)))

    return result


def _cummax_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cummax across columns using NumPy."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    chunk_size = max(1, (n_cols + OPTIMAL_WORKERS - 1) // OPTIMAL_WORKERS)

    def process_chunk(worker_id):
        start_col = worker_id * chunk_size
        end_col = min(start_col + chunk_size, n_cols)
        result[:, start_col:end_col] = np.maximum.accumulate(arr[:, start_col:end_col], axis=0)

    with ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
        list(executor.map(process_chunk, range(OPTIMAL_WORKERS)))

    return result


# ============================================================================
# Dispatch functions (choose parallel vs pandas based on shape)
# ============================================================================

def _should_use_parallel(arr):
    """Determine if parallel execution is worthwhile.

    Parallel helps when:
    - Enough columns to distribute (>= 200)
    - Enough rows per column for meaningful work (>= 5000)
    """
    n_rows, n_cols = arr.shape
    return n_cols >= MIN_COLS_FOR_PARALLEL and n_rows >= MIN_ROWS_FOR_PARALLEL


# ============================================================================
# Wrapper functions for pandas DataFrame methods
# ============================================================================

def optimized_cumsum(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cumsum - parallel for wide DataFrames, pandas for narrow."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis not in (0, 'index', None):
        raise ValueError("Only axis=0 is supported")

    numeric_cols, numeric_df = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if _should_use_parallel(arr):
        result = _cumsum_parallel(arr, skipna)
    else:
        # Fall back to pandas for small DataFrames
        raise TypeError("Use pandas for small DataFrames")

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def optimized_cumprod(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cumprod - parallel for wide DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis not in (0, 'index', None):
        raise ValueError("Only axis=0 is supported")

    numeric_cols, numeric_df = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if _should_use_parallel(arr):
        result = _cumprod_parallel(arr, skipna)
    else:
        raise TypeError("Use pandas for small DataFrames")

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def optimized_cummin(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cummin - parallel for wide DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis not in (0, 'index', None):
        raise ValueError("Only axis=0 is supported")

    numeric_cols, numeric_df = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if _should_use_parallel(arr):
        result = _cummin_parallel(arr, skipna)
    else:
        raise TypeError("Use pandas for small DataFrames")

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def optimized_cummax(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cummax - parallel for wide DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis not in (0, 'index', None):
        raise ValueError("Only axis=0 is supported")

    numeric_cols, numeric_df = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if _should_use_parallel(arr):
        result = _cummax_parallel(arr, skipna)
    else:
        raise TypeError("Use pandas for small DataFrames")

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def apply_cumulative_patches():
    """Apply cumulative operation patches to pandas.

    These patches only activate for wide DataFrames (200+ columns, 5000+ rows).
    For narrow DataFrames, falls back to pandas automatically.
    """
    from .._patch import patch

    patch(pd.DataFrame, 'cumsum', optimized_cumsum)
    patch(pd.DataFrame, 'cumprod', optimized_cumprod)
    patch(pd.DataFrame, 'cummin', optimized_cummin)
    patch(pd.DataFrame, 'cummax', optimized_cummax)
