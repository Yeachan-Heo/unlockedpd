"""Parallel DataFrame correlation and covariance operations using Numba.

This module provides optimized DataFrame.corr() and DataFrame.cov() using:
1. Serial Numba JIT for small arrays
2. Parallel prange for medium arrays
3. ThreadPool + Numba nogil for large arrays (true parallelism)

Uses Welford's online algorithm for numerical stability when computing
correlation and covariance between column pairs.
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import os

from .._compat import get_numeric_columns_fast, ensure_float64

# Threading configuration
_CPU_COUNT = os.cpu_count() or 8
THREADPOOL_WORKERS = min(_CPU_COUNT, 32)

# Thresholds for dispatch (total elements = n_rows * n_cols)
# For correlation, complexity is O(n_rows * n_cols^2) for pairwise comparisons
PARALLEL_THRESHOLD = 500_000  # Use parallel prange above this
THREADPOOL_THRESHOLD = 10_000_000  # Use ThreadPool above this


# ============================================================================
# Core Welford kernels for pairwise correlation/covariance
# ============================================================================

@njit(nogil=True, cache=True)
def _corr_pair(x, y, min_periods):
    """Compute Pearson correlation between two columns using Welford's algorithm.

    This is numerically stable and handles NaN values.

    Args:
        x: First column array
        y: Second column array
        min_periods: Minimum number of valid observations required

    Returns:
        Pearson correlation coefficient or NaN
    """
    n = len(x)
    mean_x = 0.0
    mean_y = 0.0
    M2_x = 0.0  # Sum of squared deviations for x
    M2_y = 0.0  # Sum of squared deviations for y
    C = 0.0     # Co-moment sum
    count = 0

    for i in range(n):
        vx, vy = x[i], y[i]
        if np.isnan(vx) or np.isnan(vy):
            continue
        count += 1
        delta_x = vx - mean_x
        mean_x += delta_x / count
        delta_y = vy - mean_y
        mean_y += delta_y / count
        # Note: delta2 uses updated mean for numerical stability
        delta2_x = vx - mean_x
        delta2_y = vy - mean_y
        M2_x += delta_x * delta2_x
        M2_y += delta_y * delta2_y
        C += delta_x * delta2_y

    if count < min_periods:
        return np.nan
    # Check for near-zero variance (constant columns)
    if M2_x < 1e-14 or M2_y < 1e-14:
        return np.nan

    return C / np.sqrt(M2_x * M2_y)


@njit(nogil=True, cache=True)
def _cov_pair(x, y, min_periods, ddof):
    """Compute covariance between two columns using Welford's algorithm.

    Args:
        x: First column array
        y: Second column array
        min_periods: Minimum number of valid observations required
        ddof: Delta degrees of freedom (default 1 for sample covariance)

    Returns:
        Covariance or NaN
    """
    n = len(x)
    mean_x = 0.0
    mean_y = 0.0
    C = 0.0  # Co-moment sum
    count = 0

    for i in range(n):
        vx, vy = x[i], y[i]
        if np.isnan(vx) or np.isnan(vy):
            continue
        count += 1
        delta_x = vx - mean_x
        mean_x += delta_x / count
        delta_y = vy - mean_y
        mean_y += delta_y / count
        delta2_y = vy - mean_y
        C += delta_x * delta2_y

    if count < min_periods:
        return np.nan
    if count <= ddof:
        return np.nan

    return C / (count - ddof)


# ============================================================================
# Serial kernels (for small arrays)
# ============================================================================

@njit(cache=True)
def _corr_matrix_serial(arr, min_periods):
    """Compute correlation matrix serially.

    Only computes upper triangle and mirrors to lower for efficiency.
    """
    n_cols = arr.shape[1]
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    for i in range(n_cols):
        result[i, i] = 1.0  # Diagonal is always 1.0
        for j in range(i + 1, n_cols):
            corr = _corr_pair(arr[:, i], arr[:, j], min_periods)
            result[i, j] = corr
            result[j, i] = corr  # Symmetric

    return result


@njit(cache=True)
def _cov_matrix_serial(arr, min_periods, ddof):
    """Compute covariance matrix serially.

    Only computes upper triangle and mirrors to lower for efficiency.
    """
    n_cols = arr.shape[1]
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    for i in range(n_cols):
        # Diagonal: variance of column i
        result[i, i] = _cov_pair(arr[:, i], arr[:, i], min_periods, ddof)
        for j in range(i + 1, n_cols):
            cov = _cov_pair(arr[:, i], arr[:, j], min_periods, ddof)
            result[i, j] = cov
            result[j, i] = cov  # Symmetric

    return result


# ============================================================================
# Parallel prange kernels (for medium arrays)
# ============================================================================

@njit(parallel=True, cache=True)
def _corr_matrix_parallel(arr, min_periods):
    """Compute correlation matrix using parallel prange.

    Parallelizes across columns (outer loop).
    """
    n_cols = arr.shape[1]
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    for i in prange(n_cols):
        result[i, i] = 1.0  # Diagonal
        for j in range(i + 1, n_cols):
            corr = _corr_pair(arr[:, i], arr[:, j], min_periods)
            result[i, j] = corr
            result[j, i] = corr  # Symmetric

    return result


@njit(parallel=True, cache=True)
def _cov_matrix_parallel(arr, min_periods, ddof):
    """Compute covariance matrix using parallel prange.

    Parallelizes across columns (outer loop).
    """
    n_cols = arr.shape[1]
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    for i in prange(n_cols):
        # Diagonal: variance
        result[i, i] = _cov_pair(arr[:, i], arr[:, i], min_periods, ddof)
        for j in range(i + 1, n_cols):
            cov = _cov_pair(arr[:, i], arr[:, j], min_periods, ddof)
            result[i, j] = cov
            result[j, i] = cov  # Symmetric

    return result


# ============================================================================
# ThreadPool + nogil kernels (for large arrays)
# ============================================================================

@njit(nogil=True, cache=True)
def _corr_matrix_nogil_chunk(arr, result, start_col, end_col, min_periods):
    """Compute correlation for columns start_col to end_col vs all columns.

    This kernel releases the GIL for true parallel execution with ThreadPool.
    """
    n_cols = arr.shape[1]

    for i in range(start_col, end_col):
        # Diagonal
        result[i, i] = 1.0
        # Upper triangle (j > i)
        for j in range(i + 1, n_cols):
            corr = _corr_pair(arr[:, i], arr[:, j], min_periods)
            result[i, j] = corr
            result[j, i] = corr  # Symmetric


@njit(nogil=True, cache=True)
def _cov_matrix_nogil_chunk(arr, result, start_col, end_col, min_periods, ddof):
    """Compute covariance for columns start_col to end_col vs all columns.

    This kernel releases the GIL for true parallel execution with ThreadPool.
    """
    n_cols = arr.shape[1]

    for i in range(start_col, end_col):
        # Diagonal: variance
        result[i, i] = _cov_pair(arr[:, i], arr[:, i], min_periods, ddof)
        # Upper triangle (j > i)
        for j in range(i + 1, n_cols):
            cov = _cov_pair(arr[:, i], arr[:, j], min_periods, ddof)
            result[i, j] = cov
            result[j, i] = cov  # Symmetric


def _corr_matrix_threadpool(arr, min_periods):
    """Compute correlation matrix using ThreadPool + nogil kernels.

    Distributes column ranges across worker threads for true parallelism.
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)
    result[:] = np.nan  # Initialize with NaN

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _corr_matrix_nogil_chunk(arr, result, start_col, end_col, min_periods)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _cov_matrix_threadpool(arr, min_periods, ddof):
    """Compute covariance matrix using ThreadPool + nogil kernels.

    Distributes column ranges across worker threads for true parallelism.
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)
    result[:] = np.nan  # Initialize with NaN

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _cov_matrix_nogil_chunk(arr, result, start_col, end_col, min_periods, ddof)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


# ============================================================================
# Dispatch functions (choose serial vs parallel vs threadpool)
# ============================================================================

def _corr_dispatch(arr, min_periods):
    """Dispatch to appropriate correlation implementation based on array size.

    - Small arrays: serial (avoid parallel overhead)
    - Medium arrays: prange parallel
    - Large arrays: ThreadPool + nogil (best for huge arrays)
    """
    total_elements = arr.size

    if total_elements < PARALLEL_THRESHOLD:
        return _corr_matrix_serial(arr, min_periods)
    elif total_elements < THREADPOOL_THRESHOLD:
        return _corr_matrix_parallel(arr, min_periods)
    else:
        return _corr_matrix_threadpool(arr, min_periods)


def _cov_dispatch(arr, min_periods, ddof):
    """Dispatch to appropriate covariance implementation based on array size.

    - Small arrays: serial (avoid parallel overhead)
    - Medium arrays: prange parallel
    - Large arrays: ThreadPool + nogil (best for huge arrays)
    """
    total_elements = arr.size

    if total_elements < PARALLEL_THRESHOLD:
        return _cov_matrix_serial(arr, min_periods, ddof)
    elif total_elements < THREADPOOL_THRESHOLD:
        return _cov_matrix_parallel(arr, min_periods, ddof)
    else:
        return _cov_matrix_threadpool(arr, min_periods, ddof)


# ============================================================================
# Pandas wrapper functions
# ============================================================================

def optimized_corr(self, method='pearson', min_periods=1, numeric_only=False):
    """Optimized DataFrame.corr() using Numba-accelerated Welford algorithm.

    Computes pairwise Pearson correlation of columns, excluding NA/null values.

    Args:
        self: DataFrame
        method: Only 'pearson' is supported (raises TypeError for others)
        min_periods: Minimum number of observations required per pair
        numeric_only: Not used (all numeric columns are always selected)

    Returns:
        DataFrame with correlation matrix

    Raises:
        TypeError: If optimization cannot be applied (non-DataFrame, non-pearson method)
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if method != 'pearson':
        raise TypeError("Only pearson correlation supported")

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    result = _corr_dispatch(arr, min_periods)

    return pd.DataFrame(result, index=numeric_cols, columns=numeric_cols)


def optimized_cov(self, min_periods=None, ddof=1, numeric_only=False):
    """Optimized DataFrame.cov() using Numba-accelerated Welford algorithm.

    Computes pairwise covariance of columns, excluding NA/null values.

    Args:
        self: DataFrame
        min_periods: Minimum number of observations required per pair
        ddof: Delta degrees of freedom. Divisor is N - ddof (default 1)
        numeric_only: Not used (all numeric columns are always selected)

    Returns:
        DataFrame with covariance matrix

    Raises:
        TypeError: If optimization cannot be applied (non-DataFrame)
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if min_periods is None:
        min_periods = 1

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    result = _cov_dispatch(arr, min_periods, ddof)

    return pd.DataFrame(result, index=numeric_cols, columns=numeric_cols)


# ============================================================================
# Patch application
# ============================================================================

def apply_correlation_patches():
    """Apply correlation and covariance patches to pandas DataFrame."""
    from .._patch import patch

    patch(pd.DataFrame, 'corr', optimized_corr)
    patch(pd.DataFrame, 'cov', optimized_cov)
