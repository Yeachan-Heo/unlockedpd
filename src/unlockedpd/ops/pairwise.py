"""Parallel pairwise operations (correlation, covariance) using Numba.

This module provides Numba-accelerated pairwise operations that compute
correlation and covariance matrices by parallelizing across column pairs.
"""
import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Optional

from .._compat import get_numeric_columns, ensure_float64, ensure_optimal_layout

# Threshold for parallel vs serial execution (elements)
# Parallel overhead is ~1-2ms, so we need enough work to amortize it
PARALLEL_THRESHOLD = 500_000


# ============================================================================
# Core Numba-jitted functions (PARALLEL versions)
# ============================================================================

@njit(parallel=True, cache=True)
def _corr_matrix(arr: np.ndarray, min_periods: int = 1) -> np.ndarray:
    """Compute pairwise correlation matrix in parallel.

    Uses Pearson correlation coefficient with pairwise-complete observation handling.

    Args:
        arr: 2D array (n_rows, n_cols)
        min_periods: Minimum number of observations required per pair

    Returns:
        Correlation matrix (n_cols, n_cols)
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    # Pre-compute means and standard deviations for each column
    means = np.empty(n_cols, dtype=np.float64)
    stds = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        # Compute mean using pairwise-complete (skip NaN)
        sum_val = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                sum_val += val
                count += 1

        if count >= min_periods:
            means[col] = sum_val / count
        else:
            means[col] = np.nan

        # Compute standard deviation
        sum_sq = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                sum_sq += (val - means[col]) ** 2
                count += 1

        if count >= min_periods and count > 0:
            stds[col] = np.sqrt(sum_sq / count)
        else:
            stds[col] = np.nan

    # Parallel over column pairs
    for col_i in prange(n_cols):
        for col_j in range(col_i, n_cols):
            if col_i == col_j:
                # Correlation of a column with itself is 1.0
                result[col_i, col_j] = 1.0
            else:
                # Check if either column has invalid mean/std
                if np.isnan(means[col_i]) or np.isnan(means[col_j]) or np.isnan(stds[col_i]) or np.isnan(stds[col_j]):
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan
                    continue

                if stds[col_i] == 0.0 or stds[col_j] == 0.0:
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan
                    continue

                # Compute covariance using pairwise-complete observations
                sum_cov = 0.0
                count = 0
                for row in range(n_rows):
                    val_i = arr[row, col_i]
                    val_j = arr[row, col_j]
                    if not np.isnan(val_i) and not np.isnan(val_j):
                        sum_cov += (val_i - means[col_i]) * (val_j - means[col_j])
                        count += 1

                if count >= min_periods:
                    cov = sum_cov / count
                    corr = cov / (stds[col_i] * stds[col_j])
                    result[col_i, col_j] = corr
                    result[col_j, col_i] = corr
                else:
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan

    return result


@njit(parallel=True, cache=True)
def _cov_matrix(arr: np.ndarray, min_periods: int = 1, ddof: int = 1) -> np.ndarray:
    """Compute pairwise covariance matrix in parallel.

    Uses pairwise-complete observation handling.

    Args:
        arr: 2D array (n_rows, n_cols)
        min_periods: Minimum number of observations required per pair
        ddof: Delta degrees of freedom (0 for population, 1 for sample covariance)

    Returns:
        Covariance matrix (n_cols, n_cols)
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    # Pre-compute means for each column
    means = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        sum_val = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                sum_val += val
                count += 1

        if count >= min_periods:
            means[col] = sum_val / count
        else:
            means[col] = np.nan

    # Parallel over column pairs
    for col_i in prange(n_cols):
        for col_j in range(col_i, n_cols):
            # Check if either column has invalid mean
            if np.isnan(means[col_i]) or np.isnan(means[col_j]):
                result[col_i, col_j] = np.nan
                if col_i != col_j:
                    result[col_j, col_i] = np.nan
                continue

            # Compute covariance using pairwise-complete observations
            sum_cov = 0.0
            count = 0
            for row in range(n_rows):
                val_i = arr[row, col_i]
                val_j = arr[row, col_j]
                if not np.isnan(val_i) and not np.isnan(val_j):
                    sum_cov += (val_i - means[col_i]) * (val_j - means[col_j])
                    count += 1

            if count >= min_periods and count > ddof:
                cov = sum_cov / (count - ddof)
                result[col_i, col_j] = cov
                if col_i != col_j:
                    result[col_j, col_i] = cov
            else:
                result[col_i, col_j] = np.nan
                if col_i != col_j:
                    result[col_j, col_i] = np.nan

    return result


# ============================================================================
# Core Numba-jitted functions (SERIAL versions for small arrays)
# ============================================================================

@njit(cache=True)
def _corr_matrix_serial(arr: np.ndarray, min_periods: int = 1) -> np.ndarray:
    """Serial correlation matrix for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    # Pre-compute means and standard deviations
    means = np.empty(n_cols, dtype=np.float64)
    stds = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        sum_val = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                sum_val += val
                count += 1

        if count >= min_periods:
            means[col] = sum_val / count
        else:
            means[col] = np.nan

        sum_sq = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                sum_sq += (val - means[col]) ** 2
                count += 1

        if count >= min_periods and count > 0:
            stds[col] = np.sqrt(sum_sq / count)
        else:
            stds[col] = np.nan

    # Serial computation of correlations
    for col_i in range(n_cols):
        for col_j in range(col_i, n_cols):
            if col_i == col_j:
                result[col_i, col_j] = 1.0
            else:
                if np.isnan(means[col_i]) or np.isnan(means[col_j]) or np.isnan(stds[col_i]) or np.isnan(stds[col_j]):
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan
                    continue

                if stds[col_i] == 0.0 or stds[col_j] == 0.0:
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan
                    continue

                sum_cov = 0.0
                count = 0
                for row in range(n_rows):
                    val_i = arr[row, col_i]
                    val_j = arr[row, col_j]
                    if not np.isnan(val_i) and not np.isnan(val_j):
                        sum_cov += (val_i - means[col_i]) * (val_j - means[col_j])
                        count += 1

                if count >= min_periods:
                    cov = sum_cov / count
                    corr = cov / (stds[col_i] * stds[col_j])
                    result[col_i, col_j] = corr
                    result[col_j, col_i] = corr
                else:
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan

    return result


@njit(cache=True)
def _cov_matrix_serial(arr: np.ndarray, min_periods: int = 1, ddof: int = 1) -> np.ndarray:
    """Serial covariance matrix for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    # Pre-compute means
    means = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        sum_val = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                sum_val += val
                count += 1

        if count >= min_periods:
            means[col] = sum_val / count
        else:
            means[col] = np.nan

    # Serial computation of covariances
    for col_i in range(n_cols):
        for col_j in range(col_i, n_cols):
            if np.isnan(means[col_i]) or np.isnan(means[col_j]):
                result[col_i, col_j] = np.nan
                if col_i != col_j:
                    result[col_j, col_i] = np.nan
                continue

            sum_cov = 0.0
            count = 0
            for row in range(n_rows):
                val_i = arr[row, col_i]
                val_j = arr[row, col_j]
                if not np.isnan(val_i) and not np.isnan(val_j):
                    sum_cov += (val_i - means[col_i]) * (val_j - means[col_j])
                    count += 1

            if count >= min_periods and count > ddof:
                cov = sum_cov / (count - ddof)
                result[col_i, col_j] = cov
                if col_i != col_j:
                    result[col_j, col_i] = cov
            else:
                result[col_i, col_j] = np.nan
                if col_i != col_j:
                    result[col_j, col_i] = np.nan

    return result


# ============================================================================
# Dispatch functions (choose serial vs parallel based on array size)
# ============================================================================

def _corr_dispatch(arr: np.ndarray, min_periods: int = 1) -> np.ndarray:
    """Dispatch to serial or parallel correlation matrix based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _corr_matrix_serial(arr, min_periods)
    return _corr_matrix(arr, min_periods)


def _cov_dispatch(arr: np.ndarray, min_periods: int = 1, ddof: int = 1) -> np.ndarray:
    """Dispatch to serial or parallel covariance matrix based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _cov_matrix_serial(arr, min_periods, ddof)
    return _cov_matrix(arr, min_periods, ddof)


# ============================================================================
# Wrapper functions for pandas DataFrame methods
# ============================================================================

def optimized_corr(df: pd.DataFrame, method: str = 'pearson', min_periods: int = 1,
                   numeric_only: bool = False) -> pd.DataFrame:
    """Compute pairwise correlation of columns.

    Args:
        df: Input DataFrame
        method: Currently only 'pearson' is supported
        min_periods: Minimum number of observations required per pair
        numeric_only: Include only numeric columns (ignored, always True)

    Returns:
        Correlation matrix as DataFrame
    """
    if method != 'pearson':
        raise ValueError(f"Method '{method}' not supported. Only 'pearson' is available.")

    # Extract numeric columns
    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to compute correlation")

    if len(numeric_cols) == 1:
        # Single column: correlation with itself is 1.0
        return pd.DataFrame([[1.0]], index=numeric_cols, columns=numeric_cols)

    # Optimize memory layout for column access (F-contiguous)
    arr = ensure_optimal_layout(ensure_float64(numeric_df.values), axis=0)
    result = _corr_dispatch(arr, min_periods)

    return pd.DataFrame(result, index=numeric_cols, columns=numeric_cols)


def optimized_cov(df: pd.DataFrame, min_periods: Optional[int] = None,
                  ddof: int = 1, numeric_only: bool = False) -> pd.DataFrame:
    """Compute pairwise covariance of columns.

    Args:
        df: Input DataFrame
        min_periods: Minimum number of observations required per pair (defaults to ddof + 1)
        ddof: Delta degrees of freedom (0 for population, 1 for sample covariance)
        numeric_only: Include only numeric columns (ignored, always True)

    Returns:
        Covariance matrix as DataFrame
    """
    if min_periods is None:
        min_periods = ddof + 1

    # Extract numeric columns
    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to compute covariance")

    if len(numeric_cols) == 1:
        # Single column: compute variance
        arr = ensure_optimal_layout(ensure_float64(numeric_df.values), axis=0)
        n_rows = arr.shape[0]

        sum_val = 0.0
        count = 0
        for i in range(n_rows):
            val = arr[i, 0]
            if not np.isnan(val):
                sum_val += val
                count += 1

        if count >= min_periods and count > ddof:
            mean = sum_val / count
            sum_sq = 0.0
            for i in range(n_rows):
                val = arr[i, 0]
                if not np.isnan(val):
                    sum_sq += (val - mean) ** 2
            variance = sum_sq / (count - ddof)
            return pd.DataFrame([[variance]], index=numeric_cols, columns=numeric_cols)
        else:
            return pd.DataFrame([[np.nan]], index=numeric_cols, columns=numeric_cols)

    # Optimize memory layout for column access (F-contiguous)
    arr = ensure_optimal_layout(ensure_float64(numeric_df.values), axis=0)
    result = _cov_dispatch(arr, min_periods, ddof)

    return pd.DataFrame(result, index=numeric_cols, columns=numeric_cols)


# ============================================================================
# Patch application
# ============================================================================

def apply_pairwise_patches():
    """Apply all pairwise operation patches to pandas DataFrame."""
    from .._patch import patch

    patch(pd.DataFrame, 'corr', optimized_corr)
    patch(pd.DataFrame, 'cov', optimized_cov)
