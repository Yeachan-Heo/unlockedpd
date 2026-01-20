"""Parallel statistical operations using Numba.

This module provides Numba-accelerated statistical operations (skewness, kurtosis, SEM)
that parallelize across columns for significant speedup on wide DataFrames.

Uses online algorithms for numerical stability when computing higher moments.
"""
import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Optional

from .._compat import get_numeric_columns, wrap_result, ensure_float64, ensure_optimal_layout

# Threshold for parallel vs serial execution (elements)
# Parallel overhead is ~1-2ms, so we need enough work to amortize it
PARALLEL_THRESHOLD = 500_000


# ============================================================================
# Core Numba-jitted functions - SKEWNESS (PARALLEL versions)
# ============================================================================

@njit(parallel=True, cache=True)
def _skew_2d_axis0(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Compute skewness for each column (axis=0) in parallel.

    Uses online algorithm for numerical stability (similar to Welford's method).
    Skewness = E[(X - μ)³] / σ³
    """
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0  # Sum of squared deviations
        M3 = 0.0  # Sum of cubed deviations

        # Online algorithm for computing moments
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        # Compute skewness
        if count < 3:
            result[col] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:  # Near-zero variance
                result[col] = np.nan
            else:
                std = np.sqrt(variance)
                # Adjust for sample skewness (bias correction)
                skew = (np.sqrt(count * (count - 1)) / (count - 2)) * (M3 / count) / (std ** 3)
                result[col] = skew

    return result


@njit(parallel=True, cache=True)
def _skew_2d_axis1(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Compute skewness for each row (axis=1) in parallel."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in prange(n_rows):
        count = 0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0

        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 3:
            result[row] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[row] = np.nan
            else:
                std = np.sqrt(variance)
                skew = (np.sqrt(count * (count - 1)) / (count - 2)) * (M3 / count) / (std ** 3)
                result[row] = skew

    return result


# ============================================================================
# Core Numba-jitted functions - SKEWNESS (SERIAL versions)
# ============================================================================

@njit(cache=True)
def _skew_2d_axis0_serial(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Serial skewness computation for each column."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0

        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 3:
            result[col] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[col] = np.nan
            else:
                std = np.sqrt(variance)
                skew = (np.sqrt(count * (count - 1)) / (count - 2)) * (M3 / count) / (std ** 3)
                result[col] = skew

    return result


@njit(cache=True)
def _skew_2d_axis1_serial(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Serial skewness computation for each row."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in range(n_rows):
        count = 0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0

        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 3:
            result[row] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[row] = np.nan
            else:
                std = np.sqrt(variance)
                skew = (np.sqrt(count * (count - 1)) / (count - 2)) * (M3 / count) / (std ** 3)
                result[row] = skew

    return result


# ============================================================================
# Core Numba-jitted functions - KURTOSIS (PARALLEL versions)
# ============================================================================

@njit(parallel=True, cache=True)
def _kurt_2d_axis0(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Compute kurtosis for each column (axis=0) in parallel.

    Uses online algorithm for numerical stability.
    Kurtosis = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
    """
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0  # Sum of squared deviations
        M3 = 0.0  # Sum of cubed deviations
        M4 = 0.0  # Sum of fourth-power deviations

        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M4 += (term1 * delta_n2 * (count * count - 3 * count + 3) +
                   6.0 * delta_n2 * M2 - 4.0 * delta_n * M3)
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 4:
            result[col] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[col] = np.nan
            else:
                # Excess kurtosis with bias correction
                kurt = ((count - 1) * ((count + 1) * M4 / (M2 * M2) - 3 * (count - 1)) /
                        ((count - 2) * (count - 3)))
                result[col] = kurt

    return result


@njit(parallel=True, cache=True)
def _kurt_2d_axis1(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Compute kurtosis for each row (axis=1) in parallel."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in prange(n_rows):
        count = 0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0
        M4 = 0.0

        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M4 += (term1 * delta_n2 * (count * count - 3 * count + 3) +
                   6.0 * delta_n2 * M2 - 4.0 * delta_n * M3)
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 4:
            result[row] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[row] = np.nan
            else:
                kurt = ((count - 1) * ((count + 1) * M4 / (M2 * M2) - 3 * (count - 1)) /
                        ((count - 2) * (count - 3)))
                result[row] = kurt

    return result


# ============================================================================
# Core Numba-jitted functions - KURTOSIS (SERIAL versions)
# ============================================================================

@njit(cache=True)
def _kurt_2d_axis0_serial(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Serial kurtosis computation for each column."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0
        M4 = 0.0

        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M4 += (term1 * delta_n2 * (count * count - 3 * count + 3) +
                   6.0 * delta_n2 * M2 - 4.0 * delta_n * M3)
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 4:
            result[col] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[col] = np.nan
            else:
                kurt = ((count - 1) * ((count + 1) * M4 / (M2 * M2) - 3 * (count - 1)) /
                        ((count - 2) * (count - 3)))
                result[col] = kurt

    return result


@njit(cache=True)
def _kurt_2d_axis1_serial(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Serial kurtosis computation for each row."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in range(n_rows):
        count = 0
        mean = 0.0
        M2 = 0.0
        M3 = 0.0
        M4 = 0.0

        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            delta_n = delta / count
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (count - 1)

            mean += delta_n
            M4 += (term1 * delta_n2 * (count * count - 3 * count + 3) +
                   6.0 * delta_n2 * M2 - 4.0 * delta_n * M3)
            M3 += term1 * delta_n * (count - 2) - 3.0 * delta_n * M2
            M2 += term1

        if count < 4:
            result[row] = np.nan
        else:
            variance = M2 / (count - 1)
            if variance < 1e-14:
                result[row] = np.nan
            else:
                kurt = ((count - 1) * ((count + 1) * M4 / (M2 * M2) - 3 * (count - 1)) /
                        ((count - 2) * (count - 3)))
                result[row] = kurt

    return result


# ============================================================================
# Core Numba-jitted functions - SEM (Standard Error of Mean) (PARALLEL versions)
# ============================================================================

@njit(parallel=True, cache=True)
def _sem_2d_axis0(arr: np.ndarray, skipna: bool = True, ddof: int = 1) -> np.ndarray:
    """Compute standard error of mean for each column (axis=0) in parallel.

    SEM = std / sqrt(n)
    """
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0

        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            M2 += delta * delta2

        if count <= ddof:
            result[col] = np.nan
        else:
            variance = M2 / (count - ddof)
            std = np.sqrt(variance)
            sem = std / np.sqrt(count)
            result[col] = sem

    return result


@njit(parallel=True, cache=True)
def _sem_2d_axis1(arr: np.ndarray, skipna: bool = True, ddof: int = 1) -> np.ndarray:
    """Compute standard error of mean for each row (axis=1) in parallel."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in prange(n_rows):
        count = 0
        mean = 0.0
        M2 = 0.0

        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            M2 += delta * delta2

        if count <= ddof:
            result[row] = np.nan
        else:
            variance = M2 / (count - ddof)
            std = np.sqrt(variance)
            sem = std / np.sqrt(count)
            result[row] = sem

    return result


# ============================================================================
# Core Numba-jitted functions - SEM (SERIAL versions)
# ============================================================================

@njit(cache=True)
def _sem_2d_axis0_serial(arr: np.ndarray, skipna: bool = True, ddof: int = 1) -> np.ndarray:
    """Serial SEM computation for each column."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in range(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0

        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            M2 += delta * delta2

        if count <= ddof:
            result[col] = np.nan
        else:
            variance = M2 / (count - ddof)
            std = np.sqrt(variance)
            sem = std / np.sqrt(count)
            result[col] = sem

    return result


@njit(cache=True)
def _sem_2d_axis1_serial(arr: np.ndarray, skipna: bool = True, ddof: int = 1) -> np.ndarray:
    """Serial SEM computation for each row."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in range(n_rows):
        count = 0
        mean = 0.0
        M2 = 0.0

        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue

            count += 1
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            M2 += delta * delta2

        if count <= ddof:
            result[row] = np.nan
        else:
            variance = M2 / (count - ddof)
            std = np.sqrt(variance)
            sem = std / np.sqrt(count)
            result[row] = sem

    return result


# ============================================================================
# Dispatch functions (choose serial vs parallel based on array size)
# ============================================================================

def _skew_axis0_dispatch(arr, skipna):
    """Dispatch to serial or parallel skewness based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _skew_2d_axis0_serial(arr, skipna)
    return _skew_2d_axis0(arr, skipna)


def _skew_axis1_dispatch(arr, skipna):
    """Dispatch to serial or parallel skewness based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _skew_2d_axis1_serial(arr, skipna)
    return _skew_2d_axis1(arr, skipna)


def _kurt_axis0_dispatch(arr, skipna):
    """Dispatch to serial or parallel kurtosis based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _kurt_2d_axis0_serial(arr, skipna)
    return _kurt_2d_axis0(arr, skipna)


def _kurt_axis1_dispatch(arr, skipna):
    """Dispatch to serial or parallel kurtosis based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _kurt_2d_axis1_serial(arr, skipna)
    return _kurt_2d_axis1(arr, skipna)


def _sem_axis0_dispatch(arr, skipna, ddof):
    """Dispatch to serial or parallel SEM based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _sem_2d_axis0_serial(arr, skipna, ddof)
    return _sem_2d_axis0(arr, skipna, ddof)


def _sem_axis1_dispatch(arr, skipna, ddof):
    """Dispatch to serial or parallel SEM based on array size."""
    if arr.size < PARALLEL_THRESHOLD:
        return _sem_2d_axis1_serial(arr, skipna, ddof)
    return _sem_2d_axis1(arr, skipna, ddof)


# ============================================================================
# Wrapper functions for DataFrame methods
# ============================================================================

def optimized_skew(df: pd.DataFrame, axis: Union[int, str] = 0, skipna: bool = True,
                   numeric_only: bool = True, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized skewness computation for DataFrames.

    Shape-adaptive parallelization:
    - axis=0: reduce rows → parallelize across columns (column-independent)
    - axis=1: reduce cols → parallelize across rows (row-independent)

    Args:
        df: Input DataFrame
        axis: 0 for column-wise, 1 for row-wise
        skipna: Exclude NaN values
        numeric_only: Include only numeric columns

    Returns:
        Series (axis=0) or DataFrame (axis=1) with skewness values
    """
    # Normalize axis parameter
    if axis in ['index', 0]:
        axis = 0
    elif axis in ['columns', 1]:
        axis = 1
    else:
        raise ValueError(f"No axis named {axis} for object type DataFrame")

    # Extract numeric columns
    if numeric_only:
        numeric_cols, numeric_df = get_numeric_columns(df)
        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")
    else:
        numeric_df = df
        numeric_cols = df.columns.tolist()

    # Ensure optimal memory layout for the operation axis
    arr = ensure_float64(numeric_df.values)
    arr = ensure_optimal_layout(arr, axis=axis)

    if axis == 0:
        # axis=0: reduce rows → parallelize across columns
        result = _skew_axis0_dispatch(arr, skipna)
        return pd.Series(result, index=numeric_cols)
    else:  # axis == 1
        # axis=1: reduce cols → parallelize across rows
        result = _skew_axis1_dispatch(arr, skipna)
        return pd.Series(result, index=df.index)


def optimized_kurt(df: pd.DataFrame, axis: Union[int, str] = 0, skipna: bool = True,
                   numeric_only: bool = True, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized kurtosis computation for DataFrames.

    Shape-adaptive parallelization:
    - axis=0: reduce rows → parallelize across columns (column-independent)
    - axis=1: reduce cols → parallelize across rows (row-independent)

    Args:
        df: Input DataFrame
        axis: 0 for column-wise, 1 for row-wise
        skipna: Exclude NaN values
        numeric_only: Include only numeric columns

    Returns:
        Series (axis=0) or DataFrame (axis=1) with kurtosis values
    """
    # Normalize axis parameter
    if axis in ['index', 0]:
        axis = 0
    elif axis in ['columns', 1]:
        axis = 1
    else:
        raise ValueError(f"No axis named {axis} for object type DataFrame")

    # Extract numeric columns
    if numeric_only:
        numeric_cols, numeric_df = get_numeric_columns(df)
        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")
    else:
        numeric_df = df
        numeric_cols = df.columns.tolist()

    # Ensure optimal memory layout for the operation axis
    arr = ensure_float64(numeric_df.values)
    arr = ensure_optimal_layout(arr, axis=axis)

    if axis == 0:
        # axis=0: reduce rows → parallelize across columns
        result = _kurt_axis0_dispatch(arr, skipna)
        return pd.Series(result, index=numeric_cols)
    else:  # axis == 1
        # axis=1: reduce cols → parallelize across rows
        result = _kurt_axis1_dispatch(arr, skipna)
        return pd.Series(result, index=df.index)


def optimized_sem(df: pd.DataFrame, axis: Union[int, str] = 0, skipna: bool = True,
                  ddof: int = 1, numeric_only: bool = True, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized standard error of mean computation for DataFrames.

    Shape-adaptive parallelization:
    - axis=0: reduce rows → parallelize across columns (column-independent)
    - axis=1: reduce cols → parallelize across rows (row-independent)

    Args:
        df: Input DataFrame
        axis: 0 for column-wise, 1 for row-wise
        skipna: Exclude NaN values
        ddof: Delta degrees of freedom
        numeric_only: Include only numeric columns

    Returns:
        Series (axis=0) or DataFrame (axis=1) with SEM values
    """
    # Normalize axis parameter
    if axis in ['index', 0]:
        axis = 0
    elif axis in ['columns', 1]:
        axis = 1
    else:
        raise ValueError(f"No axis named {axis} for object type DataFrame")

    # Extract numeric columns
    if numeric_only:
        numeric_cols, numeric_df = get_numeric_columns(df)
        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")
    else:
        numeric_df = df
        numeric_cols = df.columns.tolist()

    # Ensure optimal memory layout for the operation axis
    arr = ensure_float64(numeric_df.values)
    arr = ensure_optimal_layout(arr, axis=axis)

    if axis == 0:
        # axis=0: reduce rows → parallelize across columns
        result = _sem_axis0_dispatch(arr, skipna, ddof)
        return pd.Series(result, index=numeric_cols)
    else:  # axis == 1
        # axis=1: reduce cols → parallelize across rows
        result = _sem_axis1_dispatch(arr, skipna, ddof)
        return pd.Series(result, index=df.index)


def apply_stats_patches():
    """Apply all statistical operation patches to pandas DataFrame."""
    from .._patch import patch

    patch(pd.DataFrame, 'skew', optimized_skew)
    patch(pd.DataFrame, 'kurt', optimized_kurt)
    patch(pd.DataFrame, 'sem', optimized_sem)
