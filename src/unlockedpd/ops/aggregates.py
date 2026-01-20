"""Parallel DataFrame aggregate operations using Numba.

Provides optimized versions of df.mean(), df.std(), df.var(), df.sum()
for both axis=0 and axis=1.
"""
import numpy as np
from numba import njit, prange
import pandas as pd

from .._compat import get_numeric_columns_fast, ensure_float64

# ============================================================================
# Mean
# ============================================================================

@njit(parallel=True, cache=True)
def _mean_2d_axis0(arr, skipna):
    """Mean along axis=0 (reduce rows, result per column)."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in prange(n_cols):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                total = np.nan
                count = 1
                break
            total += val
            count += 1
        result[col] = total / count if count > 0 else np.nan
    return result


@njit(parallel=True, cache=True)
def _mean_2d_axis1(arr, skipna):
    """Mean along axis=1 (reduce columns, result per row)."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)
    for row in prange(n_rows):
        total = 0.0
        count = 0
        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                total = np.nan
                count = 1
                break
            total += val
            count += 1
        result[row] = total / count if count > 0 else np.nan
    return result


def optimized_mean(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized DataFrame mean."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if axis == 0 or axis == 'index':
        result = _mean_2d_axis0(arr, skipna)
        return pd.Series(result, index=numeric_cols)
    elif axis == 1 or axis == 'columns':
        result = _mean_2d_axis1(arr, skipna)
        return pd.Series(result, index=df.index)
    else:
        raise ValueError(f"Invalid axis: {axis}")


# ============================================================================
# Sum
# ============================================================================

@njit(parallel=True, cache=True)
def _sum_2d_axis0(arr, skipna):
    """Sum along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in prange(n_cols):
        total = 0.0
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                total = np.nan
                break
            total += val
        result[col] = total
    return result


@njit(parallel=True, cache=True)
def _sum_2d_axis1(arr, skipna):
    """Sum along axis=1."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)
    for row in prange(n_rows):
        total = 0.0
        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                total = np.nan
                break
            total += val
        result[row] = total
    return result


def optimized_sum(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized DataFrame sum."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if axis == 0 or axis == 'index':
        result = _sum_2d_axis0(arr, skipna)
        return pd.Series(result, index=numeric_cols)
    elif axis == 1 or axis == 'columns':
        result = _sum_2d_axis1(arr, skipna)
        return pd.Series(result, index=df.index)
    else:
        raise ValueError(f"Invalid axis: {axis}")


# ============================================================================
# Std
# ============================================================================

@njit(parallel=True, cache=True)
def _std_2d_axis0(arr, skipna, ddof):
    """Std along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in prange(n_cols):
        # First pass: mean
        total = 0.0
        count = 0
        has_nan = False
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                has_nan = True
                break
            total += val
            count += 1

        if has_nan or count <= ddof:
            result[col] = np.nan
            continue

        mean = total / count

        # Second pass: variance
        sum_sq = 0.0
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            diff = val - mean
            sum_sq += diff * diff

        result[col] = np.sqrt(sum_sq / (count - ddof))
    return result


@njit(parallel=True, cache=True)
def _std_2d_axis1(arr, skipna, ddof):
    """Std along axis=1."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)
    for row in prange(n_rows):
        total = 0.0
        count = 0
        has_nan = False
        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                has_nan = True
                break
            total += val
            count += 1

        if has_nan or count <= ddof:
            result[row] = np.nan
            continue

        mean = total / count

        sum_sq = 0.0
        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            diff = val - mean
            sum_sq += diff * diff

        result[row] = np.sqrt(sum_sq / (count - ddof))
    return result


def optimized_std(df, axis=0, skipna=True, ddof=1, *args, **kwargs):
    """Optimized DataFrame std."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if axis == 0 or axis == 'index':
        result = _std_2d_axis0(arr, skipna, ddof)
        return pd.Series(result, index=numeric_cols)
    elif axis == 1 or axis == 'columns':
        result = _std_2d_axis1(arr, skipna, ddof)
        return pd.Series(result, index=df.index)
    else:
        raise ValueError(f"Invalid axis: {axis}")


# ============================================================================
# Var
# ============================================================================

@njit(parallel=True, cache=True)
def _var_2d_axis0(arr, skipna, ddof):
    """Var along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in prange(n_cols):
        total = 0.0
        count = 0
        has_nan = False
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                has_nan = True
                break
            total += val
            count += 1

        if has_nan or count <= ddof:
            result[col] = np.nan
            continue

        mean = total / count

        sum_sq = 0.0
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            diff = val - mean
            sum_sq += diff * diff

        result[col] = sum_sq / (count - ddof)
    return result


@njit(parallel=True, cache=True)
def _var_2d_axis1(arr, skipna, ddof):
    """Var along axis=1."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.float64)
    for row in prange(n_rows):
        total = 0.0
        count = 0
        has_nan = False
        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            if not skipna and np.isnan(val):
                has_nan = True
                break
            total += val
            count += 1

        if has_nan or count <= ddof:
            result[row] = np.nan
            continue

        mean = total / count

        sum_sq = 0.0
        for col in range(n_cols):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            diff = val - mean
            sum_sq += diff * diff

        result[row] = sum_sq / (count - ddof)
    return result


def optimized_var(df, axis=0, skipna=True, ddof=1, *args, **kwargs):
    """Optimized DataFrame var."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    if axis == 0 or axis == 'index':
        result = _var_2d_axis0(arr, skipna, ddof)
        return pd.Series(result, index=numeric_cols)
    elif axis == 1 or axis == 'columns':
        result = _var_2d_axis1(arr, skipna, ddof)
        return pd.Series(result, index=df.index)
    else:
        raise ValueError(f"Invalid axis: {axis}")


# ============================================================================
# Patch Application
# ============================================================================

def apply_aggregate_patches():
    """Apply aggregate operation patches to pandas."""
    from .._patch import patch

    patch(pd.DataFrame, 'mean', optimized_mean)
    patch(pd.DataFrame, 'sum', optimized_sum)
    patch(pd.DataFrame, 'std', optimized_std)
    patch(pd.DataFrame, 'var', optimized_var)
