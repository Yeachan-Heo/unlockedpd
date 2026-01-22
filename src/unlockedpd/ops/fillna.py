"""Optimized fillna and dropna operations using Numba.

This module provides Numba-accelerated fillna and dropna operations
for handling missing values (NaN) in DataFrames.
"""
import numpy as np
from numba import njit, prange
import pandas as pd


@njit(parallel=True, cache=True)
def _fillna_scalar(arr: np.ndarray, value: float) -> np.ndarray:
    """Fill NaN values with a scalar value."""
    n_rows, n_cols = arr.shape
    result = arr.copy()
    for col in prange(n_cols):
        for row in range(n_rows):
            if np.isnan(result[row, col]):
                result[row, col] = value
    return result


@njit(parallel=True, cache=True)
def _fillna_ffill(arr: np.ndarray) -> np.ndarray:
    """Forward fill NaN values (propagate last valid observation forward)."""
    n_rows, n_cols = arr.shape
    result = arr.copy()
    for col in prange(n_cols):
        last_valid = np.nan
        for row in range(n_rows):
            if np.isnan(result[row, col]):
                if not np.isnan(last_valid):
                    result[row, col] = last_valid
            else:
                last_valid = result[row, col]
    return result


@njit(parallel=True, cache=True)
def _fillna_bfill(arr: np.ndarray) -> np.ndarray:
    """Backward fill NaN values (propagate next valid observation backward)."""
    n_rows, n_cols = arr.shape
    result = arr.copy()
    for col in prange(n_cols):
        next_valid = np.nan
        for row in range(n_rows - 1, -1, -1):
            if np.isnan(result[row, col]):
                if not np.isnan(next_valid):
                    result[row, col] = next_valid
            else:
                next_valid = result[row, col]
    return result


def optimized_fillna(df: pd.DataFrame, value=None, method=None, axis=None,
                     inplace=False, limit=None) -> pd.DataFrame:
    """Optimized fillna operation.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    value : scalar, dict, Series, or DataFrame
        Value to use to fill holes
    method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes
    axis : {0 or 'index', 1 or 'columns'}, optional
        Axis along which to fill missing values
    inplace : bool, default False
        If True, fill in-place (not optimized)
    limit : int, optional
        Maximum number of consecutive NaN values to fill (not optimized)

    Returns
    -------
    DataFrame
        DataFrame with NaN values filled
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle empty DataFrame
    if df.empty:
        return df.copy()

    # inplace not optimized
    if inplace:
        raise TypeError("inplace parameter not optimized")

    # limit not optimized
    if limit is not None:
        raise TypeError("limit parameter not optimized")

    # axis not optimized (pandas fillna doesn't really use axis parameter)
    if axis is not None:
        raise TypeError("axis parameter not optimized")

    # Dict/Series/DataFrame values not optimized
    if value is not None and not isinstance(value, (int, float)):
        raise TypeError("Only scalar values are optimized")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise TypeError("No numeric columns")

    arr = numeric_df.values.astype(np.float64)

    # Determine operation
    if value is not None:
        # Fill with scalar value
        result_arr = _fillna_scalar(arr, float(value))
    elif method in ('ffill', 'pad'):
        # Forward fill
        result_arr = _fillna_ffill(arr)
    elif method in ('bfill', 'backfill'):
        # Backward fill
        result_arr = _fillna_bfill(arr)
    elif method is None and value is None:
        raise ValueError("Must specify a fill 'value' or 'method'")
    else:
        raise ValueError(f"Invalid fill method: {method}")

    # Create result DataFrame preserving original structure
    result = df.copy()
    result[numeric_df.columns] = result_arr

    return result


@njit(parallel=True, cache=True)
def _count_nan_per_row(arr: np.ndarray) -> np.ndarray:
    """Count NaN values per row."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_rows, dtype=np.int64)
    for row in prange(n_rows):
        count = 0
        for col in range(n_cols):
            if np.isnan(arr[row, col]):
                count += 1
        result[row] = count
    return result


@njit(parallel=True, cache=True)
def _count_nan_per_col(arr: np.ndarray) -> np.ndarray:
    """Count NaN values per column."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.int64)
    for col in prange(n_cols):
        count = 0
        for row in range(n_rows):
            if np.isnan(arr[row, col]):
                count += 1
        result[col] = count
    return result


def optimized_dropna(df: pd.DataFrame, axis: int = 0, how: str = 'any',
                     thresh=None, subset=None) -> pd.DataFrame:
    """Optimized dropna operation.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    axis : int, default 0
        0 to drop rows, 1 to drop columns
    how : {'any', 'all'}, default 'any'
        'any' drops if any NaN, 'all' drops only if all NaN
    thresh : int, optional
        Require this many non-NaN values
    subset : array-like, optional
        Columns to consider (not optimized)

    Returns
    -------
    DataFrame
        DataFrame with NaN rows/columns dropped
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle empty DataFrame
    if df.empty:
        return df.copy()

    # If subset specified, fall back to pandas
    if subset is not None:
        raise TypeError("subset parameter not optimized")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise TypeError("No numeric columns")

    arr = numeric_df.values.astype(np.float64)
    n_rows, n_cols = arr.shape

    if axis == 0:
        # Drop rows
        nan_counts = _count_nan_per_row(arr)

        if thresh is not None:
            # Keep rows with at least 'thresh' non-NaN values
            non_nan_counts = n_cols - nan_counts
            mask = non_nan_counts >= thresh
        elif how == 'any':
            # Drop rows with any NaN
            mask = nan_counts == 0
        elif how == 'all':
            # Drop rows where all values are NaN
            mask = nan_counts < n_cols
        else:
            raise ValueError(f"how must be 'any' or 'all', got '{how}'")

        return df.iloc[mask]

    else:  # axis == 1
        # Drop columns
        nan_counts = _count_nan_per_col(arr)

        if thresh is not None:
            # Keep columns with at least 'thresh' non-NaN values
            non_nan_counts = n_rows - nan_counts
            keep_mask = non_nan_counts >= thresh
        elif how == 'any':
            # Drop columns with any NaN
            keep_mask = nan_counts == 0
        elif how == 'all':
            # Drop columns where all values are NaN
            keep_mask = nan_counts < n_rows
        else:
            raise ValueError(f"how must be 'any' or 'all', got '{how}'")

        # Build list of columns to keep
        keep_cols = [col for i, col in enumerate(numeric_df.columns) if keep_mask[i]]

        # Also include non-numeric columns (they're not checked)
        non_numeric_cols = [col for col in df.columns if col not in numeric_df.columns]
        result_cols = keep_cols + non_numeric_cols

        # Preserve original column order
        result_cols = [col for col in df.columns if col in result_cols]

        return df[result_cols]


def _fallback_fillna(obj, value=None, method=None, axis=None, inplace=False, limit=None):
    """Fallback to pandas fillna, handling deprecated method parameter.

    In pandas 2.1+, the method parameter was deprecated.
    In pandas 2.2+, it may raise an error.
    We convert method='ffill'/'bfill' to explicit ffill()/bfill() calls.
    """
    # Handle method parameter by calling ffill/bfill directly
    if method in ('ffill', 'pad'):
        if inplace:
            obj.ffill(axis=axis, limit=limit, inplace=True)
            return None
        return obj.ffill(axis=axis, limit=limit)
    elif method in ('bfill', 'backfill'):
        if inplace:
            obj.bfill(axis=axis, limit=limit, inplace=True)
            return None
        return obj.bfill(axis=axis, limit=limit)
    else:
        # value-based fill - use original fillna without method
        from .._patch import _PatchRegistry
        original = _PatchRegistry.get_original(type(obj), 'fillna')
        if original is not None:
            return original(obj, value=value, axis=axis, inplace=inplace, limit=limit)
        # If no original (shouldn't happen), use pandas directly
        return pd.DataFrame.fillna(obj, value=value, axis=axis, inplace=inplace, limit=limit)


def _patched_fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    """Patched fillna method for DataFrame."""
    if inplace:
        return _fallback_fillna(self, value=value, method=method, axis=axis, inplace=inplace, limit=limit)

    try:
        return optimized_fillna(self, value=value, method=method, axis=axis, inplace=inplace, limit=limit)
    except TypeError:
        # Fall back to pandas for unsupported parameters
        return _fallback_fillna(self, value=value, method=method, axis=axis, inplace=inplace, limit=limit)


def _patched_dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False, ignore_index=False):
    """Patched dropna method for DataFrame."""
    if inplace:
        # Get original for inplace operations
        from .._patch import _PatchRegistry
        original = _PatchRegistry.get_original(pd.DataFrame, 'dropna')
        return original(self, axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace, ignore_index=ignore_index)

    try:
        return optimized_dropna(self, axis=axis, how=how, thresh=thresh, subset=subset)
    except TypeError:
        # Fall back to pandas for unsupported parameters
        from .._patch import _PatchRegistry
        original = _PatchRegistry.get_original(pd.DataFrame, 'dropna')
        return original(self, axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace, ignore_index=ignore_index)


def _patched_series_fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    """Patched fillna method for Series.

    Handles the deprecated method parameter in pandas 2.1+.
    """
    return _fallback_fillna(self, value=value, method=method, axis=axis, inplace=inplace, limit=limit)


def apply_fillna_patches():
    """Apply fillna and dropna patches to pandas."""
    from .._patch import patch

    patch(pd.DataFrame, 'fillna', _patched_fillna, fallback=True)
    patch(pd.DataFrame, 'dropna', _patched_dropna, fallback=True)
    # Also patch Series.fillna to handle deprecated method parameter
    patch(pd.Series, 'fillna', _patched_series_fillna, fallback=True)
