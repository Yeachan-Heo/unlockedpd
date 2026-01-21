"""Parallel quantile operations using Numba nogil kernels with thread pooling.

This module provides parallelized quantile computation by distributing columns
across threads using Numba's nogil=True to release the GIL for true parallel
execution.

Operations support axis parameter:
- axis=0: Compute quantile for each column -> Series/DataFrame indexed by columns
- axis=1: Compute quantile for each row -> Series/DataFrame indexed by original index
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import os

from .._compat import get_numeric_columns_fast, ensure_float64

# Thresholds for parallel execution dispatch
PARALLEL_THRESHOLD = 500_000      # Use parallel prange above this
THREADPOOL_THRESHOLD = 10_000_000 # Use ThreadPool+nogil above this

# Optimal worker count
_CPU_COUNT = os.cpu_count() or 8
THREADPOOL_WORKERS = min(_CPU_COUNT, 32)

# Interpolation method codes for Numba
INTERP_LINEAR = 0
INTERP_LOWER = 1
INTERP_HIGHER = 2
INTERP_NEAREST = 3
INTERP_MIDPOINT = 4


# ============================================================================
# QUANTILE OPERATION - SERIAL
# ============================================================================

@njit(cache=True)
def _quantile_serial(arr, q_arr, interpolation):
    """Serial quantile reduction along axis=0.

    Args:
        arr: 2D float64 array (n_rows, n_cols)
        q_arr: 1D float64 array of quantile values (each in [0, 1])
        interpolation: int code for interpolation method

    Returns:
        2D array (n_quantiles, n_cols) with quantile values
    """
    n_rows, n_cols = arr.shape
    n_q = len(q_arr)
    result = np.empty((n_q, n_cols), dtype=np.float64)

    for c in range(n_cols):
        # Collect non-NaN values for this column
        values = np.empty(n_rows, dtype=np.float64)
        count = 0

        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                values[count] = val
                count += 1

        if count == 0:
            # All NaN column
            for qi in range(n_q):
                result[qi, c] = np.nan
            continue

        # Sort valid values
        valid_values = values[:count]
        valid_values = np.sort(valid_values)
        n = count

        # Compute each quantile
        for qi in range(n_q):
            q = q_arr[qi]

            # Handle edge cases
            if n == 1:
                result[qi, c] = valid_values[0]
                continue

            # Compute quantile index using linear interpolation formula
            # pandas uses (n - 1) * q formula
            idx = q * (n - 1)
            lower_idx = int(np.floor(idx))
            upper_idx = int(np.ceil(idx))

            # Clamp indices
            if lower_idx < 0:
                lower_idx = 0
            if upper_idx >= n:
                upper_idx = n - 1

            if lower_idx == upper_idx:
                result[qi, c] = valid_values[lower_idx]
            elif interpolation == INTERP_LOWER:
                result[qi, c] = valid_values[lower_idx]
            elif interpolation == INTERP_HIGHER:
                result[qi, c] = valid_values[upper_idx]
            elif interpolation == INTERP_NEAREST:
                if idx - lower_idx < upper_idx - idx:
                    result[qi, c] = valid_values[lower_idx]
                else:
                    result[qi, c] = valid_values[upper_idx]
            elif interpolation == INTERP_MIDPOINT:
                result[qi, c] = (valid_values[lower_idx] + valid_values[upper_idx]) / 2.0
            else:  # INTERP_LINEAR (default)
                frac = idx - lower_idx
                result[qi, c] = valid_values[lower_idx] + frac * (valid_values[upper_idx] - valid_values[lower_idx])

    return result


# ============================================================================
# QUANTILE OPERATION - PARALLEL PRANGE
# ============================================================================

@njit(parallel=True, cache=True)
def _quantile_parallel(arr, q_arr, interpolation):
    """Parallel quantile reduction along axis=0 using prange.

    Args:
        arr: 2D float64 array (n_rows, n_cols)
        q_arr: 1D float64 array of quantile values (each in [0, 1])
        interpolation: int code for interpolation method

    Returns:
        2D array (n_quantiles, n_cols) with quantile values
    """
    n_rows, n_cols = arr.shape
    n_q = len(q_arr)
    result = np.empty((n_q, n_cols), dtype=np.float64)

    for c in prange(n_cols):
        # Collect non-NaN values for this column
        values = np.empty(n_rows, dtype=np.float64)
        count = 0

        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                values[count] = val
                count += 1

        if count == 0:
            # All NaN column
            for qi in range(n_q):
                result[qi, c] = np.nan
            continue

        # Sort valid values
        valid_values = values[:count]
        valid_values = np.sort(valid_values)
        n = count

        # Compute each quantile
        for qi in range(n_q):
            q = q_arr[qi]

            # Handle edge cases
            if n == 1:
                result[qi, c] = valid_values[0]
                continue

            # Compute quantile index
            idx = q * (n - 1)
            lower_idx = int(np.floor(idx))
            upper_idx = int(np.ceil(idx))

            # Clamp indices
            if lower_idx < 0:
                lower_idx = 0
            if upper_idx >= n:
                upper_idx = n - 1

            if lower_idx == upper_idx:
                result[qi, c] = valid_values[lower_idx]
            elif interpolation == INTERP_LOWER:
                result[qi, c] = valid_values[lower_idx]
            elif interpolation == INTERP_HIGHER:
                result[qi, c] = valid_values[upper_idx]
            elif interpolation == INTERP_NEAREST:
                if idx - lower_idx < upper_idx - idx:
                    result[qi, c] = valid_values[lower_idx]
                else:
                    result[qi, c] = valid_values[upper_idx]
            elif interpolation == INTERP_MIDPOINT:
                result[qi, c] = (valid_values[lower_idx] + valid_values[upper_idx]) / 2.0
            else:  # INTERP_LINEAR (default)
                frac = idx - lower_idx
                result[qi, c] = valid_values[lower_idx] + frac * (valid_values[upper_idx] - valid_values[lower_idx])

    return result


# ============================================================================
# QUANTILE OPERATION - NOGIL CHUNK (for ThreadPool)
# ============================================================================

@njit(nogil=True, cache=True)
def _quantile_nogil_chunk(arr, result, q_arr, interpolation, start_col, end_col):
    """Quantile reduction for column chunk - GIL released.

    Args:
        arr: 2D float64 array (n_rows, n_cols)
        result: Pre-allocated 2D result array (n_quantiles, n_cols) to fill
        q_arr: 1D float64 array of quantile values
        interpolation: int code for interpolation method
        start_col: Starting column index (inclusive)
        end_col: Ending column index (exclusive)
    """
    n_rows = arr.shape[0]
    n_q = len(q_arr)

    for c in range(start_col, end_col):
        # Collect non-NaN values for this column
        values = np.empty(n_rows, dtype=np.float64)
        count = 0

        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                values[count] = val
                count += 1

        if count == 0:
            # All NaN column
            for qi in range(n_q):
                result[qi, c] = np.nan
            continue

        # Sort valid values
        valid_values = values[:count]
        valid_values = np.sort(valid_values)
        n = count

        # Compute each quantile
        for qi in range(n_q):
            q = q_arr[qi]

            # Handle edge cases
            if n == 1:
                result[qi, c] = valid_values[0]
                continue

            # Compute quantile index
            idx = q * (n - 1)
            lower_idx = int(np.floor(idx))
            upper_idx = int(np.ceil(idx))

            # Clamp indices
            if lower_idx < 0:
                lower_idx = 0
            if upper_idx >= n:
                upper_idx = n - 1

            if lower_idx == upper_idx:
                result[qi, c] = valid_values[lower_idx]
            elif interpolation == INTERP_LOWER:
                result[qi, c] = valid_values[lower_idx]
            elif interpolation == INTERP_HIGHER:
                result[qi, c] = valid_values[upper_idx]
            elif interpolation == INTERP_NEAREST:
                if idx - lower_idx < upper_idx - idx:
                    result[qi, c] = valid_values[lower_idx]
                else:
                    result[qi, c] = valid_values[upper_idx]
            elif interpolation == INTERP_MIDPOINT:
                result[qi, c] = (valid_values[lower_idx] + valid_values[upper_idx]) / 2.0
            else:  # INTERP_LINEAR (default)
                frac = idx - lower_idx
                result[qi, c] = valid_values[lower_idx] + frac * (valid_values[upper_idx] - valid_values[lower_idx])


# ============================================================================
# QUANTILE OPERATION - THREADPOOL WRAPPER
# ============================================================================

def _quantile_threadpool(arr, q_arr, interpolation):
    """ThreadPool quantile using nogil kernels.

    Args:
        arr: 2D float64 array (n_rows, n_cols)
        q_arr: 1D float64 array of quantile values
        interpolation: int code for interpolation method

    Returns:
        2D array (n_quantiles, n_cols) with quantile values
    """
    n_rows, n_cols = arr.shape
    n_q = len(q_arr)
    result = np.empty((n_q, n_cols), dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _quantile_nogil_chunk(arr, result, q_arr, interpolation, start_col, end_col)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


# ============================================================================
# DISPATCH FUNCTION
# ============================================================================

def _quantile_dispatch(arr, q_arr, interpolation, axis):
    """Dispatch quantile to appropriate implementation based on size.

    Args:
        arr: 2D float64 array
        q_arr: 1D float64 array of quantile values
        interpolation: int code for interpolation method
        axis: 0 for column quantiles, 1 for row quantiles

    Returns:
        2D array (n_quantiles, n_cols) for axis=0
        2D array (n_quantiles, n_rows) for axis=1 (after transpose)
    """
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _quantile_threadpool(arr, q_arr, interpolation)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _quantile_parallel(arr, q_arr, interpolation)
    else:
        result = _quantile_serial(arr, q_arr, interpolation)

    return result


# ============================================================================
# PANDAS WRAPPER FUNCTION
# ============================================================================

def optimized_quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear', method='single'):
    """Optimized quantile computation for DataFrame.

    Computes values at the given quantile(s) over requested axis.

    Args:
        self: DataFrame
        q: float or array-like, default 0.5
            Quantile(s) to compute. Must be between 0 and 1 inclusive.
        axis: {0 or 'index', 1 or 'columns'}, default 0
            - 0: compute quantile for each column
            - 1: compute quantile for each row
        numeric_only: bool, default True
            Include only float, int, boolean columns.
        interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile lies between two data points.
        method: str, default 'single'
            Ignored - for pandas compatibility only.

    Returns:
        Series or DataFrame
            - If q is a single quantile, returns Series
            - If q is a list of quantiles, returns DataFrame
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Normalize axis
    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Map interpolation string to int for Numba
    interp_map = {
        'linear': INTERP_LINEAR,
        'lower': INTERP_LOWER,
        'higher': INTERP_HIGHER,
        'nearest': INTERP_NEAREST,
        'midpoint': INTERP_MIDPOINT
    }
    interp_code = interp_map.get(interpolation, INTERP_LINEAR)

    # Normalize q to array and track if scalar
    scalar_q = np.isscalar(q)
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))

    # Validate q values
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("Quantile values must be between 0 and 1 inclusive")

    # Get numeric columns
    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Convert to float64 for Numba
    arr = ensure_float64(numeric_df.values)

    # Dispatch to appropriate implementation
    result = _quantile_dispatch(arr, q_arr, interp_code, axis)

    # Build output
    if axis == 0:
        # Column quantiles
        if scalar_q:
            return pd.Series(result[0], index=numeric_cols, name=q)
        else:
            return pd.DataFrame(result, index=q_arr, columns=numeric_cols)
    else:
        # Row quantiles
        if scalar_q:
            return pd.Series(result[0], index=self.index, name=q)
        else:
            return pd.DataFrame(result, index=q_arr, columns=self.index)


# ============================================================================
# PATCH REGISTRATION
# ============================================================================

def apply_quantile_patches():
    """Apply quantile operation patches to pandas DataFrame.

    These patches provide optimized implementations of quantile computation
    using Numba-accelerated kernels with 3-tier dispatch (serial, parallel,
    threadpool).
    """
    from .._patch import patch

    patch(pd.DataFrame, 'quantile', optimized_quantile)
