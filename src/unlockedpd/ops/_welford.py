"""Welford's algorithm for numerically stable rolling variance/std.

This provides O(n) complexity instead of O(n*w) for large windows.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def rolling_std_welford_parallel(arr: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    """Rolling std using Welford's online algorithm - O(n) per column.

    Welford's algorithm maintains running mean and M2 (sum of squared deviations)
    that can be updated incrementally as values enter/leave the window.
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0  # Sum of squared deviations from mean

        for row in range(n_rows):
            val = arr[row, col]

            # Add new value using Welford's update
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

            # Remove old value when past window (reverse Welford)
            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    count -= 1
                    if count > 0:
                        delta = old_val - mean
                        mean -= delta / count
                        delta2 = old_val - mean
                        M2 -= delta * delta2
                    else:
                        mean = 0.0
                        M2 = 0.0

            # Compute result
            if count >= min_periods and count > ddof:
                result[row, col] = np.sqrt(M2 / (count - ddof))

    return result


@njit(cache=True)
def rolling_std_welford_serial(arr: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    """Serial version for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in range(n_cols):  # range, not prange
        count = 0
        mean = 0.0
        M2 = 0.0

        for row in range(n_rows):
            val = arr[row, col]

            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    count -= 1
                    if count > 0:
                        delta = old_val - mean
                        mean -= delta / count
                        delta2 = old_val - mean
                        M2 -= delta * delta2
                    else:
                        mean = 0.0
                        M2 = 0.0

            if count >= min_periods and count > ddof:
                result[row, col] = np.sqrt(M2 / (count - ddof))

    return result


@njit(parallel=True, cache=True)
def rolling_var_welford_parallel(arr: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    """Rolling variance using Welford's algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0

        for row in range(n_rows):
            val = arr[row, col]

            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    count -= 1
                    if count > 0:
                        delta = old_val - mean
                        mean -= delta / count
                        delta2 = old_val - mean
                        M2 -= delta * delta2
                    else:
                        mean = 0.0
                        M2 = 0.0

            if count >= min_periods and count > ddof:
                result[row, col] = M2 / (count - ddof)

    return result


@njit(cache=True)
def rolling_var_welford_serial(arr: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    """Serial version for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in range(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0

        for row in range(n_rows):
            val = arr[row, col]

            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    count -= 1
                    if count > 0:
                        delta = old_val - mean
                        mean -= delta / count
                        delta2 = old_val - mean
                        M2 -= delta * delta2
                    else:
                        mean = 0.0
                        M2 = 0.0

            if count >= min_periods and count > ddof:
                result[row, col] = M2 / (count - ddof)

    return result
