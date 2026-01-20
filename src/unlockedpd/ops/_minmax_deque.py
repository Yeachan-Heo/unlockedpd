"""Monotonic deque algorithm for O(n) rolling min/max.

Instead of scanning the entire window each time (O(n*w)),
we maintain a deque of indices where values are monotonically
increasing (for min) or decreasing (for max).
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _rolling_min_1d_deque(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """O(n) rolling min for 1D array using monotonic deque."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    # Deque stores indices of potential minimums
    # Maintains monotonically increasing values
    deque_idx = np.empty(window + 1, dtype=np.int64)
    deque_front = 0
    deque_back = 0

    count = 0  # Count of non-NaN in window

    for i in range(n):
        val = arr[i]

        # Remove elements outside window from front
        while deque_front < deque_back and deque_idx[deque_front] <= i - window:
            old_idx = deque_idx[deque_front]
            if not np.isnan(arr[old_idx]):
                count -= 1
            deque_front += 1

        if not np.isnan(val):
            count += 1
            # Remove elements larger than current from back (maintain monotonic increasing)
            while deque_front < deque_back:
                back_idx = deque_idx[deque_back - 1]
                if arr[back_idx] > val:
                    deque_back -= 1
                else:
                    break

            # Add current index to back
            deque_idx[deque_back] = i
            deque_back += 1

        # Get result if enough values
        if count >= min_periods and deque_front < deque_back:
            result[i] = arr[deque_idx[deque_front]]

    return result


@njit(cache=True)
def _rolling_max_1d_deque(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """O(n) rolling max for 1D array using monotonic deque."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    # Deque stores indices of potential maximums
    # Maintains monotonically decreasing values
    deque_idx = np.empty(window + 1, dtype=np.int64)
    deque_front = 0
    deque_back = 0

    count = 0

    for i in range(n):
        val = arr[i]

        # Remove elements outside window from front
        while deque_front < deque_back and deque_idx[deque_front] <= i - window:
            old_idx = deque_idx[deque_front]
            if not np.isnan(arr[old_idx]):
                count -= 1
            deque_front += 1

        if not np.isnan(val):
            count += 1
            # Remove elements smaller than current from back (maintain monotonic decreasing)
            while deque_front < deque_back:
                back_idx = deque_idx[deque_back - 1]
                if arr[back_idx] < val:
                    deque_back -= 1
                else:
                    break

            deque_idx[deque_back] = i
            deque_back += 1

        if count >= min_periods and deque_front < deque_back:
            result[i] = arr[deque_idx[deque_front]]

    return result


@njit(parallel=True, cache=True)
def rolling_min_deque_parallel(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Parallel rolling min with O(n) per column."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        # Copy to contiguous array
        col_data = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            col_data[i] = arr[i, col]
        col_result = _rolling_min_1d_deque(col_data, window, min_periods)
        for i in range(n_rows):
            result[i, col] = col_result[i]

    return result


@njit(cache=True)
def rolling_min_deque_serial(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Serial rolling min for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in range(n_cols):
        col_data = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            col_data[i] = arr[i, col]
        col_result = _rolling_min_1d_deque(col_data, window, min_periods)
        for i in range(n_rows):
            result[i, col] = col_result[i]

    return result


@njit(parallel=True, cache=True)
def rolling_max_deque_parallel(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Parallel rolling max with O(n) per column."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        col_data = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            col_data[i] = arr[i, col]
        col_result = _rolling_max_1d_deque(col_data, window, min_periods)
        for i in range(n_rows):
            result[i, col] = col_result[i]

    return result


@njit(cache=True)
def rolling_max_deque_serial(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Serial rolling max for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in range(n_cols):
        col_data = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            col_data[i] = arr[i, col]
        col_result = _rolling_max_1d_deque(col_data, window, min_periods)
        for i in range(n_rows):
            result[i, col] = col_result[i]

    return result
