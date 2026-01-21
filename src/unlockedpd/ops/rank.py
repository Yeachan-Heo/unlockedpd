"""Parallel rank operations using Numba.

This module provides Numba-accelerated ranking operations
that parallelize across rows (axis=1) or columns (axis=0).
"""
import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union
from concurrent.futures import ThreadPoolExecutor

from .._compat import get_numeric_columns, wrap_result, ensure_float64, ensure_optimal_layout
from .._config import PARALLEL_THRESHOLD, THREADPOOL_THRESHOLD, config


# na_option encoding: 0='keep', 1='top', 2='bottom'
NA_OPTION_MAP = {'keep': 0, 'top': 1, 'bottom': 2}


@njit(cache=True)
def _apply_na_option(ranks: np.ndarray, is_nan: np.ndarray, na_option: int) -> np.ndarray:
    """Apply na_option to ranks array.

    na_option encoding: 0='keep', 1='top', 2='bottom'
    """
    n = len(ranks)
    nan_count = 0
    for i in range(n):
        if is_nan[i]:
            nan_count += 1

    if na_option == 0:  # 'keep' - NaN positions already have NaN ranks
        pass
    elif na_option == 1:  # 'top' - NaNs get lowest ranks
        # Shift valid ranks up by nan_count
        for i in range(n):
            if not is_nan[i]:
                ranks[i] += nan_count
        # Assign lowest ranks to NaNs
        rank = 1.0
        for i in range(n):
            if is_nan[i]:
                ranks[i] = rank
                rank += 1.0
    elif na_option == 2:  # 'bottom' - NaNs get highest ranks
        # Valid ranks stay as-is (1..n-nan_count)
        # Assign highest ranks to NaNs
        rank = float(n - nan_count + 1)
        for i in range(n):
            if is_nan[i]:
                ranks[i] = rank
                rank += 1.0

    return ranks


# =============================================================================
# Parallel implementations (for large arrays)
# =============================================================================

@njit(parallel=True, cache=True)
def _rank_axis1_average(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 (across columns) using average method."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in prange(n_rows):
        row_data = arr[row, :].copy()

        # Track NaN positions
        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                # Replace NaN with large value for sorting (will be handled after)
                row_data[col] = np.inf if ascending else -np.inf

        # Get sort indices
        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        # Assign ranks (handling ties with average method)
        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            # Find all tied values (excluding NaN placeholders)
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                # Don't group NaN placeholders as ties
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            # Average rank for tied values
            avg_rank = (i + j + 2) / 2  # +2 because ranks are 1-based
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1

        # Set NaN positions to NaN (for 'keep' default)
        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        # Apply na_option
        ranks = _apply_na_option(ranks, is_nan, na_option)

        result[row, :] = ranks

    return result


@njit(parallel=True, cache=True)
def _rank_axis0_average(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=0 (down columns) using average method."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        col_data = arr[:, col].copy()

        # Track NaN positions
        is_nan = np.empty(n_rows, dtype=np.bool_)
        for row in range(n_rows):
            is_nan[row] = np.isnan(col_data[row])
            if is_nan[row]:
                col_data[row] = np.inf if ascending else -np.inf

        # Get sort indices
        if ascending:
            sorted_idx = np.argsort(col_data)
        else:
            sorted_idx = np.argsort(-col_data)

        # Assign ranks (handling ties with average method)
        ranks = np.empty(n_rows, dtype=np.float64)
        i = 0
        while i < n_rows:
            j = i
            while j < n_rows - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if col_data[idx_j] != col_data[idx_j1]:
                    break
                j += 1

            avg_rank = (i + j + 2) / 2
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1

        # Set NaN positions to NaN
        for row in range(n_rows):
            if is_nan[row]:
                ranks[row] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[:, col] = ranks

    return result


@njit(parallel=True, cache=True)
def _rank_axis1_min(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using min method (ties get minimum rank)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in prange(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            # Min rank for tied values
            min_rank = float(i + 1)  # 1-based
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = min_rank
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(parallel=True, cache=True)
def _rank_axis1_max(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using max method (ties get maximum rank)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in prange(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            # Max rank for tied values
            max_rank = float(j + 1)  # 1-based
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = max_rank
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(parallel=True, cache=True)
def _rank_axis1_first(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using first method (ties broken by order)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in prange(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        for i in range(n_cols):
            ranks[sorted_idx[i]] = float(i + 1)

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(parallel=True, cache=True)
def _rank_axis1_dense(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using dense method (no gaps in ranks)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in prange(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        current_rank = 1.0
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = current_rank
            current_rank += 1.0
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


# =============================================================================
# Serial implementations (for small arrays - avoids parallel overhead)
# =============================================================================

@njit(cache=True)
def _rank_axis1_average_serial(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using average method (serial version)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            avg_rank = (i + j + 2) / 2
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(cache=True)
def _rank_axis0_average_serial(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=0 using average method (serial version)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in range(n_cols):
        col_data = arr[:, col].copy()

        is_nan = np.empty(n_rows, dtype=np.bool_)
        for row in range(n_rows):
            is_nan[row] = np.isnan(col_data[row])
            if is_nan[row]:
                col_data[row] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(col_data)
        else:
            sorted_idx = np.argsort(-col_data)

        ranks = np.empty(n_rows, dtype=np.float64)
        i = 0
        while i < n_rows:
            j = i
            while j < n_rows - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if col_data[idx_j] != col_data[idx_j1]:
                    break
                j += 1

            avg_rank = (i + j + 2) / 2
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1

        for row in range(n_rows):
            if is_nan[row]:
                ranks[row] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[:, col] = ranks

    return result


@njit(cache=True)
def _rank_axis1_min_serial(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using min method (serial version)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            min_rank = float(i + 1)
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = min_rank
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(cache=True)
def _rank_axis1_max_serial(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using max method (serial version)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            max_rank = float(j + 1)
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = max_rank
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(cache=True)
def _rank_axis1_first_serial(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using first method (serial version)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        for i in range(n_cols):
            ranks[sorted_idx[i]] = float(i + 1)

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


@njit(cache=True)
def _rank_axis1_dense_serial(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using dense method (serial version)."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        row_data = arr[row, :].copy()

        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(row_data[col])
            if is_nan[col]:
                row_data[col] = np.inf if ascending else -np.inf

        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        ranks = np.empty(n_cols, dtype=np.float64)
        current_rank = 1.0
        i = 0
        while i < n_cols:
            j = i
            while j < n_cols - 1:
                idx_j = sorted_idx[j]
                idx_j1 = sorted_idx[j + 1]
                if is_nan[idx_j] or is_nan[idx_j1]:
                    break
                if row_data[idx_j] != row_data[idx_j1]:
                    break
                j += 1

            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = current_rank
            current_rank += 1.0
            i = j + 1

        for col in range(n_cols):
            if is_nan[col]:
                ranks[col] = np.nan

        ranks = _apply_na_option(ranks, is_nan, na_option)
        result[row, :] = ranks

    return result


# =============================================================================
# ThreadPool implementations (nogil kernels for medium-sized arrays)
# =============================================================================

@njit(cache=True, nogil=True)
def _rank_single_row_average(arr: np.ndarray, ascending: bool, na_option: int) -> np.ndarray:
    """Rank a single row using average method (nogil kernel for ThreadPool)."""
    n_cols = len(arr)
    row_data = arr.copy()

    is_nan = np.empty(n_cols, dtype=np.bool_)
    for col in range(n_cols):
        is_nan[col] = np.isnan(row_data[col])
        if is_nan[col]:
            row_data[col] = np.inf if ascending else -np.inf

    if ascending:
        sorted_idx = np.argsort(row_data)
    else:
        sorted_idx = np.argsort(-row_data)

    ranks = np.empty(n_cols, dtype=np.float64)
    i = 0
    while i < n_cols:
        j = i
        while j < n_cols - 1:
            idx_j = sorted_idx[j]
            idx_j1 = sorted_idx[j + 1]
            if is_nan[idx_j] or is_nan[idx_j1]:
                break
            if row_data[idx_j] != row_data[idx_j1]:
                break
            j += 1

        avg_rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1

    for col in range(n_cols):
        if is_nan[col]:
            ranks[col] = np.nan

    return _apply_na_option(ranks, is_nan, na_option)


@njit(cache=True, nogil=True)
def _rank_single_col_average(arr: np.ndarray, ascending: bool, na_option: int) -> np.ndarray:
    """Rank a single column using average method (nogil kernel for ThreadPool)."""
    n_rows = len(arr)
    col_data = arr.copy()

    is_nan = np.empty(n_rows, dtype=np.bool_)
    for row in range(n_rows):
        is_nan[row] = np.isnan(col_data[row])
        if is_nan[row]:
            col_data[row] = np.inf if ascending else -np.inf

    if ascending:
        sorted_idx = np.argsort(col_data)
    else:
        sorted_idx = np.argsort(-col_data)

    ranks = np.empty(n_rows, dtype=np.float64)
    i = 0
    while i < n_rows:
        j = i
        while j < n_rows - 1:
            idx_j = sorted_idx[j]
            idx_j1 = sorted_idx[j + 1]
            if is_nan[idx_j] or is_nan[idx_j1]:
                break
            if col_data[idx_j] != col_data[idx_j1]:
                break
            j += 1

        avg_rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1

    for row in range(n_rows):
        if is_nan[row]:
            ranks[row] = np.nan

    return _apply_na_option(ranks, is_nan, na_option)


def _rank_axis1_average_threadpool(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 using ThreadPool."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    def process_row(row_idx):
        return row_idx, _rank_single_row_average(arr[row_idx, :], ascending, na_option)

    with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
        futures = [executor.submit(process_row, i) for i in range(n_rows)]
        for future in futures:
            row_idx, ranks = future.result()
            result[row_idx, :] = ranks

    return result


def _rank_axis0_average_threadpool(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=0 using ThreadPool."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    def process_col(col_idx):
        return col_idx, _rank_single_col_average(arr[:, col_idx], ascending, na_option)

    with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
        futures = [executor.submit(process_col, i) for i in range(n_cols)]
        for future in futures:
            col_idx, ranks = future.result()
            result[:, col_idx] = ranks

    return result


# =============================================================================
# Main entry point
# =============================================================================

def optimized_rank(
    df: pd.DataFrame,
    axis: int = 0,
    method: str = 'average',
    na_option: str = 'keep',
    ascending: bool = True,
    pct: bool = False
) -> pd.DataFrame:
    """Optimized DataFrame rank operation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    # Use optimal layout based on axis
    if axis == 1:
        arr = ensure_optimal_layout(arr, axis=1)  # C-contiguous for row ops
    else:
        arr = ensure_optimal_layout(arr, axis=0)  # F-contiguous for column ops

    na_opt = NA_OPTION_MAP.get(na_option, 0)

    # Select implementation based on size (3-tier: serial < threadpool < parallel)
    if arr.size >= PARALLEL_THRESHOLD:
        tier = 'parallel'
    elif arr.size >= THREADPOOL_THRESHOLD:
        tier = 'threadpool'
    else:
        tier = 'serial'

    # Select method and axis
    if axis == 1:
        if method == 'average':
            if tier == 'parallel':
                result = _rank_axis1_average(arr, ascending, na_opt)
            elif tier == 'threadpool':
                result = _rank_axis1_average_threadpool(arr, ascending, na_opt)
            else:
                result = _rank_axis1_average_serial(arr, ascending, na_opt)
        elif method == 'min':
            if tier == 'parallel':
                result = _rank_axis1_min(arr, ascending, na_opt)
            else:
                result = _rank_axis1_min_serial(arr, ascending, na_opt)
        elif method == 'max':
            if tier == 'parallel':
                result = _rank_axis1_max(arr, ascending, na_opt)
            else:
                result = _rank_axis1_max_serial(arr, ascending, na_opt)
        elif method == 'first':
            if tier == 'parallel':
                result = _rank_axis1_first(arr, ascending, na_opt)
            else:
                result = _rank_axis1_first_serial(arr, ascending, na_opt)
        elif method == 'dense':
            if tier == 'parallel':
                result = _rank_axis1_dense(arr, ascending, na_opt)
            else:
                result = _rank_axis1_dense_serial(arr, ascending, na_opt)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:  # axis == 0
        if method == 'average':
            if tier == 'parallel':
                result = _rank_axis0_average(arr, ascending, na_opt)
            elif tier == 'threadpool':
                result = _rank_axis0_average_threadpool(arr, ascending, na_opt)
            else:
                result = _rank_axis0_average_serial(arr, ascending, na_opt)
        else:
            # For other methods on axis=0, transpose and use axis=1 functions
            arr_t = np.ascontiguousarray(arr.T)  # Ensure C-contiguous for row operations
            if method == 'min':
                if tier == 'parallel':
                    result_t = _rank_axis1_min(arr_t, ascending, na_opt)
                else:
                    result_t = _rank_axis1_min_serial(arr_t, ascending, na_opt)
            elif method == 'max':
                if tier == 'parallel':
                    result_t = _rank_axis1_max(arr_t, ascending, na_opt)
                else:
                    result_t = _rank_axis1_max_serial(arr_t, ascending, na_opt)
            elif method == 'first':
                if tier == 'parallel':
                    result_t = _rank_axis1_first(arr_t, ascending, na_opt)
                else:
                    result_t = _rank_axis1_first_serial(arr_t, ascending, na_opt)
            elif method == 'dense':
                if tier == 'parallel':
                    result_t = _rank_axis1_dense(arr_t, ascending, na_opt)
                else:
                    result_t = _rank_axis1_dense_serial(arr_t, ascending, na_opt)
            else:
                raise ValueError(f"Unknown method: {method}")
            result = result_t.T

    if pct:
        # Convert to percentile ranks
        if axis == 1:
            n_valid = np.sum(~np.isnan(arr), axis=1, keepdims=True)
        else:
            n_valid = np.sum(~np.isnan(arr), axis=0, keepdims=True)
        result = result / n_valid

    return wrap_result(
        result, numeric_df, columns=numeric_cols,
        merge_non_numeric=True, original_df=df
    )


def _patched_rank(df, axis=0, method='average', numeric_only=False,
                  na_option='keep', ascending=True, pct=False):
    """Patched rank method for DataFrame."""
    return optimized_rank(df, axis=axis, method=method, na_option=na_option,
                          ascending=ascending, pct=pct)


def apply_rank_patches():
    """Apply rank operation patch to pandas."""
    from .._patch import patch

    patch(pd.DataFrame, 'rank', _patched_rank)
