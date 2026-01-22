"""Parallel aggregation operations using Numba nogil kernels with thread pooling.

This module provides parallelized reduction operations (sum, mean, std, var,
min, max, median, prod) by distributing columns across threads using Numba's
nogil=True to release the GIL for true parallel execution.

Operations support axis parameter:
- axis=0: Reduce rows -> output shape (n_cols,) -> Series indexed by columns
- axis=1: Reduce columns -> output shape (n_rows,) -> Series indexed by original index
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Union
from numba import njit, prange
import os

from .._compat import get_numeric_columns_fast, ensure_float64

# Thresholds for parallel execution dispatch
PARALLEL_THRESHOLD = 500_000      # Use parallel prange above this
THREADPOOL_THRESHOLD = 10_000_000 # Use ThreadPool+nogil above this

# Optimal worker count
_CPU_COUNT = os.cpu_count() or 8
THREADPOOL_WORKERS = min(_CPU_COUNT, 32)


# ============================================================================
# SUM OPERATION
# ============================================================================

@njit(cache=True)
def _sum_serial(arr, skipna):
    """Serial sum reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    total = np.nan
                    break
            else:
                total += val
                count += 1
        if count > 0 or not skipna:
            result[c] = total
        else:
            result[c] = 0.0  # pandas returns 0 for all-NaN with skipna=True

    return result


@njit(parallel=True, cache=True)
def _sum_parallel(arr, skipna):
    """Parallel sum reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    total = np.nan
                    break
            else:
                total += val
                count += 1
        if count > 0 or not skipna:
            result[c] = total
        else:
            result[c] = 0.0

    return result


@njit(nogil=True, cache=True)
def _sum_nogil_chunk(arr, result, start_col, end_col, skipna):
    """Sum reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    total = np.nan
                    break
            else:
                total += val
                count += 1
        if count > 0 or not skipna:
            result[c] = total
        else:
            result[c] = 0.0


def _sum_threadpool(arr, skipna):
    """ThreadPool sum using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _sum_nogil_chunk(arr, result, start_col, end_col, skipna)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _sum_dispatch(arr, skipna, axis):
    """Dispatch sum to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _sum_threadpool(arr, skipna)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _sum_parallel(arr, skipna)
    else:
        result = _sum_serial(arr, skipna)

    return result


# ============================================================================
# MEAN OPERATION
# ============================================================================

@njit(cache=True)
def _mean_serial(arr, skipna):
    """Serial mean reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    total = np.nan
                    count = 1  # Avoid division by zero
                    break
            else:
                total += val
                count += 1
        result[c] = total / count if count > 0 else np.nan

    return result


@njit(parallel=True, cache=True)
def _mean_parallel(arr, skipna):
    """Parallel mean reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    total = np.nan
                    count = 1
                    break
            else:
                total += val
                count += 1
        result[c] = total / count if count > 0 else np.nan

    return result


@njit(nogil=True, cache=True)
def _mean_nogil_chunk(arr, result, start_col, end_col, skipna):
    """Mean reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    total = np.nan
                    count = 1
                    break
            else:
                total += val
                count += 1
        result[c] = total / count if count > 0 else np.nan


def _mean_threadpool(arr, skipna):
    """ThreadPool mean using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _mean_nogil_chunk(arr, result, start_col, end_col, skipna)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _mean_dispatch(arr, skipna, axis):
    """Dispatch mean to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _mean_threadpool(arr, skipna)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _mean_parallel(arr, skipna)
    else:
        result = _mean_serial(arr, skipna)

    return result


# ============================================================================
# STD/VAR OPERATIONS (Welford's Algorithm for Numerical Stability)
# ============================================================================

@njit(cache=True)
def _var_serial(arr, skipna, ddof):
    """Serial variance reduction along axis=0 using Welford's algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        mean = 0.0
        M2 = 0.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

        if nan_found:
            result[c] = np.nan
        elif count > ddof:
            result[c] = M2 / (count - ddof)
        else:
            result[c] = np.nan

    return result


@njit(parallel=True, cache=True)
def _var_parallel(arr, skipna, ddof):
    """Parallel variance reduction along axis=0 using prange and Welford's."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        mean = 0.0
        M2 = 0.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

        if nan_found:
            result[c] = np.nan
        elif count > ddof:
            result[c] = M2 / (count - ddof)
        else:
            result[c] = np.nan

    return result


@njit(nogil=True, cache=True)
def _var_nogil_chunk(arr, result, start_col, end_col, skipna, ddof):
    """Variance reduction using Welford's algorithm - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        mean = 0.0
        M2 = 0.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

        if nan_found:
            result[c] = np.nan
        elif count > ddof:
            result[c] = M2 / (count - ddof)
        else:
            result[c] = np.nan


def _var_threadpool(arr, skipna, ddof):
    """ThreadPool variance using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _var_nogil_chunk(arr, result, start_col, end_col, skipna, ddof)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _var_dispatch(arr, skipna, axis, ddof):
    """Dispatch variance to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _var_threadpool(arr, skipna, ddof)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _var_parallel(arr, skipna, ddof)
    else:
        result = _var_serial(arr, skipna, ddof)

    return result


@njit(cache=True)
def _std_serial(arr, skipna, ddof):
    """Serial std reduction - sqrt of variance."""
    var_result = _var_serial(arr, skipna, ddof)
    return np.sqrt(var_result)


@njit(parallel=True, cache=True)
def _std_parallel(arr, skipna, ddof):
    """Parallel std reduction - sqrt of variance."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        mean = 0.0
        M2 = 0.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

        if nan_found:
            result[c] = np.nan
        elif count > ddof:
            result[c] = np.sqrt(M2 / (count - ddof))
        else:
            result[c] = np.nan

    return result


@njit(nogil=True, cache=True)
def _std_nogil_chunk(arr, result, start_col, end_col, skipna, ddof):
    """Std reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        mean = 0.0
        M2 = 0.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

        if nan_found:
            result[c] = np.nan
        elif count > ddof:
            result[c] = np.sqrt(M2 / (count - ddof))
        else:
            result[c] = np.nan


def _std_threadpool(arr, skipna, ddof):
    """ThreadPool std using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _std_nogil_chunk(arr, result, start_col, end_col, skipna, ddof)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _std_dispatch(arr, skipna, axis, ddof):
    """Dispatch std to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _std_threadpool(arr, skipna, ddof)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _std_parallel(arr, skipna, ddof)
    else:
        result = _std_serial(arr, skipna, ddof)

    return result


# ============================================================================
# MIN OPERATION
# ============================================================================

@njit(cache=True)
def _min_serial(arr, skipna):
    """Serial min reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        min_val = np.inf
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                if val < min_val:
                    min_val = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count > 0:
            result[c] = min_val
        else:
            result[c] = np.nan

    return result


@njit(parallel=True, cache=True)
def _min_parallel(arr, skipna):
    """Parallel min reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        min_val = np.inf
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                if val < min_val:
                    min_val = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count > 0:
            result[c] = min_val
        else:
            result[c] = np.nan

    return result


@njit(nogil=True, cache=True)
def _min_nogil_chunk(arr, result, start_col, end_col, skipna):
    """Min reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        min_val = np.inf
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                if val < min_val:
                    min_val = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count > 0:
            result[c] = min_val
        else:
            result[c] = np.nan


def _min_threadpool(arr, skipna):
    """ThreadPool min using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _min_nogil_chunk(arr, result, start_col, end_col, skipna)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _min_dispatch(arr, skipna, axis):
    """Dispatch min to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _min_threadpool(arr, skipna)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _min_parallel(arr, skipna)
    else:
        result = _min_serial(arr, skipna)

    return result


# ============================================================================
# MAX OPERATION
# ============================================================================

@njit(cache=True)
def _max_serial(arr, skipna):
    """Serial max reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        max_val = -np.inf
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                if val > max_val:
                    max_val = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count > 0:
            result[c] = max_val
        else:
            result[c] = np.nan

    return result


@njit(parallel=True, cache=True)
def _max_parallel(arr, skipna):
    """Parallel max reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        max_val = -np.inf
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                if val > max_val:
                    max_val = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count > 0:
            result[c] = max_val
        else:
            result[c] = np.nan

    return result


@njit(nogil=True, cache=True)
def _max_nogil_chunk(arr, result, start_col, end_col, skipna):
    """Max reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        max_val = -np.inf
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                if val > max_val:
                    max_val = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count > 0:
            result[c] = max_val
        else:
            result[c] = np.nan


def _max_threadpool(arr, skipna):
    """ThreadPool max using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _max_nogil_chunk(arr, result, start_col, end_col, skipna)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _max_dispatch(arr, skipna, axis):
    """Dispatch max to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _max_threadpool(arr, skipna)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _max_parallel(arr, skipna)
    else:
        result = _max_serial(arr, skipna)

    return result


# ============================================================================
# MEDIAN OPERATION
# ============================================================================

@njit(cache=True)
def _median_serial(arr, skipna):
    """Serial median reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        # Collect non-NaN values
        values = np.empty(n_rows, dtype=np.float64)
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                values[count] = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count == 0:
            result[c] = np.nan
        else:
            # Sort and find median
            valid_values = values[:count]
            valid_values = np.sort(valid_values)
            mid = count // 2
            if count % 2 == 0:
                result[c] = (valid_values[mid - 1] + valid_values[mid]) / 2.0
            else:
                result[c] = valid_values[mid]

    return result


@njit(parallel=True, cache=True)
def _median_parallel(arr, skipna):
    """Parallel median reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        values = np.empty(n_rows, dtype=np.float64)
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                values[count] = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count == 0:
            result[c] = np.nan
        else:
            valid_values = values[:count]
            valid_values = np.sort(valid_values)
            mid = count // 2
            if count % 2 == 0:
                result[c] = (valid_values[mid - 1] + valid_values[mid]) / 2.0
            else:
                result[c] = valid_values[mid]

    return result


@njit(nogil=True, cache=True)
def _median_nogil_chunk(arr, result, start_col, end_col, skipna):
    """Median reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        values = np.empty(n_rows, dtype=np.float64)
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                values[count] = val
                count += 1

        if nan_found:
            result[c] = np.nan
        elif count == 0:
            result[c] = np.nan
        else:
            valid_values = values[:count]
            valid_values = np.sort(valid_values)
            mid = count // 2
            if count % 2 == 0:
                result[c] = (valid_values[mid - 1] + valid_values[mid]) / 2.0
            else:
                result[c] = valid_values[mid]


def _median_threadpool(arr, skipna):
    """ThreadPool median using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _median_nogil_chunk(arr, result, start_col, end_col, skipna)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _median_dispatch(arr, skipna, axis):
    """Dispatch median to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _median_threadpool(arr, skipna)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _median_parallel(arr, skipna)
    else:
        result = _median_serial(arr, skipna)

    return result


# ============================================================================
# PROD OPERATION
# ============================================================================

@njit(cache=True)
def _prod_serial(arr, skipna):
    """Serial prod reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in range(n_cols):
        prod_val = 1.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                prod_val *= val
                count += 1

        if nan_found:
            result[c] = np.nan
        else:
            result[c] = prod_val  # pandas returns 1.0 for empty, which is what we have

    return result


@njit(parallel=True, cache=True)
def _prod_parallel(arr, skipna):
    """Parallel prod reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for c in prange(n_cols):
        prod_val = 1.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                prod_val *= val
                count += 1

        if nan_found:
            result[c] = np.nan
        else:
            result[c] = prod_val

    return result


@njit(nogil=True, cache=True)
def _prod_nogil_chunk(arr, result, start_col, end_col, skipna):
    """Prod reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        prod_val = 1.0
        count = 0
        nan_found = False

        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    nan_found = True
                    break
            else:
                prod_val *= val
                count += 1

        if nan_found:
            result[c] = np.nan
        else:
            result[c] = prod_val


def _prod_threadpool(arr, skipna):
    """ThreadPool prod using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _prod_nogil_chunk(arr, result, start_col, end_col, skipna)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _prod_dispatch(arr, skipna, axis):
    """Dispatch prod to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _prod_threadpool(arr, skipna)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _prod_parallel(arr, skipna)
    else:
        result = _prod_serial(arr, skipna)

    return result


# ============================================================================
# QUANTILE OPERATION
# ============================================================================

@njit(cache=True)
def _quantile_column(arr: np.ndarray, q: float) -> float:
    """Calculate quantile for a single column, handling NaN."""
    # Filter out NaN values
    n = len(arr)
    valid_count = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            valid_count += 1

    if valid_count == 0:
        return np.nan

    # Copy valid values
    valid = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            valid[idx] = arr[i]
            idx += 1

    # Sort
    valid.sort()

    # Calculate quantile using linear interpolation
    # Position in sorted array
    pos = q * (valid_count - 1)
    lower_idx = int(pos)
    upper_idx = lower_idx + 1

    if upper_idx >= valid_count:
        return valid[valid_count - 1]

    # Linear interpolation
    frac = pos - lower_idx
    return valid[lower_idx] * (1 - frac) + valid[upper_idx] * frac


@njit(parallel=True, cache=True)
def _quantile_parallel(arr: np.ndarray, q: float) -> np.ndarray:
    """Calculate quantile for each column - parallelized."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in prange(n_cols):
        col_data = arr[:, col].copy()
        result[col] = _quantile_column(col_data, q)
    return result


@njit(cache=True)
def _quantile_serial(arr: np.ndarray, q: float) -> np.ndarray:
    """Calculate quantile for each column - serial version."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in range(n_cols):
        col_data = arr[:, col].copy()
        result[col] = _quantile_column(col_data, q)
    return result


@njit(nogil=True, cache=True)
def _quantile_nogil_chunk(arr, result, start_col, end_col, q):
    """Quantile reduction - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        col_data = arr[:, c].copy()
        result[c] = _quantile_column(col_data, q)


def _quantile_threadpool(arr, q):
    """ThreadPool quantile using nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _quantile_nogil_chunk(arr, result, start_col, end_col, q)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _quantile_dispatch(arr, q, axis):
    """Dispatch quantile to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= THREADPOOL_THRESHOLD:
        result = _quantile_threadpool(arr, q)
    elif n_elements >= PARALLEL_THRESHOLD:
        result = _quantile_parallel(arr, q)
    else:
        result = _quantile_serial(arr, q)

    return result


# ============================================================================
# ALL OPERATION
# ============================================================================

@njit(cache=True)
def _all_serial(arr, skipna):
    """Serial all reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.bool_)

    for c in range(n_cols):
        all_true = True
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    all_true = False
                    break
                # If skipna=True, skip NaN values
            elif val == 0.0:
                all_true = False
                break
        result[c] = all_true

    return result


@njit(parallel=True, cache=True)
def _all_parallel(arr, skipna):
    """Parallel all reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.bool_)

    for c in prange(n_cols):
        all_true = True
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    all_true = False
                    break
            elif val == 0.0:
                all_true = False
                break
        result[c] = all_true

    return result


def _all_dispatch(arr, skipna, axis):
    """Dispatch all to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= PARALLEL_THRESHOLD:
        result = _all_parallel(arr, skipna)
    else:
        result = _all_serial(arr, skipna)

    return result


# ============================================================================
# ANY OPERATION
# ============================================================================

@njit(cache=True)
def _any_serial(arr, skipna):
    """Serial any reduction along axis=0."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.bool_)

    for c in range(n_cols):
        any_true = False
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    any_true = True  # pandas: NaN is truthy with skipna=False
                    break
            elif val != 0.0:
                any_true = True
                break
        result[c] = any_true

    return result


@njit(parallel=True, cache=True)
def _any_parallel(arr, skipna):
    """Parallel any reduction along axis=0 using prange."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.bool_)

    for c in prange(n_cols):
        any_true = False
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                if not skipna:
                    any_true = True
                    break
            elif val != 0.0:
                any_true = True
                break
        result[c] = any_true

    return result


def _any_dispatch(arr, skipna, axis):
    """Dispatch any to appropriate implementation based on size."""
    if axis == 1:
        arr = arr.T

    n_elements = arr.size
    if n_elements >= PARALLEL_THRESHOLD:
        result = _any_parallel(arr, skipna)
    else:
        result = _any_serial(arr, skipna)

    return result


# ============================================================================
# PANDAS WRAPPER FUNCTIONS
# ============================================================================

def optimized_sum(self, axis=0, skipna=True, numeric_only=False, min_count=0, **kwargs):
    """Optimized sum reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Normalize axis
    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return 0.0 for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(0.0, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _sum_dispatch(arr, skipna, axis)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_mean(self, axis=0, skipna=True, numeric_only=False, **kwargs):
    """Optimized mean reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return nan for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(np.nan, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _mean_dispatch(arr, skipna, axis)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_std(self, axis=0, skipna=True, ddof=1, numeric_only=False, **kwargs):
    """Optimized std reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return nan for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(np.nan, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _std_dispatch(arr, skipna, axis, ddof)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_var(self, axis=0, skipna=True, ddof=1, numeric_only=False, **kwargs):
    """Optimized var reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return nan for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(np.nan, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _var_dispatch(arr, skipna, axis, ddof)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_min(self, axis=0, skipna=True, numeric_only=False, **kwargs):
    """Optimized min reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return nan for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(np.nan, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _min_dispatch(arr, skipna, axis)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_max(self, axis=0, skipna=True, numeric_only=False, **kwargs):
    """Optimized max reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return nan for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(np.nan, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _max_dispatch(arr, skipna, axis)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_median(self, axis=0, skipna=True, numeric_only=False, **kwargs):
    """Optimized median reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle 0x0 empty DataFrame - MUST return empty Series like pandas
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=np.float64)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    # Handle 0 rows with columns - return nan for each column (pandas behavior)
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(np.nan, index=numeric_cols, dtype=np.float64)
        else:
            return pd.Series([], dtype=np.float64)

    arr = ensure_float64(numeric_df.values)
    result = _median_dispatch(arr, skipna, axis)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_prod(self, axis=0, skipna=True, numeric_only=False, min_count=0, **kwargs):
    """Optimized prod reduction for DataFrame."""
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    arr = ensure_float64(numeric_df.values)
    result = _prod_dispatch(arr, skipna, axis)

    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear', method='single'):
    """Optimized quantile computation.

    Parameters
    ----------
    self : DataFrame
        Input DataFrame
    q : float or array-like, default 0.5
        Quantile(s) to compute (0 <= q <= 1)
    axis : int, default 0
        0 for column-wise, 1 for row-wise
    numeric_only : bool, default True
        Include only numeric columns
    interpolation : str, default 'linear'
        Interpolation method (only 'linear' optimized)
    method : str, default 'single'
        Method for pandas compatibility (ignored in optimized version)

    Returns
    -------
    Series or DataFrame
        Quantile values
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if interpolation != 'linear':
        raise TypeError(f"Only 'linear' interpolation optimized, got '{interpolation}'")

    # Normalize axis
    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1

    # Handle empty DataFrame
    if self.empty and self.shape[1] == 0:
        if isinstance(q, (list, np.ndarray)):
            return pd.DataFrame()
        # Use Index with dtype='object' to match pandas behavior
        return pd.Series([], dtype=np.float64, index=pd.Index([], dtype='object'), name=q)

    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")

    arr = ensure_float64(numeric_df.values)

    # Handle single quantile vs multiple quantiles
    if isinstance(q, (list, np.ndarray)):
        # Multiple quantiles - return DataFrame
        q_arr = np.asarray(q, dtype=np.float64)
        results = []
        for q_val in q_arr:
            result_arr = _quantile_dispatch(arr, q_val, axis)
            if axis == 0:
                results.append(pd.Series(result_arr, index=numeric_cols, name=q_val))
            else:
                results.append(pd.Series(result_arr, index=self.index, name=q_val))
        return pd.DataFrame(results)
    else:
        # Single quantile - return Series
        q_val = float(q)
        result_arr = _quantile_dispatch(arr, q_val, axis)
        if axis == 0:
            return pd.Series(result_arr, index=numeric_cols, name=q_val)
        else:
            return pd.Series(result_arr, index=self.index, name=q_val)


def optimized_all(self, axis=0, bool_only=None, skipna=True, **kwargs):
    """Optimized boolean all operation.
    
    Parameters
    ----------
    self : DataFrame
        Input DataFrame
    axis : int, default 0
        0 for column-wise, 1 for row-wise
    bool_only : ignored
    skipna : bool, default True
        Exclude NaN values
        
    Returns
    -------
    Series
        Boolean results
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")
    
    # Normalize axis
    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1
    
    # Handle empty DataFrame
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=bool)
    
    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")
    
    # Handle 0 rows with columns
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(True, index=numeric_cols, dtype=bool)
        else:
            return pd.Series([], dtype=bool)
    
    arr = ensure_float64(numeric_df.values)
    result = _all_dispatch(arr, skipna, axis)
    
    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


def optimized_any(self, axis=0, bool_only=None, skipna=True, **kwargs):
    """Optimized boolean any operation.
    
    Parameters
    ----------
    self : DataFrame
        Input DataFrame
    axis : int, default 0
        0 for column-wise, 1 for row-wise
    bool_only : ignored
    skipna : bool, default True
        Exclude NaN values
        
    Returns
    -------
    Series
        Boolean results
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")
    
    # Normalize axis
    if axis is None or axis == 'index':
        axis = 0
    elif axis == 'columns':
        axis = 1
    
    # Handle empty DataFrame
    if self.empty and self.shape[1] == 0:
        return pd.Series([], dtype=bool)
    
    numeric_cols, numeric_df = get_numeric_columns_fast(self)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns")
    
    # Handle 0 rows with columns
    if numeric_df.shape[0] == 0:
        if axis == 0:
            return pd.Series(False, index=numeric_cols, dtype=bool)
        else:
            return pd.Series([], dtype=bool)
    
    arr = ensure_float64(numeric_df.values)
    result = _any_dispatch(arr, skipna, axis)
    
    if axis == 0:
        return pd.Series(result, index=numeric_cols)
    else:
        return pd.Series(result, index=self.index)


# ============================================================================
# PATCH REGISTRATION
# ============================================================================

def apply_aggregation_patches():
    """Apply aggregation operation patches to pandas DataFrame.

    These patches provide optimized implementations of reduction operations
    (sum, mean, std, var, min, max, median, prod, quantile, all, any) using Numba-accelerated
    kernels with 3-tier dispatch (serial, parallel, threadpool).
    """
    from .._patch import patch

    patch(pd.DataFrame, 'sum', optimized_sum)
    patch(pd.DataFrame, 'mean', optimized_mean)
    patch(pd.DataFrame, 'std', optimized_std)
    patch(pd.DataFrame, 'var', optimized_var)
    patch(pd.DataFrame, 'min', optimized_min)
    patch(pd.DataFrame, 'max', optimized_max)
    patch(pd.DataFrame, 'median', optimized_median)
    patch(pd.DataFrame, 'prod', optimized_prod)
    patch(pd.DataFrame, 'quantile', optimized_quantile)
    patch(pd.DataFrame, 'all', optimized_all)
    patch(pd.DataFrame, 'any', optimized_any)
