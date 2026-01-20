"""Parallel EWM (Exponentially Weighted Moving) operations using Numba.

This module provides Numba-accelerated EWM operations
that parallelize across columns for significant speedup on wide DataFrames.
"""
import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Optional

from .._compat import get_numeric_columns_fast, wrap_result, ensure_float64, ensure_optimal_layout

# Threshold for parallel vs serial execution (elements)
# Parallel overhead is ~1-2ms, so we need enough work to amortize it
PARALLEL_THRESHOLD = 500_000
THREADPOOL_THRESHOLD = 10_000_000  # 10M elements

import os
from concurrent.futures import ThreadPoolExecutor

_CPU_COUNT = os.cpu_count() or 8
THREADPOOL_WORKERS = min(_CPU_COUNT, 32)


# ============================================================================
# Helper function to compute alpha parameter
# ============================================================================

def _get_alpha(span=None, halflife=None, alpha=None, com=None):
    """Compute alpha from EWM parameters.

    Args:
        span: Specify decay in terms of span (N ≥ 1)
        halflife: Specify decay in terms of half-life (HL > 0)
        alpha: Specify smoothing factor α directly (0 < α ≤ 1)
        com: Specify decay in terms of center of mass (c ≥ 0)

    Returns:
        alpha: The computed alpha value

    Formulas:
        - alpha = 2 / (span + 1)  for span
        - alpha = 1 - exp(-ln(2) / halflife)  for halflife
        - alpha = 1 / (1 + com)  for com
    """
    # Count how many parameters are specified
    params = sum(x is not None for x in [span, halflife, alpha, com])

    if params == 0:
        raise ValueError("Must specify one of: span, halflife, alpha, or com")
    if params > 1:
        raise ValueError("Only one of span, halflife, alpha, or com should be specified")

    if alpha is not None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        return alpha
    elif span is not None:
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        return 2.0 / (span + 1.0)
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        return 1.0 - np.exp(-np.log(2.0) / halflife)
    elif com is not None:
        if com < 0:
            raise ValueError(f"com must be >= 0, got {com}")
        return 1.0 / (1.0 + com)


# ============================================================================
# Core Numba-jitted functions (PARALLEL versions)
# ============================================================================

@njit(parallel=True, cache=True)
def _ewm_mean_2d(arr: np.ndarray, alpha: float, adjust: bool, ignore_na: bool, min_periods: int) -> np.ndarray:
    """Compute EWM mean across columns in parallel.

    Args:
        arr: Input 2D array
        alpha: Smoothing factor (0 < alpha <= 1)
        adjust: If True, use adjusted formula; if False, use recursive formula
        ignore_na: If True, ignore NaN values in calculation
        min_periods: Minimum number of observations required

    Returns:
        2D array with EWM mean values
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        if adjust:
            # Adjusted EWM: y_t = (x_t + (1-alpha)*x_{t-1} + ...) / (1 + (1-alpha) + ...)
            weighted_sum = 0.0
            weight_sum = 0.0
            weight = 1.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        # Reset on NaN if not ignoring
                        weighted_sum = 0.0
                        weight_sum = 0.0
                        weight = 1.0
                        nobs = 0
                else:
                    weighted_sum += weight * val
                    weight_sum += weight
                    weight *= (1.0 - alpha)
                    nobs += 1

                    if nobs >= min_periods:
                        result[row, col] = weighted_sum / weight_sum
        else:
            # Recursive EWM: y_t = alpha * x_t + (1-alpha) * y_{t-1}
            ewm = 0.0
            is_first = True
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        # Reset on NaN if not ignoring
                        is_first = True
                        nobs = 0
                else:
                    if is_first:
                        ewm = val
                        is_first = False
                    else:
                        ewm = alpha * val + (1.0 - alpha) * ewm

                    nobs += 1
                    if nobs >= min_periods:
                        result[row, col] = ewm

    return result


@njit(parallel=True, cache=True)
def _ewm_var_2d(arr: np.ndarray, alpha: float, adjust: bool, ignore_na: bool, min_periods: int, bias: bool) -> np.ndarray:
    """Compute EWM variance across columns in parallel.

    Uses the formula: Var = EWM(x^2) - EWM(x)^2 with bias correction

    Args:
        arr: Input 2D array
        alpha: Smoothing factor
        adjust: If True, use adjusted formula
        ignore_na: If True, ignore NaN values
        min_periods: Minimum number of observations
        bias: If False, apply bias correction

    Returns:
        2D array with EWM variance values
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        if adjust:
            # Track both x and x^2
            weighted_sum = 0.0
            weighted_sum_sq = 0.0
            weight_sum = 0.0
            weight = 1.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        weighted_sum = 0.0
                        weighted_sum_sq = 0.0
                        weight_sum = 0.0
                        weight = 1.0
                        nobs = 0
                else:
                    weighted_sum += weight * val
                    weighted_sum_sq += weight * val * val
                    weight_sum += weight
                    weight *= (1.0 - alpha)
                    nobs += 1

                    if nobs >= min_periods:
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean

                        # Bias correction for adjusted method
                        if not bias and nobs > 1:
                            # Apply bias correction similar to pandas
                            var *= weight_sum / (weight_sum - 1.0 + alpha)

                        result[row, col] = max(0.0, var)  # Ensure non-negative
        else:
            # Recursive formula
            ewm = 0.0
            ewm_sq = 0.0
            is_first = True
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        is_first = True
                        nobs = 0
                else:
                    if is_first:
                        ewm = val
                        ewm_sq = val * val
                        is_first = False
                    else:
                        ewm = alpha * val + (1.0 - alpha) * ewm
                        ewm_sq = alpha * val * val + (1.0 - alpha) * ewm_sq

                    nobs += 1
                    if nobs >= min_periods:
                        var = ewm_sq - ewm * ewm

                        # Bias correction for recursive method
                        if not bias and nobs > 1:
                            var *= nobs / (nobs - 1.0)

                        result[row, col] = max(0.0, var)

    return result


@njit(parallel=True, cache=True)
def _ewm_std_2d(arr: np.ndarray, alpha: float, adjust: bool, ignore_na: bool, min_periods: int, bias: bool) -> np.ndarray:
    """Compute EWM standard deviation across columns in parallel.

    Simply the square root of EWM variance.

    Args:
        arr: Input 2D array
        alpha: Smoothing factor
        adjust: If True, use adjusted formula
        ignore_na: If True, ignore NaN values
        min_periods: Minimum number of observations
        bias: If False, apply bias correction

    Returns:
        2D array with EWM std values
    """
    var_result = _ewm_var_2d(arr, alpha, adjust, ignore_na, min_periods, bias)
    return np.sqrt(var_result)


# ============================================================================
# Core Numba-jitted functions (SERIAL versions for small arrays)
# ============================================================================

@njit(cache=True)
def _ewm_mean_2d_serial(arr: np.ndarray, alpha: float, adjust: bool, ignore_na: bool, min_periods: int) -> np.ndarray:
    """Serial EWM mean for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in range(n_cols):
        if adjust:
            weighted_sum = 0.0
            weight_sum = 0.0
            weight = 1.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        weighted_sum = 0.0
                        weight_sum = 0.0
                        weight = 1.0
                        nobs = 0
                else:
                    weighted_sum += weight * val
                    weight_sum += weight
                    weight *= (1.0 - alpha)
                    nobs += 1

                    if nobs >= min_periods:
                        result[row, col] = weighted_sum / weight_sum
        else:
            ewm = 0.0
            is_first = True
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        is_first = True
                        nobs = 0
                else:
                    if is_first:
                        ewm = val
                        is_first = False
                    else:
                        ewm = alpha * val + (1.0 - alpha) * ewm

                    nobs += 1
                    if nobs >= min_periods:
                        result[row, col] = ewm

    return result


@njit(cache=True)
def _ewm_var_2d_serial(arr: np.ndarray, alpha: float, adjust: bool, ignore_na: bool, min_periods: int, bias: bool) -> np.ndarray:
    """Serial EWM variance for small arrays."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in range(n_cols):
        if adjust:
            weighted_sum = 0.0
            weighted_sum_sq = 0.0
            weight_sum = 0.0
            weight = 1.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        weighted_sum = 0.0
                        weighted_sum_sq = 0.0
                        weight_sum = 0.0
                        weight = 1.0
                        nobs = 0
                else:
                    weighted_sum += weight * val
                    weighted_sum_sq += weight * val * val
                    weight_sum += weight
                    weight *= (1.0 - alpha)
                    nobs += 1

                    if nobs >= min_periods:
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean

                        if not bias and nobs > 1:
                            var *= weight_sum / (weight_sum - 1.0 + alpha)

                        result[row, col] = max(0.0, var)
        else:
            ewm = 0.0
            ewm_sq = 0.0
            is_first = True
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        is_first = True
                        nobs = 0
                else:
                    if is_first:
                        ewm = val
                        ewm_sq = val * val
                        is_first = False
                    else:
                        ewm = alpha * val + (1.0 - alpha) * ewm
                        ewm_sq = alpha * val * val + (1.0 - alpha) * ewm_sq

                    nobs += 1
                    if nobs >= min_periods:
                        var = ewm_sq - ewm * ewm

                        if not bias and nobs > 1:
                            var *= nobs / (nobs - 1.0)

                        result[row, col] = max(0.0, var)

    return result


@njit(cache=True)
def _ewm_std_2d_serial(arr: np.ndarray, alpha: float, adjust: bool, ignore_na: bool, min_periods: int, bias: bool) -> np.ndarray:
    """Serial EWM std for small arrays."""
    var_result = _ewm_var_2d_serial(arr, alpha, adjust, ignore_na, min_periods, bias)
    return np.sqrt(var_result)


# ============================================================================
# Nogil kernels for ThreadPool (GIL-released for true parallelism)
# ============================================================================

@njit(nogil=True, cache=True)
def _ewm_mean_nogil_chunk(arr, result, start_col, end_col, alpha, adjust, ignore_na, min_periods):
    """EWM mean - GIL released for ThreadPool parallelism."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        if adjust:
            weighted_sum = 0.0
            weight_sum = 0.0
            weight = 1.0
            nobs = 0
            for row in range(n_rows):
                val = arr[row, c]
                if np.isnan(val):
                    if not ignore_na:
                        weighted_sum = 0.0
                        weight_sum = 0.0
                        weight = 1.0
                        nobs = 0
                    result[row, c] = np.nan
                else:
                    weighted_sum += weight * val
                    weight_sum += weight
                    weight *= (1.0 - alpha)
                    nobs += 1
                    if nobs >= min_periods:
                        result[row, c] = weighted_sum / weight_sum
                    else:
                        result[row, c] = np.nan
        else:
            ewm = 0.0
            is_first = True
            nobs = 0
            for row in range(n_rows):
                val = arr[row, c]
                if np.isnan(val):
                    if not ignore_na:
                        is_first = True
                        nobs = 0
                    result[row, c] = np.nan
                else:
                    if is_first:
                        ewm = val
                        is_first = False
                    else:
                        ewm = alpha * val + (1.0 - alpha) * ewm
                    nobs += 1
                    if nobs >= min_periods:
                        result[row, c] = ewm
                    else:
                        result[row, c] = np.nan


@njit(nogil=True, cache=True)
def _ewm_var_nogil_chunk(arr, result, start_col, end_col, alpha, adjust, ignore_na, min_periods, bias):
    """EWM variance - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        if adjust:
            weighted_sum = 0.0
            weighted_sum_sq = 0.0
            weight_sum = 0.0
            weight = 1.0
            nobs = 0
            for row in range(n_rows):
                val = arr[row, c]
                if np.isnan(val):
                    if not ignore_na:
                        weighted_sum = 0.0
                        weighted_sum_sq = 0.0
                        weight_sum = 0.0
                        weight = 1.0
                        nobs = 0
                    result[row, c] = np.nan
                else:
                    weighted_sum += weight * val
                    weighted_sum_sq += weight * val * val
                    weight_sum += weight
                    weight *= (1.0 - alpha)
                    nobs += 1
                    if nobs >= min_periods:
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean
                        if not bias and nobs > 1:
                            var *= weight_sum / (weight_sum - 1.0 + alpha)
                        result[row, c] = max(0.0, var)
                    else:
                        result[row, c] = np.nan
        else:
            ewm = 0.0
            ewm_sq = 0.0
            is_first = True
            nobs = 0
            for row in range(n_rows):
                val = arr[row, c]
                if np.isnan(val):
                    if not ignore_na:
                        is_first = True
                        nobs = 0
                    result[row, c] = np.nan
                else:
                    if is_first:
                        ewm = val
                        ewm_sq = val * val
                        is_first = False
                    else:
                        ewm = alpha * val + (1.0 - alpha) * ewm
                        ewm_sq = alpha * val * val + (1.0 - alpha) * ewm_sq
                    nobs += 1
                    if nobs >= min_periods:
                        var = ewm_sq - ewm * ewm
                        if not bias and nobs > 1:
                            var *= nobs / (nobs - 1.0)
                        result[row, c] = max(0.0, var)
                    else:
                        result[row, c] = np.nan


# ============================================================================
# ThreadPool functions using nogil kernels (4.7x faster than prange!)
# ============================================================================

def _ewm_mean_threadpool(arr, alpha, adjust, ignore_na, min_periods):
    """Ultra-fast EWM mean using ThreadPool + nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _ewm_mean_nogil_chunk(arr, result, start_col, end_col, alpha, adjust, ignore_na, min_periods)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _ewm_var_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias):
    """Ultra-fast EWM variance using ThreadPool + nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _ewm_var_nogil_chunk(arr, result, start_col, end_col, alpha, adjust, ignore_na, min_periods, bias)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS) if i * chunk_size < n_cols]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _ewm_std_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias):
    """Ultra-fast EWM std using ThreadPool + nogil kernels."""
    var_result = _ewm_var_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias)
    return np.sqrt(var_result)


# ============================================================================
# Dispatch functions (choose serial vs parallel based on array size)
# ============================================================================

def _ewm_mean_dispatch(arr, alpha, adjust, ignore_na, min_periods):
    """Dispatch to ThreadPool (large), parallel (medium), or serial (small)."""
    if arr.size >= THREADPOOL_THRESHOLD:
        return _ewm_mean_threadpool(arr, alpha, adjust, ignore_na, min_periods)
    if arr.size < PARALLEL_THRESHOLD:
        return _ewm_mean_2d_serial(arr, alpha, adjust, ignore_na, min_periods)
    return _ewm_mean_2d(arr, alpha, adjust, ignore_na, min_periods)


def _ewm_var_dispatch(arr, alpha, adjust, ignore_na, min_periods, bias):
    """Dispatch to ThreadPool (large), parallel (medium), or serial (small)."""
    if arr.size >= THREADPOOL_THRESHOLD:
        return _ewm_var_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias)
    if arr.size < PARALLEL_THRESHOLD:
        return _ewm_var_2d_serial(arr, alpha, adjust, ignore_na, min_periods, bias)
    return _ewm_var_2d(arr, alpha, adjust, ignore_na, min_periods, bias)


def _ewm_std_dispatch(arr, alpha, adjust, ignore_na, min_periods, bias):
    """Dispatch to ThreadPool (large), parallel (medium), or serial (small)."""
    if arr.size >= THREADPOOL_THRESHOLD:
        return _ewm_std_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias)
    if arr.size < PARALLEL_THRESHOLD:
        return _ewm_std_2d_serial(arr, alpha, adjust, ignore_na, min_periods, bias)
    return _ewm_std_2d(arr, alpha, adjust, ignore_na, min_periods, bias)


# ============================================================================
# Wrapper functions for pandas EWM objects
# ============================================================================

def _make_ewm_mean_wrapper():
    """Create wrapper for EWM mean."""

    def wrapper(ewm_obj, *args, **kwargs):
        obj = ewm_obj.obj

        # Extract EWM parameters
        adjust = ewm_obj.adjust
        ignore_na = ewm_obj.ignore_na
        min_periods = ewm_obj.min_periods if ewm_obj.min_periods is not None else 0

        # Compute alpha
        alpha = _get_alpha(
            span=getattr(ewm_obj, 'span', None),
            halflife=getattr(ewm_obj, 'halflife', None),
            alpha=getattr(ewm_obj, 'alpha', None),
            com=getattr(ewm_obj, 'com', None)
        )

        # Only optimize DataFrames
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Optimization only for DataFrame")

        # Handle mixed-dtype DataFrames
        numeric_cols, numeric_df = get_numeric_columns_fast(obj)

        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")

        arr = ensure_float64(numeric_df.values)  # Keep C-contiguous (pandas default)
        result = _ewm_mean_dispatch(arr, alpha, adjust, ignore_na, min_periods)

        return wrap_result(
            result, numeric_df, columns=numeric_cols,
            merge_non_numeric=True, original_df=obj
        )

    return wrapper


def _make_ewm_var_wrapper():
    """Create wrapper for EWM variance."""

    def wrapper(ewm_obj, bias=False, *args, **kwargs):
        obj = ewm_obj.obj

        # Extract EWM parameters
        adjust = ewm_obj.adjust
        ignore_na = ewm_obj.ignore_na
        min_periods = ewm_obj.min_periods if ewm_obj.min_periods is not None else 0

        # Compute alpha
        alpha = _get_alpha(
            span=getattr(ewm_obj, 'span', None),
            halflife=getattr(ewm_obj, 'halflife', None),
            alpha=getattr(ewm_obj, 'alpha', None),
            com=getattr(ewm_obj, 'com', None)
        )

        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Optimization only for DataFrame")

        numeric_cols, numeric_df = get_numeric_columns_fast(obj)

        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")

        arr = ensure_float64(numeric_df.values)  # Keep C-contiguous (pandas default)
        result = _ewm_var_dispatch(arr, alpha, adjust, ignore_na, min_periods, bias)

        return wrap_result(
            result, numeric_df, columns=numeric_cols,
            merge_non_numeric=True, original_df=obj
        )

    return wrapper


def _make_ewm_std_wrapper():
    """Create wrapper for EWM standard deviation."""

    def wrapper(ewm_obj, bias=False, *args, **kwargs):
        obj = ewm_obj.obj

        # Extract EWM parameters
        adjust = ewm_obj.adjust
        ignore_na = ewm_obj.ignore_na
        min_periods = ewm_obj.min_periods if ewm_obj.min_periods is not None else 0

        # Compute alpha
        alpha = _get_alpha(
            span=getattr(ewm_obj, 'span', None),
            halflife=getattr(ewm_obj, 'halflife', None),
            alpha=getattr(ewm_obj, 'alpha', None),
            com=getattr(ewm_obj, 'com', None)
        )

        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Optimization only for DataFrame")

        numeric_cols, numeric_df = get_numeric_columns_fast(obj)

        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")

        arr = ensure_float64(numeric_df.values)  # Keep C-contiguous (pandas default)
        result = _ewm_std_dispatch(arr, alpha, adjust, ignore_na, min_periods, bias)

        return wrap_result(
            result, numeric_df, columns=numeric_cols,
            merge_non_numeric=True, original_df=obj
        )

    return wrapper


# Create wrapper instances
optimized_ewm_mean = _make_ewm_mean_wrapper()
optimized_ewm_var = _make_ewm_var_wrapper()
optimized_ewm_std = _make_ewm_std_wrapper()


def apply_ewm_patches():
    """Apply all EWM operation patches to pandas."""
    from .._patch import patch

    ExponentialMovingWindow = pd.core.window.ewm.ExponentialMovingWindow

    patch(ExponentialMovingWindow, 'mean', optimized_ewm_mean)
    patch(ExponentialMovingWindow, 'var', optimized_ewm_var)
    patch(ExponentialMovingWindow, 'std', optimized_ewm_std)
