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
            # New observations get weight 1.0, older observations decay by (1-alpha)
            weighted_sum = 0.0
            weight_sum = 0.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        # Decay weights even for NaN, but don't add to sum
                        weighted_sum *= (1.0 - alpha)
                        weight_sum *= (1.0 - alpha)
                    # When ignore_na=True, don't decay weights (skip entirely)
                    # Output current EWM if we have enough observations
                    if nobs >= min_periods and weight_sum > 0:
                        result[row, col] = weighted_sum / weight_sum
                else:
                    # Decay existing sums first, then add new observation with weight 1.0
                    weighted_sum = (1.0 - alpha) * weighted_sum + val
                    weight_sum = (1.0 - alpha) * weight_sum + 1.0
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
            # Track both x and x^2 - new observations get weight 1.0, older decay by (1-alpha)
            weighted_sum = 0.0
            weighted_sum_sq = 0.0
            weight_sum = 0.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        # Decay weights even for NaN, but don't add to sum
                        weighted_sum *= (1.0 - alpha)
                        weighted_sum_sq *= (1.0 - alpha)
                        weight_sum *= (1.0 - alpha)
                    # When ignore_na=True, don't decay weights
                    # Output current variance if we have enough observations
                    if nobs >= min_periods and nobs > 1 and weight_sum > 0:  # Variance requires at least 2 observations
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean
                        if not bias:
                            var *= weight_sum / (weight_sum - 1.0 + alpha)
                        result[row, col] = max(0.0, var)
                else:
                    # Decay existing sums first, then add new observation with weight 1.0
                    weighted_sum = (1.0 - alpha) * weighted_sum + val
                    weighted_sum_sq = (1.0 - alpha) * weighted_sum_sq + val * val
                    weight_sum = (1.0 - alpha) * weight_sum + 1.0
                    nobs += 1

                    if nobs >= min_periods and nobs > 1:  # Variance requires at least 2 observations
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean

                        # Bias correction for adjusted method
                        if not bias:
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
            # New observations get weight 1.0, older observations decay by (1-alpha)
            weighted_sum = 0.0
            weight_sum = 0.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        # Decay weights even for NaN, but don't add to sum
                        weighted_sum *= (1.0 - alpha)
                        weight_sum *= (1.0 - alpha)
                    # When ignore_na=True, don't decay weights
                    # Output current EWM if we have enough observations
                    if nobs >= min_periods and weight_sum > 0:
                        result[row, col] = weighted_sum / weight_sum
                else:
                    # Decay existing sums first, then add new observation with weight 1.0
                    weighted_sum = (1.0 - alpha) * weighted_sum + val
                    weight_sum = (1.0 - alpha) * weight_sum + 1.0
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
                    # Output current EWM for NaN if we have enough observations
                    elif nobs >= min_periods and not is_first:
                        result[row, col] = ewm
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
            # New observations get weight 1.0, older observations decay by (1-alpha)
            weighted_sum = 0.0
            weighted_sum_sq = 0.0
            weight_sum = 0.0
            nobs = 0

            for row in range(n_rows):
                val = arr[row, col]

                if np.isnan(val):
                    if not ignore_na:
                        # Decay weights even for NaN, but don't add to sum
                        weighted_sum *= (1.0 - alpha)
                        weighted_sum_sq *= (1.0 - alpha)
                        weight_sum *= (1.0 - alpha)
                    # When ignore_na=True, don't decay weights
                    # Output current variance if we have enough observations
                    if nobs >= min_periods and nobs > 1 and weight_sum > 0:  # Variance requires at least 2 observations
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean
                        if not bias:
                            var *= weight_sum / (weight_sum - 1.0 + alpha)
                        result[row, col] = max(0.0, var)
                else:
                    # Decay existing sums first, then add new observation with weight 1.0
                    weighted_sum = (1.0 - alpha) * weighted_sum + val
                    weighted_sum_sq = (1.0 - alpha) * weighted_sum_sq + val * val
                    weight_sum = (1.0 - alpha) * weight_sum + 1.0
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
                    # Output current variance for NaN if we have enough observations
                    elif nobs >= min_periods and not is_first:
                        var = ewm_sq - ewm * ewm
                        if not bias and nobs > 1:
                            var *= nobs / (nobs - 1.0)
                        result[row, col] = max(0.0, var)
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
            # New observations get weight 1.0, older observations decay by (1-alpha)
            weighted_sum = 0.0
            weight_sum = 0.0
            nobs = 0
            for row in range(n_rows):
                val = arr[row, c]
                if np.isnan(val):
                    if not ignore_na:
                        # Decay weights even for NaN, but don't add to sum
                        weighted_sum *= (1.0 - alpha)
                        weight_sum *= (1.0 - alpha)
                    # Output current EWM if we have enough observations
                    if nobs >= min_periods and weight_sum > 0:
                        result[row, c] = weighted_sum / weight_sum
                    else:
                        result[row, c] = np.nan
                else:
                    # Decay existing sums first, then add new observation with weight 1.0
                    weighted_sum = (1.0 - alpha) * weighted_sum + val
                    weight_sum = (1.0 - alpha) * weight_sum + 1.0
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
                    elif nobs >= min_periods and not is_first:
                        result[row, c] = ewm
                    else:
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
            # New observations get weight 1.0, older observations decay by (1-alpha)
            weighted_sum = 0.0
            weighted_sum_sq = 0.0
            weight_sum = 0.0
            nobs = 0
            for row in range(n_rows):
                val = arr[row, c]
                if np.isnan(val):
                    if not ignore_na:
                        # Decay weights even for NaN, but don't add to sum
                        weighted_sum *= (1.0 - alpha)
                        weighted_sum_sq *= (1.0 - alpha)
                        weight_sum *= (1.0 - alpha)
                    # Output current variance if we have enough observations
                    if nobs >= min_periods and weight_sum > 0:
                        mean = weighted_sum / weight_sum
                        mean_sq = weighted_sum_sq / weight_sum
                        var = mean_sq - mean * mean
                        if not bias and nobs > 1:
                            var *= weight_sum / (weight_sum - 1.0 + alpha)
                        result[row, c] = max(0.0, var)
                    else:
                        result[row, c] = np.nan
                else:
                    # Decay existing sums first, then add new observation with weight 1.0
                    weighted_sum = (1.0 - alpha) * weighted_sum + val
                    weighted_sum_sq = (1.0 - alpha) * weighted_sum_sq + val * val
                    weight_sum = (1.0 - alpha) * weight_sum + 1.0
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
                    elif nobs >= min_periods and not is_first:
                        var = ewm_sq - ewm * ewm
                        if not bias and nobs > 1:
                            var *= nobs / (nobs - 1.0)
                        result[row, c] = max(0.0, var)
                    else:
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


# ============================================================================
# Nogil kernels for EWM pairwise correlation/covariance
# ============================================================================

@njit(nogil=True, cache=True)
def _ewm_cov_single_col_nogil(arr_x, arr_y, result, alpha, adjust, ignore_na, min_periods, bias):
    """EWM covariance between two columns - GIL released.

    Uses formula: Cov = EWM(XY) - EWM(X)*EWM(Y)
    """
    n_rows = len(arr_x)

    if adjust:
        ewm_x = 0.0
        ewm_y = 0.0
        ewm_xy = 0.0
        weight_sum_x = 0.0
        weight_sum_y = 0.0
        weight_sum_xy = 0.0
        weight = 1.0
        nobs = 0

        for row in range(n_rows):
            vx = arr_x[row]
            vy = arr_y[row]

            # Both values must be non-NaN
            if np.isnan(vx) or np.isnan(vy):
                if not ignore_na:
                    # Reset on NaN
                    ewm_x = 0.0
                    ewm_y = 0.0
                    ewm_xy = 0.0
                    weight_sum_x = 0.0
                    weight_sum_y = 0.0
                    weight_sum_xy = 0.0
                    weight = 1.0
                    nobs = 0
                result[row] = np.nan
            else:
                ewm_x += weight * vx
                ewm_y += weight * vy
                ewm_xy += weight * vx * vy
                weight_sum_x += weight
                weight_sum_y += weight
                weight_sum_xy += weight
                weight *= (1.0 - alpha)
                nobs += 1

                if nobs >= min_periods:
                    mean_x = ewm_x / weight_sum_x
                    mean_y = ewm_y / weight_sum_y
                    mean_xy = ewm_xy / weight_sum_xy
                    cov = mean_xy - (mean_x * mean_y)

                    if not bias:
                        # Apply bias correction
                        cov *= weight_sum_xy / (weight_sum_xy - 1.0) if weight_sum_xy > 1.0 else 1.0

                    result[row] = cov
                else:
                    result[row] = np.nan
    else:
        # Recursive EWM
        ewm_x = 0.0
        ewm_y = 0.0
        ewm_xy = 0.0
        is_first = True
        nobs = 0

        for row in range(n_rows):
            vx = arr_x[row]
            vy = arr_y[row]

            if np.isnan(vx) or np.isnan(vy):
                if not ignore_na:
                    is_first = True
                    nobs = 0
                result[row] = np.nan
            else:
                if is_first:
                    ewm_x = vx
                    ewm_y = vy
                    ewm_xy = vx * vy
                    is_first = False
                else:
                    ewm_x = alpha * vx + (1.0 - alpha) * ewm_x
                    ewm_y = alpha * vy + (1.0 - alpha) * ewm_y
                    ewm_xy = alpha * vx * vy + (1.0 - alpha) * ewm_xy

                nobs += 1
                if nobs >= min_periods:
                    cov = ewm_xy - (ewm_x * ewm_y)
                    result[row] = cov
                else:
                    result[row] = np.nan


@njit(nogil=True, cache=True)
def _ewm_corr_single_col_nogil(arr_x, arr_y, result, alpha, adjust, ignore_na, min_periods, is_diagonal):
    """EWM correlation between two columns - GIL released.

    Pearson correlation = Cov(X,Y) / (Std(X) * Std(Y))
    """
    n_rows = len(arr_x)

    if adjust:
        ewm_x = 0.0
        ewm_y = 0.0
        ewm_x2 = 0.0
        ewm_y2 = 0.0
        ewm_xy = 0.0
        weight_sum = 0.0
        weight = 1.0
        nobs = 0

        for row in range(n_rows):
            vx = arr_x[row]
            vy = arr_y[row]

            if np.isnan(vx) or np.isnan(vy):
                if not ignore_na:
                    ewm_x = 0.0
                    ewm_y = 0.0
                    ewm_x2 = 0.0
                    ewm_y2 = 0.0
                    ewm_xy = 0.0
                    weight_sum = 0.0
                    weight = 1.0
                    nobs = 0
                result[row] = np.nan
            else:
                ewm_x += weight * vx
                ewm_y += weight * vy
                ewm_x2 += weight * vx * vx
                ewm_y2 += weight * vy * vy
                ewm_xy += weight * vx * vy
                weight_sum += weight
                weight *= (1.0 - alpha)
                nobs += 1

                if nobs >= min_periods:
                    if is_diagonal:
                        result[row] = 1.0
                    else:
                        mean_x = ewm_x / weight_sum
                        mean_y = ewm_y / weight_sum
                        mean_x2 = ewm_x2 / weight_sum
                        mean_y2 = ewm_y2 / weight_sum
                        mean_xy = ewm_xy / weight_sum

                        var_x = mean_x2 - (mean_x * mean_x)
                        var_y = mean_y2 - (mean_y * mean_y)
                        cov = mean_xy - (mean_x * mean_y)

                        if var_x > 1e-14 and var_y > 1e-14:
                            result[row] = cov / np.sqrt(var_x * var_y)
                        else:
                            result[row] = np.nan
                else:
                    result[row] = np.nan
    else:
        # Recursive EWM
        ewm_x = 0.0
        ewm_y = 0.0
        ewm_x2 = 0.0
        ewm_y2 = 0.0
        ewm_xy = 0.0
        is_first = True
        nobs = 0

        for row in range(n_rows):
            vx = arr_x[row]
            vy = arr_y[row]

            if np.isnan(vx) or np.isnan(vy):
                if not ignore_na:
                    is_first = True
                    nobs = 0
                result[row] = np.nan
            else:
                if is_first:
                    ewm_x = vx
                    ewm_y = vy
                    ewm_x2 = vx * vx
                    ewm_y2 = vy * vy
                    ewm_xy = vx * vy
                    is_first = False
                else:
                    ewm_x = alpha * vx + (1.0 - alpha) * ewm_x
                    ewm_y = alpha * vy + (1.0 - alpha) * ewm_y
                    ewm_x2 = alpha * vx * vx + (1.0 - alpha) * ewm_x2
                    ewm_y2 = alpha * vy * vy + (1.0 - alpha) * ewm_y2
                    ewm_xy = alpha * vx * vy + (1.0 - alpha) * ewm_xy

                nobs += 1
                if nobs >= min_periods:
                    if is_diagonal:
                        result[row] = 1.0
                    else:
                        var_x = ewm_x2 - (ewm_x * ewm_x)
                        var_y = ewm_y2 - (ewm_y * ewm_y)
                        cov = ewm_xy - (ewm_x * ewm_y)

                        if var_x > 1e-14 and var_y > 1e-14:
                            result[row] = cov / np.sqrt(var_x * var_y)
                        else:
                            result[row] = np.nan
                else:
                    result[row] = np.nan


@njit(nogil=True, cache=True)
def _ewm_cov_matrix_nogil_chunk(arr, result_flat, start_pair, end_pair, pairs_i, pairs_j, alpha, adjust, ignore_na, min_periods, bias, n_rows):
    """EWM covariance for multiple column pairs - GIL released."""
    for p in range(start_pair, end_pair):
        i = pairs_i[p]
        j = pairs_j[p]
        col_x = arr[:, i]
        col_y = arr[:, j]
        result_col = result_flat[:, p]
        _ewm_cov_single_col_nogil(col_x, col_y, result_col, alpha, adjust, ignore_na, min_periods, bias)


@njit(nogil=True, cache=True)
def _ewm_corr_matrix_nogil_chunk(arr, result_flat, start_pair, end_pair, pairs_i, pairs_j, alpha, adjust, ignore_na, min_periods, n_rows):
    """EWM correlation for multiple column pairs - GIL released."""
    for p in range(start_pair, end_pair):
        i = pairs_i[p]
        j = pairs_j[p]
        col_x = arr[:, i]
        col_y = arr[:, j]
        result_col = result_flat[:, p]
        is_diagonal = (i == j)
        _ewm_corr_single_col_nogil(col_x, col_y, result_col, alpha, adjust, ignore_na, min_periods, is_diagonal)


# ============================================================================
# ThreadPool functions for EWM pairwise operations
# ============================================================================

def _ewm_cov_pairwise_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias):
    """EWM covariance matrix using ThreadPool + nogil kernels."""
    n_rows, n_cols = arr.shape

    pairs = []
    for i in range(n_cols):
        for j in range(i, n_cols):
            pairs.append((i, j))

    n_pairs = len(pairs)
    pairs_i = np.array([p[0] for p in pairs], dtype=np.int64)
    pairs_j = np.array([p[1] for p in pairs], dtype=np.int64)

    result_flat = np.empty((n_rows, n_pairs), dtype=np.float64)
    result_flat[:] = np.nan

    chunk_size = max(1, (n_pairs + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_pair, end_pair = args
        _ewm_cov_matrix_nogil_chunk(arr, result_flat, start_pair, end_pair,
                                    pairs_i, pairs_j, alpha, adjust, ignore_na, min_periods, bias, n_rows)

    chunks = [(k * chunk_size, min((k + 1) * chunk_size, n_pairs))
              for k in range(THREADPOOL_WORKERS) if k * chunk_size < n_pairs]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    # Reshape to (n_rows, n_cols, n_cols) symmetric matrix
    result = np.empty((n_rows, n_cols, n_cols), dtype=np.float64)
    result[:] = np.nan

    for idx, (i, j) in enumerate(pairs):
        result[:, i, j] = result_flat[:, idx]
        if i != j:
            result[:, j, i] = result_flat[:, idx]  # Symmetric

    return result


def _ewm_corr_pairwise_threadpool(arr, alpha, adjust, ignore_na, min_periods):
    """EWM correlation matrix using ThreadPool + nogil kernels."""
    n_rows, n_cols = arr.shape

    pairs = []
    for i in range(n_cols):
        for j in range(i, n_cols):
            pairs.append((i, j))

    n_pairs = len(pairs)
    pairs_i = np.array([p[0] for p in pairs], dtype=np.int64)
    pairs_j = np.array([p[1] for p in pairs], dtype=np.int64)

    result_flat = np.empty((n_rows, n_pairs), dtype=np.float64)
    result_flat[:] = np.nan

    chunk_size = max(1, (n_pairs + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_pair, end_pair = args
        _ewm_corr_matrix_nogil_chunk(arr, result_flat, start_pair, end_pair,
                                     pairs_i, pairs_j, alpha, adjust, ignore_na, min_periods, n_rows)

    chunks = [(k * chunk_size, min((k + 1) * chunk_size, n_pairs))
              for k in range(THREADPOOL_WORKERS) if k * chunk_size < n_pairs]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    result = np.empty((n_rows, n_cols, n_cols), dtype=np.float64)
    result[:] = np.nan

    for idx, (i, j) in enumerate(pairs):
        result[:, i, j] = result_flat[:, idx]
        if i != j:
            result[:, j, i] = result_flat[:, idx]  # Symmetric

    return result


# ============================================================================
# Wrapper functions for pandas EWM objects (corr/cov)
# ============================================================================

def optimized_ewm_cov(ewm_obj, other=None, pairwise=None, bias=False, *args, **kwargs):
    """Optimized EWM covariance."""
    obj = ewm_obj.obj

    # Only optimize DataFrame pairwise case
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if other is not None:
        raise TypeError("other parameter not supported, use pairwise=True")

    if pairwise is False:
        raise TypeError("Only pairwise=True is optimized")

    numeric_cols, numeric_df = get_numeric_columns_fast(obj)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    # Extract EWM parameters
    alpha = _get_alpha(
        span=ewm_obj.span,
        halflife=ewm_obj.halflife,
        alpha=ewm_obj.alpha,
        com=ewm_obj.com
    )
    adjust = ewm_obj.adjust
    ignore_na = ewm_obj.ignore_na
    min_periods = ewm_obj.min_periods if ewm_obj.min_periods is not None else 0

    arr = ensure_float64(numeric_df.values)
    result_3d = _ewm_cov_pairwise_threadpool(arr, alpha, adjust, ignore_na, min_periods, bias)

    # Convert to pandas format: MultiIndex rows (timestamp, column), single columns
    n_rows = len(obj)
    n_cols = len(numeric_cols)

    row_tuples = [(idx, col) for idx in obj.index for col in numeric_cols]
    multi_index = pd.MultiIndex.from_tuples(row_tuples)

    # Reshape: from (n_rows, n_cols, n_cols) to (n_rows * n_cols, n_cols)
    result_2d = result_3d.reshape(n_rows * n_cols, n_cols)

    return pd.DataFrame(result_2d, index=multi_index, columns=numeric_cols)


def optimized_ewm_corr(ewm_obj, other=None, pairwise=None, *args, **kwargs):
    """Optimized EWM correlation."""
    obj = ewm_obj.obj

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if other is not None:
        raise TypeError("other parameter not supported, use pairwise=True")

    if pairwise is False:
        raise TypeError("Only pairwise=True is optimized")

    numeric_cols, numeric_df = get_numeric_columns_fast(obj)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    # Extract EWM parameters
    alpha = _get_alpha(
        span=ewm_obj.span,
        halflife=ewm_obj.halflife,
        alpha=ewm_obj.alpha,
        com=ewm_obj.com
    )
    adjust = ewm_obj.adjust
    ignore_na = ewm_obj.ignore_na
    min_periods = ewm_obj.min_periods if ewm_obj.min_periods is not None else 0

    arr = ensure_float64(numeric_df.values)
    result_3d = _ewm_corr_pairwise_threadpool(arr, alpha, adjust, ignore_na, min_periods)

    # Convert to pandas format
    n_rows = len(obj)
    n_cols = len(numeric_cols)

    row_tuples = [(idx, col) for idx in obj.index for col in numeric_cols]
    multi_index = pd.MultiIndex.from_tuples(row_tuples)

    result_2d = result_3d.reshape(n_rows * n_cols, n_cols)

    return pd.DataFrame(result_2d, index=multi_index, columns=numeric_cols)


def apply_ewm_patches():
    """Apply all EWM operation patches to pandas."""
    from .._patch import patch

    ExponentialMovingWindow = pd.core.window.ewm.ExponentialMovingWindow

    patch(ExponentialMovingWindow, 'mean', optimized_ewm_mean)
    patch(ExponentialMovingWindow, 'var', optimized_ewm_var)
    patch(ExponentialMovingWindow, 'std', optimized_ewm_std)
    patch(ExponentialMovingWindow, 'corr', optimized_ewm_corr)
    patch(ExponentialMovingWindow, 'cov', optimized_ewm_cov)
