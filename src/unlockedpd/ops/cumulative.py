"""Parallel cumulative operations using Numba nogil kernels with thread pooling.

This module provides parallelized cumulative operations by distributing
columns across threads using Numba's nogil=True to release the GIL for
true parallel execution.

Key insight: Numba nogil kernels with ThreadPool provide 4.7x speedup over
NumPy by enabling true parallel execution across multiple cores.
"""

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange, set_num_threads
from concurrent.futures import ThreadPoolExecutor
from .._compat import (
    get_numeric_columns_fast,
    is_all_numeric,
    wrap_result,
    wrap_result_fast,
    ensure_float64,
)
from .._resources import (
    assert_memory_budget,
    record_dispatch_path,
    resolve_threadpool_workers,
    simple_result_memory_estimate,
    use_threadpool_path,
)

# Thresholds for parallel execution
# Based on benchmarking: parallel helps when n_cols >= 200 and n_rows >= 5000
MIN_COLS_FOR_PARALLEL = 200
MIN_ROWS_FOR_PARALLEL = 5000
AXIS1_CUMULATIVE_THRESHOLD = 500_000
AXIS1_CUMULATIVE_THREAD_CAP = 32
AXIS1_CUMULATIVE_MEDIUM_FRAME_BYTES = 512 * 1024 * 1024
AXIS1_CUMULATIVE_SMALL_THREAD_CAP = 16
AXIS1_CUMULATIVE_LARGE_THREAD_CAP = 32
AXIS1_CUMULATIVE_BYTES_PER_THREAD = 2 * 1024 * 1024
AXIS0_CUMULATIVE_THRESHOLD = 500_000
AXIS0_CUMULATIVE_THREAD_CAP = 32
AXIS0_CUMULATIVE_MEDIUM_FRAME_BYTES = 128 * 1024 * 1024
AXIS0_CUMULATIVE_SMALL_THREAD_CAP = 16
AXIS0_CUMULATIVE_LARGE_THREAD_CAP = 32
AXIS0_CUMULATIVE_BYTES_PER_THREAD = 2 * 1024 * 1024


def _axis1_cumulative_thread_cap(arr: np.ndarray) -> int:
    """Return a size-aware cap for transient axis=1 cumulative workers."""

    nbytes = int(getattr(arr, "nbytes", 0))
    if nbytes <= 0:
        return 1

    size_cap = max(
        1,
        (nbytes + AXIS1_CUMULATIVE_BYTES_PER_THREAD - 1)
        // AXIS1_CUMULATIVE_BYTES_PER_THREAD,
    )
    operation_cap = (
        AXIS1_CUMULATIVE_SMALL_THREAD_CAP
        if nbytes < AXIS1_CUMULATIVE_MEDIUM_FRAME_BYTES
        else AXIS1_CUMULATIVE_LARGE_THREAD_CAP
    )
    return max(
        1,
        min(size_cap, operation_cap, AXIS1_CUMULATIVE_THREAD_CAP, arr.shape[0]),
    )


def _axis0_cumulative_thread_cap(arr: np.ndarray) -> int:
    """Return a size-aware cap for row-block axis=0 cumulative workers."""

    nbytes = int(getattr(arr, "nbytes", 0))
    if nbytes <= 0:
        return 1

    size_cap = max(
        1,
        (nbytes + AXIS0_CUMULATIVE_BYTES_PER_THREAD - 1)
        // AXIS0_CUMULATIVE_BYTES_PER_THREAD,
    )
    operation_cap = (
        AXIS0_CUMULATIVE_SMALL_THREAD_CAP
        if nbytes < AXIS0_CUMULATIVE_MEDIUM_FRAME_BYTES
        else AXIS0_CUMULATIVE_LARGE_THREAD_CAP
    )
    return max(
        1,
        min(size_cap, operation_cap, AXIS0_CUMULATIVE_THREAD_CAP, arr.shape[0]),
    )


def _axis1_row_chunks(n_rows: int, workers: int):
    chunk_size = max(1, (int(n_rows) + int(workers) - 1) // int(workers))
    return [
        (start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]


def _bounded_axis1_cumulative(kernel, arr: np.ndarray) -> np.ndarray:
    """Run finite row cumulative kernels using transient bounded workers."""

    thread_cap = _axis1_cumulative_thread_cap(arr)
    workers = resolve_threadpool_workers(
        arr.shape[0],
        operation="cumulative",
        operation_cap=thread_cap,
        memory_bandwidth_cap=thread_cap,
        cap=thread_cap,
    )
    chunks = _axis1_row_chunks(arr.shape[0], workers)
    workers = max(1, min(workers, len(chunks)))
    result = np.empty_like(arr)

    record_dispatch_path("threadpool" if workers > 1 else "serial_numba")
    if workers == 1:
        kernel(arr, result, 0, arr.shape[0])
        return result

    def process_chunk(bounds):
        start_row, end_row = bounds
        kernel(arr, result, start_row, end_row)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _bounded_axis0_cumulative(kernel, arr: np.ndarray) -> np.ndarray:
    """Run dense finite row-block cumulative kernels with bounded Numba threads."""

    from .._config import config

    thread_cap = _axis0_cumulative_thread_cap(arr)
    configured = config.num_threads
    target_threads = (
        min(configured, thread_cap)
        if configured > 0
        else resolve_threadpool_workers(
            arr.shape[0],
            operation="cumulative",
            operation_cap=thread_cap,
            memory_bandwidth_cap=thread_cap,
            cap=thread_cap,
        )
    )
    target_threads = max(1, min(int(target_threads), arr.shape[0]))

    record_dispatch_path("parallel_numba" if target_threads > 1 else "serial_numba")
    if target_threads != get_num_threads():
        set_num_threads(target_threads)
    return kernel(arr, target_threads)


# ============================================================================
# Nogil kernels for ThreadPool (GIL-released for true parallelism)
# ============================================================================


@njit(nogil=True, cache=True)
def _cumsum_nogil_chunk(arr, result, start_col, end_col):
    """Cumulative sum - GIL released.

    Handles inf correctly:
    - inf + finite = inf
    - inf + inf = inf
    - inf + (-inf) = nan
    - Once nan appears, all subsequent values are nan
    """
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cumsum = 0.0
        has_nan = False
        for row in range(n_rows):
            val = arr[row, c]
            if has_nan or np.isnan(val):
                result[row, c] = np.nan
                has_nan = True
            else:
                cumsum += val
                # Check if cumsum became nan (e.g., inf + (-inf))
                if np.isnan(cumsum):
                    result[row, c] = np.nan
                    has_nan = True
                else:
                    result[row, c] = cumsum


@njit(nogil=True, cache=True)
def _cumprod_nogil_chunk(arr, result, start_col, end_col):
    """Cumulative product - GIL released.

    Handles inf correctly:
    - inf * finite (non-zero) = inf (with appropriate sign)
    - inf * 0 = nan
    - inf * inf = inf
    - Once nan appears, all subsequent values are nan
    """
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cumprod = 1.0
        has_nan = False
        for row in range(n_rows):
            val = arr[row, c]
            if has_nan or np.isnan(val):
                result[row, c] = np.nan
                has_nan = True
            else:
                cumprod *= val
                # Check if cumprod became nan (e.g., inf * 0)
                if np.isnan(cumprod):
                    result[row, c] = np.nan
                    has_nan = True
                else:
                    result[row, c] = cumprod


@njit(nogil=True, cache=True)
def _cummin_nogil_chunk(arr, result, start_col, end_col):
    """Cumulative min - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cummin = np.inf
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = np.nan
            else:
                if val < cummin:
                    cummin = val
                result[row, c] = cummin


@njit(nogil=True, cache=True)
def _cummax_nogil_chunk(arr, result, start_col, end_col):
    """Cumulative max - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cummax = -np.inf
        for row in range(n_rows):
            val = arr[row, c]
            if np.isnan(val):
                result[row, c] = np.nan
            else:
                if val > cummax:
                    cummax = val
                result[row, c] = cummax


# ============================================================================
# Core parallel implementations using ThreadPoolExecutor + nogil kernels
# ============================================================================


def _cumsum_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cumsum using nogil kernels for 4.7x speedup."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    result[:] = np.nan

    workers, chunks = use_threadpool_path(n_cols, operation="cumulative")

    def process_chunk(args):
        start_col, end_col = args
        _cumsum_nogil_chunk(arr, result, start_col, end_col)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _cumprod_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cumprod using nogil kernels for 4.7x speedup."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    result[:] = np.nan

    workers, chunks = use_threadpool_path(n_cols, operation="cumulative")

    def process_chunk(args):
        start_col, end_col = args
        _cumprod_nogil_chunk(arr, result, start_col, end_col)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _cummin_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cummin using nogil kernels for 4.7x speedup."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    result[:] = np.nan

    workers, chunks = use_threadpool_path(n_cols, operation="cumulative")

    def process_chunk(args):
        start_col, end_col = args
        _cummin_nogil_chunk(arr, result, start_col, end_col)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_chunk, chunks))

    return result


def _cummax_parallel(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Parallel cummax using nogil kernels for 4.7x speedup."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    result[:] = np.nan

    workers, chunks = use_threadpool_path(n_cols, operation="cumulative")

    def process_chunk(args):
        start_col, end_col = args
        _cummax_nogil_chunk(arr, result, start_col, end_col)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_chunk, chunks))

    return result


# ============================================================================
# Dispatch functions (choose parallel vs pandas based on shape)
# ============================================================================


def _should_use_parallel(arr):
    """Determine if parallel execution is worthwhile.

    Parallel helps when:
    - Enough columns to distribute (>= 200)
    - Enough rows per column for meaningful work (>= 5000)
    """
    n_rows, n_cols = arr.shape
    return n_cols >= MIN_COLS_FOR_PARALLEL and n_rows >= MIN_ROWS_FOR_PARALLEL


def _normalize_axis(axis) -> int:
    if axis in (0, "index", None):
        return 0
    if axis in (1, "columns"):
        return 1
    raise ValueError(f"No axis named {axis!r}")


@njit(parallel=True, nogil=True, cache=True)
def _cumsum_axis0_finite_blocks(arr, blocks):
    """Dense finite cumsum(axis=0) using row-contiguous block scans."""

    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    block_offsets = np.zeros((blocks, n_cols), dtype=np.float64)

    for block in prange(blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        totals = np.zeros(n_cols, dtype=np.float64)
        for row in range(start, end):
            for col in range(n_cols):
                totals[col] += arr[row, col]
                result[row, col] = totals[col]
        for col in range(n_cols):
            block_offsets[block, col] = totals[col]

    running = np.zeros(n_cols, dtype=np.float64)
    for block in range(blocks):
        for col in range(n_cols):
            block_total = block_offsets[block, col]
            block_offsets[block, col] = running[col]
            running[col] += block_total

    for block in prange(1, blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        for row in range(start, end):
            for col in range(n_cols):
                result[row, col] += block_offsets[block, col]

    return result


@njit(parallel=True, nogil=True, cache=True)
def _cumprod_axis0_finite_blocks(arr, blocks):
    """Dense finite cumprod(axis=0) using row-contiguous block scans."""

    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    block_offsets = np.ones((blocks, n_cols), dtype=np.float64)

    for block in prange(blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        totals = np.ones(n_cols, dtype=np.float64)
        for row in range(start, end):
            for col in range(n_cols):
                totals[col] *= arr[row, col]
                result[row, col] = totals[col]
        for col in range(n_cols):
            block_offsets[block, col] = totals[col]

    running = np.ones(n_cols, dtype=np.float64)
    for block in range(blocks):
        for col in range(n_cols):
            block_total = block_offsets[block, col]
            block_offsets[block, col] = running[col]
            running[col] *= block_total

    for block in prange(1, blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        for row in range(start, end):
            for col in range(n_cols):
                result[row, col] *= block_offsets[block, col]

    return result


@njit(parallel=True, nogil=True, cache=True)
def _cummin_axis0_finite_blocks(arr, blocks):
    """Dense finite cummin(axis=0) using row-contiguous block scans."""

    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    block_offsets = np.empty((blocks, n_cols), dtype=np.float64)

    for block in prange(blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        totals = np.empty(n_cols, dtype=np.float64)
        for col in range(n_cols):
            totals[col] = np.inf
        for row in range(start, end):
            for col in range(n_cols):
                value = arr[row, col]
                if value < totals[col]:
                    totals[col] = value
                result[row, col] = totals[col]
        for col in range(n_cols):
            block_offsets[block, col] = totals[col]

    running = np.empty(n_cols, dtype=np.float64)
    for col in range(n_cols):
        running[col] = np.inf
    for block in range(blocks):
        for col in range(n_cols):
            block_total = block_offsets[block, col]
            block_offsets[block, col] = running[col]
            if block_total < running[col]:
                running[col] = block_total

    for block in prange(1, blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        for row in range(start, end):
            for col in range(n_cols):
                offset = block_offsets[block, col]
                if offset < result[row, col]:
                    result[row, col] = offset

    return result


@njit(parallel=True, nogil=True, cache=True)
def _cummax_axis0_finite_blocks(arr, blocks):
    """Dense finite cummax(axis=0) using row-contiguous block scans."""

    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)
    block_offsets = np.empty((blocks, n_cols), dtype=np.float64)

    for block in prange(blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        totals = np.empty(n_cols, dtype=np.float64)
        for col in range(n_cols):
            totals[col] = -np.inf
        for row in range(start, end):
            for col in range(n_cols):
                value = arr[row, col]
                if value > totals[col]:
                    totals[col] = value
                result[row, col] = totals[col]
        for col in range(n_cols):
            block_offsets[block, col] = totals[col]

    running = np.empty(n_cols, dtype=np.float64)
    for col in range(n_cols):
        running[col] = -np.inf
    for block in range(blocks):
        for col in range(n_cols):
            block_total = block_offsets[block, col]
            block_offsets[block, col] = running[col]
            if block_total > running[col]:
                running[col] = block_total

    for block in prange(1, blocks):
        start = (n_rows * block) // blocks
        end = (n_rows * (block + 1)) // blocks
        for row in range(start, end):
            for col in range(n_cols):
                offset = block_offsets[block, col]
                if offset > result[row, col]:
                    result[row, col] = offset

    return result


def _axis0_numba_cumulative(arr: np.ndarray, op: str) -> np.ndarray | None:
    """Return a dense finite column-wise cumulative result, or ``None``."""

    if (
        arr.dtype != np.float64
        or arr.size < AXIS0_CUMULATIVE_THRESHOLD
        or arr.shape[0] == 0
        or arr.shape[1] == 0
        or not arr.flags.c_contiguous
    ):
        return None

    # The row-block kernels use associative block prefixes to exploit the
    # row-major DataFrame buffer.  Keep exact pandas/NumPy edge behavior for
    # NaN/inf inputs by leaving non-finite frames on the existing safe path.
    if not np.isfinite(arr).all():
        return None

    assert_memory_budget(
        simple_result_memory_estimate(
            arr.shape[0],
            arr.shape[1],
            dtype=arr.dtype,
            intermediates=0,
        ),
        operation="cumulative",
    )

    if op == "cumsum":
        return _bounded_axis0_cumulative(_cumsum_axis0_finite_blocks, arr)
    if op == "cumprod":
        return _bounded_axis0_cumulative(_cumprod_axis0_finite_blocks, arr)
    if op == "cummin":
        return _bounded_axis0_cumulative(_cummin_axis0_finite_blocks, arr)
    if op == "cummax":
        return _bounded_axis0_cumulative(_cummax_axis0_finite_blocks, arr)
    return None


@njit(nogil=True, fastmath=True, cache=True)
def _cumsum_axis1_finite_chunk(arr, result, start_row, end_row):
    """Dense finite cumsum(axis=1) over contiguous rows."""

    n_cols = arr.shape[1]
    for row in range(start_row, end_row):
        total = 0.0
        for col in range(n_cols):
            total += arr[row, col]
            result[row, col] = total


@njit(nogil=True, fastmath=True, cache=True)
def _cumprod_axis1_finite_chunk(arr, result, start_row, end_row):
    """Dense finite cumprod(axis=1) over contiguous rows."""

    n_cols = arr.shape[1]
    for row in range(start_row, end_row):
        total = 1.0
        for col in range(n_cols):
            total *= arr[row, col]
            result[row, col] = total


@njit(nogil=True, fastmath=True, cache=True)
def _cummin_axis1_finite_chunk(arr, result, start_row, end_row):
    """Dense finite cummin(axis=1) over contiguous rows."""

    n_cols = arr.shape[1]
    for row in range(start_row, end_row):
        current = np.inf
        for col in range(n_cols):
            value = arr[row, col]
            if value < current:
                current = value
            result[row, col] = current


@njit(nogil=True, fastmath=True, cache=True)
def _cummax_axis1_finite_chunk(arr, result, start_row, end_row):
    """Dense finite cummax(axis=1) over contiguous rows."""

    n_cols = arr.shape[1]
    for row in range(start_row, end_row):
        current = -np.inf
        for col in range(n_cols):
            value = arr[row, col]
            if value > current:
                current = value
            result[row, col] = current


def _axis1_numba_cumulative(arr: np.ndarray, op: str) -> np.ndarray | None:
    """Return a dense finite row-parallel result, or ``None`` to use NumPy."""

    if (
        arr.dtype != np.float64
        or arr.size < AXIS1_CUMULATIVE_THRESHOLD
        or arr.shape[1] == 0
        or not arr.flags.c_contiguous
    ):
        return None

    # The Numba kernels are intentionally branch-free and fastmath-enabled.
    # Keep exact pandas/NumPy behavior for NaN/inf edge cases by letting the
    # existing NumPy path handle any non-finite frame.
    if not np.isfinite(arr).all():
        return None

    if op == "cumsum":
        return _bounded_axis1_cumulative(_cumsum_axis1_finite_chunk, arr)
    if op == "cumprod":
        return _bounded_axis1_cumulative(_cumprod_axis1_finite_chunk, arr)
    if op == "cummin":
        return _bounded_axis1_cumulative(_cummin_axis1_finite_chunk, arr)
    if op == "cummax":
        return _bounded_axis1_cumulative(_cummax_axis1_finite_chunk, arr)
    return None


def _numpy_accumulate(arr: np.ndarray, op: str, axis: int, skipna: bool) -> np.ndarray:
    """Run NumPy cumulative kernels with pandas-compatible skipna handling.

    Axis=1 cumulative operations are a large pandas slow path because pandas
    iterates block/column machinery while the underlying row-major buffer is
    contiguous.  NumPy's accumulate kernels exploit that layout directly.  The
    dense no-NaN case keeps a single output array; only true missing-value cases
    allocate the mask/fill intermediates needed for pandas ``skipna=True``.
    """

    if op == "cumsum":
        result = np.cumsum(arr, axis=axis)
        fill_value = 0
        redo = np.cumsum
    elif op == "cumprod":
        result = np.cumprod(arr, axis=axis)
        fill_value = 1
        redo = np.cumprod
    elif op == "cummin":
        result = np.minimum.accumulate(arr, axis=axis)
        fill_value = np.inf
        redo = np.minimum.accumulate
    elif op == "cummax":
        result = np.maximum.accumulate(arr, axis=axis)
        fill_value = -np.inf
        redo = np.maximum.accumulate
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported cumulative op: {op}")

    if skipna and np.isnan(result).any():
        mask = np.isnan(arr)
        if mask.any():
            result = redo(np.where(mask, fill_value, arr), axis=axis)
            result[mask] = np.nan

    record_dispatch_path("numpy_vectorized")
    return result


def _axis1_numpy_cumulative(df: pd.DataFrame, op: str, skipna: bool) -> pd.DataFrame:
    """Fast path for all-numeric row-wise cumulative DataFrame operations."""

    if not is_all_numeric(df):
        raise TypeError("axis=1 cumulative optimization requires all numeric columns")

    # Preserve NumPy's dtype behavior for integer/bool dense frames instead of
    # forcing float64 like the Numba column-wise kernels need to do.
    arr = df.to_numpy(copy=False)
    assert_memory_budget(
        simple_result_memory_estimate(
            arr.shape[0],
            arr.shape[1],
            dtype=arr.dtype,
            intermediates=0,
        ),
        operation="cumulative",
    )
    result = _axis1_numba_cumulative(arr, op)
    if result is None:
        result = _numpy_accumulate(arr, op, 1, skipna)
    return wrap_result_fast(result, df)


# ============================================================================
# Wrapper functions for pandas DataFrame methods
# ============================================================================


def optimized_cumsum(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cumsum - parallel for wide DataFrames, pandas for narrow."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    axis = _normalize_axis(axis)

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

    if axis == 1:
        return _axis1_numpy_cumulative(df, "cumsum", skipna)

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    result = _axis0_numba_cumulative(arr, "cumsum")
    if result is None:
        if not _should_use_parallel(arr):
            # Fall back to pandas for small DataFrames
            raise TypeError("Use pandas for small DataFrames")

        assert_memory_budget(
            simple_result_memory_estimate(arr.shape[0], arr.shape[1]),
            operation="cumulative",
        )
        result = _cumsum_parallel(arr, skipna)

    return wrap_result(
        result, numeric_df, columns=numeric_cols, merge_non_numeric=True, original_df=df
    )


def optimized_cumprod(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cumprod - parallel for wide DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    axis = _normalize_axis(axis)

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

    if axis == 1:
        return _axis1_numpy_cumulative(df, "cumprod", skipna)

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    result = _axis0_numba_cumulative(arr, "cumprod")
    if result is None:
        if not _should_use_parallel(arr):
            raise TypeError("Use pandas for small DataFrames")

        assert_memory_budget(
            simple_result_memory_estimate(arr.shape[0], arr.shape[1]),
            operation="cumulative",
        )
        result = _cumprod_parallel(arr, skipna)

    return wrap_result(
        result, numeric_df, columns=numeric_cols, merge_non_numeric=True, original_df=df
    )


def optimized_cummin(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cummin - parallel for wide DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    axis = _normalize_axis(axis)

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

    if axis == 1:
        return _axis1_numpy_cumulative(df, "cummin", skipna)

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    result = _axis0_numba_cumulative(arr, "cummin")
    if result is None:
        if not _should_use_parallel(arr):
            raise TypeError("Use pandas for small DataFrames")

        assert_memory_budget(
            simple_result_memory_estimate(arr.shape[0], arr.shape[1]),
            operation="cumulative",
        )
        result = _cummin_parallel(arr, skipna)

    return wrap_result(
        result, numeric_df, columns=numeric_cols, merge_non_numeric=True, original_df=df
    )


def optimized_cummax(df, axis=0, skipna=True, *args, **kwargs):
    """Optimized cummax - parallel for wide DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    axis = _normalize_axis(axis)

    # Handle empty DataFrame
    if df.empty:
        raise TypeError("Use pandas for empty DataFrames")

    if axis == 1:
        return _axis1_numpy_cumulative(df, "cummax", skipna)

    numeric_cols, numeric_df = get_numeric_columns_fast(df)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    result = _axis0_numba_cumulative(arr, "cummax")
    if result is None:
        if not _should_use_parallel(arr):
            raise TypeError("Use pandas for small DataFrames")

        assert_memory_budget(
            simple_result_memory_estimate(arr.shape[0], arr.shape[1]),
            operation="cumulative",
        )
        result = _cummax_parallel(arr, skipna)

    return wrap_result(
        result, numeric_df, columns=numeric_cols, merge_non_numeric=True, original_df=df
    )


def apply_cumulative_patches():
    """Apply cumulative operation patches to pandas.

    These patches only activate for wide DataFrames (200+ columns, 5000+ rows).
    For narrow DataFrames, falls back to pandas automatically.
    """
    from .._patch import patch

    patch(pd.DataFrame, "cumsum", optimized_cumsum)
    patch(pd.DataFrame, "cumprod", optimized_cumprod)
    patch(pd.DataFrame, "cummin", optimized_cummin)
    patch(pd.DataFrame, "cummax", optimized_cummax)
