"""Warmup module for pre-compiling Numba functions.

This module triggers JIT compilation at import time so that
the first actual call doesn't incur compilation overhead.
"""
import numpy as np


def warmup_rolling():
    """Pre-compile rolling operation functions."""
    from .ops.rolling import (
        _rolling_sum_2d, _rolling_sum_2d_serial,
        _rolling_mean_2d, _rolling_mean_2d_serial,
        _rolling_sum_2d_centered, _rolling_mean_2d_centered,
    )

    # Create small dummy array for compilation
    dummy = np.zeros((10, 4), dtype=np.float64)
    window = 3
    min_periods = 1

    # Trigger compilation of parallel versions
    try:
        _rolling_sum_2d(dummy, window, min_periods)
        _rolling_mean_2d(dummy, window, min_periods)
        _rolling_sum_2d_centered(dummy, window, min_periods)
        _rolling_mean_2d_centered(dummy, window, min_periods)
    except Exception:
        pass

    # Trigger compilation of serial versions
    try:
        _rolling_sum_2d_serial(dummy, window, min_periods)
        _rolling_mean_2d_serial(dummy, window, min_periods)
    except Exception:
        pass

    # Try to compile Welford algorithms if available
    try:
        from .ops._welford import (
            rolling_std_welford_parallel, rolling_std_welford_serial,
            rolling_var_welford_parallel, rolling_var_welford_serial,
        )
        rolling_std_welford_parallel(dummy, window, min_periods, 1)
        rolling_std_welford_serial(dummy, window, min_periods, 1)
        rolling_var_welford_parallel(dummy, window, min_periods, 1)
        rolling_var_welford_serial(dummy, window, min_periods, 1)
    except Exception:
        pass

    # Try to compile deque algorithms if available
    try:
        from .ops._minmax_deque import (
            rolling_min_deque_parallel, rolling_min_deque_serial,
            rolling_max_deque_parallel, rolling_max_deque_serial,
        )
        rolling_min_deque_parallel(dummy, window, min_periods)
        rolling_min_deque_serial(dummy, window, min_periods)
        rolling_max_deque_parallel(dummy, window, min_periods)
        rolling_max_deque_serial(dummy, window, min_periods)
    except Exception:
        pass

    # Warmup nogil chunk kernels (ThreadPool workers)
    try:
        from .ops.rolling import (
            _rolling_mean_nogil_chunk, _rolling_sum_nogil_chunk,
            _rolling_std_nogil_chunk, _rolling_var_nogil_chunk,
            _rolling_min_nogil_chunk, _rolling_max_nogil_chunk,
            _rolling_count_nogil_chunk,
        )

        # Small arrays for warmup
        arr = np.random.randn(100, 10).astype(np.float64)
        result = np.empty_like(arr)

        # Warmup each nogil kernel
        _rolling_mean_nogil_chunk(arr, result, 0, 10, 5, 1)
        _rolling_sum_nogil_chunk(arr, result, 0, 10, 5, 1)
        _rolling_std_nogil_chunk(arr, result, 0, 10, 5, 1, 1)  # extra ddof param
        _rolling_var_nogil_chunk(arr, result, 0, 10, 5, 1, 1)  # extra ddof param
        _rolling_min_nogil_chunk(arr, result, 0, 10, 5, 1)
        _rolling_max_nogil_chunk(arr, result, 0, 10, 5, 1)
        _rolling_count_nogil_chunk(arr, result, 0, 10, 5, 1)
    except Exception:
        pass


def warmup_rank():
    """Pre-compile rank operation functions."""
    from .ops.rank import (
        _rank_axis1_average, _rank_axis0_average,
        _rank_axis1_min, _rank_axis1_max,
        _rank_axis1_first, _rank_axis1_dense,
    )

    dummy = np.zeros((10, 4), dtype=np.float64)

    try:
        _rank_axis1_average(dummy, True, 0)
        _rank_axis0_average(dummy, True, 0)
        _rank_axis1_min(dummy, True, 0)
        _rank_axis1_max(dummy, True, 0)
        _rank_axis1_first(dummy, True, 0)
        _rank_axis1_dense(dummy, True, 0)
    except Exception:
        pass

    # Try serial versions if available
    try:
        from .ops.rank import (
            _rank_axis1_average_serial, _rank_axis0_average_serial,
        )
        _rank_axis1_average_serial(dummy, True, 0)
        _rank_axis0_average_serial(dummy, True, 0)
    except Exception:
        pass


def warmup_apply():
    """Pre-compile apply operation functions."""
    try:
        from .ops.apply import _make_parallel_apply_axis0, _make_parallel_apply_axis1
        from numba import njit

        @njit(cache=True)
        def _dummy_func(arr):
            return np.sum(arr)

        # Compile the dummy function
        dummy_1d = np.zeros(10, dtype=np.float64)
        _dummy_func(dummy_1d)

        # Create and compile parallel apply functions
        apply_axis0 = _make_parallel_apply_axis0(_dummy_func)
        apply_axis1 = _make_parallel_apply_axis1(_dummy_func)

        dummy_2d = np.zeros((10, 4), dtype=np.float64)
        apply_axis0(dummy_2d)
        apply_axis1(dummy_2d)
    except Exception:
        pass


def warmup_cumulative():
    """Pre-compile cumulative operation functions."""
    try:
        from .ops.cumulative import (
            _cumsum_2d, _cumsum_2d_serial,
            _cumprod_2d, _cumprod_2d_serial,
            _cummin_2d, _cummin_2d_serial,
            _cummax_2d, _cummax_2d_serial,
        )

        dummy = np.zeros((10, 4), dtype=np.float64)

        _cumsum_2d(dummy, True)
        _cumsum_2d_serial(dummy, True)
        _cumprod_2d(dummy, True)
        _cumprod_2d_serial(dummy, True)
        _cummin_2d(dummy, True)
        _cummin_2d_serial(dummy, True)
        _cummax_2d(dummy, True)
        _cummax_2d_serial(dummy, True)
    except Exception:
        pass


def warmup_expanding():
    """Pre-compile expanding operation functions."""
    try:
        from .ops.expanding import (
            _expanding_sum_2d, _expanding_sum_2d_serial,
            _expanding_mean_2d, _expanding_mean_2d_serial,
            _expanding_std_2d, _expanding_std_2d_serial,
            _expanding_var_2d, _expanding_var_2d_serial,
            _expanding_min_2d, _expanding_min_2d_serial,
            _expanding_max_2d, _expanding_max_2d_serial,
        )

        dummy = np.zeros((10, 4), dtype=np.float64)
        min_periods = 1

        _expanding_sum_2d(dummy, min_periods)
        _expanding_sum_2d_serial(dummy, min_periods)
        _expanding_mean_2d(dummy, min_periods)
        _expanding_mean_2d_serial(dummy, min_periods)
        _expanding_std_2d(dummy, min_periods, 1)
        _expanding_std_2d_serial(dummy, min_periods, 1)
        _expanding_var_2d(dummy, min_periods, 1)
        _expanding_var_2d_serial(dummy, min_periods, 1)
        _expanding_min_2d(dummy, min_periods)
        _expanding_min_2d_serial(dummy, min_periods)
        _expanding_max_2d(dummy, min_periods)
        _expanding_max_2d_serial(dummy, min_periods)
    except Exception:
        pass

    # Warmup nogil chunk kernels (ThreadPool workers)
    try:
        from .ops.expanding import (
            _expanding_mean_nogil_chunk, _expanding_sum_nogil_chunk,
            _expanding_std_nogil_chunk, _expanding_var_nogil_chunk,
            _expanding_min_nogil_chunk, _expanding_max_nogil_chunk,
            _expanding_count_nogil_chunk,
        )

        # Small arrays for warmup
        arr = np.random.randn(100, 10).astype(np.float64)
        result = np.empty_like(arr)

        # Warmup each nogil kernel (no window parameter for expanding)
        _expanding_mean_nogil_chunk(arr, result, 0, 10, 1)
        _expanding_sum_nogil_chunk(arr, result, 0, 10, 1)
        _expanding_std_nogil_chunk(arr, result, 0, 10, 1, 1)  # ddof
        _expanding_var_nogil_chunk(arr, result, 0, 10, 1, 1)  # ddof
        _expanding_min_nogil_chunk(arr, result, 0, 10, 1)
        _expanding_max_nogil_chunk(arr, result, 0, 10, 1)
        _expanding_count_nogil_chunk(arr, result, 0, 10, 1)
    except Exception:
        pass


def warmup_ewm():
    """Pre-compile EWM operation functions."""
    try:
        from .ops.ewm import (
            _ewm_mean_2d, _ewm_mean_2d_serial,
            _ewm_var_2d, _ewm_var_2d_serial,
            _ewm_std_2d, _ewm_std_2d_serial,
        )

        dummy = np.zeros((10, 4), dtype=np.float64)
        alpha = 0.5

        _ewm_mean_2d(dummy, alpha, True, True, 1)
        _ewm_mean_2d_serial(dummy, alpha, True, True, 1)
        _ewm_var_2d(dummy, alpha, True, True, 1, False)
        _ewm_var_2d_serial(dummy, alpha, True, True, 1, False)
        _ewm_std_2d(dummy, alpha, True, True, 1, False)
        _ewm_std_2d_serial(dummy, alpha, True, True, 1, False)
    except Exception:
        pass


def warmup_stats():
    """Pre-compile stats operation functions."""
    try:
        from .ops.stats import (
            _skew_2d_axis0, _skew_2d_axis0_serial,
            _skew_2d_axis1, _skew_2d_axis1_serial,
            _kurt_2d_axis0, _kurt_2d_axis0_serial,
            _kurt_2d_axis1, _kurt_2d_axis1_serial,
            _sem_2d_axis0, _sem_2d_axis0_serial,
            _sem_2d_axis1, _sem_2d_axis1_serial,
        )

        dummy = np.zeros((10, 4), dtype=np.float64)

        _skew_2d_axis0(dummy, True)
        _skew_2d_axis0_serial(dummy, True)
        _skew_2d_axis1(dummy, True)
        _skew_2d_axis1_serial(dummy, True)
        _kurt_2d_axis0(dummy, True)
        _kurt_2d_axis0_serial(dummy, True)
        _kurt_2d_axis1(dummy, True)
        _kurt_2d_axis1_serial(dummy, True)
        _sem_2d_axis0(dummy, True, 1)
        _sem_2d_axis0_serial(dummy, True, 1)
        _sem_2d_axis1(dummy, True, 1)
        _sem_2d_axis1_serial(dummy, True, 1)
    except Exception:
        pass


def warmup_pairwise():
    """Pre-compile pairwise operation functions."""
    try:
        from .ops.pairwise import (
            _corr_matrix, _corr_matrix_serial,
            _cov_matrix, _cov_matrix_serial,
        )

        dummy = np.zeros((10, 4), dtype=np.float64)

        _corr_matrix(dummy, 1)
        _corr_matrix_serial(dummy, 1)
        _cov_matrix(dummy, 1, 1)
        _cov_matrix_serial(dummy, 1, 1)
    except Exception:
        pass


def warmup_transform():
    """Pre-compile transform operation functions."""
    try:
        from .ops.transform import (
            _diff_row_parallel, _diff_col_parallel, _diff_serial,
            _pct_change_row_parallel, _pct_change_col_parallel, _pct_change_serial,
            _shift_row_parallel, _shift_col_parallel, _shift_serial,
        )

        dummy = np.zeros((10, 4), dtype=np.float64)

        # Row-parallel versions
        _diff_row_parallel(dummy, 1)
        _pct_change_row_parallel(dummy, 1)
        _shift_row_parallel(dummy, 1, np.nan)

        # Column-parallel versions
        _diff_col_parallel(dummy, 1)
        _pct_change_col_parallel(dummy, 1)
        _shift_col_parallel(dummy, 1, np.nan)

        # Serial versions
        _diff_serial(dummy, 1)
        _pct_change_serial(dummy, 1)
        _shift_serial(dummy, 1, np.nan)
    except Exception:
        pass


def warmup_all():
    """Pre-compile all Numba functions.

    This should be called at module import time to avoid
    JIT compilation overhead on first use.
    """
    warmup_rolling()
    warmup_rank()
    warmup_apply()
    warmup_cumulative()
    warmup_expanding()
    warmup_ewm()
    warmup_stats()
    warmup_pairwise()
    warmup_transform()
