"""Optimized apply operations using Numba.

This module provides Numba-accelerated apply operations
for user-defined functions that can be JIT compiled.
"""
import numpy as np
from numba import njit, prange
from numba.core.errors import TypingError, NumbaError
import pandas as pd
from typing import Callable, Union, Any
import warnings


def jit(func: Callable) -> Callable:
    """Decorator to mark function for Numba compilation in apply.

    Usage:
        @unlockedpd.jit
        def my_func(arr):
            return np.sum(arr ** 2)

        df.apply(my_func, axis=0)  # Will use parallel Numba execution
    """
    jitted = njit(cache=True)(func)
    jitted._unlockedpd_jittable = True
    return jitted


def _is_jittable(func: Callable) -> bool:
    """Check if a function is marked as jittable or is already jitted."""
    return (
        hasattr(func, '_unlockedpd_jittable') or
        hasattr(func, '_dispatcher')  # Already a Numba dispatcher
    )


def _try_jit_compile(func: Callable, sample_arr: np.ndarray) -> tuple[bool, Any]:
    """Try to JIT compile a function and test it.

    Returns:
        (success, jitted_func or None)
    """
    if hasattr(func, '_dispatcher'):
        # Already jitted
        return True, func

    try:
        jitted = njit(cache=True)(func)
        # Test compilation with sample data
        _ = jitted(sample_arr[:min(10, len(sample_arr))])
        return True, jitted
    except (TypingError, NumbaError, Exception):
        return False, None


def _make_parallel_apply_axis0(jitted_func: Callable):
    """Create parallel apply function for axis=0 (column-wise)."""
    @njit(parallel=True, cache=True)
    def parallel_apply(arr):
        n_rows, n_cols = arr.shape
        result = np.empty(n_cols, dtype=np.float64)
        for col in prange(n_cols):
            result[col] = jitted_func(arr[:, col])
        return result
    return parallel_apply


def _make_parallel_apply_axis1(jitted_func: Callable):
    """Create parallel apply function for axis=1 (row-wise)."""
    @njit(parallel=True, cache=True)
    def parallel_apply(arr):
        n_rows, n_cols = arr.shape
        result = np.empty(n_rows, dtype=np.float64)
        for row in prange(n_rows):
            result[row] = jitted_func(arr[row, :])
        return result
    return parallel_apply


def optimized_apply(
    df: pd.DataFrame,
    func: Callable,
    axis: int = 0,
    raw: bool = False,
    result_type: str = None,
    **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """Optimized DataFrame apply operation.

    Attempts to JIT compile the function and execute in parallel.
    Falls back to pandas apply if compilation fails.
    """
    from .._compat import get_numeric_columns, ensure_float64
    from .._config import config

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Get numeric data
    numeric_cols, numeric_df = get_numeric_columns(df)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)

    # Try to JIT compile
    sample = arr[:, 0] if axis == 0 else arr[0, :]
    success, jitted_func = _try_jit_compile(func, sample)

    if not success:
        if config.warn_on_fallback:
            warnings.warn(
                f"unlockedpd: Could not JIT compile function, falling back to pandas",
                RuntimeWarning
            )
        raise TypeError("Function cannot be JIT compiled")

    # Create parallel version
    if axis == 0:
        parallel_func = _make_parallel_apply_axis0(jitted_func)
        result = parallel_func(arr)
        return pd.Series(result, index=numeric_cols)
    else:
        parallel_func = _make_parallel_apply_axis1(jitted_func)
        result = parallel_func(arr)
        return pd.Series(result, index=df.index)


def _patched_apply(df, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
    """Patched apply method for DataFrame."""
    # Only attempt optimization for raw numeric apply
    if raw and not args and not kwargs:
        return optimized_apply(df, func, axis=axis, raw=raw, result_type=result_type)
    raise TypeError("Only raw numeric apply is optimized")


def apply_apply_patches():
    """Apply the apply operation patch to pandas."""
    from .._patch import patch

    # Note: We patch with fallback=True so non-optimizable cases go to pandas
    patch(pd.DataFrame, 'apply', _patched_apply)
