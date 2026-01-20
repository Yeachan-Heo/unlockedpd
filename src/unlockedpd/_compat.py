"""Pandas compatibility layer for unlockedpd.

This module handles version detection and provides helpers for
extracting/wrapping numpy arrays from pandas structures.
"""
from typing import Tuple, List, Optional, Union
import pandas as pd
import numpy as np
from packaging import version


# Version detection
PANDAS_VERSION = version.parse(pd.__version__)
NUMPY_VERSION = version.parse(np.__version__)

PANDAS_2 = PANDAS_VERSION >= version.parse("2.0.0")
PANDAS_2_1 = PANDAS_VERSION >= version.parse("2.1.0")

# Check for Copy-on-Write mode (pandas 2.0+)
try:
    COW_ENABLED = PANDAS_2 and pd.options.mode.copy_on_write
except AttributeError:
    COW_ENABLED = False

# Supported numeric dtypes for optimization
NUMERIC_DTYPES = (
    np.float64, np.float32,
    np.int64, np.int32, np.int16, np.int8,
    np.uint64, np.uint32, np.uint16, np.uint8,
)


def get_numeric_columns(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Extract numeric columns from DataFrame.

    For mixed-dtype DataFrames, this filters to only numeric columns.
    Non-numeric columns (object, datetime, categorical, etc.) are skipped.

    Args:
        df: Input DataFrame

    Returns:
        tuple: (list of numeric column names, DataFrame with only numeric columns)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return [], df.iloc[:, :0]  # Empty DataFrame with same index
    return numeric_cols, df[numeric_cols]


def is_all_numeric(df: pd.DataFrame) -> bool:
    """Fast check if all columns are numeric.

    This is much faster than select_dtypes for the common case where
    all columns are already numeric (the typical case for DataFrames
    that will be used with Numba operations).
    """
    if len(df.columns) == 0:
        return False
    # Check each column's dtype kind: b=bool, i=int, u=uint, f=float, c=complex
    for col in df.columns:
        if df[col].dtype.kind not in 'biufc':
            return False
    return True


def get_numeric_columns_fast(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Fast extraction of numeric columns with optimized fast-path.

    For all-numeric DataFrames (the common case), this is ~200x faster
    than select_dtypes. Falls back to slow path for mixed-dtype DataFrames.
    """
    # Fast path: check if all columns are numeric
    if is_all_numeric(df):
        return list(df.columns), df
    # Slow path: mixed types, need to filter
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return [], df.iloc[:, :0]
    return numeric_cols, df[numeric_cols]


def get_values(
    obj: Union[pd.DataFrame, pd.Series],
    numeric_only: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Extract numpy array from DataFrame or Series.

    Args:
        obj: DataFrame or Series
        numeric_only: If True and obj is DataFrame, select only numeric columns

    Returns:
        If numeric_only=True and obj is DataFrame:
            tuple of (numpy array, list of numeric column names)
        Otherwise:
            numpy array
    """
    if isinstance(obj, pd.DataFrame):
        if numeric_only:
            numeric_cols, numeric_df = get_numeric_columns(obj)
            if len(numeric_cols) == 0:
                return np.array([]).reshape(len(obj), 0), []
            return numeric_df.values, numeric_cols
        return obj.values
    elif isinstance(obj, pd.Series):
        return obj.to_numpy()
    return np.asarray(obj)


def wrap_result(
    result: np.ndarray,
    like: Union[pd.DataFrame, pd.Series],
    columns: Optional[List[str]] = None,
    merge_non_numeric: bool = False,
    original_df: Optional[pd.DataFrame] = None
) -> Union[pd.DataFrame, pd.Series]:
    """Wrap numpy result back into pandas structure.

    Args:
        result: numpy array with computed values
        like: original DataFrame/Series for index/structure
        columns: column names for result (for numeric-only operations)
        merge_non_numeric: if True, merge result with non-numeric columns from original
        original_df: original DataFrame when merge_non_numeric=True

    Returns:
        DataFrame or Series matching the structure of 'like'
    """
    if isinstance(like, pd.DataFrame):
        result_df = pd.DataFrame(
            result,
            index=like.index,
            columns=columns if columns is not None else like.columns
        )

        if merge_non_numeric and original_df is not None:
            # Add back non-numeric columns as NaN (matching pandas behavior for numeric ops)
            non_numeric_cols = [c for c in original_df.columns if c not in result_df.columns]
            if non_numeric_cols:
                for col in non_numeric_cols:
                    result_df[col] = np.nan
                # Restore original column order (only if we added columns)
                result_df = result_df.reindex(columns=original_df.columns)

        return result_df

    elif isinstance(like, pd.Series):
        return pd.Series(result, index=like.index, name=like.name)

    return result


def wrap_result_fast(
    result: np.ndarray,
    df: pd.DataFrame
) -> pd.DataFrame:
    """Fast wrap numpy result back into DataFrame.

    This is optimized for the common case where:
    - Input is a DataFrame (not Series)
    - All columns are numeric (no merge needed)
    - We just need to preserve index and columns

    This is ~10x faster than wrap_result with merge_non_numeric.
    """
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64 for Numba compatibility.

    Args:
        arr: Input numpy array

    Returns:
        float64 array (may be a view if already float64)
    """
    if arr.dtype == np.float64:
        return arr
    return arr.astype(np.float64)


def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous for optimal Numba performance.

    Args:
        arr: Input numpy array

    Returns:
        C-contiguous array (may be a view if already contiguous)
    """
    if arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr)


def ensure_optimal_layout(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Ensure array has optimal memory layout for the operation axis.

    For column-wise operations (axis=0), F-contiguous is optimal because
    it provides sequential memory access when iterating down columns.
    For row-wise operations (axis=1), C-contiguous is optimal.

    Args:
        arr: Input numpy array
        axis: Operation axis (0=column-wise, 1=row-wise)

    Returns:
        Array with optimal memory layout (may be a copy if layout change needed)
    """
    if axis == 0:
        # Column-parallel: F-contiguous for sequential column access
        if arr.flags['F_CONTIGUOUS']:
            return arr
        return np.asfortranarray(arr)
    else:
        # Row-parallel: C-contiguous for sequential row access
        if arr.flags['C_CONTIGUOUS']:
            return arr
        return np.ascontiguousarray(arr)


def get_optimal_parallel_axis(arr: np.ndarray, min_parallel_units: int = 64) -> int:
    """Determine optimal parallelization axis based on array shape.

    For element-independent operations, we want to parallelize across
    the dimension with more elements to maximize CPU utilization.

    Args:
        arr: Input numpy array (2D)
        min_parallel_units: Minimum parallel work units for good utilization

    Returns:
        0 for column-parallel (parallelize across columns)
        1 for row-parallel (parallelize across rows)
    """
    n_rows, n_cols = arr.shape

    # If both dimensions are small, prefer rows (C-contiguous default)
    if n_rows < min_parallel_units and n_cols < min_parallel_units:
        return 1  # Row-parallel, fewer layout changes needed

    # If rows significantly outnumber columns, parallelize across rows
    if n_rows >= 4 * n_cols:
        return 1  # Row-parallel

    # If columns significantly outnumber rows, parallelize across columns
    if n_cols >= 4 * n_rows:
        return 0  # Column-parallel

    # For balanced shapes, prefer row-parallel (C-contiguous is default)
    # This avoids memory layout conversion overhead
    return 1


def prepare_array_for_parallel(arr: np.ndarray, parallel_axis: int) -> np.ndarray:
    """Prepare array with optimal dtype and memory layout for parallel operation.

    Args:
        arr: Input numpy array
        parallel_axis: 0 for column-parallel, 1 for row-parallel

    Returns:
        float64 array with optimal memory layout
    """
    arr = ensure_float64(arr)
    if parallel_axis == 0:
        # Column-parallel: F-contiguous for sequential column access
        if not arr.flags['F_CONTIGUOUS']:
            return np.asfortranarray(arr)
    else:
        # Row-parallel: C-contiguous for sequential row access
        if not arr.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(arr)
    return arr
