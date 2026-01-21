"""Optimized DataFrame.agg() / aggregate() operations.

Dispatches to existing optimized aggregation implementations for supported functions.
Falls back to pandas for unsupported functions or complex cases.

Supported aggregations:
- sum, mean, std, var, min, max, median, prod

Usage patterns:
- df.agg('sum')                          # Single function as string
- df.agg(['sum', 'mean', 'std'])         # List of functions
- df.agg({'col_a': 'sum', 'col_b': 'mean'})  # Dict mapping columns to functions
"""
import pandas as pd
from typing import Union, List, Dict, Callable, Any

from .._compat import get_numeric_columns_fast


# Mapping of function names to optimized implementation names in aggregations.py
SUPPORTED_FUNCTIONS = {
    'sum': 'optimized_sum',
    'mean': 'optimized_mean',
    'std': 'optimized_std',
    'var': 'optimized_var',
    'min': 'optimized_min',
    'max': 'optimized_max',
    'median': 'optimized_median',
    'prod': 'optimized_prod',
}


def _get_optimized_func(func_name: str) -> Callable:
    """Get optimized function by name.

    Args:
        func_name: Name of the aggregation function (e.g., 'sum', 'mean')

    Returns:
        The optimized function from aggregations module

    Raises:
        TypeError: If the function is not supported for optimization
    """
    if func_name not in SUPPORTED_FUNCTIONS:
        raise TypeError(f"Unsupported aggregation function: {func_name}")

    from . import aggregations
    opt_func = getattr(aggregations, SUPPORTED_FUNCTIONS[func_name], None)
    if opt_func is None:
        raise TypeError(f"Optimized function not found: {func_name}")

    return opt_func


def _agg_single_function(df: pd.DataFrame, func: str, axis: int = 0) -> pd.Series:
    """Apply single aggregation function across DataFrame.

    Args:
        df: Input DataFrame
        func: Function name as string (e.g., 'sum', 'mean')
        axis: Aggregation axis (0=rows, 1=columns)

    Returns:
        pd.Series with aggregation results

    Raises:
        TypeError: If function is not supported
    """
    opt_func = _get_optimized_func(func)
    return opt_func(df, axis=axis)


def _agg_multiple_functions(df: pd.DataFrame, funcs: List[str], axis: int = 0) -> pd.DataFrame:
    """Apply multiple aggregation functions across DataFrame.

    Args:
        df: Input DataFrame
        funcs: List of function names
        axis: Aggregation axis (0=rows, 1=columns)

    Returns:
        pd.DataFrame with rows=functions, columns=original columns (axis=0)
        or rows=functions, columns=original index (axis=1)

    Raises:
        TypeError: If any function is not supported
    """
    results = {}
    for func in funcs:
        opt_func = _get_optimized_func(func)
        results[func] = opt_func(df, axis=axis)

    # Build DataFrame with function names as index
    result_df = pd.DataFrame(results).T
    result_df.index.name = None
    return result_df


def _agg_dict_mapping(df: pd.DataFrame, func_dict: Dict[str, Union[str, List[str]]], axis: int = 0) -> pd.Series:
    """Apply functions per column based on dict mapping.

    Args:
        df: Input DataFrame
        func_dict: Dict mapping column names to function names or list of functions
        axis: Aggregation axis (only axis=0 supported for dict mapping)

    Returns:
        pd.Series with column names as index and aggregated values

    Raises:
        TypeError: If any function is not supported or multiple functions per column
    """
    if axis != 0:
        raise TypeError("Dict aggregation only supported for axis=0")

    results = {}
    for col, func in func_dict.items():
        if col not in df.columns:
            continue

        if isinstance(func, str):
            opt_func = _get_optimized_func(func)
            # Apply to single column - create single-column DataFrame
            col_df = df[[col]]
            result = opt_func(col_df, axis=0)
            # Result is a Series with one element; extract the value
            results[col] = result.iloc[0]
        elif isinstance(func, list):
            # Multiple functions for one column - not supported in optimized path
            # This would require a different return structure (MultiIndex)
            raise TypeError(
                f"Multiple aggregation functions per column not supported "
                f"in optimized path (column '{col}'). Use single function per column "
                f"or fall back to pandas."
            )
        else:
            raise TypeError(f"Invalid aggregation function type for column '{col}': {type(func)}")

    return pd.Series(results)


def optimized_agg(
    self,
    func: Union[str, List[str], Dict[str, Union[str, List[str]]], Callable, None] = None,
    axis: int = 0,
    *args,
    **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """Optimized DataFrame.agg() / aggregate() implementation.

    Dispatches to optimized Numba implementations for supported functions:
    - sum, mean, std, var, min, max, median, prod

    Falls back to pandas for:
    - Custom callables
    - Unsupported function names
    - Complex aggregation patterns (e.g., multiple functions per column in dict)
    - Named aggregation with keyword arguments

    Args:
        self: DataFrame instance
        func: Aggregation function(s):
            - str: Single function name ('sum', 'mean', etc.)
            - list[str]: Multiple function names
            - dict: Column-to-function mapping
            - callable: Custom function (falls back to pandas)
        axis: Aggregation axis:
            - 0 or 'index': Aggregate over rows (result has column index)
            - 1 or 'columns': Aggregate over columns (result has row index)
        *args: Additional positional arguments (triggers fallback)
        **kwargs: Additional keyword arguments (triggers fallback)

    Returns:
        pd.Series for single function or dict mapping
        pd.DataFrame for list of functions

    Raises:
        TypeError: Triggers fallback to pandas when function is not supported
    """
    if not isinstance(self, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Normalize axis
    if axis in ('index', 0):
        axis = 0
    elif axis in ('columns', 1):
        axis = 1
    else:
        raise TypeError(f"Invalid axis value: {axis}")

    # Check for conditions that require fallback
    if args:
        # Additional positional arguments - not supported
        raise TypeError("Additional positional arguments not supported")

    if kwargs:
        # Named aggregation or other kwargs - not supported
        raise TypeError("Keyword arguments not supported in optimized path")

    if func is None:
        raise TypeError("func cannot be None")

    # Dispatch based on func type
    if isinstance(func, str):
        return _agg_single_function(self, func, axis)

    elif isinstance(func, list):
        # Verify all elements are strings
        if not all(isinstance(f, str) for f in func):
            raise TypeError("All list elements must be function name strings")
        return _agg_multiple_functions(self, func, axis)

    elif isinstance(func, dict):
        return _agg_dict_mapping(self, func, axis)

    elif callable(func):
        # Custom callable - not optimized, fall back
        raise TypeError("Custom callable functions not supported in optimized path")

    else:
        raise TypeError(f"Unsupported func type: {type(func)}")


# Alias for the full name
optimized_aggregate = optimized_agg


def apply_agg_patches():
    """Apply agg/aggregate patches to pandas DataFrame.

    These patches provide optimized implementations that dispatch to
    existing Numba-accelerated aggregation functions (sum, mean, std, etc.).
    Falls back to pandas for unsupported function types or complex patterns.
    """
    from .._patch import patch

    patch(pd.DataFrame, 'agg', optimized_agg)
    patch(pd.DataFrame, 'aggregate', optimized_aggregate)
