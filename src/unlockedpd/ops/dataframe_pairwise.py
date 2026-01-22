"""Optimized DataFrame correlation and covariance operations."""
import numpy as np
import pandas as pd
from numba import njit, prange
from typing import Optional

PARALLEL_THRESHOLD = 500_000


@njit(cache=True)
def _pairwise_cov_single(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> float:
    """Calculate covariance between two arrays with pairwise NaN handling."""
    n = len(x)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    count = 0

    for i in range(n):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            sum_x += x[i]
            sum_y += y[i]
            sum_xy += x[i] * y[i]
            count += 1

    if count <= ddof:
        return np.nan

    mean_x = sum_x / count
    mean_y = sum_y / count

    cov = (sum_xy - count * mean_x * mean_y) / (count - ddof)
    return cov


@njit(cache=True)
def _pairwise_corr_single(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation between two arrays."""
    n = len(x)
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    count = 0

    for i in range(n):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            sum_x += x[i]
            sum_y += y[i]
            sum_xx += x[i] * x[i]
            sum_yy += y[i] * y[i]
            sum_xy += x[i] * y[i]
            count += 1

    if count < 2:
        return np.nan

    mean_x = sum_x / count
    mean_y = sum_y / count

    var_x = sum_xx / count - mean_x * mean_x
    var_y = sum_yy / count - mean_y * mean_y

    if var_x <= 0 or var_y <= 0:
        return np.nan

    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    cov = sum_xy / count - mean_x * mean_y
    corr = cov / (std_x * std_y)

    return corr


@njit(parallel=True, cache=True)
def _compute_cov_matrix(arr: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Compute full covariance matrix."""
    n_cols = arr.shape[1]
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    for i in prange(n_cols):
        for j in range(i, n_cols):
            cov = _pairwise_cov_single(arr[:, i], arr[:, j], ddof)
            result[i, j] = cov
            result[j, i] = cov

    return result


@njit(parallel=True, cache=True)
def _compute_corr_matrix(arr: np.ndarray) -> np.ndarray:
    """Compute full correlation matrix."""
    n_cols = arr.shape[1]
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    for i in prange(n_cols):
        for j in range(i, n_cols):
            if i == j:
                # For diagonal, check if column has valid variance
                # If variance > 0, correlation with itself is 1.0
                # If variance = 0 or all NaN, correlation is NaN
                corr = _pairwise_corr_single(arr[:, i], arr[:, j])
                # corr will be NaN if variance=0 or insufficient data
                # corr will be ~1.0 if variance>0, set exactly to 1.0
                if not np.isnan(corr):
                    corr = 1.0
                result[i, j] = corr
            else:
                corr = _pairwise_corr_single(arr[:, i], arr[:, j])
                result[i, j] = corr
                result[j, i] = corr

    return result


def optimized_cov(df: pd.DataFrame, min_periods: Optional[int] = None,
                  ddof: int = 1) -> pd.DataFrame:
    """Optimized covariance matrix computation.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    min_periods : int, optional
        Minimum number of observations (not fully implemented)
    ddof : int, default 1
        Delta degrees of freedom

    Returns
    -------
    DataFrame
        Covariance matrix
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame()

    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise TypeError("No numeric columns")

    arr = numeric_df.values.astype(np.float64)
    cols = numeric_df.columns

    # Compute covariance matrix
    cov_matrix = _compute_cov_matrix(arr, ddof)

    return pd.DataFrame(cov_matrix, index=cols, columns=cols)


def optimized_corr(df: pd.DataFrame, method: str = 'pearson',
                   min_periods: Optional[int] = None) -> pd.DataFrame:
    """Optimized correlation matrix computation.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    method : str, default 'pearson'
        Correlation method (only 'pearson' optimized)
    min_periods : int, optional
        Minimum number of observations

    Returns
    -------
    DataFrame
        Correlation matrix
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if method != 'pearson':
        raise TypeError(f"Only 'pearson' method optimized, got '{method}'")

    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame()

    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise TypeError("No numeric columns")

    arr = numeric_df.values.astype(np.float64)
    cols = numeric_df.columns

    # Compute correlation matrix
    corr_matrix = _compute_corr_matrix(arr)

    return pd.DataFrame(corr_matrix, index=cols, columns=cols)


# Monkey-patch
_original_cov = pd.DataFrame.cov
_original_corr = pd.DataFrame.corr


def _patched_cov(self, min_periods=None, ddof=1, numeric_only=True):
    try:
        return optimized_cov(self, min_periods=min_periods, ddof=ddof)
    except TypeError:
        return _original_cov(self, min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)


def _patched_corr(self, method='pearson', min_periods=None, numeric_only=True):
    try:
        return optimized_corr(self, method=method, min_periods=min_periods)
    except TypeError:
        return _original_corr(self, method=method, min_periods=min_periods, numeric_only=numeric_only)


def apply_dataframe_pairwise_patches():
    """Apply DataFrame pairwise operation patches to pandas."""
    from .._patch import patch

    patch(pd.DataFrame, 'cov', _patched_cov)
    patch(pd.DataFrame, 'corr', _patched_corr)
