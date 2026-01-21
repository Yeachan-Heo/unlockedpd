"""Extreme edge case tests for pandas compatibility.

Tests cover:
- NaN patterns (all NaN, partial NaN, NaN at boundaries)
- Extreme values (inf, -inf, very large, very small)
- Strange dtypes and mixed scenarios
- Single element, zero-length dimensions
- Numerical stability edge cases
"""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')


class TestAggregationsExtreme:
    """Extreme edge cases for aggregation operations."""

    def test_all_nan_dataframe(self):
        """All NaN values."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean, optimized_std
        df = pd.DataFrame(np.full((100, 10), np.nan))

        result_sum = optimized_sum(df, axis=0)
        expected_sum = df.sum(axis=0)
        pd.testing.assert_series_equal(result_sum, expected_sum)

        result_mean = optimized_mean(df, axis=0)
        expected_mean = df.mean(axis=0)
        pd.testing.assert_series_equal(result_mean, expected_mean)

    def test_inf_values(self):
        """DataFrame with infinity values."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean, optimized_min, optimized_max
        df = pd.DataFrame({
            'a': [1.0, np.inf, 3.0],
            'b': [-np.inf, 2.0, 3.0],
            'c': [np.inf, -np.inf, 1.0]
        })

        result_sum = optimized_sum(df, axis=0)
        expected_sum = df.sum(axis=0)
        pd.testing.assert_series_equal(result_sum, expected_sum)

        result_min = optimized_min(df, axis=0)
        expected_min = df.min(axis=0)
        pd.testing.assert_series_equal(result_min, expected_min)

        result_max = optimized_max(df, axis=0)
        expected_max = df.max(axis=0)
        pd.testing.assert_series_equal(result_max, expected_max)

    def test_very_large_values(self):
        """Very large values near float64 limits."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        large = 1e308
        df = pd.DataFrame({'a': [large, large/2], 'b': [-large, large]})

        result_sum = optimized_sum(df, axis=0)
        expected_sum = df.sum(axis=0)
        pd.testing.assert_series_equal(result_sum, expected_sum)

    def test_very_small_values(self):
        """Very small values near zero."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        tiny = 1e-308
        df = pd.DataFrame({'a': [tiny, tiny*2], 'b': [-tiny, tiny]})

        result_sum = optimized_sum(df, axis=0)
        expected_sum = df.sum(axis=0)
        pd.testing.assert_series_equal(result_sum, expected_sum)

    def test_mixed_nan_inf(self):
        """Mixed NaN and infinity."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({
            'a': [np.nan, np.inf, 1.0],
            'b': [-np.inf, np.nan, 2.0],
            'c': [np.nan, np.nan, np.nan]
        })

        result = optimized_sum(df, axis=0, skipna=True)
        expected = df.sum(axis=0, skipna=True)
        pd.testing.assert_series_equal(result, expected)

    def test_skipna_false(self):
        """Test skipna=False behavior."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})

        result = optimized_sum(df, axis=0, skipna=False)
        expected = df.sum(axis=0, skipna=False)
        pd.testing.assert_series_equal(result, expected)

    def test_std_constant_values(self):
        """Std of constant values (zero variance)."""
        from unlockedpd.ops.aggregations import optimized_std, optimized_var
        df = pd.DataFrame({'a': [5.0, 5.0, 5.0], 'b': [1.0, 2.0, 3.0]})

        result = optimized_std(df, axis=0)
        expected = df.std(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_std_single_value(self):
        """Std with single value (undefined)."""
        from unlockedpd.ops.aggregations import optimized_std
        df = pd.DataFrame({'a': [5.0], 'b': [3.0]})

        result = optimized_std(df, axis=0)
        expected = df.std(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_median_even_odd(self):
        """Median with even and odd number of elements."""
        from unlockedpd.ops.aggregations import optimized_median
        df_odd = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        df_even = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]})

        result_odd = optimized_median(df_odd, axis=0)
        expected_odd = df_odd.median(axis=0)
        pd.testing.assert_series_equal(result_odd, expected_odd)

        result_even = optimized_median(df_even, axis=0)
        expected_even = df_even.median(axis=0)
        pd.testing.assert_series_equal(result_even, expected_even)

    def test_prod_with_zero(self):
        """Product with zero values."""
        from unlockedpd.ops.aggregations import optimized_prod
        df = pd.DataFrame({'a': [1.0, 0.0, 3.0], 'b': [2.0, 2.0, 2.0]})

        result = optimized_prod(df, axis=0)
        expected = df.prod(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_prod_with_negative(self):
        """Product with negative values."""
        from unlockedpd.ops.aggregations import optimized_prod
        df = pd.DataFrame({'a': [-1.0, 2.0, -3.0], 'b': [-2.0, -2.0, 2.0]})

        result = optimized_prod(df, axis=0)
        expected = df.prod(axis=0)
        pd.testing.assert_series_equal(result, expected)


class TestFillnaExtreme:
    """Extreme edge cases for fill operations."""

    def test_nan_at_start(self):
        """NaN at start of column (ffill can't fill)."""
        from unlockedpd.ops.fillna import optimized_ffill
        df = pd.DataFrame({'a': [np.nan, np.nan, 1.0, 2.0], 'b': [1.0, np.nan, np.nan, 4.0]})

        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected)

    def test_nan_at_end(self):
        """NaN at end of column (bfill can't fill)."""
        from unlockedpd.ops.fillna import optimized_bfill
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan, np.nan], 'b': [1.0, np.nan, np.nan, 4.0]})

        result = optimized_bfill(df)
        expected = df.bfill()
        pd.testing.assert_frame_equal(result, expected)

    def test_alternating_nan(self):
        """Alternating NaN pattern."""
        from unlockedpd.ops.fillna import optimized_ffill, optimized_bfill
        df = pd.DataFrame({'a': [1.0, np.nan, 2.0, np.nan, 3.0]})

        result_f = optimized_ffill(df)
        expected_f = df.ffill()
        pd.testing.assert_frame_equal(result_f, expected_f)

        result_b = optimized_bfill(df)
        expected_b = df.bfill()
        pd.testing.assert_frame_equal(result_b, expected_b)

    def test_fillna_with_inf(self):
        """Fill NaN with infinity."""
        from unlockedpd.ops.fillna import optimized_fillna
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, np.nan]})

        result = optimized_fillna(df, value=np.inf)
        expected = df.fillna(np.inf)
        pd.testing.assert_frame_equal(result, expected)

    def test_fillna_with_negative(self):
        """Fill NaN with negative value."""
        from unlockedpd.ops.fillna import optimized_fillna
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})

        result = optimized_fillna(df, value=-999.0)
        expected = df.fillna(-999.0)
        pd.testing.assert_frame_equal(result, expected)

    def test_single_column_all_nan(self):
        """Single column all NaN."""
        from unlockedpd.ops.fillna import optimized_ffill
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})

        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected)


class TestElementWiseExtreme:
    """Extreme edge cases for element-wise operations."""

    def test_clip_with_inf_bounds(self):
        """Clip with infinity bounds."""
        from unlockedpd.ops.element_wise import optimized_clip
        df = pd.DataFrame({'a': [1.0, 100.0, -100.0], 'b': [np.inf, -np.inf, 0.0]})

        result = optimized_clip(df, lower=-np.inf, upper=np.inf)
        expected = df.clip(lower=-np.inf, upper=np.inf)
        pd.testing.assert_frame_equal(result, expected)

    def test_clip_with_nan(self):
        """Clip preserves NaN."""
        from unlockedpd.ops.element_wise import optimized_clip
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, np.nan]})

        result = optimized_clip(df, lower=0.0, upper=2.0)
        expected = df.clip(lower=0.0, upper=2.0)
        pd.testing.assert_frame_equal(result, expected)

    def test_abs_with_inf(self):
        """Absolute value of infinity."""
        from unlockedpd.ops.element_wise import optimized_abs
        df = pd.DataFrame({'a': [np.inf, -np.inf, 0.0], 'b': [-1.0, 1.0, np.nan]})

        result = optimized_abs(df)
        expected = df.abs()
        pd.testing.assert_frame_equal(result, expected)

    def test_round_negative_decimals(self):
        """Round to negative decimals (tens, hundreds)."""
        from unlockedpd.ops.element_wise import optimized_round
        df = pd.DataFrame({'a': [123.456, 789.012], 'b': [1234.5, 5678.9]})

        # Round to nearest 10
        result = optimized_round(df, decimals=-1)
        expected = df.round(decimals=-1)
        pd.testing.assert_frame_equal(result, expected)

    def test_round_many_decimals(self):
        """Round to many decimal places."""
        from unlockedpd.ops.element_wise import optimized_round
        df = pd.DataFrame({'a': [1.123456789, 2.987654321]})

        result = optimized_round(df, decimals=6)
        expected = df.round(decimals=6)
        pd.testing.assert_frame_equal(result, expected)

    def test_clip_lower_equals_upper(self):
        """Clip where lower equals upper."""
        from unlockedpd.ops.element_wise import optimized_clip
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})

        result = optimized_clip(df, lower=2.0, upper=2.0)
        expected = df.clip(lower=2.0, upper=2.0)
        pd.testing.assert_frame_equal(result, expected)


class TestCorrelationExtreme:
    """Extreme edge cases for correlation operations."""

    def test_corr_with_nan(self):
        """Correlation with NaN values."""
        from unlockedpd.ops.correlation import optimized_corr
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 5))
        df.iloc[::5, 0] = np.nan  # 20% NaN in first column
        df.iloc[::3, 1] = np.nan  # 33% NaN in second column

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_perfectly_correlated(self):
        """Perfectly correlated columns."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [2.0, 4.0, 6.0, 8.0]})

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_negatively_correlated(self):
        """Perfectly negatively correlated columns."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [4.0, 3.0, 2.0, 1.0]})

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_all_same_value(self):
        """All columns have same value (undefined correlation)."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame({'a': [5.0, 5.0, 5.0], 'b': [5.0, 5.0, 5.0]})

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_cov_with_nan(self):
        """Covariance with NaN values."""
        from unlockedpd.ops.correlation import optimized_cov
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(50, 4))
        df.iloc[::4, :] = np.nan  # 25% NaN rows

        result = optimized_cov(df)
        expected = df.cov()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cov_ddof(self):
        """Covariance with different ddof."""
        from unlockedpd.ops.correlation import optimized_cov
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(20, 3))

        result = optimized_cov(df, ddof=0)
        expected = df.cov(ddof=0)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_two_rows(self):
        """Correlation with only two rows."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0], 'c': [5.0, 6.0]})

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_sum_catastrophic_cancellation(self):
        """Sum where catastrophic cancellation could occur."""
        from unlockedpd.ops.aggregations import optimized_sum
        # Large values that nearly cancel
        df = pd.DataFrame({'a': [1e16, 1.0, -1e16]})

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_var_numerical_stability(self):
        """Variance with values that could cause numerical issues."""
        from unlockedpd.ops.aggregations import optimized_var
        # Values offset from zero by large amount
        df = pd.DataFrame({'a': [1e10 + 1, 1e10 + 2, 1e10 + 3]})

        result = optimized_var(df, axis=0)
        expected = df.var(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_corr_near_zero_variance(self):
        """Correlation with near-zero variance."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame({
            'a': [1.0, 1.0 + 1e-15, 1.0],  # Nearly constant
            'b': [1.0, 2.0, 3.0]
        })

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestLargeArrays:
    """Tests with larger arrays to exercise parallel paths."""

    def test_aggregations_wide_dataframe(self):
        """Test aggregations on wide DataFrame (many columns)."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 500))  # 500 columns

        result_sum = optimized_sum(df, axis=0)
        expected_sum = df.sum(axis=0)
        pd.testing.assert_series_equal(result_sum, expected_sum, rtol=1e-10)

    def test_fillna_wide_dataframe(self):
        """Test fillna on wide DataFrame."""
        from unlockedpd.ops.fillna import optimized_ffill
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 300))
        df.iloc[::3, :] = np.nan  # 33% NaN

        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected)

    def test_correlation_many_columns(self):
        """Test correlation with many columns."""
        from unlockedpd.ops.correlation import optimized_corr
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(50, 50))  # 50x50 correlation matrix

        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
"""
Extreme edge case tests for unlockedpd numerical operations.
"""

import numpy as np
import pandas as pd
import pytest


class TestDtypeEdgeCases:
    """Tests for different numeric dtypes."""

    def test_int_dtypes(self):
        """Test with integer dtypes (converted to float64)."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, dtype=np.int64)

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected.astype(float))

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        from unlockedpd.ops.aggregations import optimized_sum
        df = pd.DataFrame(np.random.randn(10, 5).astype(np.float32))

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        # Lower tolerance for float32
        pd.testing.assert_series_equal(result, expected.astype(float), rtol=1e-5)

    def test_bool_dtype(self):
        """Test with boolean dtype."""
        from unlockedpd.ops.aggregations import optimized_sum
        df = pd.DataFrame({'a': [True, False, True], 'b': [False, False, True]})

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected.astype(float))


class TestShapeEdgeCases:
    """Tests for unusual DataFrame shapes."""

    def test_single_element(self):
        """Single element DataFrame."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({'a': [42.0]})

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected)

        result = optimized_mean(df, axis=0)
        expected = df.mean(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_very_wide(self):
        """Very wide DataFrame (1 row, many columns)."""
        from unlockedpd.ops.aggregations import optimized_sum
        np.random.seed(42)
        df = pd.DataFrame([np.random.randn(1000)])

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_very_tall(self):
        """Very tall DataFrame (many rows, 1 column)."""
        from unlockedpd.ops.aggregations import optimized_sum
        np.random.seed(42)
        df = pd.DataFrame({'a': np.random.randn(10000)})

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)


class TestSpecialNaNPatterns:
    """Tests for special NaN patterns."""

    def test_nan_only_first_column(self):
        """Only first column has NaN."""
        from unlockedpd.ops.aggregations import optimized_mean
        df = pd.DataFrame({
            'a': [np.nan, np.nan, np.nan],
            'b': [1.0, 2.0, 3.0],
            'c': [4.0, 5.0, 6.0]
        })

        result = optimized_mean(df, axis=0)
        expected = df.mean(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_nan_only_last_column(self):
        """Only last column has NaN."""
        from unlockedpd.ops.aggregations import optimized_mean
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
            'c': [np.nan, np.nan, np.nan]
        })

        result = optimized_mean(df, axis=0)
        expected = df.mean(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_checkerboard_nan(self):
        """Checkerboard NaN pattern."""
        from unlockedpd.ops.fillna import optimized_ffill
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan],
            'b': [np.nan, 2.0, np.nan, 4.0],
            'c': [1.0, np.nan, 3.0, np.nan],
        })

        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected)

    def test_sparse_nan(self):
        """Very sparse NaN (only 1 NaN in large array)."""
        from unlockedpd.ops.fillna import optimized_fillna
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))
        df.iloc[50, 5] = np.nan  # Single NaN

        result = optimized_fillna(df, value=0.0)
        expected = df.fillna(0.0)
        pd.testing.assert_frame_equal(result, expected)


class TestBoundaryValues:
    """Tests for boundary values."""

    def test_max_float64(self):
        """Maximum float64 value."""
        from unlockedpd.ops.aggregations import optimized_max
        max_val = np.finfo(np.float64).max
        df = pd.DataFrame({'a': [1.0, max_val, 3.0]})

        result = optimized_max(df, axis=0)
        expected = df.max(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_min_float64(self):
        """Minimum float64 value."""
        from unlockedpd.ops.aggregations import optimized_min
        min_val = np.finfo(np.float64).min
        df = pd.DataFrame({'a': [1.0, min_val, 3.0]})

        result = optimized_min(df, axis=0)
        expected = df.min(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_tiny_positive(self):
        """Smallest positive float64."""
        from unlockedpd.ops.aggregations import optimized_sum
        tiny = np.finfo(np.float64).tiny
        df = pd.DataFrame({'a': [tiny, tiny, tiny]})

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_subnormal_numbers(self):
        """Subnormal/denormalized numbers."""
        from unlockedpd.ops.aggregations import optimized_sum
        subnormal = np.finfo(np.float64).tiny / 2
        df = pd.DataFrame({'a': [subnormal, subnormal * 2, subnormal * 3]})

        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected)


class TestAxis1Operations:
    """Tests for axis=1 operations."""

    def test_sum_axis1(self):
        """Sum along axis=1."""
        from unlockedpd.ops.aggregations import optimized_sum
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))

        result = optimized_sum(df, axis=1)
        expected = df.sum(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_mean_axis1(self):
        """Mean along axis=1."""
        from unlockedpd.ops.aggregations import optimized_mean
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))

        result = optimized_mean(df, axis=1)
        expected = df.mean(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_std_axis1(self):
        """Std along axis=1."""
        from unlockedpd.ops.aggregations import optimized_std
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))

        result = optimized_std(df, axis=1)
        expected = df.std(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_min_axis1(self):
        """Min along axis=1."""
        from unlockedpd.ops.aggregations import optimized_min
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))

        result = optimized_min(df, axis=1)
        expected = df.min(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_max_axis1(self):
        """Max along axis=1."""
        from unlockedpd.ops.aggregations import optimized_max
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))

        result = optimized_max(df, axis=1)
        expected = df.max(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sum_axis1_with_nan(self):
        """Sum along axis=1 with NaN."""
        from unlockedpd.ops.aggregations import optimized_sum
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [np.nan, 2.0, 3.0],
            'c': [1.0, 2.0, np.nan]
        })

        result = optimized_sum(df, axis=1, skipna=True)
        expected = df.sum(axis=1, skipna=True)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)
