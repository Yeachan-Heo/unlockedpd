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
        from unlockedpd.ops.aggregations import optimized_mean
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan],
            'b': [np.nan, 2.0, np.nan, 4.0],
            'c': [1.0, np.nan, 3.0, np.nan],
        })

        result = optimized_mean(df, axis=0)
        expected = df.mean(axis=0)
        pd.testing.assert_series_equal(result, expected)

    def test_sparse_nan(self):
        """Very sparse NaN (only 1 NaN in large array)."""
        from unlockedpd.ops.aggregations import optimized_sum
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))
        df.iloc[50, 5] = np.nan  # Single NaN

        result = optimized_sum(df, axis=0, skipna=True)
        expected = df.sum(axis=0, skipna=True)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)


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
