"""Tests for aggregation operations."""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')


class TestAggregations:
    """Test suite for aggregation operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(np.random.randn(1000, 100))

    @pytest.fixture
    def df_with_nan(self):
        """DataFrame with NaN values."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))
        df.iloc[::3, ::2] = np.nan
        return df

    def test_sum_axis0(self, sample_df):
        """Test sum along axis=0."""
        from unlockedpd.ops.aggregations import optimized_sum
        result = optimized_sum(sample_df, axis=0)
        expected = sample_df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sum_axis1(self, sample_df):
        """Test sum along axis=1."""
        from unlockedpd.ops.aggregations import optimized_sum
        result = optimized_sum(sample_df, axis=1)
        expected = sample_df.sum(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_mean_axis0(self, sample_df):
        """Test mean along axis=0."""
        from unlockedpd.ops.aggregations import optimized_mean
        result = optimized_mean(sample_df, axis=0)
        expected = sample_df.mean(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_std_axis0(self, sample_df):
        """Test std along axis=0."""
        from unlockedpd.ops.aggregations import optimized_std
        result = optimized_std(sample_df, axis=0)
        expected = sample_df.std(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_var_axis0(self, sample_df):
        """Test var along axis=0."""
        from unlockedpd.ops.aggregations import optimized_var
        result = optimized_var(sample_df, axis=0)
        expected = sample_df.var(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_min_axis0(self, sample_df):
        """Test min along axis=0."""
        from unlockedpd.ops.aggregations import optimized_min
        result = optimized_min(sample_df, axis=0)
        expected = sample_df.min(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_max_axis0(self, sample_df):
        """Test max along axis=0."""
        from unlockedpd.ops.aggregations import optimized_max
        result = optimized_max(sample_df, axis=0)
        expected = sample_df.max(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_median_axis0(self, sample_df):
        """Test median along axis=0."""
        from unlockedpd.ops.aggregations import optimized_median
        result = optimized_median(sample_df, axis=0)
        expected = sample_df.median(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_prod_axis0(self, sample_df):
        """Test prod along axis=0."""
        from unlockedpd.ops.aggregations import optimized_prod
        # Use smaller values to avoid overflow
        df = pd.DataFrame(np.random.uniform(0.9, 1.1, (100, 10)))
        result = optimized_prod(df, axis=0)
        expected = df.prod(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sum_with_nan(self, df_with_nan):
        """Test sum with NaN values."""
        from unlockedpd.ops.aggregations import optimized_sum
        result = optimized_sum(df_with_nan, axis=0, skipna=True)
        expected = df_with_nan.sum(axis=0, skipna=True)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_mean_with_nan(self, df_with_nan):
        """Test mean with NaN values."""
        from unlockedpd.ops.aggregations import optimized_mean
        result = optimized_mean(df_with_nan, axis=0, skipna=True)
        expected = df_with_nan.mean(axis=0, skipna=True)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_empty_dataframe(self):
        """Test operations on empty DataFrame."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame()
        with pytest.raises(TypeError):
            optimized_sum(df)

    def test_all_nan_column(self):
        """Test operations on DataFrame with all-NaN column."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [1.0, 2.0, 3.0]})
        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_single_row(self):
        """Test operations on single row DataFrame."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})
        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_single_column(self):
        """Test operations on single column DataFrame."""
        from unlockedpd.ops.aggregations import optimized_sum, optimized_mean
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = optimized_sum(df, axis=0)
        expected = df.sum(axis=0)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)
