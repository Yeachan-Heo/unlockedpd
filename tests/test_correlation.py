"""Tests for correlation operations."""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')


class TestCorrelation:
    """Test suite for correlation operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(np.random.randn(100, 10))

    def test_corr(self, sample_df):
        """Test correlation matrix."""
        from unlockedpd.ops.correlation import optimized_corr
        result = optimized_corr(sample_df)
        expected = sample_df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cov(self, sample_df):
        """Test covariance matrix."""
        from unlockedpd.ops.correlation import optimized_cov
        result = optimized_cov(sample_df)
        expected = sample_df.cov()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_diagonal(self, sample_df):
        """Test correlation diagonal is all 1s."""
        from unlockedpd.ops.correlation import optimized_corr
        result = optimized_corr(sample_df)
        diag = np.diag(result.values)
        np.testing.assert_allclose(diag, np.ones(len(diag)), rtol=1e-10)

    def test_corr_symmetric(self, sample_df):
        """Test correlation matrix is symmetric."""
        from unlockedpd.ops.correlation import optimized_corr
        result = optimized_corr(sample_df)
        np.testing.assert_allclose(result.values, result.values.T, rtol=1e-10)

    def test_empty_dataframe(self):
        """Test correlation on empty DataFrame."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame()
        with pytest.raises(TypeError):
            optimized_corr(df)

    def test_constant_column(self):
        """Test correlation with constant column (zero variance)."""
        from unlockedpd.ops.correlation import optimized_corr
        np.random.seed(42)
        df = pd.DataFrame({'a': [1.0, 1.0, 1.0], 'b': np.random.randn(3)})
        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_single_column(self):
        """Test correlation on single column DataFrame."""
        from unlockedpd.ops.correlation import optimized_corr
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = optimized_corr(df)
        expected = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_min_periods(self):
        """Test correlation with min_periods."""
        from unlockedpd.ops.correlation import optimized_corr
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(5, 3))
        df.iloc[0:3, 0] = np.nan  # Only 2 valid pairs
        result = optimized_corr(df, min_periods=3)
        expected = df.corr(min_periods=3)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
