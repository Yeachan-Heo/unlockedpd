"""Tests for DataFrame corr/cov operations."""
import pytest
import pandas as pd
import numpy as np
import unlockedpd


@pytest.fixture(autouse=True)
def setup():
    unlockedpd.config.enabled = True
    yield
    unlockedpd.config.enabled = True


class TestCorr:
    def test_basic_corr(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 5), columns=list('abcde'))
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_diagonal_is_one(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df.corr()
        assert result.loc['a', 'a'] == 1.0
        assert result.loc['b', 'b'] == 1.0

    def test_corr_with_nan(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0], 'b': [4.0, 5.0, np.nan, 7.0]})
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_corr_perfect_positive(self):
        """Test perfect positive correlation."""
        df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 4, 6, 8]})
        result = df.corr()
        assert abs(result.loc['a', 'b'] - 1.0) < 1e-10

    def test_corr_perfect_negative(self):
        """Test perfect negative correlation."""
        df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [8, 6, 4, 2]})
        result = df.corr()
        assert abs(result.loc['a', 'b'] - (-1.0)) < 1e-10

    def test_corr_symmetry(self):
        """Test that correlation matrix is symmetric."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(50, 4), columns=list('abcd'))
        result = df.corr()
        for i in result.columns:
            for j in result.columns:
                assert abs(result.loc[i, j] - result.loc[j, i]) < 1e-10


class TestCov:
    def test_basic_cov(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 5), columns=list('abcde'))
        unlockedpd.config.enabled = False
        expected = df.cov()
        unlockedpd.config.enabled = True
        result = df.cov()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cov_with_nan(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0], 'b': [4.0, 5.0, np.nan, 7.0]})
        unlockedpd.config.enabled = False
        expected = df.cov()
        unlockedpd.config.enabled = True
        result = df.cov()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cov_ddof(self):
        """Test different ddof values."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(20, 3), columns=list('abc'))

        for ddof in [0, 1, 2]:
            unlockedpd.config.enabled = False
            expected = df.cov(ddof=ddof)
            unlockedpd.config.enabled = True
            result = df.cov(ddof=ddof)
            pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cov_symmetry(self):
        """Test that covariance matrix is symmetric."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(50, 4), columns=list('abcd'))
        result = df.cov()
        for i in result.columns:
            for j in result.columns:
                assert abs(result.loc[i, j] - result.loc[j, i]) < 1e-10


class TestEdgeCases:
    def test_constant_column(self):
        """Test correlation with constant column."""
        df = pd.DataFrame({'a': [1.0, 1.0, 1.0], 'b': [1.0, 2.0, 3.0]})
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pd.DataFrame()
        result = df.corr()
        assert result.empty

    def test_single_column(self):
        """Test single column DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        result = df.corr()
        assert result.loc['a', 'a'] == 1.0

    def test_two_rows(self):
        """Test DataFrame with only 2 rows."""
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_all_nan_column(self):
        """Test with all NaN column."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [np.nan, np.nan, np.nan]})
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected)

    def test_mixed_dtypes(self):
        """Test with mixed numeric dtypes."""
        df = pd.DataFrame({
            'a': np.array([1, 2, 3, 4], dtype=np.int32),
            'b': np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
            'c': np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        })
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-6)


class TestLargeData:
    def test_large_corr(self):
        """Test correlation on large DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(10000, 10))
        unlockedpd.config.enabled = False
        expected = df.corr()
        unlockedpd.config.enabled = True
        result = df.corr()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_large_cov(self):
        """Test covariance on large DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(10000, 10))
        unlockedpd.config.enabled = False
        expected = df.cov()
        unlockedpd.config.enabled = True
        result = df.cov()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
