"""Tests for DataFrame aggregate operations."""
import pytest
import pandas as pd
import numpy as np


class TestDataFrameMean:
    """Tests for df.mean()"""

    def test_mean_axis0(self):
        """Test mean along axis=0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.mean(axis=0)

        unlockedpd.config.enabled = True
        result = df.mean(axis=0)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_mean_axis1(self):
        """Test mean along axis=1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.mean(axis=1)

        unlockedpd.config.enabled = True
        result = df.mean(axis=1)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_mean_with_nan(self):
        """Test mean handles NaN correctly with skipna=True."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.mean(skipna=True)

        unlockedpd.config.enabled = True
        result = df.mean(skipna=True)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)


class TestDataFrameSum:
    """Tests for df.sum()"""

    def test_sum_axis0(self):
        """Test sum along axis=0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.sum(axis=0)

        unlockedpd.config.enabled = True
        result = df.sum(axis=0)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sum_axis1(self):
        """Test sum along axis=1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.sum(axis=1)

        unlockedpd.config.enabled = True
        result = df.sum(axis=1)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sum_with_nan(self):
        """Test sum handles NaN correctly with skipna=True."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.sum(skipna=True)

        unlockedpd.config.enabled = True
        result = df.sum(skipna=True)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)


class TestDataFrameStd:
    """Tests for df.std()"""

    def test_std_axis0(self):
        """Test std along axis=0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.std(axis=0)

        unlockedpd.config.enabled = True
        result = df.std(axis=0)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_std_axis1(self):
        """Test std along axis=1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.std(axis=1)

        unlockedpd.config.enabled = True
        result = df.std(axis=1)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_std_with_ddof(self):
        """Test std with different ddof parameter."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.std(ddof=0)

        unlockedpd.config.enabled = True
        result = df.std(ddof=0)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_std_with_nan(self):
        """Test std handles NaN correctly with skipna=True."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.std(skipna=True)

        unlockedpd.config.enabled = True
        result = df.std(skipna=True)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)


class TestDataFrameVar:
    """Tests for df.var()"""

    def test_var_axis0(self):
        """Test var along axis=0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.var(axis=0)

        unlockedpd.config.enabled = True
        result = df.var(axis=0)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_var_axis1(self):
        """Test var along axis=1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.var(axis=1)

        unlockedpd.config.enabled = True
        result = df.var(axis=1)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_var_with_ddof(self):
        """Test var with different ddof parameter."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.var(ddof=0)

        unlockedpd.config.enabled = True
        result = df.var(ddof=0)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_var_with_nan(self):
        """Test var handles NaN correctly with skipna=True."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.var(skipna=True)

        unlockedpd.config.enabled = True
        result = df.var(skipna=True)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)
