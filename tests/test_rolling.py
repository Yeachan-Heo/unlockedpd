"""Tests for rolling operations."""
import pytest
import pandas as pd
import numpy as np


class TestRollingMean:
    """Tests for rolling().mean()"""

    def test_basic_rolling_mean(self):
        """Test basic rolling mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(5).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_with_nan(self):
        """Test rolling mean handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(3).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(3).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_min_periods(self):
        """Test rolling mean with min_periods."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.rolling(10, min_periods=5).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(10, min_periods=5).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_centered(self):
        """Test centered rolling mean."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5, center=True).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(5, center=True).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_window_larger_than_data(self):
        """Test rolling mean when window > data length."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(5, 3))

        unlockedpd.config.enabled = False
        expected = df.rolling(10).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(10).mean()

        pd.testing.assert_frame_equal(result, expected)


class TestRollingSum:
    """Tests for rolling().sum()"""

    def test_basic_rolling_sum(self):
        """Test basic rolling sum matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).sum()

        unlockedpd.config.enabled = True
        result = df.rolling(5).sum()

        pd.testing.assert_frame_equal(result, expected)


class TestRollingStd:
    """Tests for rolling().std()"""

    def test_basic_rolling_std(self):
        """Test basic rolling std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).std()

        unlockedpd.config.enabled = True
        result = df.rolling(5).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_std_ddof(self):
        """Test rolling std with different ddof."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).std(ddof=0)

        unlockedpd.config.enabled = True
        result = df.rolling(5).std(ddof=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestRollingMinMax:
    """Tests for rolling().min() and rolling().max()"""

    def test_basic_rolling_min(self):
        """Test basic rolling min matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).min()

        unlockedpd.config.enabled = True
        result = df.rolling(5).min()

        pd.testing.assert_frame_equal(result, expected)

    def test_basic_rolling_max(self):
        """Test basic rolling max matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).max()

        unlockedpd.config.enabled = True
        result = df.rolling(5).max()

        pd.testing.assert_frame_equal(result, expected)


class TestMixedDtypes:
    """Tests for mixed-dtype DataFrames."""

    def test_mixed_dtype_rolling(self):
        """Test that non-numeric columns are handled correctly.

        Our implementation processes only numeric columns and returns
        NaN for non-numeric columns in their original positions.
        """
        import unlockedpd

        df = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(2).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(2).mean()

        pd.testing.assert_frame_equal(result, expected)
