"""Tests for expanding operations."""
import pytest
import pandas as pd
import numpy as np


class TestExpandingMean:
    """Tests for expanding().mean()"""

    def test_basic_expanding_mean(self):
        """Test basic expanding mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_mean_with_nan(self):
        """Test expanding mean handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_mean_min_periods(self):
        """Test expanding mean with min_periods."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding(min_periods=5).mean()

        unlockedpd.config.enabled = True
        result = df.expanding(min_periods=5).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestExpandingSum:
    """Tests for expanding().sum()"""

    def test_basic_expanding_sum(self):
        """Test basic expanding sum matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().sum()

        unlockedpd.config.enabled = True
        result = df.expanding().sum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestExpandingStd:
    """Tests for expanding().std()"""

    def test_basic_expanding_std(self):
        """Test basic expanding std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().std()

        unlockedpd.config.enabled = True
        result = df.expanding().std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestExpandingMedian:
    """Tests for expanding().median()"""

    def test_basic_expanding_median(self):
        """Test basic expanding median matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().median()

        unlockedpd.config.enabled = True
        result = df.expanding().median()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_median_with_nan(self):
        """Test expanding median handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().median()

        unlockedpd.config.enabled = True
        result = df.expanding().median()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_median_min_periods(self):
        """Test expanding median with min_periods."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding(min_periods=5).median()

        unlockedpd.config.enabled = True
        result = df.expanding(min_periods=5).median()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestExpandingQuantile:
    """Tests for expanding().quantile()"""

    def test_basic_expanding_quantile(self):
        """Test basic expanding quantile matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().quantile(0.75)

        unlockedpd.config.enabled = True
        result = df.expanding().quantile(0.75)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_quantile_with_nan(self):
        """Test expanding quantile handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().quantile(0.25)

        unlockedpd.config.enabled = True
        result = df.expanding().quantile(0.25)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_quantile_min_periods(self):
        """Test expanding quantile with min_periods."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding(min_periods=10).quantile(0.9)

        unlockedpd.config.enabled = True
        result = df.expanding(min_periods=10).quantile(0.9)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
