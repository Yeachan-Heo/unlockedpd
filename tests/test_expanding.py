"""Tests for expanding operations.

Note: Expanding operations are not yet implemented.
This file serves as a placeholder for future tests.
"""
import pytest
import pandas as pd
import numpy as np


class TestExpandingMean:
    """Tests for expanding().mean()"""

    @pytest.mark.skip(reason="Expanding operations not yet implemented")
    def test_basic_expanding_mean(self):
        """Test basic expanding mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Expanding operations not yet implemented")
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

    @pytest.mark.skip(reason="Expanding operations not yet implemented")
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

    @pytest.mark.skip(reason="Expanding operations not yet implemented")
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

    @pytest.mark.skip(reason="Expanding operations not yet implemented")
    def test_basic_expanding_std(self):
        """Test basic expanding std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.expanding().std()

        unlockedpd.config.enabled = True
        result = df.expanding().std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
