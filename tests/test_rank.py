"""Tests for rank operations."""
import pytest
import pandas as pd
import numpy as np


class TestRankAxis1:
    """Tests for rank(axis=1)"""

    def test_basic_rank_axis1(self):
        """Test basic rank axis=1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1)

        pd.testing.assert_frame_equal(result, expected)

    def test_rank_axis1_descending(self):
        """Test rank axis=1 descending."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1, ascending=False)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1, ascending=False)

        pd.testing.assert_frame_equal(result, expected)

    def test_rank_axis1_with_nan(self):
        """Test rank axis=1 handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [2.0, 2.0, np.nan],
            'c': [3.0, 1.0, 1.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1)

        pd.testing.assert_frame_equal(result, expected)

    def test_rank_axis1_na_option_top(self):
        """Test rank with na_option='top'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [2.0, 2.0, np.nan],
            'c': [3.0, 1.0, 1.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1, na_option='top')

        unlockedpd.config.enabled = True
        result = df.rank(axis=1, na_option='top')

        pd.testing.assert_frame_equal(result, expected)

    def test_rank_axis1_na_option_bottom(self):
        """Test rank with na_option='bottom'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [2.0, 2.0, np.nan],
            'c': [3.0, 1.0, 1.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1, na_option='bottom')

        unlockedpd.config.enabled = True
        result = df.rank(axis=1, na_option='bottom')

        pd.testing.assert_frame_equal(result, expected)


class TestRankAxis0:
    """Tests for rank(axis=0)"""

    def test_basic_rank_axis0(self):
        """Test basic rank axis=0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=0)

        unlockedpd.config.enabled = True
        result = df.rank(axis=0)

        pd.testing.assert_frame_equal(result, expected)


class TestRankMethods:
    """Tests for different rank methods."""

    @pytest.mark.parametrize("method", ['average', 'min', 'max', 'first', 'dense'])
    def test_rank_methods(self, method):
        """Test all rank methods."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 3.0],
            'b': [4.0, 4.0, 5.0, 6.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1, method=method)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1, method=method)

        pd.testing.assert_frame_equal(result, expected)


class TestRankPct:
    """Tests for percentage ranks."""

    def test_rank_pct(self):
        """Test percentage ranks."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1, pct=True)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1, pct=True)

        pd.testing.assert_frame_equal(result, expected)
