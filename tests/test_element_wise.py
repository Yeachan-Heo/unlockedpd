"""Tests for element-wise operations."""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')


class TestElementWise:
    """Test suite for element-wise operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(np.random.randn(100, 10))

    def test_clip(self, sample_df):
        """Test clip operation."""
        from unlockedpd.ops.element_wise import optimized_clip
        result = optimized_clip(sample_df, lower=-1.0, upper=1.0)
        expected = sample_df.clip(lower=-1.0, upper=1.0)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_abs(self, sample_df):
        """Test abs operation."""
        from unlockedpd.ops.element_wise import optimized_abs
        result = optimized_abs(sample_df)
        expected = sample_df.abs()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_round(self, sample_df):
        """Test round operation."""
        from unlockedpd.ops.element_wise import optimized_round
        result = optimized_round(sample_df, decimals=2)
        expected = sample_df.round(decimals=2)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_empty_dataframe(self):
        """Test operations on empty DataFrame."""
        from unlockedpd.ops.element_wise import optimized_clip
        df = pd.DataFrame()
        with pytest.raises(TypeError):
            optimized_clip(df, lower=0, upper=1)

    def test_all_nan(self):
        """Test operations on all-NaN DataFrame."""
        from unlockedpd.ops.element_wise import optimized_clip, optimized_abs
        df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        result = optimized_abs(df)
        expected = df.abs()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_single_row(self):
        """Test operations on single row DataFrame."""
        from unlockedpd.ops.element_wise import optimized_clip
        df = pd.DataFrame({'a': [1.5], 'b': [-0.5]})
        result = optimized_clip(df, lower=0, upper=1)
        expected = df.clip(lower=0, upper=1)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
