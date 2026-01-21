"""Tests for fill operations."""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')


class TestFillna:
    """Test suite for fill operations."""

    @pytest.fixture
    def df_with_nan(self):
        """DataFrame with NaN values."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))
        df.iloc[::3, :] = np.nan
        return df

    def test_ffill(self, df_with_nan):
        """Test forward fill."""
        from unlockedpd.ops.fillna import optimized_ffill
        result = optimized_ffill(df_with_nan)
        expected = df_with_nan.ffill()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_bfill(self, df_with_nan):
        """Test backward fill."""
        from unlockedpd.ops.fillna import optimized_bfill
        result = optimized_bfill(df_with_nan)
        expected = df_with_nan.bfill()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_fillna_scalar(self, df_with_nan):
        """Test fill with scalar."""
        from unlockedpd.ops.fillna import optimized_fillna
        result = optimized_fillna(df_with_nan, value=0.0)
        expected = df_with_nan.fillna(0.0)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_empty_dataframe(self):
        """Test fill on empty DataFrame."""
        from unlockedpd.ops.fillna import optimized_ffill
        df = pd.DataFrame()
        with pytest.raises(TypeError):
            optimized_ffill(df)

    def test_all_nan(self):
        """Test fill on all-NaN DataFrame."""
        from unlockedpd.ops.fillna import optimized_ffill, optimized_bfill
        df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_single_row(self):
        """Test fill on single row DataFrame."""
        from unlockedpd.ops.fillna import optimized_ffill
        df = pd.DataFrame({'a': [np.nan], 'b': [1.0]})
        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_no_nan(self):
        """Test fill on DataFrame with no NaN."""
        from unlockedpd.ops.fillna import optimized_ffill
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        result = optimized_ffill(df)
        expected = df.ffill()
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
