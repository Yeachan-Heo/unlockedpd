"""Tests for DataFrame.quantile() operation."""
import numpy as np
import pandas as pd
import pytest
import unlockedpd


class TestQuantile:
    """Test suite for quantile operations."""

    def test_median(self):
        """Test quantile at 0.5 (median)."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0], 'b': [5.0, 4.0, 3.0, 2.0, 1.0]})
        unlockedpd.unpatch_all()
        expected = df.quantile(0.5)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.5)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_quartiles(self):
        """Test multiple quantiles."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        unlockedpd.unpatch_all()
        expected = df.quantile([0.25, 0.5, 0.75])
        unlockedpd._apply_all_patches()
        result = df.quantile([0.25, 0.5, 0.75])
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_quantile_with_nan(self):
        """Test quantile with NaN values."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0, 5.0]})
        unlockedpd.unpatch_all()
        expected = df.quantile(0.5)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.5)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_quantile_extremes(self):
        """Test quantile at extremes (0.0 and 1.0)."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        for q in [0.0, 1.0]:
            unlockedpd.unpatch_all()
            expected = df.quantile(q)
            unlockedpd._apply_all_patches()
            result = df.quantile(q)
            pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_axis_1(self):
        """Test quantile along axis=1."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        unlockedpd.unpatch_all()
        expected = df.quantile(0.5, axis=1)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.5, axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_quantile_empty_df(self):
        """Test quantile on empty DataFrame."""
        df = pd.DataFrame()
        unlockedpd.unpatch_all()
        expected = df.quantile(0.5)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.5)
        pd.testing.assert_series_equal(result, expected)

    def test_quantile_all_nan(self):
        """Test quantile with all NaN values."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.quantile(0.5)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.5)
        pd.testing.assert_series_equal(result, expected)

    def test_quantile_linear_interpolation(self):
        """Test quantile with linear interpolation."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]})
        # 0.3 quantile with linear interpolation
        unlockedpd.unpatch_all()
        expected = df.quantile(0.3)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.3)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_quantile_multiple_columns(self):
        """Test quantile with multiple columns."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
            'c': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        unlockedpd.unpatch_all()
        expected = df.quantile(0.5)
        unlockedpd._apply_all_patches()
        result = df.quantile(0.5)
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_quantile_multiple_values(self):
        """Test multiple quantiles on multiple columns."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        unlockedpd.unpatch_all()
        expected = df.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        unlockedpd._apply_all_patches()
        result = df.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
