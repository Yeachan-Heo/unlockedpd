"""Tests for fillna and dropna operations."""
import numpy as np
import pandas as pd
import pytest
import unlockedpd


class TestFillna:
    """Tests for fillna operation."""

    def test_fillna_scalar(self):
        """Test fillna with scalar value."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.fillna(0.0)
        unlockedpd._apply_all_patches()
        result = df.fillna(0.0)
        pd.testing.assert_frame_equal(result, expected)

    def test_fillna_ffill(self):
        """Test fillna with forward fill."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, np.nan, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.fillna(method='ffill')
        unlockedpd._apply_all_patches()
        result = df.fillna(method='ffill')
        pd.testing.assert_frame_equal(result, expected)

    def test_fillna_bfill(self):
        """Test fillna with backward fill."""
        df = pd.DataFrame({'a': [np.nan, 2.0, 3.0], 'b': [np.nan, np.nan, 6.0]})
        unlockedpd.unpatch_all()
        expected = df.fillna(method='bfill')
        unlockedpd._apply_all_patches()
        result = df.fillna(method='bfill')
        pd.testing.assert_frame_equal(result, expected)

    def test_fillna_no_nan(self):
        """Test fillna with no NaN values."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        unlockedpd.unpatch_all()
        expected = df.fillna(0.0)
        unlockedpd._apply_all_patches()
        result = df.fillna(0.0)
        pd.testing.assert_frame_equal(result, expected)

    def test_fillna_all_nan(self):
        """Test fillna with all NaN values."""
        df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.fillna(0.0)
        unlockedpd._apply_all_patches()
        result = df.fillna(0.0)
        pd.testing.assert_frame_equal(result, expected)


class TestDropna:
    """Tests for dropna operation."""

    def test_dropna_rows_any(self):
        """Test dropna dropping rows with any NaN."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.dropna()
        unlockedpd._apply_all_patches()
        result = df.dropna()
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_rows_all(self):
        """Test dropna dropping rows with all NaN."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, np.nan, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.dropna(how='all')
        unlockedpd._apply_all_patches()
        result = df.dropna(how='all')
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_cols_any(self):
        """Test dropna dropping columns with any NaN."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, np.nan, 6.0]})
        unlockedpd.unpatch_all()
        expected = df.dropna(axis=1)
        unlockedpd._apply_all_patches()
        result = df.dropna(axis=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_dropna_thresh(self):
        """Test dropna with threshold."""
        df = pd.DataFrame({'a': [1.0, np.nan, np.nan], 'b': [4.0, 5.0, np.nan], 'c': [7.0, 8.0, 9.0]})
        unlockedpd.unpatch_all()
        expected = df.dropna(thresh=2)
        unlockedpd._apply_all_patches()
        result = df.dropna(thresh=2)
        pd.testing.assert_frame_equal(result, expected)

    def test_no_nan(self):
        """Test dropna with no NaN values."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        unlockedpd.unpatch_all()
        expected = df.dropna()
        unlockedpd._apply_all_patches()
        result = df.dropna()
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_after_drop(self):
        """Test dropna resulting in empty DataFrame."""
        df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        unlockedpd.unpatch_all()
        expected = df.dropna()
        unlockedpd._apply_all_patches()
        result = df.dropna()
        pd.testing.assert_frame_equal(result, expected)
