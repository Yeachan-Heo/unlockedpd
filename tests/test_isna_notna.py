"""Tests for isna/notna operations."""
import numpy as np
import pandas as pd
import pytest

import unlockedpd


class TestIsna:
    def test_basic_isna(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, np.nan]})
        unlockedpd.config.enabled = False
        expected = df.isna()
        unlockedpd.config.enabled = True
        result = df.isna()
        pd.testing.assert_frame_equal(result, expected)

    def test_no_nan(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        unlockedpd.config.enabled = False
        expected = df.isna()
        unlockedpd.config.enabled = True
        result = df.isna()
        pd.testing.assert_frame_equal(result, expected)

    def test_all_nan(self):
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        unlockedpd.config.enabled = False
        expected = df.isna()
        unlockedpd.config.enabled = True
        result = df.isna()
        pd.testing.assert_frame_equal(result, expected)

    def test_isnull_alias(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        result_isna = df.isna()
        result_isnull = df.isnull()
        pd.testing.assert_frame_equal(result_isna, result_isnull)

    def test_multiple_columns(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        unlockedpd.config.enabled = False
        expected = df.isna()
        unlockedpd.config.enabled = True
        result = df.isna()
        pd.testing.assert_frame_equal(result, expected)

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': ['x', 'y', None],
            'c': [1, 2, 3]
        })
        unlockedpd.config.enabled = False
        expected = df.isna()
        unlockedpd.config.enabled = True
        result = df.isna()
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        unlockedpd.config.enabled = False
        expected = df.isna()
        unlockedpd.config.enabled = True
        result = df.isna()
        pd.testing.assert_frame_equal(result, expected)


class TestNotna:
    def test_basic_notna(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, np.nan]})
        unlockedpd.config.enabled = False
        expected = df.notna()
        unlockedpd.config.enabled = True
        result = df.notna()
        pd.testing.assert_frame_equal(result, expected)

    def test_no_nan(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        unlockedpd.config.enabled = False
        expected = df.notna()
        unlockedpd.config.enabled = True
        result = df.notna()
        pd.testing.assert_frame_equal(result, expected)

    def test_all_nan(self):
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        unlockedpd.config.enabled = False
        expected = df.notna()
        unlockedpd.config.enabled = True
        result = df.notna()
        pd.testing.assert_frame_equal(result, expected)

    def test_notnull_alias(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        result_notna = df.notna()
        result_notnull = df.notnull()
        pd.testing.assert_frame_equal(result_notna, result_notnull)

    def test_multiple_columns(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        unlockedpd.config.enabled = False
        expected = df.notna()
        unlockedpd.config.enabled = True
        result = df.notna()
        pd.testing.assert_frame_equal(result, expected)

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': ['x', 'y', None],
            'c': [1, 2, 3]
        })
        unlockedpd.config.enabled = False
        expected = df.notna()
        unlockedpd.config.enabled = True
        result = df.notna()
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        unlockedpd.config.enabled = False
        expected = df.notna()
        unlockedpd.config.enabled = True
        result = df.notna()
        pd.testing.assert_frame_equal(result, expected)


class TestIsnaNotnaInverse:
    def test_isna_notna_are_inverse(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan],
            'b': [np.nan, 2.0, np.nan, 4.0]
        })
        isna_result = df.isna()
        notna_result = df.notna()

        # They should be exact inverses
        pd.testing.assert_frame_equal(isna_result, ~notna_result)
        pd.testing.assert_frame_equal(~isna_result, notna_result)
