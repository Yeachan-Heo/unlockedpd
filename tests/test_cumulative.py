"""Tests for cumulative operations."""
import pytest
import pandas as pd
import numpy as np


class TestCumSum:
    """Tests for cumsum()"""

    def test_basic_cumsum(self):
        """Test basic cumsum matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_with_nan(self):
        """Test cumsum handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_skipna_false(self):
        """Test cumsum with skipna=False."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cumsum(skipna=False)

        unlockedpd.config.enabled = True
        result = df.cumsum(skipna=False)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_all_nan(self):
        """Test cumsum with all NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [np.nan, np.nan, np.nan],
            'b': [1.0, 2.0, 3.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_large_values(self):
        """Test cumsum with large values."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(1000, 20) * 1e6)

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_single_column(self):
        """Test cumsum with single column DataFrame."""
        import unlockedpd

        df = pd.DataFrame({'a': np.random.randn(100)})

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestCumProd:
    """Tests for cumprod()"""

    def test_basic_cumprod(self):
        """Test basic cumprod matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 10) * 0.1)  # Small values to avoid overflow

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumprod_with_nan(self):
        """Test cumprod handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [2.0, np.nan, 3.0, 2.0, 2.0],
            'b': [np.nan, 2.0, 2.0, np.nan, 2.0],
            'c': [2.0, 2.0, 2.0, 2.0, 2.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumprod_skipna_false(self):
        """Test cumprod with skipna=False."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.5, np.nan, 1.5, 1.5, 1.5],
            'b': [1.5, 1.5, 1.5, 1.5, 1.5]
        })

        unlockedpd.config.enabled = False
        expected = df.cumprod(skipna=False)

        unlockedpd.config.enabled = True
        result = df.cumprod(skipna=False)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumprod_with_zeros(self):
        """Test cumprod with zero values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [2.0, 0.0, 3.0, 2.0, 2.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumprod_negative_values(self):
        """Test cumprod with negative values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [2.0, -1.0, 2.0, -1.0, 2.0],
            'b': [1.5, 1.5, 1.5, 1.5, 1.5]
        })

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestCumMin:
    """Tests for cummin()"""

    def test_basic_cummin(self):
        """Test basic cummin matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.cummin()

        unlockedpd.config.enabled = True
        result = df.cummin()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummin_with_nan(self):
        """Test cummin handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [5.0, np.nan, 3.0, 4.0, 1.0],
            'b': [np.nan, 5.0, 3.0, np.nan, 1.0],
            'c': [5.0, 4.0, 3.0, 2.0, 1.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummin()

        unlockedpd.config.enabled = True
        result = df.cummin()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummin_skipna_false(self):
        """Test cummin with skipna=False."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [5.0, np.nan, 3.0, 4.0, 1.0],
            'b': [5.0, 4.0, 3.0, 2.0, 1.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummin(skipna=False)

        unlockedpd.config.enabled = True
        result = df.cummin(skipna=False)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummin_ascending_values(self):
        """Test cummin with ascending values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummin()

        unlockedpd.config.enabled = True
        result = df.cummin()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummin_descending_values(self):
        """Test cummin with descending values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [5.0, 4.0, 3.0, 2.0, 1.0],
            'b': [10.0, 8.0, 6.0, 4.0, 2.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummin()

        unlockedpd.config.enabled = True
        result = df.cummin()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestCumMax:
    """Tests for cummax()"""

    def test_basic_cummax(self):
        """Test basic cummax matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.cummax()

        unlockedpd.config.enabled = True
        result = df.cummax()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummax_with_nan(self):
        """Test cummax handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 2.0, 5.0],
            'b': [np.nan, 1.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummax()

        unlockedpd.config.enabled = True
        result = df.cummax()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummax_skipna_false(self):
        """Test cummax with skipna=False."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 2.0, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummax(skipna=False)

        unlockedpd.config.enabled = True
        result = df.cummax(skipna=False)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummax_ascending_values(self):
        """Test cummax with ascending values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [2.0, 4.0, 6.0, 8.0, 10.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummax()

        unlockedpd.config.enabled = True
        result = df.cummax()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummax_descending_values(self):
        """Test cummax with descending values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [5.0, 4.0, 3.0, 2.0, 1.0],
            'b': [10.0, 8.0, 6.0, 4.0, 2.0]
        })

        unlockedpd.config.enabled = False
        expected = df.cummax()

        unlockedpd.config.enabled = True
        result = df.cummax()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestMixedDtypes:
    """Tests for mixed-dtype DataFrames."""

    def test_mixed_dtype_cumsum(self):
        """Test cumsum with mixed dtypes."""
        import unlockedpd

        df = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_mixed_dtype_cumprod(self):
        """Test cumprod with mixed dtypes."""
        import unlockedpd

        df = pd.DataFrame({
            'numeric1': [1.1, 1.2, 1.1, 1.2, 1.1],
            'numeric2': [1.5, 1.5, 1.5, 1.5, 1.5],
        })

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test cumulative operations on empty DataFrame."""
        import unlockedpd

        df = pd.DataFrame()

        unlockedpd.config.enabled = False
        try:
            expected = df.cumsum()
            expected_error = None
        except Exception as e:
            expected_error = type(e)

        unlockedpd.config.enabled = True
        if expected_error:
            with pytest.raises(expected_error):
                df.cumsum()
        else:
            result = df.cumsum()
            pd.testing.assert_frame_equal(result, expected)

    def test_single_row(self):
        """Test cumulative operations on single row."""
        import unlockedpd

        df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]])

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_very_large_dataframe(self):
        """Test cumulative operations on very large DataFrame."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(10000, 50))

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
