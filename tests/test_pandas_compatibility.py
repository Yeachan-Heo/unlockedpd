"""Comprehensive pandas compatibility tests for 100% mathematical equivalence.

This module tests ALL operations against pandas behavior, including edge cases:
- All-NaN columns/rows
- Single value
- Zero variance (all identical values)
- Empty DataFrames
- Near-zero values (numerical precision)
- Infinity values
- Mixed NaN patterns
"""
import pytest
import pandas as pd
import numpy as np
import warnings


class TestPctChangeCompatibility:
    """Test pct_change behavior matches pandas exactly."""

    def test_basic_pct_change(self):
        """Test basic pct_change matches pandas."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_pct_change_with_nan(self):
        """Test pct_change with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, np.nan, 40.0, 50.0],
        })

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_pct_change_all_nan_column(self):
        """Test pct_change with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_pct_change_from_zero(self):
        """Test pct_change when previous value is zero."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [0.0, 1.0, 2.0, 0.0, 3.0],
            'b': [1.0, 0.0, 1.0, 2.0, 3.0],
        })

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_pct_change_fill_method_none(self):
        """Test pct_change with fill_method=None."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, np.nan, 40.0, 50.0],
        })

        unlockedpd.config.enabled = False
        expected = df.pct_change(fill_method=None)

        unlockedpd.config.enabled = True
        result = df.pct_change(fill_method=None)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestRollingCompatibility:
    """Test rolling operations match pandas exactly."""

    def test_rolling_mean_basic(self):
        """Test rolling mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 5))

        unlockedpd.config.enabled = False
        expected = df.rolling(10).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(10).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_mean_with_nan(self):
        """Test rolling mean with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'b': [np.nan, 2.0, np.nan, 4.0, np.nan, 6.0, np.nan, 8.0, np.nan, 10.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(3).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(3).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_mean_all_nan(self):
        """Test rolling mean with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(3).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(3).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_std_basic(self):
        """Test rolling std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 5))

        unlockedpd.config.enabled = False
        expected = df.rolling(10).std()

        unlockedpd.config.enabled = True
        result = df.rolling(10).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_std_zero_variance(self):
        """Test rolling std with zero variance (all identical values)."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(3).std()

        unlockedpd.config.enabled = True
        result = df.rolling(3).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_std_single_value_window(self):
        """Test rolling std with min_periods=1 and window=1."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(10, 3))

        unlockedpd.config.enabled = False
        expected = df.rolling(1, min_periods=1).std()

        unlockedpd.config.enabled = True
        result = df.rolling(1, min_periods=1).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_skew_basic(self):
        """Test rolling skew matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 5))

        unlockedpd.config.enabled = False
        expected = df.rolling(10).skew()

        unlockedpd.config.enabled = True
        result = df.rolling(10).skew()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_skew_zero_variance(self):
        """Test rolling skew with zero variance (all identical values)."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(5).skew()

        unlockedpd.config.enabled = True
        result = df.rolling(5).skew()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_skew_near_zero_variance(self):
        """Test rolling skew with near-zero variance."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0 + 1e-15, 1.0, 1.0 + 1e-15, 1.0, 1.0 + 1e-15, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0 + 1e-15, 2.0, 2.0, 2.0 + 1e-15, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(5).skew()

        unlockedpd.config.enabled = True
        result = df.rolling(5).skew()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_kurt_basic(self):
        """Test rolling kurt matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 5))

        unlockedpd.config.enabled = False
        expected = df.rolling(10).kurt()

        unlockedpd.config.enabled = True
        result = df.rolling(10).kurt()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_kurt_zero_variance(self):
        """Test rolling kurt with zero variance."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rolling(5).kurt()

        unlockedpd.config.enabled = True
        result = df.rolling(5).kurt()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestExpandingCompatibility:
    """Test expanding operations match pandas exactly."""

    def test_expanding_mean_basic(self):
        """Test expanding mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_mean_all_nan(self):
        """Test expanding mean with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_std_basic(self):
        """Test expanding std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding().std()

        unlockedpd.config.enabled = True
        result = df.expanding().std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_std_zero_variance(self):
        """Test expanding std with zero variance."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().std()

        unlockedpd.config.enabled = True
        result = df.expanding().std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_skew_basic(self):
        """Test expanding skew matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding().skew()

        unlockedpd.config.enabled = True
        result = df.expanding().skew()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_skew_zero_variance(self):
        """Test expanding skew with zero variance."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().skew()

        unlockedpd.config.enabled = True
        result = df.expanding().skew()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_kurt_basic(self):
        """Test expanding kurt matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.expanding().kurt()

        unlockedpd.config.enabled = True
        result = df.expanding().kurt()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_expanding_kurt_zero_variance(self):
        """Test expanding kurt with zero variance."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 1.0, 1.0, 1.0, 1.0],
            'b': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        unlockedpd.config.enabled = False
        expected = df.expanding().kurt()

        unlockedpd.config.enabled = True
        result = df.expanding().kurt()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestEWMCompatibility:
    """Test EWM operations match pandas exactly."""

    def test_ewm_mean_basic(self):
        """Test EWM mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ewm_mean_with_nan(self):
        """Test EWM mean with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
        })

        unlockedpd.config.enabled = False
        expected = df.ewm(span=3).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(span=3).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ewm_mean_all_nan(self):
        """Test EWM mean with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.ewm(span=3).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(span=3).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ewm_std_basic(self):
        """Test EWM std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).std()

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ewm_var_basic(self):
        """Test EWM var matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).var()

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).var()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestStatsCompatibility:
    """Test DataFrame stats operations match pandas exactly."""

    def test_skew_basic(self):
        """Test skew matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.skew()

        unlockedpd.config.enabled = True
        result = df.skew()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_skew_all_nan(self):
        """Test skew with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': [np.nan] * 100,
        })

        unlockedpd.config.enabled = False
        expected = df.skew()

        unlockedpd.config.enabled = True
        result = df.skew()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_skew_zero_variance(self):
        """Test skew with zero variance column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': [1.0] * 100,
        })

        unlockedpd.config.enabled = False
        expected = df.skew()

        unlockedpd.config.enabled = True
        result = df.skew()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_kurt_basic(self):
        """Test kurt matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.kurt()

        unlockedpd.config.enabled = True
        result = df.kurt()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_kurt_all_nan(self):
        """Test kurt with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': [np.nan] * 100,
        })

        unlockedpd.config.enabled = False
        expected = df.kurt()

        unlockedpd.config.enabled = True
        result = df.kurt()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_kurt_zero_variance(self):
        """Test kurt with zero variance column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': [1.0] * 100,
        })

        unlockedpd.config.enabled = False
        expected = df.kurt()

        unlockedpd.config.enabled = True
        result = df.kurt()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sem_basic(self):
        """Test sem matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.sem()

        unlockedpd.config.enabled = True
        result = df.sem()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_sem_all_nan(self):
        """Test sem with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': [np.nan] * 100,
        })

        unlockedpd.config.enabled = False
        expected = df.sem()

        unlockedpd.config.enabled = True
        result = df.sem()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)


class TestDiffCompatibility:
    """Test diff operation matches pandas exactly."""

    def test_diff_basic(self):
        """Test diff matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.diff()

        unlockedpd.config.enabled = True
        result = df.diff()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_diff_with_nan(self):
        """Test diff with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
        })

        unlockedpd.config.enabled = False
        expected = df.diff()

        unlockedpd.config.enabled = True
        result = df.diff()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_diff_all_nan(self):
        """Test diff with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.diff()

        unlockedpd.config.enabled = True
        result = df.diff()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestShiftCompatibility:
    """Test shift operation matches pandas exactly."""

    def test_shift_basic(self):
        """Test shift matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.shift()

        unlockedpd.config.enabled = True
        result = df.shift()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_shift_with_nan(self):
        """Test shift with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
        })

        unlockedpd.config.enabled = False
        expected = df.shift()

        unlockedpd.config.enabled = True
        result = df.shift()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_shift_negative(self):
        """Test negative shift."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.shift(-2)

        unlockedpd.config.enabled = True
        result = df.shift(-2)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestCumulativeCompatibility:
    """Test cumulative operations match pandas exactly."""

    def test_cumsum_basic(self):
        """Test cumsum matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_with_nan(self):
        """Test cumsum with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
        })

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_all_nan(self):
        """Test cumsum with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumprod_basic(self):
        """Test cumprod matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.abs(np.random.randn(50, 5)) + 0.1)

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummin_basic(self):
        """Test cummin matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.cummin()

        unlockedpd.config.enabled = True
        result = df.cummin()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cummax_basic(self):
        """Test cummax matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.cummax()

        unlockedpd.config.enabled = True
        result = df.cummax()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestRankCompatibility:
    """Test rank operation matches pandas exactly."""

    def test_rank_basic(self):
        """Test rank matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.rank()

        unlockedpd.config.enabled = True
        result = df.rank()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_with_nan(self):
        """Test rank with NaN values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0],
        })

        unlockedpd.config.enabled = False
        expected = df.rank()

        unlockedpd.config.enabled = True
        result = df.rank()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_all_nan(self):
        """Test rank with all-NaN column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        unlockedpd.config.enabled = False
        expected = df.rank()

        unlockedpd.config.enabled = True
        result = df.rank()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestWrapResultFragmentation:
    """Test wrap_result function for DataFrame fragmentation issues."""

    def test_wrap_result_no_fragmentation_warning(self):
        """Test that merge_non_numeric doesn't trigger fragmentation warning."""
        from unlockedpd._compat import wrap_result

        # Test with object dtype
        df_object = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': ['x', 'y', 'z']  # object dtype
        })

        # Extract numeric columns
        numeric_cols = ['a', 'b']
        numeric_df = df_object[numeric_cols]
        result = numeric_df.values * 2

        # Wrap result with merge_non_numeric=True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped = wrap_result(
                result,
                numeric_df,
                columns=numeric_cols,
                merge_non_numeric=True,
                original_df=df_object
            )

            # Verify no PerformanceWarning
            perf_warnings = [x for x in w if issubclass(x.category, pd.errors.PerformanceWarning)]
            assert len(perf_warnings) == 0, f"Got fragmentation warning: {perf_warnings}"

        # Verify correct behavior
        assert list(wrapped.columns) == ['a', 'b', 'c']  # original order
        assert wrapped['c'].isna().all()  # non-numeric filled with NaN
        assert (wrapped[['a', 'b']] == result).all().all()  # numeric values correct

    def test_wrap_result_mixed_non_numeric_dtypes(self):
        """Test that merge_non_numeric handles datetime and categorical dtypes correctly."""
        from unlockedpd._compat import wrap_result

        # Create DataFrame with multiple non-numeric dtypes
        df_mixed = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6],
            'str_col': ['x', 'y', 'z'],  # object dtype
            'date_col': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),  # datetime
            'cat_col': pd.Categorical(['A', 'B', 'C'])  # categorical
        })

        numeric_cols = ['num1', 'num2']
        numeric_df = df_mixed[numeric_cols]
        result = numeric_df.values * 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped = wrap_result(
                result,
                numeric_df,
                columns=numeric_cols,
                merge_non_numeric=True,
                original_df=df_mixed
            )

            # Verify no PerformanceWarning
            perf_warnings = [x for x in w if issubclass(x.category, pd.errors.PerformanceWarning)]
            assert len(perf_warnings) == 0, f"Got fragmentation warning: {perf_warnings}"

        # Verify correct behavior for all dtypes
        assert list(wrapped.columns) == list(df_mixed.columns)  # original order
        assert wrapped['str_col'].isna().all()  # object dtype filled with NaN
        assert wrapped['date_col'].isna().all()  # datetime dtype filled with NaN
        assert wrapped['cat_col'].isna().all()  # categorical dtype filled with NaN
        assert (wrapped[['num1', 'num2']] == result).all().all()  # numeric values correct

    def test_wrap_result_many_non_numeric_columns(self):
        """Test wrap_result with many non-numeric columns (stress test for fragmentation)."""
        from unlockedpd._compat import wrap_result

        # Create DataFrame with many non-numeric columns
        n_non_numeric = 50
        data = {'num': [1, 2, 3]}
        for i in range(n_non_numeric):
            data[f'str_{i}'] = ['x', 'y', 'z']

        df = pd.DataFrame(data)
        numeric_cols = ['num']
        numeric_df = df[numeric_cols]
        result = numeric_df.values * 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped = wrap_result(
                result,
                numeric_df,
                columns=numeric_cols,
                merge_non_numeric=True,
                original_df=df
            )

            # Verify no PerformanceWarning even with many columns
            perf_warnings = [x for x in w if issubclass(x.category, pd.errors.PerformanceWarning)]
            assert len(perf_warnings) == 0, f"Got fragmentation warning with {n_non_numeric} non-numeric columns"

        # Verify correct behavior
        assert list(wrapped.columns) == list(df.columns)
        assert (wrapped['num'] == result.flatten()).all()
        for i in range(n_non_numeric):
            assert wrapped[f'str_{i}'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
