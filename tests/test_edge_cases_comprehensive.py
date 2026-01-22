"""Comprehensive edge case tests for pandas compatibility.

Tests verify unlockedpd matches pandas exactly for edge cases including:
- inf/-inf handling in all operations
- Empty and minimal DataFrames (0x0, 1x1, etc.)
- NaN patterns and skipna behavior
- Numerical precision edge cases
- Division edge cases in pct_change
- Window edge cases in rolling/expanding
"""
import pytest
import pandas as pd
import numpy as np
import unlockedpd


# ============== INF HANDLING TESTS ==============

INF_TEST_DATA = [
    pytest.param([np.inf, 1.0, 2.0], id='single_inf'),
    pytest.param([-np.inf, 1.0, 2.0], id='single_neg_inf'),
    pytest.param([np.inf, -np.inf, 1.0], id='mixed_inf'),
    pytest.param([np.inf, np.inf, np.inf], id='all_inf'),
]

AGG_FUNCS = ['sum', 'mean', 'std', 'var', 'min', 'max', 'median', 'prod']


class TestInfHandling:
    """Test inf/-inf handling matches pandas exactly."""

    @pytest.mark.parametrize('agg_name', AGG_FUNCS)
    @pytest.mark.parametrize('values', INF_TEST_DATA)
    def test_aggregation_inf(self, agg_name, values):
        """Test aggregation with inf values."""
        df = pd.DataFrame({'a': values, 'b': [1.0, 2.0, 3.0]})

        # Get pandas result with unlockedpd disabled
        unlockedpd.config.enabled = False
        expected = getattr(df, agg_name)()

        # Get unlockedpd result
        unlockedpd.config.enabled = True
        result = getattr(df, agg_name)()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    @pytest.mark.parametrize('values', INF_TEST_DATA)
    def test_rolling_mean_inf(self, values):
        """Test rolling mean with inf values."""
        df = pd.DataFrame({'a': values, 'b': [1.0, 2.0, 3.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(2).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(2).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.parametrize('values', INF_TEST_DATA)
    def test_expanding_mean_inf(self, values):
        """Test expanding mean with inf values."""
        df = pd.DataFrame({'a': values, 'b': [1.0, 2.0, 3.0]})

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cumsum_inf(self):
        """Test cumsum with inf values."""
        df = pd.DataFrame({'a': [1.0, np.inf, 2.0], 'b': [3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected)

    def test_cumsum_inf_minus_inf(self):
        """Test cumsum with inf and -inf (results in NaN)."""
        df = pd.DataFrame({'a': [1.0, np.inf, -np.inf], 'b': [3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected)


# ============== EMPTY DATAFRAME TESTS ==============

class TestEmptyDataFrames:
    """Test empty DataFrame handling matches pandas."""

    @pytest.mark.parametrize('agg_name', AGG_FUNCS)
    def test_zero_rows(self, agg_name):
        """Test aggregation on 0 rows, N columns."""
        df = pd.DataFrame({'a': [], 'b': []}).astype(float)

        unlockedpd.config.enabled = False
        expected = getattr(df, agg_name)()

        unlockedpd.config.enabled = True
        result = getattr(df, agg_name)()

        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize('agg_name', AGG_FUNCS)
    def test_zero_by_zero(self, agg_name):
        """Test aggregation on 0x0 DataFrame."""
        df = pd.DataFrame()

        unlockedpd.config.enabled = False
        expected = getattr(df, agg_name)()

        unlockedpd.config.enabled = True
        result = getattr(df, agg_name)()

        pd.testing.assert_series_equal(result, expected)

    def test_single_element(self):
        """Test 1x1 DataFrame."""
        df = pd.DataFrame({'a': [42.0]})

        for agg_name in AGG_FUNCS:
            unlockedpd.config.enabled = False
            expected = getattr(df, agg_name)()

            unlockedpd.config.enabled = True
            result = getattr(df, agg_name)()

            pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_single_row(self):
        """Test 1xN DataFrame."""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})

        for agg_name in ['sum', 'mean', 'min', 'max']:
            unlockedpd.config.enabled = False
            expected = getattr(df, agg_name)()

            unlockedpd.config.enabled = True
            result = getattr(df, agg_name)()

            pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_single_column(self):
        """Test Nx1 DataFrame."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        for agg_name in AGG_FUNCS:
            unlockedpd.config.enabled = False
            expected = getattr(df, agg_name)()

            unlockedpd.config.enabled = True
            result = getattr(df, agg_name)()

            pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_rolling_empty_df(self):
        """Test rolling operations on empty DataFrame."""
        df = pd.DataFrame({'a': [], 'b': []}).astype(float)

        unlockedpd.config.enabled = False
        expected = df.rolling(2).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(2).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_expanding_empty_df(self):
        """Test expanding operations on empty DataFrame."""
        df = pd.DataFrame({'a': [], 'b': []}).astype(float)

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected)


# ============== NAN PATTERN TESTS ==============

class TestNaNPatterns:
    """Test NaN handling matches pandas."""

    @pytest.mark.parametrize('agg_name', AGG_FUNCS)
    def test_all_nan(self, agg_name):
        """Test all-NaN DataFrame."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [np.nan, np.nan, np.nan]})

        unlockedpd.config.enabled = False
        expected = getattr(df, agg_name)()

        unlockedpd.config.enabled = True
        result = getattr(df, agg_name)()

        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize('agg_name', ['sum', 'mean', 'min', 'max'])
    def test_skipna_false(self, agg_name):
        """Test skipna=False behavior."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})

        unlockedpd.config.enabled = False
        expected = getattr(df, agg_name)(skipna=False)

        unlockedpd.config.enabled = True
        result = getattr(df, agg_name)(skipna=False)

        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize('agg_name', ['sum', 'mean', 'min', 'max'])
    def test_skipna_true(self, agg_name):
        """Test skipna=True behavior (default)."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})

        unlockedpd.config.enabled = False
        expected = getattr(df, agg_name)(skipna=True)

        unlockedpd.config.enabled = True
        result = getattr(df, agg_name)(skipna=True)

        pd.testing.assert_series_equal(result, expected)

    def test_alternating_nan(self):
        """Test alternating NaN pattern."""
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, np.nan],
        })

        for agg_name in ['sum', 'mean', 'min', 'max']:
            unlockedpd.config.enabled = False
            expected = getattr(df, agg_name)()

            unlockedpd.config.enabled = True
            result = getattr(df, agg_name)()

            pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_leading_nan(self):
        """Test leading NaN values."""
        df = pd.DataFrame({'a': [np.nan, np.nan, 1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0, 7.0, 8.0]})

        for agg_name in AGG_FUNCS:
            unlockedpd.config.enabled = False
            expected = getattr(df, agg_name)()

            unlockedpd.config.enabled = True
            result = getattr(df, agg_name)()

            pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_trailing_nan(self):
        """Test trailing NaN values."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, np.nan, np.nan], 'b': [4.0, 5.0, 6.0, 7.0, 8.0]})

        for agg_name in AGG_FUNCS:
            unlockedpd.config.enabled = False
            expected = getattr(df, agg_name)()

            unlockedpd.config.enabled = True
            result = getattr(df, agg_name)()

            pd.testing.assert_series_equal(result, expected, rtol=1e-10)


# ============== ROLLING EDGE CASES ==============

class TestRollingEdgeCases:
    """Test rolling operation edge cases."""

    def test_window_larger_than_data(self):
        """Test rolling with window > len(data)."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(10).sum()

        unlockedpd.config.enabled = True
        result = df.rolling(10).sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_window_size_one(self):
        """Test rolling with window=1."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(1).sum()

        unlockedpd.config.enabled = True
        result = df.rolling(1).sum()

        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op', ['sum', 'mean', 'std', 'var', 'min', 'max'])
    def test_rolling_with_inf(self, op):
        """Test rolling operations with inf values."""
        df = pd.DataFrame({'a': [1.0, np.inf, 2.0, 3.0, 4.0]})

        unlockedpd.config.enabled = False
        expected = getattr(df.rolling(3), op)()

        unlockedpd.config.enabled = True
        result = getattr(df.rolling(3), op)()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_min_periods_edge(self):
        """Test rolling with min_periods edge cases."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        # min_periods = window should result in NaN for first few rows
        unlockedpd.config.enabled = False
        expected = df.rolling(5, min_periods=5).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(5, min_periods=5).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_min_periods_one(self):
        """Test rolling with min_periods=1."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(3, min_periods=1).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(3, min_periods=1).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_all_nan(self):
        """Test rolling on all-NaN data."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan, np.nan, np.nan]})

        unlockedpd.config.enabled = False
        expected = df.rolling(2).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(2).mean()

        pd.testing.assert_frame_equal(result, expected)


# ============== EXPANDING EDGE CASES ==============

class TestExpandingEdgeCases:
    """Test expanding operation edge cases."""

    def test_expanding_single_value(self):
        """Test expanding with single value."""
        df = pd.DataFrame({'a': [42.0]})

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_expanding_with_inf(self):
        """Test expanding with inf values."""
        df = pd.DataFrame({'a': [1.0, 2.0, np.inf, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_expanding_all_nan(self):
        """Test expanding on all-NaN data."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})

        unlockedpd.config.enabled = False
        expected = df.expanding().mean()

        unlockedpd.config.enabled = True
        result = df.expanding().mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_expanding_min_periods_edge(self):
        """Test expanding with min_periods > data length."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})

        unlockedpd.config.enabled = False
        expected = df.expanding(min_periods=5).mean()

        unlockedpd.config.enabled = True
        result = df.expanding(min_periods=5).mean()

        pd.testing.assert_frame_equal(result, expected)


# ============== CUMULATIVE EDGE CASES ==============

class TestCumulativeEdgeCases:
    """Test cumulative operation edge cases."""

    def test_cumprod_with_zero(self):
        """Test cumprod with zero (everything after becomes zero)."""
        df = pd.DataFrame({'a': [1.0, 2.0, 0.0, 3.0, 4.0]})

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected)

    def test_cumprod_with_inf(self):
        """Test cumprod with inf values."""
        df = pd.DataFrame({'a': [1.0, 2.0, np.inf]})

        unlockedpd.config.enabled = False
        expected = df.cumprod()

        unlockedpd.config.enabled = True
        result = df.cumprod()

        pd.testing.assert_frame_equal(result, expected)

    def test_cumsum_empty_df(self):
        """Test cumsum on empty DataFrame."""
        df = pd.DataFrame({'a': []}).astype(float)

        unlockedpd.config.enabled = False
        expected = df.cumsum()

        unlockedpd.config.enabled = True
        result = df.cumsum()

        pd.testing.assert_frame_equal(result, expected)

    def test_cummax_all_nan(self):
        """Test cummax on all-NaN data."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})

        unlockedpd.config.enabled = False
        expected = df.cummax()

        unlockedpd.config.enabled = True
        result = df.cummax()

        pd.testing.assert_frame_equal(result, expected)

    def test_cummin_all_nan(self):
        """Test cummin on all-NaN data."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})

        unlockedpd.config.enabled = False
        expected = df.cummin()

        unlockedpd.config.enabled = True
        result = df.cummin()

        pd.testing.assert_frame_equal(result, expected)


# ============== PCT_CHANGE EDGE CASES ==============

class TestPctChangeEdgeCases:
    """Test pct_change edge cases."""

    def test_division_by_zero(self):
        """Test pct_change with division by zero (0 -> non-zero = inf)."""
        df = pd.DataFrame({'a': [0.0, 1.0, 0.0]})

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected)

    def test_zero_to_zero(self):
        """Test pct_change 0 -> 0 (0/0 = NaN)."""
        df = pd.DataFrame({'a': [0.0, 0.0, 0.0]})

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected)

    def test_inf_pct_change(self):
        """Test pct_change with inf values."""
        df = pd.DataFrame({'a': [np.inf, 1.0, np.inf]})

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected)

    def test_pct_change_inf_to_inf(self):
        """Test pct_change inf -> inf (inf/inf = NaN)."""
        df = pd.DataFrame({'a': [np.inf, np.inf, np.inf]})

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected)

    def test_pct_change_single_row(self):
        """Test pct_change with single row (first row is always NaN)."""
        df = pd.DataFrame({'a': [42.0]})

        unlockedpd.config.enabled = False
        expected = df.pct_change()

        unlockedpd.config.enabled = True
        result = df.pct_change()

        pd.testing.assert_frame_equal(result, expected)


# ============== NUMERICAL PRECISION TESTS ==============

class TestNumericalPrecision:
    """Test numerical precision edge cases."""

    def test_near_max_float64(self):
        """Test values near float64 max."""
        max_val = np.finfo(np.float64).max
        df = pd.DataFrame({'a': [max_val * 0.9, max_val * 0.95, max_val * 0.99]})

        unlockedpd.config.enabled = False
        expected = df.sum()

        unlockedpd.config.enabled = True
        result = df.sum()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_near_min_float64(self):
        """Test values near float64 min (most negative)."""
        min_val = np.finfo(np.float64).min
        df = pd.DataFrame({'a': [min_val * 0.9, min_val * 0.95, min_val * 0.99]})

        unlockedpd.config.enabled = False
        expected = df.sum()

        unlockedpd.config.enabled = True
        result = df.sum()

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_subnormal_numbers(self):
        """Test subnormal/denormalized numbers."""
        subnormal = np.finfo(np.float64).tiny / 2
        df = pd.DataFrame({'a': [subnormal, subnormal * 2, subnormal * 3]})

        unlockedpd.config.enabled = False
        expected = df.sum()

        unlockedpd.config.enabled = True
        result = df.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_catastrophic_cancellation(self):
        """Test catastrophic cancellation (large - nearly_equal_large)."""
        large = 1e16
        df = pd.DataFrame({'a': [large, -large + 1, 1.0]})

        unlockedpd.config.enabled = False
        expected = df.sum()

        unlockedpd.config.enabled = True
        result = df.sum()

        # Use looser tolerance due to floating point precision limits
        pd.testing.assert_series_equal(result, expected, rtol=1e-8)

    def test_tiny_variance(self):
        """Test variance/std with very small differences."""
        df = pd.DataFrame({'a': [1.0, 1.0 + 1e-15, 1.0 + 2e-15]})

        unlockedpd.config.enabled = False
        expected = df.std()

        unlockedpd.config.enabled = True
        result = df.std()

        pd.testing.assert_series_equal(result, expected, rtol=1e-5)
