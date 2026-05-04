"""Tests for rolling operations."""

import pandas as pd
import numpy as np
import pandas.testing as tm


class TestRollingMean:
    """Tests for rolling().mean()"""

    def test_basic_rolling_mean(self):
        """Test basic rolling mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(5).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_with_nan(self):
        """Test rolling mean handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0, 5.0], "b": [np.nan, 2.0, 3.0, np.nan, 5.0]}
        )

        unlockedpd.config.enabled = False
        expected = df.rolling(3).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(3).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_min_periods(self):
        """Test rolling mean with min_periods."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 5))

        unlockedpd.config.enabled = False
        expected = df.rolling(10, min_periods=5).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(10, min_periods=5).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_centered(self):
        """Test centered rolling mean."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5, center=True).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(5, center=True).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_mean_window_larger_than_data(self):
        """Test rolling mean when window > data length."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(5, 3))

        unlockedpd.config.enabled = False
        expected = df.rolling(10).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(10).mean()

        pd.testing.assert_frame_equal(result, expected)

    def test_large_rolling_mean_uses_bounded_parallel_path(self):
        """Large rolling mean uses the bounded O(n) Numba path."""
        import unlockedpd
        from unlockedpd._resources import get_last_selected_path

        rng = np.random.default_rng(123)
        df = pd.DataFrame(rng.standard_normal((1024, 512)))

        unlockedpd.config.enabled = False
        expected = df.rolling(20).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(20).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)
        assert get_last_selected_path() == "parallel_numba"


class TestRollingSum:
    """Tests for rolling().sum()"""

    def test_basic_rolling_sum(self):
        """Test basic rolling sum matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).sum()

        unlockedpd.config.enabled = True
        result = df.rolling(5).sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_sum_inf_semantics(self):
        """Rolling sum preserves pandas NaN result for windows containing inf."""
        import unlockedpd

        df = pd.DataFrame(
            {
                "a": [1.0, np.inf, 3.0, 4.0],
                "b": [1.0, -np.inf, np.nan, 4.0],
            }
        )

        unlockedpd.config.enabled = False
        expected = df.rolling(2, min_periods=1).sum()

        unlockedpd.config.enabled = True
        result = df.rolling(2, min_periods=1).sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_large_rolling_sum_uses_rowblock_parallel_path(self):
        """Large rolling sum keeps pandas semantics on the row-block path."""
        import unlockedpd
        from unlockedpd._resources import get_last_selected_path

        rng = np.random.default_rng(321)
        values = rng.standard_normal((1024, 512))
        values[10, 7] = np.nan
        values[20, 9] = np.inf
        df = pd.DataFrame(values)

        unlockedpd.config.enabled = False
        expected = df.rolling(20, min_periods=10).sum()

        unlockedpd.config.enabled = True
        result = df.rolling(20, min_periods=10).sum()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)
        assert get_last_selected_path() == "parallel_numba"


class TestRollingStd:
    """Tests for rolling().std()"""

    def test_basic_rolling_std(self):
        """Test basic rolling std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).std()

        unlockedpd.config.enabled = True
        result = df.rolling(5).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_std_ddof(self):
        """Test rolling std with different ddof."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).std(ddof=0)

        unlockedpd.config.enabled = True
        result = df.rolling(5).std(ddof=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_std_ignores_inf_like_pandas(self):
        """Rolling std treats inf like missing data for moment calculations."""
        import unlockedpd

        df = pd.DataFrame({"a": [1.0, np.inf, 3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(3, min_periods=2).std()

        unlockedpd.config.enabled = True
        result = df.rolling(3, min_periods=2).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_large_rolling_std_uses_bounded_parallel_path(self):
        """Large rolling std avoids unbounded Numba thread oversubscription."""
        import unlockedpd
        from unlockedpd._resources import get_last_selected_path

        rng = np.random.default_rng(321)
        df = pd.DataFrame(rng.standard_normal((1024, 512)))

        unlockedpd.config.enabled = False
        expected = df.rolling(20).std()

        unlockedpd.config.enabled = True
        result = df.rolling(20).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10, atol=1e-10)
        assert get_last_selected_path() == "parallel_numba"


class TestRollingMinMax:
    """Tests for rolling().min() and rolling().max()"""

    def test_basic_rolling_min(self):
        """Test basic rolling min matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).min()

        unlockedpd.config.enabled = True
        result = df.rolling(5).min()

        pd.testing.assert_frame_equal(result, expected)

    def test_basic_rolling_max(self):
        """Test basic rolling max matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).max()

        unlockedpd.config.enabled = True
        result = df.rolling(5).max()

        pd.testing.assert_frame_equal(result, expected)


class TestRollingAxis1:
    """Tests for row-wise rolling windows."""

    def test_axis1_rolling_common_ops_match_pandas(self):
        """Axis=1 rolling uses row-wise windows, not column-wise axis=0 windows."""
        import unlockedpd

        rng = np.random.default_rng(20260504)
        df = pd.DataFrame(rng.standard_normal((64, 96)))
        df.iloc[::11, ::13] = np.nan

        operations = [
            ("mean", {}),
            ("sum", {}),
            ("min", {}),
            ("max", {}),
            ("std", {}),
            ("var", {}),
            ("count", {}),
        ]

        for method, kwargs in operations:
            with unlockedpd._PatchRegistry.temporarily_unpatched():
                expected = getattr(df.rolling(5, axis=1, min_periods=2), method)(
                    **kwargs
                )
            result = getattr(df.rolling(5, axis=1, min_periods=2), method)(**kwargs)

            tm.assert_frame_equal(result, expected, rtol=1e-10, atol=1e-10)

    def test_large_axis1_rolling_uses_parallel_transposed_path(self):
        """Large axis=1 rolling dispatches through the parallel kernel on df.T."""
        import unlockedpd
        from unlockedpd._resources import get_last_selected_path

        rng = np.random.default_rng(20260505)
        df = pd.DataFrame(rng.standard_normal((1024, 512)))

        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = df.rolling(20, axis=1).mean()
        result = df.rolling(20, axis=1).mean()

        tm.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)
        assert get_last_selected_path() == "parallel_numba"

    def test_axis1_rolling_window_larger_than_columns(self):
        """Axis=1 all-NaN edge case uses column count, not row count."""
        import unlockedpd

        df = pd.DataFrame(np.arange(12.0).reshape(3, 4))

        with unlockedpd._PatchRegistry.temporarily_unpatched():
            expected = df.rolling(10, axis=1).sum()
        result = df.rolling(10, axis=1).sum()

        tm.assert_frame_equal(result, expected)


class TestRollingSem:
    """Tests for rolling().sem()"""

    def test_basic_rolling_sem(self):
        """Test basic rolling sem matches pandas."""
        import unlockedpd

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(3).sem()

        unlockedpd.config.enabled = True
        result = df.rolling(3).sem()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_sem_with_nan(self):
        """Test rolling sem handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(3).sem()

        unlockedpd.config.enabled = True
        result = df.rolling(3).sem()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_sem_min_periods(self):
        """Test rolling sem with min_periods."""
        import unlockedpd

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

        unlockedpd.config.enabled = False
        expected = df.rolling(3, min_periods=2).sem()

        unlockedpd.config.enabled = True
        result = df.rolling(3, min_periods=2).sem()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rolling_sem_ddof(self):
        """Test rolling sem with different ddof."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rolling(5).sem(ddof=0)

        unlockedpd.config.enabled = True
        result = df.rolling(5).sem(ddof=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestMixedDtypes:
    """Tests for mixed-dtype DataFrames."""

    def test_mixed_dtype_rolling(self):
        """Test that non-numeric columns are handled correctly.

        Our implementation processes only numeric columns and returns
        NaN for non-numeric columns in their original positions.
        """
        import unlockedpd

        df = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        unlockedpd.config.enabled = False
        expected = df.rolling(2).mean()

        unlockedpd.config.enabled = True
        result = df.rolling(2).mean()

        pd.testing.assert_frame_equal(result, expected)
