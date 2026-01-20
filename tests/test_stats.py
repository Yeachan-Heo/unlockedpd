"""Tests for statistical operations."""
import pytest
import pandas as pd
import numpy as np


class TestRank:
    """Tests for rank() operation"""

    def test_basic_rank_axis0(self):
        """Test basic rank along axis 0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=0)

        unlockedpd.config.enabled = True
        result = df.rank(axis=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_basic_rank_axis1(self):
        """Test basic rank along axis 1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_with_nan(self):
        """Test rank handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'c': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank()

        unlockedpd.config.enabled = True
        result = df.rank()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_method_min(self):
        """Test rank with method='min'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 3.0, 4.0],
            'b': [1.0, 1.0, 3.0, 3.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(method='min', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(method='min', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_method_max(self):
        """Test rank with method='max'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 3.0, 4.0],
            'b': [1.0, 1.0, 3.0, 3.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(method='max', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(method='max', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_method_first(self):
        """Test rank with method='first'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 3.0, 4.0],
            'b': [1.0, 1.0, 3.0, 3.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(method='first', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(method='first', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_method_dense(self):
        """Test rank with method='dense'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, 2.0, 3.0, 4.0],
            'b': [1.0, 1.0, 3.0, 3.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(method='dense', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(method='dense', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_ascending_false(self):
        """Test rank with ascending=False."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 10))

        unlockedpd.config.enabled = False
        expected = df.rank(ascending=False, axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(ascending=False, axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_na_option_keep(self):
        """Test rank with na_option='keep'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(na_option='keep', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(na_option='keep', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_na_option_top(self):
        """Test rank with na_option='top'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(na_option='top', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(na_option='top', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_na_option_bottom(self):
        """Test rank with na_option='bottom'."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(na_option='bottom', axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(na_option='bottom', axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_all_equal_values(self):
        """Test rank with all equal values."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [2.0, 2.0, 2.0, 2.0, 2.0],
            'b': [3.0, 3.0, 3.0, 3.0, 3.0]
        })

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_single_column(self):
        """Test rank with single column DataFrame."""
        import unlockedpd

        df = pd.DataFrame({'a': np.random.randn(100)})

        unlockedpd.config.enabled = False
        expected = df.rank()

        unlockedpd.config.enabled = True
        result = df.rank()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_rank_large_dataframe(self):
        """Test rank on large DataFrame."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(1000, 100))

        unlockedpd.config.enabled = False
        expected = df.rank(axis=1)

        unlockedpd.config.enabled = True
        result = df.rank(axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestCorr:
    """Tests for corr() operation"""

    @pytest.mark.skip(reason="Correlation operations not yet implemented")
    def test_basic_corr(self):
        """Test basic correlation matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.corr()

        unlockedpd.config.enabled = True
        result = df.corr()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestCov:
    """Tests for cov() operation"""

    @pytest.mark.skip(reason="Covariance operations not yet implemented")
    def test_basic_cov(self):
        """Test basic covariance matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.cov()

        unlockedpd.config.enabled = True
        result = df.cov()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestQuantile:
    """Tests for quantile() operation"""

    @pytest.mark.skip(reason="Quantile operations not yet implemented")
    def test_basic_quantile(self):
        """Test basic quantile matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.quantile(0.5)

        unlockedpd.config.enabled = True
        result = df.quantile(0.5)

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Quantile operations not yet implemented")
    def test_quantile_multiple(self):
        """Test quantile with multiple quantiles."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.quantile([0.25, 0.5, 0.75])

        unlockedpd.config.enabled = True
        result = df.quantile([0.25, 0.5, 0.75])

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
