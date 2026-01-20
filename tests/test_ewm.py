"""Tests for exponentially weighted moving operations."""
import pytest
import pandas as pd
import numpy as np


class TestEWMMean:
    """Tests for ewm().mean()"""

    @pytest.mark.skip(reason="EWM operations not yet implemented")
    def test_basic_ewm_mean(self):
        """Test basic EWM mean matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="EWM operations not yet implemented")
    def test_ewm_mean_with_nan(self):
        """Test EWM mean handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, 5.0],
            'b': [np.nan, 2.0, 3.0, np.nan, 5.0]
        })

        unlockedpd.config.enabled = False
        expected = df.ewm(span=3).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(span=3).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="EWM operations not yet implemented")
    def test_ewm_mean_halflife(self):
        """Test EWM mean with halflife parameter."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.ewm(halflife=10).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(halflife=10).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="EWM operations not yet implemented")
    def test_ewm_mean_alpha(self):
        """Test EWM mean with alpha parameter."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.ewm(alpha=0.5).mean()

        unlockedpd.config.enabled = True
        result = df.ewm(alpha=0.5).mean()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestEWMStd:
    """Tests for ewm().std()"""

    @pytest.mark.skip(reason="EWM operations not yet implemented")
    def test_basic_ewm_std(self):
        """Test basic EWM std matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).std()

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).std()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestEWMVar:
    """Tests for ewm().var()"""

    @pytest.mark.skip(reason="EWM operations not yet implemented")
    def test_basic_ewm_var(self):
        """Test basic EWM var matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).var()

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).var()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestEWMCorr:
    """Tests for ewm().corr()"""

    def test_basic_ewm_corr_pairwise(self):
        """Test basic EWM correlation (pairwise) matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 5))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).corr(pairwise=True)

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).corr(pairwise=True)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-8, atol=1e-10)

    def test_ewm_corr_with_nan(self):
        """Test EWM corr handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0, 5.0],
            'b': [5.0, 4.0, 3.0, np.nan, 1.0],
            'c': [2.0, np.nan, 4.0, 5.0, 6.0]
        })

        unlockedpd.config.enabled = False
        expected = df.ewm(span=3).corr(pairwise=True)

        unlockedpd.config.enabled = True
        result = df.ewm(span=3).corr(pairwise=True)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-8, atol=1e-10)


class TestEWMCov:
    """Tests for ewm().cov()"""

    def test_basic_ewm_cov_pairwise(self):
        """Test basic EWM covariance (pairwise) matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 5))

        unlockedpd.config.enabled = False
        expected = df.ewm(span=10).cov(pairwise=True)

        unlockedpd.config.enabled = True
        result = df.ewm(span=10).cov(pairwise=True)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-8, atol=1e-10)

    def test_ewm_cov_with_nan(self):
        """Test EWM cov handles NaN correctly."""
        import unlockedpd

        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0, 5.0],
            'b': [5.0, 4.0, 3.0, np.nan, 1.0],
            'c': [2.0, np.nan, 4.0, 5.0, 6.0]
        })

        unlockedpd.config.enabled = False
        expected = df.ewm(span=3).cov(pairwise=True)

        unlockedpd.config.enabled = True
        result = df.ewm(span=3).cov(pairwise=True)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-8, atol=1e-10)

    def test_ewm_cov_halflife(self):
        """Test EWM cov with halflife parameter."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(50, 4))

        unlockedpd.config.enabled = False
        expected = df.ewm(halflife=5).cov(pairwise=True)

        unlockedpd.config.enabled = True
        result = df.ewm(halflife=5).cov(pairwise=True)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-8, atol=1e-10)
