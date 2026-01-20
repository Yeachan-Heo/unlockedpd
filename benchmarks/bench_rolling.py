"""Benchmarks for rolling operations.

Run with: pytest benchmarks/ --benchmark-only
"""
import pytest
import pandas as pd
import numpy as np


# Test dimensions
SIZES = [
    (1_000, 10),
    (10_000, 100),
    (100_000, 100),
]
WINDOWS = [5, 20]


@pytest.fixture
def unlockedpd_module():
    """Import unlockedpd."""
    import unlockedpd
    return unlockedpd


class TestRollingMeanBenchmark:
    """Benchmarks for rolling mean."""

    @pytest.mark.parametrize("rows,cols", SIZES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_rolling_mean_optimized(self, benchmark, unlockedpd_module, rows, cols, window):
        """Benchmark optimized rolling mean."""
        df = pd.DataFrame(np.random.randn(rows, cols))
        unlockedpd_module.config.enabled = True

        result = benchmark(lambda: df.rolling(window).mean())
        assert result is not None

    @pytest.mark.parametrize("rows,cols", SIZES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_rolling_mean_pandas(self, benchmark, unlockedpd_module, rows, cols, window):
        """Benchmark vanilla pandas rolling mean."""
        df = pd.DataFrame(np.random.randn(rows, cols))
        unlockedpd_module.config.enabled = False

        result = benchmark(lambda: df.rolling(window).mean())
        assert result is not None


class TestRollingStdBenchmark:
    """Benchmarks for rolling std."""

    @pytest.mark.parametrize("rows,cols", [(10_000, 100)])
    @pytest.mark.parametrize("window", [20])
    def test_rolling_std_optimized(self, benchmark, unlockedpd_module, rows, cols, window):
        """Benchmark optimized rolling std."""
        df = pd.DataFrame(np.random.randn(rows, cols))
        unlockedpd_module.config.enabled = True

        result = benchmark(lambda: df.rolling(window).std())
        assert result is not None

    @pytest.mark.parametrize("rows,cols", [(10_000, 100)])
    @pytest.mark.parametrize("window", [20])
    def test_rolling_std_pandas(self, benchmark, unlockedpd_module, rows, cols, window):
        """Benchmark vanilla pandas rolling std."""
        df = pd.DataFrame(np.random.randn(rows, cols))
        unlockedpd_module.config.enabled = False

        result = benchmark(lambda: df.rolling(window).std())
        assert result is not None


class TestRankBenchmark:
    """Benchmarks for rank operations."""

    @pytest.mark.parametrize("rows,cols", SIZES)
    def test_rank_axis1_optimized(self, benchmark, unlockedpd_module, rows, cols):
        """Benchmark optimized rank axis=1."""
        df = pd.DataFrame(np.random.randn(rows, cols))
        unlockedpd_module.config.enabled = True

        result = benchmark(lambda: df.rank(axis=1))
        assert result is not None

    @pytest.mark.parametrize("rows,cols", SIZES)
    def test_rank_axis1_pandas(self, benchmark, unlockedpd_module, rows, cols):
        """Benchmark vanilla pandas rank axis=1."""
        df = pd.DataFrame(np.random.randn(rows, cols))
        unlockedpd_module.config.enabled = False

        result = benchmark(lambda: df.rank(axis=1))
        assert result is not None
