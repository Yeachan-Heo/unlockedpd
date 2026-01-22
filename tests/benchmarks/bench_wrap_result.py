"""Performance benchmark for wrap_result fragmentation fix.

This benchmark measures the performance of wrap_result() when merging
non-numeric columns back into numeric operation results.

The fix replaced iterative column assignment (which caused DataFrame
fragmentation) with batch pd.concat() operation.

Run with: pytest tests/benchmarks/bench_wrap_result.py -v
"""
import pytest
import pandas as pd
import numpy as np

from unlockedpd._compat import wrap_result


@pytest.fixture
def mixed_dtype_df_small():
    """Small mixed-dtype DataFrame: 1000 rows, 10 numeric, 10 non-numeric."""
    np.random.seed(42)
    data = {}
    for i in range(10):
        data[f'num_{i}'] = np.random.randn(1000)
    for i in range(10):
        data[f'str_{i}'] = ['x'] * 1000
    return pd.DataFrame(data)


@pytest.fixture
def mixed_dtype_df_medium():
    """Medium mixed-dtype DataFrame: 10000 rows, 10 numeric, 50 non-numeric."""
    np.random.seed(42)
    data = {}
    for i in range(10):
        data[f'num_{i}'] = np.random.randn(10000)
    for i in range(50):
        data[f'str_{i}'] = ['x'] * 10000
    return pd.DataFrame(data)


@pytest.fixture
def mixed_dtype_df_large():
    """Large mixed-dtype DataFrame: 10000 rows, 10 numeric, 100 non-numeric."""
    np.random.seed(42)
    data = {}
    for i in range(10):
        data[f'num_{i}'] = np.random.randn(10000)
    for i in range(100):
        data[f'str_{i}'] = ['x'] * 10000
    return pd.DataFrame(data)


def _wrap_result_helper(df, n_numeric):
    """Helper to run wrap_result with merge_non_numeric."""
    numeric_cols = [f'num_{i}' for i in range(n_numeric)]
    numeric_df = df[numeric_cols]
    result = numeric_df.values
    return wrap_result(
        result,
        numeric_df,
        columns=numeric_cols,
        merge_non_numeric=True,
        original_df=df
    )


class TestWrapResultBenchmarks:
    """Benchmarks for wrap_result with merge_non_numeric=True.

    These benchmarks measure the performance improvement from using
    pd.concat() instead of iterative column assignment.

    Expected improvement: 10-100x faster for DataFrames with many
    non-numeric columns.
    """

    def test_bench_wrap_result_10_non_numeric(self, benchmark, mixed_dtype_df_small):
        """Benchmark wrap_result with 10 non-numeric columns."""
        benchmark(lambda: _wrap_result_helper(mixed_dtype_df_small, 10))

    def test_bench_wrap_result_50_non_numeric(self, benchmark, mixed_dtype_df_medium):
        """Benchmark wrap_result with 50 non-numeric columns."""
        benchmark(lambda: _wrap_result_helper(mixed_dtype_df_medium, 10))

    def test_bench_wrap_result_100_non_numeric(self, benchmark, mixed_dtype_df_large):
        """Benchmark wrap_result with 100 non-numeric columns."""
        benchmark(lambda: _wrap_result_helper(mixed_dtype_df_large, 10))


if __name__ == '__main__':
    # Quick manual benchmark without pytest-benchmark
    import time

    print("Manual benchmark for wrap_result performance")
    print("=" * 60)

    for n_non_numeric in [10, 50, 100]:
        np.random.seed(42)
        n_rows = 10000
        n_numeric = 10

        # Create mixed DataFrame
        data = {}
        for i in range(n_numeric):
            data[f'num_{i}'] = np.random.randn(n_rows)
        for i in range(n_non_numeric):
            data[f'str_{i}'] = ['x'] * n_rows

        df = pd.DataFrame(data)
        numeric_cols = [f'num_{i}' for i in range(n_numeric)]
        numeric_df = df[numeric_cols]
        result = numeric_df.values

        # Benchmark
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            wrapped = wrap_result(
                result,
                numeric_df,
                columns=numeric_cols,
                merge_non_numeric=True,
                original_df=df
            )
        elapsed = time.perf_counter() - start

        print(f"n_non_numeric={n_non_numeric:3d}: {elapsed:.3f}s for {iterations} iterations "
              f"({elapsed/iterations*1000:.2f}ms per call)")

    print("=" * 60)
    print("Benchmark complete")
