"""Benchmarks for aggregation operations."""
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')

from unlockedpd.ops.aggregations import (
    optimized_sum, optimized_mean, optimized_std, optimized_var,
    optimized_min, optimized_max, optimized_median, optimized_prod
)


def benchmark_operation(name, optimized_func, pandas_func, df, runs=5):
    """Benchmark optimized vs pandas."""
    # Warmup
    optimized_func(df, axis=0)
    pandas_func(axis=0)

    # Time optimized
    start = time.perf_counter()
    for _ in range(runs):
        optimized_func(df, axis=0)
    opt_time = (time.perf_counter() - start) / runs

    # Time pandas
    start = time.perf_counter()
    for _ in range(runs):
        pandas_func(axis=0)
    pd_time = (time.perf_counter() - start) / runs

    speedup = pd_time / opt_time
    print(f"{name:12s}: optimized={opt_time*1000:8.2f}ms, pandas={pd_time*1000:8.2f}ms, speedup={speedup:.2f}x")
    return speedup


def main():
    print("=" * 60)
    print("Aggregation Benchmarks (10K x 1K DataFrame)")
    print("=" * 60)

    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(10000, 1000))

    results = {}
    results['sum'] = benchmark_operation('sum', optimized_sum, df.sum, df)
    results['mean'] = benchmark_operation('mean', optimized_mean, df.mean, df)
    results['std'] = benchmark_operation('std', optimized_std, df.std, df)
    results['var'] = benchmark_operation('var', optimized_var, df.var, df)
    results['min'] = benchmark_operation('min', optimized_min, df.min, df)
    results['max'] = benchmark_operation('max', optimized_max, df.max, df)
    results['median'] = benchmark_operation('median', optimized_median, df.median, df)

    # Small values for prod
    df_small = pd.DataFrame(np.random.uniform(0.99, 1.01, (1000, 100)))
    results['prod'] = benchmark_operation('prod', optimized_prod, df_small.prod, df_small)

    print("-" * 60)
    avg_speedup = np.mean(list(results.values()))
    print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == '__main__':
    main()
