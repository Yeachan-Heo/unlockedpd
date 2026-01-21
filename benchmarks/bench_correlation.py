"""Benchmarks for correlation operations."""
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')

from unlockedpd.ops.correlation import optimized_corr, optimized_cov


def benchmark_operation(name, optimized_func, pandas_func, df, runs=5):
    """Benchmark optimized vs pandas."""
    # Warmup
    optimized_func(df)
    pandas_func()

    # Time optimized
    start = time.perf_counter()
    for _ in range(runs):
        optimized_func(df)
    opt_time = (time.perf_counter() - start) / runs

    # Time pandas
    start = time.perf_counter()
    for _ in range(runs):
        pandas_func()
    pd_time = (time.perf_counter() - start) / runs

    speedup = pd_time / opt_time
    print(f"{name:12s}: optimized={opt_time*1000:8.2f}ms, pandas={pd_time*1000:8.2f}ms, speedup={speedup:.2f}x")
    return speedup


def main():
    print("=" * 60)
    print("Correlation Benchmarks (1K x 100 DataFrame)")
    print("=" * 60)

    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(1000, 100))

    results = {}
    results['corr'] = benchmark_operation('corr', optimized_corr, df.corr, df)
    results['cov'] = benchmark_operation('cov', optimized_cov, df.cov, df)

    print("-" * 60)
    avg_speedup = np.mean(list(results.values()))
    print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == '__main__':
    main()
