"""Benchmarks for fill operations."""
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')

from unlockedpd.ops.fillna import optimized_ffill, optimized_bfill, optimized_fillna


def benchmark_operation(name, optimized_func, pandas_func, df, runs=5, **kwargs):
    """Benchmark optimized vs pandas."""
    # Warmup
    optimized_func(df, **kwargs)
    pandas_func(**kwargs)

    # Time optimized
    start = time.perf_counter()
    for _ in range(runs):
        optimized_func(df, **kwargs)
    opt_time = (time.perf_counter() - start) / runs

    # Time pandas
    start = time.perf_counter()
    for _ in range(runs):
        pandas_func(**kwargs)
    pd_time = (time.perf_counter() - start) / runs

    speedup = pd_time / opt_time
    print(f"{name:12s}: optimized={opt_time*1000:8.2f}ms, pandas={pd_time*1000:8.2f}ms, speedup={speedup:.2f}x")
    return speedup


def main():
    print("=" * 60)
    print("Fill Benchmarks (10K x 1K DataFrame with 30% NaN)")
    print("=" * 60)

    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(10000, 1000))
    df.iloc[::3, :] = np.nan  # 30% NaN

    results = {}
    results['ffill'] = benchmark_operation('ffill', optimized_ffill, df.ffill, df)
    results['bfill'] = benchmark_operation('bfill', optimized_bfill, df.bfill, df)
    results['fillna'] = benchmark_operation('fillna',
        lambda d, **kw: optimized_fillna(d, value=0.0),
        lambda **kw: df.fillna(0.0), df)

    print("-" * 60)
    avg_speedup = np.mean(list(results.values()))
    print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == '__main__':
    main()
