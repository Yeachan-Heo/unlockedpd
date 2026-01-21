"""Benchmarks for element-wise operations."""
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/bellman/Workspace/unlockedpd-numerical-ops/src')

from unlockedpd.ops.element_wise import optimized_clip, optimized_abs, optimized_round


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
    print("Element-wise Benchmarks (10K x 1K DataFrame)")
    print("=" * 60)

    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(10000, 1000))

    results = {}
    results['clip'] = benchmark_operation('clip',
        lambda d, **kw: optimized_clip(d, lower=-1.0, upper=1.0),
        lambda **kw: df.clip(lower=-1.0, upper=1.0), df)
    results['abs'] = benchmark_operation('abs', optimized_abs, df.abs, df)
    results['round'] = benchmark_operation('round',
        lambda d, **kw: optimized_round(d, decimals=2),
        lambda **kw: df.round(decimals=2), df)

    print("-" * 60)
    avg_speedup = np.mean(list(results.values()))
    print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == '__main__':
    main()
