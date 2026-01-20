"""Comprehensive benchmarks for unlockedpd operations.

Benchmarks 2D arrays from 1MB to 10GB with different shapes.
Compares unlockedpd (Numba-parallel) vs vanilla pandas.

Run: python benchmarks/bench_comprehensive.py

Memory calculation:
- float64 = 8 bytes per element
- 1MB = 131,072 elements
- 10MB = 1,310,720 elements
- 100MB = 13,107,200 elements
- 1GB = 134,217,728 elements
- 10GB = 1,342,177,280 elements
"""
import time
import gc
import sys
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    operation: str
    shape: Tuple[int, int]
    size_mb: float
    pandas_time: float
    optimized_time: float
    speedup: float


def get_test_sizes() -> List[Tuple[Tuple[int, int], float]]:
    """Generate test sizes from 1MB to 10GB with different shapes.

    Returns list of ((rows, cols), size_mb) tuples.
    """
    sizes = []

    # 1MB tests - different aspect ratios
    sizes.append(((16384, 8), 1.0))       # Tall/narrow: 16K x 8
    sizes.append(((1024, 128), 1.0))      # Medium: 1K x 128
    sizes.append(((128, 1024), 1.0))      # Wide: 128 x 1K

    # 10MB tests
    sizes.append(((163840, 8), 10.0))     # Tall/narrow
    sizes.append(((10240, 128), 10.0))    # Medium
    sizes.append(((1280, 1024), 10.0))    # Wide

    # 100MB tests
    sizes.append(((1638400, 8), 100.0))   # Tall/narrow
    sizes.append(((102400, 128), 100.0))  # Medium
    sizes.append(((12800, 1024), 100.0))  # Wide

    # 1GB tests
    sizes.append(((16777216, 8), 1024.0))    # Tall/narrow: 16M x 8
    sizes.append(((1048576, 128), 1024.0))   # Medium: 1M x 128
    sizes.append(((131072, 1024), 1024.0))   # Wide: 128K x 1K

    # 5GB tests (if memory allows)
    sizes.append(((83886080, 8), 5120.0))    # Tall/narrow: 84M x 8
    sizes.append(((5242880, 128), 5120.0))   # Medium: 5M x 128
    sizes.append(((655360, 1024), 5120.0))   # Wide: 640K x 1K

    # 10GB tests (if memory allows)
    sizes.append(((167772160, 8), 10240.0))   # Tall/narrow: 168M x 8
    sizes.append(((10485760, 128), 10240.0))  # Medium: 10M x 128
    sizes.append(((1310720, 1024), 10240.0))  # Wide: 1.3M x 1K

    return sizes


def time_operation(df: pd.DataFrame, operation: Callable, warmup: int = 1, runs: int = 3) -> float:
    """Time an operation with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        _ = operation(df)

    # Timed runs
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        _ = operation(df)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)  # Return best time


def benchmark_operation(
    name: str,
    operation: Callable,
    shape: Tuple[int, int],
    size_mb: float,
    unlockedpd_module
) -> BenchmarkResult:
    """Benchmark a single operation."""
    rows, cols = shape

    # Create DataFrame
    df = pd.DataFrame(np.random.randn(rows, cols))

    # Benchmark with unlockedpd enabled
    unlockedpd_module.config.enabled = True
    try:
        optimized_time = time_operation(df, operation, warmup=1, runs=3)
    except Exception as e:
        print(f"  ERROR (optimized): {e}")
        optimized_time = float('inf')

    # Benchmark with unlockedpd disabled (vanilla pandas)
    unlockedpd_module.config.enabled = False
    try:
        pandas_time = time_operation(df, operation, warmup=1, runs=3)
    except Exception as e:
        print(f"  ERROR (pandas): {e}")
        pandas_time = float('inf')

    # Re-enable optimizations
    unlockedpd_module.config.enabled = True

    # Calculate speedup
    if optimized_time > 0 and pandas_time < float('inf'):
        speedup = pandas_time / optimized_time
    else:
        speedup = 0.0

    # Cleanup
    del df
    gc.collect()

    return BenchmarkResult(
        operation=name,
        shape=shape,
        size_mb=size_mb,
        pandas_time=pandas_time,
        optimized_time=optimized_time,
        speedup=speedup
    )


def get_operations() -> Dict[str, Callable]:
    """Get dictionary of operations to benchmark."""
    return {
        # Cumulative operations
        "cumsum": lambda df: df.cumsum(),
        "cumprod": lambda df: df.cumprod(),
        "cummin": lambda df: df.cummin(),
        "cummax": lambda df: df.cummax(),

        # Stats operations
        "skew": lambda df: df.skew(),
        "kurt": lambda df: df.kurt(),
        "sem": lambda df: df.sem(),

        # Pairwise operations (only for smaller shapes due to n^2 complexity)
        "corr": lambda df: df.corr(),
        "cov": lambda df: df.cov(),

        # Transform operations
        "diff": lambda df: df.diff(),
        "pct_change": lambda df: df.pct_change(),
        "shift": lambda df: df.shift(1),

        # Rolling operations (window=20)
        "rolling_mean": lambda df: df.rolling(20).mean(),
        "rolling_std": lambda df: df.rolling(20).std(),
        "rolling_min": lambda df: df.rolling(20).min(),
        "rolling_max": lambda df: df.rolling(20).max(),
        "rolling_sum": lambda df: df.rolling(20).sum(),
        "rolling_count": lambda df: df.rolling(20).count(),

        # Expanding operations
        "expanding_mean": lambda df: df.expanding().mean(),
        "expanding_std": lambda df: df.expanding().std(),
        "expanding_sum": lambda df: df.expanding().sum(),

        # EWM operations
        "ewm_mean": lambda df: df.ewm(span=20).mean(),
        "ewm_std": lambda df: df.ewm(span=20).std(),

        # Rank operation
        "rank_axis0": lambda df: df.rank(axis=0),
        "rank_axis1": lambda df: df.rank(axis=1),
    }


def format_time(t: float) -> str:
    """Format time in human-readable form."""
    if t == float('inf'):
        return "ERROR"
    elif t < 0.001:
        return f"{t*1000000:.1f}µs"
    elif t < 1:
        return f"{t*1000:.1f}ms"
    else:
        return f"{t:.2f}s"


def format_speedup(s: float) -> str:
    """Format speedup with color indicator."""
    if s == 0:
        return "N/A"
    elif s >= 2:
        return f"{s:.1f}x ⚡"
    elif s >= 1:
        return f"{s:.1f}x ✓"
    else:
        return f"{s:.1f}x ⚠"


def check_memory_available(size_mb: float) -> bool:
    """Check if we likely have enough memory for this test."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        # Need ~3x the data size (original + 2 results)
        required_mb = size_mb * 3
        return available_mb > required_mb + 1000  # 1GB buffer
    except ImportError:
        # No psutil, be conservative
        return size_mb <= 2048  # Max 2GB without psutil


def run_benchmarks(max_size_mb: float = 10240, operations: List[str] = None):
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("UNLOCKEDPD COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print()

    # Import and warmup unlockedpd
    print("Importing and warming up unlockedpd...")
    import unlockedpd
    unlockedpd._warmup_all()
    print("Warmup complete.")
    print()

    # Get test sizes (filter by max_size_mb)
    all_sizes = get_test_sizes()
    sizes = [(shape, size) for shape, size in all_sizes if size <= max_size_mb]

    # Get operations
    all_ops = get_operations()
    if operations:
        ops = {k: v for k, v in all_ops.items() if k in operations}
    else:
        ops = all_ops

    print(f"Testing {len(ops)} operations across {len(sizes)} array sizes")
    print(f"Max size: {max_size_mb}MB")
    print()

    results: List[BenchmarkResult] = []

    for (shape, size_mb) in sizes:
        rows, cols = shape

        # Check memory
        if not check_memory_available(size_mb):
            print(f"\n⚠ Skipping {size_mb}MB tests (insufficient memory)")
            continue

        print(f"\n{'='*60}")
        print(f"SIZE: {size_mb}MB | Shape: {rows:,} x {cols}")
        print(f"{'='*60}")

        for op_name, op_func in ops.items():
            # Skip corr/cov for very wide matrices (O(n^2) columns)
            if op_name in ("corr", "cov") and cols > 256:
                print(f"  {op_name:20s} SKIPPED (too many columns for pairwise)")
                continue

            # Skip slow operations for very large arrays
            if size_mb >= 5120 and op_name in ("rolling_std", "expanding_std", "ewm_std"):
                print(f"  {op_name:20s} SKIPPED (too slow for 5GB+)")
                continue

            try:
                result = benchmark_operation(op_name, op_func, shape, size_mb, unlockedpd)
                results.append(result)

                print(f"  {op_name:20s} pandas={format_time(result.pandas_time):>10s}  "
                      f"optimized={format_time(result.optimized_time):>10s}  "
                      f"speedup={format_speedup(result.speedup):>10s}")
            except MemoryError:
                print(f"  {op_name:20s} OUT OF MEMORY")
            except Exception as e:
                print(f"  {op_name:20s} ERROR: {e}")

        # Force garbage collection between sizes
        gc.collect()

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by operation
    op_speedups = {}
    for r in results:
        if r.operation not in op_speedups:
            op_speedups[r.operation] = []
        if r.speedup > 0:
            op_speedups[r.operation].append(r.speedup)

    print("\nAverage speedup by operation:")
    for op, speedups in sorted(op_speedups.items()):
        if speedups:
            avg = sum(speedups) / len(speedups)
            max_s = max(speedups)
            print(f"  {op:20s} avg={avg:.1f}x  max={max_s:.1f}x  (n={len(speedups)})")

    # Overall stats
    all_speedups = [r.speedup for r in results if r.speedup > 0]
    if all_speedups:
        print(f"\nOverall: avg={sum(all_speedups)/len(all_speedups):.1f}x  "
              f"median={sorted(all_speedups)[len(all_speedups)//2]:.1f}x  "
              f"max={max(all_speedups):.1f}x")

    print("\nBenchmark complete!")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark unlockedpd operations")
    parser.add_argument("--max-size", type=float, default=1024,
                        help="Maximum array size in MB (default: 1024)")
    parser.add_argument("--operations", nargs="+",
                        help="Specific operations to benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 100MB max, core operations only")

    args = parser.parse_args()

    if args.quick:
        run_benchmarks(
            max_size_mb=100,
            operations=["cumsum", "rolling_mean", "rank_axis1", "diff"]
        )
    else:
        run_benchmarks(
            max_size_mb=args.max_size,
            operations=args.operations
        )
