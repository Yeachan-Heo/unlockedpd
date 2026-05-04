"""Manual benchmark for rolling pairwise corr/cov memory-guard behavior.

Run:
    PYTHONPATH=src python3 benchmarks/bench_pairwise_rolling.py
"""

import gc
import time

import numpy as np
import pandas as pd

import unlockedpd
from unlockedpd.ops.pairwise import _estimate_pairwise_memory


def _time_call(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _bench_case(rows, cols, window):
    rng = np.random.default_rng(12345)
    df = pd.DataFrame(rng.normal(size=(rows, cols)))
    estimate = _estimate_pairwise_memory(rows, cols, df.values.nbytes)

    print(f"shape=({rows}, {cols}), window={window}")
    print(
        "estimated_output_mb="
        f"{estimate['output_bytes'] / 1024 / 1024:.2f}, "
        "estimated_scratch_mb="
        f"{estimate['scratch_bytes'] / 1024 / 1024:.2f}"
    )

    for op_name in ("corr", "cov"):
        unlockedpd.config.enabled = False
        pandas_time = _time_call(
            lambda: getattr(df.rolling(window), op_name)(), repeats=1
        )
        unlockedpd.config.enabled = True
        optimized_time = _time_call(
            lambda: getattr(df.rolling(window), op_name)(), repeats=3
        )
        speedup = pandas_time / optimized_time if optimized_time else float("inf")
        print(
            f"  rolling_{op_name}: pandas={pandas_time:.4f}s "
            f"optimized={optimized_time:.4f}s speedup={speedup:.2f}x"
        )


def main():
    for rows, cols, window in [
        (128, 8, 20),
        (512, 16, 20),
    ]:
        _bench_case(rows, cols, window)


if __name__ == "__main__":
    main()
