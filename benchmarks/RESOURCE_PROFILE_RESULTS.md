# Resource profile results

Latest broad run: `.omx/artifacts/broad-profile-final-20260504T122846Z.json`.
Latest targeted run for the current optimization pass: `.omx/artifacts/profile-targeted-final-20260504T122612Z.json`.

Commands:

```bash
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 5 --output .omx/artifacts/broad-profile-final-20260504T122846Z.json
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'aggregation-wide-10mb' \
  --case-filter 'aggregation-medium-100mb' \
  --case-filter 'transform-axis1-wide-32mb' \
  --output .omx/artifacts/profile-targeted-final-20260504T122612Z.json
```

The resource budget is speedup-weighted:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

## Current pass enhancements

- Dense mid-sized `axis=0` `DataFrame.sum/mean` can use a bounded OpenBLAS GEMV path (`arr.T @ ones`) with the OpenBLAS thread count capped during the call and restored afterwards.
- `DataFrame.pct_change(axis=1, fill_method=None)` now uses a bounded row-parallel Numba kernel for large real-numeric frames, avoiding pandas `shift`/binary-op intermediates.
- `DataFrame.diff(axis=1)` intentionally remains on the pandas-primitives path: direct Numba and Rust/Rayon prototypes improved wall time in some runs but exceeded the speedup-weighted CPU budget or stayed far below the 10x target.

## Targeted evidence for changed areas

| case | operation | speedup | CPU ratio | RSS ratio | budget | optimized path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| aggregation-wide-10mb | dataframe_mean | 5.595x | 2.120x | 0.795x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_sum | 5.622x | 1.654x | 0.788x | pass | numpy_vectorized |
| aggregation-medium-100mb | dataframe_mean | 23.681x | 1.671x | 0.080x | pass | parallel_numba |
| aggregation-medium-100mb | dataframe_sum | 20.508x | 1.484x | 0.080x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_diff | 0.988x | 1.006x | 1.002x | pass | pandas_primitives |
| transform-axis1-wide-32mb | dataframe_shift | 11.179x | 0.089x | 0.031x | pass | pandas_native |
| transform-axis1-wide-32mb | dataframe_pct_change | 3.311x | 2.454x | 0.502x | pass | parallel_numba |

## Broad profile summary

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| transform-axis1-wide-32mb | dataframe_diff | 0.655x | 1.529x | 1.002x | FAIL | pandas_primitives |
| cumulative-axis1-wide-32mb | dataframe_cumprod | 0.705x | 1.418x | 1.007x | FAIL | numpy_vectorized |
| import-only | import_unlockedpd | 0.981x | 1.015x | 1.000x | pass | optimized_import |
| rank-wide-1mb-control | rank_axis1 | 1.002x | 0.996x | 0.993x | pass | pandas_native |
| cumulative-axis1-wide-32mb | dataframe_cummax | 1.343x | 0.746x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cumsum | 1.553x | 0.646x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cummin | 3.110x | 0.322x | 1.007x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_mean | 4.510x | 2.407x | 0.795x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_sum | 5.457x | 1.624x | 0.790x | pass | numpy_vectorized |
| aggregation-axis1-wide-32mb | dataframe_max | 10.202x | 2.819x | 0.254x | pass | parallel_numba |
| rank-axis1-wide-32mb | rank_axis1 | 11.027x | 0.601x | 1.121x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_max | 11.542x | 0.272x | 0.960x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_min | 11.637x | 0.269x | 0.955x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_corr | 12.271x | 0.320x | 0.857x | pass | threadpool |
| aggregation-axis1-wide-32mb | dataframe_min | 13.030x | 3.857x | 0.254x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_cov | 16.314x | 0.315x | 0.903x | pass | threadpool |
| aggregation-axis1-wide-32mb | dataframe_sum | 16.828x | 0.951x | 0.254x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_mean | 18.554x | 0.806x | 0.254x | pass | parallel_numba |
| aggregation-medium-100mb | dataframe_mean | 21.815x | 1.408x | 0.080x | pass | parallel_numba |
| expanding-wide-10mb | expanding_mean | 21.980x | 0.637x | 0.501x | pass | parallel_numba |
| aggregation-medium-100mb | dataframe_sum | 24.375x | 1.433x | 0.080x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_std | 25.741x | 0.330x | 0.885x | pass | parallel_numba |
| rolling-wide-10mb | rolling_mean | 26.489x | 0.479x | 0.998x | pass | parallel_numba |
| rolling-wide-10mb | rolling_sum | 28.256x | 0.667x | 0.999x | pass | parallel_numba |
| rolling-medium-100mb | rolling_mean | 30.558x | 0.482x | 1.000x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_var | 30.953x | 0.313x | 0.838x | pass | parallel_numba |
| rolling-medium-100mb | rolling_std | 35.645x | 0.418x | 1.000x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_sum | 35.857x | 0.293x | 0.858x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_mean | 36.500x | 0.294x | 0.885x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_std | 50.626x | 0.020x | 0.251x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_var | 61.697x | 0.249x | 0.251x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_pct_change | 289.832x | 0.027x | 0.502x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_shift | 634.057x | 0.002x | 0.031x | pass | pandas_native |

## Remaining gap to the universal 10x objective

Universal 10x is still **not** met.  The latest targeted evidence leaves `axis=1 diff` near parity, `axis=1 pct_change` at ~3.3x, and 10MB `axis=0` sum/mean at ~5.6x.  The latest broad run also shows noisy cumulative-axis1 results below 10x, despite prior and local targeted cumulative measurements showing NumPy row-wise accumulation can be much faster than pandas on the same shape.

Rejected experiments include low-threshold axis=0 row-block dispatch for the 10MB case, unbounded BLAS/dot dispatch that failed CPU budget, direct Numba `axis=1 diff` because CPU ratio exceeded the speedup-weighted budget, NumExpr, Bottleneck, and a Rust ctypes/Rayon prototype for axis=1 diff/pct/shift.  The Rust prototype improved copying transforms only to roughly 2-6x locally and made axis=1 shift slower than pandas' metadata-based native shift, so it was not adopted into the package.
