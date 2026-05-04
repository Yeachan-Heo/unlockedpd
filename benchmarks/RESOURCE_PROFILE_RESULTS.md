# Resource profile results

Latest broad run: `.omx/artifacts/broad-profile-native-c-cap8-final-20260504T125650Z.json`.
Latest targeted run for changed/high-risk areas: `.omx/artifacts/profile-targeted-native-c-cap8-final-20260504T130116Z.json`.

Commands:

```bash
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 5 --output .omx/artifacts/broad-profile-native-c-cap8-final-20260504T125650Z.json
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'aggregation-wide-10mb' \
  --case-filter 'aggregation-medium-100mb' \
  --case-filter 'cumulative-axis1-wide-32mb' \
  --case-filter 'transform-axis1-wide-32mb' \
  --output .omx/artifacts/profile-targeted-native-c-cap8-final-20260504T130116Z.json
```

The resource budget is speedup-weighted:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

## Current pass enhancements

- Dense mid-sized `axis=0` `DataFrame.sum/mean` can use a bounded OpenBLAS GEMV path (`arr.T @ ones`) with the OpenBLAS thread count capped during the call and restored afterwards.
- Large real-numeric `axis=1 diff` and `pct_change(fill_method=None)` can use an optional cached native C/pthread kernel.  It compiles into the user cache on first use when a C compiler is available, falls back cleanly otherwise, and does not leave extra worker threads alive after each call.
- `axis=1 shift(fill_value=None)` remains on pandas' no-fill native metadata path, which is still the fastest resource-clean option.

## Targeted evidence for changed/high-risk areas

| case | operation | speedup | CPU ratio | RSS ratio | budget | optimized path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| aggregation-wide-10mb | dataframe_mean | 5.253x | 1.238x | 0.796x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_sum | 5.584x | 2.195x | 0.790x | pass | numpy_vectorized |
| aggregation-medium-100mb | dataframe_mean | 20.389x | 1.042x | 0.080x | pass | parallel_numba |
| aggregation-medium-100mb | dataframe_sum | 17.245x | 2.295x | 0.080x | pass | parallel_numba |
| cumulative-axis1-wide-32mb | dataframe_cumsum | 11.863x | 0.084x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cumprod | 11.566x | 0.087x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cummin | 11.210x | 0.089x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cummax | 10.739x | 0.093x | 1.007x | pass | numpy_vectorized |
| transform-axis1-wide-32mb | dataframe_diff | 3.118x | 1.620x | 1.004x | pass | native_c |
| transform-axis1-wide-32mb | dataframe_shift | 11.181x | 0.091x | 0.031x | pass | pandas_native |
| transform-axis1-wide-32mb | dataframe_pct_change | 6.379x | 0.888x | 0.502x | pass | native_c |

## Broad profile summary

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| import-only | import_unlockedpd | 0.992x | 1.015x | 1.000x | pass | optimized_import |
| rank-wide-1mb-control | rank_axis1 | 1.063x | 0.938x | 0.995x | pass | pandas_native |
| transform-axis1-wide-32mb | dataframe_diff | 2.374x | 1.763x | 1.004x | pass | native_c |
| transform-axis1-wide-32mb | dataframe_pct_change | 4.037x | 0.972x | 0.502x | pass | native_c |
| aggregation-wide-10mb | dataframe_sum | 4.952x | 2.406x | 0.790x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_mean | 5.463x | 0.891x | 0.794x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cummax | 11.078x | 0.090x | 1.007x | pass | numpy_vectorized |
| rank-axis1-wide-32mb | rank_axis1 | 11.200x | 0.606x | 1.121x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_min | 11.507x | 0.270x | 0.955x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_shift | 11.526x | 0.088x | 0.031x | pass | pandas_native |
| cumulative-axis1-wide-32mb | dataframe_cummin | 11.559x | 0.087x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cumprod | 11.918x | 0.084x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cumsum | 12.206x | 0.082x | 1.007x | pass | numpy_vectorized |
| aggregation-axis1-wide-32mb | dataframe_min | 12.304x | 2.028x | 0.254x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_max | 12.560x | 2.049x | 0.254x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_max | 12.567x | 0.249x | 0.964x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_mean | 14.680x | 1.533x | 0.254x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_cov | 15.263x | 0.325x | 0.898x | pass | threadpool |
| pairwise-safe-rolling-corr | rolling_corr | 15.514x | 0.350x | 0.895x | pass | threadpool |
| aggregation-axis1-wide-32mb | dataframe_sum | 15.840x | 0.539x | 0.254x | pass | parallel_numba |
| aggregation-medium-100mb | dataframe_sum | 17.844x | 1.723x | 0.080x | pass | parallel_numba |
| expanding-wide-10mb | expanding_mean | 22.079x | 0.795x | 0.501x | pass | parallel_numba |
| aggregation-medium-100mb | dataframe_mean | 24.453x | 1.311x | 0.080x | pass | parallel_numba |
| rolling-wide-10mb | rolling_mean | 26.021x | 0.544x | 0.998x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_std | 26.394x | 0.323x | 0.953x | pass | parallel_numba |
| rolling-wide-10mb | rolling_sum | 28.409x | 0.580x | 0.999x | pass | parallel_numba |
| rolling-medium-100mb | rolling_mean | 30.058x | 0.501x | 1.000x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_var | 31.419x | 0.314x | 0.839x | pass | parallel_numba |
| rolling-medium-100mb | rolling_std | 35.026x | 0.423x | 1.000x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_sum | 35.078x | 0.282x | 0.834x | pass | parallel_numba |
| rolling-axis1-wide-32mb | rolling_mean | 41.726x | 0.257x | 0.858x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_std | 51.301x | 0.404x | 0.251x | pass | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_var | 52.255x | 0.394x | 0.251x | pass | parallel_numba |

## Remaining gap to the universal 10x objective

Universal 10x is still **not** met.  The latest broad and targeted profiles are resource-clean, and almost all large-frame cases now clear 10x.  Remaining speed gaps are `axis=1 diff` (~2.4-3.1x), `axis=1 pct_change(fill_method=None)` (~4.0-6.4x), and the 10MB `axis=0` `sum/mean` case (~5x).  The small import/rank control cases are intentionally not large-dataframe optimization targets.

Rejected or bounded experiments include low-threshold axis=0 row-block dispatch for the 10MB case, unbounded BLAS/dot dispatch that failed CPU budget, direct Numba `axis=1 diff` because CPU ratio exceeded the speedup-weighted budget, NumExpr, Bottleneck, and Rust/Rayon prototypes.  OpenMP native transform kernels were faster than pthreads but left extra worker threads alive after the call, so the committed native path uses pthread create/join to avoid thread leaks.
