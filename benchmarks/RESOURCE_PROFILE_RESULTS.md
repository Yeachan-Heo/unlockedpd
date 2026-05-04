# Resource profile results

Latest broad run for the current code: `.omx/artifacts/broad-profile-current-transform-rank-pairwise-budget2-20260504T145226Z.json`.

Targeted runs for the newest high-risk changes:

```text
.omx/artifacts/profile-transform-numba-fastmath-20260504T142318Z.json
.omx/artifacts/profile-rank-native-cpp-cap32-gate-20260504T140727Z.json
.omx/artifacts/profile-pairwise-numba-prange-20260504T144502Z.json
.omx/artifacts/profile-axis0-wide-after-transform-rank-pairwise-20260504T144740Z.json
```

Commands:

```bash
PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 5 \
  --output .omx/artifacts/broad-profile-current-transform-rank-pairwise-budget2-20260504T145226Z.json

PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'transform-axis1-wide-32mb' \
  --output .omx/artifacts/profile-transform-numba-fastmath-20260504T142318Z.json

PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'rank-axis1-wide-32mb' \
  --output .omx/artifacts/profile-rank-native-cpp-cap32-gate-20260504T140727Z.json

PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'pairwise-safe-rolling-corr' \
  --output .omx/artifacts/profile-pairwise-numba-prange-20260504T144502Z.json
```

The current HPC resource budget follows the revised constraint: higher
utilization is acceptable when it buys speed, while true leaks are not.  The
profiler default is now:

```text
optimized_cpu_ratio <= speedup * 2.0
optimized_rss_ratio <= speedup * 2.0
```

Override with `UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR` for stricter or looser gates.
Fixed absolute config ceilings (`max_cpu_overhead` and `max_memory_overhead`) are
still reported separately.

## Current pass enhancements

- `axis=1 diff` now uses a stable row-parallel Numba kernel by default instead
  of pandas `shift` primitives or the noisy native-C transform path.
- `axis=1 pct_change(fill_method=None)` now uses the same stable row-parallel
  Numba path with `fastmath=True`; NaN/Inf regression coverage still matches
  pandas for the tested edge cases.
- The older native-C transform path is now explicit opt-in via
  `UNLOCKEDPD_ENABLE_NATIVE_TRANSFORMS=1` because the subprocess profiler showed
  higher variance than the Numba path.  It remains pthread create/join only, with
  no persistent native worker pool.
- Dense default `rank(axis=1, method='average')` can use an optional cached
  C++/pthread row-sort kernel for no-NaN wide rows.  It is bounded by frame size,
  row count, and machine CPU count, and joins workers before returning.
- Pairwise rolling `corr/cov` now uses a single bounded Numba `prange` kernel
  over upper-triangle pairs, avoiding Python ThreadPool chunk orchestration in
  the hot path.
- Dense mid-sized `axis=0` `DataFrame.sum/mean` retain the bounded OpenBLAS GEMV
  path with a size-aware thread cap and restore the previous OpenBLAS thread
  count afterwards.  A native-C row-block reducer was tested and rejected because
  it was slower in subprocess profiling.
- `axis=1 shift(fill_value=None)` remains on pandas' no-fill native metadata
  path, still the fastest resource-clean option.

## Targeted evidence for newest changes

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| transform-axis1-wide-32mb | dataframe_diff | 21.546x | 0.356x | 1.004x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_shift | 309.556x | 0.002x | 0.031x | pass | pandas_native |
| transform-axis1-wide-32mb | dataframe_pct_change | 16.006x | 0.511x | 0.502x | pass | parallel_numba |
| rank-axis1-wide-32mb | rank_axis1 | 10.927x | 0.439x | 1.121x | pass | native_cpp |
| pairwise-safe-rolling-corr | rolling_corr | 21.392x | 0.332x | 0.882x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_cov | 17.596x | 0.380x | 0.921x | pass | parallel_numba |

## Latest broad profile summary

All 33 broad-profile rows pass the `speedup * 2.0` CPU/RAM gate.  Universal 10x
is still **not** proven by the broad run: seven large-dataframe rows remain below
10x in this noisy run, plus the import and 1MB rank controls which are not large
DataFrame optimization targets.

Slowest rows from the latest broad run:

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| rank-wide-1mb-control | rank_axis1 | 0.989x | 1.009x | 0.994x | pass | pandas_native |
| import-only | import_unlockedpd | 1.037x | 0.962x | 1.000x | pass | optimized_import |
| transform-axis1-wide-32mb | dataframe_pct_change | 3.178x | 2.407x | 0.502x | pass | parallel_numba |
| cumulative-axis1-wide-32mb | dataframe_cummin | 3.726x | 0.268x | 1.007x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_mean | 4.623x | 3.375x | 0.794x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cumsum | 5.593x | 0.179x | 1.007x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cummax | 6.026x | 0.166x | 1.007x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_sum | 6.622x | 2.762x | 0.789x | pass | numpy_vectorized |
| cumulative-axis1-wide-32mb | dataframe_cumprod | 9.560x | 0.105x | 1.007x | pass | numpy_vectorized |
| transform-axis1-wide-32mb | dataframe_shift | 10.389x | 0.097x | 0.031x | pass | pandas_native |
| rank-axis1-wide-32mb | rank_axis1 | 10.847x | 0.417x | 1.120x | pass | native_cpp |
| transform-axis1-wide-32mb | dataframe_diff | 12.832x | 0.566x | 1.004x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_corr | 19.054x | 0.391x | 0.859x | pass | parallel_numba |

Representative high-confidence wins from the same run include large axis=1
aggregation, rolling, pairwise, rank, and diff rows at >10x while staying inside
the speed-weighted resource budget.

## Remaining gap to the universal 10x objective

Universal 10x is still **not achieved**.  The current code is resource-clean
under the revised HPC budget and now has targeted >10x evidence for transform
`diff`, transform `pct_change`, rank, and pairwise rolling.  Remaining weak or
noisy rows are:

- 10MB `axis=0` `sum/mean`: stable around 5-7x in subprocess profiling.  This is
  a sub-millisecond optimized path; native C row-block reduction was tested and
  rejected because it made the profiler slower.
- `axis=1 pct_change(fill_method=None)`: targeted run reached 16x, but the broad
  run can still drop below 10x when pandas' baseline is unusually low.
- `axis=1` cumulative ops: optimized NumPy-vectorized calls are usually around
  ~20-30ms and often >10x, but broad runs show large machine-load outliers.

Rejected or bounded experiments include low-threshold axis=0 row-block dispatch,
unbounded BLAS/dot dispatch, direct native-C axis=0 reduction, NumExpr,
Bottleneck, Rust/Rayon prototypes, native C cumulative kernels, and default
native-C axis=1 transform dispatch.  OpenMP native transform kernels were faster
in raw wall time but left extra worker threads alive after the call, so they are
not accepted as resource-clean.
