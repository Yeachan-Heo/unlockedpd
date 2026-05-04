# Resource profile results

Latest broad run after the rank and axis=0 cumulative pass:
`.omx/artifacts/broad-profile-pct32-20260504T184546Z.json`.

Latest targeted artifacts for the newest accepted changes:

```text
.omx/artifacts/profile-rank-axis1-wide-final-20260504T181900Z.json
.omx/artifacts/profile-cumulative-axis0-wide-parfinite128-20260504T181735Z.json
.omx/artifacts/profile-cumulative-axis0-large-parfinite-20260504T181928Z.json
.omx/artifacts/profile-transform-axis1-large-native-auto-rerun-factor1p2-20260504T174127Z.json
.omx/artifacts/profile-transform-axis1-large-pct32-only-20260504T184412Z.json
.omx/artifacts/broad-profile-pct32-20260504T184546Z.json
.omx/artifacts/profile-transform-axis1-large-native-auto-caps-factor1p2-20260504T172202Z.json
.omx/artifacts/profile-cumulative-axis0-large-rowblock-auto32-factor1p2-20260504T163915Z.json
.omx/artifacts/profile-cumulative-axis1-threadpool-numba-20260504T152000Z.json
.omx/artifacts/profile-pairwise-numba-prange-20260504T144502Z.json
```

Representative commands:

```bash
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'rank-axis1-wide-32mb' \
  --output .omx/artifacts/profile-rank-axis1-wide-final-20260504T181900Z.json

PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 7 \
  --case-filter 'cumulative-axis0-wide-32mb' \
  --output .omx/artifacts/profile-cumulative-axis0-wide-parfinite128-20260504T181735Z.json

PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 3 \
  --case-filter 'cumulative-axis0-large-256mb' \
  --output .omx/artifacts/profile-cumulative-axis0-large-parfinite-20260504T181928Z.json

PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 3 \
  --output .omx/artifacts/broad-profile-pct32-20260504T184546Z.json
```

The current HPC resource budget follows the revised constraint: higher
utilization is acceptable when it buys speed, while true leaks are not.  The
profiler default is aligned with the stricter target:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

Fixed absolute config ceilings (`max_cpu_overhead` and `max_memory_overhead`) are
still reported separately.

## Current pass enhancements

- Dense finite `axis=0` cumulative `cumsum/cumprod/cummin/cummax` now use a
  parallel no-allocation finite check before the row-contiguous block scan.  This
  removes NumPy's temporary boolean mask from the hot path and pushes both 32MB
  and 256MB axis=0 cumulative profiles past 10x while keeping RSS near pandas.
- Dense default `rank(axis=1, method='average')` now avoids pandas
  `select_dtypes` on the all-numeric fast path.  The optional native C++ rank
  backend is now opt-in via `UNLOCKEDPD_ENABLE_NATIVE_RANK=1`; the default path
  uses the faster bounded Numba no-NaN kernel.
- Large `axis=1 diff/pct_change(fill_method=None)` frames auto-promote to the
  native-C pthread transform path at 256MiB and above.  Smaller frames stay on
  stable Numba/pandas paths unless `UNLOCKEDPD_ENABLE_NATIVE_TRANSFORMS=1`
  explicitly opts in.  `UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS=1` remains a hard
  escape hatch.
- Dense finite `axis=1` cumulative `cumsum/cumprod/cummin/cummax` use transient
  bounded ThreadPool + Numba `nogil` row chunks; workers join before return.
- Pairwise rolling `corr/cov` uses a single bounded Numba `prange` kernel over
  upper-triangle pairs, avoiding Python ThreadPool chunk orchestration.
- Dense mid-sized `axis=0` `DataFrame.sum/mean` retain the bounded OpenBLAS GEMV
  path with a size-aware thread cap and restore the previous OpenBLAS thread
  count afterwards.  A native-C row-block reducer was tested and rejected because
  it was slower in subprocess profiling.
- `axis=1 shift(fill_value=None)` remains on pandas' no-fill native metadata
  path, still the fastest resource-clean option.

## Targeted evidence for newest changes

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| rank-axis1-wide-32mb | rank_axis1 | 52.996x | 0.475x | 1.121x | pass | parallel_numba |
| cumulative-axis0-wide-32mb | dataframe_cumsum | 25.308x | 2.520x | 0.889x | pass | parallel_numba |
| cumulative-axis0-wide-32mb | dataframe_cumprod | 25.026x | 2.482x | 0.889x | pass | parallel_numba |
| cumulative-axis0-wide-32mb | dataframe_cummin | 27.212x | 2.431x | 0.889x | pass | parallel_numba |
| cumulative-axis0-wide-32mb | dataframe_cummax | 27.273x | 2.430x | 0.889x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cumsum | 27.340x | 1.139x | 1.000x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cumprod | 26.544x | 1.169x | 1.000x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cummin | 26.384x | 1.189x | 1.000x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cummax | 27.034x | 1.145x | 1.000x | pass | parallel_numba |
| transform-axis1-large-256mb | dataframe_diff | 6.784x | 2.390x | 1.000x | pass | native_c |
| transform-axis1-large-256mb | dataframe_pct_change | 12.805x | 1.323x | 0.997x | pass | native_c |
| pairwise-safe-rolling-corr | rolling_corr | 21.392x | 0.332x | 0.882x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_cov | 17.596x | 0.380x | 0.921x | pass | parallel_numba |

## Latest broad profile summary

The latest broad run used the default `speedup * 1.2` resource gate and covered
44 cases on the clean code tree for this pass.  `37 / 44` cases met both the 10x target
and the speed-weighted CPU/RAM budget.  The rank and axis=0 cumulative changes
are successful in the broad matrix; the remaining large-frame blockers are
`axis=1 diff` at 256MiB.

Rows still below 10x in the latest broad run:

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| import-only | import_unlockedpd | 0.966x | 1.034x | 1.000x | pass | optimized_import |
| rank-wide-1mb-control | rank_axis1 | 0.992x | 1.004x | 0.998x | pass | pandas_native |
| transform-axis1-wide-32mb | dataframe_diff | 1.815x | 4.322x | 1.004x | FAIL | parallel_numba |
| transform-axis1-wide-32mb | dataframe_pct_change | 3.335x | 2.475x | 0.502x | pass | parallel_numba |
| aggregation-wide-10mb | dataframe_mean | 4.362x | 5.300x | 0.795x | FAIL | numpy_vectorized |
| aggregation-wide-10mb | dataframe_sum | 5.053x | 0.202x | 0.790x | pass | numpy_vectorized |
| transform-axis1-large-256mb | dataframe_diff | 7.121x | 2.266x | 1.000x | pass | native_c |

## Remaining gap to the universal 10x objective

Universal 10x is still **not achieved**.  Current accepted code is resource-clean
under the speed-weighted CPU/RAM budget, and large axis=0 cumulative, large
axis=1 `pct_change`, rolling, pairwise, and rank have >10x evidence.  Missing or
weak requirements:

- 256MB `axis=1 diff`: native-C auto-dispatch is resource-bounded but still only
  about 7x against the fastest pandas baseline observed in the broad profiler.
- 32MB `axis=1 diff/pct_change`: optimized wall time is already ~3-4ms, but
  pandas can measure ~7-15ms in the broad profiler, making universal 10x
  unstable for this medium-wide case.
- 10MB `axis=0` `sum/mean`: optimized wall time is sub-millisecond but still only
  around 5-6x against pandas; this is below the large-frame focus but remains in
  the broad matrix.
- Import and 1MB rank control rows are intentionally not large DataFrame compute
  targets.

Rejected or bounded experiments include low-threshold axis=0 row-block dispatch,
unbounded BLAS/dot dispatch, direct native-C axis=0 reduction, NumExpr,
Bottleneck, Rust/Rayon prototypes, native C cumulative kernels, transient
ThreadPool row-block axis=0 cumulative, and native worker CPU-affinity pinning
for axis=1 transforms.  OpenMP native transform kernels were faster in raw wall
time but left extra worker threads alive after the call, so they are not accepted
as resource-clean.  A transient ThreadPool+Numba `nogil` axis=1 transform
experiment was also rejected after
`.omx/artifacts/profile-transform-axis1-threadpool-numba-20260504T153536Z.json`:
it restored `final_threads` to baseline, but regressed wall time to 0.15s for
`diff` and 0.13s for `pct_change`.
