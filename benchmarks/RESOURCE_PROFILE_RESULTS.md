# Resource profile results

Latest broad run before the axis=0 cumulative pass: `.omx/artifacts/broad-profile-with-large-transform-20260504T154000Z.json`.

Latest targeted axis=0 cumulative run for this pass: `.omx/artifacts/profile-cumulative-axis0-large-rowblock-auto32-factor1p2-20260504T163915Z.json`.

Latest targeted large axis=1 transform run for the current auto-native gate:
`.omx/artifacts/profile-transform-axis1-large-native-auto-caps-factor1p2-20260504T172202Z.json`.

Targeted runs for the newest high-risk changes:

```text
.omx/artifacts/profile-cumulative-axis0-large-rowblock-auto32-factor1p2-20260504T163915Z.json
.omx/artifacts/profile-cumulative-axis0-current-20260504T155529Z.json
.omx/artifacts/profile-cumulative-axis0-threadpool-rowblock-20260504T161105Z.json
.omx/artifacts/profile-cumulative-axis0-native-rowblock-20260504T161655Z.json
.omx/artifacts/profile-transform-axis1-large-native-auto-caps-factor1p2-20260504T172202Z.json
.omx/artifacts/profile-transform-axis1-large-native-auto-factor1p2-20260504T171501Z.json
.omx/artifacts/profile-cumulative-axis1-threadpool-numba-20260504T152000Z.json
.omx/artifacts/profile-transform-axis1-large-256mb-20260504T153910Z.json
.omx/artifacts/profile-transform-numba-fastmath-20260504T142318Z.json
.omx/artifacts/profile-rank-native-cpp-cap32-gate-20260504T140727Z.json
.omx/artifacts/profile-pairwise-numba-prange-20260504T144502Z.json
.omx/artifacts/profile-axis0-wide-after-transform-rank-pairwise-20260504T144740Z.json
```

Commands:

```bash
PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 5 \
  --output .omx/artifacts/broad-profile-with-large-transform-20260504T154000Z.json

PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=1.2 poetry run python \
  benchmarks/profile_resources.py --repeats 3 \
  --case-filter 'cumulative-axis0-large-256mb' \
  --output .omx/artifacts/profile-cumulative-axis0-large-rowblock-auto32-factor1p2-20260504T163915Z.json

PYTHONPATH=src poetry run python \
  benchmarks/profile_resources.py --repeats 3 \
  --case-filter 'transform-axis1-large-256mb:dataframe_diff' \
  --case-filter 'transform-axis1-large-256mb:dataframe_pct_change' \
  --output .omx/artifacts/profile-transform-axis1-large-native-auto-caps-factor1p2-20260504T172202Z.json

PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 5 \
  --case-filter 'cumulative-axis1-wide-32mb' \
  --thread-sample-interval 0.001 \
  --output .omx/artifacts/profile-cumulative-axis1-threadpool-numba-20260504T152000Z.json

PYTHONPATH=src UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR=2.0 poetry run python \
  benchmarks/profile_resources.py --repeats 3 \
  --case-filter 'transform-axis1-large-256mb' \
  --thread-sample-interval 0.005 \
  --output .omx/artifacts/profile-transform-axis1-large-256mb-20260504T153910Z.json

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
profiler default is now aligned with the stricter target:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

Override with `UNLOCKEDPD_SPEEDUP_RESOURCE_FACTOR` for stricter or looser gates.
Fixed absolute config ceilings (`max_cpu_overhead` and `max_memory_overhead`) are
still reported separately.

## Current pass enhancements

- Non-large `axis=1 diff` uses a stable row-parallel Numba kernel instead of
  pandas `shift` primitives or the noisy small-frame native-C transform path.
- Non-large `axis=1 pct_change(fill_method=None)` uses the same stable
  row-parallel Numba path with `fastmath=True`; NaN/Inf regression coverage
  still matches pandas for the tested edge cases.
- Large `axis=1 diff/pct_change(fill_method=None)` frames now auto-promote to
  the native-C pthread transform path at 256MiB and above.  Smaller frames stay
  on the stable Numba/pandas paths unless `UNLOCKEDPD_ENABLE_NATIVE_TRANSFORMS=1`
  explicitly opts in.  `UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS=1` remains a hard
  escape hatch.  The native path creates and joins pthread workers per call, so
  it does not leave a persistent native worker pool behind.
- The profiler's speedup-weighted resource factor now defaults consistently to
  `1.2` in both the run config snapshot and the actual summary gate.
- Dense default `rank(axis=1, method='average')` can use an optional cached
  C++/pthread row-sort kernel for no-NaN wide rows.  It is bounded by frame size,
  row count, and machine CPU count, and joins workers before returning.
- Pairwise rolling `corr/cov` now uses a single bounded Numba `prange` kernel
  over upper-triangle pairs, avoiding Python ThreadPool chunk orchestration in
  the hot path.
- Dense finite `axis=1` cumulative `cumsum/cumprod/cummin/cummax` now use
  transient bounded ThreadPool + Numba `nogil` row chunks.  The cap scales with
  frame bytes, row count, and machine CPU count; workers join before return, so
  profiler `final_threads` stays at the pre-call baseline.
- Dense finite `axis=0` cumulative `cumsum/cumprod/cummin/cummax` now use
  row-contiguous Numba block scans with a size-aware 16/32 thread cap.  This
  targets large column-wise cumulative cases that previously parallelized by
  strided columns.  Non-finite frames stay on the existing pandas-compatible
  safe path.
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
| transform-axis1-large-256mb | dataframe_diff | 2.032x | 3.653x | 1.000x | pass | parallel_numba |
| transform-axis1-large-256mb | dataframe_shift | 22.596x | 0.044x | 0.004x | pass | pandas_native |
| transform-axis1-large-256mb | dataframe_pct_change | 3.511x | 2.113x | 0.997x | pass | parallel_numba |
| transform-axis1-large-256mb | dataframe_diff | 7.747x | 2.816x | 1.000x | pass | native_c |
| transform-axis1-large-256mb | dataframe_pct_change | 11.618x | 0.867x | 0.997x | pass | native_c |
| cumulative-axis0-large-256mb | dataframe_cumsum | 20.705x | 0.207x | 1.000x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cumprod | 18.136x | 0.287x | 1.000x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cummin | 108.049x | 0.124x | 1.000x | pass | parallel_numba |
| cumulative-axis0-large-256mb | dataframe_cummax | 12.452x | 1.160x | 1.000x | pass | parallel_numba |
| cumulative-axis1-wide-32mb | dataframe_cumsum | 22.478x | 0.097x | 1.003x | pass | threadpool |
| cumulative-axis1-wide-32mb | dataframe_cumprod | 22.099x | 0.097x | 1.003x | pass | threadpool |
| cumulative-axis1-wide-32mb | dataframe_cummin | 20.454x | 0.116x | 1.003x | pass | threadpool |
| cumulative-axis1-wide-32mb | dataframe_cummax | 20.252x | 0.115x | 1.003x | pass | threadpool |
| rank-axis1-wide-32mb | rank_axis1 | 10.927x | 0.439x | 1.121x | pass | native_cpp |
| pairwise-safe-rolling-corr | rolling_corr | 21.392x | 0.332x | 0.882x | pass | parallel_numba |
| pairwise-safe-rolling-corr | rolling_cov | 17.596x | 0.380x | 0.921x | pass | parallel_numba |

The latest native auto-dispatch transform artifact was captured while the host
was heavily loaded by unrelated long-running CPU jobs and with the tree dirty;
it is useful dispatch/regression evidence, not a clean completion proof for the
universal 10x objective.

## Latest broad profile summary

The cumulative axis=1 rows now clear 20x in both the targeted and broad runs,
and the broad matrix now includes an explicit 256MB axis=1 transform case.  All 36 rows in that pre-axis0 broad run passed the then-current
`speedup * 2.0` CPU/RAM gate, but universal 10x was still **not** proven: explicit 256MB transform `diff`/`pct_change`, 32MB
`pct_change`, 10MB `axis=0` sum/mean, and a few noisy broad rows remain below
10x.

Slowest rows from the latest broad run:

| case | operation | speedup | CPU ratio | RSS ratio | budget | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| rank-wide-1mb-control | rank_axis1 | 0.971x | 1.028x | 0.993x | pass | pandas_native |
| transform-axis1-large-256mb | dataframe_diff | 2.455x | 3.163x | 1.000x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_pct_change | 3.687x | 2.148x | 0.502x | pass | parallel_numba |
| transform-axis1-large-256mb | dataframe_pct_change | 3.840x | 2.003x | 0.997x | pass | parallel_numba |
| aggregation-wide-10mb | dataframe_sum | 5.603x | 3.300x | 0.791x | pass | numpy_vectorized |
| aggregation-wide-10mb | dataframe_mean | 5.941x | 0.171x | 0.799x | pass | numpy_vectorized |
| rank-axis1-wide-32mb | rank_axis1 | 7.934x | 0.476x | 1.121x | pass | native_cpp |
| rolling-axis1-wide-32mb | rolling_max | 9.456x | 0.274x | 0.953x | pass | parallel_numba |
| transform-axis1-wide-32mb | dataframe_diff | 9.656x | 0.814x | 1.004x | pass | parallel_numba |
| import-only | import_unlockedpd | 1.144x | 0.875x | 1.000x | pass | optimized_import |
| transform-axis1-large-256mb | dataframe_shift | 58.892x | 0.017x | 0.004x | pass | pandas_native |

Representative high-confidence wins from the same run include large axis=1
aggregation, rolling, pairwise, cumulative, and shift rows at >10x while staying
inside the speed-weighted resource budget.

## Remaining gap to the universal 10x objective

Universal 10x is still **not achieved**.  The current code is inside the
speed-weighted CPU/RAM budget under targeted profiling and now has targeted
>10x evidence for axis=0 large cumulative, axis=1 cumulative, rank, pairwise
rolling, and some 32MB transform runs.  Remaining weak or noisy
rows are:

- 256MB `axis=1` transform `diff`: large-frame native auto-dispatch improved
  the loaded-host profiler from 2-4x to 7.747x mean / 7.545x median, but it is
  still below the universal 10x target and remains the main true large-DataFrame
  blocker.
- 256MB `axis=1 pct_change(fill_method=None)`: now clears 10x in the latest
  targeted native auto-dispatch run, but needs a fresh clean broad run before it
  can be removed from the watch list.
- 10MB `axis=0` `sum/mean`: stable around 5-7x in subprocess profiling.  This is
  a sub-millisecond optimized path; native C row-block reduction was tested and
  rejected because it made the profiler slower.
- `axis=1 pct_change(fill_method=None)`: targeted run reached 16x, but broad
  runs can still drop below 10x when pandas' baseline is unusually low.
- `axis=1 diff`: targeted and previous broad runs cleared 10x, but the newest
  broad run dropped below 10x because pandas diff measured unusually fast.
- 32MB axis=0 cumulative remains noisy and can fall below 10x in subprocess
  means; the accepted target for this pass is the 256MB large-frame case.

Rejected or bounded experiments include low-threshold axis=0 row-block dispatch,
unbounded BLAS/dot dispatch, direct native-C axis=0 reduction, NumExpr,
Bottleneck, Rust/Rayon prototypes, native C cumulative kernels, transient
ThreadPool row-block axis=0 cumulative, and native worker CPU-affinity pinning
for axis=1 transforms.  OpenMP native transform kernels were faster
in raw wall time but left extra worker threads alive after the call, so they are
not accepted as resource-clean.  A transient ThreadPool+Numba `nogil` axis=1
transform experiment was also rejected after
`.omx/artifacts/profile-transform-axis1-threadpool-numba-20260504T153536Z.json`:
it restored `final_threads` to baseline, but regressed wall time to 0.15s for
`diff` and 0.13s for `pct_change`.
