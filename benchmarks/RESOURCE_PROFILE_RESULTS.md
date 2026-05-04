# Resource profile results

Latest verified run: `.omx/artifacts/broad-profile-rowblocks-expanding-final-20260504T105429Z.json`.

Command:

```bash
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 5 --output .omx/artifacts/broad-profile-rowblocks-expanding-final-20260504T105429Z.json
```

Validation paired with the run:

```bash
poetry run ruff check src/unlockedpd/ops/rolling.py src/unlockedpd/ops/expanding.py src/unlockedpd/ops/rank.py src/unlockedpd/ops/aggregations.py tests/test_rolling.py tests/test_axis1_aggregation_fastpaths.py
PYTHONPATH=src poetry run pytest -q
PYTHONPATH=src poetry run python -m compileall -q src tests benchmarks
```

Result: `0` speedup-weighted CPU/RAM failures in the broad profile.  The
speedup-weighted budget is:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

## Current broad profile summary

| case | operation | speedup | CPU ratio | RSS ratio | optimized path |
| --- | ---: | ---: | ---: | ---: | --- |
| rolling-axis1-wide-32mb | rolling_mean/sum | ~33-36x | ~0.30-0.32x | ~0.93-0.97x | parallel_numba |
| rolling-medium-100mb | rolling_mean/std | ~24-29x | ~0.44-0.53x | ~1.0x | parallel_numba |
| rolling-wide-10mb | rolling_mean/sum | ~12x | ~1.2-1.3x | ~1.0x | parallel_numba |
| rolling-axis1-wide-32mb | rolling_std/var/min/max | ~13-28x | ~0.23-0.34x | ~0.96-0.99x | parallel_numba |
| pairwise-safe-rolling-corr | rolling_corr/cov | ~18-19x | ~0.26-0.28x | ~0.89-0.92x | threadpool |
| cumulative-axis1-wide-32mb | cumsum/cumprod/cummin/cummax | ~10-11x | <0.1x | ~1.0x | numpy_vectorized |
| rank-axis1-wide-32mb | rank_axis1 | ~10.7x | ~0.61x | ~1.12x | parallel_numba |
| expanding-wide-10mb | expanding_mean | ~10.1x | ~1.55x | ~0.51x | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_std/var | ~6x | ~4.9-5.1x | ~0.04x | parallel_numba |
| aggregation-medium-100mb | dataframe_mean/sum | ~5.2-5.3x | ~2.8-3.0x | ~0.014x | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_sum/mean/min/max | ~1.9-2.3x | ~0.40-0.51x | ~0.04x | numpy_vectorized |
| transform-axis1-wide-32mb | diff/shift/pct_change | ~1x | ~1.0x | ~1.0x | pandas primitives/native |
| aggregation-wide-10mb | dataframe_mean/sum | ~0.84-0.90x | ~0.73-0.77x | ~0.12x | numpy_vectorized |

## Optimization commits covered

| commit | area | benchmark evidence |
| --- | --- | --- |
| `85dca9e` | broad row-wise fast paths | earlier broad axis=1 profiles under `.omx/artifacts/` |
| `c728b47` | axis=1 rolling kernels and resource fixes | `.omx/artifacts/broad-profile-axis1-final-all-pass-20260504T091204Z.json` |
| `d6bc805` | rolling/expanding fast wrapping, stable pct axis=1 | `.omx/artifacts/broad-profile-fastwrap-primpct-20260504T093744Z.json` |
| `f14f259` | rank fast wrapping | `.omx/artifacts/broad-profile-rank-fastwrap-20260504T095206Z.json` |
| `35a568b` | large-frame axis=0 row-block reducers and 100MB aggregation profile cases | `.omx/artifacts/broad-profile-large-axis0-agg-rerun-20260504T100846Z.json` |
| current commit | row-block rolling sum/mean/std/var, row-block expanding sum/mean, rank cap widening | `.omx/artifacts/broad-profile-rowblocks-expanding-final-20260504T105429Z.json` |

## Remaining gap to the 10x objective

The latest profile is resource-clean but does **not** satisfy universal 10x
speedup.  Rolling, pairwise, cumulative, rank(axis=1), and expanding mean now
meet or exceed 10x in the broad profile.  Remaining below-target families are
simple axis=0 aggregations, simple axis=1 aggregations, and axis=1
diff/shift/pct_change.  These remaining rows are already single-digit
millisecond pandas/NumPy primitives in the benchmark, so DataFrame dispatch and
result wrapping dominate.  Rejected experiments include low-threshold row-block
axis=0 aggregation, bounded Numba axis=1 transforms, Bottleneck, and wider
axis=1 aggregation thread caps that failed the speedup-weighted CPU budget.
Rust/PyO3 remains permissible but was not adopted because the benchmarked
Numba row-block rewrites delivered the verified wins without adding build-system
risk.
