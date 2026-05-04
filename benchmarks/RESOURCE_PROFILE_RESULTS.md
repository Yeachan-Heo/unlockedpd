# Resource profile results

Latest verified broad run: `.omx/artifacts/broad-profile-axis1-dense-unrolled-final-20260504T114035Z.json`.

Command:

```bash
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 5 --output .omx/artifacts/broad-profile-axis1-dense-unrolled-final-20260504T114035Z.json
```

Validation paired with this optimization pass:

```bash
poetry run ruff check src/unlockedpd/ops/aggregations.py benchmarks/profile_resources.py tests/test_axis1_aggregation_fastpaths.py
PYTHONPATH=src poetry run pytest -q tests/test_axis1_aggregation_fastpaths.py tests/test_resource_profile.py tests/test_resource_profile_verification.py
```

The broad profile has `0` speedup-weighted CPU/RAM failures.  The budget is:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

## Current broad profile summary

| case | operation | speedup | CPU ratio | RSS ratio | optimized path |
| --- | ---: | ---: | ---: | ---: | --- |
| aggregation-axis1-wide-32mb | dataframe_std/var | ~40-43x | ~0.28x | ~0.25x | parallel_numba |
| rolling-axis1-wide-32mb | rolling_mean/sum | ~35-36x | ~0.29-0.31x | ~0.83-0.89x | parallel_numba |
| rolling-medium-100mb | rolling_mean/std | ~28-34x | ~0.43-0.52x | ~1.0x | parallel_numba |
| rolling-axis1-wide-32mb | rolling_std/var | ~26-31x | ~0.32-0.33x | ~0.84-0.88x | parallel_numba |
| rolling-wide-10mb | rolling_mean/sum | ~23-24x | ~0.61-0.63x | ~1.0x | parallel_numba |
| pairwise-safe-rolling-corr | rolling_corr/cov | ~16-18x | ~0.32-0.33x | ~0.91x | threadpool |
| expanding-wide-10mb | expanding_mean | ~19x | ~0.90x | ~0.50x | parallel_numba |
| aggregation-medium-100mb | dataframe_mean/sum | ~14-19x | ~1.7-2.8x | ~0.08x | parallel_numba |
| rolling-axis1-wide-32mb | rolling_min/max | ~11.8x | ~0.26x | ~0.96-0.97x | parallel_numba |
| cumulative-axis1-wide-32mb | cumsum/cumprod/cummin/cummax | ~11x | ~0.09x | ~1.0x | numpy_vectorized |
| rank-axis1-wide-32mb | rank_axis1 | ~11.3x | ~0.59x | ~1.12x | parallel_numba |
| aggregation-axis1-wide-32mb | dataframe_sum/mean/min/max | ~10.7-11.4x | ~1.1-2.1x | ~0.25x | parallel_numba |
| aggregation-wide-10mb | dataframe_mean/sum | ~2.2-2.3x | ~0.43-0.46x | ~0.79x | numpy_vectorized |
| transform-axis1-wide-32mb | diff/shift/pct_change | ~1x | ~1.0x | ~1.0x | pandas primitives/native |

## Optimization commits covered

| commit | area | benchmark evidence |
| --- | --- | --- |
| `85dca9e` | broad row-wise fast paths | earlier broad axis=1 profiles under `.omx/artifacts/` |
| `c728b47` | axis=1 rolling kernels and resource fixes | `.omx/artifacts/broad-profile-axis1-final-all-pass-20260504T091204Z.json` |
| `d6bc805` | rolling/expanding fast wrapping, stable pct axis=1 | `.omx/artifacts/broad-profile-fastwrap-primpct-20260504T093744Z.json` |
| `f14f259` | rank fast wrapping | `.omx/artifacts/broad-profile-rank-fastwrap-20260504T095206Z.json` |
| `35a568b` | large-frame axis=0 row-block reducers and 100MB aggregation profile cases | `.omx/artifacts/broad-profile-large-axis0-agg-rerun-20260504T100846Z.json` |
| `315bcc1` | row-block rolling/expanding kernels and rank cap widening | `.omx/artifacts/broad-profile-rowblocks-expanding-final-20260504T105429Z.json` |
| current commit | branch-free dense axis=1 aggregation reducers and non-intrusive profiler accounting | `.omx/artifacts/broad-profile-axis1-dense-unrolled-final-20260504T114035Z.json` |

## Remaining gap to the universal 10x objective

The latest profile is resource-clean and all rolling, expanding, pairwise,
cumulative, rank(axis=1 large), 100MB axis=0 aggregation, and 32MB axis=1
aggregation rows meet or exceed 10x.  Universal 10x is still **not** met:
axis=1 diff/shift/pct_change remain ~1x because the fastest correct paths are
pandas primitives/native, and the 10MB axis=0 sum/mean rows remain ~2.2x because
NumPy/pandas already complete them in ~1-2 ms.

Rejected experiments include low-threshold axis=0 row-block dispatch for the
10MB case, bounded Numba axis=1 transforms, NumExpr, Bottleneck, and a Rust
ctypes/Rayon prototype for axis=1 diff/pct/shift.  The Rust prototype improved
copying transforms only to roughly 2-6x locally and made axis=1 shift slower
than pandas' metadata-based native shift, so it was not adopted into the package.
