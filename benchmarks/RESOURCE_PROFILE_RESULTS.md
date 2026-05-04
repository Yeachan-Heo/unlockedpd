# Resource profile results

Latest verified run: `.omx/artifacts/broad-profile-large-axis0-agg-rerun-20260504T100846Z.json`.

Command:

```bash
PYTHONPATH=src poetry run python benchmarks/profile_resources.py --repeats 5 --output .omx/artifacts/broad-profile-large-axis0-agg-rerun-20260504T100846Z.json
```

Validation paired with the run:

```bash
poetry run ruff check src/unlockedpd/ops/aggregations.py benchmarks/profile_resources.py benchmarks/compare_resource_profiles.py
PYTHONPATH=src poetry run pytest -q
PYTHONPATH=src poetry run python -m compileall -q src tests benchmarks
```

Result: `0` speedup-weighted CPU/RAM failures.  The speedup-weighted budget is:

```text
optimized_cpu_ratio <= speedup * 1.2
optimized_rss_ratio <= speedup * 1.2
```

## Current broad profile summary

| case | operation | speedup | CPU ratio | RSS ratio | optimized path |
| --- | ---: | ---: | ---: | ---: | --- |
| rolling-axis1-wide-32mb | rolling_sum | ~24x | <0.2x | ~1.0x | parallel_numba |
| rolling-axis1-wide-32mb | rolling_mean | ~23x | <0.2x | ~1.0x | parallel_numba |
| pairwise-safe-rolling-corr | rolling_corr/cov | ~18-19x | <0.3x | ~0.9x | threadpool |
| cumulative-axis1-wide-32mb | cumsum/cumprod/cummin/cummax | ~10-11x | <0.1x | ~1.0x | numpy_vectorized |
| rank-axis1-wide-32mb | rank_axis1 | ~8.3x | ~0.46x | ~1.0x | parallel_numba |
| aggregation-medium-100mb | dataframe_mean/sum | ~5.7x | ~0.6-0.7x | ~0.015x | parallel_numba |
| rolling-medium-100mb | rolling_mean/std | ~4-6x | <1.0x | ~1.0x | threadpool / parallel_numba |
| expanding-wide-10mb | expanding_mean | ~5x | ~1.4x | ~0.5x | parallel_numba |
| transform-axis1-wide-32mb | diff/shift/pct_change | ~1x | ~1.0x | ~1.0x | pandas primitives/native |
| aggregation-wide-10mb | dataframe_mean/sum | ~0.9x | ~0.7x | ~0.13x | numpy_vectorized |

## Optimization commits covered

| commit | area | benchmark evidence |
| --- | --- | --- |
| `85dca9e` | broad row-wise fast paths | earlier broad axis=1 profiles under `.omx/artifacts/` |
| `c728b47` | axis=1 rolling kernels and resource fixes | `.omx/artifacts/broad-profile-axis1-final-all-pass-20260504T091204Z.json` |
| `d6bc805` | rolling/expanding fast wrapping, stable pct axis=1 | `.omx/artifacts/broad-profile-fastwrap-primpct-20260504T093744Z.json` |
| `f14f259` | rank fast wrapping | `.omx/artifacts/broad-profile-rank-fastwrap-20260504T095206Z.json` |
| `35a568b` | large-frame axis=0 row-block reducers and 100MB aggregation profile cases | `.omx/artifacts/broad-profile-large-axis0-agg-rerun-20260504T100846Z.json` |

## Remaining gap to the 10x objective

The latest profile is resource-clean but does **not** satisfy universal 10x speedup.  Remaining below-target families are axis=0 aggregations, axis=1 transforms, axis=0 rolling/expanding, and rank.  Rejected experiments include low-threshold row-block aggregation, bounded Numba axis=1 transforms, Bottleneck, and wider rolling thread caps because they were slower or failed the speedup-weighted CPU/RAM budget.
