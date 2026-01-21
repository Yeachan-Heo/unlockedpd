"""
Benchmark all patched operations on >100MB DataFrames.

100MB = 12.5M float64 elements (100MB / 8 bytes)
We'll use 5000 rows x 2500 cols = 12.5M elements = 100MB
"""
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Create >100MB DataFrame (5000 x 2500 = 12.5M elements = 100MB)
ROWS = 5000
COLS = 2500
print(f"Creating {ROWS}x{COLS} DataFrame ({ROWS*COLS*8/1e6:.1f} MB)...")
np.random.seed(42)
data = np.random.randn(ROWS, COLS)
# Add some NaNs for realistic testing
data[::100, ::50] = np.nan

df_base = pd.DataFrame(data)
print(f"DataFrame created: {df_base.shape}, {df_base.values.nbytes/1e6:.1f} MB\n")

def benchmark(name, pandas_func, unlockedpd_func, n_runs=3):
    """Benchmark pandas vs unlockedpd and return speedup."""
    # Warmup
    try:
        pandas_func()
        unlockedpd_func()
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
        return None

    # Benchmark pandas
    pandas_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        pandas_func()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = min(pandas_times)

    # Benchmark unlockedpd
    unlockedpd_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        unlockedpd_func()
        unlockedpd_times.append(time.perf_counter() - start)
    unlockedpd_time = min(unlockedpd_times)

    speedup = pandas_time / unlockedpd_time if unlockedpd_time > 0 else 0
    status = "✓ PASS" if speedup >= 1.0 else "✗ FAIL"
    print(f"  {name}: pandas={pandas_time*1000:.1f}ms, unlockedpd={unlockedpd_time*1000:.1f}ms, speedup={speedup:.2f}x {status}")
    return speedup

results = {}

# Import unlockedpd to apply patches
import unlockedpd
unlockedpd.config.enabled = True

print("=" * 70)
print("BENCHMARKING ALL PATCHED OPERATIONS ON >100MB DATAFRAME")
print("=" * 70)

# ============================================================
# AGGREGATIONS
# ============================================================
print("\n[AGGREGATIONS]")
df = df_base.copy()

# Disable patches for pandas baseline
unlockedpd.unpatch_all()

for op in ['sum', 'mean', 'std', 'var', 'min', 'max', 'median', 'prod']:
    unlockedpd.unpatch_all()
    pandas_func = lambda op=op: getattr(df, op)()

    # Re-enable patches
    from unlockedpd.ops.aggregations import apply_aggregation_patches
    apply_aggregation_patches()
    unlockedpd_func = lambda op=op: getattr(df, op)()

    results[f'agg_{op}'] = benchmark(f"df.{op}()", pandas_func, unlockedpd_func)

# ============================================================
# FILLNA
# ============================================================
print("\n[FILLNA]")
df = df_base.copy()

unlockedpd.unpatch_all()
pandas_ffill = lambda: df.ffill()
pandas_bfill = lambda: df.bfill()
pandas_fillna = lambda: df.fillna(0.0)

from unlockedpd.ops.fillna import apply_fillna_patches
apply_fillna_patches()
unlockedpd_ffill = lambda: df.ffill()
unlockedpd_bfill = lambda: df.bfill()
unlockedpd_fillna = lambda: df.fillna(0.0)

unlockedpd.unpatch_all()
results['fillna_ffill'] = benchmark("df.ffill()", pandas_ffill, unlockedpd_ffill)
results['fillna_bfill'] = benchmark("df.bfill()", pandas_bfill, unlockedpd_bfill)
results['fillna_scalar'] = benchmark("df.fillna(0)", pandas_fillna, unlockedpd_fillna)

# ============================================================
# ELEMENT-WISE
# ============================================================
print("\n[ELEMENT-WISE]")
df = df_base.abs().copy()  # Ensure positive for clip

unlockedpd.unpatch_all()
pandas_clip = lambda: df.clip(lower=-1, upper=1)
pandas_abs = lambda: df.abs()
pandas_round = lambda: df.round(2)

from unlockedpd.ops.element_wise import apply_element_wise_patches
apply_element_wise_patches()
unlockedpd_clip = lambda: df.clip(lower=-1, upper=1)
unlockedpd_abs = lambda: df.abs()
unlockedpd_round = lambda: df.round(2)

unlockedpd.unpatch_all()
results['elem_clip'] = benchmark("df.clip()", pandas_clip, unlockedpd_clip)
results['elem_abs'] = benchmark("df.abs()", pandas_abs, unlockedpd_abs)
results['elem_round'] = benchmark("df.round()", pandas_round, unlockedpd_round)

# ============================================================
# CORRELATION
# ============================================================
print("\n[CORRELATION]")
# Use smaller subset for correlation (it's O(n*m^2))
df_small = df_base.iloc[:, :100].copy()

unlockedpd.unpatch_all()
pandas_corr = lambda: df_small.corr()
pandas_cov = lambda: df_small.cov()

from unlockedpd.ops.correlation import apply_correlation_patches
apply_correlation_patches()
unlockedpd_corr = lambda: df_small.corr()
unlockedpd_cov = lambda: df_small.cov()

unlockedpd.unpatch_all()
results['corr'] = benchmark("df.corr()", pandas_corr, unlockedpd_corr)
results['cov'] = benchmark("df.cov()", pandas_cov, unlockedpd_cov)

# ============================================================
# QUANTILE
# ============================================================
print("\n[QUANTILE]")
df = df_base.copy()

unlockedpd.unpatch_all()
pandas_quantile = lambda: df.quantile(0.5)
pandas_quantile_multi = lambda: df.quantile([0.25, 0.5, 0.75])

from unlockedpd.ops.quantile import apply_quantile_patches
apply_quantile_patches()
unlockedpd_quantile = lambda: df.quantile(0.5)
unlockedpd_quantile_multi = lambda: df.quantile([0.25, 0.5, 0.75])

unlockedpd.unpatch_all()
results['quantile_single'] = benchmark("df.quantile(0.5)", pandas_quantile, unlockedpd_quantile)
results['quantile_multi'] = benchmark("df.quantile([.25,.5,.75])", pandas_quantile_multi, unlockedpd_quantile_multi)

# ============================================================
# ROLLING
# ============================================================
print("\n[ROLLING]")
df = df_base.copy()

unlockedpd.unpatch_all()
pandas_roll_mean = lambda: df.rolling(20).mean()
pandas_roll_sum = lambda: df.rolling(20).sum()
pandas_roll_std = lambda: df.rolling(20).std()

from unlockedpd.ops.rolling import apply_rolling_patches
apply_rolling_patches()
unlockedpd_roll_mean = lambda: df.rolling(20).mean()
unlockedpd_roll_sum = lambda: df.rolling(20).sum()
unlockedpd_roll_std = lambda: df.rolling(20).std()

unlockedpd.unpatch_all()
results['rolling_mean'] = benchmark("df.rolling(20).mean()", pandas_roll_mean, unlockedpd_roll_mean)
results['rolling_sum'] = benchmark("df.rolling(20).sum()", pandas_roll_sum, unlockedpd_roll_sum)
results['rolling_std'] = benchmark("df.rolling(20).std()", pandas_roll_std, unlockedpd_roll_std)

# ============================================================
# EXPANDING
# ============================================================
print("\n[EXPANDING]")
df = df_base.copy()

unlockedpd.unpatch_all()
pandas_exp_mean = lambda: df.expanding().mean()
pandas_exp_sum = lambda: df.expanding().sum()

from unlockedpd.ops.expanding import apply_expanding_patches
apply_expanding_patches()
unlockedpd_exp_mean = lambda: df.expanding().mean()
unlockedpd_exp_sum = lambda: df.expanding().sum()

unlockedpd.unpatch_all()
results['expanding_mean'] = benchmark("df.expanding().mean()", pandas_exp_mean, unlockedpd_exp_mean)
results['expanding_sum'] = benchmark("df.expanding().sum()", pandas_exp_sum, unlockedpd_exp_sum)

# ============================================================
# TRANSFORM (diff, pct_change, shift)
# ============================================================
print("\n[TRANSFORM]")
df = df_base.copy()

unlockedpd.unpatch_all()
pandas_diff = lambda: df.diff()
pandas_pct = lambda: df.pct_change()
pandas_shift = lambda: df.shift(1)

from unlockedpd.ops.transform import apply_transform_patches
apply_transform_patches()
unlockedpd_diff = lambda: df.diff()
unlockedpd_pct = lambda: df.pct_change()
unlockedpd_shift = lambda: df.shift(1)

unlockedpd.unpatch_all()
results['transform_diff'] = benchmark("df.diff()", pandas_diff, unlockedpd_diff)
results['transform_pct_change'] = benchmark("df.pct_change()", pandas_pct, unlockedpd_pct)
results['transform_shift'] = benchmark("df.shift()", pandas_shift, unlockedpd_shift)

# ============================================================
# RANK
# ============================================================
print("\n[RANK]")
df = df_base.copy()

unlockedpd.unpatch_all()
pandas_rank0 = lambda: df.rank(axis=0)
pandas_rank1 = lambda: df.rank(axis=1)

from unlockedpd.ops.rank import apply_rank_patches
apply_rank_patches()
unlockedpd_rank0 = lambda: df.rank(axis=0)
unlockedpd_rank1 = lambda: df.rank(axis=1)

unlockedpd.unpatch_all()
results['rank_axis0'] = benchmark("df.rank(axis=0)", pandas_rank0, unlockedpd_rank0)
results['rank_axis1'] = benchmark("df.rank(axis=1)", pandas_rank1, unlockedpd_rank1)

# ============================================================
# STATS (skew, kurt, sem)
# ============================================================
print("\n[STATS]")
df = df_base.copy()

# These are already in pandas, just testing our optimized versions
# Note: stats operations may not be patched by default

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for v in results.values() if v is not None and v >= 1.0)
failed = sum(1 for v in results.values() if v is not None and v < 1.0)
errors = sum(1 for v in results.values() if v is None)

print(f"\nTotal: {len(results)} operations")
print(f"  PASSED (speedup >= 1.0x): {passed}")
print(f"  FAILED (speedup < 1.0x): {failed}")
print(f"  ERRORS: {errors}")

if failed > 0:
    print("\nFailed operations (slower than pandas):")
    for name, speedup in results.items():
        if speedup is not None and speedup < 1.0:
            print(f"  - {name}: {speedup:.2f}x")

print("\nTop performers:")
sorted_results = sorted([(k, v) for k, v in results.items() if v is not None], key=lambda x: -x[1])
for name, speedup in sorted_results[:5]:
    print(f"  - {name}: {speedup:.2f}x")

# Exit with status
exit_code = 0 if failed == 0 and errors == 0 else 1
print(f"\nExit code: {exit_code}")
exit(exit_code)
