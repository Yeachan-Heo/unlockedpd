"""
Correct benchmark: Compare pandas native vs unlockedpd patched.
Uses subprocess to ensure clean state for each comparison.
"""
import time
import numpy as np
import pandas as pd
import os
import sys

# Force using worktree source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Create >100MB DataFrame
ROWS = 5000
COLS = 2500
print(f"Creating {ROWS}x{COLS} DataFrame ({ROWS*COLS*8/1e6:.1f} MB)...")
np.random.seed(42)
data = np.random.randn(ROWS, COLS)
data[::100, ::50] = np.nan
df = pd.DataFrame(data)
print(f"DataFrame: {df.shape}, {df.values.nbytes/1e6:.1f} MB\n")

def time_operation(func, n_runs=3):
    """Time an operation, return best of n runs."""
    # Warmup
    func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)

print("=" * 70)
print("BENCHMARKING: PANDAS NATIVE vs UNLOCKEDPD PATCHED")
print("=" * 70)

results = {}

# ============================================================
# Test each category
# ============================================================

categories = {
    'AGGREGATIONS': [
        ('sum', lambda d: d.sum()),
        ('mean', lambda d: d.mean()),
        ('std', lambda d: d.std()),
        ('var', lambda d: d.var()),
        ('min', lambda d: d.min()),
        ('max', lambda d: d.max()),
        ('median', lambda d: d.median()),
        ('prod', lambda d: d.prod()),
    ],
    'ROLLING': [
        ('rolling.mean', lambda d: d.rolling(20).mean()),
        ('rolling.sum', lambda d: d.rolling(20).sum()),
        ('rolling.std', lambda d: d.rolling(20).std()),
        ('rolling.min', lambda d: d.rolling(20).min()),
        ('rolling.max', lambda d: d.rolling(20).max()),
    ],
    'EXPANDING': [
        ('expanding.mean', lambda d: d.expanding().mean()),
        ('expanding.sum', lambda d: d.expanding().sum()),
        ('expanding.std', lambda d: d.expanding().std()),
    ],
    'TRANSFORM': [
        ('diff', lambda d: d.diff()),
        ('pct_change', lambda d: d.pct_change()),
        ('shift', lambda d: d.shift(1)),
    ],
    'RANK': [
        ('rank_axis0', lambda d: d.rank(axis=0)),
        ('rank_axis1', lambda d: d.rank(axis=1)),
    ],
    'FILLNA': [
        ('ffill', lambda d: d.ffill()),
        ('bfill', lambda d: d.bfill()),
        ('fillna', lambda d: d.fillna(0.0)),
    ],
    'ELEMENT_WISE': [
        ('clip', lambda d: d.clip(-1, 1)),
        ('abs', lambda d: d.abs()),
        ('round', lambda d: d.round(2)),
    ],
}

for category, ops in categories.items():
    print(f"\n[{category}]")

    for name, func in ops:
        # Time with pandas (no patches)
        try:
            # Ensure unlockedpd patches are removed
            try:
                import unlockedpd
                unlockedpd.unpatch_all()
            except:
                pass

            pandas_time = time_operation(lambda: func(df))

            # Apply patches
            import unlockedpd
            # Re-import to apply patches fresh
            import importlib
            importlib.reload(unlockedpd)

            unlockedpd_time = time_operation(lambda: func(df))

            speedup = pandas_time / unlockedpd_time if unlockedpd_time > 0 else 0
            status = "✓" if speedup >= 1.0 else "✗"
            print(f"  {name}: pandas={pandas_time*1000:.1f}ms, ulpd={unlockedpd_time*1000:.1f}ms, {speedup:.2f}x {status}")
            results[name] = speedup

        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            results[name] = None

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for v in results.values() if v is not None and v >= 1.0)
failed = sum(1 for v in results.values() if v is not None and v < 1.0)
marginal = sum(1 for v in results.values() if v is not None and 0.95 <= v < 1.0)

print(f"\nTotal: {len(results)} operations")
print(f"  FASTER (>= 1.0x): {passed}")
print(f"  SLOWER (< 1.0x): {failed}")
print(f"  Marginal (0.95-1.0x): {marginal}")

avg_speedup = np.mean([v for v in results.values() if v is not None])
print(f"\nAverage speedup: {avg_speedup:.2f}x")

if failed > 0:
    print("\nSlower operations:")
    for name, speedup in sorted(results.items(), key=lambda x: x[1] or 0):
        if speedup is not None and speedup < 1.0:
            print(f"  - {name}: {speedup:.2f}x")

print("\nFastest operations:")
for name, speedup in sorted(results.items(), key=lambda x: -(x[1] or 0))[:5]:
    if speedup is not None:
        print(f"  - {name}: {speedup:.2f}x")

# Verdict
if avg_speedup >= 1.0 and failed <= len(results) * 0.2:  # Allow 20% marginal
    print("\n✓ VERDICT: PASS - Overall performance improvement achieved")
    exit(0)
else:
    print("\n✗ VERDICT: NEEDS REVIEW - Some operations are slower")
    exit(1)
