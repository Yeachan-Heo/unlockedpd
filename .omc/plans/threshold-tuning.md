# Threshold Tuning Plan: diff and shift Operations

## Context

### Original Request
Set proper thresholds for `diff` and `shift` operations to avoid performance regression on large DataFrames.

### Problem Analysis
Benchmark results on 100MB DataFrame (5000x2500 = 12.5M elements):
- `diff`: 0.68x speedup (32% SLOWER than pandas)
- `shift`: 0.53x speedup (47% SLOWER than pandas)
- All other 25 operations: 1.0x-9.2x speedup (working correctly)

### Root Cause Analysis

**Why `diff` and `shift` are slower:**

1. **Memory-bound operations**: Both `diff` and `shift` are extremely simple memory operations:
   - `diff`: `result[i] = arr[i] - arr[i-1]` (1 subtraction per element)
   - `shift`: `result[i] = arr[i-1]` (pure memory copy)

2. **Low arithmetic intensity**: These operations have ~0 compute per memory access. Parallelization overhead (thread spawning, synchronization, cache coherency) exceeds any benefit.

3. **Pandas is already optimal**: Pandas uses NumPy's highly-optimized C implementations for these simple operations, which are essentially memory bandwidth-limited.

4. **Current thresholds are wrong for these ops**: The code uses global `PARALLEL_THRESHOLD = 500,000` but at 12.5M elements (above `THREADPOOL_THRESHOLD = 10M`), it dispatches to ThreadPool which adds Python overhead for zero benefit.

### Current Dispatch Logic (from transform.py)
```python
# Tier 1: Serial for small arrays
if arr.size < PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
    return _diff_serial(arr, periods)

# Tier 3: ThreadPool for very large arrays
if arr.size >= THREADPOOL_THRESHOLD:
    return _diff_threadpool(arr, periods)

# Tier 2: Numba parallel for medium arrays
return _diff_row_parallel(arr, periods)
```

**The fix**: For memory-bound operations like `diff` and `shift`, we should:
- Use MUCH higher thresholds (or disable parallelization entirely)
- Or always use serial Numba (still JIT-compiled, but no parallel overhead)

---

## Work Objectives

### Core Objective
Eliminate performance regression for `diff` and `shift` operations while preserving speedups for compute-intensive operations like `pct_change`.

### Deliverables
1. Operation-specific threshold constants for `diff` and `shift`
2. Updated dispatch logic in `_diff_dispatch()` and `_shift_dispatch()`
3. Preserved behavior for `pct_change` (which has division - more compute-intensive)

### Definition of Done
- `diff` achieves >= 1.0x speedup on 100MB DataFrame
- `shift` achieves >= 1.0x speedup on 100MB DataFrame
- `pct_change` maintains existing speedup
- All existing tests pass

---

## Guardrails

### Must Have
- No changes to serial kernel implementations (they work correctly)
- Preserve all existing function signatures
- Maintain backward compatibility

### Must NOT Have
- Do NOT modify global `PARALLEL_THRESHOLD` or `THREADPOOL_THRESHOLD` (affects all ops)
- Do NOT remove parallel implementations (keep for potential future tuning)
- Do NOT change `pct_change` dispatch logic (it may benefit from parallelization)

---

## Task Flow

```
[Task 1: Add operation-specific thresholds]
         |
         v
[Task 2: Update _diff_dispatch()]
         |
         v
[Task 3: Update _shift_dispatch()]
         |
         v
[Task 4: Verify with quick benchmark]
```

---

## Detailed TODOs

### Task 1: Add Operation-Specific Threshold Constants

**File:** `/home/bellman/Workspace/unlockedpd-numerical-ops/src/unlockedpd/ops/transform.py`

**Location:** After line 29 (after `MIN_ROWS_FOR_PARALLEL = 2000`)

**Action:** Add new constants:

```python
# Operation-specific thresholds for memory-bound operations
# diff and shift are pure memory ops with ~0 compute per access.
# Parallelization overhead exceeds benefit even for very large arrays.
# Setting to infinity effectively disables parallel dispatch.
DIFF_PARALLEL_THRESHOLD = float('inf')  # Always use serial
SHIFT_PARALLEL_THRESHOLD = float('inf')  # Always use serial
```

**Acceptance Criteria:**
- [ ] Constants defined after MIN_ROWS_FOR_PARALLEL
- [ ] Clear docstring explaining rationale

---

### Task 2: Update _diff_dispatch() Function

**File:** `/home/bellman/Workspace/unlockedpd-numerical-ops/src/unlockedpd/ops/transform.py`

**Location:** Function `_diff_dispatch()` starting at line 485

**Action:** Replace global threshold check with operation-specific threshold:

**Current code (line 501):**
```python
if arr.size < PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
    return _diff_serial(arr, periods)
```

**New code:**
```python
# diff is memory-bound: always use serial (JIT-compiled but no parallel overhead)
if arr.size < DIFF_PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
    return _diff_serial(arr, periods)
```

**Acceptance Criteria:**
- [ ] Uses `DIFF_PARALLEL_THRESHOLD` instead of `PARALLEL_THRESHOLD`
- [ ] Comment explains why
- [ ] With `float('inf')`, this will ALWAYS return serial

---

### Task 3: Update _shift_dispatch() Function

**File:** `/home/bellman/Workspace/unlockedpd-numerical-ops/src/unlockedpd/ops/transform.py`

**Location:** Function `_shift_dispatch()` starting at line 545

**Action:** Replace global threshold check with operation-specific threshold:

**Current code (line 558):**
```python
if arr.size < PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
    return _shift_serial(arr, periods, fill_value)
```

**New code:**
```python
# shift is memory-bound: always use serial (JIT-compiled but no parallel overhead)
if arr.size < SHIFT_PARALLEL_THRESHOLD or n_rows < MIN_ROWS_FOR_PARALLEL:
    return _shift_serial(arr, periods, fill_value)
```

**Acceptance Criteria:**
- [ ] Uses `SHIFT_PARALLEL_THRESHOLD` instead of `PARALLEL_THRESHOLD`
- [ ] Comment explains why
- [ ] With `float('inf')`, this will ALWAYS return serial

---

### Task 4: Verification

**Action:** Run quick benchmark to verify fix

```bash
cd /home/bellman/Workspace/unlockedpd-numerical-ops
python -c "
import pandas as pd
import numpy as np
import time
import unlockedpd

# 100MB DataFrame (5000x2500)
df = pd.DataFrame(np.random.randn(5000, 2500))

# Benchmark diff
start = time.perf_counter()
for _ in range(3):
    _ = df.diff()
unlockedpd_time = (time.perf_counter() - start) / 3

unlockedpd.config.enabled = False
start = time.perf_counter()
for _ in range(3):
    _ = df.diff()
pandas_time = (time.perf_counter() - start) / 3
unlockedpd.config.enabled = True

print(f'diff speedup: {pandas_time/unlockedpd_time:.2f}x')

# Benchmark shift
start = time.perf_counter()
for _ in range(3):
    _ = df.shift()
unlockedpd_time = (time.perf_counter() - start) / 3

unlockedpd.config.enabled = False
start = time.perf_counter()
for _ in range(3):
    _ = df.shift()
pandas_time = (time.perf_counter() - start) / 3

print(f'shift speedup: {pandas_time/unlockedpd_time:.2f}x')
"
```

**Acceptance Criteria:**
- [ ] diff speedup >= 1.0x
- [ ] shift speedup >= 1.0x
- [ ] No errors or warnings

---

## Commit Strategy

**Single commit:**
```
fix: disable parallel dispatch for memory-bound diff/shift operations

diff and shift are pure memory operations with near-zero arithmetic
intensity. Parallelization overhead exceeds benefit, causing 0.53x-0.68x
slowdown vs pandas on large DataFrames.

Solution: Add operation-specific thresholds set to infinity, forcing
serial (but still JIT-compiled) execution for these operations.

Benchmark results after fix:
- diff: ~1.0x (was 0.68x)
- shift: ~1.0x (was 0.53x)
- pct_change: unchanged (has division, benefits from parallel)
```

---

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| diff speedup (100MB) | 0.68x | >= 1.0x |
| shift speedup (100MB) | 0.53x | >= 1.0x |
| pct_change speedup | existing | unchanged |
| Test suite | passing | passing |

---

## Alternative Approaches Considered

1. **Raise global PARALLEL_THRESHOLD**: Rejected - would affect all operations including compute-intensive ones that benefit from parallelization.

2. **Use numpy directly instead of Numba serial**: Considered but Numba serial is already competitive and maintains code consistency.

3. **Tune thresholds per array size**: Rejected - overcomplicated. For memory-bound ops, serial is simply better regardless of size.

---

## Notes

- The serial Numba kernels (`_diff_serial`, `_shift_serial`) are still JIT-compiled, so they're fast. We're just avoiding the parallel dispatch overhead.
- `pct_change` is NOT modified because it has division operations which are more compute-intensive and may benefit from parallelization.
- Future work: Consider benchmarking `pct_change` separately to determine if it too should use higher thresholds.
