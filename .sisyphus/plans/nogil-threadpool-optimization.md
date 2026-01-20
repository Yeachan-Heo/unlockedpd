# Work Plan: @njit(nogil=True) + ThreadPool Optimization

## Context

### Original Request
Apply `@njit(nogil=True)` + ThreadPool optimization pattern to all unlockedpd operations for maximum performance.

### Key Benchmark Finding
```
ThreadPool + Numba nogil: 0.456s (4.7x faster than pandas)
ThreadPool + NumPy:       0.517s (4.1x faster)
Numba prange:             1.748s (1.2x faster)
Pandas baseline:          2.130s
```

### Why This Works
1. `@njit(nogil=True)` explicitly releases Python's Global Interpreter Lock
2. ThreadPoolExecutor threads can then run Numba-compiled code truly in parallel
3. Numba's fast scalar loop (~0.22ms/col) beats NumPy (~0.98ms/col) per-column overhead
4. Result: Best of both worlds - Numba's compilation speed + ThreadPool's true parallelism

### Technical Pattern
```python
# NEW: Add nogil kernel (single-threaded, GIL-released)
@njit(nogil=True)
def _operation_nogil_chunk(arr, result, start_col, end_col, ...params):
    """Process a chunk of columns without holding GIL."""
    for c in range(start_col, end_col):
        # Per-column computation
        for i in range(arr.shape[0]):
            # Row-wise logic
            ...

# UPDATED: ThreadPool calls nogil kernel instead of inline Python
def _operation_threadpool(arr, ...params):
    result = np.empty_like(arr)
    n_cols = arr.shape[1]
    chunk_size = (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS

    def process(args):
        s, e = args
        _operation_nogil_chunk(arr, result, s, e, ...params)  # GIL released!

    chunks = [(i*chunk_size, min((i+1)*chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS)]
    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as ex:
        list(ex.map(process, chunks))
    return result
```

---

## Work Objectives

### Core Objective
Achieve consistent 4.7x+ speedup over pandas for all rolling and expanding operations by replacing ThreadPool's Python/NumPy code with nogil-compiled Numba kernels.

### Deliverables
1. Rolling operations with nogil kernels: mean, sum, std, var, min, max, skew, kurt, count
2. Expanding operations with nogil kernels: mean, sum, std, var, min, max, skew, kurt, count
3. Updated warmup module for new nogil functions
4. All existing tests passing
5. Benchmark validation showing 4.5x+ speedup

### Definition of Done
- [ ] All rolling ThreadPool functions use nogil kernels
- [ ] All expanding ThreadPool functions use nogil kernels
- [ ] Warmup functions trigger compilation of new nogil kernels
- [ ] All tests in `tests/test_rolling.py` pass
- [ ] All tests in `tests/test_expanding.py` pass
- [ ] Benchmark shows >= 4.5x speedup vs pandas baseline

---

## Guardrails

### MUST Have
- Maintain exact numerical correctness (match pandas output)
- Handle NaN values correctly (use fallback path when NaN present)
- Preserve min_periods behavior
- Preserve ddof parameter for std/var operations
- Keep serial/parallel/threadpool dispatch logic intact
- Maintain backward compatibility with existing API

### MUST NOT Have
- Breaking changes to public API
- Regressions in numerical accuracy
- Memory leaks or increased memory usage
- Changes to non-threadpool code paths (serial, prange parallel)
- Removal of NaN handling capability

---

## Task Flow

```
Phase 1: Rolling Operations (rolling.py)
    |
    +-> 1.1 Add nogil kernels for basic ops (mean, sum)
    +-> 1.2 Add nogil kernels for statistical ops (std, var)
    +-> 1.3 Add nogil kernels for extrema ops (min, max)
    +-> 1.4 Add nogil kernels for moment ops (skew, kurt, count)
    +-> 1.5 Update all _threadpool functions to use nogil kernels
    |
Phase 2: Expanding Operations (expanding.py)
    |
    +-> 2.1 Add nogil kernels for basic ops (mean, sum)
    +-> 2.2 Add nogil kernels for statistical ops (std, var)
    +-> 2.3 Add nogil kernels for extrema ops (min, max)
    +-> 2.4 Add nogil kernels for moment ops (skew, kurt, count)
    +-> 2.5 Update all _threadpool functions to use nogil kernels
    |
Phase 3: Warmup & Validation
    |
    +-> 3.1 Update _warmup.py with nogil function warmup
    +-> 3.2 Run test suite
    +-> 3.3 Run benchmarks
    +-> 3.4 Document performance gains
```

---

## Detailed TODOs

### Phase 1: Rolling Operations (`src/unlockedpd/ops/rolling.py`)

#### TODO 1.1: Add nogil kernels for basic rolling ops
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py`

Add after the serial function definitions (around line 483):

```python
# ============================================================================
# Nogil kernels for ThreadPool (GIL-released for true parallelism)
# ============================================================================

@njit(nogil=True, cache=True)
def _rolling_mean_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Process chunk of columns for rolling mean - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cumsum = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                cumsum += val
                count += 1
            if row >= window:
                old_val = arr[row - window, c]
                if not np.isnan(old_val):
                    cumsum -= old_val
                    count -= 1
            if count >= min_periods:
                result[row, c] = cumsum / count
            else:
                result[row, c] = np.nan

@njit(nogil=True, cache=True)
def _rolling_sum_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Process chunk of columns for rolling sum - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cumsum = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                cumsum += val
                count += 1
            if row >= window:
                old_val = arr[row - window, c]
                if not np.isnan(old_val):
                    cumsum -= old_val
                    count -= 1
            if count >= min_periods:
                result[row, c] = cumsum
            else:
                result[row, c] = np.nan
```

**Acceptance Criteria**:
- Functions compile with `@njit(nogil=True)`
- Handle NaN values correctly
- Process column range [start_col, end_col)

---

#### TODO 1.2: Add nogil kernels for statistical rolling ops
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py`

```python
@njit(nogil=True, cache=True)
def _rolling_std_nogil_chunk(arr, result, start_col, end_col, window, min_periods, ddof):
    """Rolling std using Welford's algorithm - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        count = 0
        mean = 0.0
        M2 = 0.0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2
            if row >= window:
                old_val = arr[row - window, c]
                if not np.isnan(old_val):
                    count -= 1
                    if count > 0:
                        delta = old_val - mean
                        mean -= delta / count
                        delta2 = old_val - mean
                        M2 -= delta * delta2
                    else:
                        mean = 0.0
                        M2 = 0.0
            if count >= min_periods and count > ddof:
                result[row, c] = np.sqrt(M2 / (count - ddof))
            else:
                result[row, c] = np.nan

@njit(nogil=True, cache=True)
def _rolling_var_nogil_chunk(arr, result, start_col, end_col, window, min_periods, ddof):
    """Rolling variance using Welford's algorithm - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        count = 0
        mean = 0.0
        M2 = 0.0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2
            if row >= window:
                old_val = arr[row - window, c]
                if not np.isnan(old_val):
                    count -= 1
                    if count > 0:
                        delta = old_val - mean
                        mean -= delta / count
                        delta2 = old_val - mean
                        M2 -= delta * delta2
                    else:
                        mean = 0.0
                        M2 = 0.0
            if count >= min_periods and count > ddof:
                result[row, c] = M2 / (count - ddof)
            else:
                result[row, c] = np.nan
```

**Acceptance Criteria**:
- Uses numerically stable Welford's algorithm
- Handles ddof parameter correctly
- Matches existing test expectations

---

#### TODO 1.3: Add nogil kernels for extrema rolling ops
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py`

```python
@njit(nogil=True, cache=True)
def _rolling_min_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Rolling min - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        for row in range(n_rows):
            if row < min_periods - 1:
                result[row, c] = np.nan
                continue
            start = max(0, row - window + 1)
            min_val = np.inf
            count = 0
            for k in range(start, row + 1):
                val = arr[k, c]
                if not np.isnan(val):
                    if val < min_val:
                        min_val = val
                    count += 1
            if count >= min_periods:
                result[row, c] = min_val
            else:
                result[row, c] = np.nan

@njit(nogil=True, cache=True)
def _rolling_max_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Rolling max - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        for row in range(n_rows):
            if row < min_periods - 1:
                result[row, c] = np.nan
                continue
            start = max(0, row - window + 1)
            max_val = -np.inf
            count = 0
            for k in range(start, row + 1):
                val = arr[k, c]
                if not np.isnan(val):
                    if val > max_val:
                        max_val = val
                    count += 1
            if count >= min_periods:
                result[row, c] = max_val
            else:
                result[row, c] = np.nan
```

**Acceptance Criteria**:
- Correctly finds min/max in rolling window
- Handles NaN values by excluding from comparison

---

#### TODO 1.4: Add nogil kernels for moment rolling ops
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py`

```python
@njit(nogil=True, cache=True)
def _rolling_skew_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Rolling skewness - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        for row in range(n_rows):
            if row < min_periods - 1:
                result[row, c] = np.nan
                continue
            start = max(0, row - window + 1)
            # Collect window values
            values = np.empty(window, dtype=np.float64)
            count = 0
            for k in range(start, row + 1):
                val = arr[k, c]
                if not np.isnan(val):
                    values[count] = val
                    count += 1
            if count >= min_periods and count >= 3:
                # Compute mean
                mean = 0.0
                for i in range(count):
                    mean += values[i]
                mean /= count
                # Compute moments
                m2 = 0.0
                m3 = 0.0
                for i in range(count):
                    delta = values[i] - mean
                    m2 += delta * delta
                    m3 += delta * delta * delta
                m2 /= count
                m3 /= count
                if m2 > 1e-14:
                    skew = m3 / (m2 ** 1.5)
                    if count > 2:
                        adjust = np.sqrt(count * (count - 1)) / (count - 2)
                        result[row, c] = adjust * skew
                    else:
                        result[row, c] = skew
                else:
                    result[row, c] = 0.0
            else:
                result[row, c] = np.nan

@njit(nogil=True, cache=True)
def _rolling_kurt_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Rolling kurtosis - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        for row in range(n_rows):
            if row < min_periods - 1:
                result[row, c] = np.nan
                continue
            start = max(0, row - window + 1)
            values = np.empty(window, dtype=np.float64)
            count = 0
            for k in range(start, row + 1):
                val = arr[k, c]
                if not np.isnan(val):
                    values[count] = val
                    count += 1
            if count >= min_periods and count >= 4:
                mean = 0.0
                for i in range(count):
                    mean += values[i]
                mean /= count
                m2 = 0.0
                m4 = 0.0
                for i in range(count):
                    delta = values[i] - mean
                    delta2 = delta * delta
                    m2 += delta2
                    m4 += delta2 * delta2
                m2 /= count
                m4 /= count
                if m2 > 1e-14:
                    kurt = m4 / (m2 * m2) - 3.0
                    if count > 3:
                        adjust = (count - 1) / ((count - 2) * (count - 3))
                        term1 = (count + 1) * kurt
                        term2 = 3.0 * (count - 1)
                        result[row, c] = adjust * (term1 + term2)
                    else:
                        result[row, c] = kurt
                else:
                    result[row, c] = 0.0
            else:
                result[row, c] = np.nan

@njit(nogil=True, cache=True)
def _rolling_count_nogil_chunk(arr, result, start_col, end_col, window, min_periods):
    """Rolling count - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                count += 1
            if row >= window:
                old_val = arr[row - window, c]
                if not np.isnan(old_val):
                    count -= 1
            if count >= min_periods:
                result[row, c] = float(count)
            else:
                result[row, c] = np.nan
```

**Acceptance Criteria**:
- Skew/kurt use bias-corrected formulas matching pandas
- Count tracks non-NaN values correctly

---

#### TODO 1.5: Update ThreadPool functions to use nogil kernels
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py`

Replace the existing `_rolling_*_threadpool` functions with updated versions that call nogil kernels:

```python
def _rolling_mean_threadpool(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Ultra-fast rolling mean using ThreadPool + nogil Numba kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan  # Initialize to NaN

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _rolling_mean_nogil_chunk(arr, result, start_col, end_col, window, min_periods)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS)]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result
```

Apply same pattern to: `_rolling_sum_threadpool`, `_rolling_std_threadpool`, `_rolling_var_threadpool`, `_rolling_min_threadpool`, `_rolling_max_threadpool`

Add new threadpool functions for skew, kurt, count (currently not using threadpool):

```python
def _rolling_skew_threadpool(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Rolling skew using ThreadPool + nogil kernels."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    chunk_size = max(1, (n_cols + THREADPOOL_WORKERS - 1) // THREADPOOL_WORKERS)

    def process_chunk(args):
        start_col, end_col = args
        _rolling_skew_nogil_chunk(arr, result, start_col, end_col, window, min_periods)

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_cols))
              for i in range(THREADPOOL_WORKERS)]

    with ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS) as executor:
        list(executor.map(process_chunk, chunks))

    return result
```

**Update dispatch functions** to use threadpool for large arrays:
```python
def _rolling_skew_dispatch(arr, window, min_periods):
    if arr.size >= THREADPOOL_THRESHOLD:
        return _rolling_skew_threadpool(arr, window, min_periods)
    if arr.size < PARALLEL_THRESHOLD:
        return _rolling_skew_2d_serial(arr, window, min_periods)
    return _rolling_skew_2d(arr, window, min_periods)
```

**Acceptance Criteria**:
- All threadpool functions use nogil kernels
- Dispatch functions route large arrays to threadpool
- Tests pass with identical results

---

### Phase 2: Expanding Operations (`src/unlockedpd/ops/expanding.py`)

#### TODO 2.1: Add nogil kernels for basic expanding ops
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/expanding.py`

```python
@njit(nogil=True, cache=True)
def _expanding_mean_nogil_chunk(arr, result, start_col, end_col, min_periods):
    """Expanding mean - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cumsum = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                cumsum += val
                count += 1
            if count >= min_periods:
                result[row, c] = cumsum / count
            else:
                result[row, c] = np.nan

@njit(nogil=True, cache=True)
def _expanding_sum_nogil_chunk(arr, result, start_col, end_col, min_periods):
    """Expanding sum - GIL released."""
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        cumsum = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, c]
            if not np.isnan(val):
                cumsum += val
                count += 1
            if count >= min_periods:
                result[row, c] = cumsum
            else:
                result[row, c] = np.nan
```

---

#### TODO 2.2-2.4: Add remaining nogil kernels for expanding
Apply same pattern as rolling for: std, var, min, max, skew, kurt, count

**Acceptance Criteria**:
- All expanding operations have nogil kernel variants
- Expanding operations do NOT have window parameter (simpler than rolling)

---

#### TODO 2.5: Update ThreadPool functions for expanding
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/expanding.py`

Update existing threadpool functions and add new ones for skew, kurt, count.

---

### Phase 3: Warmup & Validation

#### TODO 3.1: Update warmup module
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/_warmup.py`

Add warmup for new nogil kernels:

```python
def warmup_rolling_nogil():
    """Pre-compile rolling nogil kernels."""
    from .ops.rolling import (
        _rolling_mean_nogil_chunk, _rolling_sum_nogil_chunk,
        _rolling_std_nogil_chunk, _rolling_var_nogil_chunk,
        _rolling_min_nogil_chunk, _rolling_max_nogil_chunk,
        _rolling_skew_nogil_chunk, _rolling_kurt_nogil_chunk,
        _rolling_count_nogil_chunk,
    )

    dummy = np.zeros((10, 4), dtype=np.float64)
    result = np.zeros((10, 4), dtype=np.float64)

    try:
        _rolling_mean_nogil_chunk(dummy, result, 0, 4, 3, 1)
        _rolling_sum_nogil_chunk(dummy, result, 0, 4, 3, 1)
        _rolling_std_nogil_chunk(dummy, result, 0, 4, 3, 1, 1)
        _rolling_var_nogil_chunk(dummy, result, 0, 4, 3, 1, 1)
        _rolling_min_nogil_chunk(dummy, result, 0, 4, 3, 1)
        _rolling_max_nogil_chunk(dummy, result, 0, 4, 3, 1)
        _rolling_skew_nogil_chunk(dummy, result, 0, 4, 3, 1)
        _rolling_kurt_nogil_chunk(dummy, result, 0, 4, 3, 1)
        _rolling_count_nogil_chunk(dummy, result, 0, 4, 3, 1)
    except Exception:
        pass

def warmup_expanding_nogil():
    """Pre-compile expanding nogil kernels."""
    # Similar pattern for expanding kernels
    ...
```

Update `warmup_all()` to include the new warmup functions.

**Acceptance Criteria**:
- All nogil kernels are warmed up at import time
- No first-call JIT compilation delay

---

#### TODO 3.2: Run test suite
```bash
cd /home/bellman/Workspace/MyNumbaDataFrame
pytest tests/test_rolling.py -v
pytest tests/test_expanding.py -v
pytest tests/ -v  # Full suite
```

**Acceptance Criteria**:
- All existing tests pass
- No numerical accuracy regressions

---

#### TODO 3.3: Run benchmarks
```bash
cd /home/bellman/Workspace/MyNumbaDataFrame
pytest benchmarks/bench_rolling.py --benchmark-only -v
```

Or create a quick benchmark script:
```python
import time
import numpy as np
import pandas as pd

# Test data: 100K rows x 100 cols = 10M elements (above THREADPOOL_THRESHOLD)
df = pd.DataFrame(np.random.randn(100_000, 100))

# Warmup
import unlockedpd
unlockedpd.config.enabled = True
_ = df.rolling(20).mean()

# Benchmark optimized
start = time.perf_counter()
for _ in range(5):
    result = df.rolling(20).mean()
optimized_time = (time.perf_counter() - start) / 5

# Benchmark pandas
unlockedpd.config.enabled = False
start = time.perf_counter()
for _ in range(5):
    result = df.rolling(20).mean()
pandas_time = (time.perf_counter() - start) / 5

print(f"Optimized: {optimized_time:.3f}s")
print(f"Pandas:    {pandas_time:.3f}s")
print(f"Speedup:   {pandas_time/optimized_time:.1f}x")
```

**Acceptance Criteria**:
- Rolling mean achieves >= 4.5x speedup on 10M+ element arrays
- Other operations show similar improvements

---

#### TODO 3.4: Document performance
Update module docstrings with benchmark results.

---

## Commit Strategy

### Commit 1: Rolling nogil kernels
```
feat(rolling): add @njit(nogil=True) kernels for ThreadPool parallelism

- Add nogil kernel variants for all rolling operations
- Update threadpool functions to use nogil kernels
- Add threadpool path for skew, kurt, count operations
- Update dispatch functions for large array routing

Achieves 4.7x speedup over pandas for 10M+ element arrays by
releasing GIL during Numba-compiled column processing.
```

### Commit 2: Expanding nogil kernels
```
feat(expanding): add @njit(nogil=True) kernels for ThreadPool parallelism

- Add nogil kernel variants for all expanding operations
- Update threadpool functions to use nogil kernels
- Add threadpool path for skew, kurt, count operations
```

### Commit 3: Warmup and validation
```
chore(warmup): add warmup for nogil kernels

- Add warmup_rolling_nogil() and warmup_expanding_nogil()
- Update warmup_all() to include new functions
- Ensure no first-call JIT compilation delay
```

---

## Success Criteria

| Metric | Target | Verification |
|--------|--------|--------------|
| Rolling mean speedup | >= 4.5x | Benchmark vs pandas |
| Rolling sum speedup | >= 4.5x | Benchmark vs pandas |
| Rolling std speedup | >= 3.5x | Benchmark vs pandas |
| Test pass rate | 100% | pytest tests/ |
| NaN handling | Correct | test_rolling_mean_with_nan |
| Memory usage | No increase | Visual inspection |

---

## Risk Identification

### Risk 1: Numba compilation cache invalidation
**Likelihood**: Medium
**Impact**: Low (one-time recompilation)
**Mitigation**: Document that users may need to clear `__pycache__` after update

### Risk 2: Numerical differences from algorithm change
**Likelihood**: Low
**Impact**: High
**Mitigation**: Use same algorithms (Welford for std/var), run comprehensive tests

### Risk 3: ThreadPool overhead for small arrays
**Likelihood**: Medium
**Impact**: Low
**Mitigation**: Keep existing THREADPOOL_THRESHOLD (10M elements) dispatch logic

### Risk 4: nogil kernel doesn't release GIL properly
**Likelihood**: Very Low
**Impact**: High (no speedup)
**Mitigation**: Verify with threading benchmark that threads run in parallel

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `src/unlockedpd/ops/rolling.py` | Add 9 nogil kernels, update 6 threadpool functions, add 3 new threadpool functions |
| `src/unlockedpd/ops/expanding.py` | Add 9 nogil kernels, update 6 threadpool functions, add 3 new threadpool functions |
| `src/unlockedpd/_warmup.py` | Add warmup_rolling_nogil(), warmup_expanding_nogil() |

---

## Verification Commands

```bash
# Run tests
pytest tests/test_rolling.py tests/test_expanding.py -v

# Run benchmarks
pytest benchmarks/bench_rolling.py --benchmark-only

# Quick manual benchmark
python -c "
import time
import numpy as np
import pandas as pd
import unlockedpd

df = pd.DataFrame(np.random.randn(100_000, 100))

# Warmup
_ = df.rolling(20).mean()

# Benchmark
unlockedpd.config.enabled = True
start = time.perf_counter()
for _ in range(5): df.rolling(20).mean()
opt = (time.perf_counter() - start) / 5

unlockedpd.config.enabled = False
start = time.perf_counter()
for _ in range(5): df.rolling(20).mean()
pd_time = (time.perf_counter() - start) / 5

print(f'Speedup: {pd_time/opt:.1f}x (target: 4.5x+)')
"
```

---

**Plan Generated**: 2026-01-20
**Author**: Prometheus (Strategic Planning Consultant)
