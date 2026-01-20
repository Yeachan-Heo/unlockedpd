# Performance Optimization Plan: unlockedpd

## Executive Summary

The unlockedpd library needs critical performance optimizations to deliver on its promise of transparent pandas acceleration. Oracle analysis identified **6 major issues** causing the library to be **slower than pandas on small data** and only **marginally faster on large data**.

### Key Findings

| Issue | Impact | Priority |
|-------|--------|----------|
| 1. Wrapper overhead | 10-50x slowdown on small data | CRITICAL |
| 2. Memory layout mismatch | 2-3x cache miss penalty | HIGH |
| 3. O(N*W) algorithms for std/var/min/max | 10-100x for large windows | HIGH |
| 4. Excessive copies in rank operations | 2x memory overhead | MEDIUM |
| 5. No size thresholds for parallelization | Thread overhead dominates small data | CRITICAL |
| 6. Redundant type conversions | Extra memory allocations | MEDIUM |

### Expected Performance After Fixes

| Data Size | Current vs Pandas | Expected After Fixes |
|-----------|------------------|---------------------|
| 1k x 10 | 0.5x (slower) | 0.9-1.0x (parity) |
| 10k x 100 | 1.2x | 3-5x |
| 100k x 100 | 1.5x | 5-10x |
| 100k x 100 (rolling_std) | 0.8x | 8-15x |

---

## Phase 1: Critical Fixes (Wrapper & Parallelization Threshold)

### Task 1.1: Add Size Thresholds for Parallel Execution

**File**: `src/unlockedpd/ops/rolling.py`

Add serial versions of all Numba functions and dispatch based on array size:

```python
# Constants at module level
PARALLEL_THRESHOLD = 10_000  # Elements before parallel is beneficial

@njit(cache=True)  # Serial version - NO parallel
def _rolling_sum_2d_serial(arr, window, min_periods):
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan
    for col in range(n_cols):  # range, not prange
        cumsum = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                cumsum += val
                count += 1
            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    cumsum -= old_val
                    count -= 1
            if count >= min_periods:
                result[row, col] = cumsum
    return result

# Dispatch in wrapper
def rolling_sum(arr, window, min_periods):
    if arr.size < PARALLEL_THRESHOLD:
        return _rolling_sum_2d_serial(arr, window, min_periods)
    return _rolling_sum_2d(arr, window, min_periods)
```

**Apply to**: All rolling operations (sum, mean, std, var, min, max)

### Task 1.2: Move Imports to Module Level

**File**: `src/unlockedpd/ops/rolling.py` (lines 272-273)

**Before**:
```python
def wrapper(rolling_obj, *args, **kwargs):
    from .._compat import get_numeric_columns, wrap_result, ensure_float64  # Slow!
```

**After**:
```python
# At module top level
from .._compat import get_numeric_columns, wrap_result, ensure_float64

def wrapper(rolling_obj, *args, **kwargs):
    # Imports already available
```

### Task 1.3: Add Configuration for Parallel Threshold

**File**: `src/unlockedpd/_config.py`

Add configurable threshold:

```python
_parallel_threshold: int = field(default=10_000, repr=False)

@property
def parallel_threshold(self) -> int:
    """Minimum array size before parallel execution is used."""
    with self._lock:
        return self._parallel_threshold

@parallel_threshold.setter
def parallel_threshold(self, value: int) -> None:
    with self._lock:
        self._parallel_threshold = int(value)
```

---

## Phase 2: Memory Layout Optimization

### Task 2.1: Use F-Contiguous Layout for Column-Parallel Operations

**File**: `src/unlockedpd/_compat.py`

The existing `ensure_contiguous()` function (lines 135-146) is **never called**. Add a new function that enforces the optimal memory layout:

```python
def ensure_optimal_layout(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Ensure array has optimal memory layout for the operation axis.

    For column-wise operations (axis=0), F-contiguous is optimal.
    For row-wise operations (axis=1), C-contiguous is optimal.
    """
    if axis == 0:
        # Column-parallel: F-contiguous for sequential column access
        if arr.flags['F_CONTIGUOUS']:
            return arr
        return np.asfortranarray(arr)
    else:
        # Row-parallel: C-contiguous for sequential row access
        if arr.flags['C_CONTIGUOUS']:
            return arr
        return np.ascontiguousarray(arr)
```

### Task 2.2: Call ensure_optimal_layout in Rolling Wrapper

**File**: `src/unlockedpd/ops/rolling.py`

Update wrapper to use optimal layout:

```python
def wrapper(rolling_obj, *args, **kwargs):
    # ... existing code ...
    arr = ensure_float64(numeric_df.values)
    arr = ensure_optimal_layout(arr, axis=0)  # Column-parallel operations
    # ... rest of wrapper ...
```

### Task 2.3: Update Rank Operations for Row-Wise Layout

**File**: `src/unlockedpd/ops/rank.py`

For axis=1 (row-wise) ranking, ensure C-contiguous:

```python
def _optimized_rank(df, axis=0, method='average', ...):
    arr = ensure_float64(numeric_df.values)
    if axis == 1:
        arr = ensure_optimal_layout(arr, axis=1)  # C-contiguous for row ops
    else:
        arr = ensure_optimal_layout(arr, axis=0)  # F-contiguous for col ops
```

---

## Phase 3: Algorithm Optimization (O(n) vs O(n*w))

### Task 3.1: Implement Welford's Algorithm for Rolling Std/Var

**File**: `src/unlockedpd/ops/rolling.py`

Replace the current O(n*w) two-pass algorithm with O(n) Welford's online algorithm:

```python
@njit(parallel=True, cache=True)
def _rolling_std_2d_welford(arr, window, min_periods, ddof=1):
    """Rolling standard deviation using Welford's online algorithm.

    Complexity: O(n) per column instead of O(n*w)
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0  # Sum of squared differences from current mean

        for row in range(n_rows):
            val = arr[row, col]

            # Add new value using Welford's update
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

            # Remove old value (reverse Welford) when past window
            if row >= window:
                old_val = arr[row - window, col]
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

            # Compute result
            if count >= min_periods and count > ddof:
                result[row, col] = np.sqrt(M2 / (count - ddof))

    return result

@njit(parallel=True, cache=True)
def _rolling_var_2d_welford(arr, window, min_periods, ddof=1):
    """Rolling variance using Welford's algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        count = 0
        mean = 0.0
        M2 = 0.0

        for row in range(n_rows):
            val = arr[row, col]

            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                M2 += delta * delta2

            if row >= window:
                old_val = arr[row - window, col]
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
                result[row, col] = M2 / (count - ddof)

    return result
```

### Task 3.2: Implement Monotonic Deque for Rolling Min/Max

**File**: `src/unlockedpd/ops/rolling.py`

Current O(n*w) min/max can be replaced with O(n) using a monotonic deque:

```python
@njit(cache=True)
def _rolling_min_1d_deque(arr, window, min_periods):
    """O(n) rolling min using monotonic deque."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    # Deque stores indices, not values
    # deque_idx[deque_front:deque_back] contains valid indices
    deque_idx = np.empty(window, dtype=np.int64)
    deque_front = 0
    deque_back = 0

    for i in range(n):
        val = arr[i]

        # Remove elements outside window
        while deque_front < deque_back and deque_idx[deque_front] <= i - window:
            deque_front += 1

        if not np.isnan(val):
            # Remove elements larger than current (maintain monotonic increasing)
            while deque_front < deque_back and arr[deque_idx[deque_back - 1]] >= val:
                deque_back -= 1

            # Add current index
            if deque_back < window:
                deque_idx[deque_back] = i
                deque_back += 1

        # Count non-NaN values in window
        count = 0
        for j in range(max(0, i - window + 1), i + 1):
            if not np.isnan(arr[j]):
                count += 1

        if count >= min_periods and deque_front < deque_back:
            result[i] = arr[deque_idx[deque_front]]

    return result

@njit(parallel=True, cache=True)
def _rolling_min_2d_optimized(arr, window, min_periods):
    """Parallel rolling min with O(n) per column."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        result[:, col] = _rolling_min_1d_deque(arr[:, col], window, min_periods)

    return result
```

---

## Phase 4: Reduce Memory Copies

### Task 4.1: Eliminate Unnecessary Copies in Rank Operations

**File**: `src/unlockedpd/ops/rank.py`

The current code copies each row/column to replace NaN with infinity. NumPy's argsort handles NaN properly (sorts to end), so copies are unnecessary:

**Before** (line 60):
```python
for row in prange(n_rows):
    row_data = arr[row, :].copy()  # COPY for every row!
    is_nan = np.isnan(row_data)
    row_data[is_nan] = np.inf
    sorted_idx = np.argsort(row_data)
```

**After**:
```python
for row in prange(n_rows):
    # argsort handles NaN by placing them at the end
    sorted_idx = np.argsort(arr[row, :])
    # Handle NaN during rank assignment instead
```

### Task 4.2: Pre-allocate Result Arrays Outside Parallel Loop

Ensure result arrays are allocated once before the parallel loop, not inside:

```python
# Good - allocate once
result = np.empty((n_rows, n_cols), dtype=np.float64)
for col in prange(n_cols):
    # Use pre-allocated result

# Bad - avoid this pattern
for col in prange(n_cols):
    temp = np.empty(n_rows)  # Allocation inside parallel loop
```

---

## Phase 5: Import-Time Warmup & Eager Compilation

### Task 5.1: Add Eager Compilation with Type Signatures

**File**: `src/unlockedpd/ops/rolling.py`

Add explicit type signatures for eager compilation:

```python
from numba import njit, prange, float64
from numba.types import Array

# Eager compilation with explicit signatures
@njit(
    float64[:, :](float64[:, :], int64, int64),
    parallel=True,
    cache=True
)
def _rolling_sum_2d(arr, window, min_periods):
    # ... implementation ...
```

### Task 5.2: Add Warmup on Module Import

**File**: `src/unlockedpd/ops/rolling.py`

Add warmup at module load to trigger JIT compilation:

```python
def _warmup_rolling_functions():
    """Pre-compile Numba functions with representative data."""
    dummy = np.zeros((10, 4), dtype=np.float64)

    # Trigger compilation
    _rolling_sum_2d(dummy, 3, 1)
    _rolling_mean_2d(dummy, 3, 1)
    _rolling_std_2d_welford(dummy, 3, 1, 1)
    _rolling_var_2d_welford(dummy, 3, 1, 1)
    _rolling_min_2d_optimized(dummy, 3, 1)
    _rolling_max_2d_optimized(dummy, 3, 1)

# Run at import time if caching is enabled
try:
    _warmup_rolling_functions()
except Exception:
    pass  # Silently fail - will compile on first use
```

### Task 5.3: Add Warmup to __init__.py

**File**: `src/unlockedpd/__init__.py`

Add warmup call after patches are applied:

```python
def _warmup_all():
    """Pre-compile all Numba functions."""
    from .ops.rolling import _warmup_rolling_functions
    from .ops.rank import _warmup_rank_functions

    _warmup_rolling_functions()
    _warmup_rank_functions()

# Auto-patch on import if enabled
if config.enabled:
    _apply_all_patches()
    _warmup_all()  # Trigger compilation
```

---

## Phase 6: 2D Matrix Optimization (User Request)

### Task 6.1: Implement Block-Based Processing for Large Matrices

For very large 2D matrices, process in cache-friendly blocks:

```python
BLOCK_SIZE = 256  # Rows per block for cache efficiency

@njit(parallel=True, cache=True)
def _rolling_mean_2d_blocked(arr, window, min_periods):
    """Block-based rolling mean for better cache utilization."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    # Process columns in parallel, but use blocks for row iteration
    for col in prange(n_cols):
        cumsum = 0.0
        count = 0

        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                cumsum += val
                count += 1

            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    cumsum -= old_val
                    count -= 1

            if count >= min_periods:
                result[row, col] = cumsum / count

    return result
```

### Task 6.2: Add Tile-Based Operations for Rank

For rank operations on large matrices, process in tiles:

```python
TILE_SIZE = 1024  # Rows per tile

def _rank_tiled(arr, axis, method):
    """Tile-based ranking for memory efficiency."""
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    if axis == 1:
        for start in range(0, n_rows, TILE_SIZE):
            end = min(start + TILE_SIZE, n_rows)
            tile = arr[start:end, :]
            result[start:end, :] = _rank_axis1_kernel(tile, method)

    return result
```

---

## Implementation Order

1. **Phase 1** (Critical - Immediate Impact)
   - Task 1.1: Size thresholds
   - Task 1.2: Module-level imports
   - Task 1.3: Config for threshold

2. **Phase 2** (High Impact)
   - Task 2.1: ensure_optimal_layout function
   - Task 2.2: Update rolling wrapper
   - Task 2.3: Update rank operations

3. **Phase 3** (High Impact for Large Windows)
   - Task 3.1: Welford's algorithm
   - Task 3.2: Monotonic deque for min/max

4. **Phase 4** (Medium Impact)
   - Task 4.1: Remove rank copies
   - Task 4.2: Pre-allocate arrays

5. **Phase 5** (First-Call Impact)
   - Task 5.1: Eager compilation signatures
   - Task 5.2: Rolling warmup
   - Task 5.3: Init warmup

6. **Phase 6** (Large Matrix Support)
   - Task 6.1: Block-based processing
   - Task 6.2: Tile-based rank

---

## Verification Steps

After each phase, run:

```bash
# Unit tests
pytest tests/ -v

# Benchmark comparison
python benchmarks/rolling_benchmark.py

# Memory profiling
python -m memory_profiler benchmarks/memory_test.py
```

### Expected Benchmark Improvements

| Phase | Small Data (1k x 10) | Large Data (100k x 100) |
|-------|---------------------|-------------------------|
| Baseline | 0.5x pandas | 1.5x pandas |
| After Phase 1 | 0.9x pandas | 2.0x pandas |
| After Phase 2 | 1.0x pandas | 4.0x pandas |
| After Phase 3 | 1.0x pandas | 8.0x pandas |
| After All | 1.0x pandas | 10x+ pandas |

---

## Trade-offs

| Optimization | Benefit | Cost |
|-------------|---------|------|
| Size thresholds | Fixes small-data regression | Two code paths |
| F-contiguous | 2-3x cache improvement | Extra copy if data is C-order |
| Welford algorithm | 10-100x for large windows | More complex, edge cases |
| Remove copies | 2x memory reduction | Careful NaN handling |
| Warmup | Faster first call | Slower import (~200ms) |

---

## Success Criteria

1. **Small data parity**: unlockedpd should not be slower than pandas for DataFrames < 10k elements
2. **Large data speedup**: 5-10x faster than pandas for DataFrames > 100k elements
3. **Memory efficiency**: No more than 2x memory overhead vs pandas
4. **All tests pass**: Maintain 100% compatibility with pandas output
