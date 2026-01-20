# Work Plan: UnlockedPD Coverage Gaps

## Context

### Original Request
Address all gaps in unlockedpd pandas optimization coverage based on comprehensive analysis.

### Research Findings
Analysis of the codebase reveals the following structure:
- **Package**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/`
- **Operations modules**: `ops/rolling.py`, `ops/expanding.py`, `ops/ewm.py`, `ops/cumulative.py`, `ops/rank.py`, `ops/transform.py`, `ops/pairwise.py`, `ops/stats.py`
- **Pattern**: ThreadPool + Numba `nogil=True` kernels for true parallelism (4.7x speedup)
- **Dispatch strategy**: Serial (<500K elements), Parallel (500K-10M), ThreadPool (>10M)

### Verified Gaps

| Gap ID | Category | Issue | Impact | Effort |
|--------|----------|-------|--------|--------|
| GAP-1 | BUG | `apply_stats_patches()` not called in `__init__.py` | HIGH (dead code) | TRIVIAL |
| GAP-2 | Expanding | Missing `median`, `quantile` | MEDIUM | LOW |
| GAP-3 | EWM | Missing `corr`, `cov` | MEDIUM | MEDIUM |
| GAP-4 | Rolling | Missing `apply`, `rank`, `sem` | LOW | MEDIUM |
| GAP-5 | DataFrame Stats | Missing `mean`, `std`, `var`, `sum`, `corr`, `cov` | MEDIUM | MEDIUM |
| GAP-6 | GroupBy | 0% coverage - major gap | HIGH | HIGH |

---

## Work Objectives

### Core Objective
Achieve comprehensive pandas operation coverage by fixing the stats bug and implementing missing operations across expanding, EWM, rolling, and DataFrame aggregations.

### Deliverables
1. **GAP-1 FIX**: Stats patches enabled (skew, kurt, sem working)
2. **GAP-2 IMPL**: `expanding().median()` and `expanding().quantile()`
3. **GAP-3 IMPL**: `ewm().corr()` and `ewm().cov()`
4. **GAP-4 IMPL**: `rolling().sem()` (apply/rank deferred - complex)
5. **GAP-5 IMPL**: `DataFrame.mean()`, `.std()`, `.var()`, `.sum()`
6. **Tests**: Full test coverage for all new implementations

### Definition of Done
- [ ] All patches correctly applied on import
- [ ] All new operations match pandas output (assert_frame_equal with rtol=1e-10)
- [ ] All existing tests pass
- [ ] New tests cover edge cases (NaN handling, empty DataFrames, single column)
- [ ] Benchmarks show speedup >= 1.5x on 10M+ element DataFrames

---

## Guardrails

### Must Have
- Exact numerical match with pandas (rtol=1e-10)
- NaN handling identical to pandas
- Fallback to pandas for unsupported cases (Series, non-numeric columns)
- ThreadPool + nogil pattern for large arrays
- Cache=True on all njit functions
- Tests comparing optimized vs pandas output

### Must NOT Have
- Breaking changes to existing patched operations
- Memory leaks (ensure ThreadPoolExecutor properly cleaned up)
- Hanging threads (ensure proper exception handling in ThreadPool)
- Silent failures (must raise TypeError for unsupported cases)
- Modifications to pandas internals (only monkey-patching allowed)

---

## Task Flow

```
PHASE 1: Critical Bug Fix (Immediate)
    |
    v
[T1] Enable stats patches --> Run tests --> Verify skew/kurt/sem work
    |
    v
PHASE 2: Quick Wins (Low effort, high pattern reuse)
    |
    +---> [T2] Expanding median (port from rolling.py)
    |
    +---> [T3] Expanding quantile (port from rolling.py)
    |
    +---> [T4] Rolling sem (reuse rolling std pattern)
    |
    v
PHASE 3: Medium Complexity (New implementations)
    |
    +---> [T5] EWM corr (pairwise EWM)
    |
    +---> [T6] EWM cov (pairwise EWM)
    |
    v
PHASE 4: DataFrame Reductions
    |
    +---> [T7] DataFrame.mean/std/var/sum axis=0/1
    |
    v
PHASE 5: Full Test Suite
    |
    +---> [T8] Add tests for all new operations
    |
    v
[DONE] Full coverage achieved
```

---

## Detailed Tasks

### PHASE 1: Critical Bug Fix

#### T1: Enable stats patches
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/__init__.py`

**Current state** (lines 58-62 in `__init__.py`):
```python
    Not patched (marginal or negative benefit):
    - cumulative (cumsum, cumprod, etc.): NumPy SIMD is faster
    - ewm: Same speed as pandas (no benefit)
    - stats: Marginal benefit
    - pairwise: Complex semantics
```

**Rationale for enabling stats despite the comment**:
The comment "stats: Marginal benefit" was written during initial benchmarking. However:
1. The stats module (`ops/stats.py`) is fully implemented with parallel and serial versions
2. The implementations use online algorithms (Welford's method) for numerical stability
3. For wide DataFrames (many columns), the parallel execution provides measurable speedup
4. The dead code provides no benefit - enabling it at minimum activates the existing work
5. **Action**: Update the comment to reflect that stats patches are now enabled, with a note that benefit is shape-dependent (better for wide DataFrames)

**Required changes**:
```python
# In _apply_all_patches(), after line 90 (after apply_ewm_patches()):
from .ops.stats import apply_stats_patches
apply_stats_patches()

# Update the docstring comment (lines 58-62) to:
    Shape-dependent benefits:
    - stats (skew, kurt, sem): Enabled - benefits wide DataFrames with parallel column processing

    Not patched (marginal or negative benefit):
    - cumulative (cumsum, cumprod, etc.): NumPy SIMD is faster
```

**Acceptance criteria**:
- [ ] `df.skew()` returns optimized result
- [ ] `df.kurt()` returns optimized result
- [ ] `df.sem()` returns optimized result
- [ ] All three match pandas output exactly
- [ ] Comment in `__init__.py` updated to explain the change

---

### PHASE 2: Quick Wins

#### T2: Expanding median
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/expanding.py`

**Implementation approach**:
1. Port `_rolling_median_nogil_chunk` pattern from rolling.py (lines 662-699)
2. Adapt for expanding window (window grows with each row: window = row + 1)
3. Create `_expanding_median_nogil_chunk` with nogil=True
4. Create ThreadPool wrapper `_expanding_median_threadpool`
5. Create dispatch function `_expanding_median_dispatch`
6. Create `optimized_expanding_median` wrapper
7. Add to `apply_expanding_patches()`

**Algorithm complexity note**:
The expanding median algorithm uses insertion sort at each row, resulting in O(n^2) total complexity per column.
- For row `i`, we sort `i+1` elements using insertion sort: O(i) comparisons
- Total: O(1 + 2 + 3 + ... + n) = O(n^2) per column
- **This is acceptable** because:
  1. Matches the same pattern used in `rolling.py` for rolling median
  2. True O(n log n) incremental median requires complex data structures (two heaps) that don't JIT well
  3. The parallelization across columns compensates for the per-column complexity
  4. For typical DataFrame sizes (10K-100K rows), this is fast enough in practice
- **Alternative not implemented**: A heap-based O(n log n) algorithm would require maintaining two heaps (max-heap for lower half, min-heap for upper half), which adds significant complexity and doesn't work well with Numba's limitations

**Complete algorithm**:
```python
@njit(nogil=True, cache=True)
def _expanding_median_nogil_chunk(arr, result, start_col, end_col, min_periods):
    """Expanding median using insertion sort - GIL released.

    Complexity: O(n^2) per column where n = number of rows.
    This is acceptable as we parallelize across columns.
    """
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        # Buffer grows with each row (expanding window)
        buffer = np.empty(n_rows, dtype=np.float64)

        for row in range(n_rows):
            # Build sorted buffer from all values up to current row
            count = 0
            for k in range(row + 1):  # expanding = all rows [0, row]
                val = arr[k, c]
                if not np.isnan(val):
                    # Insertion sort into buffer
                    i = count
                    while i > 0 and buffer[i - 1] > val:
                        buffer[i] = buffer[i - 1]
                        i -= 1
                    buffer[i] = val
                    count += 1

            if count >= min_periods:
                # Get median from sorted buffer
                if count % 2 == 1:
                    result[row, c] = buffer[count // 2]
                else:
                    result[row, c] = (buffer[count // 2 - 1] + buffer[count // 2]) / 2.0
            else:
                result[row, c] = np.nan
```

**Acceptance criteria**:
- [ ] `df.expanding().median()` matches pandas exactly
- [ ] Handles NaN values correctly (skipped in count)
- [ ] ThreadPool dispatch for large arrays (>10M elements)
- [ ] Respects `min_periods` parameter

---

#### T3: Expanding quantile
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/expanding.py`

**Implementation approach**: Same pattern as T2, but with quantile interpolation.

**Reference**: Port from `_rolling_quantile_nogil_chunk` at rolling.py lines 702-734.

**Supported interpolation modes**:
| Mode | Description | Algorithm |
|------|-------------|-----------|
| `linear` (default) | Linear interpolation between data points | `idx = q * (n-1); result = arr[floor(idx)] * (1-frac) + arr[ceil(idx)] * frac` |
| `lower` | Use lower data point | `result = arr[floor(q * (n-1))]` |
| `higher` | Use higher data point | `result = arr[ceil(q * (n-1))]` |
| `nearest` | Use nearest data point | `result = arr[round(q * (n-1))]` |
| `midpoint` | Average of lower and higher | `result = (arr[floor] + arr[ceil]) / 2` |

**Note**: pandas default is `linear`. Start with `linear` only and fallback to pandas for other modes.

**Complete algorithm**:
```python
@njit(nogil=True, cache=True)
def _expanding_quantile_nogil_chunk(arr, result, start_col, end_col, min_periods, quantile):
    """Expanding quantile using insertion sort with linear interpolation.

    Args:
        quantile: Value between 0 and 1 (e.g., 0.25 for first quartile)

    Complexity: O(n^2) per column (same as expanding median).
    """
    n_rows = arr.shape[0]
    for c in range(start_col, end_col):
        buffer = np.empty(n_rows, dtype=np.float64)

        for row in range(n_rows):
            count = 0
            for k in range(row + 1):  # expanding window
                val = arr[k, c]
                if not np.isnan(val):
                    # Insertion sort
                    i = count
                    while i > 0 and buffer[i - 1] > val:
                        buffer[i] = buffer[i - 1]
                        i -= 1
                    buffer[i] = val
                    count += 1

            if count >= min_periods:
                # Linear interpolation for quantile (pandas default)
                idx = quantile * (count - 1)
                lower = int(idx)
                upper = min(lower + 1, count - 1)
                frac = idx - lower
                result[row, c] = buffer[lower] * (1 - frac) + buffer[upper] * frac
            else:
                result[row, c] = np.nan
```

**Wrapper function**:
```python
def _make_expanding_quantile_wrapper():
    """Create wrapper for expanding quantile."""
    def wrapper(expanding_obj, quantile, interpolation='linear', *args, **kwargs):
        # Only support 'linear' interpolation; fallback to pandas for others
        if interpolation != 'linear':
            raise TypeError(f"interpolation='{interpolation}' not supported, use pandas")

        obj = expanding_obj.obj
        min_periods = expanding_obj.min_periods if expanding_obj.min_periods is not None else 1

        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Optimization only for DataFrame")

        numeric_cols, numeric_df = get_numeric_columns_fast(obj)
        if len(numeric_cols) == 0:
            raise TypeError("No numeric columns to process")

        arr = ensure_float64(numeric_df.values)
        result = _expanding_quantile_dispatch(arr, min_periods, quantile)

        return wrap_result(result, numeric_df, columns=numeric_cols,
                          merge_non_numeric=True, original_df=obj)
    return wrapper
```

**Acceptance criteria**:
- [ ] `df.expanding().quantile(0.25)` matches pandas exactly
- [ ] `df.expanding().quantile(0.5)` matches pandas exactly (same as median)
- [ ] `df.expanding().quantile(0.75)` matches pandas exactly
- [ ] Supports `interpolation='linear'` parameter (default)
- [ ] Falls back to pandas for other interpolation modes (`lower`, `higher`, `nearest`, `midpoint`)
- [ ] Handles NaN values correctly

---

#### T4: Rolling sem
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py`

**Implementation approach**:
1. SEM = std / sqrt(count)
2. Reuse rolling std computation pattern
3. Divide by sqrt(count) at the end

**Acceptance criteria**:
- [ ] `df.rolling(20).sem()` matches pandas exactly
- [ ] Supports `ddof` parameter

---

### PHASE 3: EWM Pairwise Operations

#### T5: EWM corr
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/ewm.py`

**Output format** (matching pandas `df.ewm(span=20).corr()`):
- Returns a DataFrame with MultiIndex rows: `(original_index, column_name)`
- Columns: same as original DataFrame columns
- Shape: `(n_rows * n_cols, n_cols)`
- Example for a 100x3 DataFrame with columns ['A', 'B', 'C']:
  ```
  Row Index (MultiIndex)     | A    | B    | C
  (0, 'A')                   | 1.0  | NaN  | NaN
  (0, 'B')                   | NaN  | 1.0  | NaN
  (0, 'C')                   | NaN  | NaN  | 1.0
  (1, 'A')                   | 1.0  | 0.xx | 0.xx
  ...
  ```

**pairwise parameter**:
- `pairwise=True` (default for DataFrames): Compute correlation between all column pairs
- `pairwise=False` or `pairwise=None`: Not optimized, fallback to pandas
- `other` parameter: Not supported, fallback to pandas

**Reference implementation**: Follow the pattern in `pairwise.py` (lines 230-305) which implements `rolling.corr()` with the same output format.

**Complete algorithm**:
```python
@njit(nogil=True, cache=True)
def _ewm_corr_pairwise_nogil(arr_x, arr_y, result, alpha, adjust, ignore_na, min_periods, is_diagonal):
    """EWM correlation between two columns - GIL released.

    Pearson correlation using EWM:
    corr(X,Y) = cov(X,Y) / (std(X) * std(Y))

    Where:
    - ewm_cov(X,Y) = ewm(X*Y) - ewm(X)*ewm(Y)
    - ewm_var(X) = ewm(X^2) - ewm(X)^2

    For diagonal (same column), correlation is always 1.0.
    """
    n_rows = len(arr_x)

    if adjust:
        # Track EWM of: x, y, x^2, y^2, x*y
        ewm_x = 0.0
        ewm_y = 0.0
        ewm_x2 = 0.0
        ewm_y2 = 0.0
        ewm_xy = 0.0
        weight_sum = 0.0
        weight = 1.0
        nobs = 0

        for row in range(n_rows):
            vx = arr_x[row]
            vy = arr_y[row]

            # Check for NaN
            if np.isnan(vx) or np.isnan(vy):
                if not ignore_na:
                    # Reset on NaN
                    ewm_x = ewm_y = ewm_x2 = ewm_y2 = ewm_xy = 0.0
                    weight_sum = 0.0
                    weight = 1.0
                    nobs = 0
                result[row] = np.nan
                continue

            # Update EWM for all streams
            ewm_x += weight * vx
            ewm_y += weight * vy
            ewm_x2 += weight * vx * vx
            ewm_y2 += weight * vy * vy
            ewm_xy += weight * vx * vy
            weight_sum += weight
            weight *= (1.0 - alpha)
            nobs += 1

            if nobs >= min_periods:
                if is_diagonal:
                    result[row] = 1.0
                else:
                    # Compute means
                    mean_x = ewm_x / weight_sum
                    mean_y = ewm_y / weight_sum
                    mean_x2 = ewm_x2 / weight_sum
                    mean_y2 = ewm_y2 / weight_sum
                    mean_xy = ewm_xy / weight_sum

                    # Compute variances and covariance
                    var_x = mean_x2 - mean_x * mean_x
                    var_y = mean_y2 - mean_y * mean_y
                    cov_xy = mean_xy - mean_x * mean_y

                    # Correlation
                    if var_x > 1e-14 and var_y > 1e-14:
                        result[row] = cov_xy / np.sqrt(var_x * var_y)
                    else:
                        result[row] = np.nan
            else:
                result[row] = np.nan
    else:
        # Non-adjusted (recursive) version - similar logic
        # ... (implement similarly)
```

**Wrapper function**:
```python
def optimized_ewm_corr(ewm_obj, other=None, pairwise=None, *args, **kwargs):
    """Optimized EWM correlation."""
    obj = ewm_obj.obj

    # Only optimize DataFrame pairwise case
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    if other is not None:
        raise TypeError("other parameter not supported, use pairwise=True")

    if pairwise is False:
        raise TypeError("Only pairwise=True is optimized")

    # Extract EWM parameters
    alpha = _get_alpha(
        span=getattr(ewm_obj, 'span', None),
        halflife=getattr(ewm_obj, 'halflife', None),
        alpha=getattr(ewm_obj, 'alpha', None),
        com=getattr(ewm_obj, 'com', None)
    )
    adjust = ewm_obj.adjust
    ignore_na = ewm_obj.ignore_na
    min_periods = ewm_obj.min_periods if ewm_obj.min_periods is not None else 0

    numeric_cols, numeric_df = get_numeric_columns_fast(obj)
    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = ensure_float64(numeric_df.values)
    result_3d = _ewm_corr_pairwise_threadpool(arr, alpha, adjust, ignore_na, min_periods)

    # Convert to pandas format: MultiIndex rows (index, column), columns
    n_rows = len(obj)
    n_cols = len(numeric_cols)

    row_tuples = [(idx, col) for idx in obj.index for col in numeric_cols]
    multi_index = pd.MultiIndex.from_tuples(row_tuples)

    result_2d = result_3d.reshape(n_rows * n_cols, n_cols)
    return pd.DataFrame(result_2d, index=multi_index, columns=numeric_cols)
```

**Acceptance criteria**:
- [ ] `df.ewm(span=20).corr()` matches pandas exactly
- [ ] Returns MultiIndex DataFrame with structure `(original_index, column_name)` x columns
- [ ] Handles NaN correctly (skip or reset based on `ignore_na`)
- [ ] Supports all EWM parameter forms: `span`, `halflife`, `alpha`, `com`
- [ ] Diagonal elements (self-correlation) are always 1.0

---

#### T6: EWM cov
**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/ewm.py`

**Implementation approach**: Similar to T5 but without the std normalization.

**Output format**: Same MultiIndex DataFrame structure as EWM corr.

**Complete algorithm**:
```python
@njit(nogil=True, cache=True)
def _ewm_cov_pairwise_nogil(arr_x, arr_y, result, alpha, adjust, ignore_na, min_periods, bias):
    """EWM covariance between two columns - GIL released.

    cov(X,Y) = ewm(X*Y) - ewm(X)*ewm(Y)

    With optional bias correction when bias=False.
    """
    n_rows = len(arr_x)

    if adjust:
        ewm_x = 0.0
        ewm_y = 0.0
        ewm_xy = 0.0
        weight_sum = 0.0
        weight = 1.0
        nobs = 0

        for row in range(n_rows):
            vx = arr_x[row]
            vy = arr_y[row]

            if np.isnan(vx) or np.isnan(vy):
                if not ignore_na:
                    ewm_x = ewm_y = ewm_xy = 0.0
                    weight_sum = 0.0
                    weight = 1.0
                    nobs = 0
                result[row] = np.nan
                continue

            ewm_x += weight * vx
            ewm_y += weight * vy
            ewm_xy += weight * vx * vy
            weight_sum += weight
            weight *= (1.0 - alpha)
            nobs += 1

            if nobs >= min_periods:
                mean_x = ewm_x / weight_sum
                mean_y = ewm_y / weight_sum
                mean_xy = ewm_xy / weight_sum

                cov = mean_xy - mean_x * mean_y

                # Bias correction
                if not bias and nobs > 1:
                    cov *= weight_sum / (weight_sum - 1.0 + alpha)

                result[row] = cov
            else:
                result[row] = np.nan
```

**Acceptance criteria**:
- [ ] `df.ewm(span=20).cov()` matches pandas exactly
- [ ] Returns MultiIndex DataFrame like pandas
- [ ] Supports `bias` parameter (default False for bias correction)
- [ ] Handles NaN correctly

---

### PHASE 4: DataFrame Reductions

#### T7: DataFrame aggregate operations
**File**: NEW `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/aggregates.py`

**Operations**: `mean`, `std`, `var`, `sum` for axis=0 and axis=1

**Implementation approach**:
1. axis=0: reduce rows, parallelize across columns (column-independent)
2. axis=1: reduce columns, parallelize across rows (row-independent)
3. Use existing stats.py patterns

**Example kernel**:
```python
@njit(parallel=True, cache=True)
def _mean_2d_axis0(arr, skipna=True):
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)
    for col in prange(n_cols):
        total = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if skipna and np.isnan(val):
                continue
            total += val
            count += 1
        result[col] = total / count if count > 0 else np.nan
    return result
```

**Acceptance criteria**:
- [ ] `df.mean()` matches pandas exactly (axis=0)
- [ ] `df.mean(axis=1)` matches pandas exactly
- [ ] Same for std, var, sum
- [ ] Respects `skipna` parameter

---

### PHASE 5: Test Suite

#### T8: Comprehensive tests
**Files**:
- Update `/home/bellman/Workspace/MyNumbaDataFrame/tests/test_stats.py`
- Update `/home/bellman/Workspace/MyNumbaDataFrame/tests/test_expanding.py`
- Update `/home/bellman/Workspace/MyNumbaDataFrame/tests/test_ewm.py`
- Update `/home/bellman/Workspace/MyNumbaDataFrame/tests/test_rolling.py`

**Test file modifications**:

1. **test_expanding.py**:
   - **Remove** `@pytest.mark.skip(reason="Expanding operations not yet implemented")` decorators from existing tests (lines 14, 29, 47, 66, 85)
   - The expanding operations ARE implemented (see `expanding.py`) - the tests were just never enabled
   - **Add new tests** for `expanding().median()` and `expanding().quantile()`

2. **test_ewm.py**:
   - **Remove** `@pytest.mark.skip(reason="EWM operations not yet implemented")` decorators from existing tests (lines 14, 29, 47, 62, 81, 100)
   - The EWM operations ARE implemented (see `ewm.py`) - the tests were just never enabled
   - **Add new tests** for `ewm().corr()` and `ewm().cov()`

3. **test_stats.py**:
   - **Add tests** for `df.skew()`, `df.kurt()`, `df.sem()` (these will work once T1 enables the patches)

**Test pattern** (from existing tests):
```python
def test_xxx(self):
    import unlockedpd
    df = pd.DataFrame(np.random.randn(100, 10))

    unlockedpd.config.enabled = False
    expected = df.xxx()

    unlockedpd.config.enabled = True
    result = df.xxx()

    pd.testing.assert_frame_equal(result, expected, rtol=1e-10)
```

**Required tests**:
- [ ] Stats: skew, kurt, sem (enable tests now that patches work)
- [ ] Expanding: mean, sum, std, var, min, max (remove skip decorators), median, quantile (new)
- [ ] EWM: mean, std, var (remove skip decorators), corr, cov (new)
- [ ] Rolling: sem
- [ ] DataFrame: mean, std, var, sum (axis=0 and axis=1)
- [ ] Edge cases: NaN, empty, single column, single row

---

## Commit Strategy

| Commit | Description |
|--------|-------------|
| 1 | fix(stats): enable stats patches in __init__.py and update docstring |
| 2 | feat(expanding): add median and quantile operations |
| 3 | feat(rolling): add sem operation |
| 4 | feat(ewm): add pairwise corr and cov operations |
| 5 | feat(aggregates): add DataFrame mean/std/var/sum reductions |
| 6 | test: enable existing skipped tests, add tests for new operations |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical precision mismatch with pandas | MEDIUM | HIGH | Use rtol=1e-10, test with diverse data |
| ThreadPool deadlock | LOW | HIGH | Use context managers, set timeouts |
| Memory bloat with large pairwise ops | MEDIUM | MEDIUM | Chunk processing, lazy evaluation |
| Performance regression | LOW | MEDIUM | Benchmark before/after each change |
| GroupBy complexity underestimated | HIGH | LOW | Explicitly out of scope for this plan |
| O(n^2) expanding median too slow | LOW | MEDIUM | Acceptable for typical sizes; document limitation |

---

## Out of Scope (Deferred)

1. **GroupBy operations** - Requires significant architectural work (group detection, per-group aggregation). Should be a separate work plan.
2. **Rolling apply** - Requires user-defined function compilation, complex Numba interop
3. **Rolling rank** - Niche use case, lower priority
4. **DataFrame corr/cov** - Already have rolling versions, static versions less critical
5. **Non-linear quantile interpolation modes** - Start with `linear` only, fallback to pandas for others

---

## Success Criteria

1. **Functional**: All new operations produce identical output to pandas
2. **Performance**: >= 1.5x speedup on DataFrames with 10M+ elements
3. **Quality**: All tests pass, no regressions in existing functionality
4. **Coverage**: Stats bug fixed, expanding/EWM gaps closed, DataFrame reductions available

---

## Verification Commands

```bash
# Run all tests
cd /home/bellman/Workspace/MyNumbaDataFrame
pytest tests/ -v

# Run specific test file
pytest tests/test_stats.py -v
pytest tests/test_expanding.py -v
pytest tests/test_ewm.py -v

# Verify stats patch is applied
python -c "import unlockedpd; from unlockedpd._patch import is_patched; import pandas as pd; print('skew patched:', is_patched(pd.DataFrame, 'skew'))"

# Quick benchmark
python -c "
import pandas as pd
import numpy as np
import time

df = pd.DataFrame(np.random.randn(10000, 1000))

import unlockedpd
unlockedpd.config.enabled = False
start = time.time()
_ = df.rolling(20).mean()
pandas_time = time.time() - start

unlockedpd.config.enabled = True
start = time.time()
_ = df.rolling(20).mean()
optimized_time = time.time() - start

print(f'Pandas: {pandas_time:.3f}s, Optimized: {optimized_time:.3f}s, Speedup: {pandas_time/optimized_time:.1f}x')
"
```

---

*Plan generated by Prometheus*
*Timestamp: 2026-01-20*
*Revision: 2 (Momus feedback incorporated)*
