# Plan: Unimplemented Features to Enhance Pandas Coverage

## Context

### Original Request
Identify pandas numerical operations not yet optimized in unlockedpd and plan their implementation to enhance pandas coverage.

### Research Findings

#### Currently Implemented Operations (Active Patches)
Based on analysis of `/src/unlockedpd/__init__.py` and the ops modules:

| Category | Operations | Status |
|----------|------------|--------|
| **Rolling** | mean, sum, std, var, min, max, count, skew, kurt, median, quantile, corr, cov | IMPLEMENTED |
| **Expanding** | mean, sum, std, var, min, max, count, skew, kurt | IMPLEMENTED |
| **EWM** | mean, std, var (with span/halflife/alpha/com params) | IMPLEMENTED |
| **Cumulative** | cumsum, cumprod, cummin, cummax | IMPLEMENTED |
| **Transform** | diff, pct_change, shift | IMPLEMENTED |
| **Rank** | rank (all methods: average, min, max, first, dense) | IMPLEMENTED |
| **Aggregations** | sum, mean, std, var, min, max, median, prod | IMPLEMENTED |
| **Fillna** | ffill, bfill, fillna (scalar) | IMPLEMENTED |
| **Element-wise** | clip, abs, round | IMPLEMENTED |
| **Correlation** | DataFrame.corr, DataFrame.cov | IMPLEMENTED |
| **Pairwise** | rolling corr/cov | IMPLEMENTED |

#### Unimplemented Operations (from Skipped Tests)
Found 27 skipped tests indicating planned but unimplemented features:

1. **Apply Operations** (4 tests)
   - `test_basic_apply_axis0`
   - `test_basic_apply_axis1`
   - `test_apply_numpy_function`
   - `test_apply_lambda`
   - Reason: "Apply operations implementation needs verification"

2. **Transform Operations** (3 tests)
   - `test_basic_transform`
   - `test_transform_multiple_functions`
   - `test_transform_dict`
   - Reason: "Transform operations not yet implemented"

3. **Pipe Operations** (2 tests)
   - `test_basic_pipe`
   - `test_pipe_chaining`
   - Reason: "Pipe operations not yet implemented"

4. **Agg Operations** (3 tests)
   - `test_basic_agg`
   - `test_agg_multiple_functions`
   - `test_agg_dict`
   - Reason: "Agg operations not yet implemented"

5. **Quantile Operations** (2 tests)
   - `test_basic_quantile`
   - `test_quantile_multiple`
   - Reason: "Quantile operations not yet implemented"

**Note:** The tests in `test_expanding.py`, `test_ewm.py`, and `test_stats.py` marked as skipped appear to be stale - the operations ARE implemented in the actual code modules. These test files need updating to remove incorrect skip markers.

---

## Work Objectives

### Core Objective
Implement missing pandas numerical operations to increase pandas API coverage, focusing on operations that offer meaningful speedup potential.

### Deliverables
1. Optimized `DataFrame.agg()` / `DataFrame.aggregate()` implementation
2. Optimized `DataFrame.transform()` implementation
3. Optimized `DataFrame.quantile()` implementation
4. Enhanced `DataFrame.apply()` (currently partial)
5. Optimized `DataFrame.pipe()` (if beneficial)
6. Test file cleanup (remove stale skip markers)

### Definition of Done
- All new operations pass pandas compatibility tests
- Performance benchmarks show >= 1.5x speedup for large DataFrames
- Full parameter support matching pandas API
- Tests unskipped and passing

---

## Must Have / Must NOT Have

### Must Have
- Pandas API compatibility for all parameters
- NaN handling matching pandas behavior
- Both axis=0 and axis=1 support where applicable
- 3-tier dispatch (serial/parallel/threadpool) for optimal performance
- Comprehensive test coverage

### Must NOT Have
- Operations with marginal speedup potential (< 1.2x)
- Breaking changes to existing API
- Dependencies beyond numpy/numba
- Support for non-numeric operations

---

## Feature Priority Ranking

### HIGH Priority (Common operations, high speedup potential)

| Feature | Usage Frequency | Speedup Potential | Complexity |
|---------|-----------------|-------------------|------------|
| `DataFrame.agg()` | Very High | HIGH (parallel column processing) | Medium |
| `DataFrame.quantile()` | High | MEDIUM (parallel per-column quantile) | Low |
| Test cleanup (expanding/ewm) | N/A | N/A (correctness) | Low |

### MEDIUM Priority (Useful but specialized)

| Feature | Usage Frequency | Speedup Potential | Complexity |
|---------|-----------------|-------------------|------------|
| `DataFrame.transform()` | Medium | MEDIUM (depends on function) | Medium |
| Enhanced `DataFrame.apply()` | High | MEDIUM (JIT-able functions only) | High |

### LOW Priority (Limited optimization benefit)

| Feature | Usage Frequency | Speedup Potential | Complexity |
|---------|-----------------|-------------------|------------|
| `DataFrame.pipe()` | Low | LOW (just function composition) | Low |

---

## Task Flow and Dependencies

```
Phase 1: Foundation & Cleanup
    |
    +-- Task 1.1: Clean up stale skipped tests
    |
    +-- Task 1.2: Implement DataFrame.quantile()

Phase 2: Aggregation Enhancement
    |
    +-- Task 2.1: Implement DataFrame.agg() (single function)
    |
    +-- Task 2.2: Extend agg() for multiple functions
    |
    +-- Task 2.3: Extend agg() for dict-based column mapping

Phase 3: Transform Operations
    |
    +-- Task 3.1: Implement DataFrame.transform() (single function)
    |
    +-- Task 3.2: Extend transform() for multiple functions

Phase 4: Apply Enhancement (Optional)
    |
    +-- Task 4.1: Improve apply() JIT compilation coverage
    |
    +-- Task 4.2: Add raw=True optimization path
```

---

## Detailed TODOs

### Phase 1: Foundation & Cleanup

#### Task 1.1: Clean Up Stale Skipped Tests
**Priority:** HIGH
**Estimated Effort:** 1 hour

**Description:**
Remove `@pytest.mark.skip` decorators from tests that test already-implemented functionality:
- `tests/test_expanding.py` - Expanding operations ARE implemented
- `tests/test_ewm.py` - EWM operations ARE implemented
- `tests/test_stats.py` (TestCorr, TestCov) - Correlation operations ARE implemented

**Acceptance Criteria:**
- [ ] All tests for implemented operations run (not skipped)
- [ ] All unskipped tests pass
- [ ] No regressions in existing functionality

**Files to Modify:**
- `/home/bellman/Workspace/unlockedpd-numerical-ops/tests/test_expanding.py`
- `/home/bellman/Workspace/unlockedpd-numerical-ops/tests/test_ewm.py`
- `/home/bellman/Workspace/unlockedpd-numerical-ops/tests/test_stats.py`

---

#### Task 1.2: Implement DataFrame.quantile()
**Priority:** HIGH
**Estimated Effort:** 3-4 hours

**Description:**
Implement optimized `DataFrame.quantile()` using Numba parallel kernels.

**Implementation Approach:**
1. Create `src/unlockedpd/ops/quantile.py` with:
   - `_quantile_serial()` - Serial kernel for small arrays
   - `_quantile_parallel()` - prange kernel for medium arrays
   - `_quantile_threadpool()` - nogil kernel for large arrays
2. Support parameters: `q` (single or list), `axis`, `numeric_only`, `interpolation`
3. Use quickselect or partial sort for O(n) median finding

**Acceptance Criteria:**
- [ ] `df.quantile(0.5)` returns Series matching pandas
- [ ] `df.quantile([0.25, 0.5, 0.75])` returns DataFrame matching pandas
- [ ] axis=0 and axis=1 support
- [ ] NaN handling matches pandas
- [ ] >= 1.5x speedup on 1M+ element DataFrames

**Files to Create:**
- `/home/bellman/Workspace/unlockedpd-numerical-ops/src/unlockedpd/ops/quantile.py`

**Files to Modify:**
- `/home/bellman/Workspace/unlockedpd-numerical-ops/src/unlockedpd/__init__.py` (add patch)
- `/home/bellman/Workspace/unlockedpd-numerical-ops/tests/test_stats.py` (unskip and expand tests)

---

### Phase 2: Aggregation Enhancement

#### Task 2.1: Implement DataFrame.agg() (Single Function)
**Priority:** HIGH
**Estimated Effort:** 4-5 hours

**Description:**
Implement `DataFrame.agg()` / `DataFrame.aggregate()` for single string function names.

**Implementation Approach:**
1. Create `src/unlockedpd/ops/agg.py`
2. Map string function names ('sum', 'mean', 'std', etc.) to existing optimized implementations
3. Dispatch to appropriate optimized function from `aggregations.py`
4. Support: `agg('sum')`, `agg('mean')`, etc.

**Acceptance Criteria:**
- [ ] `df.agg('sum')` returns Series matching pandas
- [ ] `df.agg('mean')` returns Series matching pandas
- [ ] All standard aggregation functions supported
- [ ] Falls back to pandas for unsupported functions

**Files to Create:**
- `/home/bellman/Workspace/unlockedpd-numerical-ops/src/unlockedpd/ops/agg.py`

---

#### Task 2.2: Extend agg() for Multiple Functions
**Priority:** MEDIUM
**Estimated Effort:** 3-4 hours

**Description:**
Extend `DataFrame.agg()` to support list of functions.

**Implementation Approach:**
1. Support `df.agg(['sum', 'mean', 'std'])`
2. Run each aggregation in sequence (or parallel if independent)
3. Stack results into DataFrame with function names as index

**Acceptance Criteria:**
- [ ] `df.agg(['sum', 'mean'])` returns DataFrame matching pandas
- [ ] Multiple functions run efficiently
- [ ] Result index matches pandas behavior

---

#### Task 2.3: Extend agg() for Dict-based Column Mapping
**Priority:** MEDIUM
**Estimated Effort:** 3-4 hours

**Description:**
Extend `DataFrame.agg()` to support dict mapping columns to functions.

**Implementation Approach:**
1. Support `df.agg({'col_a': 'sum', 'col_b': 'mean'})`
2. Apply appropriate function to each specified column
3. Return Series with column names as index

**Acceptance Criteria:**
- [ ] `df.agg({'a': 'sum', 'b': 'mean'})` matches pandas
- [ ] Missing columns handled correctly
- [ ] Multiple functions per column supported

---

### Phase 3: Transform Operations

#### Task 3.1: Implement DataFrame.transform() (Single Function)
**Priority:** MEDIUM
**Estimated Effort:** 4-5 hours

**Description:**
Implement `DataFrame.transform()` for single function that returns same-shape output.

**Implementation Approach:**
1. Create `src/unlockedpd/ops/dataframe_transform.py`
2. For string functions: map to optimized element-wise operations
3. For callables: attempt JIT compilation, fallback to pandas
4. Key difference from apply: transform preserves shape

**Acceptance Criteria:**
- [ ] `df.transform(np.sqrt)` matches pandas
- [ ] `df.transform('abs')` matches pandas
- [ ] Shape is always preserved
- [ ] Falls back gracefully for non-JIT-able functions

---

#### Task 3.2: Extend transform() for Multiple Functions
**Priority:** LOW
**Estimated Effort:** 3-4 hours

**Description:**
Extend `transform()` to support list or dict of functions.

**Acceptance Criteria:**
- [ ] `df.transform(['sqrt', 'exp'])` creates multi-level column output
- [ ] `df.transform({'a': 'sqrt', 'b': 'exp'})` applies per-column

---

### Phase 4: Apply Enhancement (Optional)

#### Task 4.1: Improve apply() JIT Compilation Coverage
**Priority:** LOW
**Estimated Effort:** 6-8 hours

**Description:**
Enhance `apply.py` to support more function types.

**Current Limitation:** Only works with `raw=True` and JIT-able functions

**Implementation Approach:**
1. Add detection for common numpy ufuncs
2. Implement specialized kernels for common patterns
3. Better fallback detection and warnings

**Acceptance Criteria:**
- [ ] More numpy functions work without explicit `@unlockedpd.jit`
- [ ] Clear error messages for unsupported functions
- [ ] No performance regression for supported cases

---

## Commit Strategy

1. **Commit 1:** "fix: remove stale skip markers from tests for implemented features"
2. **Commit 2:** "feat: add optimized DataFrame.quantile() with 3-tier dispatch"
3. **Commit 3:** "feat: add DataFrame.agg() for single function aggregation"
4. **Commit 4:** "feat: extend agg() to support multiple functions"
5. **Commit 5:** "feat: extend agg() to support dict column mapping"
6. **Commit 6:** "feat: add DataFrame.transform() for element-wise operations"

---

## Success Criteria

### Functional Requirements
- [ ] All new operations match pandas output exactly
- [ ] Full parameter support for each operation
- [ ] NaN handling matches pandas behavior
- [ ] Works with both small and large DataFrames

### Performance Requirements
- [ ] `quantile()`: >= 2x speedup on 1M+ elements
- [ ] `agg()`: >= 1.5x speedup on 1M+ elements (dispatches to optimized aggregations)
- [ ] `transform()`: >= 2x speedup for JIT-able functions

### Quality Requirements
- [ ] All tests pass (including previously skipped)
- [ ] No regressions in existing functionality
- [ ] Code follows existing patterns in codebase
- [ ] Proper error handling and fallbacks

---

## Additional Pandas Operations for Future Consideration

These operations were not found in current skipped tests but are commonly used:

| Operation | Priority | Notes |
|-----------|----------|-------|
| `describe()` | Low | Can be built from existing aggregations |
| `nlargest() / nsmallest()` | Medium | Heap-based selection |
| `idxmin() / idxmax()` | Medium | Argmin/argmax with index |
| `mode()` | Low | Statistical mode |
| `value_counts()` | Medium | Histogram-like counting |
| `nunique()` | Low | Count unique values |
| `resample()` | High | Time series resampling |
| `groupby()` | High | Group-based operations (complex) |

---

## Notes

1. **Stale Tests:** The test files appear to have been created before implementations were completed. Many skipped tests test functionality that IS implemented in the ops modules.

2. **Apply Complexity:** The `apply()` function is inherently complex because it must handle arbitrary Python callables. Full optimization requires either user cooperation (`@unlockedpd.jit`) or aggressive fallback.

3. **GroupBy:** Not currently in scope but would be the highest-impact addition for pandas coverage. Would require significant new infrastructure.

4. **Pipe:** Low value for optimization - it's just syntactic sugar for function composition and has no computational overhead.
