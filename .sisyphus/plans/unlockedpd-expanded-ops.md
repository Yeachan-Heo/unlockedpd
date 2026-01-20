# Work Plan: Expand unlockedpd with Additional Optimized Operations

## Context

### Original Request
Expand unlockedpd to optimize additional pandas operations beyond the existing rolling (sum, mean, std, var, min, max), rank, and apply operations. Create a comprehensive implementation plan covering all pandas Series/DataFrame operations that could benefit from Numba parallelization.

### Current Implementation Analysis

The existing codebase follows a well-established pattern:
- **Location**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/`
- **Pattern**: `@njit(parallel=True, cache=True)` with `prange` for parallel operations
- **Dispatch**: Serial/parallel dispatch based on `PARALLEL_THRESHOLD` (10,000 elements)
- **Fallback**: Automatic fallback to pandas on any error
- **Compatibility**: `_compat.py` handles numeric column extraction and result wrapping

### Existing Optimized Operations
| Category | Operations | File |
|----------|-----------|------|
| Rolling | sum, mean, std, var, min, max | `ops/rolling.py` |
| Rank | all methods (average, min, max, first, dense) | `ops/rank.py` |
| Apply | raw numeric apply with jit-decorated functions | `ops/apply.py` |

---

## Comprehensive Operation Catalog

### Category 1: Cumulative Operations (Priority: HIGH)
Operations that compute running aggregates down columns.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **cumsum** | `df.cumsum()` | Yes (across columns) | Low | Simple running sum |
| **cumprod** | `df.cumprod()` | Yes (across columns) | Low | May have overflow issues |
| **cummax** | `df.cummax()` | Yes (across columns) | Low | Running maximum |
| **cummin** | `df.cummin()` | Yes (across columns) | Low | Running minimum |

**Benefit**: Very high - these are O(n) per column and trivially parallel across columns.

### Category 2: Expanding Window Operations (Priority: HIGH)
Expanding windows compute aggregates from start to current position.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **expanding.sum** | `df.expanding().sum()` | Yes | Low | Equivalent to cumsum with min_periods |
| **expanding.mean** | `df.expanding().mean()` | Yes | Low | Running mean |
| **expanding.std** | `df.expanding().std()` | Yes | Medium | Welford's algorithm |
| **expanding.var** | `df.expanding().var()` | Yes | Medium | Welford's algorithm |
| **expanding.min** | `df.expanding().min()` | Yes | Low | Running min (same as cummin) |
| **expanding.max** | `df.expanding().max()` | Yes | Low | Running max (same as cummax) |
| **expanding.count** | `df.expanding().count()` | Yes | Low | Count non-NaN |
| **expanding.skew** | `df.expanding().skew()` | Yes | Medium | 3rd moment calculation |
| **expanding.kurt** | `df.expanding().kurt()` | Yes | Medium | 4th moment calculation |

**Benefit**: High - expanding operations are simpler than rolling and highly parallelizable.

### Category 3: Exponentially Weighted Moving (EWM) Operations (Priority: HIGH)
EWM operations use exponential decay weighting.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **ewm.mean** | `df.ewm(span=N).mean()` | Yes (columns) | Medium | Single-pass with alpha |
| **ewm.std** | `df.ewm(span=N).std()` | Yes (columns) | Medium | Need online variance |
| **ewm.var** | `df.ewm(span=N).var()` | Yes (columns) | Medium | Online variance |
| **ewm.corr** | `df.ewm(span=N).corr()` | Yes | High | Pairwise EWM correlation |
| **ewm.cov** | `df.ewm(span=N).cov()` | Yes | High | Pairwise EWM covariance |

**Benefit**: Very high - EWM operations are heavily used in finance and currently slow in pandas.

### Category 4: Rolling Window Extensions (Priority: MEDIUM)
Additional rolling operations beyond the current implementation.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **rolling.skew** | `df.rolling(N).skew()` | Yes | Medium | 3rd moment |
| **rolling.kurt** | `df.rolling(N).kurt()` | Yes | Medium | 4th moment |
| **rolling.count** | `df.rolling(N).count()` | Yes | Low | Count non-NaN in window |
| **rolling.median** | `df.rolling(N).median()` | Yes | High | Requires sorting/heap |
| **rolling.quantile** | `df.rolling(N).quantile(q)` | Yes | High | Requires partial sorting |
| **rolling.corr** | `df.rolling(N).corr(other)` | Yes | Medium | Pairwise rolling correlation |
| **rolling.cov** | `df.rolling(N).cov(other)` | Yes | Medium | Pairwise rolling covariance |

**Benefit**: Medium-High - skew/kurt/count are straightforward; median/quantile require more complex algorithms.

### Category 5: Basic Statistical Operations (Priority: MEDIUM)
Simple aggregation operations on entire DataFrame/Series.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **mean** | `df.mean()` | Yes (columns) | Low | Parallel reduction |
| **std** | `df.std()` | Yes (columns) | Low | Two-pass or Welford |
| **var** | `df.var()` | Yes (columns) | Low | Same as std |
| **sum** | `df.sum()` | Yes (columns) | Low | Parallel reduction |
| **min** | `df.min()` | Yes (columns) | Low | Parallel reduction |
| **max** | `df.max()` | Yes (columns) | Low | Parallel reduction |
| **skew** | `df.skew()` | Yes (columns) | Medium | 3rd moment |
| **kurt** | `df.kurt()` | Yes (columns) | Medium | 4th moment |
| **sem** | `df.sem()` | Yes (columns) | Low | std / sqrt(n) |

**Benefit**: Medium - pandas is already reasonably fast for these, but multi-column parallel helps.

### Category 6: Pairwise Statistical Operations (Priority: LOW)
Operations that compute relationships between columns.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **corr** | `df.corr()` | Yes (pairs) | High | O(n*m^2) pairwise |
| **cov** | `df.cov()` | Yes (pairs) | High | O(n*m^2) pairwise |

**Benefit**: Medium - O(n*m^2) complexity means significant work, parallelizable across pairs.

### Category 7: Transform Operations (Priority: LOW)
Element-wise or group-wise transformations.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **diff** | `df.diff(periods)` | Yes | Low | Simple subtraction |
| **pct_change** | `df.pct_change()` | Yes | Low | (x - x_lag) / x_lag |
| **shift** | `df.shift(periods)` | Yes | Low | Array slicing |
| **clip** | `df.clip(lower, upper)` | Yes | Low | Element-wise bounds |
| **abs** | `df.abs()` | Limited | Low | NumPy already vectorized |

**Benefit**: Low-Medium - most are already vectorized in NumPy; minimal improvement expected.

### Category 8: Advanced Window Operations (Priority: LOW - Future)
Complex operations that may require significant development.

| Operation | Pandas Method | Parallelizable | Complexity | Notes |
|-----------|--------------|----------------|------------|-------|
| **rolling.apply** | `df.rolling(N).apply(func)` | Depends | High | User function dependent |
| **resample** | `df.resample('D').mean()` | Partial | High | Time-based grouping |
| **groupby.transform** | `df.groupby(col).transform('mean')` | Partial | High | Group-wise operations |

**Benefit**: Varies - complex to implement correctly; defer to future versions.

---

## Implementation Plan

### Phase 1: Cumulative Operations (Estimated: 2-3 hours)

#### TODO 1.1: Create `ops/cumulative.py`

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/cumulative.py`

**Operations**: cumsum, cumprod, cummin, cummax

**Implementation Pattern**:
```python
@njit(parallel=True, cache=True)
def _cumsum_2d(arr: np.ndarray, skipna: bool = True) -> np.ndarray:
    """Cumulative sum across columns in parallel."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for col in prange(n_cols):
        cumsum = 0.0
        for row in range(n_rows):
            val = arr[row, col]
            if np.isnan(val):
                if skipna:
                    result[row, col] = cumsum
                else:
                    result[row, col] = np.nan
                    cumsum = np.nan
            else:
                cumsum += val
                result[row, col] = cumsum
    return result
```

**Acceptance Criteria**:
- [ ] cumsum matches `df.cumsum()` for all numeric dtypes
- [ ] cumprod matches `df.cumprod()` with overflow handling
- [ ] cummin/cummax match pandas behavior including NaN handling
- [ ] skipna parameter supported (default True)
- [ ] axis parameter supported (0=down columns, 1=across rows)
- [ ] Serial/parallel dispatch based on PARALLEL_THRESHOLD
- [ ] Tests pass for edge cases: all NaN, single column, empty DataFrame

#### TODO 1.2: Patch Cumulative Methods

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/cumulative.py`

```python
def apply_cumulative_patches():
    """Apply cumulative operation patches to pandas."""
    from .._patch import patch

    patch(pd.DataFrame, 'cumsum', _patched_cumsum)
    patch(pd.DataFrame, 'cumprod', _patched_cumprod)
    patch(pd.DataFrame, 'cummin', _patched_cummin)
    patch(pd.DataFrame, 'cummax', _patched_cummax)
```

**Acceptance Criteria**:
- [ ] Patches applied on `import unlockedpd`
- [ ] Fallback to pandas on error
- [ ] config.enabled=False uses original pandas

#### TODO 1.3: Add Warmup for Cumulative

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/_warmup.py`

Add `warmup_cumulative()` function and call from `warmup_all()`.

---

### Phase 2: Expanding Window Operations (Estimated: 3-4 hours)

#### TODO 2.1: Create `ops/expanding.py`

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/expanding.py`

**Operations**: sum, mean, std, var, min, max, count, skew, kurt

**Key Insight**: Expanding is just rolling with window=current_position, but more efficiently implemented as cumulative.

**Implementation Pattern**:
```python
@njit(parallel=True, cache=True)
def _expanding_mean_2d(arr: np.ndarray, min_periods: int = 1) -> np.ndarray:
    """Expanding mean across columns in parallel."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        cumsum = 0.0
        count = 0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                cumsum += val
                count += 1
            if count >= min_periods:
                result[row, col] = cumsum / count
    return result

@njit(parallel=True, cache=True)
def _expanding_std_welford_2d(arr: np.ndarray, min_periods: int = 1, ddof: int = 1) -> np.ndarray:
    """Expanding std using Welford's algorithm."""
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

            if count >= min_periods and count > ddof:
                result[row, col] = np.sqrt(M2 / (count - ddof))
    return result
```

**Skew and Kurt Implementation**:
```python
@njit(parallel=True, cache=True)
def _expanding_skew_2d(arr: np.ndarray, min_periods: int = 3) -> np.ndarray:
    """Expanding skewness using online algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        n = 0
        M1 = 0.0  # Mean
        M2 = 0.0  # Variance * n
        M3 = 0.0  # 3rd central moment * n

        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                n1 = n
                n += 1
                delta = val - M1
                delta_n = delta / n
                term1 = delta * delta_n * n1
                M1 += delta_n
                M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
                M2 += term1

            if n >= min_periods:
                if M2 > 0:
                    skew = np.sqrt(n) * M3 / (M2 ** 1.5)
                    # Bias correction factor
                    result[row, col] = skew * np.sqrt(n * (n - 1)) / (n - 2)
    return result
```

**Acceptance Criteria**:
- [ ] All expanding operations match pandas within numerical tolerance
- [ ] min_periods parameter works correctly
- [ ] Welford's algorithm used for std/var for numerical stability
- [ ] Online algorithms for skew/kurt
- [ ] Serial/parallel dispatch

#### TODO 2.2: Patch Expanding Methods

Patch `pd.core.window.expanding.Expanding` class methods.

**Acceptance Criteria**:
- [ ] All methods: sum, mean, std, var, min, max, count, skew, kurt
- [ ] Fallback on unsupported parameters
- [ ] Tests for pandas compatibility

---

### Phase 3: EWM Operations (Estimated: 4-5 hours)

#### TODO 3.1: Create `ops/ewm.py`

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/ewm.py`

**Operations**: mean, std, var (corr/cov deferred due to complexity)

**EWM Alpha Calculation**:
```python
def _get_alpha(span=None, halflife=None, alpha=None, com=None):
    """Convert EWM parameters to alpha."""
    if alpha is not None:
        return alpha
    elif span is not None:
        return 2.0 / (span + 1)
    elif halflife is not None:
        return 1 - np.exp(-np.log(2) / halflife)
    elif com is not None:
        return 1.0 / (com + 1)
    else:
        raise ValueError("Must specify span, halflife, alpha, or com")
```

**EWM Mean Implementation**:
```python
@njit(parallel=True, cache=True)
def _ewm_mean_2d(arr: np.ndarray, alpha: float, adjust: bool = True,
                  ignore_na: bool = False, min_periods: int = 0) -> np.ndarray:
    """Exponentially weighted mean across columns in parallel.

    When adjust=True:
        y_t = (x_t + (1-alpha)*x_{t-1} + (1-alpha)^2*x_{t-2} + ...) /
              (1 + (1-alpha) + (1-alpha)^2 + ...)

    When adjust=False:
        y_t = (1-alpha)*y_{t-1} + alpha*x_t
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        if adjust:
            weighted_sum = 0.0
            weight_sum = 0.0
            decay = 1.0
            count = 0

            for row in range(n_rows):
                val = arr[row, col]
                if not np.isnan(val):
                    weighted_sum = weighted_sum * (1 - alpha) + val
                    weight_sum = weight_sum * (1 - alpha) + 1.0
                    count += 1
                elif ignore_na:
                    weighted_sum = weighted_sum * (1 - alpha)
                    weight_sum = weight_sum * (1 - alpha)

                if count >= min_periods:
                    result[row, col] = weighted_sum / weight_sum
        else:
            # Non-adjusted (recursive)
            ewm = np.nan
            count = 0
            for row in range(n_rows):
                val = arr[row, col]
                if not np.isnan(val):
                    count += 1
                    if np.isnan(ewm):
                        ewm = val
                    else:
                        ewm = alpha * val + (1 - alpha) * ewm
                elif not ignore_na:
                    ewm = np.nan

                if count >= min_periods:
                    result[row, col] = ewm
    return result
```

**EWM Variance (for std)**:
```python
@njit(parallel=True, cache=True)
def _ewm_var_2d(arr: np.ndarray, alpha: float, adjust: bool = True,
                 bias: bool = False, min_periods: int = 0) -> np.ndarray:
    """Exponentially weighted variance using online algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        mean = 0.0
        var = 0.0
        sum_weights = 0.0
        sum_weights_sq = 0.0
        count = 0

        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                count += 1
                weight = 1.0
                sum_weights_old = sum_weights
                sum_weights = sum_weights * (1 - alpha) + weight
                sum_weights_sq = sum_weights_sq * (1 - alpha)**2 + weight**2

                delta = val - mean
                mean += (weight / sum_weights) * delta
                var = (1 - alpha) * (var + (weight * delta * delta * sum_weights_old / sum_weights))

            if count >= min_periods:
                if bias:
                    result[row, col] = var / sum_weights
                else:
                    # Unbiased correction
                    correction = sum_weights ** 2 / (sum_weights ** 2 - sum_weights_sq)
                    result[row, col] = var * correction / sum_weights
    return result
```

**Acceptance Criteria**:
- [ ] ewm.mean() matches pandas for span, halflife, alpha, com parameters
- [ ] adjust=True/False behavior matches pandas exactly
- [ ] ignore_na parameter supported
- [ ] ewm.std() and ewm.var() with bias correction
- [ ] min_periods parameter works
- [ ] Serial/parallel dispatch

#### TODO 3.2: Patch EWM Methods

Patch `pd.core.window.ewm.ExponentialMovingWindow` class.

---

### Phase 4: Rolling Extensions (Estimated: 4-5 hours)

#### TODO 4.1: Add rolling.skew and rolling.kurt

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/rolling.py` (extend existing)

**Rolling Skew**:
```python
@njit(parallel=True, cache=True)
def _rolling_skew_2d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Rolling skewness using online algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        # Use circular buffer for window
        buffer = np.empty(window, dtype=np.float64)
        buffer[:] = np.nan
        buf_idx = 0

        for row in range(n_rows):
            buffer[buf_idx] = arr[row, col]
            buf_idx = (buf_idx + 1) % window

            if row >= min_periods - 1:
                # Compute skew from buffer
                n = 0
                mean = 0.0
                M2 = 0.0
                M3 = 0.0

                for i in range(window):
                    val = buffer[i]
                    if not np.isnan(val):
                        n1 = n
                        n += 1
                        delta = val - mean
                        delta_n = delta / n
                        term1 = delta * delta_n * n1
                        mean += delta_n
                        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
                        M2 += term1

                if n >= min_periods and M2 > 0:
                    skew = np.sqrt(n) * M3 / (M2 ** 1.5)
                    # Fisher's bias correction
                    result[row, col] = skew * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else np.nan
    return result
```

**Acceptance Criteria**:
- [ ] rolling.skew() matches pandas
- [ ] rolling.kurt() matches pandas (similar implementation)
- [ ] rolling.count() simple count of non-NaN in window
- [ ] Tests for various window sizes and data patterns

#### TODO 4.2: Add rolling.median and rolling.quantile (DEFERRED)

**Status**: Deferred due to complexity. Rolling median requires maintaining a sorted data structure (heap or skip list) which is complex in Numba.

**Notes for future**:
- Consider using a two-heap approach (max-heap for lower half, min-heap for upper half)
- Numba does not have built-in heap support; would need custom implementation

---

### Phase 5: Basic Statistical Operations (Estimated: 2-3 hours)

#### TODO 5.1: Create `ops/stats.py`

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/stats.py`

**Operations**: mean, std, var, sum, min, max, skew, kurt, sem

**Note**: Many of these are already well-optimized in pandas/numpy. Focus on:
1. `df.skew()` and `df.kurt()` - less optimized in pandas
2. Multi-column parallel processing

**Implementation**:
```python
@njit(parallel=True, cache=True)
def _skew_2d_axis0(arr: np.ndarray) -> np.ndarray:
    """Compute skewness for each column in parallel."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in prange(n_cols):
        n = 0
        M1 = 0.0
        M2 = 0.0
        M3 = 0.0

        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                n1 = n
                n += 1
                delta = val - M1
                delta_n = delta / n
                term1 = delta * delta_n * n1
                M1 += delta_n
                M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
                M2 += term1

        if n > 2 and M2 > 0:
            skew = np.sqrt(n) * M3 / (M2 ** 1.5)
            result[col] = skew * np.sqrt(n * (n - 1)) / (n - 2)
        else:
            result[col] = np.nan
    return result
```

**Acceptance Criteria**:
- [ ] df.skew() matches pandas
- [ ] df.kurt() matches pandas
- [ ] axis=0 and axis=1 supported
- [ ] skipna parameter works
- [ ] Serial/parallel dispatch

---

### Phase 6: Pairwise Correlation/Covariance (Estimated: 3-4 hours)

#### TODO 6.1: Create `ops/pairwise.py`

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/pairwise.py`

**Operations**: df.corr(), df.cov()

**Implementation Strategy**: Parallelize across column pairs.

```python
@njit(parallel=True, cache=True)
def _corr_matrix(arr: np.ndarray, min_periods: int = 1) -> np.ndarray:
    """Compute correlation matrix with parallel pair computation."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_cols, n_cols), dtype=np.float64)

    # Pre-compute means and stds for efficiency
    means = np.empty(n_cols, dtype=np.float64)
    stds = np.empty(n_cols, dtype=np.float64)
    counts = np.empty(n_cols, dtype=np.int64)

    for col in range(n_cols):
        n = 0
        mean = 0.0
        M2 = 0.0
        for row in range(n_rows):
            val = arr[row, col]
            if not np.isnan(val):
                n += 1
                delta = val - mean
                mean += delta / n
                M2 += delta * (val - mean)
        means[col] = mean
        stds[col] = np.sqrt(M2 / (n - 1)) if n > 1 else 0.0
        counts[col] = n

    # Compute pairwise correlations in parallel
    # Flatten to 1D iteration for prange
    n_pairs = n_cols * (n_cols + 1) // 2

    for col_i in prange(n_cols):
        for col_j in range(col_i, n_cols):
            if col_i == col_j:
                result[col_i, col_j] = 1.0
            else:
                # Compute correlation between col_i and col_j
                sum_xy = 0.0
                n = 0
                for row in range(n_rows):
                    val_i = arr[row, col_i]
                    val_j = arr[row, col_j]
                    if not np.isnan(val_i) and not np.isnan(val_j):
                        sum_xy += (val_i - means[col_i]) * (val_j - means[col_j])
                        n += 1

                if n >= min_periods and stds[col_i] > 0 and stds[col_j] > 0:
                    corr = sum_xy / ((n - 1) * stds[col_i] * stds[col_j])
                    result[col_i, col_j] = corr
                    result[col_j, col_i] = corr
                else:
                    result[col_i, col_j] = np.nan
                    result[col_j, col_i] = np.nan

    return result
```

**Acceptance Criteria**:
- [ ] df.corr() matches pandas for method='pearson'
- [ ] df.cov() matches pandas
- [ ] min_periods parameter works
- [ ] numeric_only parameter handled (via _compat.py pattern)
- [ ] Result is symmetric matrix with 1.0 on diagonal (for corr)

---

### Phase 7: Transform Operations (Estimated: 2-3 hours)

#### TODO 7.1: Create `ops/transform.py`

**File**: `/home/bellman/Workspace/MyNumbaDataFrame/src/unlockedpd/ops/transform.py`

**Operations**: diff, pct_change, shift

**Note**: These are simpler operations. The main benefit is parallel processing across many columns.

```python
@njit(parallel=True, cache=True)
def _diff_2d(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute difference with lag in parallel across columns."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        if periods > 0:
            for row in range(periods, n_rows):
                result[row, col] = arr[row, col] - arr[row - periods, col]
        else:
            for row in range(n_rows + periods):
                result[row, col] = arr[row, col] - arr[row - periods, col]
    return result

@njit(parallel=True, cache=True)
def _pct_change_2d(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute percentage change in parallel across columns."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        if periods > 0:
            for row in range(periods, n_rows):
                prev = arr[row - periods, col]
                if prev != 0 and not np.isnan(prev):
                    result[row, col] = (arr[row, col] - prev) / prev
    return result
```

**Acceptance Criteria**:
- [ ] df.diff(periods) matches pandas for positive and negative periods
- [ ] df.pct_change(periods) matches pandas
- [ ] NaN handling matches pandas behavior
- [ ] Serial/parallel dispatch

---

## Testing Strategy

### Test File Structure

```
tests/
  test_cumulative.py      # Phase 1
  test_expanding.py       # Phase 2
  test_ewm.py            # Phase 3
  test_rolling_ext.py    # Phase 4 (skew, kurt, count)
  test_stats.py          # Phase 5
  test_pairwise.py       # Phase 6
  test_transform.py      # Phase 7
```

### Standard Test Pattern

Each test file should include:

```python
class TestOperationName:
    def test_basic_operation(self):
        """Test basic operation matches pandas."""
        import unlockedpd
        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.operation()

        unlockedpd.config.enabled = True
        result = df.operation()

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_with_nan(self):
        """Test NaN handling matches pandas."""
        # ...

    def test_edge_cases(self):
        """Test edge cases: empty, single row/col, all NaN."""
        # ...

    def test_parameters(self):
        """Test various parameter combinations."""
        # ...
```

---

## Integration Points

### Update `__init__.py`

```python
# src/unlockedpd/__init__.py

def _apply_all_patches():
    """Apply all optimization patches to pandas."""
    from .ops.rolling import apply_rolling_patches
    from .ops.rank import apply_rank_patches
    from .ops.apply import apply_apply_patches
    from .ops.cumulative import apply_cumulative_patches      # NEW
    from .ops.expanding import apply_expanding_patches        # NEW
    from .ops.ewm import apply_ewm_patches                    # NEW
    from .ops.stats import apply_stats_patches                # NEW
    from .ops.pairwise import apply_pairwise_patches          # NEW
    from .ops.transform import apply_transform_patches        # NEW

    apply_rolling_patches()
    apply_rank_patches()
    apply_apply_patches()
    apply_cumulative_patches()   # NEW
    apply_expanding_patches()    # NEW
    apply_ewm_patches()          # NEW
    apply_stats_patches()        # NEW
    apply_pairwise_patches()     # NEW
    apply_transform_patches()    # NEW
```

### Update `_warmup.py`

Add warmup functions for each new operation category.

### Update `ops/__init__.py`

Export new operation modules.

---

## Priority Summary

| Phase | Category | Priority | Est. Hours | Benefit |
|-------|----------|----------|------------|---------|
| 1 | Cumulative (cumsum, etc.) | HIGH | 2-3 | Very High |
| 2 | Expanding Window | HIGH | 3-4 | High |
| 3 | EWM Operations | HIGH | 4-5 | Very High |
| 4 | Rolling Extensions (skew, kurt) | MEDIUM | 4-5 | Medium-High |
| 5 | Basic Stats (skew, kurt) | MEDIUM | 2-3 | Medium |
| 6 | Pairwise (corr, cov) | LOW | 3-4 | Medium |
| 7 | Transform (diff, pct_change) | LOW | 2-3 | Low-Medium |

**Total Estimated Time**: 21-27 hours

---

## Deferred Operations (Future Versions)

| Operation | Reason for Deferral |
|-----------|---------------------|
| rolling.median | Requires complex heap data structure |
| rolling.quantile | Requires partial sorting algorithm |
| ewm.corr/cov | Complex pairwise EWM computation |
| groupby operations | Complex pandas internals |
| resample operations | Time-based indexing complexity |

---

## Commit Strategy

| Phase | Commit Message |
|-------|----------------|
| 1 | `feat(cumulative): add parallel cumsum, cumprod, cummin, cummax` |
| 2 | `feat(expanding): add parallel expanding window operations` |
| 3 | `feat(ewm): add parallel exponentially weighted operations` |
| 4 | `feat(rolling): add rolling skew, kurt, count operations` |
| 5 | `feat(stats): add parallel skew and kurtosis computation` |
| 6 | `feat(pairwise): add parallel correlation and covariance matrices` |
| 7 | `feat(transform): add parallel diff and pct_change` |

---

## Success Criteria

### Performance Targets
- All new operations should show >= 1.5x speedup on 100+ column DataFrames
- EWM operations should show >= 2x speedup (currently very slow in pandas)
- Cumulative operations should match or exceed numpy performance

### Quality Targets
- 100% pandas compatibility for supported parameter combinations
- Zero silent data corruption
- Clear fallback to pandas for unsupported cases

---

## Next Steps

Execute this plan with:
```
/start-work .sisyphus/plans/unlockedpd-expanded-ops.md
```

Or for automated execution:
```
/ralph-loop
```
