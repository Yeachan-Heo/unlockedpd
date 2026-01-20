# Work Plan: unlockedpd - Transparent Pandas Performance Optimization

## Context

### Original Request
Create "unlockedpd" - a pandas override/wrapper library that enables transparent performance optimization. Users import it alongside pandas, and it automatically patches DataFrame/Series methods with parallelized Numba implementations.

```python
import pandas as pd
import unlockedpd  # This import patches pandas automatically

df = pd.DataFrame(...)  # Now uses optimized methods transparently
df.rolling(20).mean()   # Parallelized automatically
df.rank(axis=1)         # Parallelized automatically
```

### Research Findings Summary

#### Pandas Internals
- **Data Storage**: DataFrames use internal arrays (historically BlockManager, evolving in pandas 2.x)
- **ExtensionArray**: Core interface for custom array types; can implement `__array_ufunc__` for NumPy integration
- **Custom Accessors**: Can register custom namespaces via `@pd.api.extensions.register_dataframe_accessor`
- **Priority System**: `__pandas_priority__` controls operation delegation (DataFrame=4000, Series=3000)
- **Copy-on-Write**: Pandas 2.0+ has CoW mode changing view/copy semantics

#### Numba Capabilities
- **`@jit(parallel=True)`**: Automatic parallelization of array operations, reductions, ufuncs
- **`prange`**: Explicit parallel loop ranges with automatic reduction inference
- **`@stencil`**: Rolling window operations with relative indexing and neighborhood definitions
- **`@guvectorize`**: Generalized ufuncs for complex array operations
- **Threading Layers**: TBB, OpenMP, or workqueue backends; configurable thread count
- **Limitations**: Homogeneous lists only, no class definitions in nopython mode, restricted exception handling

#### Existing Solutions Analysis

| Library | Approach | Pros | Cons |
|---------|----------|------|------|
| **Modin** | Distributed DataFrames (Ray/Dask) | Drop-in replacement, scales to TB | Heavy dependencies, changes execution model |
| **Swifter** | Smart apply() optimization | Auto-vectorization, benchmarks approaches | Only targets apply(), accessor-based |
| **Pandarallel** | Multiprocessing for apply/map | Simple API, progress bars | Process overhead, limited operations, unmaintained |

**Key Insight**: None of these provide Numba JIT compilation for core pandas methods. This is our differentiation.

### Gap Analysis (Metis Consultation)

Critical considerations incorporated:
1. **Version Compatibility**: Support pandas 1.5+ and 2.x with conditional patching
2. **Fallback Strategy**: Graceful degradation when Numba fails (dtype issues, compilation errors)
3. **CoW Compatibility**: Respect pandas 2.x Copy-on-Write semantics
4. **Thread Management**: Configurable parallelism to avoid oversubscription
5. **Disable Mechanism**: Ability to restore original methods for debugging

---

## Work Objectives

### Core Objective
Build a Python library that monkey-patches pandas DataFrame/Series methods with Numba-accelerated parallel implementations, providing 2-10x speedups on multi-core systems with zero API changes.

### Deliverables

1. **`unlockedpd` Python package** with:
   - Auto-patching on import
   - Parallel rolling operations (mean, std, sum, min, max)
   - Parallel rank operation
   - Parallel apply with Numba-compilable functions
   - Configuration API for thread count and feature toggles
   - Disable/restore mechanism

2. **Benchmark Suite** demonstrating:
   - Performance vs vanilla pandas across operation types
   - Scaling behavior with core count
   - Memory efficiency comparisons

3. **Documentation** including:
   - Installation guide
   - Usage examples
   - Configuration options
   - Compatibility matrix

### Definition of Done

- [ ] All targeted operations show measurable speedup (>1.5x) on 4+ core systems with 10K+ row, 100+ column DataFrames
- [ ] All pandas tests for patched methods pass with unlockedpd active
- [ ] No memory leaks in parallel operations (verified via memory profiling)
- [ ] Works with pandas 1.5.x, 2.0.x, 2.1.x, 2.2.x
- [ ] Works with Python 3.9, 3.10, 3.11, 3.12
- [ ] CI pipeline passing with matrix testing

---

## Guardrails

### MUST Have
- Zero API changes from user perspective (drop-in enhancement)
- Graceful fallback to original pandas on any Numba failure
- Thread-safe operations that don't corrupt data
- Configurable parallelism (respect user's thread limits)
- Support for common numeric dtypes: float64, float32, int64, int32
- Handle mixed-dtype DataFrames: skip non-numeric columns automatically, process only numeric subset
- Memory-efficient: no unnecessary copies of large arrays

### MUST NOT Have
- Breaking changes to pandas behavior or return types
- Hidden state that affects reproducibility
- Dependencies beyond numpy, numba, pandas, packaging (for version parsing)
- Automatic patching of methods we haven't explicitly tested
- Silent data corruption on edge cases

---

## Architecture Decisions

### AD-1: Patching Mechanism
**Decision**: Use method replacement on DataFrame/Series classes, not accessor pattern.

**Rationale**:
- Accessor pattern (like swifter) requires `df.swifter.apply()` - not transparent
- Method replacement achieves true drop-in behavior
- Store original methods for fallback/restoration

**Implementation**:
```python
# Store original
_original_rolling = pd.core.window.rolling.Rolling.mean

# Replace with optimized
pd.core.window.rolling.Rolling.mean = _optimized_rolling_mean
```

### AD-2: Fallback Strategy
**Decision**: Wrap all patched methods with try/except that falls back to original.

**Rationale**:
- Numba compilation can fail for various reasons (unsupported dtypes, complex expressions)
- User should never see errors from our optimization layer
- Log warnings (configurable) when fallback occurs

**Implementation**:
```python
def _optimized_rolling_mean(self, *args, **kwargs):
    try:
        return _numba_rolling_mean(self, *args, **kwargs)
    except (NumbaError, TypeError) as e:
        if unlockedpd.config.warn_on_fallback:
            warnings.warn(f"Falling back to pandas: {e}")
        return _original_rolling_mean(self, *args, **kwargs)
```

### AD-3: Parallel Implementation Pattern
**Decision**: Use `@njit(parallel=True)` with `prange` for column-wise parallelization.

**Rationale**:
- Rolling operations are naturally parallel across columns
- `prange` gives explicit control over parallelization axis
- `@stencil` is good for single-column but we need multi-column

**Implementation Pattern**:
```python
@njit(parallel=True, cache=True)
def _rolling_mean_2d(arr, window):
    n_rows, n_cols = arr.shape
    result = np.empty_like(arr)

    for col in prange(n_cols):  # Parallel across columns
        for row in range(n_rows):
            if row < window - 1:
                result[row, col] = np.nan
            else:
                result[row, col] = np.mean(arr[row-window+1:row+1, col])

    return result
```

### AD-4: Memory Layout Handling
**Decision**: Ensure C-contiguous arrays before parallel operations, avoid copies when possible.

**Rationale**:
- Numba parallel performs best on C-contiguous memory
- Column operations on C-order arrays have poor cache locality
- Use `np.ascontiguousarray()` only when necessary, warn user if copy required

### AD-5: Configuration System
**Decision**: Module-level configuration object with environment variable overrides.

```python
import unlockedpd
unlockedpd.config.num_threads = 4
unlockedpd.config.warn_on_fallback = True
unlockedpd.config.enabled = True  # Set False to disable all patches
```

Environment variables: `UNLOCKEDPD_NUM_THREADS`, `UNLOCKEDPD_ENABLED`

---

## Task Flow

```
[Phase 1: Foundation]
    |
    v
[1.1] Project scaffolding --> [1.2] Configuration system --> [1.3] Patch infrastructure
    |
    v
[Phase 2: Core Operations]
    |
    v
[2.1] Rolling operations --> [2.2] Rank operation --> [2.3] Apply optimization
    |
    v
[Phase 3: Quality & Polish]
    |
    v
[3.1] Benchmark suite --> [3.2] Pandas compatibility tests --> [3.3] Documentation
    |
    v
[Phase 4: Release]
    |
    v
[4.1] Package for PyPI --> [4.2] CI/CD setup
```

---

## Detailed TODOs

### Phase 1: Foundation (Priority: Critical)

#### TODO 1.1: Project Scaffolding
**Acceptance Criteria**:
- [ ] Standard Python package structure with `src/unlockedpd/`
- [ ] `pyproject.toml` with dependencies: `numpy>=1.21`, `numba>=0.56`, `pandas>=1.5`, `packaging>=21.0`
- [ ] Development dependencies: pytest, pytest-benchmark, memory_profiler
- [ ] `.gitignore` for Python projects
- [ ] Basic `__init__.py` that auto-patches on import

**Files to Create**:
```
unlockedpd/
  pyproject.toml
  src/
    unlockedpd/
      __init__.py
      _config.py
      _patch.py
      _compat.py
  tests/
    __init__.py
    conftest.py
  benchmarks/
    __init__.py
```

#### TODO 1.2: Configuration System
**Acceptance Criteria**:
- [ ] `UnlockedConfig` class with typed attributes
- [ ] Environment variable loading on module import
- [ ] `num_threads` setting that configures Numba's thread count
- [ ] `enabled` flag to globally disable patches
- [ ] `warn_on_fallback` flag for debugging
- [ ] Thread-safe configuration access using threading.Lock for mutable state

**Implementation Details**:
```python
# src/unlockedpd/_config.py
import os
import threading
from dataclasses import dataclass, field
import numba

@dataclass
class UnlockedConfig:
    """Thread-safe configuration for unlockedpd.

    Uses a lock for all mutable attribute access to ensure
    thread-safety when config is modified from multiple threads.
    """
    _enabled: bool = field(default=True, repr=False)
    _num_threads: int = field(default=0, repr=False)  # 0 = auto (numba default)
    _warn_on_fallback: bool = field(default=False, repr=False)
    _cache_compiled: bool = field(default=True, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        # Load from environment
        self._enabled = os.environ.get('UNLOCKEDPD_ENABLED', 'true').lower() == 'true'
        threads = os.environ.get('UNLOCKEDPD_NUM_THREADS', '0')
        self._num_threads = int(threads) if threads.isdigit() else 0

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        with self._lock:
            self._enabled = value

    @property
    def num_threads(self) -> int:
        with self._lock:
            return self._num_threads

    @num_threads.setter
    def num_threads(self, value: int):
        with self._lock:
            self._num_threads = value
            if value > 0:
                numba.set_num_threads(value)

    @property
    def warn_on_fallback(self) -> bool:
        with self._lock:
            return self._warn_on_fallback

    @warn_on_fallback.setter
    def warn_on_fallback(self, value: bool):
        with self._lock:
            self._warn_on_fallback = value

    @property
    def cache_compiled(self) -> bool:
        with self._lock:
            return self._cache_compiled

    @cache_compiled.setter
    def cache_compiled(self, value: bool):
        with self._lock:
            self._cache_compiled = value

    def apply_thread_config(self):
        with self._lock:
            if self._num_threads > 0:
                numba.set_num_threads(self._num_threads)

config = UnlockedConfig()
```

#### TODO 1.3: Patch Infrastructure
**Acceptance Criteria**:
- [ ] `_PatchRegistry` class tracking all applied patches
- [ ] `patch()` function to apply a single method patch with fallback wrapper
- [ ] `unpatch()` function to restore original method
- [ ] `unpatch_all()` to restore all original methods
- [ ] `is_patched()` query function
- [ ] Context manager for temporary unpatching

**Implementation Details**:
```python
# src/unlockedpd/_patch.py
from typing import Callable, Dict, Tuple, Any
import functools
import warnings

class _PatchRegistry:
    _patches: Dict[Tuple[type, str], Callable] = {}
    _originals: Dict[Tuple[type, str], Callable] = {}

    @classmethod
    def patch(cls, target_class: type, method_name: str,
              optimized_func: Callable, fallback: bool = True):
        key = (target_class, method_name)
        original = getattr(target_class, method_name)
        cls._originals[key] = original

        if fallback:
            @functools.wraps(original)
            def wrapper(self, *args, **kwargs):
                from . import config
                if not config.enabled:
                    return original(self, *args, **kwargs)
                try:
                    return optimized_func(self, *args, **kwargs)
                except Exception as e:
                    if config.warn_on_fallback:
                        warnings.warn(f"unlockedpd fallback for {method_name}: {e}")
                    return original(self, *args, **kwargs)
            replacement = wrapper
        else:
            replacement = optimized_func

        setattr(target_class, method_name, replacement)
        cls._patches[key] = replacement

    @classmethod
    def unpatch(cls, target_class: type, method_name: str):
        key = (target_class, method_name)
        if key in cls._originals:
            setattr(target_class, method_name, cls._originals[key])
            del cls._patches[key]
            del cls._originals[key]

    @classmethod
    def unpatch_all(cls):
        for (target_class, method_name), original in list(cls._originals.items()):
            setattr(target_class, method_name, original)
        cls._patches.clear()
        cls._originals.clear()
```

#### TODO 1.4: Pandas Compatibility Layer
**Acceptance Criteria**:
- [ ] Version detection for pandas (1.5.x vs 2.x)
- [ ] Version detection for numpy and numba
- [ ] Compatibility flags for CoW mode detection
- [ ] Helper to extract underlying numpy array from DataFrame/Series
- [ ] Helper to reconstruct DataFrame/Series from numpy result

**Implementation Details**:
```python
# src/unlockedpd/_compat.py
import pandas as pd
import numpy as np
from packaging import version

PANDAS_VERSION = version.parse(pd.__version__)
PANDAS_2 = PANDAS_VERSION >= version.parse("2.0.0")
COW_ENABLED = PANDAS_2 and pd.options.mode.copy_on_write

NUMERIC_DTYPES = (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8,
                  np.uint64, np.uint32, np.uint16, np.uint8)

def get_numeric_columns(df: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Extract numeric columns from DataFrame.

    Returns:
        tuple: (list of numeric column names, DataFrame with only numeric columns)

    For mixed-dtype DataFrames, this filters to only numeric columns.
    Non-numeric columns (object, datetime, categorical, etc.) are skipped.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols, df[numeric_cols]

def get_values(obj, numeric_only: bool = False):
    """Extract numpy array from DataFrame or Series.

    Args:
        obj: DataFrame or Series
        numeric_only: If True and obj is DataFrame, select only numeric columns

    Returns:
        numpy array (and optionally the numeric column names for DataFrames)
    """
    if isinstance(obj, pd.DataFrame):
        if numeric_only:
            numeric_cols, numeric_df = get_numeric_columns(obj)
            return numeric_df.values, numeric_cols
        return obj.values
    elif isinstance(obj, pd.Series):
        return obj.to_numpy()
    return np.asarray(obj)

def wrap_result(result: np.ndarray, like: pd.DataFrame | pd.Series,
                columns: list = None, merge_non_numeric: bool = False,
                original_df: pd.DataFrame = None):
    """Wrap numpy result back into pandas structure.

    Args:
        result: numpy array with computed values
        like: original DataFrame/Series for index/structure
        columns: column names for result (for numeric-only operations)
        merge_non_numeric: if True, merge result with non-numeric columns from original
        original_df: original DataFrame when merge_non_numeric=True
    """
    if isinstance(like, pd.DataFrame):
        result_df = pd.DataFrame(result, index=like.index, columns=columns or like.columns)

        if merge_non_numeric and original_df is not None:
            # Add back non-numeric columns as NaN (matching pandas behavior for numeric ops)
            non_numeric_cols = [c for c in original_df.columns if c not in result_df.columns]
            for col in non_numeric_cols:
                result_df[col] = np.nan
            # Restore original column order
            result_df = result_df[original_df.columns]

        return result_df
    elif isinstance(like, pd.Series):
        return pd.Series(result, index=like.index, name=like.name)
    return result
```

---

### Phase 2: Core Operations (Priority: High)

#### TODO 2.1: Parallel Rolling Operations
**Acceptance Criteria**:
- [ ] `rolling().mean()` parallelized across columns
- [ ] `rolling().sum()` parallelized across columns
- [ ] `rolling().std()` parallelized across columns (use Welford's algorithm)
- [ ] `rolling().min()` and `rolling().max()` parallelized
- [ ] Support for `min_periods` parameter
- [ ] Support for `center` parameter (see centering algorithm below)
- [ ] Handle NaN values correctly (match pandas behavior exactly)
- [ ] Handle edge case: `window > len(df)` returns all NaN (consistent with pandas)
- [ ] Benchmark showing >1.5x speedup on 100+ column DataFrames

**Implementation Approach**:

1. **Target Class**: `pd.core.window.rolling.Rolling`
2. **Methods to Patch**: `mean`, `sum`, `std`, `var`, `min`, `max`
3. **Core Algorithm**: For each method, create Numba-jitted function operating on 2D array

```python
# src/unlockedpd/ops/rolling.py
import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _rolling_mean_2d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Compute rolling mean across columns in parallel."""
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

            if row >= window:
                old_val = arr[row - window, col]
                if not np.isnan(old_val):
                    cumsum -= old_val
                    count -= 1

            if count >= min_periods:
                result[row, col] = cumsum / count

    return result

@njit(parallel=True, cache=True)
def _rolling_std_2d(arr: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    """Compute rolling std using Welford's online algorithm."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    for col in prange(n_cols):
        # Use circular buffer approach for numerically stable std
        buffer = np.empty(window, dtype=np.float64)
        buffer[:] = np.nan
        buf_idx = 0

        for row in range(n_rows):
            buffer[buf_idx] = arr[row, col]
            buf_idx = (buf_idx + 1) % window

            if row >= min_periods - 1:
                # Compute std from buffer
                valid_count = 0
                mean = 0.0
                for i in range(window):
                    if not np.isnan(buffer[i]):
                        mean += buffer[i]
                        valid_count += 1

                if valid_count >= min_periods:
                    mean /= valid_count
                    variance = 0.0
                    for i in range(window):
                        if not np.isnan(buffer[i]):
                            variance += (buffer[i] - mean) ** 2
                    if valid_count > ddof:
                        result[row, col] = np.sqrt(variance / (valid_count - ddof))

    return result

@njit(parallel=True, cache=True)
def _rolling_mean_2d_centered(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Compute centered rolling mean across columns in parallel.

    Centering algorithm:
    - For window W, we need (W-1)//2 values before AND after the current position
    - half_left = (window - 1) // 2
    - half_right = window // 2
    - For row i, the window spans [i - half_left, i + half_right] inclusive
    - First half_left rows get NaN (not enough values before)
    - Last half_right rows get NaN (not enough values after)
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    result[:] = np.nan

    half_left = (window - 1) // 2
    half_right = window // 2

    for col in prange(n_cols):
        for row in range(half_left, n_rows - half_right):
            cumsum = 0.0
            count = 0
            for k in range(row - half_left, row + half_right + 1):
                val = arr[k, col]
                if not np.isnan(val):
                    cumsum += val
                    count += 1

            if count >= min_periods:
                result[row, col] = cumsum / count

    return result
```

5. **Edge Case: window > len(df)**:
```python
# In the wrapper, check window size first
def _patched_rolling_mean(rolling_obj, *args, **kwargs):
    obj = rolling_obj.obj
    window = rolling_obj.window

    # If window > length, return all NaN (pandas behavior)
    if window > len(obj):
        return obj.copy() * np.nan  # Returns DataFrame of same shape, all NaN
    # ... rest of implementation
```

6. **Wrapper Integration (with mixed-dtype support)**:
```python
def _patched_rolling_mean(rolling_obj, *args, **kwargs):
    from .._compat import get_values, wrap_result, get_numeric_columns
    from .._config import config

    # Get underlying DataFrame/Series
    obj = rolling_obj.obj
    window = rolling_obj.window
    min_periods = rolling_obj.min_periods or window
    center = rolling_obj.center

    # Edge case: window > len(df)
    if window > len(obj):
        return obj.copy() * np.nan

    # Check if we can optimize (2D numeric data)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Optimization only for DataFrame")

    # Handle mixed-dtype DataFrames: extract only numeric columns
    numeric_cols, numeric_df = get_numeric_columns(obj)

    if len(numeric_cols) == 0:
        raise TypeError("No numeric columns to process")

    arr = numeric_df.values.astype(np.float64)  # Numba works best with float64

    # Choose centered or non-centered implementation
    if center:
        result = _rolling_mean_2d_centered(arr, window, min_periods)
    else:
        result = _rolling_mean_2d(arr, window, min_periods)

    # Wrap result, merging back non-numeric columns as NaN
    return wrap_result(result, numeric_df, columns=numeric_cols,
                       merge_non_numeric=True, original_df=obj)
```

#### TODO 2.2: Parallel Rank Operation
**Acceptance Criteria**:
- [ ] `df.rank(axis=1)` parallelized across rows
- [ ] `df.rank(axis=0)` parallelized across columns
- [ ] Support for `method` parameter: 'average', 'min', 'max', 'first', 'dense'
- [ ] Support for `na_option` parameter: 'keep', 'top', 'bottom'
- [ ] Support for `ascending` parameter
- [ ] Support for `pct` parameter (percentage ranks)
- [ ] Benchmark showing >1.5x speedup on wide DataFrames

**na_option Implementation Guidance**:
- `'keep'` (default): NaN values remain NaN in output
- `'top'`: NaN values receive the lowest ranks (1, 2, ... for multiple NaNs), other ranks shift up
- `'bottom'`: NaN values receive the highest ranks (n, n-1, ... for multiple NaNs)

**Algorithm**:
```python
# Step 1: Count NaNs and compute ranks for non-NaN values only
# Step 2: Based on na_option:
#   - 'keep': Leave NaN positions as NaN (default behavior)
#   - 'top': Shift all non-NaN ranks up by nan_count, assign 1..nan_count to NaN positions
#   - 'bottom': Keep non-NaN ranks as-is (1..n-nan_count), assign (n-nan_count+1)..n to NaN positions
```

**Full Implementation Approach**:

```python
# src/unlockedpd/ops/rank.py
import numpy as np
from numba import njit, prange

@njit(cache=True)
def _apply_na_option(ranks: np.ndarray, is_nan: np.ndarray, na_option: int) -> np.ndarray:
    """Apply na_option to ranks array.

    na_option encoding: 0='keep', 1='top', 2='bottom'
    """
    n = len(ranks)
    nan_count = 0
    for i in range(n):
        if is_nan[i]:
            nan_count += 1

    if na_option == 0:  # 'keep'
        # Already handled: NaN positions have NaN ranks
        pass
    elif na_option == 1:  # 'top' - NaNs get lowest ranks
        # Shift valid ranks up by nan_count
        for i in range(n):
            if not is_nan[i]:
                ranks[i] += nan_count
        # Assign lowest ranks to NaNs
        rank = 1.0
        for i in range(n):
            if is_nan[i]:
                ranks[i] = rank
                rank += 1.0
    elif na_option == 2:  # 'bottom' - NaNs get highest ranks
        # Valid ranks stay as-is (1..n-nan_count)
        # Assign highest ranks to NaNs
        rank = float(n - nan_count + 1)
        for i in range(n):
            if is_nan[i]:
                ranks[i] = rank
                rank += 1.0

    return ranks

@njit(parallel=True, cache=True)
def _rank_axis1_average(arr: np.ndarray, ascending: bool = True, na_option: int = 0) -> np.ndarray:
    """Rank values along axis=1 (across columns) using average method."""
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for row in prange(n_rows):
        row_data = arr[row, :].copy()

        # Get sort indices
        if ascending:
            sorted_idx = np.argsort(row_data)
        else:
            sorted_idx = np.argsort(-row_data)

        # Assign ranks (handling ties with average method)
        ranks = np.empty(n_cols, dtype=np.float64)
        i = 0
        while i < n_cols:
            j = i
            # Find all tied values
            while j < n_cols - 1 and row_data[sorted_idx[j]] == row_data[sorted_idx[j + 1]]:
                j += 1
            # Average rank for tied values
            avg_rank = (i + j + 2) / 2  # +2 because ranks are 1-based
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1

        # Track NaN positions
        is_nan = np.empty(n_cols, dtype=np.bool_)
        for col in range(n_cols):
            is_nan[col] = np.isnan(arr[row, col])
            if is_nan[col]:
                ranks[col] = np.nan  # Default for 'keep'

        # Apply na_option
        ranks = _apply_na_option(ranks, is_nan, na_option)

        result[row, :] = ranks

    return result
```

**Wrapper maps string na_option to int**:
```python
NA_OPTION_MAP = {'keep': 0, 'top': 1, 'bottom': 2}

def _patched_rank(df, axis=0, method='average', na_option='keep', ascending=True, pct=False):
    na_opt_int = NA_OPTION_MAP.get(na_option, 0)
    # ... call appropriate Numba function with na_opt_int
```

#### TODO 2.3: Optimized Apply for Numba-Compilable Functions
**Acceptance Criteria**:
- [ ] `df.apply(func, engine='numba')` uses parallel execution
- [ ] Detect if function is already Numba-jitted
- [ ] Support axis=0 (column-wise) and axis=1 (row-wise) apply
- [ ] Provide `@unlockedpd.jit` decorator for user functions
- [ ] Clear error messages when function cannot be compiled

**Implementation Approach**:

```python
# src/unlockedpd/ops/apply.py
from numba import njit, prange, types
from numba.core.errors import TypingError
import numpy as np

def jit(func):
    """Decorator to mark function for Numba compilation in apply."""
    jitted = njit(cache=True)(func)
    jitted._unlockedpd_jittable = True
    return jitted

@njit(parallel=True, cache=True)
def _apply_axis0_template(arr: np.ndarray, func) -> np.ndarray:
    """Apply function along axis 0 (to each column)."""
    n_rows, n_cols = arr.shape
    result = np.empty(n_cols, dtype=np.float64)

    for col in prange(n_cols):
        result[col] = func(arr[:, col])

    return result

def make_parallel_apply(user_func):
    """Create a parallel apply function for the user's function."""
    jitted_func = njit(cache=True)(user_func)

    @njit(parallel=True, cache=True)
    def parallel_apply_axis0(arr):
        n_rows, n_cols = arr.shape
        result = np.empty(n_cols, dtype=np.float64)
        for col in prange(n_cols):
            result[col] = jitted_func(arr[:, col])
        return result

    return parallel_apply_axis0
```

#### TODO 2.4: Parallel GroupBy Aggregations (DEFERRED - Future Work)

**Status**: Removed from current scope. Documented here for future consideration.

**Rationale for Deferral**:
- GroupBy operations involve complex pandas internals (GroupByEngine, Splitter, etc.)
- Parallel implementation requires careful handling of group boundaries
- Benefits are less clear-cut than rolling/rank (pandas groupby is already reasonably optimized)
- Risk of subtle correctness issues outweighs potential performance gains in v1.0

**Future Implementation Notes** (for reference):
- Target: `df.groupby(col).agg('mean')` parallelized across groups
- Approach: Pre-compute group indices, then parallel aggregate per group
- Consider: Integration with pandas ExtensionArray for custom group handling

This can be revisited in v1.1+ after core operations prove stable.

---

### Phase 3: Quality and Polish (Priority: Medium)

#### TODO 3.1: Benchmark Suite
**Acceptance Criteria**:
- [ ] Benchmarks for each optimized operation
- [ ] Compare against vanilla pandas at various DataFrame sizes (1K, 10K, 100K, 1M rows)
- [ ] Compare at various column counts (10, 100, 1000 columns)
- [ ] Measure memory usage
- [ ] Generate benchmark report (markdown or HTML)
- [ ] CI integration for performance regression detection

**Benchmark Dimensions**:
```python
# benchmarks/bench_rolling.py
import pytest
import pandas as pd
import numpy as np

SIZES = [(1_000, 10), (10_000, 100), (100_000, 100), (1_000_000, 50)]
WINDOWS = [5, 20, 50]

@pytest.mark.parametrize("rows,cols", SIZES)
@pytest.mark.parametrize("window", WINDOWS)
def test_rolling_mean_performance(benchmark, rows, cols, window):
    df = pd.DataFrame(np.random.randn(rows, cols))

    # Disable unlockedpd for baseline
    import unlockedpd
    unlockedpd.config.enabled = False
    baseline = benchmark.pedantic(lambda: df.rolling(window).mean(), rounds=5)

    # Enable for optimized
    unlockedpd.config.enabled = True
    optimized = benchmark.pedantic(lambda: df.rolling(window).mean(), rounds=5)

    # Assert speedup
    speedup = baseline / optimized
    print(f"Speedup: {speedup:.2f}x")
```

#### TODO 3.2: Pandas Compatibility Tests
**Acceptance Criteria**:
- [ ] Run pandas own test suite for patched methods with unlockedpd active
- [ ] Document any known divergences from pandas behavior
- [ ] Edge case tests: empty DataFrames, single row/column, all NaN
- [ ] Dtype tests: int, float, mixed dtypes (verify non-numeric columns handled)
- [ ] Index tests: default, named, MultiIndex
- [ ] Integration tests for import-and-patch workflow (see below)

**Test Strategy**:
```python
# tests/test_rolling_compat.py
import pandas as pd
import numpy as np
import pytest
import unlockedpd

def test_rolling_mean_matches_pandas():
    df = pd.DataFrame(np.random.randn(100, 10))

    # Get pandas result (with unlockedpd disabled)
    unlockedpd.config.enabled = False
    expected = df.rolling(5).mean()

    # Get unlockedpd result
    unlockedpd.config.enabled = True
    result = df.rolling(5).mean()

    pd.testing.assert_frame_equal(result, expected)

def test_rolling_with_nan():
    df = pd.DataFrame({
        'a': [1, np.nan, 3, 4, 5],
        'b': [np.nan, 2, 3, np.nan, 5]
    })

    unlockedpd.config.enabled = False
    expected = df.rolling(3).mean()

    unlockedpd.config.enabled = True
    result = df.rolling(3).mean()

    pd.testing.assert_frame_equal(result, expected)

def test_mixed_dtype_dataframe():
    """Test that non-numeric columns are handled correctly."""
    df = pd.DataFrame({
        'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'string_col': ['a', 'b', 'c', 'd', 'e'],
        'numeric2': [10, 20, 30, 40, 50],
        'datetime_col': pd.date_range('2020-01-01', periods=5),
    })

    unlockedpd.config.enabled = False
    expected = df.rolling(2).mean()  # pandas skips non-numeric

    unlockedpd.config.enabled = True
    result = df.rolling(2).mean()

    pd.testing.assert_frame_equal(result, expected)
```

**Integration Tests for Import-and-Patch Workflow**:
```python
# tests/test_integration.py
import subprocess
import sys

def test_import_patches_automatically():
    """Verify that importing unlockedpd patches pandas methods."""
    code = '''
import pandas as pd
# Store original method reference
original_mean = pd.core.window.rolling.Rolling.mean

import unlockedpd

# Check that method has been patched
patched_mean = pd.core.window.rolling.Rolling.mean
assert original_mean is not patched_mean, "Method should be patched after import"

# Verify patch registry tracks it
from unlockedpd._patch import _PatchRegistry
assert _PatchRegistry.is_patched(pd.core.window.rolling.Rolling, 'mean')
print("PASS: Import patches automatically")
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f"Test failed: {result.stderr}"

def test_unpatch_restores_original():
    """Verify unpatch_all restores original methods."""
    code = '''
import pandas as pd
original_mean = pd.core.window.rolling.Rolling.mean

import unlockedpd
unlockedpd.unpatch_all()

restored_mean = pd.core.window.rolling.Rolling.mean
assert original_mean is restored_mean, "Method should be restored after unpatch_all"
print("PASS: Unpatch restores original")
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f"Test failed: {result.stderr}"

def test_config_enabled_false_uses_original():
    """Verify config.enabled=False bypasses optimization."""
    code = '''
import pandas as pd
import numpy as np
import unlockedpd

# Small DataFrame where optimization won't help
df = pd.DataFrame(np.random.randn(10, 3))

# With optimization
unlockedpd.config.enabled = True
result1 = df.rolling(3).mean()

# Without optimization (uses original pandas)
unlockedpd.config.enabled = False
result2 = df.rolling(3).mean()

# Results should be identical
pd.testing.assert_frame_equal(result1, result2)
print("PASS: config.enabled=False uses original")
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f"Test failed: {result.stderr}"

def test_fallback_on_unsupported_dtype():
    """Verify graceful fallback for unsupported operations."""
    code = '''
import pandas as pd
import unlockedpd

# DataFrame with only non-numeric columns (should fallback)
df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})

# Should not crash, should fallback to pandas
result = df.rolling(2).mean()

# Pandas returns empty DataFrame for all-string rolling mean
assert result.empty or result.isna().all().all()
print("PASS: Fallback on unsupported dtype")
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, f"Test failed: {result.stderr}"
```

#### TODO 3.3: Documentation
**Acceptance Criteria**:
- [ ] README.md with quick start guide
- [ ] API reference for configuration options
- [ ] Performance guide with benchmarks
- [ ] Compatibility matrix (pandas versions, Python versions)
- [ ] Troubleshooting guide for common issues

---

### Phase 4: Release (Priority: Low)

#### TODO 4.1: Package for PyPI
**Acceptance Criteria**:
- [ ] `pyproject.toml` complete with metadata
- [ ] `LICENSE` file (MIT recommended)
- [ ] `CHANGELOG.md` initialized
- [ ] Package builds successfully with `python -m build`
- [ ] Package installable from wheel

#### TODO 4.2: CI/CD Setup
**Acceptance Criteria**:
- [ ] GitHub Actions workflow for testing
- [ ] Matrix testing: Python 3.9-3.12, pandas 1.5/2.0/2.1/2.2
- [ ] Automated benchmarks on PR
- [ ] Automated PyPI release on tag

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pandas internal API changes break patches | Medium | High | Version detection in `_compat.py`, conditional patching |
| Numba compilation failures on user systems | Medium | Medium | Fallback wrappers, clear error messages, documentation |
| Thread contention with user's parallel code | Medium | Medium | Configurable `num_threads`, document interaction |
| Numerical precision differences from pandas | Low | High | Extensive testing, use same algorithms where possible |
| Performance regression in edge cases | Medium | Low | Benchmarks, automatic fallback for small DataFrames |
| Memory bloat from array copies | Medium | Medium | Careful memory management, avoid copies when possible |

---

## Commit Strategy

| Phase | Commits |
|-------|---------|
| 1.1 | `feat: Initialize unlockedpd project structure` |
| 1.2 | `feat: Add configuration system with env var support` |
| 1.3 | `feat: Add patch infrastructure with fallback mechanism` |
| 1.4 | `feat: Add pandas compatibility layer` |
| 2.1 | `feat: Add parallel rolling operations (mean, sum, std)` |
| 2.2 | `feat: Add parallel rank operation` |
| 2.3 | `feat: Add optimized apply with Numba support` |
| 3.1 | `test: Add benchmark suite` |
| 3.2 | `test: Add pandas compatibility tests` |
| 3.3 | `docs: Add documentation` |
| 4.x | `chore: Prepare for release` |

---

## Success Criteria

### Performance Targets
- **Rolling operations**: 1.5-5x speedup on 100+ column DataFrames (minimum 1.5x guaranteed)
- **Rank operations**: 1.5-4x speedup on axis=1 with 100+ columns (minimum 1.5x guaranteed)
- **Apply operations**: 2-10x speedup for Numba-compilable functions (minimum 2x for jit-compatible functions)

### Quality Targets
- **Compatibility**: 100% pass rate on pandas tests for patched methods
- **Reliability**: Zero data corruption or silent failures
- **Usability**: Single import line, zero configuration required for basic use

### Adoption Targets
- **Documentation**: Complete enough for new user onboarding in <5 minutes
- **Error messages**: Clear, actionable messages for all failure modes

---

## Appendix: Numba Performance Patterns

### Parallel Column Processing
```python
@njit(parallel=True, cache=True)
def parallel_column_op(arr_2d, func):
    n_rows, n_cols = arr_2d.shape
    result = np.empty_like(arr_2d)

    for col in prange(n_cols):  # Parallel
        for row in range(n_rows):  # Sequential
            result[row, col] = func(arr_2d[row, col])

    return result
```

### Parallel Row Processing
```python
@njit(parallel=True, cache=True)
def parallel_row_op(arr_2d, func):
    n_rows, n_cols = arr_2d.shape
    result = np.empty(n_rows, dtype=np.float64)

    for row in prange(n_rows):  # Parallel
        result[row] = func(arr_2d[row, :])

    return result
```

### Stencil for Window Operations (Alternative)
```python
from numba import stencil

@stencil
def rolling_mean_kernel(a, window=5):
    total = 0.0
    for i in range(-window+1, 1):
        total += a[i]
    return total / window
```

---

## Next Steps

After plan approval, execute with:
```
/start-work .sisyphus/plans/unlockedpd.md
```

Or for automated execution:
```
/ralph-loop
```
