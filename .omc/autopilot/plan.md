# Implementation Plan

## Execution Order

### Batch 1: Foundation (Sequential - Dependencies)
1. **Centralize thresholds to _config.py** - Must be done first as all other modules will import from here

### Batch 2: HIGH Impact (Parallel)
2. **Fix rolling median dispatch bug** - rolling.py
3. **Use monotonic deque for rolling min/max** - rolling.py
4. **Add ThreadPool tier to stats.py**
5. **Add ThreadPool tier to rank.py**
6. **Add ThreadPool tier to transform.py**

### Batch 3: MEDIUM Impact (Parallel)
7. **Add optimal layout to aggregations.py**
8. **Add optimal layout to fillna.py**
9. **Add optimal layout to ewm.py**
10. **Add ThreadPool to expanding skew/kurt**

### Batch 4: LOW Impact
11. **Make threadpool workers configurable** - Already part of _config.py changes

### Batch 5: Validation
12. **Run full test suite**
13. **Architect verification**

## Detailed Tasks

### Task 1: Centralize Thresholds (_config.py)
Add to _config.py:
```python
# Dispatch thresholds
PARALLEL_THRESHOLD = 500_000  # Elements for parallel prange
THREADPOOL_THRESHOLD = 10_000_000  # Elements for ThreadPool+nogil

# ThreadPool configuration
@property
def threadpool_workers(self):
    return min(os.cpu_count() or 4, 32)
```

### Task 2-3: Fix Rolling (rolling.py)
- Fix lines 1301-1306: Add serial implementation for median
- Fix lines 1309-1313: Add serial implementation for quantile
- Integrate _minmax_deque.py for O(1) rolling min/max

### Task 4-6: Add ThreadPool Tier
Pattern for each module:
1. Add `_*_nogil` kernel with `@njit(nogil=True, cache=True)`
2. Add `_*_threadpool` function using ThreadPoolExecutor
3. Update dispatch to use 3-tier

### Task 7-9: Add Optimal Layout
For column-parallel operations:
```python
from .._compat import ensure_optimal_layout
arr, was_converted = ensure_optimal_layout(df.values, axis=0)
```

### Task 10: Expanding Skew/Kurt ThreadPool
Add 3rd tier to expanding.py lines 932-950
