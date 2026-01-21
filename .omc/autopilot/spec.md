# Performance Improvements Specification

## Requirements

### HIGH Impact (Must Fix)
1. **BUG: Rolling median dispatch** - `rolling.py:1301-1306` both branches call threadpool
2. **Threshold inconsistency** - Centralize thresholds to `_config.py`
3. **Missing ThreadPool tier** - Add 3rd tier to `stats.py`, `rank.py`, `transform.py`
4. **O(n*window) rolling min/max** - Use monotonic deque from `_minmax_deque.py`

### MEDIUM Impact (Should Fix)
5. **Missing memory layout optimization** - Use `ensure_optimal_layout()` in `aggregations.py`, `fillna.py`, `ewm.py`
6. **Inconsistent expanding skew/kurt** - Add 3rd tier to `expanding.py:932-950`

### LOW Impact (Nice to Have)
7. **Configurable thread count** - Make `THREADPOOL_WORKERS` configurable via `_config.py`

## Technical Approach

### Threshold Centralization
- Add constants to `_config.py`: `PARALLEL_THRESHOLD = 500_000`, `THREADPOOL_THRESHOLD = 10_000_000`
- Update all ops modules to import from `_config.py`

### ThreadPool Tier Pattern
```python
# Existing pattern to replicate:
def _dispatch(arr, ...):
    n = arr.size
    if n >= THREADPOOL_THRESHOLD:
        return _threadpool_impl(arr, ...)
    elif n >= PARALLEL_THRESHOLD:
        return _parallel_impl(arr, ...)
    else:
        return _serial_impl(arr, ...)
```

### Monotonic Deque for Rolling Min/Max
- `_minmax_deque.py` already exists with `MonotonicDeque` class
- Integrate into `rolling.py` for O(1) amortized min/max

## Files to Modify

| File | Changes |
|------|---------|
| `src/unlockedpd/_config.py` | Add PARALLEL_THRESHOLD, THREADPOOL_THRESHOLD, threadpool_workers |
| `src/unlockedpd/ops/rolling.py` | Fix dispatch bug, use deque for min/max, import thresholds |
| `src/unlockedpd/ops/stats.py` | Add ThreadPool tier, import thresholds |
| `src/unlockedpd/ops/rank.py` | Add ThreadPool tier, import thresholds |
| `src/unlockedpd/ops/transform.py` | Add ThreadPool tier, import thresholds |
| `src/unlockedpd/ops/expanding.py` | Add ThreadPool to skew/kurt, import thresholds |
| `src/unlockedpd/ops/aggregations.py` | Add optimal layout, import thresholds |
| `src/unlockedpd/ops/fillna.py` | Add optimal layout, import thresholds |
| `src/unlockedpd/ops/ewm.py` | Add optimal layout, import thresholds |
| `src/unlockedpd/ops/cumulative.py` | Import thresholds (align with standard) |

## Definition of Done
- All tests pass (229+)
- Thresholds centralized in _config.py
- All ops modules use imported thresholds
- Rolling min/max use monotonic deque
- Rolling median dispatch bug fixed
- ThreadPool tier added to stats/rank/transform/expanding
- Memory layout optimization added where beneficial
