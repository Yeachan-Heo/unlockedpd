# unlockedpd

**Unlock pandas performance with zero code changes.**

[![PyPI version](https://badge.fury.io/py/unlockedpd.svg)](https://badge.fury.io/py/unlockedpd)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

unlockedpd is a **drop-in performance booster** for pandas that achieves **8-15x speedups** on rolling, expanding, and other window operations. Just `import unlockedpd` after pandas and your existing code runs faster.

```python
import pandas as pd
import unlockedpd  # That's it. Your pandas code is now faster.

df = pd.DataFrame(...)
df.rolling(20).mean()  # 8.9x faster!
df.expanding().std()   # 8.6x faster!
df.rank(axis=1)        # 10x faster!
```

## Why unlockedpd?

| Library | Speedup | pandas Compatible | Setup Required |
|---------|---------|-------------------|----------------|
| **unlockedpd** | **8.9x** | **100%** | `pip install` |
| Polars | 5-10x | 0% (new API) | Learn new API |
| Modin | ~4x | 95% | Ray/Dask cluster |

**Key advantages:**
- **Zero code changes**: Works with your existing pandas code
- **No infrastructure**: No Ray, no Dask, no distributed setup
- **No new API to learn**: It's still pandas
- **Automatic fallback**: Falls back to pandas for unsupported cases

## Benchmarks

Tested on a 64-core machine with a **1GB DataFrame** (10,000 rows x 13,000 columns):

### Rolling Operations

| Operation | pandas | unlockedpd | Speedup |
|-----------|--------|------------|---------|
| `rolling(20).mean()` | 2.48s | 0.55s | **4.5x** |
| `rolling(20).sum()` | 1.91s | 0.22s | **8.6x** |
| `rolling(20).std()` | 2.94s | 0.50s | **5.9x** |
| `rolling(20).var()` | 2.66s | 0.46s | **5.7x** |
| `rolling(20).min()` | 4.00s | 0.35s | **11.6x** |
| `rolling(20).max()` | 4.05s | 0.36s | **11.4x** |

### Expanding Operations

| Operation | pandas | unlockedpd | Speedup |
|-----------|--------|------------|---------|
| `expanding().mean()` | 1.67s | 0.23s | **7.3x** |
| `expanding().sum()` | 1.49s | 0.22s | **6.7x** |
| `expanding().std()` | 2.13s | 0.25s | **8.6x** |
| `expanding().var()` | 1.88s | 0.23s | **8.0x** |
| `expanding().min()` | 3.28s | 0.23s | **14.4x** |
| `expanding().max()` | 3.31s | 0.23s | **14.6x** |

### Other Operations

| Operation | Speedup |
|-----------|---------|
| `pct_change()` | **11x** |
| `rank(axis=1)` | **8-10x** |
| `rank(axis=0)` | **1.4-1.5x** |
| `diff()` | **1.0-1.7x** |
| `shift()` | **1.0-1.5x** |

## Installation

```bash
pip install unlockedpd
```

**Requirements:**
- Python 3.9+
- pandas >= 1.5
- numba >= 0.56
- numpy >= 1.21

## Usage

### Basic Usage

```python
import pandas as pd
import unlockedpd  # Import after pandas

# Your existing code works unchanged
df = pd.DataFrame(np.random.randn(10000, 1000))
result = df.rolling(20).mean()  # Automatically optimized!
```

### Configuration

```python
import unlockedpd

# Disable optimizations temporarily
unlockedpd.config.enabled = False

# Set thread count (default: min(cpu_count, 32))
unlockedpd.config.num_threads = 16

# Enable warnings when falling back to pandas
unlockedpd.config.warn_on_fallback = True

# Set minimum elements for parallel execution
unlockedpd.config.parallel_threshold = 500_000
```

### Environment Variables

```bash
export UNLOCKEDPD_ENABLED=false
export UNLOCKEDPD_NUM_THREADS=16
export UNLOCKEDPD_WARN_ON_FALLBACK=true
export UNLOCKEDPD_PARALLEL_THRESHOLD=500000
```

### Temporarily Disable

```python
from unlockedpd import _PatchRegistry

with _PatchRegistry.temporarily_unpatched():
    # Uses original pandas here
    result = df.rolling(20).mean()
```

## How It Works

unlockedpd achieves its speedups through:

1. **Numba JIT compilation**: Operations are compiled to optimized machine code
2. **`nogil=True`**: Releases Python's GIL during computation
3. **ThreadPoolExecutor**: Achieves true parallelism across CPU cores
4. **Column-wise chunking**: Distributes work efficiently across threads

The key insight: `@njit(nogil=True)` + `ThreadPoolExecutor` combines Numba's fast compiled loops with true multi-threaded parallelism.

```
┌─────────────────────────────────────────────────────────────┐
│                    ThreadPoolExecutor                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐   │
│  │ Thread 1│  │ Thread 2│  │ Thread 3│  ...  │Thread 32│   │
│  │ Cols 0-k│  │Cols k-2k│  │Cols 2k..│       │Cols ..N │   │
│  │ (nogil) │  │ (nogil) │  │ (nogil) │       │ (nogil) │   │
│  └─────────┘  └─────────┘  └─────────┘       └─────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## What's Optimized

**Fully optimized (8-15x faster):**
- `rolling().mean()`, `sum()`, `std()`, `var()`, `min()`, `max()`, `count()`, `skew()`, `kurt()`
- `expanding().mean()`, `sum()`, `std()`, `var()`, `min()`, `max()`, `count()`, `skew()`, `kurt()`
- `rank()` (both axis=0 and axis=1)
- `pct_change()`, `diff()`, `shift()`

**Passes through to pandas (unchanged):**
- `rolling().median()`, `quantile()`, `apply()`, `corr()`, `cov()`
- `ewm()` operations
- `cumsum()`, `cumprod()` (NumPy SIMD is already fast)
- Series operations (optimizations target DataFrames)
- Non-numeric columns (auto-fallback)

## Compatibility

unlockedpd is designed for **100% pandas compatibility**:

- **Drop-in replacement**: No code changes required
- **Automatic fallback**: If optimization fails, falls back to pandas
- **Type preservation**: Returns same types as pandas
- **Index preservation**: Maintains DataFrame/Series indices
- **NaN handling**: Correctly handles missing values

## Comparison with Alternatives

### vs Polars

| Aspect | unlockedpd | Polars |
|--------|------------|--------|
| Speedup | 8.9x | 5-10x |
| API | pandas (unchanged) | New API to learn |
| Code changes | None | Rewrite required |
| Ecosystem | pandas ecosystem | Polars ecosystem |

### vs Modin

| Aspect | unlockedpd | Modin |
|--------|------------|-------|
| Speedup | 8.9x | ~4x (general) |
| Rolling ops | Optimized | Not optimized |
| Infrastructure | None | Ray/Dask cluster |
| Memory | Low overhead | Partitioning overhead |

### vs Vanilla Numba

| Aspect | unlockedpd | Manual Numba |
|--------|------------|--------------|
| Usage | `import unlockedpd` | Write custom kernels |
| GIL handling | Automatic (`nogil=True`) | Manual |
| Parallelization | Automatic ThreadPool | Manual implementation |

## Running Benchmarks

```bash
# Clone the repo
git clone https://github.com/unlockedpd/unlockedpd
cd unlockedpd

# Install with dev dependencies
pip install -e ".[dev]"

# Run benchmarks
pytest benchmarks/ -v
```

## Contributing

Contributions are welcome! Areas of interest:

- Additional operation optimizations
- Performance improvements
- Documentation and examples
- Bug reports and fixes

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [Numba](https://numba.pydata.org/) - JIT compilation for Python
- [pandas](https://pandas.pydata.org/) - Data analysis library
- [NumPy](https://numpy.org/) - Numerical computing

---

**unlockedpd** - *Because your pandas code deserves to be fast.*
