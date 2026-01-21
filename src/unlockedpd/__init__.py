"""unlockedpd - Transparent pandas performance optimization.

Import this module to automatically patch pandas with Numba-accelerated
parallel implementations of common operations.

Usage:
    import pandas as pd
    import unlockedpd  # This patches pandas automatically

    df = pd.DataFrame(...)
    df.rolling(20).mean()  # Now parallelized!
    df.rank(axis=1)        # Now parallelized!

Configuration:
    import unlockedpd
    unlockedpd.config.enabled = False  # Disable all optimizations
    unlockedpd.config.num_threads = 4  # Set thread count
    unlockedpd.config.warn_on_fallback = True  # Warn when falling back to pandas
    unlockedpd.config.parallel_threshold = 5000  # Min elements for parallel

    # Or via environment variables:
    # UNLOCKEDPD_ENABLED=false
    # UNLOCKEDPD_NUM_THREADS=4
    # UNLOCKEDPD_WARN_ON_FALLBACK=true
    # UNLOCKEDPD_PARALLEL_THRESHOLD=10000
"""

__version__ = "0.2.1"

# Import configuration
from ._config import config

# Import patch utilities
from ._patch import (
    patch,
    unpatch,
    unpatch_all,
    is_patched,
    _PatchRegistry,
)

# Import the jit decorator for user functions
from .ops.apply import jit


def _apply_all_patches():
    """Apply all optimization patches to pandas.

    Benchmarked speedups vs pandas (64 threads, various shapes):
    - pct_change: 11x faster (major win for financial calculations)
    - rank(axis=1): 8-10x faster (row-parallel, massive win for wide DataFrames)
    - rolling (mean, sum, std, var, min, max): 1.5x - 2.2x faster
    - expanding (mean): 1.5x - 1.6x faster
    - rank(axis=0): 1.4x - 1.5x faster
    - diff: 1.0x - 1.7x faster (shape-dependent)
    - shift: 1.0x - 1.5x faster (shape-dependent)

    Not patched (marginal or negative benefit):
    - cumulative (cumsum, cumprod, etc.): NumPy SIMD is faster
    - ewm: Same speed as pandas (no benefit)
    - stats: Marginal benefit
    - pairwise: Complex semantics
    """
    from .ops.rank import apply_rank_patches
    from .ops.transform import apply_transform_patches
    from .ops.rolling import apply_rolling_patches
    from .ops.expanding import apply_expanding_patches
    from .ops.pairwise import apply_pairwise_patches
    # Transform operations: diff 1.0-1.7x, shift 1.0-1.5x, pct_change 11x
    apply_transform_patches()

    # Rank: axis=0 is 1.4x faster, axis=1 is 8-10x faster (row-parallel)
    apply_rank_patches()

    # Rolling: 1.5x - 2.2x faster
    apply_rolling_patches()

    # Expanding: 1.5x - 1.6x faster
    apply_expanding_patches()

    # Pairwise: rolling corr/cov
    apply_pairwise_patches()

    # Cumulative ops with nogil kernels
    from .ops.cumulative import apply_cumulative_patches
    apply_cumulative_patches()

    # EWM operations with nogil kernels
    from .ops.ewm import apply_ewm_patches
    apply_ewm_patches()


def _warmup_all():
    """Pre-compile all Numba functions to avoid first-call overhead."""
    try:
        from ._warmup import warmup_all
        warmup_all()
    except Exception:
        # Silently fail - functions will compile on first use
        pass


# Auto-patch on import if enabled
if config.enabled:
    _apply_all_patches()
    # Warmup after patching to pre-compile functions
    _warmup_all()


__all__ = [
    # Version
    "__version__",
    # Configuration
    "config",
    # Patch management
    "patch",
    "unpatch",
    "unpatch_all",
    "is_patched",
    "_PatchRegistry",
    # User utilities
    "jit",
    # Internal
    "_apply_all_patches",
    "_warmup_all",
]
