"""Thread-safe configuration for unlockedpd.

This module provides the configuration system that controls unlockedpd behavior.
All configuration access is thread-safe using a lock.

Environment Variables:
    UNLOCKEDPD_ENABLED: Set to 'false' to disable all patches (default: 'true')
    UNLOCKEDPD_NUM_THREADS: Number of threads for Numba parallel operations (default: 0 = auto)
    UNLOCKEDPD_THREADPOOL_WORKERS: Worker cap for Python ThreadPool paths (default: 0 = auto)
    UNLOCKEDPD_WARN_ON_FALLBACK: Set to 'true' to warn when falling back to pandas (default: 'false')
    UNLOCKEDPD_PARALLEL_THRESHOLD: Minimum array size for parallel execution (default: 10000)
    UNLOCKEDPD_MAX_MEMORY_OVERHEAD: Maximum optimized-vs-pandas RSS ratio budget (default: 6.0)
    UNLOCKEDPD_MAX_CPU_OVERHEAD: Maximum optimized-vs-pandas CPU seconds ratio budget (default: 6.0)
"""
import math
import os
import threading
from dataclasses import dataclass, field

import numba


def _parse_auto_int(value: object, default: int = 0) -> int:
    """Parse an integer where unset, 0, and 'auto' mean adaptive/default."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "0", "auto", "default"}:
            return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else 0


def _parse_positive_float_env(name: str, default: float) -> float:
    """Parse a positive finite float env var, falling back on invalid values."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _coerce_positive_float(value: object, name: str) -> float:
    """Coerce a runtime config value to a positive finite float."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


@dataclass
class UnlockedConfig:
    """Thread-safe configuration for unlockedpd.

    Uses a lock for all mutable attribute access to ensure
    thread-safety when config is modified from multiple threads.

    Attributes:
        enabled: If False, all patches bypass to original pandas methods
        num_threads: Number of threads for Numba (0 = auto/default)
        threadpool_workers: Worker cap for Python ThreadPool paths (0 = adaptive)
        warn_on_fallback: If True, emit warnings when falling back to pandas
        cache_compiled: If True, cache Numba compiled functions
        parallel_threshold: Minimum array size before parallel execution is used
        max_memory_overhead: Maximum optimized-vs-pandas RSS ratio budget
        max_cpu_overhead: Maximum optimized-vs-pandas CPU seconds ratio budget
    """
    _enabled: bool = field(default=True, repr=False)
    _num_threads: int = field(default=0, repr=False)
    _threadpool_workers: int = field(default=0, repr=False)
    _warn_on_fallback: bool = field(default=False, repr=False)
    _cache_compiled: bool = field(default=True, repr=False)
    _parallel_threshold: int = field(default=10_000, repr=False)
    _max_memory_overhead: float = field(default=6.0, repr=False)
    _max_cpu_overhead: float = field(default=6.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._enabled = os.environ.get('UNLOCKEDPD_ENABLED', 'true').lower() == 'true'

        threads_str = os.environ.get('UNLOCKEDPD_NUM_THREADS', '0')
        self._num_threads = int(threads_str) if threads_str.isdigit() else 0

        self._threadpool_workers = _parse_auto_int(os.environ.get('UNLOCKEDPD_THREADPOOL_WORKERS'), 0)

        self._warn_on_fallback = os.environ.get('UNLOCKEDPD_WARN_ON_FALLBACK', 'false').lower() == 'true'

        threshold_str = os.environ.get('UNLOCKEDPD_PARALLEL_THRESHOLD', '10000')
        self._parallel_threshold = int(threshold_str) if threshold_str.isdigit() else 10_000

        self._max_memory_overhead = _parse_positive_float_env('UNLOCKEDPD_MAX_MEMORY_OVERHEAD', 6.0)
        self._max_cpu_overhead = _parse_positive_float_env('UNLOCKEDPD_MAX_CPU_OVERHEAD', 6.0)

        # Apply thread config on initialization
        if self._num_threads > 0:
            numba.set_num_threads(self._num_threads)

    @property
    def enabled(self) -> bool:
        """Whether unlockedpd optimizations are enabled."""
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)

    @property
    def num_threads(self) -> int:
        """Number of threads for Numba parallel operations (0 = auto)."""
        with self._lock:
            return self._num_threads

    @num_threads.setter
    def num_threads(self, value: int) -> None:
        with self._lock:
            parsed = int(value)
            self._num_threads = parsed
            if parsed > 0:
                numba.set_num_threads(parsed)

    @property
    def threadpool_workers(self) -> int:
        """Worker cap for Python ThreadPool paths (0 = adaptive/default)."""
        with self._lock:
            return self._threadpool_workers

    @threadpool_workers.setter
    def threadpool_workers(self, value: int) -> None:
        with self._lock:
            self._threadpool_workers = _parse_auto_int(value, 0)

    @property
    def warn_on_fallback(self) -> bool:
        """Whether to warn when falling back to original pandas."""
        with self._lock:
            return self._warn_on_fallback

    @warn_on_fallback.setter
    def warn_on_fallback(self, value: bool) -> None:
        with self._lock:
            self._warn_on_fallback = bool(value)

    @property
    def cache_compiled(self) -> bool:
        """Whether to cache Numba compiled functions."""
        with self._lock:
            return self._cache_compiled

    @cache_compiled.setter
    def cache_compiled(self, value: bool) -> None:
        with self._lock:
            self._cache_compiled = bool(value)

    @property
    def parallel_threshold(self) -> int:
        """Minimum array size before parallel execution is used."""
        with self._lock:
            return self._parallel_threshold

    @parallel_threshold.setter
    def parallel_threshold(self, value: int) -> None:
        with self._lock:
            self._parallel_threshold = int(value)

    @property
    def max_memory_overhead(self) -> float:
        """Maximum optimized-vs-pandas RSS ratio budget."""
        with self._lock:
            return self._max_memory_overhead

    @max_memory_overhead.setter
    def max_memory_overhead(self, value: float) -> None:
        with self._lock:
            self._max_memory_overhead = _coerce_positive_float(value, "max_memory_overhead")

    @property
    def max_cpu_overhead(self) -> float:
        """Maximum optimized-vs-pandas CPU seconds ratio budget."""
        with self._lock:
            return self._max_cpu_overhead

    @max_cpu_overhead.setter
    def max_cpu_overhead(self, value: float) -> None:
        with self._lock:
            self._max_cpu_overhead = _coerce_positive_float(value, "max_cpu_overhead")

    def apply_thread_config(self) -> None:
        """Apply the current thread configuration to Numba."""
        with self._lock:
            if self._num_threads > 0:
                numba.set_num_threads(self._num_threads)

    def as_dict(self) -> dict:
        """Return a thread-safe snapshot for profiling/reporting."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "num_threads": self._num_threads,
                "threadpool_workers": self._threadpool_workers,
                "warn_on_fallback": self._warn_on_fallback,
                "cache_compiled": self._cache_compiled,
                "parallel_threshold": self._parallel_threshold,
                "max_memory_overhead": self._max_memory_overhead,
                "max_cpu_overhead": self._max_cpu_overhead,
            }

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"UnlockedConfig(enabled={self._enabled}, "
                f"num_threads={self._num_threads}, "
                f"threadpool_workers={self._threadpool_workers}, "
                f"warn_on_fallback={self._warn_on_fallback}, "
                f"cache_compiled={self._cache_compiled}, "
                f"parallel_threshold={self._parallel_threshold}, "
                f"max_memory_overhead={self._max_memory_overhead}, "
                f"max_cpu_overhead={self._max_cpu_overhead})"
            )


# Global configuration instance
config = UnlockedConfig()
